# Copyright (c) 2026, Tri Dao.

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SplitSchedulePlan:
    num_splits: int
    split_counts: list[int] | None


@dataclass(frozen=True)
class SplitScheduleLaunch:
    num_splits: int
    scheduler_metadata: torch.Tensor | None


class SplitSchedulerPlanner:
    """Own device scheduling policy and graph-stable workspace."""
    _NUM_HOST_WORKSPACES = 2

    def __init__(self, *, device: torch.device, max_batch_size: int):
        self.device = device
        self.max_batch_size = max_batch_size
        self._workspace: torch.Tensor | None = None
        self._host_workspaces: list[torch.Tensor] | None = None
        self._host_workspace_events: list[torch.cuda.Event] | None = None
        self._host_workspace_event_recorded = [False] * self._NUM_HOST_WORKSPACES
        self._next_host_workspace = 0

    def __call__(
        self,
        query_start_loc_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        *,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        head_dim_v: int,
        has_qv: bool,
        cp_world_size: int,
        window_size: tuple[int, int] | None,
        cuda_graph_max_num_splits: int | None = None,
        fast_build: bool = False,
    ) -> SplitScheduleLaunch | None:
        plan = plan_hopper_split_schedule(
            query_start_loc_cpu,
            seq_lens_cpu,
            device=self.device,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            has_qv=has_qv,
            cp_world_size=cp_world_size,
            window_size=window_size,
            cuda_graph_max_num_splits=cuda_graph_max_num_splits,
            fast_build=fast_build,
        )
        if plan is None:
            return None
        scheduler_metadata = None
        if plan.split_counts is not None:
            if self._workspace is None:
                self._workspace = torch.ones(
                    self.max_batch_size,
                    dtype=torch.int32,
                    device=self.device,
                )
                self._host_workspaces = [
                    torch.ones(
                        self.max_batch_size,
                        dtype=torch.int32,
                        pin_memory=True,
                    )
                    for _ in range(self._NUM_HOST_WORKSPACES)
                ]
                self._host_workspace_events = [
                    torch.cuda.Event() for _ in range(self._NUM_HOST_WORKSPACES)
                ]
            assert self._host_workspaces is not None
            assert self._host_workspace_events is not None
            counts = (
                [-1, *plan.split_counts[1:]]
                if max(plan.split_counts) == 1
                else plan.split_counts
            )
            count = len(counts)
            slot = self._next_host_workspace
            event = self._host_workspace_events[slot]
            if self._host_workspace_event_recorded[slot] and not event.query():
                event.synchronize()
            host_workspace = self._host_workspaces[slot]
            host_workspace[:count].copy_(torch.tensor(counts, dtype=torch.int32))
            self._workspace[:count].copy_(
                host_workspace[:count],
                non_blocking=True,
            )
            event.record(torch.cuda.current_stream(self.device))
            self._host_workspace_event_recorded[slot] = True
            self._next_host_workspace = (slot + 1) % self._NUM_HOST_WORKSPACES
            scheduler_metadata = self._workspace[:count]
        return SplitScheduleLaunch(
            num_splits=plan.num_splits,
            scheduler_metadata=scheduler_metadata,
        )


def plan_hopper_split_schedule(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    *,
    device: torch.device,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    head_dim_v: int,
    has_qv: bool,
    cp_world_size: int,
    window_size: tuple[int, int] | None,
    cuda_graph_max_num_splits: int | None = None,
    fast_build: bool = False,
) -> SplitSchedulePlan | None:
    """Return FA3-style SplitKV counts for measured Hopper Dense decode."""
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        return None
    q_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    active_q_lens = q_lens[q_lens > 0]
    all_single_token = (
        active_q_lens.numel() > 0 and active_q_lens.max().item() == 1
    )
    mixed_decode_extend = (
        active_q_lens.numel() > 0
        and active_q_lens.min().item() == 1
        and not all_single_token
    )
    if (
        q_lens.numel() == 0
        or q_lens.numel() > 128
        or has_qv
        or not (all_single_token or mixed_decode_extend)
        or cp_world_size != 1
        or fast_build
    ):
        return None
    qheads_per_kvhead = math.ceil(num_heads_q / num_heads_kv)
    max_query_len = active_q_lens.max().item()
    if head_dim == head_dim_v == 128:
        tile_m = 64 if max_query_len * qheads_per_kvhead <= 64 else 128
        tile_n = 128
    elif head_dim == head_dim_v == 256:
        tile_m, tile_n = 128, 80
    else:
        return None

    keep_all_one_schedule = cuda_graph_max_num_splits is not None
    max_num_splits = cuda_graph_max_num_splits or 128
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_m_blocks = [
        math.ceil(q_len * qheads_per_kvhead / tile_m)
        for q_len in q_lens.tolist()
    ]
    if window_size is None or (window_size[0] < 0 and window_size[1] < 0):
        effective_k_lens = seq_lens_cpu.tolist()
    else:
        window_left, window_right = window_size
        effective_k_lens = [
            max(
                0,
                min(
                    kv_len,
                    (kv_len if window_left < 0 else window_left)
                    + (kv_len if window_right < 0 else window_right)
                    + 1
                    + tile_m,
                ),
            )
            for kv_len in seq_lens_cpu.tolist()
        ]
    num_n_blocks = [
        math.ceil(effective_k_len / tile_n)
        for effective_k_len in effective_k_lens
    ]
    total_blocks = sum(
        m_blocks * n_blocks
        for m_blocks, n_blocks in zip(num_m_blocks, num_n_blocks)
    )
    graph_split_bound = max_num_splits
    if keep_all_one_schedule:
        min_active_m_blocks = min(blocks for blocks in num_m_blocks if blocks > 0)
        graph_split_bound = min(
            max_num_splits,
            math.ceil(num_sms / (1.1 * num_heads_kv * min_active_m_blocks)),
        )
        if graph_split_bound <= 1:
            return SplitSchedulePlan(num_splits=1, split_counts=None)
    blocks_per_sm = max(
        1, math.ceil(total_blocks * 1.1 * num_heads_kv / num_sms)
    )
    splits = [
        max(1, min(math.ceil(n_blocks / blocks_per_sm), graph_split_bound))
        for n_blocks in num_n_blocks
    ]
    if not keep_all_one_schedule and len(set(splits)) == 1:
        if splits[0] == 1 and len(splits) > 1:
            return None
        return SplitSchedulePlan(num_splits=splits[0], split_counts=None)
    return SplitSchedulePlan(
        num_splits=graph_split_bound if keep_all_one_schedule else max(splits),
        split_counts=splits,
    )
