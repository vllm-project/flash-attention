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

    def __init__(self, *, device: torch.device, max_batch_size: int):
        self.device = device
        self.max_batch_size = max_batch_size
        self._workspace: torch.Tensor | None = None

    def __call__(self, *args, **kwargs) -> SplitScheduleLaunch | None:
        plan = plan_hopper_split_schedule(*args, device=self.device, **kwargs)
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
            count = len(plan.split_counts)
            self._workspace[:count].copy_(
                torch.tensor(
                    (
                        [-1, *plan.split_counts[1:]]
                        if max(plan.split_counts) == 1
                        else plan.split_counts
                    ),
                    dtype=torch.int32, device=self.device,
                ),
                non_blocking=True,
            )
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
    cuda_graph_max_num_splits: int | None = None,
    fast_build: bool = False,
) -> SplitSchedulePlan | None:
    """Return FA3-style SplitKV counts for measured Hopper Dense decode."""
    capability = torch.cuda.get_device_capability(device)
    if capability != (9, 0):
        return None
    q_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    all_single_token = q_lens.numel() > 0 and q_lens.max().item() == 1
    mixed_decode_extend = (
        q_lens.numel() > 0
        and q_lens.min().item() == 1
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
    if head_dim != 256 or head_dim_v != 256:
        return None

    keep_all_one_schedule = cuda_graph_max_num_splits is not None
    max_num_splits = cuda_graph_max_num_splits or 128
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    qheads_per_kvhead = math.ceil(num_heads_q / num_heads_kv)
    num_m_blocks = [
        math.ceil(q_len * qheads_per_kvhead / 128)
        for q_len in q_lens.tolist()
    ]
    num_n_blocks = [
        math.ceil(kv_len / 80) for kv_len in seq_lens_cpu.tolist()
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
            return None
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
