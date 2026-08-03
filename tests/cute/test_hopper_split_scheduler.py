import pytest
import torch

from flash_attn.cute.split_scheduler import plan_hopper_split_schedule


def _plan(monkeypatch, **overrides):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Props", (), {"multi_processor_count": 132})(),
    )
    args = {
        "query_start_loc_cpu": torch.arange(5, dtype=torch.int32),
        "seq_lens_cpu": torch.full((4,), 4096, dtype=torch.int32),
        "device": torch.device("cuda"),
        "num_heads_q": 16,
        "num_heads_kv": 8,
        "head_dim": 256,
        "head_dim_v": 256,
        "has_qv": False,
        "cp_world_size": 1,
    }
    args.update(overrides)
    return plan_hopper_split_schedule(**args)


def test_standard_d256_split_planner_matches_fa3(monkeypatch):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([512, 4096], dtype=torch.int32),
    )
    assert plan is not None
    assert plan.num_splits == 13
    assert plan.split_counts == [2, 13]


def test_standard_d256_mixed_query_split_planner_matches_fa3(monkeypatch):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor(
            [0, 257, 258, 259, 260, 261, 262, 263, 264, 268, 272, 276, 405, 534],
            dtype=torch.int32,
        ),
        seq_lens_cpu=torch.tensor(
            [257, 1057, 1057, 1057, 1057, 1057, 1057, 1057, 2051, 2051, 2051, 4093, 4093],
            dtype=torch.int32,
        ),
    )
    assert plan is not None
    assert plan.num_splits == 2
    assert plan.split_counts == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]


def test_standard_d256_split_planner_rejects_pure_prefill(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            query_start_loc_cpu=torch.tensor([0, 257, 514], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4093, 4093], dtype=torch.int32),
        )
        is None
    )


def test_mla_split_planner_rejects_multitoken_queries(monkeypatch):
    assert (
        _plan(
            monkeypatch,
            query_start_loc_cpu=torch.tensor([0, 1, 5], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([512, 4096], dtype=torch.int32),
            num_heads_q=16,
            num_heads_kv=1,
            head_dim=64,
            head_dim_v=256,
            has_qv=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("head_dim_v", "seqlen_k", "split_count"),
    ((256, 1, 1), (512, 4096, 32)),
    ids=("dv256", "dv512"),
)
def test_mla_graph_plan_uses_positive_per_request_count(
    monkeypatch, head_dim_v, seqlen_k, split_count
):
    plan = _plan(
        monkeypatch,
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([seqlen_k], dtype=torch.int32),
        num_heads_q=16,
        num_heads_kv=1,
        head_dim=64,
        head_dim_v=head_dim_v,
        has_qv=True,
        cuda_graph_max_num_splits=32,
    )
    assert plan is not None
    assert plan.num_splits == 32
    assert plan.split_counts == [split_count]
