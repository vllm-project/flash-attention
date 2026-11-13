# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import enum


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()
    # FP8-KV (flash_fwd_sm90.py): DequantK is unused; DequantV0/1 sync each WG's V half.
    DequantK = enum.auto()
    DequantV0 = enum.auto()
    DequantV1 = enum.auto()
    # FP8-KV bf16-Q in-place narrow (flash_fwd_sm90.py): 256-thread sync so the full
    # fp16 Q (cast in place from bf16) is visible to both MMA warpgroups before QK.
    NarrowQ = enum.auto()
    # hdim>256: PV split along hdimv across 2 WGs, both read the full P tile.
    PSync = enum.auto()
    # hdim>256: RAW on sP, after the P store + proxy fence, before the PV GEMM.
    PSyncRAW = enum.auto()
    # hdim 512 with single_qk_wg: final per-row O scale from MMA WG 0 to WG 1.
    RowStat = enum.auto()


class NamedBarrierFwdSm100(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    TmemPtr = enum.auto()
    SoftmaxStatsW0 = enum.auto()
    SoftmaxStatsW1 = enum.auto()
    SoftmaxStatsW2 = enum.auto()
    SoftmaxStatsW3 = enum.auto()
    SoftmaxStatsW4 = enum.auto()
    SoftmaxStatsW5 = enum.auto()
    SoftmaxStatsW6 = enum.auto()
    SoftmaxStatsW7 = enum.auto()


class NamedBarrierFwdSm100SwapAB(enum.IntEnum):
    """Barrier IDs of the SM100 SwapAB decode kernel; each *WG2 ID is its *WG1 ID + 1.

    IDs 1-15 plus sync_threads on 0 use the full hardware budget of 16 named barriers.
    """

    TmemPtr = 1
    MaxFull = 2
    MaxEmpty = 3
    SumFullWG1 = 4
    SumFullWG2 = 5
    SumEmptyWG1 = 6
    SumEmptyWG2 = 7
    MaxDone = 8
    ODone = 9
    SumDone = 10
    QKIssued = 11
    PVIssued = 12
    MaxTurnWG1 = 13
    MaxTurnWG2 = 14
    # Separate from MaxDone: reusing one ID lets a fast correction warp arrive twice
    # before all warps have contributed to the first generation.
    StatsReady = 15


class NamedBarrierBwd(enum.IntEnum):
    Epilogue = enum.auto()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PdS = enum.auto()
    dQFullWG0 = enum.auto()
    dQFullWG1 = enum.auto()
    dQFullWG2 = enum.auto()
    dQEmptyWG0 = enum.auto()
    dQEmptyWG1 = enum.auto()
    dQEmptyWG2 = enum.auto()


class NamedBarrierBwdSm100(enum.IntEnum):
    EpilogueWG1 = enum.auto()
    EpilogueWG2 = enum.auto()
    Compute = enum.auto()
    dQaccReduce = enum.auto()
    TmemPtr = enum.auto()


class NamedBarrierFwdSm100_MLA2CTA(enum.IntEnum):
    Epilogue = enum.auto()
    TmemPtr = enum.auto()
    Cpasync = enum.auto()
    Softmax = enum.auto()
    SoftmaxStatsFull = enum.auto()
    SoftmaxStatsEmpty = enum.auto()


class NamedBarrierBwdSm100_MLA2CTA(enum.IntEnum):
    Epilogue = enum.auto()
    TmemPtr = enum.auto()
    Cpasync = enum.auto()
    Softmax = enum.auto()
