"""Public contract for the bounded SM120 paged-decode specialization."""

from dataclasses import dataclass
import os

import torch


_PAGE_SIZES = (16, 784)
_SPLITS = (16, 32, 48)


@dataclass(frozen=True)
class Sm120PagedDecodeD256Metadata:
    """Runtime facts used to decide whether the specialization is available."""

    capability: int
    dtype: torch.dtype
    batch_size: int
    query_tokens: int
    query_heads: int
    kv_heads: int
    head_dim: int
    page_size: int
    kv_tokens: int
    paged_kv: bool
    has_cu_seqlens_q: bool
    has_seqused_k: bool
    causal_or_full_decode: bool = True
    has_unsupported_feature: bool = False


@dataclass(frozen=True)
class Sm120PagedDecodeCompileUnit:
    """One exact, bounded compile request with its runtime witness."""

    kernel: str
    page_size: int
    num_splits: int
    kv_tokens_witness: int


def select_sm120_paged_decode_d256_num_splits(kv_tokens: int) -> int:
    """Select from the only supported SplitKV variants: 1/16/32/48."""
    if kv_tokens < 0:
        raise ValueError("kv_tokens must be non-negative")
    if kv_tokens <= 256:
        return 1
    if kv_tokens <= 512:
        return 16
    if kv_tokens <= 1024:
        return 32
    return 48


def sm120_paged_decode_d256_enabled() -> bool:
    """Return whether the experimental vendor specialization is explicitly enabled."""
    return os.environ.get("FLASH_ATTENTION_SM120_DECODE_KERNEL", "0").lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def sm120_paged_decode_d256_available(metadata: Sm120PagedDecodeD256Metadata) -> bool:
    """Fail closed unless every supported geometry and feature condition holds."""
    return (
        metadata.capability == 120
        and metadata.dtype == torch.bfloat16
        and metadata.batch_size == 1
        and metadata.query_tokens == 1
        and metadata.query_heads == 24
        and metadata.kv_heads == 4
        and metadata.head_dim == 256
        and metadata.page_size in _PAGE_SIZES
        and metadata.kv_tokens > 256
        and metadata.paged_kv
        and metadata.has_cu_seqlens_q
        and metadata.has_seqused_k
        and metadata.causal_or_full_decode
        and not metadata.has_unsupported_feature
    )


def build_sm120_paged_decode_d256_compile_plan(
    metadata: Sm120PagedDecodeD256Metadata,
) -> tuple[Sm120PagedDecodeCompileUnit, ...]:
    """Return the forward/combine pair for one selector-derived witness."""
    if not sm120_paged_decode_d256_available(metadata):
        raise ValueError("metadata is outside the SM120 BF16 D256 paged-decode contract")
    num_splits = select_sm120_paged_decode_d256_num_splits(metadata.kv_tokens)
    if num_splits not in _SPLITS:
        raise ValueError("metadata does not select a compiled SplitKV specialization")
    return (
        Sm120PagedDecodeCompileUnit(
            kernel="sm120_paged_decode_d256_forward",
            page_size=metadata.page_size,
            num_splits=num_splits,
            kv_tokens_witness=metadata.kv_tokens,
        ),
        Sm120PagedDecodeCompileUnit(
            kernel="sm120_splitkv_combine",
            page_size=metadata.page_size,
            num_splits=num_splits,
            kv_tokens_witness=metadata.kv_tokens,
        ),
    )


def build_sm120_paged_decode_d256_warmup_plan(
    metadata: Sm120PagedDecodeD256Metadata, max_kv_tokens: int
) -> tuple[Sm120PagedDecodeCompileUnit, ...]:
    """Enumerate at most three selector witnesses up to an engine's K bound.

    The witnesses are selector boundary representatives, not a length/page/batch
    Cartesian expansion.  Every returned forward unit is paired with its exact
    SplitKV combine unit.
    """
    if max_kv_tokens < 0:
        raise ValueError("max_kv_tokens must be non-negative")
    units = []
    for witness in (257, 513, 1025):
        if witness > max_kv_tokens:
            continue
        witness_metadata = Sm120PagedDecodeD256Metadata(
            capability=metadata.capability,
            dtype=metadata.dtype,
            batch_size=metadata.batch_size,
            query_tokens=metadata.query_tokens,
            query_heads=metadata.query_heads,
            kv_heads=metadata.kv_heads,
            head_dim=metadata.head_dim,
            page_size=metadata.page_size,
            kv_tokens=witness,
            paged_kv=metadata.paged_kv,
            has_cu_seqlens_q=metadata.has_cu_seqlens_q,
            has_seqused_k=metadata.has_seqused_k,
            causal_or_full_decode=metadata.causal_or_full_decode,
            has_unsupported_feature=metadata.has_unsupported_feature,
        )
        units.extend(build_sm120_paged_decode_d256_compile_plan(witness_metadata))
    return tuple(units)
