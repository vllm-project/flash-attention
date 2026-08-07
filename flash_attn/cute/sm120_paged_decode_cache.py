"""Bounded, lifecycle-safe runtime state for the SM120 decode specialization."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import weakref

import torch


_MAX_ITEMS_PER_STREAM = 8
_T = TypeVar("_T")


def current_stream_key(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    """Return a stream key without retaining the stream or input tensor."""
    device = tensor.device
    if device.type != "cuda":
        return (device.type, device.index, 0)
    stream = torch.cuda.current_stream(device)
    return (device.type, device.index, int(stream.cuda_stream))


def is_current_stream_capturing() -> bool:
    """Keep mutable caches outside graph capture; captured calls use fallback state."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


@dataclass(frozen=True)
class Sm120PagedDecodeHotPlan:
    """A weak-reference-only witness for a safe hot-edge cache hit."""

    tensor_refs: tuple[weakref.ReferenceType[torch.Tensor], ...]
    tensor_signatures: tuple[tuple[object, ...], ...]
    stream_key: tuple[str, int | None, int]
    launcher_identity: tuple[int, int]
    page_size: int
    num_splits: int

    def matches(
        self,
        tensors: tuple[torch.Tensor, ...],
        stream_key: tuple[str, int | None, int],
        launcher_identity: tuple[int, int],
        page_size: int,
        num_splits: int,
    ) -> bool:
        if (
            self.stream_key != stream_key
            or self.launcher_identity != launcher_identity
            or self.page_size != page_size
            or self.num_splits != num_splits
            or len(tensors) != len(self.tensor_refs)
        ):
            return False
        return all(
            ref() is tensor and signature == _tensor_signature(tensor)
            for ref, signature, tensor in zip(
                self.tensor_refs, self.tensor_signatures, tensors, strict=True
            )
        )


class Sm120PagedDecodeRuntimeCache(Generic[_T]):
    """Per-stream LRU cache with no strong reference to caller-owned tensors.

    Workspace tensors are owned by this cache deliberately and are bounded to
    eight entries per stream. Input, output, page-table, and sequence-length
    tensors are represented only by weak references in the hot plan.
    """

    def __init__(self, max_items_per_stream: int = _MAX_ITEMS_PER_STREAM) -> None:
        self.max_items_per_stream = max_items_per_stream
        self._hot_plans: dict[
            tuple[str, int | None, int],
            OrderedDict[tuple[object, ...], Sm120PagedDecodeHotPlan],
        ] = {}
        self._workspaces: dict[
            tuple[str, int | None, int], OrderedDict[tuple[object, ...], _T]
        ] = {}

    def _bounded_put(self, ordered: OrderedDict[tuple[object, ...], _T], key, value) -> None:
        ordered[key] = value
        ordered.move_to_end(key)
        while len(ordered) > self.max_items_per_stream:
            ordered.popitem(last=False)

    def get_hot_plan(
        self,
        plan_key: tuple[object, ...],
        tensors: tuple[torch.Tensor, ...],
        stream_key: tuple[str, int | None, int],
        launcher_identity: tuple[int, int],
        page_size: int,
        num_splits: int,
    ) -> Sm120PagedDecodeHotPlan | None:
        plan = self._hot_plans.get(stream_key, {}).get(plan_key)
        if plan is None or not plan.matches(
            tensors, stream_key, launcher_identity, page_size, num_splits
        ):
            return None
        self._hot_plans[stream_key].move_to_end(plan_key)
        return plan

    def record_hot_plan(
        self,
        plan_key: tuple[object, ...],
        tensors: tuple[torch.Tensor, ...],
        stream_key: tuple[str, int | None, int],
        launcher_identity: tuple[int, int],
        page_size: int,
        num_splits: int,
    ) -> Sm120PagedDecodeHotPlan | None:
        try:
            plan = Sm120PagedDecodeHotPlan(
                tuple(weakref.ref(tensor) for tensor in tensors),
                tuple(_tensor_signature(tensor) for tensor in tensors),
                stream_key,
                launcher_identity,
                page_size,
                num_splits,
            )
        except (RuntimeError, TypeError):
            return None
        plans = self._hot_plans.setdefault(stream_key, OrderedDict())
        self._bounded_put(plans, plan_key, plan)
        return plan

    def get_workspace(
        self,
        workspace_key: tuple[object, ...],
        stream_key: tuple[str, int | None, int],
        factory: Callable[[], _T],
        *,
        allow_cache: bool,
    ) -> _T:
        if not allow_cache:
            return factory()
        workspaces = self._workspaces.setdefault(stream_key, OrderedDict())
        value = workspaces.get(workspace_key)
        if value is None:
            value = factory()
            self._bounded_put(workspaces, workspace_key, value)
        else:
            workspaces.move_to_end(workspace_key)
        return value

    def workspace_count(self, stream_key: tuple[str, int | None, int]) -> int:
        return len(self._workspaces.get(stream_key, ()))

    def hot_plan_count(self, stream_key: tuple[str, int | None, int]) -> int:
        return len(self._hot_plans.get(stream_key, ()))

    def clear(self) -> None:
        self._hot_plans.clear()
        self._workspaces.clear()
