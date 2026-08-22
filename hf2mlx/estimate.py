from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from typing_extensions import assert_never

from hf2mlx.arch import ModelArch, kv_cache_bytes
from hf2mlx.utils import OutputFormat, Quant

_INFERENCE_OVERHEAD: Final = 1.15
DEFAULT_CTX: Final = 4096


@dataclass(frozen=True, slots=True)
class BytesPerParam:
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class SizeEstimate:
    param_count: int
    size_low_bytes: int
    size_high_bytes: int
    size_mid_bytes: int
    inference_low_bytes: int
    inference_high_bytes: int
    kv_bytes: int
    ctx: int


def bytes_per_param(quant: Quant) -> BytesPerParam:
    match quant:
        case Quant.FOUR_BIT:
            return BytesPerParam(low=0.55, high=0.7)
        case Quant.EIGHT_BIT:
            return BytesPerParam(low=1.0, high=1.2)
        case Quant.BF16:
            return BytesPerParam(low=2.0, high=2.0)
        case _ as unreachable:
            assert_never(unreachable)


def bytes_per_param_gguf(quant: Quant) -> BytesPerParam:
    match quant:
        case Quant.FOUR_BIT:
            return BytesPerParam(low=0.50, high=0.65)
        case Quant.EIGHT_BIT:
            return BytesPerParam(low=1.0, high=1.1)
        case Quant.BF16:
            return BytesPerParam(low=2.0, high=2.0)
        case _ as unreachable:
            assert_never(unreachable)


def estimate_mlx(
    param_count: int,
    quant: Quant,
    ctx: int = DEFAULT_CTX,
    arch: ModelArch | None = None,
) -> SizeEstimate:
    return _estimate(param_count, bytes_per_param(quant), ctx, arch)


def estimate_gguf(
    param_count: int,
    quant: Quant,
    ctx: int = DEFAULT_CTX,
    arch: ModelArch | None = None,
) -> SizeEstimate:
    return _estimate(param_count, bytes_per_param_gguf(quant), ctx, arch)


def estimate_for(
    param_count: int,
    quant: Quant,
    fmt: OutputFormat,
    ctx: int = DEFAULT_CTX,
    arch: ModelArch | None = None,
) -> SizeEstimate:
    match fmt:
        case OutputFormat.MLX:
            return estimate_mlx(param_count, quant, ctx, arch)
        case OutputFormat.GGUF:
            return estimate_gguf(param_count, quant, ctx, arch)
        case _ as unreachable:
            assert_never(unreachable)


def _estimate(
    param_count: int,
    bpp: BytesPerParam,
    ctx: int,
    arch: ModelArch | None,
) -> SizeEstimate:
    size_low = int(param_count * bpp.low)
    size_high = int(param_count * bpp.high)
    size_mid = (size_low + size_high) // 2
    kv = kv_cache_bytes(arch, param_count, ctx)
    return SizeEstimate(
        param_count=param_count,
        size_low_bytes=size_low,
        size_high_bytes=size_high,
        size_mid_bytes=size_mid,
        inference_low_bytes=size_mid + kv,
        inference_high_bytes=int(size_high * _INFERENCE_OVERHEAD) + kv,
        kv_bytes=kv,
        ctx=ctx,
    )


def source_weight_bytes(param_count: int) -> int:
    return int(param_count * 2.0)
