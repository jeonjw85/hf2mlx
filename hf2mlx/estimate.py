from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import assert_never

from hf2mlx.utils import Quant

_INFERENCE_LOW: float = 1.4
_INFERENCE_HIGH: float = 1.9


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


def estimate_mlx(param_count: int, quant: Quant) -> SizeEstimate:
    bpp = bytes_per_param(quant)
    size_low = int(param_count * bpp.low)
    size_high = int(param_count * bpp.high)
    size_mid = (size_low + size_high) // 2
    return SizeEstimate(
        param_count=param_count,
        size_low_bytes=size_low,
        size_high_bytes=size_high,
        size_mid_bytes=size_mid,
        inference_low_bytes=int(size_mid * _INFERENCE_LOW),
        inference_high_bytes=int(size_mid * _INFERENCE_HIGH),
    )


def source_weight_bytes(param_count: int) -> int:
    return int(param_count * 2.0)
