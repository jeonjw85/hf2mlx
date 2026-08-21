from __future__ import annotations

from hf2mlx.estimate import (
    bytes_per_param,
    bytes_per_param_gguf,
    estimate_gguf,
    estimate_mlx,
)
from hf2mlx.utils import Quant


def test_four_bit_uses_half_to_point_seven_bytes_per_param() -> None:
    bpp = bytes_per_param(Quant.FOUR_BIT)
    assert bpp.low == 0.55
    assert bpp.high == 0.7


def test_eight_bit_uses_one_to_one_point_two_bytes_per_param() -> None:
    bpp = bytes_per_param(Quant.EIGHT_BIT)
    assert bpp.low == 1.0
    assert bpp.high == 1.2


def test_bf16_uses_two_bytes_per_param() -> None:
    bpp = bytes_per_param(Quant.BF16)
    assert bpp.low == 2.0
    assert bpp.high == 2.0


def test_seven_b_four_bit_size_is_about_four_gb() -> None:
    est = estimate_mlx(7_000_000_000, Quant.FOUR_BIT)
    assert 3_500_000_000 <= est.size_mid_bytes <= 5_000_000_000
    assert est.inference_low_bytes > est.size_mid_bytes
    assert est.inference_high_bytes > est.inference_low_bytes


def test_gguf_four_bit_is_q4_k_m_sized() -> None:
    bpp = bytes_per_param_gguf(Quant.FOUR_BIT)
    assert bpp.low == 0.50
    assert bpp.high == 0.65
    est = estimate_gguf(7_000_000_000, Quant.FOUR_BIT)
    mlx = estimate_mlx(7_000_000_000, Quant.FOUR_BIT)
    assert est.size_mid_bytes < mlx.size_mid_bytes
