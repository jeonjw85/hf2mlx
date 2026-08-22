from __future__ import annotations

import pytest
from hf2mlx.errors import DoesNotFitError
from hf2mlx.fit import fit_quant
from hf2mlx.utils import OutputFormat, Quant

_16_GB = 16 * 1024**3
_24_GB = 24 * 1024**3
_8_GB = 8 * 1024**3


def test_fit_picks_four_bit_for_7b_on_16gb() -> None:
    quant = fit_quant(
        7_000_000_000,
        OutputFormat.MLX,
        ram_bytes=_16_GB,
        ctx=4096,
    )
    assert quant is Quant.FOUR_BIT


def test_fit_picks_bf16_for_3b_on_16gb() -> None:
    quant = fit_quant(
        3_000_000_000,
        OutputFormat.MLX,
        ram_bytes=_16_GB,
        ctx=4096,
    )
    assert quant is Quant.BF16


def test_fit_picks_eight_bit_or_better_for_7b_on_24gb() -> None:
    quant = fit_quant(
        7_000_000_000,
        OutputFormat.MLX,
        ram_bytes=_24_GB,
        ctx=4096,
    )
    assert quant in {Quant.EIGHT_BIT, Quant.BF16}


def test_fit_raises_when_even_four_bit_overflows() -> None:
    with pytest.raises(DoesNotFitError):
        fit_quant(
            7_000_000_000,
            OutputFormat.MLX,
            ram_bytes=_8_GB,
            ctx=4096,
        )


def test_fit_raises_when_long_ctx_blows_16gb_7b() -> None:
    with pytest.raises(DoesNotFitError):
        fit_quant(
            7_000_000_000,
            OutputFormat.MLX,
            ram_bytes=_16_GB,
            ctx=131072,
        )
