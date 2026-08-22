from __future__ import annotations

from typing import Final

from hf2mlx.arch import ModelArch
from hf2mlx.errors import DoesNotFitError
from hf2mlx.estimate import estimate_for
from hf2mlx.utils import OutputFormat, Quant, format_bytes

_RESERVE_BYTES: Final = 7 * 1024**3
_QUANT_PREF: Final = (Quant.BF16, Quant.EIGHT_BIT, Quant.FOUR_BIT)


def usable_ram_bytes(total_ram: int) -> int:
    return max(0, total_ram - _RESERVE_BYTES)


def fit_quant(
    param_count: int,
    fmt: OutputFormat,
    ram_bytes: int,
    ctx: int,
    arch: ModelArch | None = None,
) -> Quant:
    usable = usable_ram_bytes(ram_bytes)
    last_high = 0
    for quant in _QUANT_PREF:
        est = estimate_for(param_count, quant, fmt, ctx, arch)
        last_high = est.inference_high_bytes
        if est.inference_high_bytes <= usable:
            return quant
    raise DoesNotFitError(
        needed_label=format_bytes(last_high),
        available_label=format_bytes(usable),
    )
