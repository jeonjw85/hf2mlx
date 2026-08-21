from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import assert_never

from hf2mlx.errors import (
    ConversionFailedError,
    MlxMissingError,
    UnsupportedArchitectureError,
)
from hf2mlx.utils import Quant


@dataclass(frozen=True, slots=True)
class MlxConvertArgs:
    quantize: bool
    q_bits: int | None
    dtype: str | None


def quant_to_mlx_args(quant: Quant) -> MlxConvertArgs:
    match quant:
        case Quant.FOUR_BIT:
            return MlxConvertArgs(quantize=True, q_bits=4, dtype=None)
        case Quant.EIGHT_BIT:
            return MlxConvertArgs(quantize=True, q_bits=8, dtype=None)
        case Quant.BF16:
            return MlxConvertArgs(quantize=False, q_bits=None, dtype="bfloat16")
        case _ as unreachable:
            assert_never(unreachable)


def convert_to_mlx(hf_path: Path, out: Path, quant: Quant) -> None:
    try:
        from mlx_lm import convert as mlx_convert  # noqa: PLC0415
    except ImportError as exc:
        raise MlxMissingError from exc
    args = quant_to_mlx_args(quant)
    try:
        mlx_convert(
            str(hf_path),
            mlx_path=str(out),
            quantize=args.quantize,
            q_bits=args.q_bits,
            dtype=args.dtype,
        )
    except ValueError as exc:
        raise UnsupportedArchitectureError(reason=str(exc)) from exc
    except FileNotFoundError as exc:
        raise ConversionFailedError(reason=f"model files missing: {exc}") from exc
    except (OSError, RuntimeError) as exc:
        raise ConversionFailedError(reason=str(exc)) from exc
