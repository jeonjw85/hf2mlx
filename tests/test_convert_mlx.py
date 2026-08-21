from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hf2mlx.convert_mlx import convert_to_mlx, quant_to_mlx_args
from hf2mlx.errors import ConversionFailedError, UnsupportedArchitectureError
from hf2mlx.utils import Quant


def test_four_bit_enables_quantization() -> None:
    args = quant_to_mlx_args(Quant.FOUR_BIT)
    assert args.quantize is True
    assert args.q_bits == 4
    assert args.dtype is None


def test_eight_bit_enables_quantization() -> None:
    args = quant_to_mlx_args(Quant.EIGHT_BIT)
    assert args.quantize is True
    assert args.q_bits == 8


def test_bf16_disables_quantization() -> None:
    args = quant_to_mlx_args(Quant.BF16)
    assert args.quantize is False
    assert args.q_bits is None
    assert args.dtype == "bfloat16"


def _install_fake_mlx(
    monkeypatch: pytest.MonkeyPatch,
    convert_fn: object,
) -> None:
    fake = SimpleNamespace(convert=convert_fn)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)


def test_value_error_becomes_unsupported_architecture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def convert(*args: object, **kwargs: object) -> None:
        raise ValueError("Model type foo not supported")

    _install_fake_mlx(monkeypatch, convert)
    with pytest.raises(UnsupportedArchitectureError, match="foo"):
        convert_to_mlx(tmp_path, tmp_path / "out", Quant.FOUR_BIT)


def test_runtime_error_becomes_conversion_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def convert(*args: object, **kwargs: object) -> None:
        raise RuntimeError("metal OOM")

    _install_fake_mlx(monkeypatch, convert)
    with pytest.raises(ConversionFailedError, match="metal OOM"):
        convert_to_mlx(tmp_path, tmp_path / "out", Quant.FOUR_BIT)
