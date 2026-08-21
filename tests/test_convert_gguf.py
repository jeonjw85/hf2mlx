from __future__ import annotations

from pathlib import Path

import pytest
from hf2mlx.convert_gguf import convert_to_gguf, quant_to_gguf_plan
from hf2mlx.errors import GgufToolsMissingError
from hf2mlx.utils import Quant


def test_four_bit_plan_uses_q4_k_m() -> None:
    plan = quant_to_gguf_plan(Quant.FOUR_BIT)
    assert plan.outtype == "bf16"
    assert plan.quantize_type == "Q4_K_M"


def test_eight_bit_plan_writes_q8_directly() -> None:
    plan = quant_to_gguf_plan(Quant.EIGHT_BIT)
    assert plan.outtype == "q8_0"
    assert plan.quantize_type is None


def test_bf16_plan_skips_llama_quantize() -> None:
    plan = quant_to_gguf_plan(Quant.BF16)
    assert plan.outtype == "bf16"
    assert plan.quantize_type is None


def _stub_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    converter = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.write_text("# converter", encoding="utf-8")
    quantize = tmp_path / "llama-quantize"
    quantize.write_text("#!/bin/sh\n", encoding="utf-8")
    recorded: list[list[str]] = []

    def fake_run(argv: list[str], cwd: Path | None = None) -> None:
        recorded.append(list(argv))
        if "--outfile" in argv:
            dest = Path(argv[argv.index("--outfile") + 1])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"raw")
            return
        Path(argv[2]).write_bytes(b"quant")

    monkeypatch.setattr("hf2mlx.convert_gguf._ensure_converter", lambda: converter)
    monkeypatch.setattr("hf2mlx.convert_gguf._resolve_quantize", lambda: quantize)
    monkeypatch.setattr("hf2mlx.convert_gguf._run", fake_run)
    return recorded


def test_bf16_convert_writes_model_gguf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded = _stub_tools(monkeypatch, tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    convert_to_gguf(src, out, Quant.BF16)
    assert (out / "model.gguf").is_file()
    assert len(recorded) == 1
    assert "--outtype" in recorded[0]
    assert "bf16" in recorded[0]


def test_four_bit_convert_calls_llama_quantize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded = _stub_tools(monkeypatch, tmp_path)
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    convert_to_gguf(src, out, Quant.FOUR_BIT)
    assert (out / "model.gguf").read_bytes() == b"quant"
    assert any("Q4_K_M" in argv for argv in recorded)


def test_four_bit_without_quantize_binary_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    converter = tmp_path / "convert_hf_to_gguf.py"
    converter.write_text("# converter", encoding="utf-8")
    monkeypatch.setattr("hf2mlx.convert_gguf._ensure_converter", lambda: converter)
    def no_which(_name: str) -> str | None:
        return None

    monkeypatch.setattr("hf2mlx.convert_gguf.shutil.which", no_which)

    def fake_run(argv: list[str], cwd: Path | None = None) -> None:
        dest = Path(argv[argv.index("--outfile") + 1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"raw")

    monkeypatch.setattr("hf2mlx.convert_gguf._run", fake_run)
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(GgufToolsMissingError, match="llama-quantize"):
        convert_to_gguf(src, tmp_path / "out", Quant.FOUR_BIT)
