from __future__ import annotations

from pathlib import Path

import pytest
from hf2mlx.errors import OutputExistsError
from hf2mlx.utils import (
    OutputFormat,
    Quant,
    default_out_dir,
    dir_size,
    format_bytes,
    format_gb_range,
    model_basename,
    prepare_out_dir,
)


def test_model_basename_strips_org_prefix() -> None:
    assert model_basename("Qwen/Qwen2.5-7B-Instruct") == "Qwen2.5-7B-Instruct"


def test_default_out_dir_includes_format_and_quant() -> None:
    path = default_out_dir("Qwen/Qwen2.5-7B-Instruct", OutputFormat.MLX, Quant.FOUR_BIT)
    assert path == Path("converted/Qwen2.5-7B-Instruct-mlx-4bit")


def test_format_bytes_uses_gb_for_billions() -> None:
    assert format_bytes(4_100_000_000) == "4.1 GB"


def test_format_gb_range_includes_ctx() -> None:
    text = format_gb_range(6_000_000_000, 8_000_000_000, ctx=4096)
    assert "6" in text
    assert "8" in text
    assert "GB" in text
    assert "4096" in text


def test_format_gb_range_uses_mb_when_under_one_gb() -> None:
    text = format_gb_range(400_000_000, 600_000_000)
    assert "MB" in text
    assert "400" in text
    assert "600" in text


def test_prepare_out_dir_raises_without_force(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(OutputExistsError):
        prepare_out_dir(target, force=False)


def test_prepare_out_dir_removes_with_force(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    (target / "file.bin").write_bytes(b"x")
    prepare_out_dir(target, force=True)
    assert not target.exists()


def test_dir_size_sums_nested_files(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "w.bin").write_bytes(b"\x00" * 100)
    (tmp_path / "top.bin").write_bytes(b"\x00" * 50)
    assert dir_size(tmp_path) == 150
