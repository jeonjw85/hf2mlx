from __future__ import annotations

import re
from pathlib import Path

import pytest
from hf2mlx.cli import app
from hf2mlx.errors import GatedModelError
from hf2mlx.hf_utils import HubModel, LocalModel
from hf2mlx.utils import Quant
from typer.testing import CliRunner

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
runner = CliRunner()


def _visible(text: str) -> str:
    return _ANSI.sub("", text)


def test_help_exits_zero_when_invoked() -> None:
    result = runner.invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    out = _visible(result.stdout)
    assert "hf2mlx" in out
    assert "--format" in out
    assert "--quant" in out
    assert "--estimate" in out


def test_invalid_quant_fails_with_clear_message() -> None:
    result = runner.invoke(app, ["Qwen/Qwen2.5-3B-Instruct", "--quant", "3bit"])
    assert result.exit_code != 0
    combined = f"{result.stdout}{result.stderr}"
    assert "4bit" in combined or "invalid" in combined.lower()


def test_invalid_format_fails_with_clear_message() -> None:
    result = runner.invoke(app, ["Qwen/Qwen2.5-3B-Instruct", "--format", "onnx"])
    assert result.exit_code != 0
    combined = f"{result.stdout}{result.stderr}"
    assert "mlx" in combined.lower() or "invalid" in combined.lower()


def test_gguf_fails_without_converting(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"convert": False}

    def fake_convert(hf_path: Path, out: Path, quant: Quant) -> None:
        called["convert"] = True

    monkeypatch.setattr("hf2mlx.cli.convert_to_mlx", fake_convert)
    result = runner.invoke(
        app,
        ["Qwen/Qwen2.5-3B-Instruct", "--format", "gguf"],
    )
    assert result.exit_code == 1
    assert "GGUF" in result.stdout
    assert called["convert"] is False


def test_estimate_does_not_convert(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"convert": False, "download": False}

    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=7_000_000_000)

    def fake_convert(hf_path: Path, out: Path, quant: Quant) -> None:
        called["convert"] = True

    def fake_download(repo_id: str, token: str | None) -> Path:
        called["download"] = True
        return Path("/var/empty/fake-model")

    monkeypatch.setattr("hf2mlx.cli.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.cli.convert_to_mlx", fake_convert)
    monkeypatch.setattr("hf2mlx.cli.download_model", fake_download)
    result = runner.invoke(app, ["Qwen/Qwen2.5-7B-Instruct", "--estimate"])
    assert result.exit_code == 0
    assert "Qwen/Qwen2.5-7B-Instruct" in result.stdout
    assert "MLX" in result.stdout
    assert "4bit" in result.stdout
    assert "GB" in result.stdout
    assert "Done." not in result.stdout
    assert called["convert"] is False
    assert called["download"] is False


def test_existing_output_without_force_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out = tmp_path / "already"
    out.mkdir()
    (out / "marker.txt").write_text("keep", encoding="utf-8")

    def fake_resolve(model: str, token: str | None) -> LocalModel:
        return LocalModel(label=model, path=tmp_path, param_count=1_000_000)

    monkeypatch.setattr("hf2mlx.cli.resolve_model", fake_resolve)
    result = runner.invoke(app, ["dummy-model", "--out", str(out)])
    assert result.exit_code == 1
    assert "already exists" in result.stdout
    assert (out / "marker.txt").is_file()


def test_force_overwrites_and_converts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out = tmp_path / "already"
    out.mkdir()
    (out / "old.txt").write_text("old", encoding="utf-8")
    source = tmp_path / "src"
    source.mkdir()

    def fake_resolve(model: str, token: str | None) -> LocalModel:
        return LocalModel(label=model, path=source, param_count=1_000_000)

    def fake_convert(hf_path: Path, dest: Path, quant: Quant) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        (dest / "weights.bin").write_bytes(b"\x00" * 2048)

    monkeypatch.setattr("hf2mlx.cli.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.cli.convert_to_mlx", fake_convert)
    result = runner.invoke(app, ["dummy-model", "--out", str(out), "--force"])
    assert result.exit_code == 0, result.stdout
    assert "Done." in result.stdout
    assert not (out / "old.txt").exists()
    assert (out / "config.json").is_file()


def test_gated_model_prints_token_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str, token: str | None) -> HubModel:
        raise GatedModelError(model=model)

    monkeypatch.setattr("hf2mlx.cli.resolve_model", boom)
    result = runner.invoke(app, ["meta-llama/Llama-3.2-3B-Instruct", "--estimate"])
    assert result.exit_code == 1
    assert "gated" in result.stdout
    assert "HF_TOKEN" in result.stdout
    assert "Traceback" not in result.stdout


def test_unexpected_error_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str, token: str | None) -> HubModel:
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("hf2mlx.cli.resolve_model", boom)
    result = runner.invoke(app, ["Qwen/Qwen2.5-3B-Instruct", "--estimate"])
    assert result.exit_code == 1
    assert "unexpected failure" in result.stdout
    assert "disk exploded" in result.stdout
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr


def test_convert_stops_when_ram_is_too_small(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    called = {"convert": False}

    def fake_resolve(model: str, token: str | None) -> LocalModel:
        return LocalModel(label=model, path=source, param_count=2_000_000_000)

    def fake_convert(hf_path: Path, dest: Path, quant: Quant) -> None:
        called["convert"] = True

    monkeypatch.setattr("hf2mlx.cli.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.cli.convert_to_mlx", fake_convert)
    monkeypatch.setattr("hf2mlx.cli.total_ram_bytes", lambda: 100)
    result = runner.invoke(app, ["dummy-model", "--out", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "insufficient memory" in result.stdout
    assert called["convert"] is False
