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


def _no_config(repo_id: str, token: str | None) -> Path | None:
    return None


def _no_ready(repo_id: str, quant: Quant, token: str | None) -> str | None:
    return None


def _ready_dummy(repo_id: str, quant: Quant, token: str | None) -> str | None:
    return "mlx-community/dummy-4bit"


def test_help_exits_zero_when_invoked() -> None:
    result = runner.invoke(app, ["--help"], color=False)
    assert result.exit_code == 0
    out = _visible(result.stdout)
    assert "hf2mlx" in out
    assert "--format" in out
    assert "--quant" in out
    assert "--estimate" in out
    assert "--fit" in out
    assert "--ctx" in out
    assert "--rebuild" in out


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


def test_gguf_estimate_does_not_convert(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"gguf": False, "mlx": False}

    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=7_000_000_000)

    def fake_gguf(hf_path: Path, out: Path, quant: Quant) -> None:
        called["gguf"] = True

    def fake_mlx(hf_path: Path, out: Path, quant: Quant) -> None:
        called["mlx"] = True

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.convert_to_gguf", fake_gguf)
    monkeypatch.setattr("hf2mlx.job.convert_to_mlx", fake_mlx)
    result = runner.invoke(
        app,
        ["Qwen/Qwen2.5-7B-Instruct", "--format", "gguf", "--estimate"],
    )
    assert result.exit_code == 0
    assert "GGUF" in result.stdout
    assert called["gguf"] is False
    assert called["mlx"] is False


def test_gguf_convert_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()

    def fake_resolve(model: str, token: str | None) -> LocalModel:
        return LocalModel(label=model, path=source, param_count=1_000_000)

    def fake_gguf(hf_path: Path, dest: Path, quant: Quant) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.gguf").write_bytes(b"\x00" * 2048)

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.convert_to_gguf", fake_gguf)
    monkeypatch.setattr("hf2mlx.job.total_ram_bytes", lambda: 10_000_000_000)
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["dummy-model", "--format", "gguf", "--out", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert "Done." in result.stdout
    assert "llama-cli" in result.stdout
    assert (out / "model.gguf").is_file()


def test_estimate_does_not_convert(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"convert": False, "download": False}

    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=7_000_000_000)

    def fake_convert(hf_path: Path, out: Path, quant: Quant) -> None:
        called["convert"] = True

    def fake_download(repo_id: str, token: str | None) -> Path:
        called["download"] = True
        return Path("/var/empty/fake-model")

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.convert_to_mlx", fake_convert)
    monkeypatch.setattr("hf2mlx.job.download_model", fake_download)
    monkeypatch.setattr("hf2mlx.job.download_config", _no_config)
    monkeypatch.setattr("hf2mlx.job.find_ready_mlx", _no_ready)
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

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
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

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.convert_to_mlx", fake_convert)
    result = runner.invoke(app, ["dummy-model", "--out", str(out), "--force"])
    assert result.exit_code == 0, result.stdout
    assert "Done." in result.stdout
    assert not (out / "old.txt").exists()
    assert (out / "config.json").is_file()


def test_gated_model_prints_token_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str, token: str | None) -> HubModel:
        raise GatedModelError(model=model)

    monkeypatch.setattr("hf2mlx.job.resolve_model", boom)
    result = runner.invoke(app, ["meta-llama/Llama-3.2-3B-Instruct", "--estimate"])
    assert result.exit_code == 1
    assert "gated" in result.stdout
    assert "HF_TOKEN" in result.stdout
    assert "Traceback" not in result.stdout


def test_unexpected_error_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(model: str, token: str | None) -> HubModel:
        raise RuntimeError("disk exploded")

    monkeypatch.setattr("hf2mlx.job.resolve_model", boom)
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

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.convert_to_mlx", fake_convert)
    monkeypatch.setattr("hf2mlx.job.total_ram_bytes", lambda: 100)
    result = runner.invoke(app, ["dummy-model", "--out", str(tmp_path / "out")])
    assert result.exit_code == 1
    assert "insufficient memory" in result.stdout
    assert called["convert"] is False


def test_fit_estimate_picks_four_bit_for_7b_on_16gb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=7_000_000_000)

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.download_config", _no_config)
    monkeypatch.setattr("hf2mlx.job.find_ready_mlx", _no_ready)
    monkeypatch.setattr("hf2mlx.job.total_ram_bytes", lambda: 16 * 1024**3)
    result = runner.invoke(app, ["Qwen/Qwen2.5-7B-Instruct", "--fit", "--estimate"])
    assert result.exit_code == 0, result.stdout
    assert "4bit" in result.stdout
    assert "bf16" not in result.stdout


def test_estimate_prints_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=7_000_000_000)

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.download_config", _no_config)
    monkeypatch.setattr("hf2mlx.job.find_ready_mlx", _no_ready)
    result = runner.invoke(
        app,
        ["Qwen/Qwen2.5-7B-Instruct", "--estimate", "--ctx", "32768"],
    )
    assert result.exit_code == 0, result.stdout
    assert "32768" in result.stdout


def test_ready_repo_downloads_instead_of_converting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = {"convert": False, "download_to": False}

    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=1_000_000)

    def fake_convert(hf_path: Path, dest: Path, quant: Quant) -> None:
        called["convert"] = True

    def fake_download_to(repo_id: str, token: str | None, dest: Path) -> Path:
        called["download_to"] = True
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        (dest / "weights.safetensors").write_bytes(b"\x00" * 2048)
        return dest

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.download_config", _no_config)
    monkeypatch.setattr("hf2mlx.job.find_ready_mlx", _ready_dummy)
    monkeypatch.setattr("hf2mlx.job.convert_to_mlx", fake_convert)
    monkeypatch.setattr("hf2mlx.job.download_to", fake_download_to)
    monkeypatch.setattr("hf2mlx.job.total_ram_bytes", lambda: 10_000_000_000)
    out = tmp_path / "out"
    result = runner.invoke(app, ["org/dummy", "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "Ready: mlx-community/dummy-4bit" in result.stdout
    assert "Done." in result.stdout
    assert called["convert"] is False
    assert called["download_to"] is True
    assert (out / "config.json").is_file()


def test_rebuild_converts_even_when_ready_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = {"convert": False, "find": False}
    source = tmp_path / "src"
    source.mkdir()

    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=1_000_000)

    def fake_convert(hf_path: Path, dest: Path, quant: Quant) -> None:
        called["convert"] = True
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")

    def fake_find(repo_id: str, quant: Quant, token: str | None) -> str | None:
        called["find"] = True
        return "mlx-community/dummy-4bit"

    def fake_download(repo_id: str, token: str | None) -> Path:
        return source

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.convert_to_mlx", fake_convert)
    monkeypatch.setattr("hf2mlx.job.find_ready_mlx", fake_find)
    monkeypatch.setattr("hf2mlx.job.download_config", _no_config)
    monkeypatch.setattr("hf2mlx.job.download_model", fake_download)
    monkeypatch.setattr("hf2mlx.job.total_ram_bytes", lambda: 10_000_000_000)
    out = tmp_path / "out"
    result = runner.invoke(app, ["org/dummy", "--out", str(out), "--rebuild"])
    assert result.exit_code == 0, result.stdout
    assert called["convert"] is True
    assert called["find"] is False


def test_ready_lookup_skips_gated_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_resolve(model: str, token: str | None) -> HubModel:
        if model.startswith("meta-llama/"):
            raise GatedModelError(model=model)
        return HubModel(label=model, repo_id=model, param_count=1_000_000)

    def fake_download_to(repo_id: str, token: str | None, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        (dest / "weights.safetensors").write_bytes(b"\x00" * 2048)
        return dest

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.download_config", _no_config)
    monkeypatch.setattr("hf2mlx.job.find_ready_mlx", _ready_dummy)
    monkeypatch.setattr("hf2mlx.job.download_to", fake_download_to)
    monkeypatch.setattr("hf2mlx.job.total_ram_bytes", lambda: 10_000_000_000)
    out = tmp_path / "out"
    result = runner.invoke(app, ["meta-llama/secret", "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "Ready: mlx-community/dummy-4bit" in result.stdout


def test_fit_without_ram_prints_short_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_resolve(model: str, token: str | None) -> HubModel:
        return HubModel(label=model, repo_id=model, param_count=7_000_000_000)

    monkeypatch.setattr("hf2mlx.job.resolve_model", fake_resolve)
    monkeypatch.setattr("hf2mlx.job.download_config", _no_config)
    monkeypatch.setattr("hf2mlx.job.total_ram_bytes", lambda: None)
    result = runner.invoke(app, ["Qwen/Qwen2.5-7B-Instruct", "--fit", "--estimate"])
    assert result.exit_code == 1
    assert "system memory" in result.stdout
    assert "--quant" in result.stdout
