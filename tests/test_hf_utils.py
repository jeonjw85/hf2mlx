from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from hf2mlx.errors import (
    GatedModelError,
    InvalidModelIdError,
    LocalModelNotDirError,
    LocalModelNotFoundError,
    ModelNotFoundError,
    NetworkError,
)
from hf2mlx.hf_utils import (
    LocalModel,
    hub_param_count,
    local_param_count,
    looks_like_local_path,
    resolve_model,
)
from huggingface_hub.errors import (
    GatedRepoError,
    HFValidationError,
    RepositoryNotFoundError,
)


def test_looks_like_local_path_for_relative_and_absolute() -> None:
    assert looks_like_local_path("./models/foo") is True
    assert looks_like_local_path("/Users/demo/models/foo") is True
    assert looks_like_local_path("Qwen/Qwen2.5-7B-Instruct") is False


def test_local_param_count_from_index_total_size(tmp_path: Path) -> None:
    index = {
        "metadata": {"total_size": 20},
        "weight_map": {},
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    assert local_param_count(tmp_path) == 10


def test_local_param_count_from_weight_files(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 10)
    assert local_param_count(tmp_path) == 5


def test_resolve_missing_local_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(LocalModelNotFoundError):
        resolve_model(str(missing), token=None)


def test_hub_param_count_falls_back_to_weight_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True, slots=True)
    class Sibling:
        rfilename: str
        size: int

    @dataclass(frozen=True, slots=True)
    class Info:
        safetensors: None
        siblings: list[Sibling]

    def fake_info(
        repo_id: str,
        token: str | None = None,
        files_metadata: bool = False,
    ) -> Info:
        assert repo_id == "org/tiny"
        assert files_metadata is True
        return Info(
            safetensors=None,
            siblings=[
                Sibling(rfilename="config.json", size=100),
                Sibling(rfilename="pytorch_model.bin", size=20),
            ],
        )

    monkeypatch.setattr("hf2mlx.hf_utils.model_info", fake_info)
    assert hub_param_count("org/tiny", token=None) == 10


def test_resolve_empty_model_id_raises() -> None:
    with pytest.raises(InvalidModelIdError):
        resolve_model("   ", token=None)


def test_resolve_file_is_not_a_directory(tmp_path: Path) -> None:
    blob = tmp_path / "weights.bin"
    blob.write_bytes(b"\x00")
    with pytest.raises(LocalModelNotDirError):
        resolve_model(str(blob), token=None)


def test_hub_oserror_becomes_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(
        repo_id: str,
        token: str | None = None,
        files_metadata: bool = False,
    ) -> None:
        raise OSError("timed out")

    monkeypatch.setattr("hf2mlx.hf_utils.model_info", boom)
    with pytest.raises(NetworkError, match="timed out"):
        hub_param_count("org/tiny", token=None)


def test_hub_validation_error_becomes_invalid_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(
        repo_id: str,
        token: str | None = None,
        files_metadata: bool = False,
    ) -> None:
        raise HFValidationError("invalid repo id")

    monkeypatch.setattr("hf2mlx.hf_utils.model_info", boom)
    with pytest.raises(InvalidModelIdError):
        hub_param_count("not a repo", token=None)


def test_hub_gated_repo_becomes_gated_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(
        repo_id: str,
        token: str | None = None,
        files_metadata: bool = False,
    ) -> None:
        request = httpx.Request("GET", "https://huggingface.co/api/models/x")
        response = httpx.Response(status_code=403, request=request)
        raise GatedRepoError("gated", response=response)

    monkeypatch.setattr("hf2mlx.hf_utils.model_info", boom)
    with pytest.raises(GatedModelError):
        hub_param_count("meta-llama/secret", token=None)


def test_hub_missing_repo_becomes_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(
        repo_id: str,
        token: str | None = None,
        files_metadata: bool = False,
    ) -> None:
        request = httpx.Request("GET", "https://huggingface.co/api/models/x")
        response = httpx.Response(status_code=404, request=request)
        raise RepositoryNotFoundError("org/missing", response=response)

    monkeypatch.setattr("hf2mlx.hf_utils.model_info", boom)
    with pytest.raises(ModelNotFoundError):
        hub_param_count("org/missing", token=None)


def test_broken_index_json_returns_none(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text("{not json", encoding="utf-8")
    assert local_param_count(tmp_path) is None


def test_resolve_existing_local_dir(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 8)
    resolved = resolve_model(str(tmp_path), token=None)
    assert isinstance(resolved, LocalModel)
    assert resolved.param_count == 4
