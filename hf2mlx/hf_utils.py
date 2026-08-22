from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, TypeVar

from huggingface_hub import (
    ModelInfo,
    hf_hub_download,
    model_info,
    repo_exists,
    snapshot_download,
)
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)

from hf2mlx.errors import (
    GatedModelError,
    HubRequestError,
    InvalidModelIdError,
    LocalModelNotDirError,
    LocalModelNotFoundError,
    ModelNotFoundError,
    NetworkError,
)

_WEIGHT_SUFFIXES: Final = frozenset({".safetensors", ".bin", ".pt", ".npz", ".gguf"})
_INDEX_NAMES: Final = (
    "model.safetensors.index.json",
    "model.safetensors.index.fp32.json",
)
_AUTH_STATUSES: Final = frozenset({401, 403})
_NOT_FOUND: Final = 404

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LocalModel:
    label: str
    path: Path
    param_count: int | None


@dataclass(frozen=True, slots=True)
class HubModel:
    label: str
    repo_id: str
    param_count: int | None


ResolvedModel = LocalModel | HubModel


def looks_like_local_path(model: str) -> bool:
    expanded = Path(model).expanduser()
    if expanded.exists():
        return True
    return model.startswith(("./", "/", "../", "~/"))


def resolve_model(model: str, token: str | None) -> ResolvedModel:
    stripped = model.strip()
    if stripped == "":
        raise InvalidModelIdError(model=model)
    expanded = Path(stripped).expanduser()
    if expanded.exists():
        if not expanded.is_dir():
            raise LocalModelNotDirError(path=expanded)
        resolved = expanded.resolve()
        return LocalModel(
            label=model,
            path=resolved,
            param_count=local_param_count(resolved),
        )
    if looks_like_local_path(stripped):
        raise LocalModelNotFoundError(path=expanded)
    return HubModel(
        label=model,
        repo_id=stripped,
        param_count=hub_param_count(stripped, token),
    )


def download_model(repo_id: str, token: str | None) -> Path:
    downloader = partial(snapshot_download, repo_id=repo_id, token=token)
    cached = _call_hub(repo_id, downloader)
    return Path(cached)


def download_to(repo_id: str, token: str | None, dest: Path) -> Path:
    downloader = partial(
        snapshot_download,
        repo_id=repo_id,
        token=token,
        local_dir=str(dest),
    )
    cached = _call_hub(repo_id, downloader)
    return Path(cached)


def hub_repo_available(repo_id: str, token: str | None) -> bool:
    return _call_hub(
        repo_id,
        partial(repo_exists, repo_id, token=token, repo_type="model"),
    )


def download_config(repo_id: str, token: str | None) -> Path | None:
    downloader = partial(
        hf_hub_download,
        repo_id=repo_id,
        filename="config.json",
        token=token,
    )
    try:
        return Path(_call_hub(repo_id, downloader))
    except ModelNotFoundError:
        return None
    except HubRequestError as exc:
        if exc.status == _NOT_FOUND:
            return None
        raise


def hub_param_count(repo_id: str, token: str | None) -> int | None:
    info = _fetch_info(repo_id, token)
    count = _count_from_safetensors(info)
    if count is not None:
        return count
    return _count_from_siblings(info)


def _fetch_info(repo_id: str, token: str | None) -> ModelInfo:
    return _call_hub(
        repo_id,
        partial(model_info, repo_id, token=token, files_metadata=True),
    )


def _call_hub(repo_id: str, op: Callable[[], T]) -> T:
    try:
        return op()
    except GatedRepoError as exc:
        raise GatedModelError(model=repo_id) from exc
    except RepositoryNotFoundError as exc:
        raise ModelNotFoundError(model=repo_id) from exc
    except HFValidationError as exc:
        raise InvalidModelIdError(model=repo_id) from exc
    except HfHubHTTPError as exc:
        status = exc.response.status_code
        if status in _AUTH_STATUSES:
            raise GatedModelError(model=repo_id) from exc
        raise HubRequestError(model=repo_id, status=status) from exc
    except OSError as exc:
        raise NetworkError(reason=str(exc)) from exc


def _count_from_safetensors(info: ModelInfo) -> int | None:
    safetensors = info.safetensors
    if safetensors is None:
        return None
    total = safetensors.total
    if total <= 0:
        return None
    return total


def _count_from_siblings(info: ModelInfo) -> int | None:
    siblings = info.siblings
    if siblings is None:
        return None
    total = 0
    for sib in siblings:
        suffix = Path(sib.rfilename).suffix
        if suffix not in _WEIGHT_SUFFIXES:
            continue
        size = sib.size
        if size is None:
            continue
        total += size
    if total <= 0:
        return None
    return total // 2


def local_param_count(model_dir: Path) -> int | None:
    for name in _INDEX_NAMES:
        index_path = model_dir / name
        if index_path.is_file():
            count = _params_from_index(index_path)
            if count is not None:
                return count
    weight_bytes = _weight_bytes(model_dir)
    if weight_bytes > 0:
        return weight_bytes // 2
    return None


def _params_from_index(index_path: Path) -> int | None:
    try:
        raw: object = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        return None
    total_size = metadata.get("total_size")
    if isinstance(total_size, bool) or not isinstance(total_size, int):
        return None
    if total_size <= 0:
        return None
    return total_size // 2


def _weight_bytes(model_dir: Path) -> int:
    try:
        children = list(model_dir.iterdir())
    except OSError:
        return 0
    total = 0
    for child in children:
        if child.is_file() and child.suffix in _WEIGHT_SUFFIXES:
            total += child.stat().st_size
    return total
