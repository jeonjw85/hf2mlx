from __future__ import annotations

from collections.abc import Callable
from typing import Final

from typing_extensions import assert_never

from hf2mlx.errors import ModelNotFoundError
from hf2mlx.hf_utils import hub_repo_available
from hf2mlx.utils import Quant

ExistsFn = Callable[[str, str | None], bool]

_ORG: Final = "mlx-community"
_INFIXES: Final = ("", "-MLX", "-mlx")


def quant_suffixes(quant: Quant) -> tuple[str, ...]:
    match quant:
        case Quant.FOUR_BIT:
            return ("4bit", "4-bit")
        case Quant.EIGHT_BIT:
            return ("8bit", "8-bit")
        case Quant.BF16:
            return ("bf16", "bfloat16")
        case _ as unreachable:
            assert_never(unreachable)


def model_stem(name: str) -> str:
    stem = name
    lowered = stem.lower()
    tags = ("4bit", "4-bit", "8bit", "8-bit", "bf16", "bfloat16")
    for tag in tags:
        suffix = f"-{tag}"
        if lowered.endswith(suffix):
            stem = stem[: -len(suffix)]
            lowered = stem.lower()
            break
    if lowered.endswith("-mlx"):
        stem = stem[:-4]
    return stem


def name_matches_quant(name: str, quant: Quant) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(f"-{tag}") for tag in quant_suffixes(quant))


def candidate_repo_ids(repo_id: str, quant: Quant) -> tuple[str, ...]:
    base = repo_id.rsplit("/", 1)[-1]
    stem = model_stem(base)
    found: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        if item in seen:
            return
        seen.add(item)
        found.append(item)

    if name_matches_quant(base, quant):
        add(repo_id)
    for tag in quant_suffixes(quant):
        for infix in _INFIXES:
            add(f"{_ORG}/{stem}{infix}-{tag}")
    return tuple(found)


def find_ready_mlx(
    repo_id: str,
    quant: Quant,
    token: str | None,
    exists: ExistsFn | None = None,
) -> str | None:
    check = exists if exists is not None else _exists
    for candidate in candidate_repo_ids(repo_id, quant):
        if check(candidate, token):
            return candidate
    return None


def _exists(repo_id: str, token: str | None) -> bool:
    try:
        return hub_repo_available(repo_id, token)
    except ModelNotFoundError:
        return False
