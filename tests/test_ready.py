from __future__ import annotations

from hf2mlx.ready import candidate_repo_ids, find_ready_mlx
from hf2mlx.utils import Quant


def test_candidates_prefer_mlx_community_quant_suffix() -> None:
    ids = candidate_repo_ids("Qwen/Qwen2.5-7B-Instruct", Quant.FOUR_BIT)
    assert ids[0] == "mlx-community/Qwen2.5-7B-Instruct-4bit"
    assert "mlx-community/Qwen2.5-7B-Instruct-MLX-4bit" in ids


def test_candidates_for_8bit_and_bf16() -> None:
    eight = candidate_repo_ids("meta-llama/Llama-3.2-3B-Instruct", Quant.EIGHT_BIT)
    bf16 = candidate_repo_ids("google/gemma-2-9b-it", Quant.BF16)
    assert eight[0] == "mlx-community/Llama-3.2-3B-Instruct-8bit"
    assert bf16[0] == "mlx-community/gemma-2-9b-it-bf16"


def test_already_quantized_source_is_first_candidate() -> None:
    ids = candidate_repo_ids(
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        Quant.FOUR_BIT,
    )
    assert ids[0] == "mlx-community/Qwen2.5-7B-Instruct-4bit"


def test_find_ready_returns_first_live_candidate() -> None:
    def exists(repo_id: str, token: str | None) -> bool:
        return repo_id == "mlx-community/Qwen2.5-7B-Instruct-MLX-4bit"

    found = find_ready_mlx(
        "Qwen/Qwen2.5-7B-Instruct",
        Quant.FOUR_BIT,
        None,
        exists=exists,
    )
    assert found == "mlx-community/Qwen2.5-7B-Instruct-MLX-4bit"


def test_find_ready_returns_none_when_nothing_exists() -> None:
    found = find_ready_mlx(
        "org/unknown-model",
        Quant.FOUR_BIT,
        None,
        exists=lambda repo_id, token: False,
    )
    assert found is None
