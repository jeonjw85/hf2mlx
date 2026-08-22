from __future__ import annotations

import json
from pathlib import Path

from hf2mlx.arch import ModelArch, arch_from_config_file, kv_bytes_for


def test_kv_bytes_for_is_k_and_v_fp16() -> None:
    arch = ModelArch(n_layers=32, n_kv_heads=8, head_dim=128)
    assert kv_bytes_for(arch, ctx=4096) == 2 * 32 * 8 * 128 * 4096 * 2


def test_arch_from_config_file_reads_llama_style(tmp_path: Path) -> None:
    config = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    arch = arch_from_config_file(path)
    assert arch == ModelArch(n_layers=28, n_kv_heads=4, head_dim=128)


def test_arch_from_config_file_uses_head_dim_when_present(tmp_path: Path) -> None:
    config = {
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "head_dim": 128,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    arch = arch_from_config_file(path)
    assert arch == ModelArch(n_layers=36, n_kv_heads=8, head_dim=128)


def test_arch_from_missing_config_returns_none(tmp_path: Path) -> None:
    assert arch_from_config_file(tmp_path / "config.json") is None
