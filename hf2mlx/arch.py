from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_K_AND_V: Final = 2
_FP16_BYTES: Final = 2
_HEURISTIC_DIV: Final = 65536


@dataclass(frozen=True, slots=True)
class ModelArch:
    n_layers: int
    n_kv_heads: int
    head_dim: int


def kv_bytes_for(arch: ModelArch, ctx: int) -> int:
    per_layer = arch.n_kv_heads * arch.head_dim * ctx * _FP16_BYTES
    return _K_AND_V * arch.n_layers * per_layer


def heuristic_kv_bytes(param_count: int, ctx: int) -> int:
    return param_count * ctx // _HEURISTIC_DIV


def kv_cache_bytes(arch: ModelArch | None, param_count: int, ctx: int) -> int:
    if arch is None:
        return heuristic_kv_bytes(param_count, ctx)
    return kv_bytes_for(arch, ctx)


def arch_from_config_file(path: Path) -> ModelArch | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    layers = _positive_int(loaded.get("num_hidden_layers"))
    if layers is None:
        layers = _positive_int(loaded.get("n_layer"))
    heads = _positive_int(loaded.get("num_attention_heads"))
    if heads is None:
        heads = _positive_int(loaded.get("n_head"))
    hidden = _positive_int(loaded.get("hidden_size"))
    if hidden is None:
        hidden = _positive_int(loaded.get("n_embd"))
    kv_heads = _positive_int(loaded.get("num_key_value_heads"))
    if kv_heads is None:
        kv_heads = heads
    head_dim = _positive_int(loaded.get("head_dim"))
    if head_dim is None and hidden is not None and heads is not None:
        head_dim = hidden // heads
    if layers is None or kv_heads is None or head_dim is None or head_dim <= 0:
        return None
    return ModelArch(n_layers=layers, n_kv_heads=kv_heads, head_dim=head_dim)


def _positive_int(value: float | str | bool | None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value
