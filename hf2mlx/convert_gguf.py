from __future__ import annotations

from pathlib import Path

from hf2mlx.errors import GgufNotImplementedError
from hf2mlx.utils import Quant


def convert_to_gguf(_source: Path, _out: Path, _quant: Quant) -> None:
    raise GgufNotImplementedError
