from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from hf2mlx.errors import InsufficientDiskError, OutputExistsError


class OutputFormat(str, Enum):
    MLX = "mlx"
    GGUF = "gguf"


class Quant(str, Enum):
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"
    BF16 = "bf16"


_BYTES_PER_GB: Final = 1_000_000_000
_BYTES_PER_MB: Final = 1_000_000
_LOW_DISK_FACTOR: Final = 2


@dataclass(frozen=True, slots=True)
class DiskCheck:
    available_bytes: int
    is_low: bool


def model_basename(model: str) -> str:
    return Path(model.rstrip("/")).name


def default_out_dir(model: str, fmt: OutputFormat, quant: Quant) -> Path:
    return Path("converted") / f"{model_basename(model)}-{fmt.value}-{quant.value}"


def format_bytes(n: int) -> str:
    if n >= _BYTES_PER_GB:
        return f"{n / _BYTES_PER_GB:.1f} GB"
    if n >= _BYTES_PER_MB:
        return f"{n / _BYTES_PER_MB:.1f} MB"
    return f"{n} B"


def format_gb_range(low: int, high: int) -> str:
    lo_bytes = max(low, 0)
    hi_bytes = max(high, lo_bytes)
    if hi_bytes < _BYTES_PER_GB:
        lo_mb = lo_bytes / _BYTES_PER_MB
        hi_mb = hi_bytes / _BYTES_PER_MB
        return f"~{lo_mb:.0f}–{hi_mb:.0f} MB (depends on context length)"
    lo_gb = lo_bytes / _BYTES_PER_GB
    hi_gb = hi_bytes / _BYTES_PER_GB
    return f"~{lo_gb:.0f}–{hi_gb:.0f} GB (depends on context length)"


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            if file_path.is_file() and not file_path.is_symlink():
                total += file_path.stat().st_size
    return total


def existing_parent(path: Path) -> Path:
    resolved = path.expanduser()
    try:
        resolved = resolved.resolve()
    except OSError:
        resolved = path.expanduser()
    for candidate in [resolved, *resolved.parents]:
        if candidate.exists():
            return candidate
    return Path("/")


def prepare_out_dir(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if not force:
        raise OutputExistsError(path=path)
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def check_disk(path: Path, needed_bytes: int) -> DiskCheck:
    available = shutil.disk_usage(existing_parent(path)).free
    if needed_bytes > 0 and available < needed_bytes:
        raise InsufficientDiskError(
            needed_bytes=needed_bytes,
            available_bytes=available,
            needed_label=format_bytes(needed_bytes),
            available_label=format_bytes(available),
        )
    is_low = needed_bytes > 0 and available < needed_bytes * _LOW_DISK_FACTOR
    return DiskCheck(available_bytes=available, is_low=is_low)


def total_ram_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return int(pages * page_size)
