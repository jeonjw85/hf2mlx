from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from typing_extensions import assert_never

from hf2mlx.errors import ConversionFailedError, GgufToolsMissingError
from hf2mlx.utils import Quant

_LLAMA_CPP_REPO: Final = "https://github.com/ggml-org/llama.cpp.git"
_CONVERT_SCRIPT: Final = "convert_hf_to_gguf.py"


@dataclass(frozen=True, slots=True)
class GgufQuantPlan:
    outtype: str
    quantize_type: str | None


def quant_to_gguf_plan(quant: Quant) -> GgufQuantPlan:
    match quant:
        case Quant.FOUR_BIT:
            return GgufQuantPlan(outtype="bf16", quantize_type="Q4_K_M")
        case Quant.EIGHT_BIT:
            return GgufQuantPlan(outtype="q8_0", quantize_type=None)
        case Quant.BF16:
            return GgufQuantPlan(outtype="bf16", quantize_type=None)
        case _ as unreachable:
            assert_never(unreachable)


def convert_to_gguf(hf_path: Path, out: Path, quant: Quant) -> None:
    plan = quant_to_gguf_plan(quant)
    converter = _ensure_converter()
    out.mkdir(parents=True, exist_ok=True)
    raw_gguf = out / f"model-{plan.outtype}.gguf"
    final_gguf = out / "model.gguf"
    _run_hf_convert(converter, hf_path, raw_gguf, plan.outtype)
    if plan.quantize_type is None:
        raw_gguf.replace(final_gguf)
        return
    quantize = _resolve_quantize()
    _run(
        [str(quantize), str(raw_gguf), str(final_gguf), plan.quantize_type],
    )
    raw_gguf.unlink(missing_ok=True)


def cache_dir() -> Path:
    override = os.environ.get("HF2MLX_CACHE")
    if override is not None and override != "":
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg is not None and xdg != "":
        return Path(xdg) / "hf2mlx"
    return Path.home() / ".cache" / "hf2mlx"


def _ensure_converter() -> Path:
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir is not None and env_dir != "":
        script = Path(env_dir) / _CONVERT_SCRIPT
        if script.is_file():
            return script.resolve()
    cached_root = cache_dir() / "llama.cpp"
    script = cached_root / _CONVERT_SCRIPT
    if script.is_file():
        return script.resolve()
    _clone_converter(cached_root)
    if not script.is_file():
        raise GgufToolsMissingError(
            reason=(
                "convert_hf_to_gguf.py missing. "
                "Set LLAMA_CPP_DIR to a llama.cpp checkout."
            )
        )
    return script.resolve()


def _clone_converter(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            _LLAMA_CPP_REPO,
            str(dest),
        ]
    )
    _run_git(
        ["git", "sparse-checkout", "set", _CONVERT_SCRIPT, "conversion"],
        cwd=dest,
    )


def _resolve_quantize() -> Path:
    found = shutil.which("llama-quantize")
    if found is None:
        raise GgufToolsMissingError(
            reason="llama-quantize not found. Install llama.cpp: brew install llama.cpp"
        )
    return Path(found)


def _run_hf_convert(
    converter: Path,
    hf_path: Path,
    outfile: Path,
    outtype: str,
) -> None:
    _run(
        [
            sys.executable,
            converter.name,
            str(hf_path.resolve()),
            "--outfile",
            str(outfile.resolve()),
            "--outtype",
            outtype,
        ],
        cwd=converter.parent,
    )
    if not outfile.is_file():
        raise ConversionFailedError(reason=f"converter did not write {outfile}")


def _run(argv: list[str], cwd: Path | None = None) -> None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ConversionFailedError(reason=str(exc)) from exc
    if completed.returncode == 0:
        return
    detail = completed.stderr.strip() or completed.stdout.strip()
    if "No module named 'torch'" in detail:
        raise GgufToolsMissingError(
            reason="torch is required for GGUF. Run: uv pip install 'hf2mlx[gguf]'"
        )
    if detail == "":
        detail = f"exit {completed.returncode}"
    raise ConversionFailedError(reason=detail[-2000:])


def _run_git(argv: list[str], cwd: Path | None = None) -> None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise GgufToolsMissingError(
            reason=f"git is required to fetch the GGUF converter. {exc}"
        ) from exc
    if completed.returncode == 0:
        return
    detail = completed.stderr.strip() or completed.stdout.strip()
    if detail == "":
        detail = "git command failed"
    raise GgufToolsMissingError(reason=detail[-2000:])
