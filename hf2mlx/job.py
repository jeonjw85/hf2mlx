from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from typing_extensions import assert_never

from hf2mlx.arch import ModelArch, arch_from_config_file
from hf2mlx.convert_gguf import convert_to_gguf
from hf2mlx.convert_mlx import convert_to_mlx
from hf2mlx.errors import (
    InsufficientMemoryError,
    MemoryUnknownError,
    ParamCountUnknownError,
)
from hf2mlx.estimate import SizeEstimate, estimate_for, source_weight_bytes
from hf2mlx.fit import fit_quant
from hf2mlx.hf_utils import (
    HubModel,
    LocalModel,
    ResolvedModel,
    download_config,
    download_model,
    download_to,
    looks_like_local_path,
    resolve_model,
)
from hf2mlx.ready import find_ready_mlx
from hf2mlx.utils import (
    OutputFormat,
    Quant,
    check_disk,
    default_out_dir,
    dir_size,
    format_bytes,
    format_gb_range,
    prepare_out_dir,
    total_ram_bytes,
)

console = Console()


@dataclass(frozen=True, slots=True)
class ConvertJob:
    model: str
    fmt: OutputFormat
    quant: Quant
    out: Path | None
    estimate_only: bool
    hf_token: str | None
    force: bool
    fit: bool
    ctx: int
    rebuild: bool


def execute(job: ConvertJob) -> None:
    ready = _probe_ready(job)
    target = ready if ready is not None else job.model
    resolved = resolve_model(target, job.hf_token)
    params = resolved.param_count
    if params is None:
        raise ParamCountUnknownError(model=job.model)
    arch = _arch_for(resolved, job.hf_token)
    quant = _choose_quant(job, params, arch)
    if ready is None:
        ready = _ready_repo(job, resolved, quant)
    est = estimate_for(params, quant, job.fmt, job.ctx, arch)
    out = job.out
    if out is None:
        out = default_out_dir(job.model, job.fmt, quant)
    _print_plan(job, quant, est, out, ready)
    if job.estimate_only:
        return
    if ready is not None:
        _reuse(job, ready, est, out)
        return
    _convert(job, resolved, est, out, quant)


def _choose_quant(job: ConvertJob, params: int, arch: ModelArch | None) -> Quant:
    if not job.fit:
        return job.quant
    ram = total_ram_bytes()
    if ram is None:
        raise MemoryUnknownError
    return fit_quant(params, job.fmt, ram, job.ctx, arch)


def _arch_for(resolved: ResolvedModel, token: str | None) -> ModelArch | None:
    match resolved:
        case LocalModel(path=path):
            return arch_from_config_file(path / "config.json")
        case HubModel(repo_id=repo_id):
            config = download_config(repo_id, token)
            if config is None:
                return None
            return arch_from_config_file(config)
        case _ as unreachable:
            assert_never(unreachable)


def _probe_ready(job: ConvertJob) -> str | None:
    if job.rebuild or job.fit or job.fmt is not OutputFormat.MLX:
        return None
    if looks_like_local_path(job.model):
        return None
    return find_ready_mlx(job.model.strip(), job.quant, job.hf_token)


def _ready_repo(job: ConvertJob, resolved: ResolvedModel, quant: Quant) -> str | None:
    if job.rebuild or job.fmt is not OutputFormat.MLX:
        return None
    match resolved:
        case LocalModel():
            return None
        case HubModel(repo_id=repo_id):
            return find_ready_mlx(repo_id, quant, job.hf_token)
        case _ as unreachable:
            assert_never(unreachable)


def _reuse(job: ConvertJob, repo_id: str, est: SizeEstimate, out: Path) -> None:
    prepare_out_dir(out, job.force)
    _check_resources(out, est.size_high_bytes, est)
    with console.status("Downloading ready MLX model..."):
        _ = download_to(repo_id, job.hf_token, out)
    _print_done(job.fmt, out, dir_size(out), est)


def _convert(
    job: ConvertJob,
    resolved: ResolvedModel,
    est: SizeEstimate,
    out: Path,
    quant: Quant,
) -> None:
    prepare_out_dir(out, job.force)
    source = _materialize_source(job, resolved, est, out)
    match job.fmt:
        case OutputFormat.MLX:
            with console.status("Converting to MLX..."):
                convert_to_mlx(source, out, quant)
        case OutputFormat.GGUF:
            with console.status("Converting to GGUF..."):
                convert_to_gguf(source, out, quant)
        case _ as unreachable:
            assert_never(unreachable)
    _print_done(job.fmt, out, dir_size(out), est)


def _materialize_source(
    job: ConvertJob,
    resolved: ResolvedModel,
    est: SizeEstimate,
    out: Path,
) -> Path:
    needed = est.size_high_bytes
    match resolved:
        case LocalModel(path=path):
            _check_resources(out, needed, est)
            return path
        case HubModel(repo_id=repo_id):
            needed += source_weight_bytes(est.param_count)
            _check_resources(out, needed, est)
            with console.status("Downloading model..."):
                return download_model(repo_id, job.hf_token)
        case _ as unreachable:
            assert_never(unreachable)


def _check_resources(out: Path, needed: int, est: SizeEstimate) -> None:
    disk = check_disk(out, needed)
    if disk.is_low:
        console.print("[yellow]Warning: disk space is low.[/yellow]")
    ram = total_ram_bytes()
    if ram is None:
        return
    if est.size_high_bytes > ram:
        raise InsufficientMemoryError(
            needed_label=format_bytes(est.size_high_bytes),
            available_label=format_bytes(ram),
        )


def _print_plan(
    job: ConvertJob,
    quant: Quant,
    est: SizeEstimate,
    out: Path,
    ready: str | None,
) -> None:
    console.print(f"Model: {job.model}")
    console.print(f"Format: {job.fmt.value.upper()}")
    console.print(f"Quant: {quant.value}")
    console.print(f"Output: {out}")
    if ready is not None:
        console.print(f"Ready: {ready}")
    if not job.estimate_only:
        return
    console.print(f"Size: {format_bytes(est.size_mid_bytes)}")
    memory = format_gb_range(est.inference_low_bytes, est.inference_high_bytes, est.ctx)
    console.print(f"Est. inference memory: {memory}")


def _print_done(
    fmt: OutputFormat,
    out: Path,
    actual_size: int,
    est: SizeEstimate,
) -> None:
    console.print(f"Size: {format_bytes(actual_size)}")
    memory = format_gb_range(est.inference_low_bytes, est.inference_high_bytes, est.ctx)
    console.print(f"Est. inference memory: {memory}")
    console.print("Done.")
    match fmt:
        case OutputFormat.MLX:
            console.print(f'Next: mlx_lm.generate --model {out} --prompt "Hello"')
        case OutputFormat.GGUF:
            gguf_path = out / "model.gguf"
            console.print(f'Next: llama-cli -m {gguf_path} -p "Hello"')
        case _ as unreachable:
            assert_never(unreachable)
