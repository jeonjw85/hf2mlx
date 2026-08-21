from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from typing_extensions import assert_never

from hf2mlx.convert_mlx import convert_to_mlx
from hf2mlx.errors import (
    GgufNotImplementedError,
    Hf2mlxError,
    InsufficientMemoryError,
    ParamCountUnknownError,
    UnexpectedError,
)
from hf2mlx.estimate import SizeEstimate, estimate_mlx, source_weight_bytes
from hf2mlx.hf_utils import (
    HubModel,
    LocalModel,
    ResolvedModel,
    download_model,
    resolve_model,
)
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
app = typer.Typer(
    name="hf2mlx",
    help="Convert Hugging Face models to MLX for Apple Silicon.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)


@dataclass(frozen=True, slots=True)
class ConvertJob:
    model: str
    fmt: OutputFormat
    quant: Quant
    out: Path | None
    estimate_only: bool
    hf_token: str | None
    force: bool


@app.command(name="hf2mlx")
def main(
    model: Annotated[
        str,
        typer.Argument(help="Hugging Face model ID or local model directory."),
    ],
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format."),
    ] = OutputFormat.MLX,
    quant: Annotated[
        Quant,
        typer.Option("--quant", help="Quantization."),
    ] = Quant.FOUR_BIT,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output directory."),
    ] = None,
    estimate_only: Annotated[
        bool,
        typer.Option("--estimate", help="Estimate size and memory only."),
    ] = False,
    hf_token: Annotated[
        str | None,
        typer.Option("--hf-token", envvar="HF_TOKEN", help="Hugging Face token."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing output."),
    ] = False,
) -> None:
    job = ConvertJob(
        model=model,
        fmt=fmt,
        quant=quant,
        out=out,
        estimate_only=estimate_only,
        hf_token=hf_token,
        force=force,
    )
    try:
        execute(job)
    except Hf2mlxError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]{UnexpectedError(reason=str(exc))}[/bold red]")
        raise typer.Exit(code=1) from None


def execute(job: ConvertJob) -> None:
    match job.fmt:
        case OutputFormat.GGUF:
            raise GgufNotImplementedError
        case OutputFormat.MLX:
            pass
        case _ as unreachable:
            assert_never(unreachable)
    resolved = resolve_model(job.model, job.hf_token)
    params = resolved.param_count
    if params is None:
        raise ParamCountUnknownError(model=job.model)
    est = estimate_mlx(params, job.quant)
    out = job.out
    if out is None:
        out = default_out_dir(job.model, job.fmt, job.quant)
    _print_plan(job, est, out)
    if job.estimate_only:
        return
    _convert_mlx(job, resolved, est, out)


def _convert_mlx(
    job: ConvertJob,
    resolved: ResolvedModel,
    est: SizeEstimate,
    out: Path,
) -> None:
    prepare_out_dir(out, job.force)
    source = _materialize_source(job, resolved, est, out)
    with console.status("Converting to MLX..."):
        convert_to_mlx(source, out, job.quant)
    _print_done(out, dir_size(out), est)


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


def _print_plan(job: ConvertJob, est: SizeEstimate, out: Path) -> None:
    console.print(f"Model: {job.model}")
    console.print(f"Format: {job.fmt.value.upper()}")
    console.print(f"Quant: {job.quant.value}")
    console.print(f"Output: {out}")
    if not job.estimate_only:
        return
    console.print(f"Size: {format_bytes(est.size_mid_bytes)}")
    memory = format_gb_range(est.inference_low_bytes, est.inference_high_bytes)
    console.print(f"Est. inference memory: {memory}")


def _print_done(out: Path, actual_size: int, est: SizeEstimate) -> None:
    console.print(f"Size: {format_bytes(actual_size)}")
    memory = format_gb_range(est.inference_low_bytes, est.inference_high_bytes)
    console.print(f"Est. inference memory: {memory}")
    console.print("Done.")
    console.print(f'Next: mlx_lm.generate --model {out} --prompt "Hello"')
