from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from hf2mlx.errors import Hf2mlxError, UnexpectedError
from hf2mlx.estimate import DEFAULT_CTX
from hf2mlx.job import ConvertJob, execute
from hf2mlx.utils import OutputFormat, Quant

console = Console()
app = typer.Typer(
    name="hf2mlx",
    help="Convert Hugging Face models to MLX or GGUF.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
)


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
    fit: Annotated[
        bool,
        typer.Option("--fit", help="Pick the largest quant that fits this Mac."),
    ] = False,
    ctx: Annotated[
        int,
        typer.Option("--ctx", min=1, help="Context length for RAM estimates."),
    ] = DEFAULT_CTX,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help="Convert from source even if a ready MLX repo exists.",
        ),
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
        fit=fit,
        ctx=ctx,
        rebuild=rebuild,
    )
    try:
        execute(job)
    except Hf2mlxError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(code=1) from None
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]{UnexpectedError(reason=str(exc))}[/bold red]")
        raise typer.Exit(code=1) from None
