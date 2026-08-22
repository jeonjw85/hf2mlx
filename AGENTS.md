# AGENTS.md

## Project

**HF2MLX** - CLI that converts Hugging Face models into Mac-friendly local formats.

Primary target:

- **MLX** (4-bit / 8-bit / bf16)

Secondary:

- **GGUF** via llama.cpp `convert_hf_to_gguf.py` and `llama-quantize`

Goal: one command to get a usable local LLM on Apple Silicon.

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct --format mlx --quant 4bit
```

Typical flow:

1. Resolve a Hugging Face model ID or local path
2. Download if needed
3. Convert to the requested format/quantization
4. Print output path, size, and rough memory estimate
5. Fail clearly when memory or disk is insufficient

For MLX, check `mlx-community/<name>-<quant>` first and download that instead of converting. `--fit` picks quant from this machine's RAM. `--rebuild` forces a local convert.

## Core principles

- Mac-first: Apple Silicon + unified memory
- Simple CLI: one command for the common case
- Safe defaults: 4-bit MLX on ≤24GB machines
- Clear errors: no silent failures, no vague stack traces when avoidable
- No cloud dependency at runtime after the model is downloaded

## CLI

```bash
hf2mlx <model> [options]
```

Examples:

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct --format mlx --quant 4bit
hf2mlx ./models/my-model --format mlx --quant 8bit
hf2mlx google/gemma-2-9b-it --estimate
hf2mlx Qwen/Qwen2.5-3B-Instruct --out ./converted
hf2mlx Qwen/Qwen2.5-7B-Instruct --fit
hf2mlx Qwen/Qwen2.5-7B-Instruct --estimate --ctx 8192
```

| Option | Description | Default |
|---|---|---|
| `--format` | `mlx` or `gguf` | `mlx` |
| `--quant` | `4bit`, `8bit`, `bf16` | `4bit` |
| `--fit` | pick the largest quant that fits this Mac | `false` |
| `--ctx` | context length for RAM estimates | `4096` |
| `--out` | output directory | `./converted/<model-name>` |
| `--estimate` | estimate size/memory only; do not convert | `false` |
| `--rebuild` | convert from source even if a ready MLX repo exists | `false` |
| `--hf-token` | Hugging Face token | env `HF_TOKEN` |
| `--force` | overwrite existing output | `false` |

Keep this interface stable. Do not rename flags or the command unless the user asks.

## Stack

- Python 3.10+
- `uv` for packaging and running
- `mlx-lm` for MLX conversion/loading
- `huggingface_hub` for downloads
- `typer` for CLI
- `rich` for terminal output
- `pytest` for tests

## Layout

```text
hf2mlx/
  __init__.py
  cli.py              # CLI entrypoint
  job.py              # convert / reuse orchestration
  convert_mlx.py      # MLX conversion
  convert_gguf.py     # GGUF path (secondary)
  estimate.py         # size / RAM estimates
  arch.py             # config.json → KV shape
  fit.py              # --fit quant picker
  ready.py            # mlx-community reuse
  hf_utils.py         # download / auth / path resolve
  utils.py            # shared helpers
pyproject.toml
README.md
AGENTS.md
```

Display name: **HF2MLX**. Package and CLI: `hf2mlx`. Console script: `hf2mlx`.

## Conversion

1. Resolve input: Hugging Face repo id, or local model directory
2. Check disk space and warn if low
3. Reuse a matching mlx-community repo when `--format mlx` and not `--rebuild`
4. Download the model if remote and a convert is needed
5. Convert with the selected quant
6. Write artifacts to the output dir
7. Print output path, final size, and RAM estimate at `--ctx`

## Estimates

Rough guidance, not exact science.

MLX:

- 4-bit ≈ 0.55-0.7 bytes/param
- 8-bit ≈ 1.0-1.2 bytes/param
- bf16 ≈ 2.0 bytes/param

Always mention the `--ctx` used. KV cache is computed from `config.json` when available, otherwise a conservative heuristic.

## UX

Good output:

```text
Model: Qwen/Qwen2.5-7B-Instruct
Format: MLX
Quant: 4bit
Output: ./converted/Qwen2.5-7B-Instruct-mlx-4bit
Ready: mlx-community/Qwen2.5-7B-Instruct-4bit
Size: 4.1 GB
Est. inference memory: ~6-8 GB @ 4096 ctx
Done.
```

Handle these cleanly:

- invalid model id
- gated model without token
- insufficient disk space
- unsupported architecture
- output dir already exists (unless `--force`)

Prefer:

```text
Error: model is gated. Set HF_TOKEN and retry.
```

over long unrelated tracebacks.

## Non-goals

Do not turn this into:

- an LM Studio replacement
- a training / fine-tuning framework
- a multi-backend inference server

Stay on conversion + estimation + simple packaging.

## Tests

Minimum:

- CLI help works
- `--estimate` works without converting
- converting a tiny model works end-to-end
- invalid quant/format fails with a clear message
- existing output without `--force` fails safely

If full model conversion tests are too heavy, use a tiny fixture or mock the conversion layer.

## README

Keep it short and practical:

- What it does
- Install
- Quick start
- Options
- Mac memory guidance (16GB / 24GB)
- HF token setup for gated models

Update README when CLI options change.

## Agent working rules

When modifying this project:

- Keep the CLI interface stable unless necessary
- Smallest change that solves the request
- Do not add unrelated features
- Prefer the MLX path first; GGUF is secondary unless explicitly requested
- Optimize for Apple Silicon usability
- Update README when CLI options change
- Do not leave TODOs, placeholders, or incomplete code unless asked
- Do not remove existing functionality unless asked

### Comments

Almost never. No narrative comments, no restating the code, no section banners.

Allowed only when the code cannot express the reason (a non-obvious invariant, a workaround for an upstream bug). Keep those to one short line.

### Commits

Conventional commits:

```text
feat: add mlx 4bit conversion
fix: handle gated hf models
docs: add 24gb memory guidance
refactor: split estimate helpers
```

Never add AI as author, co-author, reviewer, or trailer. Forbidden:

- `Co-Authored-By: ...`
- `Generated-By: ...`
- `Signed-off-by:` with an AI name
- Claude, Anthropic, OpenAI, Copilot, Cursor, Codex, Grok, or any other AI identity in commit metadata or message body

Commit as the local git user only. Do not invent or inject extra authors.

Do not commit unless the user explicitly asks.

## Success

A user on a Mac can run:

```bash
uv pip install .
hf2mlx Qwen/Qwen2.5-3B-Instruct --format mlx --quant 4bit
```

and get a usable local model directory with clear next-step guidance.
