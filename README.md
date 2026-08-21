<div align="center">

# HF2MLX

[![CI](https://img.shields.io/github/actions/workflow/status/jeonjw85/hf2mlx/ci.yml?branch=main)](https://github.com/jeonjw85/hf2mlx/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/release/jeonjw85/hf2mlx)](https://github.com/jeonjw85/hf2mlx/releases)

[한국어](README.ko.md)

</div>

Convert Hugging Face models to **MLX** for Apple Silicon.

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct --format mlx --quant 4bit
```

Downloads if needed, converts, then prints the output path, size, and a rough RAM estimate. After that the model is local - no cloud.

## Install

Apple Silicon, Python 3.10+. Intel Macs are not a target.

```bash
git clone https://github.com/jeonjw85/hf2mlx.git
cd hf2mlx
uv pip install .
```

```bash
hf2mlx --help
```

## Quick start

```bash
hf2mlx Qwen/Qwen2.5-3B-Instruct --quant 4bit
```

Already downloaded:

```bash
hf2mlx ./models/my-model --quant 8bit
```

Size / RAM only, no convert:

```bash
hf2mlx google/gemma-2-9b-it --estimate
```

Custom output dir:

```bash
hf2mlx Qwen/Qwen2.5-3B-Instruct --out ./converted
```

Then run it:

```bash
mlx_lm.generate --model ./converted/Qwen2.5-3B-Instruct-mlx-4bit --prompt "Hello"
```

## Options

| Option | Description | Default |
|---|---|---|
| `--format` | `mlx` or `gguf` | `mlx` |
| `--quant` | `4bit`, `8bit`, `bf16` | `4bit` |
| `--out` | output directory | `./converted/<name>-mlx-<quant>` |
| `--estimate` | print size/memory only | off |
| `--hf-token` | Hugging Face token | `HF_TOKEN` |
| `--force` | overwrite existing output | off |

GGUF is not implemented. Use `--format mlx`.

## Mac memory

Weights and KV cache share unified memory. Longer context costs extra RAM. Numbers below are rough.

| Machine | Comfortable | Tight | Skip |
|---|---|---|---|
| 16 GB | 3B 4-bit | 7B 4-bit | 7B 8-bit / 13B+ |
| 24 GB | 7B 4-bit or 8-bit | 14B 4-bit | 32B+ unquantized |
| 32 GB+ | 14B 4-bit, 7B bf16 | 32B 4-bit | 70B unless 4-bit |

Default is 4-bit because that is what actually fits on ≤24 GB machines.

## Gated models

Llama, Gemma, and friends need a token with access on the model card.

```bash
export HF_TOKEN=hf_...
hf2mlx meta-llama/Llama-3.2-3B-Instruct --quant 4bit
```

Token: https://huggingface.co/settings/tokens

`Error: model is gated. Set HF_TOKEN and retry.` means the token is missing, or it does not have access.

## Develop

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run basedpyright
```

## License

MIT. See [LICENSE](LICENSE).
