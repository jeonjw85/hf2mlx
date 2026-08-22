<div align="center">

# HF2MLX

Hugging Face → MLX on Apple Silicon

[![CI](https://img.shields.io/github/actions/workflow/status/jeonjw85/hf2mlx/ci.yml?branch=main)](https://github.com/jeonjw85/hf2mlx/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/release/jeonjw85/hf2mlx)](https://github.com/jeonjw85/hf2mlx/releases)

<a href="README.md"><strong>English</strong></a>
&nbsp;·&nbsp;
<a href="README.ko.md">한국어</a>

</div>

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct
```

If `mlx-community` already has that quant, hf2mlx downloads it. Otherwise it converts locally.  
When it finishes: path, size, and a RAM estimate at the context you asked for.

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct --fit
hf2mlx Qwen/Qwen2.5-7B-Instruct --estimate --ctx 8192
hf2mlx ./models/my-model --quant 8bit
```

> `--fit` reads this Mac's RAM and picks `bf16` / `8bit` / `4bit` ("will it load", not "will it be comfortable")
> Leaves ~7 GB for the OS. If nothing fits, it exits before downloading.

Then:

```bash
mlx_lm.generate --model ./converted/Qwen2.5-7B-Instruct-mlx-4bit --prompt "Hello"
```

## Install

Apple Silicon, Python 3.10+

```bash
git clone https://github.com/jeonjw85/hf2mlx.git
cd hf2mlx
uv pip install .
```

## Options

| Option       | Default                          |                                        |
| ------------ | -------------------------------- | -------------------------------------- |
| `--format`   | `mlx`                            | `mlx` or `gguf`                        |
| `--quant`    | `4bit`                           | `4bit`, `8bit`, `bf16`                 |
| `--fit`      | off                              | ignore `--quant`, pick what fits       |
| `--ctx`      | `4096`                           | context length used for RAM estimates  |
| `--out`      | `./converted/<name>-mlx-<quant>` | output directory                       |
| `--estimate` | off                              | print size/RAM only                    |
| `--rebuild`  | off                              | convert from source even if a ready MLX repo exists |
| `--hf-token` | `HF_TOKEN`                       | Hugging Face token                     |
| `--force`    | off                              | overwrite existing output              |

Ready lookup only runs for `--format mlx`. GGUF always converts.

## GGUF

```bash
uv pip install '.[gguf]'
brew install llama.cpp
hf2mlx Qwen/Qwen2.5-3B-Instruct --format gguf --quant 4bit
```

| `--quant` | GGUF type                 |
| --------- | ------------------------- |
| `4bit`    | Q4_K_M (`llama-quantize`) |
| `8bit`    | Q8_0                      |
| `bf16`    | BF16                      |

4-bit needs `llama-quantize` on `PATH`.  
The first GGUF convert may clone llama.cpp's Python converter into `~/.cache/hf2mlx`. Point `LLAMA_CPP_DIR` at an existing checkout to skip that.

## Memory

Weights and KV cache share unified memory.  
Trust `--estimate --ctx N`, not this table.

| Machine | Usually fine        | Tight     | Skip             |
| ------- | ------------------- | --------- | ---------------- |
| 16 GB   | 3B 4-bit            | 7B 4-bit  | 7B 8-bit / 13B+  |
| 24 GB   | 7B 4-bit or 8-bit   | 14B 4-bit | 32B+ unquantized |
| 32 GB+  | 14B 4-bit, 7B bf16  | 32B 4-bit | 70B unless 4-bit |

> Default is 4-bit because that is what actually fits on ≤24 GB machines.

## Gated models

Llama, Gemma, and similar need a token with access on the model card.

```bash
export HF_TOKEN=hf_...
hf2mlx meta-llama/Llama-3.2-3B-Instruct --quant 4bit
```

[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

`Error: model is gated. Set HF_TOKEN and retry.` means the token is missing or does not have access.

If `mlx-community` already has the quant you asked for, hf2mlx uses that and does not need the original gate.

## Develop

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run basedpyright
```

## License

[MIT](LICENSE)
