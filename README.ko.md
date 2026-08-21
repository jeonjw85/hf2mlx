<div align="center">

# HF2MLX

[![CI](https://img.shields.io/github/actions/workflow/status/jeonjw85/hf2mlx/ci.yml?branch=main)](https://github.com/jeonjw85/hf2mlx/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/release/jeonjw85/hf2mlx)](https://github.com/jeonjw85/hf2mlx/releases)

[English](README.md)

</div>

Hugging Face 모델을 맥에서 돌릴 수 있게 **MLX**로 바꿔주는 CLI예요. Apple Silicon용입니다.

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct --format mlx --quant 4bit
```

모델 받고, 변환하고, 용량이랑 메모리 감 정도를 찍어줍니다. 한 번 받아두면 그다음엔 로컬만 씁니다.

## 설치

Apple Silicon, Python 3.10+ 필요합니다. 인텔 맥은 대상이 아니에요.

```bash
git clone https://github.com/jeonjw85/hf2mlx.git
cd hf2mlx
uv pip install .
```

```bash
hf2mlx --help
```

## 바로 쓰기

```bash
hf2mlx Qwen/Qwen2.5-3B-Instruct --quant 4bit
```

이미 받아둔 폴더면 경로만 넘기면 됩니다.

```bash
hf2mlx ./models/my-model --quant 8bit
```

변환 없이 용량만 보고 싶을 때:

```bash
hf2mlx google/gemma-2-9b-it --estimate
```

출력 위치 지정:

```bash
hf2mlx Qwen/Qwen2.5-3B-Instruct --out ./converted
```

변환이 끝나면 이렇게 돌려보면 됩니다.

```bash
mlx_lm.generate --model ./converted/Qwen2.5-3B-Instruct-mlx-4bit --prompt "안녕"
```

## 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--format` | `mlx` 또는 `gguf` | `mlx` |
| `--quant` | `4bit`, `8bit`, `bf16` | `4bit` |
| `--out` | 출력 폴더 | `./converted/<이름>-mlx-<quant>` |
| `--estimate` | 용량/메모리만 계산하고 변환은 안 함 | 꺼짐 |
| `--hf-token` | Hugging Face 토큰 | `HF_TOKEN` |
| `--force` | 이미 있는 출력 폴더를 덮어씀 | 꺼짐 |

GGUF는 아직 없습니다. `--format mlx`로 쓰세요.

## 맥 메모리

통합 메모리에 가중치랑 KV 캐시가 같이 올라갑니다. 컨텍스트가 길어지면 그만큼 더 먹어요. 아래 숫자는 감입니다. 과학 아닙니다.

| 머신 | 여유 있음 | 빠듯함 | 비추 |
|---|---|---|---|
| 16 GB | 3B 4-bit | 7B 4-bit | 7B 8-bit / 13B+ |
| 24 GB | 7B 4-bit 또는 8-bit | 14B 4-bit | 양자화 안 한 32B+ |
| 32 GB+ | 14B 4-bit, 7B bf16 | 32B 4-bit | 70B는 4-bit여도 부담 |

기본값이 4-bit인 이유는 단순합니다. 24GB 이하에서 실제로 들어가는 쪽이 그거라서요.

## 게이트된 모델

Llama, Gemma 같은 건 모델 카드에서 권한 받은 토큰이 필요합니다.

```bash
export HF_TOKEN=hf_...
hf2mlx meta-llama/Llama-3.2-3B-Instruct --quant 4bit
```

토큰은 여기서 만듭니다: https://huggingface.co/settings/tokens

`Error: model is gated. Set HF_TOKEN and retry.` 가 뜨면 토큰이 없거나, 있어도 그 모델 권한이 없는 겁니다.

## 개발

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run basedpyright
```

## 라이선스

MIT입니다. [LICENSE](LICENSE) 참고.
