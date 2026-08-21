<div align="center">

# HF2MLX

[![CI](https://img.shields.io/github/actions/workflow/status/jeonjw85/hf2mlx/ci.yml?branch=main)](https://github.com/jeonjw85/hf2mlx/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/release/jeonjw85/hf2mlx)](https://github.com/jeonjw85/hf2mlx/releases)

[English](README.md)

</div>

Hugging Face 모델을 Apple Silicon용 **MLX**로 변환하는 CLI입니다.

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct --format mlx --quant 4bit
```

모델을 내려받고 변환한 뒤, 출력 경로와 용량, 대략적인 메모리 사용량을 출력합니다. 변환이 끝나면 로컬에서만 사용합니다.

## 설치

Apple Silicon, Python 3.10+가 필요합니다. Intel Mac은 지원하지 않습니다.

```bash
git clone https://github.com/jeonjw85/hf2mlx.git
cd hf2mlx
uv pip install .
```

```bash
hf2mlx --help
```

## 빠른 시작

```bash
hf2mlx Qwen/Qwen2.5-3B-Instruct --quant 4bit
```

이미 받아 둔 로컬 폴더:

```bash
hf2mlx ./models/my-model --quant 8bit
```

변환 없이 용량과 메모리만 확인할 때:

```bash
hf2mlx google/gemma-2-9b-it --estimate
```

출력 디렉터리 지정:

```bash
hf2mlx Qwen/Qwen2.5-3B-Instruct --out ./converted
```

변환 후 실행 예시:

```bash
mlx_lm.generate --model ./converted/Qwen2.5-3B-Instruct-mlx-4bit --prompt "안녕"
```

## 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--format` | `mlx` 또는 `gguf` | `mlx` |
| `--quant` | `4bit`, `8bit`, `bf16` | `4bit` |
| `--out` | 출력 폴더 | `./converted/<이름>-mlx-<quant>` |
| `--estimate` | 용량/메모리만 계산하고 변환하지 않음 | 꺼짐 |
| `--hf-token` | Hugging Face 토큰 | `HF_TOKEN` |
| `--force` | 기존 출력 폴더를 덮어씀 | 꺼짐 |

GGUF 변환은 아직 지원하지 않습니다. `--format mlx`를 사용하세요.

## 맥 메모리

통합 메모리에는 가중치와 KV 캐시가 함께 올라갑니다. 컨텍스트가 길수록 추가 메모리가 필요합니다. 아래 수치는 대략적인 기준입니다.

| 머신 | 여유 있음 | 빠듯함 | 비권장 |
|---|---|---|---|
| 16 GB | 3B 4-bit | 7B 4-bit | 7B 8-bit / 13B+ |
| 24 GB | 7B 4-bit 또는 8-bit | 14B 4-bit | 양자화하지 않은 32B+ |
| 32 GB+ | 14B 4-bit, 7B bf16 | 32B 4-bit | 70B는 4-bit여도 부담 |

기본값은 4-bit입니다. 24GB 이하 환경에서 실제로 올리기 쉬운 설정이기 때문입니다.

## 게이트된 모델

Llama, Gemma 등 게이트된 모델은 모델 카드에서 접근 권한을 받은 토큰이 필요합니다.

```bash
export HF_TOKEN=hf_...
hf2mlx meta-llama/Llama-3.2-3B-Instruct --quant 4bit
```

토큰 발급: https://huggingface.co/settings/tokens

`Error: model is gated. Set HF_TOKEN and retry.`가 나오면 토큰이 없거나, 해당 모델 권한이 없는 경우입니다.

## 개발

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run basedpyright
```

## 라이선스

MIT. [LICENSE](LICENSE)를 참고하세요.
