<div align="center">

# HF2MLX

Hugging Face 모델을 Apple Silicon에서 쓰는 MLX로 변환하는 툴

[![CI](https://img.shields.io/github/actions/workflow/status/jeonjw85/hf2mlx/ci.yml?branch=main)](https://github.com/jeonjw85/hf2mlx/actions/workflows/ci.yml)
[![Version](https://img.shields.io/github/v/release/jeonjw85/hf2mlx)](https://github.com/jeonjw85/hf2mlx/releases)

<a href="README.md">English</a>
&nbsp;·&nbsp;
<a href="README.ko.md"><strong>한국어</strong></a>

</div>

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct
```

`mlx-community`에 같은 양자화본이 있으면 그걸 받고 없으면 로컬에서 변환합니다.  
끝나면 경로, 용량, 지정한 컨텍스트 기준 RAM 예상치를 출력합니다.

```bash
hf2mlx Qwen/Qwen2.5-7B-Instruct --fit
hf2mlx Qwen/Qwen2.5-7B-Instruct --estimate --ctx 8192
hf2mlx ./models/my-model --quant 8bit
```

> `--fit`은 이 맥의 RAM을 보고 `bf16` / `8bit` / `4bit` 중 들어가는 걸 고릅니다 ("편하냐"가 아니라 "올라가냐")
> OS용으로 약 7GB를 남겨 두고 아무것도 안 들어가면 받기 전에 끝냅니다

변환 후:

```bash
mlx_lm.generate --model ./converted/Qwen2.5-7B-Instruct-mlx-4bit --prompt "안녕"
```

## 설치

Apple Silicon, Python 3.10+

```bash
git clone https://github.com/jeonjw85/hf2mlx.git
cd hf2mlx
uv pip install .
```

## 옵션

| 옵션         | 기본값                           |                                        |
| ------------ | -------------------------------- | -------------------------------------- |
| `--format`   | `mlx`                            | `mlx` 또는 `gguf`                      |
| `--quant`    | `4bit`                           | `4bit`, `8bit`, `bf16`                 |
| `--fit`      | 꺼짐                             | `--quant` 무시, 이 머신에 맞는 값 선택 |
| `--ctx`      | `4096`                           | RAM 계산에 쓰는 컨텍스트 길이          |
| `--out`      | `./converted/<이름>-mlx-<quant>` | 출력 폴더                              |
| `--estimate` | 꺼짐                             | 용량/RAM만 출력                        |
| `--rebuild`  | 꺼짐                             | ready MLX가 있어도 소스에서 다시 변환  |
| `--hf-token` | `HF_TOKEN`                       | Hugging Face 토큰                      |
| `--force`    | 꺼짐                             | 있는 출력 폴더를 덮어씀                |

ready 조회는 `--format mlx`일 때만 / GGUF는 항상 변환

## GGUF

```bash
uv pip install '.[gguf]'
brew install llama.cpp
hf2mlx Qwen/Qwen2.5-3B-Instruct --format gguf --quant 4bit
```

| `--quant` | GGUF 형식                 |
| --------- | ------------------------- |
| `4bit`    | Q4_K_M (`llama-quantize`) |
| `8bit`    | Q8_0                      |
| `bf16`    | BF16                      |

4-bit는 `PATH`에 `llama-quantize`가 필요합니다.  
첫 GGUF 변환 때 llama.cpp 변환 스크립트를 `~/.cache/hf2mlx`에 받을 수 있고 이미 있는 체크아웃은 `LLAMA_CPP_DIR`로 지정하면 됩니다.

## 메모리

가중치와 KV 캐시가 통합 메모리를 같이 씁니다.  
표보다 `--estimate --ctx N`을 믿으세요.

| 머신   | 보통 괜찮음         | 빠듯함    | 비권장               |
| ------ | ------------------- | --------- | -------------------- |
| 16 GB  | 3B 4-bit            | 7B 4-bit  | 7B 8-bit / 13B+      |
| 24 GB  | 7B 4-bit 또는 8-bit | 14B 4-bit | 양자화하지 않은 32B+ |
| 32 GB+ | 14B 4-bit, 7B bf16  | 32B 4-bit | 70B는 4-bit여도 부담 |

> 기본값이 4-bit인 이유 : 24GB 이하에서 실제로 올라가는 설정이기 때문

## 게이트된 모델

Llama, Gemma 등은 모델 카드에서 권한 받은 토큰 필수

```bash
export HF_TOKEN=hf_...
hf2mlx meta-llama/Llama-3.2-3B-Instruct --quant 4bit
```

[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

`Error: model is gated. Set HF_TOKEN and retry.`는 토큰이 없거나 해당 모델 권한이 없는 경우 발생합니다.

요청한 양자화본이 `mlx-community`에 있으면 원본 게이트 없이 그걸 받습니다.

## 개발

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run basedpyright
```

## 라이선스

[MIT](LICENSE)
