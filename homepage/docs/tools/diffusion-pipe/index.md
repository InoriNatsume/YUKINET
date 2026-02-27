# diffusion-pipe

<span class="version-badge">{{ ver.diffusion_pipe }}</span>

## 개요

[diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)은 DeepSpeed 기반의 고효율 diffusion 모델 훈련 도구입니다. Pipeline parallelism을 통해 단일 GPU에서도 대형 모델의 Full Fine-tune이 가능합니다.

## 지원 모델

| 모델 | LoRA | Full FT | Pipeline Parallel |
|---|---|---|---|
| Flux | ✅ | ✅ | ✅ |
| SD3 | ✅ | ✅ | ✅ |
| HunyuanVideo | ✅ | ✅ | ✅ |
| Wan | ✅ | ✅ | ✅ |

## kohya sd-scripts와의 차이

| 기능 | sd-scripts ({{ ver.sdscripts }}) | diffusion-pipe ({{ ver.diffusion_pipe }}) |
|---|---|---|
| 병렬 전략 | Block swapping | Pipeline parallelism (DeepSpeed) |
| 옵티마이저 | 다양 (bitsandbytes 등) | pytorch-optimizer (Prodigy 등) |
| 설정 방식 | CLI 인자 | YAML 설정 파일 |
| LoRA 병합 | 별도 스크립트 | `fuse_adapters` 내장 |
| 비디오 훈련 | Wan 등 일부 | HunyuanVideo, Wan 등 폭넓음 |

## 코드 위치

프로젝트 루트: `C:\Projects\kohya_anima\Reference\diffusion-pipe\`
