# Flux

<span class="version-badge">Black Forest Labs / Flux.1</span>

## 🔗 모델 카드

| 모델 | HuggingFace |
|---|---|
| **Flux.1-dev** | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Flux.1-schnell | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

## 부품 조합

| 부품 | 선택 | 이론 참조 |
|---|---|---|
| **아키텍처** | DiT (Diffusion Transformer) | [→ 이론/아키텍처](../theory/architecture/index.md) |
| **확산 방식** | Flow Matching (직선 OT 경로) | [→ 이론/확산수학/Flow Matching](../theory/diffusion/flow-matching.md) |
| **예측 유형** | Velocity ($\epsilon - x_0$) | [→ 이론/훈련이론/목적함수](../theory/training/objectives.md) |
| **Text Encoder** | CLIP + T5-XXL | [→ 이론/아키텍처](../theory/architecture/index.md) |
| **VAE** | Flux VAE (16ch latent) | — |

## Flux 고유 특징

### 1. 해상도 적응형 Timestep Shift

Flux는 이미지 해상도에 따라 timestep 분포를 조절합니다:

\[
\mu = 0.5 + \frac{H \times W}{256^2} \cdot 1.15
\]

\[
t = \sigma(\mu + s \cdot z), \quad z \sim \mathcal{N}(0,1)
\]

큰 이미지 → 더 많은 노이즈가 필요 (content 밀도 증가) → $\mu$를 키워 높은 $t$ 쪽을 더 샘플.

### 2. Discrete Flow Shift

추론 시 sigma를 비선형 변환:

\[
\sigma' = \frac{s \cdot \sigma}{1 + (s-1)\sigma}
\]

- `dev` 모델: $s = 3.0$
- `schnell` 모델: $s = 1.0$

### 3. 변형 모델

| 변형 | 특징 |
|---|---|
| flux1-dev | 기본 (guidance distillation, $s=3.0$) |
| flux1-schnell | 빠른 추론 (step distillation, 1~4 steps) |

## 도구별 구현

| 도구 | Flux 지원 | 상세 |
|---|---|---|
| ComfyUI | ✅ 네이티브 | [→ 도구/ComfyUI](../tools/comfyui/index.md) |
| kohya (sd-scripts) | ✅ `flux_train.py` | [→ 도구/kohya](../tools/kohya/index.md) |
| DiffSynth | ✅ | [→ 도구/DiffSynth](../tools/diffsynth/index.md) |
| HuggingFace diffusers | ✅ `FluxPipeline` | [→ 도구/HuggingFace](../tools/huggingface/index.md) |

