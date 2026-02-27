# Z-Image — DiffSynth-Studio 이미지 생성 모델 시리즈

> DiffSynth-Studio · Flux 기반 이미지 생성

## 🔗 모델 카드

| 모델 | HuggingFace |
|---|---|
| **Z-Image (ComfyUI)** | [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) |
| Z-Image Turbo Training Adapter | [ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter) |

## Z-Image 시리즈 변형

| 모델 | 기반 | 특징 |
|---|---|---|
| **Z-Image** | Flux 기반 | 기본 이미지 생성 |
| **Z-Image-Turbo** | Flux Distillation | 빠른 추론 (소수 step) |
| **Z-Image-Omni-Base** | Flux | 범용 기본 모델 |
| **Z-Image-Turbo-Fun-Controlnet-Tile** | Flux + ControlNet | 타일 기반 제어 |
| **Z-Image-Turbo-Fun-Controlnet-Union** | Flux + ControlNet | 유니온 제어 |

## 부품 조합

| 부품 | 선택 | 이론 참조 |
|---|---|---|
| **아키텍처** | DiT (Flux 계열) | [→ 이론/아키텍처](../theory/architecture/index.md) |
| **확산 방식** | Flow Matching | [→ 이론/Flow Matching](../theory/diffusion/flow-matching.md) |
| **텍스트 인코더** | Qwen3-4B (Lumina2 타입) | — |
| **VAE** | Flux VAE | — |

## 훈련 설정 (diffusion-pipe)

```toml
[model]
type = 'z_image'
diffusion_model = '/path/to/z_image_turbo_bf16.safetensors'
vae = '/path/to/flux_vae.safetensors'
text_encoders = [
    {path = '/path/to/qwen_3_4b.safetensors', type = 'lumina2'}
]
# Z-Image-Turbo 훈련 시 필수
merge_adapters = ['/path/to/zimage_turbo_training_adapter_v1.safetensors']
dtype = 'bfloat16'
```

!!! tip "Turbo 훈련 시 adapter 필수"
    Z-Image-Turbo를 훈련할 때는 `merge_adapters`에 [ostris의 training adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter)를 반드시 포함해야 합니다.

## 특수 학습 전략 (DiffSynth-Studio)

### Trajectory Imitation (궤적 모방)
Teacher 모델의 중간 상태(trajectory)를 모방하여 학습.

### Differential Training (차분 학습)
기존 LoRA에서 변경된 부분만 추가 학습하는 최적화 전략.

## 도구별 지원

| 도구 | 버전 | 지원 |
|---|---|---|
| DiffSynth-Studio | {{ ver.diffsynth }} | ✅ 네이티브 (훈련+추론) |
| diffusion-pipe | {{ ver.diffusion_pipe }} | ✅ LoRA + Full FT + fp8 |
| ComfyUI | {{ ver.comfyui }} | ✅ 추론 (ComfyUI 모델 파일) |
