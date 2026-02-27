# Anima — Cosmos-Predict2 기반 애니메이션/일러스트 모델

> CircleStone Labs × Comfy Org · 기반: NVIDIA Cosmos-Predict2-2B

<span class="version-badge">Preview</span>

## 🔗 모델 카드

| 모델 | HuggingFace |
|---|---|
| **Anima (Preview)** | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) |
| NVIDIA Cosmos-Predict2 2B (베이스) | [nvidia/Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image) |
| NVIDIA Cosmos-Predict2 14B | [nvidia/Cosmos-Predict2-14B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image) |

## Cosmos-Predict2 vs Anima

### NVIDIA Cosmos-Predict2 (베이스 모델)

NVIDIA가 개발한 **물리 AI용 세계 기반 모델(World Foundation Model)**. 실세계 동역학을 이해하고 합성 데이터를 생성하기 위해 설계됨.

| 항목 | 내용 |
|---|---|
| **파라미터** | 2B / 14B |
| **아키텍처** | DiT (Diffusion Transformer) |
| **VAE** | Wan 2.1 VAE (공유) |
| **텍스트 인코더** | Old T5-XXL (구 버전 — 일반 T5와 다름) |
| **확산 방식** | Flow Matching |
| **라이선스** | NVIDIA Open Model License (상업적 사용 가능) |
| **목적** | Physical AI — 로보틱스, 자율주행 시뮬레이션용 |

### Anima (파인튜닝 파생 모델)

CircleStone Labs가 Cosmos-Predict2-2B를 **애니메이션/일러스트 도메인에 특화하여 파인튜닝**한 모델.

| 항목 | Cosmos-Predict2 | **Anima** |
|---|---|---|
| **파라미터** | 2B | 2B (동일 구조) |
| **텍스트 인코더** | Old T5-XXL | **Qwen3-0.6B** (변경) |
| **강점** | 실사, 물리 시뮬레이션 | **애니/일러스트, 아트워크** |
| **일본어 프롬프트** | 제한적 | ✅ Qwen3 덕분에 반응성 우수 |
| **라이선스** | NVIDIA Open Model | 비상업적 (CircleStone) + NVIDIA OML |
| **상태** | 안정 릴리스 | **Preview** (최종 버전에서 변경될 수 있음) |

!!! warning "Preview 주의사항"
    - Preview 버전에서 훈련한 LoRA는 **최종 버전과 호환되지 않을 가능성이 높음**
    - 기반 모델이 아직 훈련 중이므로 가중치가 변경될 예정
    - LoRA를 공유할 때는 반드시 "Preview 버전용"이라고 명시할 것

## 부품 조합

| 부품 | 선택 | 이론 참조 |
|---|---|---|
| **아키텍처** | DiT (Cosmos-Predict2 계열) | [→ 이론/아키텍처](../theory/architecture/index.md) |
| **확산 방식** | Flow Matching | [→ 이론/Flow Matching](../theory/diffusion/flow-matching.md) |
| **텍스트 인코더** | Qwen3-0.6B | — |
| **VAE** | Wan 2.1 VAE (Qwen-Image VAE 호환) | — |

## 훈련 설정 (diffusion-pipe)

```toml
[model]
type = 'anima'
transformer_path = '/path/to/anima-preview.safetensors'
vae_path = '/path/to/qwen_image_vae.safetensors'
llm_path = '/path/to/qwen_3_06b_base.safetensors'
dtype = 'bfloat16'
# LLM adapter 학습률 — 0이면 adapter 학습 비활성화
llm_adapter_lr = 0
```

### 훈련 팁

| 설정 | 권장 | 이유 |
|---|---|---|
| **학습률** | 다른 모델보다 낮게 설정 | Anima가 더 민감 |
| **`llm_adapter_lr`** | `0` (소규모 데이터셋) | 안정적 훈련. 새 개념이 많으면 활성화 시도 |
| **LoRA 저장 형식** | ComfyUI format | — |

## 도구별 지원

| 도구 | 버전 | 지원 |
|---|---|---|
| diffusion-pipe | {{ ver.diffusion_pipe }} | ✅ LoRA + Full FT + fp8 |
| ComfyUI | {{ ver.comfyui }} | ✅ 추론 (공식 ComfyUI 모델 파일) |

## Cosmos 계보

```
NVIDIA Cosmos 1.0 (Text2World, 7B)
    │ 비디오 생성 — 물리 시뮬레이션 특화
    │ VAE: Cosmos CV8x8x8
    │ TE: Old T5-XXL
    │ ⚠ 파인튜닝 어려움 (고정 해상도, 높은 VRAM)
    │
    ├── Cosmos-Predict2 (2B / 14B)
    │       이미지 생성으로 전환
    │       VAE: Wan 2.1 VAE
    │       TE: Old T5-XXL
    │       ✅ LoRA + Full FT 지원
    │
    └───── **Anima** (2B, CircleStone Labs)
                Cosmos-Predict2-2B 파인튜닝
                TE: Qwen3-0.6B (변경)
                도메인: 애니메이션/일러스트
                상태: Preview
```
