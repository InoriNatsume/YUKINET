# Diffusion/Flow 모델 훈련 코드 vs 추론/ComfyUI 처리 방식 (모델별 개발자 비교)

## 문서 목적
- 이 문서는 현재 워크스페이스의 코드 기준으로, 모델별로 훈련 코드와 추론 코드(특히 ComfyUI)의 처리 방식 차이를 개발자 관점에서 정리한다.
- 목표 독자는 다음이다.
  - 스크립트 의존을 줄이고 자체 훈련/추론 코드를 작성하려는 개발자
  - 기존 코드베이스를 안전하게 커스터마이징하려는 개발자
- 근거는 줄번호가 아니라 파일/클래스/함수 단위로 제시한다.

## 1) 코드 근거 맵 (줄번호 없음)
### 훈련 중심 코드베이스
- `ai-toolkit/toolkit/stable_diffusion_model.py` (`StableDiffusion.load_model`, `StableDiffusion.generate_images`, `StableDiffusion.save`)
- `ai-toolkit/jobs/process/BaseSDTrainProcess.py` (`BaseSDTrainProcess.run`, `BaseSDTrainProcess.save`, `BaseSDTrainProcess.setup_adapter`)
- `sd-scripts/sdxl_train.py`, `sd-scripts/flux_train.py`, `sd-scripts/sd3_train.py`
- `sd-scripts/sdxl_train_network.py`, `sd-scripts/flux_train_network.py`, `sd-scripts/sd3_train_network.py`, `sd-scripts/train_network.py`
- `diffusion-pipe/train.py`
- `diffusion-pipe/models/sdxl.py`, `diffusion-pipe/models/flux.py`, `diffusion-pipe/models/flux2.py`, `diffusion-pipe/models/sd3.py`, `diffusion-pipe/models/qwen_image.py`, `diffusion-pipe/models/wan/wan.py`
- `diffusion-pipe/models/base.py`, `diffusion-pipe/utils/saver.py`
- `DiffSynth-Studio/diffsynth/diffusion/base_pipeline.py`, `DiffSynth-Studio/diffsynth/diffusion/training_module.py`, `DiffSynth-Studio/diffsynth/diffusion/runner.py`
- `DiffSynth-Studio/examples/*/model_training/train.py`

### 추론/워크플로 중심 코드베이스
- `ComfyUI-0.13.0/ComfyUI-0.13.0/execution.py` (`execute_async`에서 `torch.inference_mode` 실행)
- `ComfyUI-0.13.0/ComfyUI-0.13.0/nodes.py` (`CheckpointLoaderSimple`, `LoraLoader`, `UNETLoader`, `CLIPLoader`, `VAELoader`)
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/sd.py` (`load_checkpoint_guess_config`, `load_lora_for_models`)
- `ComfyUI-0.13.0/ComfyUI-0.13.0/folder_paths.py` (모델 카테고리 경로)
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/supported_models.py`, `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/model_detection.py`
- `sd-scripts/sdxl_minimal_inference.py`, `sd-scripts/flux_minimal_inference.py`, `sd-scripts/sd3_minimal_inference.py`
- `DiffSynth-Studio/examples/*/model_inference/*.py`

### 참고
- ComfyUI도 `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy_extras/nodes_train.py`에 실험적 학습 노드가 있지만, 아키텍처 중심은 여전히 추론/그래프 실행이다.

## 2) 훈련 코드와 추론 코드의 구조적 차이 (모델 공통)
| 구분 | 훈련 코드 | 추론 코드 / ComfyUI |
|---|---|---|
| 실행 컨텍스트 | `forward -> loss -> backward -> optimizer.step` | `no_grad/inference_mode` 기반 샘플링 |
| 핵심 상태 | 모델 + optimizer/scheduler + dataloader 진행 상태 | 모델 객체 + 노드 출력 캐시 + 메모리 오프로딩 상태 |
| 파라미터 제어 | `requires_grad` 대상 엄격 선택 (LoRA/full) | 대부분 고정, 런타임 LoRA patch/merge |
| 저장물 | adapter/LoRA/full model + 학습 재개 상태 | 이미지/영상 결과 + 워크플로(JSON), 모델은 별도 로더로 참조 |
| 실패 패턴 | 학습 재개 불일치, optimizer state 누락, fp8/오프로딩 안정성 | 모델 타입 오탐지, LoRA 키 매핑 불일치, VRAM 오프로딩 타이밍 |

## 3) 모델 패밀리별 상세 비교

### 3-1) SDXL
#### 핵심 컴포넌트
- U-Net + VAE + 텍스트 인코더 2개(CLIP-L, OpenCLIP 계열) + size/time embedding 보조 입력.

#### 훈련 코드에서의 처리
- `sd-scripts/sdxl_train.py`
  - `train`에서 UNet/TE1/TE2 학습 여부를 LR 플래그로 분리해 제어한다.
  - `sdxl_train_util.load_target_model`로 `ckpt` vs `diffusers`를 자동 분기 로딩한다.
- `sd-scripts/sdxl_train_network.py`
  - `SdxlNetworkTrainer`가 공통 `NetworkTrainer`를 상속해 LoRA/OFT 등 네트워크 학습 경로를 사용한다.
- `diffusion-pipe/models/sdxl.py`
  - `SDXLPipeline.configure_adapter`에서 UNet 블록과 텍스트 인코더 선형층을 LoRA 타깃으로 지정한다.
  - `SDXLPipeline.save_adapter`는 Kohya 호환 LoRA로 변환 저장한다.
  - `SDXLPipeline.save_model`은 UNet/VAE/TE를 SDXL checkpoint 형태로 다시 매핑 저장한다.
- `ai-toolkit/toolkit/stable_diffusion_model.py`
  - `StableDiffusion.load_model`의 `is_xl` 분기에서 SDXL 파이프라인 로딩.
  - `StableDiffusion.save`에서 `safetensors` 단일파일 또는 diffusers 폴더 저장을 분기.

#### 추론/ComfyUI에서의 처리
- `ComfyUI-0.13.0/ComfyUI-0.13.0/nodes.py`
  - `CheckpointLoaderSimple`, `DualCLIPLoader(type=sdxl)`, `VAELoader`를 조합해 워크플로를 구성.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/sd.py`
  - `load_checkpoint_guess_config`가 키 구조 기반으로 모델 타입을 추론 로딩.
- `sd-scripts/sdxl_minimal_inference.py`
  - 훈련 코드 의존 없이 TE 임베딩/size embedding/샘플러를 직접 조합하는 최소 추론 예제를 제공.

#### 커스터마이징 포인트
- SDXL은 TE 2개 처리, add_time_ids 생성, UNet 입력 결합 순서가 틀리면 결과가 즉시 붕괴한다.
- LoRA 저장 포맷(Kohya/diffusers/Comfy)이 섞일 때 키 매핑 계층을 분리 구현하는 것이 안전하다.

### 3-2) FLUX.1 / FLUX.2
#### 핵심 컴포넌트
- FLUX.1: Flow-matching Transformer + CLIP-L + T5 + VAE.
- FLUX.2: 대형 DiT 변형 + Qwen/Mistral 계열 텍스트 인코더 조합 + VAE.

#### 훈련 코드에서의 처리
- `sd-scripts/flux_train.py`, `sd-scripts/flux_train_network.py`
  - full fine-tune 경로와 network(LoRA) 경로를 분리 운영.
  - `FluxNetworkTrainer`가 CLIP/T5 캐시, fp8, block swap 옵션을 학습 경로에 통합.
- `diffusion-pipe/models/flux.py`
  - `FluxPipeline.prepare_inputs`에서 flow-matching 타깃(`x0-x1`)과 이미지/텍스트 토큰 구조를 구성.
  - `FluxPipeline.save_model`은 diffusers 키를 BFL 단일 safetensors 구조로 재매핑 저장.
- `diffusion-pipe/models/flux2.py`
  - `Flux2Pipeline`이 `ComfyPipeline` 기반으로 동작하며, 모델 차원으로 4B/9B/32B를 자동 감지.
  - `prepare_inputs`가 edit/control latents까지 포함한 입력 경로를 처리.
- `DiffSynth-Studio/examples/flux/model_training/train.py`, `DiffSynth-Studio/examples/flux2/model_training/train.py`
  - `DiffusionTrainingModule.switch_pipe_to_training_mode`로 학습 대상 동결/LoRA 주입을 공통화.
  - `launch_training_task`로 optimizer/accumulate 루프를 표준화.

#### 추론/ComfyUI에서의 처리
- `sd-scripts/flux_minimal_inference.py`로 CLIP/T5/AE/Flow model 직접 구성 추론.
- `DiffSynth-Studio/diffsynth/pipelines/flux_image.py`, `DiffSynth-Studio/diffsynth/pipelines/flux2_image.py`
  - `PipelineUnit` 체인으로 prompt/noise/control/edit를 단계적으로 구성.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/ldm/flux/model.py`
  - Comfy 내부 모델 구현을 기반으로 그래프 노드에서 조합 실행.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/supported_models.py`, `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/model_detection.py`
  - `flux`, `flux2` 타입 감지 및 텍스트 인코더 조합 분기.

#### 커스터마이징 포인트
- FLUX 계열은 timestep 분포(`logit_normal`/`uniform`, shift)와 packed latent 규칙이 정확해야 학습/추론 일관성이 유지된다.
- block swap, offload, fp8 정책이 성능보다 먼저 안정성에 영향하므로 런타임 정책 객체로 분리하는 편이 좋다.

### 3-3) SD3
#### 핵심 컴포넌트
- MMDiT + CLIP-L + CLIP-G + T5XXL + VAE.

#### 훈련 코드에서의 처리
- `sd-scripts/sd3_train.py`, `sd-scripts/sd3_train_network.py`
  - `Sd3NetworkTrainer.load_target_model`에서 MMDiT/TE 3종/VAE를 구성.
  - 텍스트 인코더 캐시와 학습 대상(CLIP vs T5) 분리를 명시적으로 제어.
- `diffusion-pipe/models/sd3.py`
  - `SD3Pipeline.prepare_inputs`에서 CLIP+T5 임베딩 결합, flow-matching 입력/타깃 구성.
- `ai-toolkit/toolkit/stable_diffusion_model.py`
  - `is_v3` 분기에서 SD3 로딩/샘플링 경로를 제공.

#### 추론/ComfyUI에서의 처리
- `sd-scripts/sd3_minimal_inference.py`
  - MMDiT 샘플링 루프를 최소 코드로 재현.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/nodes.py`
  - `CLIPLoader(type=sd3)` 등으로 SD3용 인코더 조합을 선택.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/text_encoders/sd3_clip.py`
  - SD3 계열 텍스트 인코더 조합 처리.

#### 커스터마이징 포인트
- SD3는 TE가 3개라서 메모리 정책과 캐시 정책을 분리하지 않으면 학습 루프가 쉽게 불안정해진다.
- SD3 LoRA를 다른 런타임에서 쓰려면 키 prefix 규칙(`transformer.*`, `diffusion_model.*`) 통일이 먼저다.

### 3-4) Qwen-Image
#### 핵심 컴포넌트
- Qwen 이미지 Transformer + Qwen 계열 텍스트/VLM 인코더 + VAE + (edit용) processor.

#### 훈련 코드에서의 처리
- `diffusion-pipe/models/qwen_image.py`
  - text encoder/vae/transformer를 safetensors에서 수동 구성하고 키를 재매핑 로드.
  - `prepare_inputs`에서 가변 길이 prompt embedding 패딩/attention mask/control latents를 처리.
  - `save_adapter`가 ComfyUI 호환 prefix(`diffusion_model.`)를 붙여 저장.
- `DiffSynth-Studio/examples/qwen_image/model_training/train.py`
  - `QwenImagePipeline.from_pretrained` + `switch_pipe_to_training_mode` 패턴으로 LoRA/full 경로를 공통 처리.

#### 추론/ComfyUI에서의 처리
- `DiffSynth-Studio/diffsynth/pipelines/qwen_image.py`
  - edit/inpaint/layered/controlnet을 `PipelineUnit` 체인으로 조합.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/ldm/qwen_image/model.py`
  - Qwen-Image 모델 본체 추론 구현.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/text_encoders/qwen_image.py`
  - Qwen 텍스트 인코딩/토크나이저 처리.

#### 커스터마이징 포인트
- Qwen-Image는 텍스트 길이와 VLM 입력(특히 edit 모드) 처리 규칙이 모델 품질과 직접 연결된다.
- 가변 길이 배치에서 padding/attention mask 정합성을 백엔드 공통 유틸로 고정하는 것이 중요하다.

### 3-5) Wan (video)
#### 핵심 컴포넌트
- Wan video Transformer + UMT5 text encoder + video VAE + (variant별) CLIP/audio 등 보조 모듈.

#### 훈련 코드에서의 처리
- `diffusion-pipe/models/wan/wan.py`
  - 모델 키/설정으로 `t2v`, `i2v`, `flf2v`, `ti2v`를 자동 감지.
  - `get_call_vae_fn`에서 variant별 조건(`y`, `clip_context`)을 생성.
  - `prepare_inputs`에서 5D latent(영상) 기준 timestep 샘플링과 target 구성.
- `DiffSynth-Studio/examples/wanvideo/model_training/train.py`
  - `WanVideoPipeline.from_pretrained` + 태스크별 입력(video/audio/control) 통합.

#### 추론/ComfyUI에서의 처리
- `DiffSynth-Studio/diffsynth/pipelines/wan_video.py`
  - `PipelineUnit` 기반으로 t2v/i2v/s2v/vace/animate 경로를 조합.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/ldm/wan/model.py`
  - Wan 계열 video model 추론 구현.
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/text_encoders/wan.py`
  - Wan 텍스트 인코더 로딩 처리.

#### 커스터마이징 포인트
- Wan은 variant별 입력 계약이 크게 달라서, 단일 `forward`에서 분기하는 대신 variant 어댑터 계층을 별도로 두는 편이 유지보수에 유리하다.
- 영상 학습은 공간/시간 해상도 반올림 규칙(height/width/frame division factor)을 통일하지 않으면 데이터-모델 shape 불일치가 자주 발생한다.

## 4) 저장 포맷/호환성 관점 요약
| 모델 | 훈련 코드의 대표 저장물 | ComfyUI 쪽 사용 시 핵심 |
|---|---|---|
| SDXL | LoRA(`safetensors`), full checkpoint 또는 diffusers 폴더 | LoRA 키 포맷(Kohya/PEFT)과 checkpoint 키 매핑 정합성 |
| FLUX.1/2 | LoRA + model(`safetensors`) + 학습 상태 | `transformer`/`diffusion_model` prefix 규칙 맞추기 |
| SD3 | LoRA 중심 + (환경에 따라) full model | SD3 전용 TE 조합(CLIP-L/G/T5)과 모델 타입 감지 |
| Qwen-Image | adapter_model/model.safetensors + 설정 | edit 모드용 processor, text/image 조건 입력 키 정합성 |
| Wan | adapter/model.safetensors + variant 설정 | t2v/i2v/ti2v variant별 입력 계약과 VAE/TE 조합 일치 |

## 5) 스크립트 의존을 줄이기 위한 권장 아키텍처
### 5-1) 최소 인터페이스(권장)
- `ModelFamilyBackend`
  - 책임: 모델별 컴포넌트 로딩, `prepare_inputs`, `forward_model`, `compute_target`.
- `ConditionEncoder`
  - 책임: 텍스트/이미지/영상 조건 인코딩 통합.
- `AdapterManager`
  - 책임: LoRA/adapter 주입, merge/unmerge, 저장 포맷 변환.
- `CheckpointIO`
  - 책임: full/adapter 저장, optimizer/scheduler/dataloader 상태 포함 재개.
- `RuntimePolicy`
  - 책임: dtype(fp16/bf16/fp8), offload/block-swap, grad-checkpoint 정책.
- `InferenceExecutor`
  - 책임: 샘플링 루프 또는 그래프 실행(Comfy 방식) 추상화.

### 5-2) 모델별 구현 우선 체크리스트
- SDXL
  - TE 2개 결합 규칙, add_time_ids, VAE scale/shift 처리.
- FLUX
  - latent pack/unpack, timestep 샘플링, guidance/shift 규칙.
- SD3
  - CLIP-L/G + T5 결합, 메모리 정책, fp8 캐스팅 경계.
- Qwen-Image
  - 가변 길이 텍스트 배치, edit 모드 processor, control latents.
- Wan
  - 5D latent 처리, variant별 추가 조건(y/clip/audio), frame 규칙.

### 5-3) 현실적인 마이그레이션 순서
1. 최소 추론 엔진부터 고정한다.
2. LoRA 학습만 먼저 독립시킨다.
3. full fine-tune과 분산/재개 로직을 마지막에 붙인다.
4. 마지막 단계에서 ComfyUI 워크플로 입출력과 포맷 호환을 맞춘다.

## 6) 결론
- 훈련 코드와 ComfyUI의 본질적 차이는 `가중치 업데이트 중심` vs `그래프 추론 실행 중심`이다.
- 실제 개발에서는 모델별 수식보다 먼저, 입력 계약/저장 포맷/런타임 정책을 분리한 아키텍처가 유지보수성과 재사용성을 결정한다.
- 이 문서의 파일 단위 근거를 기준으로 모듈 경계를 먼저 설계하면, 특정 스크립트에 묶이지 않는 자체 구현으로 넘어가기 훨씬 쉬워진다.