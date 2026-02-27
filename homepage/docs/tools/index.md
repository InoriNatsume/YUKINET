# 🔧 도구

각 도구가 [이론](../theory/index.md)과 [모델](../models/index.md)을 **실제 코드로 어떻게 구현하는지** 분석합니다.

## ComfyUI

> v{{ ver.comfyui }}

노드 기반 추론. KSampler → CFGGuider → k_diffusion 체인.
sigma 파라미터화, Custom Sampling 그래프.

→ [상세 보기](comfyui/index.md)

## kohya (sd-scripts)

> v{{ ver.sdscripts }}

CLI 기반 훈련. LoRA, DreamBooth, Full Fine-tune.
데이터 파이프라인, 메모리 최적화, fused backward pass.

→ [상세 보기](kohya/index.md)

## diffusion-pipe

> v{{ ver.diffusion_pipe }}

DeepSpeed Pipeline Parallelism 기반 훈련.
대형 모델 Full FT를 단일 GPU에서 가능하게 하는 고효율 도구.

→ [상세 보기](diffusion-pipe/index.md)

## HuggingFace

> diffusers v{{ ver.diffusers }} · transformers v{{ ver.transformers }}

diffusers Pipeline, PEFT (LoRA), Accelerate (분산학습).
Scheduler 추상화, 모델 허브 통합.

→ [상세 보기](huggingface/index.md)

## DiffSynth-Studio

> v{{ ver.diffsynth }}

통합 파이프라인. 이미지 + 비디오 생성/훈련.
Bell-shaped 가중치, UnifiedDataset.

→ [상세 보기](diffsynth/index.md)
