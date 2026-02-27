# Anima 모델 개발자 문서 (diffusion-pipe / ComfyUI / sd-scripts SD3 참고)

## 문서 정보
- 목적: Anima를 기준으로 학습 코드와 추론 코드의 차이를 정리하고, 스크립트 의존 없이 자체 학습/추론 코드를 구현하거나 기존 코드를 커스터마이징할 때 필요한 기술 포인트를 제공한다.
- 인코딩: UTF-8 with BOM.
- 근거 표기 방식: 파일 경로 + 클래스/함수 이름 기준으로만 표기한다. 라인 번호는 의도적으로 제외한다.
- 우선순위 원칙: **diffusion-pipe 구현을 기준(정본)으로 우선 적용**하고, SD3 쪽 Anima 관련 내용은 개발중 기능/비교 레퍼런스로만 다룬다.

## 1) 핵심 결론
1. Anima는 코드 구조상 `Cosmos Predict2` 계열 DiT에 `LLM Adapter`를 결합한 변형 모델이다.
2. **개발 기준은 diffusion-pipe**다. Anima 제작자 측 구현이므로, 기능 해석/커스터마이징/재구현 시 diffusion-pipe 동작을 우선 진실원천으로 본다.
3. `ComfyUI`는 추론 런타임 기준 구현으로 사용하고, `sd-scripts`의 SD3 관련 코드는 Anima 자체라기보다 "개발중 기능 또는 Flow Matching 패턴 비교 레퍼런스"로 사용한다.

## 2) 근거 코드 맵 (라인번호 없음)

### diffusion-pipe (Anima 학습 핵심)
- 모델 라우팅: `diffusion-pipe/train.py`의 `model_type == 'anima'` 분기에서 `CosmosPredict2Pipeline` 사용.
- Anima 파이프라인: `diffusion-pipe/models/cosmos_predict2.py`.
  - `CosmosPredict2Pipeline.__init__`
  - `CosmosPredict2Pipeline.load_diffusion_model`
  - `CosmosPredict2Pipeline.get_call_text_encoder_fn`
  - `CosmosPredict2Pipeline.prepare_inputs`
  - `CosmosPredict2Pipeline.to_layers`
  - `CosmosPredict2Pipeline.get_param_groups`
  - `CosmosPredict2Pipeline.save_adapter`
  - `CosmosPredict2Pipeline.save_model`
  - 내부 레이어: `InitialLayer`, `LLMAdapterLayer`, `TransformerLayer`, `FinalLayer`.
- 체크포인트 저장: `diffusion-pipe/utils/saver.py`.
- 공식 지원 모델 설명: `diffusion-pipe/docs/supported_models.md`의 `Anima` 섹션.

### ComfyUI (Anima 추론 핵심)
- 모델 감지: `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/model_detection.py`.
  - `detect_unet_config`: `llm_adapter...` 키가 있으면 `image_model='anima'`로 판별.
  - `unet_prefix_from_state_dict`: `net.` prefix를 후보로 지원.
- 모델 등록: `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/supported_models.py`의 `Anima` 클래스.
- 베이스 모델 조건 처리: `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/model_base.py`의 `Anima.extra_conds`.
- 텍스트 인코더/토크나이저: `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/text_encoders/anima.py`.
  - `AnimaTokenizer`, `AnimaTEModel.encode_token_weights`.
- 디퓨전 모델 본체: `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/ldm/anima/model.py`.
  - `Anima(MiniTrainDIT)` + `LLMAdapter` + `preprocess_text_embeds`.
- 텍스트 인코더 로딩 분기: `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/sd.py`의 `TEModel.QWEN3_06B` 처리.
- LoRA 키 매핑 일반 규칙: `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/lora.py`.

### sd-scripts SD3 (Anima 개발중 기능/비교 레퍼런스)
- 학습 엔트리: `sd-scripts/sd3_train.py`, `sd-scripts/sd3_train_network.py`.
- 최소 추론: `sd-scripts/sd3_minimal_inference.py`.
- 텍스트 전략: `sd-scripts/library/strategy_sd3.py`.
- 노이즈/스케줄/로스 가중치: `sd-scripts/library/sd3_train_utils.py`.
- 모델 로딩: `sd-scripts/library/sd3_utils.py`.
- SD3 LoRA 대상 모듈 구성: `sd-scripts/networks/lora_sd3.py`.

## 3) Anima 학습 코드 vs 추론 코드 비교

| 구분 | diffusion-pipe (학습) | ComfyUI (추론) | 개발 시 의미 |
|---|---|---|---|
| 모델 진입점 | `train.py`에서 `anima -> CosmosPredict2Pipeline` | state_dict 감지 후 `supported_models.Anima` | 학습/추론에서 클래스 이름이 다르므로 "동일 개념 매핑표"가 필요 |
| 텍스트 인코딩 | `get_call_text_encoder_fn`: Qwen 경로 토크나이즈 + T5 토크나이즈 동시 수행, `prompt_embeds`와 `t5_input_ids`를 함께 반환 | `AnimaTokenizer`: Qwen + T5 동시 토크나이즈, `AnimaTEModel`이 `t5xxl_ids`, `t5xxl_weights`를 extra cond로 출력 | Anima는 단일 텍스트 인코더가 아니라 "Qwen 임베딩 + T5 토큰 ID" 조합 모델 |
| LLM Adapter 적용 | `LLMAdapterLayer.forward`에서 `source_hidden_states`, `target_input_ids`, `target_attention_mask`, `source_attention_mask`를 모두 전달 가능 | `model_base.Anima.extra_conds -> diffusion_model.preprocess_text_embeds(cross_attn, t5xxl_ids)` 형태로 적용, 이후 weight 적용/패딩 | 마스크 사용 유무/시점이 다르다. 자체 엔진에서 어느 방식으로 통일할지 먼저 결정해야 함 |
| 시퀀스 처리 | `_tokenize(..., max_length=512)` + 학습 시 attention mask 사용 | 추론 시 `cross_attn` 길이 < 512면 pad | 길이 규약 불일치로 shape 오류가 나기 쉬움 |
| 노이즈 구성 | `prepare_inputs`: `x_t=(1-t)x0+t*noise`, `target=noise-x0` (rectified-flow velocity target) | 샘플링 단계에서 flow 계열 시그마 스케줄(Anima 기본 shift=3.0)로 역적분 | 학습 목표와 샘플러 수식이 쌍으로 맞아야 함 |
| 파라미터 학습 제어 | `get_param_groups`: `llm_adapter_lr` 별도 분리, `0`이면 사실상 동결 | 추론은 학습률 개념 없음 | 작은 데이터셋에서는 `llm_adapter_lr=0`부터 시작하는 전략이 안전 |
| 저장 포맷 | `save_adapter`: `diffusion_model.` prefix, `save_model`: `net.` prefix | `model_detection.unet_prefix_from_state_dict`가 `net.` 인식, LoRA는 `diffusion_model.` 관례 지원 | 포맷 호환을 위해 prefix 규약 유지가 매우 중요 |

## 4) Anima와 SD3 구현의 구체적 차이

해석 원칙:
- 아래 표는 "차이점 파악"을 위한 비교표다.
- 실제 설계/구현 결정은 diffusion-pipe 쪽 동작을 우선한다.

| 항목 | Anima (diffusion-pipe + ComfyUI) | SD3 (sd-scripts 기준) |
|---|---|---|
| 백본 | `MiniTrainDIT` 계열 (`cosmos_predict2` 기반) + `LLMAdapter` | `MMDiT` |
| 텍스트 조건 | Qwen hidden states + T5 token IDs/weights | CLIP-L + CLIP-G + T5XXL 임베딩 결합 |
| 학습 타깃 표현 | `target = noise - latents`를 직접 예측하는 형태 | `sd3_train.py`는 preconditioning 후 `target = latents` 형태를 사용 |
| timestep 표현 | Anima 학습 코드는 기본적으로 0~1 스케일 `t` | SD3는 0~1000 timestep 인덱스/시그마 체계를 강하게 사용 |
| 텍스트 캐시 | Anima는 `cache_text_embeddings` 단일 플래그 중심 | SD3는 CLIP/T5 부분 캐시, partial cache, dropout 대응 등 전략화 |
| LoRA 학습 범위 | diffusion-pipe Anima는 `Block` + `TransformerBlock(LLM Adapter)` 대상 | SD3 LoRA는 text encoder(선택) + MMDiT 대상 모듈 체계 |

핵심 포인트:
- SD3 코드를 그대로 가져오면 Anima의 조건 결합(특히 T5 ID를 LLM Adapter에 넣는 경로)이 비어버리기 쉽다.
- Anima 자체 엔진을 만들 때는 "SD3의 Flow Matching 프레임"을 참고하되, 텍스트 조건 파이프라인은 Anima 방식으로 별도 구현해야 한다.

## 5) 아키텍처 스펙 (diffusion-pipe 정본 기준)

### 5-1) DiT 본체 (`MiniTrainDIT`) 핵심 스펙
기준 코드:
- `diffusion-pipe/models/cosmos_predict2.py`의 `get_dit_config`
- `diffusion-pipe/models/cosmos_predict2_modeling.py`의 `MiniTrainDIT`

주요 값:
- 입력/출력: `in_channels`(체크포인트에서 자동 추론), `out_channels=16`
- 패치: `patch_spatial=2`, `patch_temporal=1`
- 최대 길이 축: `max_img_h=512`, `max_img_w=512`, `max_frames=128`
- 어텐션 컨텍스트 폭: `crossattn_emb_channels=1024`
- 위치 임베딩: `pos_emb_cls='rope3d'`, `pos_emb_learnable=true`, `pos_emb_interpolation='crop'`
- FPS 범위: `min_fps=1`, `max_fps=30`
- AdaLN LoRA: `use_adaln_lora=true`, `adaln_lora_dim=256`
- RoPE 외삽:
  - `in_channels=16`일 때 `h=4.0`, `w=4.0`, `t=1.0`
  - `in_channels=17`일 때 `h=3.0`, `w=3.0`, `t=1.0`
  - 공통: `extra_h/w/t_extrapolation_ratio=1.0`, `rope_enable_fps_modulation=false`
- 블록/헤드 자동 결정(`model_channels` 기반):
  - `2048 -> num_blocks=28, num_heads=16`
  - `5120 -> num_blocks=36, num_heads=40`
  - `1280 -> num_blocks=20, num_heads=20`

### 5-2) LLM Adapter 스펙
기준 코드:
- `diffusion-pipe/models/llm_adapter.py`의 `LLMAdapter`
- `diffusion-pipe/models/cosmos_predict2_modeling.py`의 `MiniTrainDIT(use_llm_adapter)`

주요 값(Anima 경로):
- `source_dim=1024`, `target_dim=1024`, `model_dim=1024`
- `num_layers=6`, `num_heads=16`, `self_attn=true`
- 토큰 임베딩 테이블: `Embedding(32128, target_dim)`
- 구조: `in_proj -> TransformerBlock x N -> out_proj -> RMSNorm`

### 5-3) 텍스트 조건 스펙
기준 코드:
- `diffusion-pipe/models/cosmos_predict2.py`의 `_tokenize`, `get_call_text_encoder_fn`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/text_encoders/anima.py`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/model_base.py`의 `Anima.extra_conds`

핵심 규약:
- 학습(diffusion-pipe): Qwen 경로 토큰 + T5 토큰을 둘 다 생성하고 `LLMAdapter` 입력으로 결합
- 학습 토크나이즈 길이: `_tokenize(..., max_length=512)`
- ComfyUI 추론: `t5xxl_ids`, `t5xxl_weights`를 extra cond로 전달
- ComfyUI에서 cross-attn 길이가 512 미만이면 512로 pad
- Qwen tokenizer 특이값: pad token id `151643` 사용

### 5-4) VAE / latent 스펙
기준 코드:
- `diffusion-pipe/models/cosmos_predict2.py`의 `WanVAE`

핵심 규약:
- 잠재 채널 수는 16 기준
- 내부적으로 채널별 `mean/std` 정규화 스케일을 고정값으로 보유
- 파이프라인 텐서는 비디오 호환 shape(`B,C,T,H,W`)를 사용 (이미지는 보통 `T=1`)

## 6) 학습 하이퍼파라미터 레퍼런스 (Anima)

### 6-1) 모델 설정 키
기준 코드: `diffusion-pipe/models/cosmos_predict2.py`, `diffusion-pipe/docs/supported_models.md`

필수:
- `type='anima'`
- `transformer_path`
- `vae_path`
- `llm_path`
- `dtype`

선택(기본값 포함):
- `transformer_dtype` (기본 `dtype`)
- `llm_adapter_path` (없으면 transformer 내장 어댑터 키 탐지)
- `cache_text_embeddings=true`
- `text_encoder_nf4=false`
- `text_encoder_fp8=false`
- `timestep_sample_method='logit_normal'` (`uniform` 가능)
- `sigmoid_scale=1.0` (logit_normal일 때)
- `shift` (미지정 시 사용 안 함)
- `flux_shift=false` (`shift`가 없을 때만 사용)
- `multiscale_loss_weight` (미지정 시 비활성)

### 6-2) 학습률 분리 키
기준 코드: `diffusion-pipe/models/cosmos_predict2.py`의 `get_param_groups`

- `optimizer.lr`를 base로 사용
- 아래 값 미지정 시 base lr 상속:
  - `self_attn_lr`
  - `cross_attn_lr`
  - `mlp_lr`
  - `mod_lr`
  - `llm_adapter_lr`
- `llm_adapter_lr=0`이면 adapter 파라미터는 동결(학습 제외)

### 6-3) 손실/목표 함수
기준 코드: `diffusion-pipe/models/cosmos_predict2.py`의 `prepare_inputs`, `get_loss_fn`

- 입력 구성: `x_t=(1-t)x0+t*noise`
- 타깃: `noise-x0` (rectified-flow velocity 형태)
- 기본 loss: MSE
- `pseudo_huber_c`가 있으면 pseudo-Huber 사용

### 6-4) 문서에 명시된 Anima 권장사항
기준 코드: `diffusion-pipe/docs/supported_models.md`의 Anima 섹션

- 다른 모델보다 더 낮은 lr이 필요할 수 있음
- 소규모 데이터셋에서는 `llm_adapter_lr=0`부터 시작하는 전략 권장
- preview 기반 LoRA는 final 모델에서 재학습이 필요할 가능성이 큼

## 7) 추론 하이퍼파라미터 레퍼런스 (ComfyUI)

기준 코드:
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/supported_models.py`의 `Anima`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/model_base.py`의 `Anima.extra_conds`

모델 기본 추론 설정:
- `sampling_settings.multiplier = 1.0`
- `sampling_settings.shift = 3.0`
- `supported_inference_dtypes = [bfloat16, float16, float32]`

조건 처리 설정:
- `t5xxl_ids`가 있으면 `preprocess_text_embeds`(LLMAdapter 경유) 실행
- `t5xxl_weights`가 있으면 cross-attn에 가중치 적용
- cross-attn 최소 길이 512 보장(pad)

운영 팁:
- steps/CFG/sampler/seed/해상도는 워크플로우에서 제어되는 값이며, Anima 클래스 자체의 고정값은 아님
- 체크포인트 prefix는 `net.` 감지가 가능해야 하고, LoRA는 `diffusion_model.` prefix 규약을 맞추는 편이 안전

## 8) 자체 구현 청사진 (스크립트 의존 최소화)

### A. 최소 추론 엔진 (ComfyUI 호환 지향)
1. 체크포인트 로딩 시 prefix(`net.`, `model.diffusion_model.` 등) 정규화.
2. Anima DiT(`MiniTrainDIT`+`LLMAdapter`) 구성 후 weight 로드.
3. Qwen tokenizer/model로 `cross_attn` 생성.
4. T5 tokenizer로 `t5_ids`(필요 시 weight 포함) 생성.
5. `cross_attn = llm_adapter(cross_attn, t5_ids)` 적용 후 시퀀스 길이 규약 정렬.
6. Flow sampler(shift=3.0 기본값) + CFG로 latent 샘플링.
7. Wan 계열 VAE로 decode.

### B. 최소 학습 엔진 (diffusion-pipe 호환 지향)
1. 이미지/비디오 -> VAE latent 변환.
2. 캡션 -> Qwen 임베딩 + T5 IDs 생성.
3. `t` 샘플링(logit-normal 또는 uniform + shift).
4. `x_t=(1-t)x0+t*noise`, `target=noise-x0` 구성.
5. DiT forward(LLM Adapter 경유) 후 MSE/Huber loss 계산.
6. 필요 시 mask/multiscale loss 적용.
7. 파라미터 그룹 분리(`llm_adapter_lr`, self/cross_attn lr 등).
8. 저장 시 full model은 `net.`, LoRA는 `diffusion_model.` prefix 유지.

## 9) 실전 커스터마이징 포인트
- 기준 정렬:
  - 동작이 충돌하면 `diffusion-pipe`를 우선하고, 그다음 `ComfyUI` 추론 동작으로 맞춘다.
  - SD3 쪽은 아이디어 차용/검증 보조 지표로만 사용한다.
- `llm_adapter_lr`:
  - 작은 데이터셋/스타일 LoRA: 0 또는 매우 낮은 값부터 시작.
  - 새로운 개념 대량 주입: adapter 학습을 켜고 별도 lr 튜닝.
- timestep 분포:
  - `logit_normal + shift` 조합은 구조/디테일 밸런스를 크게 바꿈.
- 텍스트 길이:
  - tokenizer max length, cross-attn pad 규약(512) 불일치 시 shape 오류.
- 포맷 상호운용:
  - 추론 타깃이 ComfyUI라면 LoRA 키 prefix를 `diffusion_model.`로 통일.
- 모델 버전:
  - `supported_models.md`의 Anima 주석대로 preview 기반 LoRA는 final 모델에서 성능이 크게 흔들릴 수 있음.

## 10) 개발 체크리스트
- [ ] Qwen 임베딩 경로와 T5 ID 경로가 모두 forward에 들어가는가?
- [ ] LLM Adapter를 학습할지 동결할지 명시적으로 설정했는가?
- [ ] 학습 타깃 수식(`noise-latents` vs `latents`)이 샘플러/모델 출력 정의와 일치하는가?
- [ ] 체크포인트 prefix 변환 로직을 중앙화했는가?
- [ ] LoRA 저장 키가 ComfyUI에서 바로 읽히는가?

## 11) 구현 시 권장 검증 순서
1. 추론 parity 먼저 검증: 동일 프롬프트/시드에서 ComfyUI와 유사 출력이 나오는지 확인.
2. 학습 1 step 검증: loss 하강, gradient flow, adapter grad 유무 확인.
3. 저장/재로딩 검증: `net.` full model, `diffusion_model.` LoRA 각각 round-trip 테스트.
4. 마지막에 성능 튜닝: lr 분리, shift, tokenizer 길이, 캐시 전략 순서로 조정.

---
이 문서는 Anima 전용으로 작성되었고, 구현 우선순위는 `diffusion-pipe > ComfyUI > SD3 참고` 순서를 따른다.
