# FLUX.1 — DiT 기반 Flow Matching

> 백본: DiT (Dual-stream → Single-stream) · 확산: Flow Matching (연속) · 파라미터: ~12B · 유형: Image · sd-scripts ✓ · diffusion-pipe ✓ · DiffSynth-Studio ✓

## 아키텍처 구조

$$
z_0 \xrightarrow{x_t=(1-t)z_0+t\epsilon} x_t \xrightarrow{\text{DiT}_\theta(x_t,t,c_\text{CLIP},c_\text{T5})} v_\theta \approx \epsilon-z_0
$$

### DiT 블록 구조

| 구간 | 블록 수 | 구조 | 특징 |
|---|---|---|---|
| Dual-stream | 19 | 이미지 스트림 ↔ 텍스트 스트림 병렬 + cross-attention | 이미지와 텍스트가 독립 어텐션 후 교차 |
| Single-stream | 38 | 이미지+텍스트 토큰 concat → self-attention | 통합 어텐션으로 깊은 융합 |

### 텍스트 인코더

> CLIP-L: 전역 의미 벡터 (pooled)조건 임베딩 $c_\text{CLIP}\in\mathbb{R}^{768}$

> T5-XXL: 상세 토큰별 임베딩$c_\text{T5}\in\mathbb{R}^{L\times 4096}$, 최대 512 토큰

## Flow Matching 훈련 목적함수

$$
\mathcal{L}_\text{FM}=\mathbb{E}_{z_0,\epsilon,t}\left[w(t)\cdot\|v_\theta(x_t,t,c)-(\epsilon-z_0)\|^2\right]
$$

$$
x_t=(1-t)\,z_0+t\,\epsilon,\quad t\in[0,1]
$$

## Timestep 샘플링 전략 비교

| 전략 | 코드 플래그 | 수식 | 코드베이스 |
|---|---|---|---|
| Sigmoid | timestep_sampling=sigmoid | $t=\sigma(s\cdot z)$, $z\sim\mathcal{N}$ | sd-scripts |
| Flux Shift | timestep_sampling=flux_shift | $\mu=0.5+\frac{\text{area}}{256^2}\cdot 1.15$$t=\sigma(\mu+s\cdot z)$ | sd-scripts |
| Logit-Normal | timestep_sampling=logit_normal | $t=\sigma(\mu+s\cdot z)$ | diffusion-pipe, DiffSynth |
| Shift | discrete_flow_shift=3.0 | $t'=\frac{s\cdot t}{1+(s-1)t}$ | sd-scripts |

## 코드베이스별 특수 기능

> ### sd-scripts             Full FT + LoRA + ControlNet       --blocks_to_swap (블록 CPU 오프로드)       --fp8_base (FP8 양자화)       --model_prediction_type: raw / additive / sigma_scaled       Chroma (--model_type chroma) 공유

> ### diffusion-pipe             Full FT + LoRA       Pipeline parallelism       Block swapping (LoRA)       FP8 transformer dtype       fuse_adapters: 기존 LoRA 병합 후 새 LoRA       Flux Kontext 지원

> ### DiffSynth-Studio             LoRA (PEFT)       SFT + Distillation       2단계 학습 (data_process + train)       Civitai 포맷 변환 지원       FP8 모델 로딩       preset_lora_path (기존 LoRA 퓨전)

## LoRA 타겟 모듈

```python
# sd-scripts (networks/lora_flux.py)
# Dual-stream blocks: img_attn.qkv, img_attn.proj, img_mlp.0, img_mlp.2
#                     txt_attn.qkv, txt_attn.proj, txt_mlp.0, txt_mlp.2
# Single-stream blocks: linear1, linear2, modulation.lin

# DiffSynth-Studio 자동 감지 (min weight dim >= 512):
# a_to_qkv, b_to_qkv, ff_a.0, ff_a.2, ff_b.0, ff_b.2,
# a_to_out, b_to_out, proj_out, norm.linear, ...

# diffusion-pipe: PEFT auto-detect all nn.Linear in target blocks
```

## 권장 설정

| 설정 | LoRA | Full FT |
|---|---|---|
| Learning Rate | 1e-4 ~ 5e-4 | 1e-6 ~ 5e-6 |
| LoRA Rank | 16~64 | — |
| discrete_flow_shift | 3.0 (dev), 1.0 (schnell) | 동일 |
| timestep_sampling | sigmoid (scale=1.0) | 동일 |
| Mixed Precision | BF16 | BF16 |
| Gradient Checkpointing | 권장 | 필수 |
