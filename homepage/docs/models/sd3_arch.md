# Stable Diffusion 3 / 3.5 — MMDiT

> 백본: MMDiT (Multi-Modal DiT) · 확산: Rectified Flow · 파라미터: 2B / 8B · 유형: Image · sd-scripts ✓ · diffusion-pipe ✓ · DiffSynth-Studio ✗

## MMDiT 아키텍처

\[
z_0 \xrightarrow{x_t=(1-\sigma_t)z_0+\sigma_t\epsilon} x_t \xrightarrow{\text{MMDiT}_\theta} v_\theta
\]

### 핵심 혁신: Multi-Modal Attention

이미지 토큰과 텍스트 토큰이 **동일 어텐션 공간**에서 상호작용. 각 스트림은 독립적 Linear projection → 합산된 QKV로 joint attention → 각 스트림이 독립적 출력 projection.

\[
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{[Q_\text{img};\,Q_\text{txt}]\cdot[K_\text{img};\,K_\text{txt}]^\top}{\sqrt{d_\text{head}}}\right)\cdot[V_\text{img};\,V_\text{txt}]
\]

### Triple Text Encoder

| 인코더 | 출력 차원 | 역할 |
|---|---|---|
| CLIP-L | $768$ | Pooled + token sequence |
| CLIP-G (OpenCLIP) | $1280$ | Pooled + token sequence |
| T5-XXL | $4096$ | Token sequence only |

SD3.5 Medium은 CLIP-L + CLIP-G만 사용 (T5 제외). `--clip_l`, `--clip_g`, `--t5xxl` 플래그로 개별 제어.

## Rectified Flow 훈련

\[
\mathcal{L}_\text{RF}=\mathbb{E}_{z_0,\epsilon,t}\left[\|v_\theta(x_t,t,c)-(\epsilon-z_0)\|^2\right]
\]

### Timestep / 시그마 변환

\[
\sigma(t)=t,\quad x_t=(1-\sigma)z_0+\sigma\epsilon \quad\text{(linear interpolation)}
\]

SD3는 **shift=3.0**을 기본 사용: $\sigma'=\frac{s\cdot\sigma}{1+(s-1)\sigma}$

## 훈련 모드

| 모드 | 설명 | sd-scripts | diffusion-pipe |
|---|---|---|---|
| Full Fine-tune | MMDiT 전체 파라미터 업데이트 | sd3_train.py | ✓ |
| LoRA | Low-Rank Adaptation 삽입 | sd3_train_network.py | ✓ |
| ControlNet | 조건 분기 학습 | 지원 예정 | — |

## sd-scripts 전용 옵션

```python
# SD3 특수 파라미터
--model_prediction_type=raw        # v_prediction 직접 사용
--timestep_sampling=sigma          # sigma-based 스케줄
--discrete_flow_shift=3.0          # default shift
--weighting_scheme=logit_normal    # 가중치 방식 (none/sigma_sqrt/mode/cosmap/logit_normal)

# 메모리 절약
--learning_rate_te1=0  --learning_rate_te2=0  # TE 동결
--fp8_base                                     # FP8 양자화
--blocks_to_swap=5                             # CPU offload
```

### Weighting Schemes 비교

| 방식 | 수식 | 특징 |
|---|---|---|
| none | $w(t)=1$ | 균일 가중 |
| sigma_sqrt | $w(\sigma)=\sqrt{\sigma}$ | 고노이즈 강조 |
| mode | $w(t)=\frac{1}{\pi\sigma_n}\cdot\frac{1}{e^{(t-\mu)/\sigma_n}+e^{-(t-\mu)/\sigma_n}}$ | logit-normal 모드 중심 |
| cosmap | $w(t)=1-\cos^2\!\left(\frac{\pi t}{2}\right)$ | 중앙부 강조 |
| logit_normal | $w(t)\propto\frac{1}{t(1-t)}\cdot\exp\!\left(-\frac{(\text{logit}(t)-\mu)^2}{2s^2}\right)$ | 중앙 집중, 양 끝 억제 |

## LoRA 타겟 모듈

```python
# sd-scripts: networks/lora_sd3.py
# 대상 레이어:
# - context_block.attn.{qkv, proj}     # 텍스트 스트림
# - x_block.attn.{qkv, proj}           # 이미지 스트림
# - context_block.mlp.{fc1, fc2}
# - x_block.mlp.{fc1, fc2}
# - final_layer.*

# 제어 파라미터:
# --network_train_unet_only   (기본: DiT 만 학습)
# --network_train_text_encoder (TE도 학습에 포함)
```

## 권장 설정

| 설정 | LoRA | Full FT |
|---|---|---|
| Learning Rate | 1e-4 ~ 3e-4 | 5e-7 ~ 5e-6 |
| LoRA Rank | 16~64 | — |
| timestep_sampling | sigma (default) | 동일 |
| discrete_flow_shift | 3.0 | 동일 |
| weighting_scheme | logit_normal | logit_normal |
| Mixed Precision | BF16 | BF16 |
