# SDXL — Stable Diffusion XL

> 백본: UNet (확장) · 확산: DDPM (ε/v) · 유형: Image · sd-scripts ✓ · diffusion-pipe ✓

## 아키텍처 특징

\[
\text{SDXL}=\underbrace{\text{UNet}_{2.6B}}_{\text{확장된 attention}}+\underbrace{\text{CLIP-L}+\text{OpenCLIP-G}}_{\text{이중 텍스트 인코더}}+\underbrace{\text{VAE}}_{\text{f=8}}
\]

| SD 1.x와 차이 | SDXL |
|---|---|
| 파라미터 수 | ~860M → 2.6B |
| 텍스트 인코더 | CLIP 1개 → CLIP-L + OpenCLIP-G 듀얼 |
| 해상도 | 512 → 1024 |
| 조건 | 텍스트만 → 텍스트 + crop_coords + original_size + target_size |
| Attention 레이어 | 4단계 → 3단계 (최하단 제거) |

## 훈련 특수 하이퍼파라미터

| 파라미터 | 수식/효과 | sd-scripts | diffusion-pipe |
|---|---|---|---|
| crop conditioning | $c_\text{crop}=(c_\text{top},c_\text{left})$ — 크롭 좌표 조건 | 자동 | 자동 |
| original_size | $c_\text{size}=(H_\text{orig},W_\text{orig})$ — 원본 크기 조건 | 자동 | 자동 |
| min_snr_gamma | $w(t)=\min(\text{SNR},\gamma)/\text{SNR}$ | --min_snr_gamma 5 | min_snr_gamma = 5 |
| adaptive_noise_scale | 이미지 평균 밝기 기반 노이즈 오프셋 | --adaptive_noise_scale | 미지원 |
| fused_backward_pass | backward+optim 결합 (Adafactor 전용) | --fused_backward_pass | 미지원 |

## diffusion-pipe에서의 SDXL

> diffusion-pipe는 SDXL에서 v-prediction과 debiased estimation loss를 추가로 지원합니다. 또한 min_snr_gamma가 v-pred에서도 올바르게 동작하도록 구현되어 있습니다.

## 코드 매핑

```python
# sd-scripts
python sdxl_train.py --pretrained_model_name_or_path model.safetensors \
    --resolution 1024 --train_batch_size 1 \
    --learning_rate 2e-6 --optimizer_type AdamW

python sdxl_train_network.py --network_module networks.lora \
    --network_dim 16 --network_alpha 8 \
    --learning_rate 1e-4

# diffusion-pipe (config.toml)
[model]
type = "sdxl"
dtype = "bfloat16"
v_pred = false
min_snr_gamma = 5
```
