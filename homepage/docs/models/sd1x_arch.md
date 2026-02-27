# Stable Diffusion 1.x / 2.x — UNet 기반 Latent Diffusion

> 백본: UNet · 확산: DDPM (이산) · 유형: Image · sd-scripts ✓

## 아키텍처 구조

$$
x_0 \xrightarrow{\text{VAE Encoder}} z_0 \xrightarrow{q(z_t|z_0)} z_t \xrightarrow{\text{UNet}_\theta(z_t,t,c)} \hat\epsilon \xrightarrow{\text{VAE Decoder}} \hat x_0
$$

### UNet 구성

| 블록 | 구성 | LoRA 적용 |
|---|---|---|
| Down blocks | ResBlock + CrossAttention × 4단계 | CrossAttention의 Q,K,V,O 프로젝션 |
| Mid block | ResBlock + CrossAttention | 동일 |
| Up blocks | ResBlock + CrossAttention × 4단계 | 동일 |
| Text Encoder | CLIP (SD1: ViT-L/14, SD2: OpenCLIP ViT-H/14) | 선택적 적용 |

## 훈련 목적함수

> ### ε-prediction (기본)       $$ $$\mathcal{L}=\mathbb{E}_{z_0,\epsilon,t}\left[\|\epsilon_\theta(z_t,t,c)-\epsilon\|^2\right]$$ $$      SD 1.x 기본. 노이즈를 직접 예측.

> ### v-prediction (SD 2.x 선택)       $$ $$v_t=\sqrt{\bar\alpha_t}\epsilon-\sqrt{1-\bar\alpha_t}z_0$$ $$       $$ $$\mathcal{L}=\mathbb{E}\left[\|v_\theta(z_t,t,c)-v_t\|^2\right]$$ $$      --v_parameterization 플래그로 활성화.

## 훈련 모드 (sd-scripts)

| 모드 | 스크립트 | 학습 대상 | 권장 LR |
|---|---|---|---|
| DreamBooth | train_db.py | UNet 전체 (+정규화 이미지) | 1e-6 |
| Fine-Tune | fine_tune.py | UNet 전체 | 2e-6 |
| LoRA | train_network.py | LoRA A,B 행렬 | 1e-4 |
| Textual Inversion | train_textual_inversion.py | 토큰 임베딩 | 5e-3 |
| ControlNet | train_controlnet.py | ControlNet 인코더 | 1e-5 |

## 핵심 하이퍼파라미터

| 파라미터 | 영향 | 권장 |
|---|---|---|
| resolution | 학습 해상도 | SD1: 512, SD2: 768 |
| clip_skip | CLIP 레이어 깊이 | SD1: 2 (anime), 1 (photo) |
| noise_offset | 밝기 학습 개선 | 0.05~0.1 |
| min_snr_gamma | 손실 가중치 평활화 | 5 |
