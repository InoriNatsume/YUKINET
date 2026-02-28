# Wan 2.1 / 2.2 — 비디오 확산 DiT

> 백본: 3D-DiT (Video) · 확산: Flow Matching · 유형: Video (T2V / I2V) · sd-scripts ✗ · diffusion-pipe ✓ · DiffSynth-Studio ✓

## 아키텍처 개요

\[
\text{Video }(B,C,F,H,W)\xrightarrow{\text{3D-VAE}}\text{Latent }(B,C_z,F',H',W')\xrightarrow{\text{3D-DiT}_\theta}v_\theta
\]

### 3D-DiT 블록 구조

Wan 모델은 시공간(Spatio-Temporal) 어텐션을 위한 3D Transformer 아키텍처를 사용합니다.

| 구성 요소 | 설명 |
|---|---|
| 3D VAE | 시간축 4× + 공간축 8× 압축. 비디오 프레임을 연속 잠재 공간으로 인코딩 |
| Spatial Attention | 프레임 내부의 H×W 공간 관계 처리 |
| Temporal Attention | 프레임 간 시간적 일관성 유지 |
| Cross Attention | 텍스트 조건 임베딩 주입 |
| Text Encoder | T5-XXL (또는 UMT5-XXL) |
| Image Encoder (I2V) | CLIP-ViT (Image-to-Video 조건용) |

## Flow Matching 훈련

\[
\mathcal{L}=\mathbb{E}_{z_0,\epsilon,t}\left[\|v_\theta(x_t,t,c)-(\epsilon-z_0)\|^2\right]
\]

\[
x_t=(1-t)z_0+t\epsilon
\]

### Timestep 분포

DiffSynth-Studio는 **bell-shaped** 분포를 사용하여 중간 타임스텝에 더 많은 가중치를 부여합니다:

\[
p(t)\propto\frac{1}{t(1-t)}\cdot\exp\!\left(-\frac{(\text{logit}(t)-\mu)^2}{2s^2}\right)
\]

## 지원 변형 모델

| 모델 | 크기 | 유형 | diffusion-pipe | DiffSynth |
|---|---|---|---|---|
| Wan 2.1 T2V | 14B | Text-to-Video | ✓ | ✓ |
| Wan 2.1 I2V | 14B | Image-to-Video | ✓ | ✓ |
| Wan 2.1 1.3B | 1.3B | T2V (small) | ✓ | — |
| Wan 2.1 FLF2V | 14B | First-Last-Frame-to-Video | ✓ | — |
| Wan 2.2 | 14B | 차세대 버전 | ✓ | ✓ |

## 비디오 훈련 특수 하이퍼파라미터

| 파라미터 | 설명 | 코드베이스 |
|---|---|---|
| num_frames | 훈련 프레임 수 (보통 16~81) | Both |
| frame_step | 원본 비디오에서 프레임 추출 간격 | diffusion-pipe |
| frame_buckets | 프레임 수별 버킷팅 (예: [1,33,65]) | diffusion-pipe |
| resolutions | 해상도 버킷 (멀티 해상도 훈련) | diffusion-pipe |
| block_swap_quantile | VRAM 절약을 위한 블록 스와핑 비율 | diffusion-pipe |
| gradient_checkpointing | 필수 (비디오 메모리 요구량) | Both |

## 코드 매핑

> ### diffusion-pipe       ```python # configs: examples/wan/ # TOML 설정 예시: [model] type = "wan"         # 또는 "wan_i2v", "wan_flf2v" transformer_path = "..." vae_path = "..." text_encoder_path = "..."  [dataset] frame_buckets = [1, 33, 65] resolutions = [[480, 832], [832, 480]]  [optimizer] type = "adamw_optimi" lr = 2e-5  # 메모리 최적화 block_swap_quantile = 0.25   # GPU VRAM 24GB 대응 pipeline_stages = 1          # multi-GPU 파이프라인 ```

> ### DiffSynth-Studio       ```python # examples/wanvideo/ # 2단계 학습: # 1) data_process: 라벨링 + 전처리 # 2) train: LoRA 학습  # 핵심 설정 dataset:   video_folder: "data/videos"   text_folder: "data/labels"   num_frames: 81  training:   learning_rate: 1e-4   lora_rank: 16   use_gradient_checkpointing: true   training_steps: 10000  # 지원 Loss: # - FlowMatchSFTLoss (기본 SFT) # - DirectDistillLoss (증류) # - TrajectoryImitationLoss (궤적 모방) ```

## 비디오 훈련 VRAM 추정

| 설정 | LoRA r=16 | Full FT |
|---|---|---|
| 1 frame (이미지 모드) | ~12 GB | ~48 GB |
| 33 frames (480p) | ~24 GB | ~80 GB+ |
| 65 frames (480p) | ~40 GB | Multi-GPU 필수 |
| + Block Swap (0.25) | —30~40% | —30~40% |

1 frame으로 훈련하면 이미지 LoRA처럼 작동합니다 (비디오 모델의 정지 이미지 학습).

## DiffSynth 특수 Loss 함수

### FlowMatchSFTLoss (기본)

\[
\mathcal{L}_\text{SFT}=\|v_\theta-(\epsilon-z_0)\|^2
\]

### DirectDistillLoss (증류)

\[
\mathcal{L}_\text{distill}=\|v_\theta^{\text{student}}-v_{\theta'}^{\text{teacher}}\|^2
\]

Teacher 모델의 출력을 직접 모방하여 빠른 수렴 달성

### TrajectoryImitationLoss (궤적 모방)

\[
\mathcal{L}_\text{traj}=\sum_{i}\|x_{t_i}^{\text{student}}-x_{t_i}^{\text{teacher}}\|^2
\]

ODE 궤적 전체를 모방하여 더 안정적인 증류

## 권장 설정

| 설정 | LoRA (추천) |
|---|---|
| Learning Rate | 1e-4 ~ 2e-4 |
| LoRA Rank | 16~32 |
| Frames | 33 (시작), 점진 증가 |
| Optimizer | AdamW / AdamW-optimi |
| Mixed Precision | BF16 |
| Gradient Checkpointing | 필수 |
| Block Swap | VRAM

