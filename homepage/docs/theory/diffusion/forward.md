# 정방향 확산 (Forward Process)

!!! tip "모티베이션"
    데이터 분포 $p_\text{data}(x)$를 다루기 쉬운 $\mathcal{N}(0,I)$로 변환하는 매핑을 학습하는 것이 목표.
    점진적으로 노이즈를 추가하면 각 스텝의 역변환이 가우시안으로 근사 가능해진다 (Feller, 1949).

## DDPM 이산 정방향

$T$개의 이산 스텝 마르코프 체인:

$$
q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)
$$

**핵심 성질** — 임의의 $t$에서 직접 샘플 가능:

$$
q(x_t|x_0) = \mathcal{N}\!\left(x_t;\;\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\right)
$$

$$
\boxed{x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)}
$$

여기서 $\bar\alpha_t = \prod_{s=1}^t (1-\beta_s)$는 누적 신호 보존율.

```python
# sd-scripts: library/train_util.py
alphas = 1.0 - betas                          # α_t = 1 - β_t
alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t = Π(α_s)

# 임의 timestep에서 noisy sample 생성
sqrt_alpha = alphas_cumprod[t] ** 0.5
sqrt_one_minus_alpha = (1 - alphas_cumprod[t]) ** 0.5
x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
```

## 연속 시간 SDE

이산 과정을 연속으로 확장하면 (Song et al., 2021):

$$
dx = f(x,t)\,dt + g(t)\,dW_t
$$

DDPM의 VP-SDE:

$$
dx = -\frac{\beta(t)}{2}x\,dt + \sqrt{\beta(t)}\,dW_t
$$

- $f(x,t) = -\frac{\beta(t)}{2}x$ — drift (신호 감쇠)
- $g(t) = \sqrt{\beta(t)}$ — diffusion (노이즈 주입)

## Flow Matching 정방향

!!! info "왜 직선 경로인가?"
    DDPM의 SDE 경로는 곡선이라 많은 추론 스텝이 필요.
    Flow Matching은 **직선(optimal transport) 경로**를 사용하여 적은 NFE로 고품질 샘플 생성.

$$
\boxed{x_t = (1-t)\,x_0 + t\,\epsilon, \quad t \in [0,1]}
$$

$$
v_t = \frac{dx_t}{dt} = \epsilon - x_0 \quad \text{(상수 velocity — 경로가 직선)}
$$

```python
# sd-scripts: flux_train.py
noisy_x = (1.0 - t) * x_0 + t * noise    # 직선 보간
target = noise - x_0                       # velocity target (상수!)
```

## DDPM vs Flow Matching 비교

| 특성 | DDPM | Flow Matching |
|---|---|---|
| **수학적 프레임워크** | 이산 마르코프 체인 / VP-SDE | 연속 ODE / Optimal Transport |
| **경로 형태** | 곡선 (variance preserving) | 직선 (constant velocity) |
| **보간 공식** | $\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ | $(1-t)x_0 + t\epsilon$ |
| **필요 추론 스텝** | 20~50+ (DDPM/DDIM) | 1~4 (Euler) |
| **사용 모델** | SD 1.x, SDXL | Flux, SD3, Wan |

## SNR (Signal-to-Noise Ratio)

$$
\text{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t} = \frac{\text{signal power}}{\text{noise power}}
$$

!!! warning "왜 SNR이 중요한가?"
    단순 MSE 손실을 사용하면 각 timestep 기여도가 암묵적으로 SNR에 의존.
    높은 SNR($t \approx 0$)에서 손실이 과도해지면 세밀한 디테일에만 집중하고 구조적 학습을 소홀히 함.
    → [Min-SNR-γ](../training/objectives.md) 등의 가중치 기법이 이 불균형을 보정.

Flow Matching에서는:

$$
\text{SNR}_\text{FM}(t) = \left(\frac{1-t}{t}\right)^2
$$

$t=0.5$에서 $\text{SNR}=1$ (균형점). 이 대칭적 구조가 logit-normal 샘플링의 이론적 배경.
