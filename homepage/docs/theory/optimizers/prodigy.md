# Prodigy — Adaptive Learning Rate Estimation

> 분류: 적응형 (D-Adaptation) · sd-scripts ✓ · diffusion-pipe ✓

## 수학 정의

Mishchenko & Defazio (2023). 학습률의 "최적 크기" $d^*$를 online으로 추정.

$$
d_t = d_{t-1} + \eta_\text{lr} \cdot \frac{|\langle g_t, s_t \rangle|}{d_{t-1} \|s_t\|^2}
$$\text{effective LR} = d_t \cdot \eta_\text{lr}$$
$$

여기서 $s_t$는 누적 그래디언트 합 $\sum_{i \leq t} g_i$의 가중 추정. $d_t$는 **distance to solution** 추정치.

## 핵심 아이디어

> 전통적 학습률 튜닝: "lr=1e-4가 좋은가 5e-5가 좋은가?" 를 수동 실험. Prodigy: $\eta_\text{lr}\approx 1.0$으로 두면, $d_t$가 자동으로 최적 학습률 스케일을 추정합니다.  $$ $$d_0 \approx 0 \;\longrightarrow\; d_T \approx d^* \approx \|\theta^*-\theta_0\|$$ $$  $d^*$는 초기 파라미터에서 최적 파라미터까지의 "거리". 이를 그래디언트 내적으로 추정.

## 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| lr | 1.0 | 스케일링 계수. 1.0 권장 |
| d_coef | 1.0 | $d$ 초기화 계수 |
| growth_rate | ∞ | $d_t$ 최대 성장 비율 (안정화용) |
| betas | (0.9, 0.999) | Adam 모멘텀 파라미터 |
| weight_decay | 0 | Decoupled weight decay |

## 장점 vs 단점

> ### 장점             학습률 수동 탐색 불필요       다양한 모델/데이터에 자동 적응       LoRA 훈련에서 특히 편리

> ### 단점             $d_t$ 수렴까지 불안정할 수 있음       추가 상태 변수 → 메모리 약간 증가       학습 초기 oscillation 가능

## 코드 매핑

```python
# sd-scripts: --optimizer_type Prodigy --learning_rate 1.0
from prodigyopt import Prodigy
optimizer = Prodigy(params, lr=1.0, d_coef=1.0)

# diffusion-pipe: type = "Prodigy" (pytorch-optimizer 라이브러리)
from pytorch_optimizer import Prodigy
optimizer = Prodigy(params, lr=1.0)
```
