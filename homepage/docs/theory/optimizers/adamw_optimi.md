# AdamW (optimi) — Kahan Summation AdamW

> 분류: Adam 계열 · diffusion-pipe ✓ (권장 기본)

## 수학 정의

`optimi` 라이브러리의 AdamW. 표준 AdamW + **Kahan Summation**으로 BF16 정밀도 보정.

### Kahan Summation 보정

$$
\text{일반 BF16:}\quad \theta_{t+1} = \text{round}_\text{BF16}(\theta_t - \eta \cdot u_t) \quad\text{→ 반올림 오차 누적}
$$

$$
\text{Kahan 보정:}\quad \begin{aligned}
y &= (\theta_t - \eta \cdot u_t) + c_t \\
\theta_{t+1} &= \text{round}_\text{BF16}(y) \\
c_{t+1} &= y - \theta_{t+1} \quad\text{(보상 텐서에 오차 저장)}
\end{aligned}
$$

BF16은 가수부 7비트 → 큰 값에 작은 업데이트가 반올림으로 사라질 수 있음. Kahan이 이를 보정.

## 왜 diffusion-pipe에서 기본 권장인가?

> 대부분의 diffusion 훈련은 BF16 mixed precision을 사용합니다. 이때 파라미터 업데이트가 가수부 정밀도보다 작으면 "사라지는 업데이트" 문제가 발생합니다. 예: $\theta=1024.0$, 업데이트=$0.001$ → BF16에서 $1024+0.001=1024$ (변화 없음) Kahan Summation은 이 누실된 $0.001$을 보상 변수에 축적하여 나중에 반영합니다.

## 코드 매핑

```python
# diffusion-pipe: type = "adamw_optimi" (권장)
from optimi import AdamW
optimizer = AdamW(params, lr=lr, betas=betas)
```
