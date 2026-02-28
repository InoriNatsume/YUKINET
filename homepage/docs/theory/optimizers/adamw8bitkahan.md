# AdamW8bitKahan — 8bit + Kahan 보정

> 분류: 저메모리 + 누적오차 보정 · diffusion-pipe ✓

## 핵심 아이디어

1. 상태 텐서는 8bit로 저장(메모리 절감)
2. 파라미터 업데이트 누적 오차는 Kahan 보상 텐서로 보정

## 수식 스케치

일반 업데이트:

\[
\theta_{t+1}=\operatorname{round}(\theta_t-\eta u_t)
\]

Kahan 보정 포함:

\[
y=(\theta_t-\eta u_t)+c_t,\quad
\theta_{t+1}=\operatorname{round}(y),\quad
c_{t+1}=y-\theta_{t+1}
\]

여기서 `c_t`는 누적 반올림 오차를 저장하는 보상 상태입니다.

## 언제 유리한가

| 상황 | 효과 |
|---|---|
| VRAM이 매우 제한적 | 8bit state로 메모리 절감 |
| 긴 학습(step 많음) | 보상 텐서로 누적 오차 완화 |
| BF16/혼합정밀 | 수치 안정성 보조 |

## 코드 매핑

```python
# diffusion-pipe
from optimizers.adamw_8bit import AdamW8bitKahan
optimizer = AdamW8bitKahan(params, lr=lr, stabilize=True)
```
