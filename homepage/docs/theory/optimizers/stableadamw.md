# StableAdamW — 안정화된 AdamW

> 분류: AdamW 변형(안정성 강화) · diffusion-pipe ✓

## 핵심 아이디어

일반 AdamW 업데이트의 분모/스케일을 안정화해
큰 gradient 스파이크 구간에서 step 폭주를 줄입니다.

## 수식 스케치

RMS 스케일:

\[
\operatorname{RMS}(v_t)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}\hat{v}_{t,i}}
\]

안정화된 update 방향(대표형):

\[
u_t=
\frac{\hat{m}_t}{
\operatorname{RMS}(v_t)\cdot
\max\!\left(1,\frac{\|\hat{m}_t/\sqrt{\hat{v}_t}\|_\infty}{\operatorname{RMS}(v_t)}\right)}
\]

\[
\theta_{t+1}=\theta_t-\eta\,u_t-\eta\lambda\theta_t
\]

## 언제 고려할까

| 상황 | 이유 |
|---|---|
| 고해상도/고배치에서 loss spike | step 안정화 필요 |
| mixed precision에서 불안정 | scale 제어 이점 |
| 기본 AdamW가 자주 발산 | 보수적 대안 |

## 코드 매핑

```python
# diffusion-pipe
# type = "stableadamw"
```
