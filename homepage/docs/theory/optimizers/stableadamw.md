# StableAdamW — Stabilized AdamW

> 분류: Adam 계열 (안정화) · diffusion-pipe ✓

## 수학 정의

Wortsman et al. AdamW의 adaptive learning rate를 RMS 정규화하여 안정화.

$$
\text{RMS}(v_t) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} \hat{v}_{t,i}}
$$u_t = \frac{\hat{m}_t}{\text{RMS}(v_t) \cdot \max\!\left(1, \frac{\|\hat{m}_t / \sqrt{\hat{v}_t}\|_\infty}{\text{RMS}(v_t)}\right)}$$
$$

큰 기울기가 들어와도 업데이트 크기가 폭발하지 않도록 제한.

## 언제 사용?

> 대규모 모델에서 학습 초기 불안정 발생 시   기울기 크기의 분산이 클 때 (다양한 해상도, 비디오 등)   BF16 + Kahan이 활성화된 optimi 구현 사용

## 코드 매핑

```python
# diffusion-pipe: type = "stableadamw"
from optimi import StableAdamW
optimizer = StableAdamW(params, lr=lr, betas=betas)
```
