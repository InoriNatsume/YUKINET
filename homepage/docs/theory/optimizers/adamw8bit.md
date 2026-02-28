# AdamW8bit — 8-bit Quantized AdamW

> 분류: Adam 계열(양자화) · sd-scripts ✓ · diffusion-pipe ✓

## 핵심 아이디어

AdamW의 상태 텐서(`m_t`, `v_t`)를 FP32 대신 8bit 블록 양자화로 저장해
메모리를 절감합니다.

\[
m_t^{(8)}=Q_8(m_t),\qquad v_t^{(8)}=Q_8(v_t)
\]

\[
Q_8(x)=\operatorname{round}\!\left(\frac{x}{\max(|x_{\text{block}}|)}\cdot127\right)
\]

## 메모리 관점

\[
\text{FP32 AdamW state}\approx 8|\theta|\ \text{bytes}
\quad\rightarrow\quad
\text{8bit state}\approx 2|\theta|\ \text{bytes}
\]

즉 상태 메모리 기준 약 4배 절감이 가능합니다.

## Kahan 보정과의 관계

8bit 자체는 저장량을 줄이지만 누적 오차가 커질 수 있습니다.
그래서 일부 구현은 Kahan 보정(또는 유사 보상 텐서)을 함께 사용합니다.

\[
s_{t+1}=s_t + (x_{\text{exact}}-x_{\text{rounded}})
\]

## 코드 매핑

```python
# sd-scripts
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(params, lr=lr, betas=betas)

# diffusion-pipe
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(params, lr=lr, betas=betas)
```
