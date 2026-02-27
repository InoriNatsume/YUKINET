# AdamW8bit — 8-bit Quantized AdamW

> 분류: Adam 계열 (양자화) · sd-scripts ✓ · diffusion-pipe ✓

## 수학 정의

bitsandbytes (Dettmers et al.) 구현. AdamW의 모멘텀/분산을 블록 단위 8-bit 동적양자화.

$$
m_t^{(8)} = Q_8(m_t),\quad v_t^{(8)} = Q_8(v_t)
$$Q_8(x) = \text{round}\!\left(\frac{x}{\max(|x_\text{block}|)} \cdot 127\right),\quad \text{block size}=2048$$
업데이트 시 FP32로 디퀀타이즈 → 계산 → 재양자화.

## 메모리 절감
$$\text{FP32 AdamW}: 8|\theta|\text{ bytes} \quad\longrightarrow\quad \text{8-bit}: 2|\theta|\text{ bytes}\quad (\times4\text{ 절감})$$
1B 파라미터: 8GB → 2GB optimizer states

## 변형: AdamW8bitKahan (diffusion-pipe)
$$\text{Kahan Summation: }\; s_{t+1} = s_t + (x_\text{exact} - x_\text{rounded})$$
$$

BF16 파라미터에서 누적되는 반올림 오차를 보상 텐서 $s$에 축적. 장기 학습에서 정밀도 유지.

## 코드 매핑

```python
# sd-scripts: --optimizer_type AdamW8bit
from bitsandbytes.optim import AdamW8bit
optimizer = AdamW8bit(params, lr=lr, betas=betas)

# diffusion-pipe: type = "adamw8bit"
from bitsandbytes.optim import AdamW8bit

# diffusion-pipe Kahan 변형: type = "adamw8bitkahan"
from optimizers.adamw_8bit import AdamW8bitKahan
optimizer = AdamW8bitKahan(params, lr=lr, stabilize=True)
```
