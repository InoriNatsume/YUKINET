# AdamW8bitKahan — 8-bit + Kahan + StableAdamW

> 분류: Adam 계열 (양자화+보정) · diffusion-pipe ✓

## 수학 정의

bitsandbytes의 8-bit 양자화 + optimi의 Kahan Summation + StableAdamW 모드를 결합한 하이브리드.

$$
\text{8-bit 양자화: }m_t^{(8)}=Q_8(m_t),\; v_t^{(8)}=Q_8(v_t) \quad\text{→ 4x 메모리 절감}
$$\text{Kahan 보정: }c_{t+1}=y-\text{round}_\text{BF16}(y) \quad\text{→ BF16 정밀도 보상}$$
\text{StableAdamW: }\text{stabilize}=\text{True}\quad\text{→ RMS 정규화 안정화}
$$

## 3중 최적화의 의미

> 메모리 (8-bit): 옵티마이저 상태 VRAM 4x 절감 정밀도 (Kahan): BF16 파라미터의 누적 반올림 오차 보정 안정성 (StableAdamW): 큰 기울기에 대한 업데이트 폭발 방지 세 기법이 서로 보완적: 8-bit는 메모리를, Kahan은 양자화 오차를, Stable은 수치 안정성을 처리.

## 코드 매핑

```python
# diffusion-pipe: type = "adamw8bitkahan"
from optimizers.adamw_8bit import AdamW8bitKahan
optimizer = AdamW8bitKahan(
    params, lr=lr, betas=betas,
    stabilize=True  # StableAdamW 모드 활성화
)
```
