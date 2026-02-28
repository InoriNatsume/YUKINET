# LoRA 수학

## 핵심 아이디어

원본 가중치 $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$를 직접 수정하지 않고, **저랭크 행렬 쌍**으로 변화량을 근사:

\[
W' = W + \Delta W = W + \frac{\alpha}{r} B A
\]

- $A \in \mathbb{R}^{r \times d_\text{in}}$ — down-projection (가우시안 초기화)
- $B \in \mathbb{R}^{d_\text{out} \times r}$ — up-projection (**0 초기화**)
- $r \ll \min(d_\text{in}, d_\text{out})$ — 랭크

## 왜 저랭크인가?

!!! info "파라미터 절감"
    $d = 4096, r = 16$일 때:
    
    - Full: $d^2 = 16,777,216$개
    - LoRA: $2 \times d \times r = 131,072$개 (**0.78%**)

## 핵심 파라미터

| 기호 | 코드 | 의미 | 전형적 값 |
|---|---|---|---|
| $r$ | `--network_dim` / `lora_rank` | 랭크 | 4~128 |
| $\alpha$ | `--network_alpha` / `lora_alpha` | 스케일링 계수 | $r$의 절반~동일 |
| $\alpha / r$ | — | 실질적 학습률 스케일 | 0.5~1.0 |

!!! warning "α/r 과 학습률의 관계"
    $\Delta W = \frac{\alpha}{r} B A$이므로, `α`를 올리는 것과 학습률을 올리는 것은 같은 효과.
    보통 $\alpha = r$로 두고 학습률로 조절하거나, $\alpha = r/2$로 두고 보수적으로 훈련.

## 초기화 전략

- $B = 0$ → $\Delta W = 0$에서 출발 (원본 보존)
- $A \sim \mathcal{N}(0, \sigma^2)$ → 방향만 랜덤, 크기는 $B=0$이라 무관

학습이 진행되면서 $B$가 0에서 벗어나기 시작하면 $\Delta W$가 의미 있는 변화를 만듦.

## 구현 위치

| 도구 | 구현 | 상세 |
|---|---|---|
| kohya (sd-scripts) | `networks/lora.py` | [→ 도구/kohya](../../tools/kohya/index.md) |
| HuggingFace | `peft` 라이브러리 | [→ 도구/HuggingFace](../../tools/huggingface/index.md) |
| DiffSynth | 자체 구현 | [→ 도구/DiffSynth](../../tools/diffsynth/index.md) |
