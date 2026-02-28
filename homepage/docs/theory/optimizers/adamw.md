# AdamW — Decoupled Weight Decay Adam

> 분류: Adam 계열 · sd-scripts ✓ · diffusion-pipe ✓ · DiffSynth-Studio ✓

## 수학 정의

Loshchilov & Hutter (2019). Weight decay를 gradient update에서 분리한 Adam 변형.

\[
\begin{aligned}
g_t &= \nabla_\theta \mathcal{L}(\theta_t) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad &\text{(1st moment — momentum)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad &\text{(2nd moment — adaptive LR)} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \quad &\text{(bias correction)} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \quad &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\theta_t\right) \quad &\text{(decoupled WD)}
\end{aligned}
\]

## 파라미터 상세

| 파라미터 | 기본값 | 수식 대응 | 튜닝 가이드 |
|---|---|---|---|
| lr | 2e-6 (sd-scripts)1e-4 (diffusion-pipe, DiffSynth) | $\eta$ | LoRA: 1e-4, Full FT: 1e-6~5e-6 |
| betas | (0.9, 0.999) | $(\beta_1, \beta_2)$ | $\beta_1$↑: 모멘텀 관성 증가. $\beta_2$↑: LR 적응 지연 |
| eps | 1e-8 | $\epsilon$ | 수치 안정성. 보통 변경 불필요 |
| weight_decay | 0.01 | $\lambda$ | 0.01~0.1. 과적합 방지 |

## 메모리 사용량

\[
\text{Optimizer States}=\underbrace{4|\theta|}_{\text{m (FP32)}}+\underbrace{4|\theta|}_{\text{v (FP32)}}=8|\theta|\text{ bytes}
\]

예: 1B 파라미터 → 8GB optimizer states

## 변형

| 변형 | 차이점 | 코드베이스 |
|---|---|---|
| AdamW8bit | m, v를 8-bit 동적양자화. 메모리 ~2N | sd-scripts, diffusion-pipe |
| PagedAdamW | 가상 메모리 페이징으로 OOM 방지 | sd-scripts (bitsandbytes) |
| AdamW (optimi) | Kahan Summation → BF16 누적 오차 보정 | diffusion-pipe |
| StableAdamW | 분산 안정화 | diffusion-pipe |
| AdamW8bitKahan | 8-bit + Kahan + 선택적 StableAdamW 모드 | diffusion-pipe |

## 코드매핑

```python
# sd-scripts: library/train_util.py
optimizer = torch.optim.AdamW(
    params, lr=args.learning_rate,
    betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
)

# diffusion-pipe: train.py → create_optimizer()
if optimizer_type == 'adamw':
    optimizer = torch.optim.AdamW(params, **optimizer_kwargs)

# DiffSynth-Studio: diffsynth/diffusion/runner.py
optimizer = torch.optim.AdamW(
    trainable_params, lr=learning_rate, weight_decay=weight_decay
)
```

## 직관

> AdamW는 모든 diffusion 훈련의 기본 선택입니다. 1st moment ($m$)는 SGD의 모멘텀에 해당하고, 2nd moment ($v$)는 파라미터별 학습률을 적응적으로 조절합니다. Decoupled WD의 의미: 원래 Adam에서는 L2 정규화가 그래디언트에 섞여 $v$에 영향을 줬지만, AdamW는 WD를 분리하여 적응적 학습률을 왜곡하지 않습니다. $\beta_2=0.999$는 약 1000 step의 이동평균을 의미하므로, "최근 1000 step의 기울기 크기 평균"으로 학습률을 조절한다고 생각하면 됩니다.
