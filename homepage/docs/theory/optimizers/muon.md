# Muon / GenericOptim — Newton-Schulz 직교화 옵티마이저

> 분류: Advanced · diffusion-pipe ✓

## 수학 정의

diffusion-pipe의 `GenericOptim`은 Muon, AdaMuon, NorMuon, Polar Express 등 다양한 직교화 기반 옵티마이저를 통합 구현합니다.

### Newton-Schulz 반복 (핵심)

$$
X_0 = G / \|G\|_F
$$X_{k+1} = \frac{15}{8}X_k - \frac{5}{4}X_k^3 + \frac{3}{8}X_k^5$$
이 반복은 그래디언트 행렬 $G$의 **극 분해 (polar decomposition)**의 직교 인자 $U$로 수렴합니다:
$$G = US \;\Rightarrow\; X_\infty \to U$$
$U$는 "방향만" 보존하고 스케일을 제거. 모든 특이값을 1로 균일화.

### 변형들

| 변형 | 수식/설명 | 설정 |
|---|---|---|
| Muon | $\theta_{t+1} = \theta_t - \eta \cdot \text{NS}(\text{momentum}(g_t))$모멘텀 → NS 직교화 → 업데이트 | muon = true |
| AdaMuon | NS 후 Adam-like 적응적 스케일링.$u = \text{NS}(m),\quad \theta \mathrel{-}= \eta \cdot u / (\sqrt{v}+\epsilon)$ | adamuon = true |
| NorMuon | NS 후 열 노름 정규화 | normuon = true |
| Polar Express | $\theta_{t+1} = U_t \cdot \\|W_t\\|_*$극 분해의 직교 인자로 가중치 자체를 제약 | polar_express = true |

### Subspace Momentum (SM)
$$G_\text{proj} = P_r(G) \quad\text{(랜덤 투영으로 차원 축소)}$$
m_t = \beta m_{t-1} + G_\text{proj} \quad\text{(저차원에서 모멘텀)}
$$\theta \mathrel{-}= \eta \cdot P_r^{-1}(m_t) \quad\text{(원래 차원으로 복원)}$$
$$

`rank` 파라미터로 투영 차원 설정. `update_proj_gap`으로 투영 행렬 갱신 주기 결정.

## GenericOptim 설정 예시

```python
# diffusion-pipe config.toml
[optimizer]
type = "genericoptim"
lr = 3e-4
betas = [0.95, 0.999]
weight_decay = 0.01
muon = true                 # Newton-Schulz 직교화
ns_steps = 5                # NS 반복 횟수
momentum_type = "ema"        # ema / heavy_ball / nag
cpu_offload = true           # 옵티마이저 상태 CPU 오프로드
# rank = 64                 # Subspace Momentum 활성화
# automagic = true          # 자동 LR 방식 결합
```

## 직관

> 왜 직교화? 일반 그래디언트는 방향과 크기가 혼합되어 있습니다. 큰 특이값 방향이 지배적이 되면 학습이 불균형해집니다. Newton-Schulz 직교화는 모든 방향의 업데이트 크기를 균일화하여:    특이값이 큰 방향 = 과도한 업데이트 방지   특이값이 작은 방향 = 충분한 업데이트 보장   결과: 더 균일한 학습, 특히 대규모 모델에서 효과적
