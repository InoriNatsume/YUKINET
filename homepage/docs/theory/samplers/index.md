# 샘플러 개요

## 통합 상태천이 모델

개별 샘플러를 서로 다른 도구로 보지 않고, **동일한 상태천이식**에서 `drift / correction / noise` 항을 교체하는 방식으로 통합합니다.

### 이산 시간 업데이트

\[
x_{k+1} = \Phi_{\mathrm{drift}}(x_k, \sigma_k, \sigma_{k+1}) + \Phi_{\mathrm{corr}}(\text{history}) + \Phi_{\mathrm{noise}}(\eta, s_{\text{noise}}, \xi_k)
\]

여기서 방향 벡터(denoised-to-direction)는:

\[
d(x, \sigma) = \frac{x - \hat{x}_0}{\sigma} \quad \text{(ComfyUI: } \texttt{to\_d}\text{)}
\]

### 연속 시간 동역학

=== "역시간 SDE"

    \[
    dx = \left( f_\theta(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right) dt + g(t) \, d\bar{W}_t
    \]
    
    확산항 포함 — ancestral/SDE 샘플러가 이것의 수치 근사.

=== "확률흐름 ODE"

    \[
    dx = \left( f_\theta(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right) dt
    \]
    
    확산항 제거 — Euler, DPM++, UniPC 등 결정론 샘플러가 이것의 수치 근사.

### 구현 레이어 분해

| 레이어 | 핵심 객체 | 수학 역할 | 튜닝 포인트 |
|---|---|---|---|
| Model | `model(x, σ, **args)` | $\hat{x}_0$ / $v_\theta$ / $\epsilon_\theta$ 추정 | cond, adapter, cfg |
| Schedule | `calculate_sigmas` | $\{\sigma_k\}$ mesh 생성 | karras, exp, kl_optimal |
| Stepper | `sample_*` | local/global error 및 stability | family, order, solver_type |
| Noise | `η, s_noise, seed` | 분산/재현성 제어 | diversity vs structure |
| Controller | `rtol/atol/PID` | accept/reject 기반 오차 제어 | dpm_adaptive 계열 |

```python
# 통합 stepper 스켈레톤
for k in range(N):
    sigma, sigma_next = sigmas[k], sigmas[k+1]
    den = model(x, sigma, **extra_args)
    x = drift_update(x, den, sigma, sigma_next)
    x = correction_update(x, history, sigma, sigma_next)
    x = noise_update(x, eta, s_noise, seed, k)
    history = update_history(history, den)
```

## 분류

### 수렴 차수별

| 분류 | 대표 샘플러 | 특징 |
|---|---|---|
| **1차 단일스텝** | Euler, Euler Ancestral | 빠르지만 step 수 낮으면 bias 큼 |
| **2차 단일스텝** | Heun, DPM_2, DPM++ 2S | 같은 step에서 1차보다 안정적 |
| **다단계 (Multistep)** | LMS, DPM++ 2M, DEIS, UniPC | 과거 기울기 활용, 초기 워밍업 필요 |
| **적응 스텝** | DPM Adaptive | 내부 오차 제어로 step 크기 자동 조정 |

### Stochastic 여부

| 유형 | 동작 | 제어 파라미터 |
|---|---|---|
| **결정론 (ODE)** | 확률흐름 ODE 적분 | seed 고정 → 완전 재현 |
| **확률론 (SDE)** | 역시간 SDE 적분 + 노이즈 재주입 | `eta`, `s_noise` |

## 개별 샘플러

| 샘플러 | Family | Stochastic | 수렴 차수 | 상세 |
|---|---|---|---|---|
| [Euler](sampler/euler.md) | Euler | No | 1차 | 가장 기본적인 ODE 솔버 |
| [Euler Ancestral](sampler/euler_ancestral.md) | Euler | Yes | 1차 | ancestral 노이즈 재주입 |
| [Heun](sampler/heun.md) | Heun | No | 2차 | predictor-corrector 계열 |
| [DPM_2](sampler/dpm_2.md) | DPM | No | 2차 | DPM 계열 기본형 |
| [DPM++ 2M](sampler/dpmpp_2m.md) | DPM++ | No | multistep | 실무에서 자주 쓰는 균형형 |
| [DPM++ 2M SDE](sampler/dpmpp_2m_sde.md) | DPM++ | Yes | multistep | 품질/다양성 절충 |
| [LMS](sampler/lms.md) | LMS | No | multistep | 선형 다단계 적분 |
| [DDIM](sampler/ddim.md) | DDIM(alias) | No | 1차 근사 | ComfyUI 내부 alias 경로 |
| [DEIS](sampler/deis.md) | DEIS | No | multistep | 고차 지수 적분 계열 |
| [UniPC](sampler/uni_pc.md) | UniPC | No | predictor-corrector | 낮은 step 효율형 |
| [SA-Solver](sampler/sa_solver.md) | SA | Yes | multistep | 확률적 Adams 계열 |
| [SEEDS 2](sampler/seeds_2.md) | SEEDS | Yes | multistage | 확률형 exp-integrator |

!!! tip "읽기 순서"
    1. 먼저 이 개요에서 **family와 분류**를 파악
    2. 관심 있는 개별 샘플러 페이지에서 **수식/파라미터** 확인
    3. [도구/ComfyUI/샘플러](../../tools/comfyui/samplers.md)에서 **구현 코드** 확인
