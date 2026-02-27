# ComfyUI 샘플러 구현

<span class="version-badge">ComfyUI v0.14.2</span>
<span class="version-badge">comfy/k_diffusion/sampling.py</span>

이 페이지는 ComfyUI의 `sample_*` 함수들이 [이론/샘플러](../../theory/samplers/index.md)의 수식을 **어떻게 코드로 구현하는지** 분석합니다.

## `sample_euler` 구현

**이론 참조**: [→ 이론/샘플러/Euler](../../theory/samplers/sampler/euler.md)

```python
# comfy/k_diffusion/sampling.py — sample_euler
@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, 
                 disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), 
                 s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    for i in trange(len(sigmas) - 1, disable=disable):
        # churn (선택적 perturbation)
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) \
                if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        
        # 핵심: model → denoised → direction → euler step
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)           # d = (x - denoised) / σ
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 
                      'sigma_hat': sigma_hat, 'denoised': denoised})
        
        dt = sigmas[i + 1] - sigma_hat              # Δσ (음수)
        x = x + d * dt                              # Euler step
    
    return x
```

### 코드 ↔ 수식 매핑

| 코드 | 수식 | 역할 |
|---|---|---|
| `denoised = model(x, σ)` | $D_\theta(x_k, \sigma_k)$ | 모델의 $\hat{x}_0$ 예측 |
| `d = to_d(x, σ, denoised)` | $d_k = \frac{x_k - D_\theta}{\sigma_k}$ | 방향 벡터 |
| `dt = σ[i+1] - σ_hat` | $\Delta\sigma = \sigma_{k+1} - \sigma_k$ | step 크기 |
| `x = x + d * dt` | $x_{k+1} = x_k + d_k \cdot \Delta\sigma$ | Euler 업데이트 |
| `gamma`, `eps` | $\gamma$, $s_\text{noise} \cdot \xi_k$ | churn perturbation |

## 전체 샘플러 목록

총 **44개** 샘플러 (`KSAMPLER_NAMES` + `ddim`, `uni_pc`, `uni_pc_bh2`):

| 샘플러 | Family | Stochastic | 이론 | 분석 상태 |
|---|---|---|---|---|
| `euler` | Euler | No | [Euler](../../theory/samplers/sampler/euler.md) | ✅ |
| `euler_ancestral` | Euler | Yes | — | 예정 |
| `heun` | Heun | No | — | 예정 |
| `dpmpp_2m_sde` | DPM++ | Yes | — | 예정 |
| ... | — | — | — | — |

*기존 sampler_site의 44개 샘플러 분석을 순차적으로 마이그레이션 예정.*
