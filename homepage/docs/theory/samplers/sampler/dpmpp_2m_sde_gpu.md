

# Sampler: dpmpp_2m_sde_gpu
**ComfyUI 함수 시그니처**
`sample_dpmpp_2m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint')`

\[
x\leftarrow\frac{\sigma_t}{\sigma_s}e^{-\eta h}x+\alpha_t(-\mathrm{expm1}(-h_\eta))\hat{x}_0+\mathrm{corr}_{2M}+\sigma_t\sqrt{-\mathrm{expm1}(-2\eta h)}\,s_{\mathrm{noise}}\xi
\]

2M 계열은 이전 step 정보를 활용하는 반-다단계 구조. solver_type(midpoint/heun)에 따라 correction 항의 안정성/정확도 특성이 달라진다.

FPE 관점에서 drift + diffusion가 모두 활성화된다: $\partial_t\rho=-\nabla\cdot(\rho b)+\frac12 g^2\Delta\rho$. OT 관점에서는 entropic regularization이 있는 bridge 해석이 자연스럽다.

## 수학 심화 프로파일

### 순수수학 관점

| 항목 | 내용 |
|---|---|
| method class | 2-step multistep / exponential integrator |
| local truncation | $O(h^3)\ \text{(smooth regime)}$ |
| global error | $O(h^2)$ |
| strong/weak 관점 | SDE 항이 있으면 diffusion 분산은 유지, correction은 deterministic bias 완화에 기여 |
| stability 메모 | history 기반 보정으로 저스텝에서도 비교적 안정적인 품질 |

\[
\mathcal{L}_t\varphi=b_t\cdot\nabla\varphi+\frac12 g_t^2\Delta\varphi,\quad \partial_t\rho_t=\mathcal{L}_t^\star\rho_t
\]

\[
\rho_{k+1}\approx\arg\min_\rho\left(\frac{W_2^2(\rho,\rho_k)}{2\tau_k}+\mathcal{F}(\rho)\right)
\]

\[
\|x(t_{k+1})-x_{k+1}\|\le C h_k^{p+1},\quad \|x(T)-x_N\|\le C\max_k h_k^p,\quad h_k:=|\lambda_{k+1}-\lambda_k|
\]

### 수치해석/구현 관점

| 구현 항목 | 내용 |
|---|---|
| 스텝 커널 구조 | $x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k$ |
| 모델 평가량(경향) | history 재사용 구조라 step당 평가 횟수는 낮지만, 버퍼/계수 업데이트가 추가된다. |
| history 버퍼 | 직전 단계 history(보통 1-step memory)를 유지한다. |
| 스텝 제어 | 고정 mesh 위에서 noise injection 파라미터(eta, s_noise 등)로 분산 제어. |
| 메쉬 변수 | $\lambda=\log\alpha-\log\sigma,\ h_k=\|\lambda_{k+1}-\lambda_k\|$ |
| 저장/정밀도 메모 | 노이즈 샘플링/연산 일부가 GPU 경로로 최적화될 수 있다. |

\[
\lambda=\log\alpha-\log\sigma,\quad x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k
\]

\[
v_{\mathrm{cfg}}=v_u+w(v_c-v_u)
\]

\[
x_{k+1}=m_k(x_k)+G_k\xi_k,\ \xi_k\sim\mathcal{N}(0,I),\ \mathrm{Cov}[x_{k+1}|x_k]=G_kG_k^\top
\]

**family:** DPM++ / **stochastic:** yes

## 유도 스케치(순수수학) / 구현 절차(수치해석)

**대상:** `dpmpp_2m_sde_gpu` / **family:** DPM++

### 순수수학 유도 스케치

\[
x_{k+1}=x_k+\mathcal{I}_k^{(drift)}+\mathcal{C}_k^{(history)}+\mathcal{N}_k^{(noise)}
\]

이 항 분해에서 drift/correction/noise를 어떤 차수로 근사하는지가 sampler family의 본질이다.

### 수치해석 구현 절차

1. scheduler로 mesh를 고정한 뒤 stepper를 선택한다.
2. history 버퍼와 모델 평가 횟수의 비용-정확도 균형을 맞춘다.
3. 필요 시 stochastic 항을 조절해 분산과 구조 보존을 트레이드오프한다.

**ComfyUI 경로:** `comfy/k_diffusion/sampling.py::sample_dpmpp_2m_sde_gpu`

**독립 구현 전략:** standalone_sampler_lab/samplers.py에 구현됨

**참고:** 노이즈 샘플링 경로 GPU 변형

## 기호 계약(정의역/공역/조건)

기호별 상세 위키: **Symbol Wiki Index**

| 항목 | 수식 | 설명 |
|---|---|---|
| 상태 변수 | $x_k:(\Omega,\mathcal{F}_k)\to(\mathcal{X},\mathcal{B}(\mathcal{X}))$ | $\mathcal{X}$는 보통 $\mathbb{R}^d$ (유한차원 힐베르트 공간). |
| 조건 변수 | $c:(\Omega,\mathcal{F})\to(\mathcal{C},\mathcal{G})$ 또는 고정 매개변수 $c\in\mathcal{C}$ | 조건은 가측 사상 또는 상수 매개변수로 모델링. |
| 스케줄 사상 | $S:\{0,\dots,N\}\to\Sigma$, $k\mapsto\sigma_k$ | 단조감소 가정이 일반적이며 $h_k=\|\lambda_{k+1}-\lambda_k\|$가 오차를 지배. |
| 모형 사상 | $D_\theta:(\mathcal{X}\times\Sigma\times\mathcal{C})\to\mathcal{X}$ | 측도론적으로는 $(\mathcal{B}(\mathcal{X})\otimes\mathcal{B}(\Sigma)\otimes\mathcal{G},\mathcal{B}(\mathcal{X}))$-가측 사상. |
| 이산시간 전이 사상 | $\Phi_k:(\mathcal{X}\times\mathcal{H}_k\times\Omega_k)\to\mathcal{X}$ | 시간지수 $k$에서 $k+1$로 가는 상태전이 사상. |
| 다단계 과거값 갱신 사상 | $\Psi_k:(\mathcal{H}_k\times\mathcal{X})\to\mathcal{H}_{k+1}$ | 과거값 벡터를 다음 단계의 과거값 벡터로 옮기는 사상. |
| 적응성/가측성 | $x_k$는 $\mathcal{F}_k$-가측, $\Phi_k$는 $(\mathcal{B}(\mathcal{X})\otimes\mathcal{A}_k\otimes\mathcal{F}_k,\mathcal{B}(\mathcal{X}))$-가측 | 미래 잡음 미참조(non-anticipative) 조건을 형식화. |
| 기저 확률공간 | $\Omega=(\mathbb{R}^d)^N$, $\mathcal{F}=\mathcal{B}(\Omega)$, $\mathbb{P}=\bigotimes_{k=0}^{N-1}\mathcal{N}(0,I_d)$ | 이산 stochastic sampler의 표준 곱측도 모델. |
| 잡음 확률변수 | $\xi_k:(\Omega,\mathcal{F},\mathbb{P})\to(\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d))$, $\xi_k\sim\mathcal{N}(0,I_d)$ | $\xi_k\in L^2(\Omega;\mathbb{R}^d)$. 상관잡음이면 공분산 연산자를 명시해야 함. |
| 필트레이션 | $\mathcal{F}_k=\sigma(\xi_0,\dots,\xi_{k-1})$ 및 $x_k$의 $\mathcal{F}_k$-가측성 | 현재 상태가 과거 정보에만 의존한다는 적응성 조건. |
| 접두 경로공간 | $\Omega_k=(\mathbb{R}^d)^k$ | k-step까지 사용한 난수 경로를 나타내는 부분공간. |

해석 팁: `k`는 이산 step index, `t`는 연속시간 변수로 구분한다. 또한 $\mathcal{X}\times\mathcal{H}_k\times\Omega_k$ 위에서 정의된 $\Phi_k$의 가측성은 코드에서 난수 소비 순서(시드 재현성)와 직접 연결된다.

### 직관/구체 원소 예시

| 기호 | 원소 예시 | 직관 |
|---|---|---|
| $x_k\in\mathcal{X}$ | $d=4$ 예시에서 $x_k=(0.12,-0.34,1.08,0.00)$ | 현재 latent 상태의 한 점. |
| $h_k\in\mathcal{H}_k$ | 2-step이면 $h_k=(x_{k-1},x_k)$ | 다단계 solver의 과거값 벡터. |
| $\Phi_k$ | $x_{k+1}=\Phi_k(x_k,h_k,\omega_k)$ | 한 step에서 상태를 다음 상태로 보내는 사상. |
| $\Psi_k$ | $\Psi_k((x_{k-1},x_k),x_{k+1})=(x_k,x_{k+1})$ | 슬라이딩 윈도우 형태의 과거값 갱신. |
| $\omega\in\Omega$ | $\omega=(\xi_0,\xi_1,\dots,\xi_{N-1})$ | 전체 샘플링 과정에서 소비될 난수 경로 하나. |
| $\xi_k$ | $d=3$ 예시: $\xi_k=(0.31,-1.24,0.08)$ | k번째 step의 가우시안 잡음 벡터. |

### 해당 sampler의 추가 제약

| 제약 | 조건 | 의미 |
|---|---|---|
| mesh 단조성 | $\sigma_{k+1}\le\sigma_k$, $h_k:=\|\lambda_{k+1}-\lambda_k\|>0$ | 역적분 안정성 및 오차 분석의 기본 가정. |
| drift 정칙성 | $\lVert b_\theta(x,t)-b_\theta(y,t)\rVert\le L\lVert x-y\rVert$ | 존재/유일성과 수치해석 수렴률에 필요한 대표 가정. |
| 확률강도 | $\eta\ge0$ | noise 주입 강도/드리프트 감쇠 결합. |
| 노이즈 배율 | $s_{noise}\ge0$ | 분산 스케일 파라미터. |
| 보정자 선택 | $solver\_type\in\{\mathrm{midpoint},\mathrm{heun}\}$ | 보정식 계열 전환. |

## 공통 인자(시그니처 공통부)

| 인자 | 타입/의미 | 역할 |
|---|---|---|
| model | denoiser callable | 모델 함수 |
| x | latent | 현재 상태 |
| sigmas | sigma schedule | 스텝별 노이즈 스케일 |
| extra_args | dict | seed/model_options 등 추가 인자 |
| callback | callable | 진행 콜백 |
| disable | bool | progress disable |

## sampler 고유 파라미터 상세

| 파라미터 | 기본값 | 수학/알고리즘 역할 | KSampler 노출 경로 |
|---|---|---|---|
| eta | 1. | 확률항 강도 및 drift 감쇠에 반영 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| s_noise | 1. | noise term 배수 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| noise_sampler | None | 코드 레벨 노이즈 샘플러 함수 주입 | 코드 레벨 파라미터. |
| solver_type | 'midpoint' | 보정 방식(midpoint/heun 등) 선택 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |

## 파라미터-수식 기호 대응

| 코드 파라미터 | 수식 기호 | 들어가는 항 | 해석 |
|---|---|---|---|
| model | $\hat{x}_0(\cdot;\theta,c)$ | drift 항 | denoiser/score 기반 추정기 |
| x | $x_k$ | 상태 변수 | 현재 latent 상태 |
| sigmas | $\{\sigma_k\}_{k=0}^{N}$ | 시간 재매개화 | 노이즈 스케줄 격자 |
| extra_args | $c,\ \text{options}$ | 조건 벡터장 | conditioning/옵션 전달 |
| callback | $\mathcal{C}_k$ | 관측 함수 | 수치 궤적 모니터링 |
| disable | $-$ | UI/로그 제어 | 수학 항에는 직접 미참여 |
| eta | $\eta$ | diffusion 강도 | 노이즈 주입 강도 및 drift 감쇠와 결합 |
| s_noise | $s_{\text{noise}}$ | noise 스케일 | \xi_k \mapsto s_{\text{noise}}\xi_k |
| noise_sampler | $noise_sampler$ | 구현 의존 | sampler-specific tuning parameter |
| solver_type | $\psi\in\{\text{midpoint, heun}\}$ | 보정 연산자 | corrector 식 선택 |

## 원본 구현 스니펫
```python
def sample_dpmpp_2m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=extra_args.get("seed", None), cpu=False) if noise_sampler is None else noise_sampler
    return sample_dpmpp_2m_sde(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, solver_type=solver_type)
```


