

# Sampler: dpm_adaptive
family: DPMstochastic: optionalcfg_pp: nogpu_variant: nostandalone: no
**ComfyUI 함수 시그니처**
`sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False)`

**docstring:** DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927.
$$ \text{adaptive DPM-solver with local error test }(rtol,atol)\ \&\ \text{PID step controller} $$
적응형 step-size 제어(PID + local error test). rtol/atol과 accept_safety가 품질-시간 Pareto를 결정한다.

결정론 경로로 보면 확산항이 제거된 continuity equation 관점: $\partial_t\rho+\nabla\cdot(\rho v)=0$. 동적 OT(Benamou-Brenier) 형태의 수송 해석이 용이하다.

## 수학 심화 프로파일

### 순수수학 관점

| 항목 | 내용 |
|---|---|
| method class | adaptive DPM solver |
| local truncation | $\\|e_k\\|\le \text{atol} + \text{rtol}\cdot\\|x_k\\|$ |
| global error | $허용오차 기반으로 자동 제어(고정 차수 표현보다 tolerance 해석이 적합)$ |
| strong/weak 관점 | accept/reject와 PID 계수가 계산량-오차 균형을 직접 제어 |
| stability 메모 | accept_safety와 초기 step(h_init) 설정이 수렴/속도의 핵심 |
$$\mathcal{L}_t\varphi=b_t\cdot\nabla\varphi+\frac12 g_t^2\Delta\varphi,\quad \partial_t\rho_t=\mathcal{L}_t^\star\rho_t$$$$\rho_{k+1}\approx\arg\min_\rho\left(\frac{W_2^2(\rho,\rho_k)}{2\tau_k}+\mathcal{F}(\rho)\right)$$$$\|x(t_{k+1})-x_{k+1}\|\le C h_k^{p+1},\quad \|x(T)-x_N\|\le C\max_k h_k^p,\quad h_k:=|\lambda_{k+1}-\lambda_k|$$
### 수치해석/구현 관점

| 구현 항목 | 내용 |
|---|---|
| 스텝 커널 구조 | $x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k$ |
| 모델 평가량(경향) | 고정된 step당 횟수보다 accept/reject 및 내부 제어 루프에 의해 총 평가 횟수가 결정된다. |
| history 버퍼 | 명시적 history 버퍼 의존이 낮은 단일스텝 구조. |
| 스텝 제어 | 오차 추정 + PID(step controller) 기반 accept/reject 제어. |
| 메쉬 변수 | $\lambda=\log\alpha-\log\sigma,\ h_k=\|\lambda_{k+1}-\lambda_k\|$ |
| 저장/정밀도 메모 | 기본 latent + 중간 stage 텐서 저장 비용이 주된 메모리 사용처. |
$$\lambda=\log\alpha-\log\sigma,\quad x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k$$$$v_{\mathrm{cfg}}=v_u+w(v_c-v_u)$$$$x_{k+1}=m_k(x_k)+G_k\xi_k,\ \xi_k\sim\mathcal{N}(0,I),\ \mathrm{Cov}[x_{k+1}|x_k]=G_kG_k^\top$$
**family:** DPM / **stochastic:** optional

## 유도 스케치(순수수학) / 구현 절차(수치해석)

**대상:** `dpm_adaptive` / **family:** DPM

### 순수수학 유도 스케치
$$x_{k+1}^{[p]},\ x_{k+1}^{[p-1]}\ \text{를 동시 계산},\quad err_k=\frac{\|x_{k+1}^{[p]}-x_{k+1}^{[p-1]}\|}{atol+rtol\|x_{k+1}^{[p]}\|}$$$$err_k\le1\ \Rightarrow\ accept,\quad h_{new}=h\cdot s\cdot err_k^{-\beta}\cdot err_{k-1}^{\gamma}$$
주요 오차원천: embedded error estimator 편향, 과도한 reject 반복, 모델 비매끄러움.

### 수치해석 구현 절차

1. rtol/atol를 기준으로 accept/reject 루프를 돈다.
2. pcoeff/icoeff/dcoeff로 PID step 제어를 수행한다.
3. accept_safety가 작을수록 보수적(안정)이나 step 수가 증가한다.

**ComfyUI 경로:** `comfy/samplers.py wrapper -> comfy/k_diffusion/sampling.py::sample_dpm_adaptive`

**독립 구현 전략:** 현재는 comfy_native 위임 권장(후속 standalone 확장)

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
| 다단계 곱공간 | $\mathcal{H}_k=\mathcal{X}^{m_k}$, $\mathcal{A}_k=\mathcal{B}(\mathcal{X})^{\otimes m_k}$, $m_k\le m$ | 초기 구간에서는 $m_k$가 작고, 진행되며 최대 차수까지 증가. |
| 적분 구간 | $\sigma\in[\sigma_{\min},\sigma_{\max}]$ | 경계 선택이 해상도 보존과 계산량 균형에 직접 영향. |

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
| drift 정칙성 | $\\|b_\theta(x,t)-b_\theta(y,t)\\|\le L\\|x-y\\|$ | 존재/유일성과 수치해석 수렴률에 필요한 대표 가정. |
| 확률강도 | $\eta\ge0$ | noise 주입 강도/드리프트 감쇠 결합. |
| 노이즈 배율 | $s_{noise}\ge0$ | 분산 스케일 파라미터. |
| 상대오차 허용치 | $rtol>0$ | adaptive accept/reject 기준. |
| 절대오차 허용치 | $atol>0$ | adaptive accept/reject 기준. |
| 안전계수 | $0<accept\_safety\le1$ | 수락 조건의 보수성 제어. |

## 공통 인자(시그니처 공통부)

| 인자 | 타입/의미 | 역할 |
|---|---|---|
| model | denoiser callable | 모델 함수 |
| x | latent | 현재 상태 |
| sigma_min/sigma_max | integration interval | 적분 구간 경계 |
| extra_args | dict | 추가 인자 |
| callback | callable | 스텝 콜백 |
| disable | bool | progress disable |

## sampler 고유 파라미터 상세

| 파라미터 | 기본값 | 수학/알고리즘 역할 | KSampler 노출 경로 |
|---|---|---|---|
| order | 3 | 솔버 차수 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| rtol | 0.05 | 상대오차 허용치 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| atol | 0.0078 | 절대오차 허용치 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| h_init | 0.05 | 초기 step size | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| pcoeff | 0. | PID P 계수 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| icoeff | 1. | PID I 계수 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| dcoeff | 0. | PID D 계수 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| accept_safety | 0.81 | accept 안전계수 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| eta | 0. | 확률항 강도 및 drift 감쇠에 반영 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| s_noise | 1. | noise term 배수 | Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어. |
| noise_sampler | None | 코드 레벨 노이즈 샘플러 함수 주입 | 코드 레벨 파라미터. |
| return_info | False | 디버그 정보 반환 | 코드 레벨 파라미터. |

## 파라미터-수식 기호 대응

| 코드 파라미터 | 수식 기호 | 들어가는 항 | 해석 |
|---|---|---|---|
| model | $\hat{x}_0(\cdot;\theta,c)$ | drift 항 | denoiser/score 기반 추정기 |
| x | $x_k$ | 상태 변수 | 현재 latent 상태 |
| sigma_min/sigma_max | $\sigma_{\min},\sigma_{\max}$ | 적분 구간 | 경계값 |
| extra_args | $c,\ \text{options}$ | 조건 벡터장 | conditioning/옵션 전달 |
| callback | $\mathcal{C}_k$ | 관측 함수 | 수치 궤적 모니터링 |
| disable | $-$ | UI/로그 제어 | 수학 항에는 직접 미참여 |
| order | $m$ | 다단계 차수 | 히스토리 계수 차수 |
| rtol | $\varepsilon_{\text{rel}}$ | 오차 기준 | 상대오차 허용치 |
| atol | $\varepsilon_{\text{abs}}$ | 오차 기준 | 절대오차 허용치 |
| h_init | $h_0$ | 초기 스텝 | 적응형 시작 크기 |
| pcoeff | $K_P$ | PID 제어 | 오차 기반 step 조정 |
| icoeff | $K_I$ | PID 제어 | 누적 오차 반영 |
| dcoeff | $K_D$ | PID 제어 | 오차 변화율 반영 |
| accept_safety | $s_{\text{acc}}$ | 수락 조건 | 보수적 수락 계수 |
| eta | $\eta$ | diffusion 강도 | 노이즈 주입 강도 및 drift 감쇠와 결합 |
| s_noise | $s_{\text{noise}}$ | noise 스케일 | \xi_k \mapsto s_{\text{noise}}\xi_k |
| noise_sampler | $noise_sampler$ | 구현 의존 | sampler-specific tuning parameter |
| return_info | $return_info$ | 구현 의존 | sampler-specific tuning parameter |

## 원본 구현 스니펫
```python
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)
    if return_info:
        return x, info
    return x
```
