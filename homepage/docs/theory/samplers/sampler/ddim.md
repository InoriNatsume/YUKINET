

# Sampler: ddim
**ComfyUI 함수 시그니처**
`ddim alias (sampler_object('ddim') -> euler path)`

**docstring:** ComfyUI alias path

\[
\text{ComfyUI 내부 구현 경로상 Euler alias}
\]

DDIM(alias) 계열 deterministic 해석.

결정론 경로로 보면 확산항이 제거된 continuity equation 관점: $\partial_t\rho+\nabla\cdot(\rho v)=0$. 동적 OT(Benamou-Brenier) 형태의 수송 해석이 용이하다.

## 수학 심화 프로파일

### 순수수학 관점

| 항목 | 내용 |
|---|---|
| method class | generic |
| local truncation | $O(h^2)$ |
| global error | $O(h)$ |
| strong/weak 관점 | 구체 구현에 의존 |
| stability 메모 | 스케줄/노이즈 설정에 의존 |

\[
\mathcal{L}_t\varphi=v_t\cdot\nabla\varphi,\quad \partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0
\]

\[
\min_{\rho,v}\int_0^1\!\!\int \frac12\|v_t(x)\|^2\rho_t(x)\,dx\,dt,\quad \partial_t\rho+\nabla\cdot(\rho v)=0
\]

\[
\|x(t_{k+1})-x_{k+1}\|\le C h_k^{p+1},\quad \|x(T)-x_N\|\le C\max_k h_k^p,\quad h_k:=|\lambda_{k+1}-\lambda_k|
\]

### 수치해석/구현 관점

| 구현 항목 | 내용 |
|---|---|
| 스텝 커널 구조 | $x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k$ |
| 모델 평가량(경향) | 구현/분기 조건에 따라 변동. |
| history 버퍼 | 명시적 history 버퍼 의존이 낮은 단일스텝 구조. |
| 스텝 제어 | 고정 mesh의 deterministic stepper 제어. |
| 메쉬 변수 | $\lambda=\log\alpha-\log\sigma,\ h_k=\|\lambda_{k+1}-\lambda_k\|$ |
| 저장/정밀도 메모 | 기본 latent + 중간 stage 텐서 저장 비용이 주된 메모리 사용처. |

\[
\lambda=\log\alpha-\log\sigma,\quad x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k
\]

\[
v_{\mathrm{cfg}}=v_u+w(v_c-v_u)
\]

\[
x_{k+1}=m_k(x_k),\quad \partial_t\rho+\nabla\cdot(\rho v)=0
\]

**family:** DDIM(alias) / **stochastic:** no

## 유도 스케치(순수수학) / 구현 절차(수치해석)

**대상:** `ddim` / **family:** DDIM(alias)

### 순수수학 유도 스케치

\[
x_{k+1}=x_k+\mathcal{I}_k^{(drift)}+\mathcal{C}_k^{(history)}+\mathcal{N}_k^{(noise)}
\]

이 항 분해에서 drift/correction/noise를 어떤 차수로 근사하는지가 sampler family의 본질이다.

### 수치해석 구현 절차

1. scheduler로 mesh를 고정한 뒤 stepper를 선택한다.
2. history 버퍼와 모델 평가 횟수의 비용-정확도 균형을 맞춘다.
3. 필요 시 stochastic 항을 조절해 분산과 구조 보존을 트레이드오프한다.

**ComfyUI 경로:** `comfy/samplers.py::sampler_object('ddim') -> ksampler('euler', random inpaint)`

**독립 구현 전략:** 현재는 comfy_native 위임 권장(후속 standalone 확장)

**참고:** ComfyUI 내부에서 euler+random inpaint option alias로 취급

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
| 결정론적 모델 | $\Omega=\{\omega_0\}$, $\mathcal{F}=\{\varnothing,\Omega\}$, $\mathbb{P}(\Omega)=1$ | 확률기호는 형식적으로만 남고 난수항은 제거된다. |
| step 사상 축약 | $\Phi_k:\mathcal{X}\times\mathcal{H}_k\to\mathcal{X}$ | 난수 인자 $\Omega_k$가 소거된 결정론적 사상으로 동작. |

해석 팁: `k`는 이산 step index, `t`는 연속시간 변수로 구분한다. 또한 $\mathcal{X}\times\mathcal{H}_k\times\Omega_k$ 위에서 정의된 $\Phi_k$의 가측성은 코드에서 난수 소비 순서(시드 재현성)와 직접 연결된다.

### 직관/구체 원소 예시

| 기호 | 원소 예시 | 직관 |
|---|---|---|
| $x_k\in\mathcal{X}$ | $d=4$ 예시에서 $x_k=(0.12,-0.34,1.08,0.00)$ | 현재 latent 상태의 한 점. |
| $h_k\in\mathcal{H}_k$ | 2-step이면 $h_k=(x_{k-1},x_k)$ | 다단계 solver의 과거값 벡터. |
| $\Phi_k$ | $x_{k+1}=\Phi_k(x_k,h_k,\omega_k)$ | 한 step에서 상태를 다음 상태로 보내는 사상. |
| $\Psi_k$ | $\Psi_k((x_{k-1},x_k),x_{k+1})=(x_k,x_{k+1})$ | 슬라이딩 윈도우 형태의 과거값 갱신. |
| 결정론적 경우 | $\Omega=\{\omega_0\}$ | 난수 경로가 하나뿐이라 잡음항이 사라짐. |

### 해당 sampler의 추가 제약

| 제약 | 조건 | 의미 |
|---|---|---|
| mesh 단조성 | $\sigma_{k+1}\le\sigma_k$, $h_k:=\|\lambda_{k+1}-\lambda_k\|>0$ | 역적분 안정성 및 오차 분석의 기본 가정. |
| drift 정칙성 | $\lVert b_\theta(x,t)-b_\theta(y,t)\rVert\le L\lVert x-y\rVert$ | 존재/유일성과 수치해석 수렴률에 필요한 대표 가정. |

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

추가 파라미터가 거의 없거나 공통 인자 중심으로 동작합니다.

## 파라미터-수식 기호 대응

| 코드 파라미터 | 수식 기호 | 들어가는 항 | 해석 |
|---|---|---|---|
| model | $\hat{x}_0(\cdot;\theta,c)$ | drift 항 | denoiser/score 기반 추정기 |
| x | $x_k$ | 상태 변수 | 현재 latent 상태 |
| sigmas | $\{\sigma_k\}_{k=0}^{N}$ | 시간 재매개화 | 노이즈 스케줄 격자 |
| extra_args | $c,\ \text{options}$ | 조건 벡터장 | conditioning/옵션 전달 |
| callback | $\mathcal{C}_k$ | 관측 함수 | 수치 궤적 모니터링 |
| disable | $-$ | UI/로그 제어 | 수학 항에는 직접 미참여 |

## 원본 구현 스니펫
```python
sampler_object('ddim') -> ksampler('euler', inpaint_options={'random': True})
```


