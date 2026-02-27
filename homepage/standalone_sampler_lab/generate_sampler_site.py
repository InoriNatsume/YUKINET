"""
Per-sampler website generator.

Outputs:
- sampler_site/index.html
- sampler_site/sampler/<sampler_name>.html (44 pages)
- sampler_site/scheduler/<scheduler_name>.html
- sampler_site/symbol/<symbol_id>.html + sampler_site/symbol/index.html
- sampler_site/assets/style.css

Run:
python standalone_sampler_lab\\generate_sampler_site.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from generate_sampler_homepage import bridge_rows, build_data, scheduler_rows
from taxonomy import SCHEDULER_NAMES_ALL


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "sampler_site"
SAMPLER_DIR = SITE / "sampler"
SCHED_DIR = SITE / "scheduler"
SYMBOL_DIR = SITE / "symbol"
ASSET_DIR = SITE / "assets"


SYMBOL_WIKI: list[dict[str, Any]] = [
    {
        "id": "X",
        "tex": r"\mathcal{X}",
        "name": "상태공간 (State space)",
        "strict": [
            r"\mathcal{X}는 실수 힐베르트 공간으로 둔다.",
            r"실무 구현에서는 \mathcal{X}=\mathbb{R}^{d},\ d=C\cdot H\cdot W.",
            r"측도론에서는 (\mathcal{X},\mathcal{B}(\mathcal{X}))를 사용한다.",
        ],
        "intuition": [
            "모든 latent 벡터가 놓이는 바탕 좌표공간이다.",
            "샘플러는 결국 이 공간 위 점을 이동시키는 규칙이다.",
        ],
        "examples": [
            r"d=4일 때 x=(0.2,-1.1,0.0,3.4)\in\mathbb{R}^4.",
            r"x_k\in\mathcal{X}는 k번째 step의 latent 상태.",
        ],
        "related": ["B_X", "Hk_Ak", "Phi_k", "D_theta"],
    },
    {
        "id": "B_X",
        "tex": r"\mathcal{B}(\mathcal{X})",
        "name": "보렐 σ-대수 (Borel sigma-algebra)",
        "strict": [
            r"\mathcal{B}(\mathcal{X})는 \mathcal{X}의 열린집합이 생성하는 sigma-대수.",
            r"확률변수/사상의 가측성은 보통 이 sigma-대수 기준으로 정의한다.",
        ],
        "intuition": [
            "확률을 부여할 수 있는 사건의 모음(허용 가능한 부분집합들)이다.",
        ],
        "examples": [
            r"\mathcal{X}=\mathbb{R}^d이면 \mathcal{B}(\mathbb{R}^d).",
            r"x_k:(\Omega,\mathcal{F})\to(\mathcal{X},\mathcal{B}(\mathcal{X}))가 가측.",
        ],
        "related": ["X", "Omega_F_P", "Phi_k"],
    },
    {
        "id": "Sigma",
        "tex": r"\Sigma",
        "name": "노이즈 스케일 구간 (Noise scale interval)",
        "strict": [
            r"\Sigma=[\sigma_{\min},\sigma_{\max}]\subset(0,\infty).",
            r"스케줄 사상 S:\{0,\dots,N\}\to\Sigma의 공역.",
        ],
        "intuition": [
            "샘플링 중 현재 노이즈 강도(해상도 단계)를 표시하는 축이다.",
        ],
        "examples": [
            r"\sigma_0=14.6,\ \sigma_N=0.",
            r"S(k)=\sigma_k.",
        ],
        "related": ["S_map", "D_theta"],
    },
    {
        "id": "C_G",
        "tex": r"(\mathcal{C},\mathcal{G})",
        "name": "조건변수 가측공간 (Measurable conditioning space)",
        "strict": [
            r"\mathcal{C}는 조건변수 값공간, \mathcal{G}는 그 위 sigma-대수.",
            r"c:(\Omega,\mathcal{F})\to(\mathcal{C},\mathcal{G})를 가측 사상으로 본다.",
        ],
        "intuition": [
            "프롬프트/컨트롤 같은 외생 조건을 수학적으로 담는 그릇이다.",
        ],
        "examples": [
            r"c\in\mathcal{C}를 고정 파라미터로 둘 수도 있다.",
            r"D_\theta(x,\sigma,c)\in\mathcal{X}.",
        ],
        "related": ["D_theta", "b_theta"],
    },
    {
        "id": "Hk_Ak",
        "tex": r"(\mathcal{H}_k,\mathcal{A}_k)",
        "name": "다단계 과거값 곱공간 (Multi-step history product space)",
        "strict": [
            r"\mathcal{H}_k=\mathcal{X}^{m_k},\ \mathcal{A}_k=\mathcal{B}(\mathcal{X})^{\otimes m_k}.",
            r"m_k=0이면 한 점 공간(one-point space)으로 본다.",
        ],
        "intuition": [
            "multistep sampler가 들고 다니는 과거값 버퍼의 수학적 모델이다.",
        ],
        "examples": [
            r"2-step이면 h_k=(x_{k-1},x_k)\in\mathcal{X}^2.",
            r"\Psi_k((x_{k-1},x_k),x_{k+1})=(x_k,x_{k+1}).",
        ],
        "related": ["X", "Psi_k", "Phi_k"],
    },
    {
        "id": "K_set",
        "tex": r"\mathcal{K}",
        "name": "허용집합 (Admissible set)",
        "strict": [
            r"\mathcal{K}\subset\mathcal{X}를 닫힌 집합(필요시 닫힌 볼록집합)으로 둔다.",
            r"\Pi_{\mathcal{K}}:\mathcal{X}\to\mathcal{K}는 metric projection.",
        ],
        "intuition": [
            "해가 벗어나면 다시 집합 안으로 되돌리는 제약 영역이다.",
        ],
        "examples": [
            r"\mathcal{K}=[-1,1]^d이면 좌표별 clip과 동치.",
        ],
        "related": ["Pi_K", "X"],
    },
    {
        "id": "M_manifold",
        "tex": r"\mathcal{M}",
        "name": "매장 부분다양체 (Embedded submanifold)",
        "strict": [
            r"\mathcal{M}\subset\mathcal{X}를 C^r\ (r\ge1) 매장 부분다양체로 둔다.",
            r"x\in\mathcal{M}에서 접공간 T_x\mathcal{M}이 정의된다.",
        ],
        "intuition": [
            "전체 공간이 아니라 곡면/곡다양체 위에서 상태를 진화시키는 설정이다.",
        ],
        "examples": [
            r"\mathcal{M}=S^{d-1}\subset\mathbb{R}^d 같은 구면 제약.",
        ],
        "related": ["TxM", "R_x", "X"],
    },
    {
        "id": "TxM",
        "tex": r"T_x\mathcal{M}",
        "name": "접공간 (Tangent space)",
        "strict": [
            r"x\in\mathcal{M}에서의 접벡터들이 이루는 선형공간.",
        ],
        "intuition": [
            "다양체 위에서 순간적으로 움직일 수 있는 방향들의 선형근사.",
        ],
        "examples": [
            r"구면에서는 중심반경에 직교하는 벡터들이 T_x\mathcal{M}.",
        ],
        "related": ["M_manifold", "R_x"],
    },
    {
        "id": "S_map",
        "tex": r"S",
        "name": "스케줄 사상 (Schedule map)",
        "strict": [
            r"S:\{0,\dots,N\}\to\Sigma,\ k\mapsto\sigma_k.",
            r"보통 \sigma_{k+1}\le\sigma_k.",
        ],
        "intuition": [
            "이산 시간 인덱스를 노이즈 강도 축으로 재매개화한다.",
        ],
        "examples": [
            r"S(0)=\sigma_{\max},\ S(N)=0.",
        ],
        "related": ["Sigma", "Phi_k"],
    },
    {
        "id": "D_theta",
        "tex": r"D_\theta",
        "name": "모형 사상 (Model map: denoiser/score)",
        "strict": [
            r"D_\theta:(\mathcal{X}\times\Sigma\times\mathcal{C})\to\mathcal{X}.",
            r"보통 가측이며 국소 Lipschitz 가정을 둔다.",
        ],
        "intuition": [
            "현재 상태와 노이즈 수준, 조건을 넣으면 이동 방향 정보를 돌려주는 블랙박스.",
        ],
        "examples": [
            r"\hat{x}_{0,k}=D_\theta(x_k,\sigma_k,c).",
        ],
        "related": ["X", "Sigma", "C_G", "b_theta"],
    },
    {
        "id": "b_theta",
        "tex": r"b_\theta",
        "name": "drift 사상 (Drift map)",
        "strict": [
            r"b_\theta:\mathcal{X}\times[0,1]\times\mathcal{C}\to\mathcal{X}.",
        ],
        "intuition": [
            "확률과 무관한 평균적 이동 성분(결정론적 흐름)이다.",
        ],
        "examples": [
            r"ODE: \dot x_t=b_\theta(x_t,t,c).",
        ],
        "related": ["D_theta", "g_func", "Rho_t"],
    },
    {
        "id": "g_func",
        "tex": r"g",
        "name": "diffusion 계수 함수 (Diffusion coefficient function)",
        "strict": [
            r"g:[0,1]\to[0,\infty).",
        ],
        "intuition": [
            "언제 얼마나 잡음을 섞을지 정하는 시간 의존 스칼라 함수.",
        ],
        "examples": [
            r"g(t)=0이면 probability-flow ODE로 축약.",
        ],
        "related": ["b_theta", "Rho_t"],
    },
    {
        "id": "Phi_k",
        "tex": r"\Phi_k",
        "name": "이산시간 전이 사상 (Discrete-time transition map)",
        "strict": [
            r"\Phi_k:(\mathcal{X}\times\mathcal{H}_k\times\Omega_k)\to\mathcal{X}.",
            r"적응성 조건: \Phi_k는 \mathcal{F}_k 기준 비예견(non-anticipative).",
        ],
        "intuition": [
            "한 step 업데이트 함수 자체를 수학적으로 부르는 이름.",
        ],
        "examples": [
            r"x_{k+1}=\Phi_k(x_k,h_k,\omega_k).",
        ],
        "related": ["Psi_k", "S_map", "Omega_F_P"],
    },
    {
        "id": "Psi_k",
        "tex": r"\Psi_k",
        "name": "과거값 갱신 사상 (History update map)",
        "strict": [
            r"\Psi_k:(\mathcal{H}_k\times\mathcal{X})\to\mathcal{H}_{k+1}.",
        ],
        "intuition": [
            "multistep에서 과거 정보를 밀어 넣고 업데이트하는 연산.",
        ],
        "examples": [
            r"\Psi_k((x_{k-1},x_k),x_{k+1})=(x_k,x_{k+1}).",
        ],
        "related": ["Hk_Ak", "Phi_k"],
    },
    {
        "id": "Pi_K",
        "tex": r"\Pi_{\mathcal{K}}",
        "name": "집합 투영 연산자 (Projection operator)",
        "strict": [
            r"\Pi_{\mathcal{K}}:\mathcal{X}\to\mathcal{K}.",
            r"\mathcal{K}가 닫힌 볼록집합이면 각 x에 대해 유일.",
        ],
        "intuition": [
            "제약을 위반한 점을 가장 가까운 합법 점으로 되돌린다.",
        ],
        "examples": [
            r"\mathcal{K}=[-1,1]^d이면 \Pi_{\mathcal{K}}는 coordinate-wise clipping.",
        ],
        "related": ["K_set", "X"],
    },
    {
        "id": "R_x",
        "tex": r"R_x",
        "name": "리트랙션 (Retraction)",
        "strict": [
            r"R_x:T_x\mathcal{M}\to\mathcal{M},\ R_x(0)=x,\ dR_x(0)=\mathrm{Id}.",
        ],
        "intuition": [
            "접공간의 작은 벡터를 다시 다양체 위 점으로 올리는 지도.",
        ],
        "examples": [
            r"정규화 retraction: R_x(v)=\frac{x+v}{\|x+v\|}\ (\text{구면의 경우}).",
        ],
        "related": ["M_manifold", "TxM"],
    },
    {
        "id": "Omega_F_P",
        "tex": r"(\Omega,\mathcal{F},\mathbb{P})",
        "name": "확률공간 (Probability space)",
        "strict": [
            r"\Omega: 표본공간,\ \mathcal{F}: sigma-대수,\ \mathbb{P}: 확률측도.",
        ],
        "intuition": [
            "샘플링 중 발생 가능한 모든 난수 시나리오를 담는 무대.",
        ],
        "examples": [
            r"\Omega=(\mathbb{R}^d)^N,\ \mathbb{P}=\bigotimes_{k=0}^{N-1}\mathcal{N}(0,I_d).",
        ],
        "related": ["Xi_k", "Fk", "Phi_k"],
    },
    {
        "id": "Xi_k",
        "tex": r"\xi_k",
        "name": "잡음 확률변수 (Noise random variable)",
        "strict": [
            r"\xi_k:(\Omega,\mathcal{F},\mathbb{P})\to(\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d)).",
            r"표준 가정: \xi_k\sim\mathcal{N}(0,I_d),\ i.i.d.",
        ],
        "intuition": [
            "k번째 step에서 실제로 샘플되는 난수 벡터.",
        ],
        "examples": [
            r"d=3에서 \xi_k=(0.31,-1.24,0.08).",
        ],
        "related": ["Omega_F_P", "Fk", "Phi_k"],
    },
    {
        "id": "Fk",
        "tex": r"\mathcal{F}_k",
        "name": "자연 필트레이션 (Natural filtration)",
        "strict": [
            r"\mathcal{F}_k=\sigma(\xi_0,\dots,\xi_{k-1}).",
        ],
        "intuition": [
            "k시점까지 알고 있는 정보의 집합.",
        ],
        "examples": [
            r"x_k가 \mathcal{F}_k-가측이면 미래 잡음을 미리 보지 않는다는 뜻.",
        ],
        "related": ["Xi_k", "Phi_k", "Omega_F_P"],
    },
    {
        "id": "Rho_t",
        "tex": r"\rho_t",
        "name": "시간별 분포 (Time-dependent distribution)",
        "strict": [
            r"\rho_t\in\mathcal{P}_2(\mathcal{X})\ \text{(2차 모멘트 유한 확률측도)}.",
        ],
        "intuition": [
            "개별 샘플이 아니라 전체 샘플 집단이 시간에 따라 어떻게 퍼지는지 표현.",
        ],
        "examples": [
            r"FPE: \partial_t\rho=-\nabla\cdot(b_\theta\rho)+\frac12\nabla\cdot((gg^\top)\nabla\rho).",
        ],
        "related": ["V_t", "b_theta", "g_func"],
    },
    {
        "id": "V_t",
        "tex": r"v_t",
        "name": "속도장 (Velocity field)",
        "strict": [
            r"v_t\in L^2_{\rho_t}(\mathcal{X};\mathcal{X}).",
        ],
        "intuition": [
            "분포를 어느 방향으로 운반하는지 나타내는 벡터장.",
        ],
        "examples": [
            r"\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0.",
        ],
        "related": ["Rho_t", "Phi_k"],
    },
]


SYMBOL_GUIDE: dict[str, dict[str, Any]] = {
    "X": {
        "summary": "샘플러가 매 step 업데이트하는 ‘상태 벡터(=latent)’가 놓이는 공간이다. 구현에서는 대부분 실수 벡터공간 $\\mathbb{R}^d$로 생각하면 충분하다.",
        "why": [
            r"ComfyUI에서 `KSampler`(및 변형 노드)는 매 step마다 `latent` 텐서 `x`를 업데이트한다. 이때 `x`를 수학적으로 $x\in\mathcal{X}$라고 선언해 두면, 샘플링은 “공간 $\mathcal{X}$ 위의 점을 반복적으로 이동시키는 알고리즘”으로 한 문장에 정리된다.",
            r"샘플러/스케줄러를 바꾸면 한 step이 만드는 이동의 크기와 방향이 달라진다. 이는 보통 $\|x_{k+1}-x_k\|_2$ 같은 거리로 ‘얼마나 움직였는지’를 측정하며, 수치해석의 안정성/오차(그리고 결과의 뭉개짐·과샤프·색폭주 같은 체감 품질)와 직접 연결된다.",
            r"편집 작업(인페인트/아웃페인트, 마스크 고정, latent 합성, clip/normalize, 후처리 denoise 등)은 결국 $\mathcal{X}\to\mathcal{X}$ 사상들의 합성이다. 어떤 연산이 “같은 공간에서” 합법적으로 합성되는지(예: latent끼리 합치기 vs 픽셀에서 합치기)를 명확히 하려면 먼저 $\mathcal{X}$를 고정해야 한다.",
            r"모델 출력(denoised/score/velocity)은 ‘다음으로 가는 방향’을 나타내는 같은 차원의 벡터다. 즉 $D_\theta(x,\sigma,c)\in\mathcal{X}$, $b_\theta(x,t,c)\in\mathcal{X}$ 같은 타입 선언이 서야 ODE/SDE 해석(벡터장, 흐름, 적분)이 문법적으로 성립한다.",
            r"영상 생성(AnimateDiff 등)처럼 시간축이 붙으면 텐서 차원은 늘지만, 수학적으로는 여전히 거대한 유클리드 공간의 한 점이다(예: $\mathbb{R}^{C\cdot T\cdot H\cdot W}$). 이 관점이 있어야 “프레임 간 일관성” 같은 문제도 결국 같은 상태공간에서의 제약/수송 문제로 통일해 볼 수 있다.",
        ],
        "formal": [
            r"$\mathcal{X}$를 실수 힐베르트 공간(내적공간의 완비화)으로 두고, 내적 $\langle\cdot,\cdot\rangle$과 노름 $\|x\|=\sqrt{\langle x,x\rangle}$을 사용한다.",
            r"실무(유한차원)에서는 $\mathcal{X}=\mathbb{R}^d$로 두고 $\langle x,y\rangle=x^\top y$, $\|x\|=\|x\|_2$를 쓴다.",
            r"확률적 샘플링을 엄밀히 쓰면 각 step의 상태는 가측 사상 $x_k:(\Omega,\mathcal{F}_k)\to(\mathcal{X},\mathcal{B}(\mathcal{X}))$로 본다.",
        ],
        "formal_explain": [
            r"첫 문장은 “$x$가 어떤 ‘공간’의 원소인지”를 명시하는 선언이다. 힐베르트 공간을 가정하면 내적과 노름이 자동으로 따라오고, 이는 수치해석에서 오차(예: $\|x_{k+1}-x_k\|$)를 재는 기준이 된다.",
            r"두 번째 문장은 실무에서 거의 항상 쓰는 특수화다. 모델과 샘플러가 다루는 텐서는 유한차원이므로, 결국 $\mathbb{R}^d$의 한 벡터로 보면 된다. 여기서 $d$는 채널/해상도 등을 곱한 총 자유도다.",
            r"세 번째 문장은 “샘플러가 난수를 쓰는 확률 과정”이라는 점을 엄밀히 적는 방식이다. 보통 구현을 이해하는 데는 꼭 필요하진 않지만, 분포 $\rho_t$, FPE/OT 같은 문서로 넘어가면 결국 이 문장들이 문법을 책임진다.",
        ],
        "intuition": [
            "샘플러는 ‘점 $x$를 조금씩 옮기는 규칙’이고, $\\mathcal{X}$는 그 점이 사는 좌표공간이다.",
            "이미지 픽셀 공간이 아니라 ‘압축된 좌표(latent)’라 해도, 수학적으로는 그냥 벡터공간의 한 점이다.",
            r"ComfyUI에서는 `latent` 텐서가 바로 $x\in\mathcal{X}$이고, `KSampler`의 각 step은 이 $x$를 업데이트한다. 미리보기(preview)는 보통 VAE 디코더 $\\mathrm{Dec}:\\mathcal{X}\\to\\mathbb{R}^{H\\times W\\times 3}$를 적용해 $x$를 이미지로 ‘관측’하는 것이라고 이해하면 된다.",
            "모델 출력(denoised/score/velocity)도 결국 같은 차원의 벡터이므로 $\\mathcal{X}$의 원소로 취급된다.",
        ],
        "examples": [
            r"$\mathcal{X}=\mathbb{R}^3$라고 두면 $x=(0.2,\,-1.1,\,3.4)\in\mathcal{X}$는 ‘구체적인 상태’ 한 개다.",
            r"$\mathcal{X}=\mathbb{R}^4$에서 $x_0=(1.0,0.0,0.0,0.0)$, $x_1=(0.7,-0.2,0.1,0.0)$처럼 step마다 값이 바뀐다.",
            r"텐서가 `x.shape=(1,4,2,2)`라면 펼쳐서 $d=16$인 벡터로 볼 수 있고, 예를 들어 첫 몇 성분이 $(0.1,-0.3,0.0,0.8,...)$처럼 실수로 채워진다.",
            r"(ComfyUI 예) 512×512 기준으로 latent가 `x.shape=(1,4,64,64)`라면 $d=4\cdot64\cdot64=16384$이고, 예를 들어 펼친 벡터의 앞부분이 $(0.12,-0.34,0.05,0.80,\dots)$처럼 실수로 채워진 한 점 $x\in\mathbb{R}^{16384}$로 볼 수 있다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\mathcal{X}=\mathbb{R}^d",
                "explain": [
                    r"유한차원 구현에서는 상태를 길이 $d$인 실수 벡터로 본다. 여기서 $d$는 ‘latent 텐서의 총 성분 수’다.",
                    r"텐서 모양이 $(C,H,W)$이면 $d=C\cdot H\cdot W$이고, 배치까지 포함하면 샘플 단위로는 각 배치 원소가 $\mathbb{R}^d$의 한 점이다.",
                    r"이렇게 보면 sampler는 $\mathbb{R}^d$ 위에서 $x\mapsto x'$를 반복하는 알고리즘으로 해석된다(수치적분 관점).",
                ],
                "intuition": [
                    r"‘이미지’가 아니라 ‘좌표’라고 생각하자. 좌표가 몇 차원이든, 샘플러는 그 좌표를 움직인다.",
                    r"코드의 `x` 텐서는 결국 한 점 $x\in\mathbb{R}^d$로 읽으면 된다.",
                ],
                "examples": [
                    r"$d=4$라면 $x=(1.0,\,-0.2,\,0.1,\,0.0)\in\mathbb{R}^4$.",
                    r"$C=4,H=2,W=2$면 $d=16$이고, 예를 들어 $x=(0.1,-0.3,0.0,0.8,\,0.2,0.0,\dots)$ 같은 벡터가 된다.",
                ],
            },
            {
                "stmt": r"\langle x,y\rangle=x^\top y,\qquad \|x\|_2=\sqrt{x^\top x}",
                "explain": [
                    r"내적/노름은 ‘크기’와 ‘거리’를 정의한다. 예를 들어 projection $\Pi_{\mathcal{K}}$는 $\|x-y\|_2$를 최소화하는 $y$를 찾는 연산이다.",
                    r"solver의 step-size가 작을수록 $\|x_{k+1}-x_k\|_2$가 작아지는 경향이 있고, 이는 수치 안정성 직관과 연결된다.",
                ],
                "intuition": [
                    r"실무에서는 “유클리드 거리”로 생각하면 대부분 맞는다.",
                ],
                "examples": [
                    r"$x=(1,2)$, $y=(3,-1)$이면 $\langle x,y\rangle=1\cdot3+2\cdot(-1)=1$.",
                    r"$x=(3,4)$이면 $\|x\|_2=5$.",
                ],
            },
            {
                "stmt": r"x_k:(\Omega,\mathcal{F}_k)\to(\mathcal{X},\mathcal{B}(\mathcal{X}))",
                "explain": [
                    r"이 문장은 “$x_k$는 난수에 의해 결정되는 값(확률변수)”라는 뜻이다. 입력은 $\omega\in\Omega$ (난수 시나리오)이고 출력은 상태공간의 점 $x_k(\omega)\in\mathcal{X}$다.",
                    r"가측성은 ‘사건을 사건으로 보낸다’는 조건인데, 실무에서는 보통 “코드가 난수로부터 $x_k$를 계산한다” 정도로 받아들이면 된다.",
                ],
                "intuition": [
                    r"seed를 바꾸면 $\omega$가 바뀌고, 그 결과로 $x_k$ 값도 바뀐다. 이 관계를 엄밀히 적은 표현이다.",
                ],
                "examples": [
                    r"예를 들어 $\Omega=\{1,2\}$, $\mathbb{P}(1)=\mathbb{P}(2)=0.5$라 두고, $x_k(1)=(0.0,0.0)$, $x_k(2)=(1.0,0.0)$처럼 정의하면 $x_k$는 $\mathbb{R}^2$값 확률변수다.",
                ],
            },
        ],
        "code_map": [
            "`x`, `latent`, `x_t` 같은 변수들이 보통 $\\mathcal{X}$의 원소다.",
            "KSampler step 함수의 입력/출력 첫 인자가 이 공간에서 움직인다.",
        ],
        "pitfalls": [
            "공간을 명시하지 않으면 노름/내적/거리(수치오차 분석의 기준)가 모호해진다.",
        ],
    },
    "B_X": {
        "summary": r"상태공간 $\mathcal{X}$에서 ‘확률을 붙일 수 있는 집합들’을 모아 둔 표준 $\sigma$-대수(보렐 $\sigma$-대수)다. 확률변수/분포를 엄밀히 쓰려면 배경으로 거의 항상 깔린다.",
        "why": [
            r"ComfyUI에서 같은 프롬프트/모델이라도 `seed`, `sampler_name`, `scheduler`, (sampler에 따라) `eta/s_noise/s_churn` 같은 선택에 따라 결과가 달라진다. 이는 출력이 ‘랜덤’이라는 뜻이고, 수학적으로는 각 결과가 어떤 확률변수 $x_k$의 실현값이라는 뜻이다. 확률변수의 문법을 제대로 쓰려면 값공간 $\mathcal{X}$ 위의 사건들을 모아 둔 $\sigma$-대수 $\mathcal{B}(\mathcal{X})$가 필요하다.",
            r"‘특정 현상이 발생할 확률’(예: latent 성분이 범위 밖으로 튀는 사건, 마스크 고정이 깨지는 사건, 특정 영역이 과노출되는 사건)을 말하려면, 그 현상이 $\mathcal{X}$의 부분집합 $A\subset\mathcal{X}$로 먼저 정의돼야 한다. $\mathcal{B}(\mathcal{X})$는 이런 사건들을 놓을 “표준 사건 공간”을 제공한다.",
            r"유한차원 $\mathcal{X}=\mathbb{R}^d$에서는 우리가 직관적으로 정의하는 거의 모든 집합(구간, 원판, 다각형, 반공간 등)이 Borel 집합이다. 그래서 실제 생성/편집 논의에서 “사건이 너무 기괴해서 확률을 못 붙인다” 같은 기술적 문제를 대부분 피할 수 있다.",
            r"샘플러를 SDE로 보고 밀도 $\rho_t$의 진화를 다루면(FPE 관점), $\rho_t$는 $\mathcal{B}(\mathcal{X})$ 위의 측도(또는 그에 대한 밀도)로 해석된다. 즉 ‘분포가 시간에 따라 변한다’는 문장을 엄밀하게 쓰기 위한 최소 배경이 $\mathcal{B}(\mathcal{X})$다.",
            r"커스텀 샘플러를 만들 때는 조건 분기/클리핑/리젝션 같은 규칙을 넣는 경우가 많다(예: $\|x\|_2>R$이면 보정). 이런 규칙이 확률 과정으로서 잘 정의되려면, 역상 $x_k^{-1}(A)$가 사건이 되는 가측성 조건이 깔려 있어야 하고, 그때 표준 선택이 $\mathcal{B}(\mathcal{X})$다.",
        ],
        "formal": [
            r"$\mathcal{B}(\mathcal{X})$는 $\mathcal{X}$의 열린집합이 생성하는 Borel $\sigma$-대수다.",
            r"가측성은 보통 $(\Omega,\mathcal{F})\to(\mathcal{X},\mathcal{B}(\mathcal{X}))$ 기준으로 정의한다.",
        ],
        "formal_explain": [
            r"첫 문장은 $\mathcal{B}(\mathcal{X})$가 “열린집합들을 포함하는 가장 작은 $\sigma$-대수”라는 뜻이다. 왜 하필 열린집합인가? 거리/위상 구조가 있을 때 열린집합이 가장 기본적인 ‘관측 가능한’ 집합이기 때문이다(볼, 근방 등).",
            r"둘째 문장은 확률변수의 문법이다. $x_k$가 가측이라는 말은, 상태공간에서 사건 $A\in\mathcal{B}(\mathcal{X})$를 하나 잡으면 “$x_k$가 $A$ 안에 들어간다”라는 사건이 표본공간에서 $\mathcal{F}$-사건으로 표현된다는 뜻이다.",
            r"코드 레벨에서는 $\sigma$-대수를 직접 다루지 않는다. 대신 “$x_k$는 난수로부터 계산되는 어떤 함수”라는 상식이 가측성 가정을 사실상 대신한다.",
        ],
        "intuition": [
            r"쉽게 말해 ‘$\mathcal{X}$에서 우리가 보통 생각하는 자연스러운 집합들(구간, 원판, 직사각형 등)’은 거의 다 Borel 집합이라서, 실무에서는 큰 스트레스 없이 쓴다.",
            r"$\sigma$-대수는 ‘여집합/가산합집합을 해도 닫혀 있는 집합족’이라서 확률의 연산 규칙과 잘 맞는다.",
            r"ComfyUI 관점에서는 “어떤 조건을 만족하는 latent들의 모음”을 사건(event) $A\subset\mathcal{X}$로 생각하면 된다. 예를 들어 ‘$\|x\|_2\le 10$’ 같은 조건은 전형적으로 $A\in\mathcal{B}(\mathcal{X})$에 속하고, 샘플러가 그 사건을 얼마나 자주 방문하는지(=확률)를 말할 수 있다.",
        ],
        "examples": [
            r"$\mathcal{X}=\mathbb{R}$이면 구간 $(-1,2)$, $[0,1]$ 같은 집합은 모두 $\mathcal{B}(\mathbb{R})$에 속한다.",
            r"$\mathcal{X}=\mathbb{R}^2$이면 원판 $A=\{x\in\mathbb{R}^2:\|x\|_2\le 1\}$도 Borel 집합이다. 예를 들어 $(0.6,0.8)\in A$이고 $(2,0)\notin A$.",
            r"(ComfyUI/코드 예) $\mathcal{X}=\mathbb{R}^2$, 사건 $A=\{x:\|x\|_2\le 1\}$라 하자. 어떤 step에서 $x=(1.4,-0.5)$면 $\|x\|_2\approx1.49$로 $x\notin A$다. 반면 정규화(예: $x':=x/\|x\|_2\approx(0.94,-0.34)$)를 적용하면 $x'\in A$가 된다. 이런 “사건을 만족하도록 보정한다”는 논의는 결국 $A\in\mathcal{B}(\mathcal{X})$ 위에서 진행된다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\mathcal{B}(\mathcal{X})=\sigma(\{U\subset\mathcal{X}:U\ \text{open}\})",
                "explain": [
                    r"$\sigma(\cdot)$는 “주어진 집합족을 포함하는 가장 작은 $\sigma$-대수”를 뜻한다. 따라서 $\mathcal{B}(\mathcal{X})$는 열린집합들로부터 출발해, 여집합/가산합집합을 반복해서 만들어지는 모든 집합들의 모음이다.",
                    r"이렇게 정의하면 거리/위상으로 자연스럽게 표현되는 사건들(예: ‘$\|x\|<1$’)이 자동으로 사건집합에 포함된다.",
                ],
                "intuition": [
                    r"‘구간/원판/사각형 같은 평범한 집합들’을 확률 사건으로 허용하기 위한 최소한의 틀이라고 보면 된다.",
                ],
                "examples": [
                    r"$\mathcal{X}=\mathbb{R}$에서 열린집합은 (예: $(-1,2)$) 같은 구간들의 합집합으로 표현된다. 따라서 $(-1,2)\in\mathcal{B}(\mathbb{R})$.",
                    r"$A_n=(-1/n,1/n)$이면 $\bigcap_{n\ge1}A_n=\{0\}$이고, $\{0\}$도 Borel 집합이다(가산 연산에 닫혀 있으므로).",
                ],
            },
            {
                "stmt": r"x_k\ \text{measurable}\iff \forall A\in\mathcal{B}(\mathcal{X}),\ x_k^{-1}(A)\in\mathcal{F}",
                "explain": [
                    r"가측성의 정의는 ‘역상(preimage)이 사건이 된다’는 한 문장으로 요약된다.",
                    r"여기서 $x_k^{-1}(A)=\{\omega\in\Omega: x_k(\omega)\in A\}$는 “난수 시나리오 중에서 상태가 사건 $A$에 들어가는 경우”를 모아 둔 집합이다.",
                ],
                "intuition": [
                    r"상태공간에서 “$x$가 이 집합 안에 있나?”를 물으면, 표본공간에서도 “그 질문이 참인 $\omega$는 어떤 것들인가?”를 물을 수 있어야 한다. 그게 가측성이다.",
                ],
                "examples": [
                    r"유한 표본공간 예: $\Omega=\{1,2,3\}$, $\mathcal{F}=2^\Omega$. $x_k(1)=0.1$, $x_k(2)=0.7$, $x_k(3)=-0.2$ (값공간 $\mathcal{X}=\mathbb{R}$). 사건 $A=(0,1)$이면 $x_k^{-1}(A)=\{1,2\}\in\mathcal{F}$라서 가측성 조건을 만족한다.",
                ],
            },
        ],
        "code_map": [
            "코드에서 직접 쓰진 않지만, `x`의 확률분포/기대값/밀도를 다룰 때 배경 가정으로 항상 필요하다.",
        ],
        "pitfalls": [
            r"표본공간의 $\sigma$-대수와 상태공간의 Borel $\sigma$-대수를 혼동하면 사상 방향이 뒤집힌다.",
        ],
    },
    "Sigma": {
        "summary": r"샘플링 과정에서 ‘현재 노이즈의 세기’를 나타내는 스칼라 $\sigma$가 움직이는 값의 범위(구간)이다. 코드에서는 보통 `sigmas` (예: `sigmas=[10,5,2,1,0]`)처럼 step별 $\sigma$ 값 배열로 나타난다.",
        "why": [
            r"ComfyUI `KSampler`에서 `scheduler`는 사실상 `sigmas` 배열 $(\sigma_0,\dots,\sigma_N)$을 만드는 규칙이다. 사용자가 고르는 `steps`와 `scheduler`가 바뀌면 “어느 $\sigma$에서 모델을 몇 번 호출하느냐”가 달라지고, 그 차이가 전역 구성, 디테일, 질감(그레인) 차이로 체감된다.",
            r"img2img/inpaint에서 `denoise`(또는 `start_at_step/end_at_step` 같은 고급 옵션)는 전체 $\Sigma$ 구간을 다 쓰지 않고 부분 구간만 따라가게 만든다. 큰 $\sigma$ 영역을 덜 거치면 원본을 더 보존하지만, 동시에 전역 구성이 바뀔 여지도 줄어든다. 이 관계를 설명하려면 $\Sigma$라는 ‘적분 구간’ 개념이 필요하다.",
            r"대부분의 sampler update는 결국 $\sigma$-축에서의 수치적분(ODE 또는 SDE의 근사)이다. 따라서 $\Sigma$ 위에서 step을 어디에 배치하느냐는 곧 step-size 설계이며, 같은 `steps`라도 저노이즈($\sigma\approx 0$) 근방을 촘촘히 배치하면 미세 디테일이, 고노이즈($\sigma$ 큰) 쪽을 촘촘히 배치하면 전역 구성이 상대적으로 잘 잡히는 경향이 생긴다.",
            r"ComfyUI/k-diffusion 계열에서 `eta`, `s_churn`, `s_tmin/s_tmax`, `s_noise` 같은 하이퍼파라미터는 “특정 $\sigma$ 구간에서만” 잡음을 더 넣거나(=SDE 성분 강화) step을 뒤틀어 탐색을 넓히는 형태로 설계돼 있다. 즉 이 파라미터들을 이해하려면 먼저 $\Sigma$와 구간 선택을 이해해야 한다.",
            r"문헌/코드마다 시간 변수를 $t\in[0,1]$로 쓰기도 하고, $\sigma$를 직접 쓰기도 한다. 둘은 단조 변환으로 연결되지만, 변수를 섞어 쓰면 ‘같은 공식’이 서로 다른 의미가 된다. $\Sigma$를 명시해 두면 “이 sampler는 어떤 축에서 적분하고 있는가”가 깔끔해진다.",
            r"영상 생성에서는 프레임 간 노이즈/세부 질감의 일관성이 특히 중요하다. $\sigma$가 큰 구간에서의 확률적 변동(추가 noise)이 크면 프레임마다 질감이 들쭉날쭉해질 수 있고, 반대로 저노이즈 구간을 너무 성급히 지나가면 디테일이 흔들릴 수 있다. 이 트레이드오프도 결국 $\Sigma$ 위에서의 설계 문제다.",
        ],
        "formal": [
            r"$\Sigma=[\sigma_{\min},\sigma_{\max}]\subset[0,\infty)$를 노이즈 스케일 값공간으로 둔다.",
            r"스케줄은 보통 $S:\{0,\dots,N\}\to\Sigma,\ k\mapsto\sigma_k$로 정의한다.",
        ],
        "formal_explain": [
            r"첫 문장은 “$\sigma$가 음수가 아닌 실수 값이며, 그중에서 우리가 실제로 사용할 범위가 $[\sigma_{\min},\sigma_{\max}]$”라는 뜻이다. 즉 $\Sigma$는 ‘축(axis)’의 역할을 한다.",
            r"둘째 문장은 “이산 step 번호 $k$를 실제 $\sigma$ 값으로 바꿔 주는 함수(룩업)”를 정의한다. 이때 solver는 사실상 $\sigma$ 축 위에서 적분/근사를 수행한다.",
            r"많은 구현에서 $\sigma_N=0$까지 내려가지만, 이론/구현에 따라 $\sigma_{\min}>0$으로 두고 마지막에 별도의 디노이즈 단계로 마무리하기도 한다.",
        ],
        "intuition": [
            r"큰 $\sigma$일수록 상태가 더 ‘흐릿/랜덤’하고, 작은 $\sigma$일수록 더 ‘정제/결정적’이다.",
            r"스케줄은 ‘큰 $\sigma$ 영역을 빨리 지나갈지, 작은 $\sigma$ 근방을 촘촘히 볼지’를 정하는 손잡이다.",
            r"ComfyUI에서 `steps=N`은 $\Sigma$ 위에 점을 $N+1$개 찍는다는 뜻이고, `scheduler`는 그 점들이 어디에 놓일지(=스케줄 $S$)를 정한다. img2img/inpaint의 `denoise`는 이 구간 중 일부만 따라가게 만들어 ‘원본 보존 vs 변화’ 정도를 조절한다.",
        ],
        "examples": [
            r"$\Sigma=[0,10]$이고 step 수 $N=4$라면 한 예로 $\sigma$ 배열을 $(10,\,5,\,2,\,1,\,0)$처럼 둘 수 있다.",
            r"같은 $N=4$에서도 $(10,\,4,\,1,\,0.2,\,0)$처럼 ‘작은 $\sigma$ 쪽을 더 촘촘히’ 두는 스케줄도 가능하다.",
            r"(ComfyUI 단순화) `sigmas=[10,5,2,1,0]`에서 img2img `denoise=0.5`를 “큰 $\sigma$ 절반은 생략”으로 생각하면, 대략 $\sigma$가 5에서 0으로 내려가는 부분 $(5,2,1,0)$만 사용하게 된다(정확한 시작점은 구현/스케줄 정의에 따라 달라질 수 있다).",
        ],
        "walkthrough": [
            {
                "stmt": r"\Sigma=[\sigma_{\min},\sigma_{\max}]\subset[0,\infty)",
                "explain": [
                    r"$\Sigma$는 $\sigma$가 가질 수 있는 값들의 집합이다. 보통 시작점은 큰 노이즈 $\sigma_{\max}$, 끝점은 작은 노이즈 $\sigma_{\min}$ (또는 0)이다.",
                    r"‘구간’으로 잡는 이유는, solver가 결국 이 구간에서 여러 $\sigma$ 값들을 찍어가며 모델을 평가하기 때문이다.",
                ],
                "intuition": [
                    r"지도에서 x축 범위를 정하는 것과 같다. 그 범위 안에서만 step을 배치한다.",
                ],
                "examples": [
                    r"예: $\sigma_{\max}=10$, $\sigma_{\min}=0$이면 $\Sigma=[0,10]$.",
                    r"예: $\sigma_{\max}=14.6$, $\sigma_{\min}=0.03$이면 $\Sigma=[0.03,14.6]$.",
                ],
            },
            {
                "stmt": r"S:\{0,\dots,N\}\to\Sigma,\qquad S(k)=\sigma_k",
                "explain": [
                    r"$S$는 step 번호 $k$를 실제 노이즈 값 $\sigma_k$로 보내는 함수다. 코드에서는 `sigmas[k]` 룩업으로 구현된다.",
                    r"보통 $\sigma_{k+1}\le\sigma_k$로 단조감소하게 둔다. 그래야 ‘점점 노이즈를 줄이며 정제한다’는 그림이 일관된다.",
                    r"solver의 실질적인 step-size는 $(\sigma_{k+1}-\sigma_k)$ 또는 그에 준하는 표현으로 나타나므로, 스케줄이 곧 step-size 설계다.",
                ],
                "intuition": [
                    r"같은 $N$이라도 $S$를 어떻게 고르느냐에 따라 “어느 구간을 자세히 보느냐”가 달라진다.",
                ],
                "examples": [
                    r"$N=4$이고 `sigmas=[10,5,2,1,0]`면 $\sigma_0=10$, $\sigma_4=0$.",
                    r"같은 $N=4$에서 `sigmas=[10,4,1,0.2,0]`면 작은 $\sigma$ 구간이 더 촘촘하다($1\to0.2\to0$).",
                ],
            },
        ],
        "code_map": [
            "`sigmas`, `sigma`, `sigma_hat` 같은 인자/변수가 여기에 해당한다.",
        ],
        "pitfalls": [
            r"시간 $t$와 $\sigma$를 같은 변수처럼 다루면 solver 차수 해석이 꼬인다.",
        ],
    },
    "C_G": {
        "summary": r"프롬프트/컨트롤 같은 ‘조건 입력’ $c$가 어떤 값들을 가질 수 있는지(공간 $\mathcal{C}$)와, 그 위에서 확률/가측성을 말하기 위한 $\sigma$-대수 $\mathcal{G}$를 묶어 $(\mathcal{C},\mathcal{G})$로 적는다.",
        "why": [
            r"ComfyUI에서 ‘조건(conditioning)’은 텍스트 프롬프트(`positive/negative`), ControlNet의 컨트롤 신호, IPAdapter/참조 이미지 특징, 각종 메타데이터까지 한데 묶인 입력이다. 이들을 $c\in\mathcal{C}$로 보고 $D_\theta(x,\sigma,c)$처럼 표기하면, “모델은 상태/노이즈/조건을 받아 같은 공간의 벡터를 돌려준다”는 타입이 고정돼 구현과 수학이 같이 정리된다.",
            r"CFG(Classifier-Free Guidance)는 같은 $x,\sigma$에서 조건 $c$와 비조건(보통 ‘빈 프롬프트’) $c_{\emptyset}$를 동시에 평가해서 두 출력을 선형결합한다. 이때 $c$와 $c_{\emptyset}$가 같은 값공간 $\mathcal{C}$의 원소라는 점이 핵심이며, 그렇지 않으면 CFG 공식 자체가 타입이 맞지 않는다.",
            r"프롬프트를 step에 따라 바꾸는 ‘프롬프트 스케줄’, 영상에서 프레임별로 다른 조건을 주는 작업은 사실상 $k\mapsto c_k$ 또는 $t\mapsto c(t)$를 설계하는 것이다. $(\mathcal{C},\mathcal{G})$를 명시하면 “조건이 바뀌는 축”과 “노이즈가 줄어드는 축($\sigma$)”을 혼동하지 않고 설명할 수 있다.",
            r"편집(inpaint, reference 기반 생성)에서는 조건이 단순 텍스트가 아니라 ‘고정해야 할 정보’를 포함한다(마스크, 참조 특징, 깊이맵 등). 이를 $c$에 포함시키면, 샘플러는 동일한 전이 규칙을 쓰되 조건만 바꿔 다양한 편집 파이프라인을 한 언어로 설명할 수 있다.",
            r"조건을 고정 파라미터로 둘 수도 있고(대부분의 UI 추론), 데이터셋에서 랜덤으로 뽑는 확률변수로 둘 수도 있다(학습/증강). $(\mathcal{C},\mathcal{G})$를 두면 이 두 관점을 같은 문법 안에서 오가며 쓸 수 있다.",
        ],
        "formal": [
            r"$\mathcal{C}$는 조건값의 집합, $\mathcal{G}\subseteq 2^{\mathcal{C}}$는 그 위 $\sigma$-대수다.",
            r"확률적 조건은 $c:(\Omega,\mathcal{F})\to(\mathcal{C},\mathcal{G})$인 가측 사상으로 둔다.",
        ],
        "formal_explain": [
            r"첫 문장은 “조건값이 어떤 형태의 대상인지”를 고정한다. 가장 흔한 경우는 $\mathcal{C}=\mathbb{R}^n$ 같은 유한차원 실수공간(임베딩 벡터)이다.",
            r"둘째 문장은 조건이 랜덤일 수도 있다는 점을 포함한다. 예를 들어 데이터셋에서 조건을 랜덤으로 뽑거나, 조건 자체가 노이즈를 포함하는 파이프라인이면 $c$를 확률변수로 보는 게 자연스럽다.",
            r"반대로 대부분의 UI/추론에서는 조건을 고정하고 샘플링만 랜덤으로 한다. 그때는 $c\in\mathcal{C}$인 상수 매개변수로 보고, 가측성 문장은 배경 가정으로만 남는다.",
        ],
        "intuition": [
            r"조건은 ‘문제 설정’을 바꾸는 외생 입력이다. 같은 노이즈 $x$라도 조건 $c$가 달라지면 모델이 다른 방향을 제시한다.",
            r"실무에선 조건이 텍스트 임베딩/이미지 피처/컨트롤 신호 등 ‘실수 벡터’인 경우가 많다.",
            r"추론(inference)에서는 보통 $c$를 고정 매개변수로 두고, 난수는 $\xi_k$에서만 들어온다. 따라서 “같은 조건 $c$에서 seed만 바꿔 여러 결과를 뽑는다”는 말은 $\rho_t(\cdot\mid c)$에서의 샘플링을 뜻한다.",
            r"ComfyUI의 CFG는 같은 $(x,\sigma)$에서 조건 $c$와 비조건 $c_\emptyset$를 각각 넣어 $D_\theta$를 두 번 평가하고, `cfg`로 두 출력을 affine combination 한다. 수학적으로는 “조건이 벡터장을 어떻게 바꾸는가”를 가장 직접적으로 보여 주는 구현이다.",
            r"코드 구현 관점에선 조건 객체가 복잡해 보여도(positive/negative, control, ipadapter 등), 샘플러 입장에서는 결국 `cond` 인자 하나로 들어오며 ‘값공간 $\mathcal{C}$의 한 원소’로 취급된다. 그래서 커스텀 샘플러는 $\mathcal{C}$의 내부 표현을 몰라도 된다(인터페이스 분리).",
        ],
        "examples": [
            r"가장 단순히 $\mathcal{C}=\mathbb{R}^2$로 두고 $c=(1.5,\,-0.3)\in\mathcal{C}$ 같은 벡터를 조건으로 쓸 수 있다.",
            r"조건을 고정하면 $c$는 상수다. 예: 모든 step에서 동일하게 $c=(0.2,0.2)$를 넣는다.",
            r"(CFG 수치 예) 어떤 $(x,\sigma)$에서 $D_\theta(x,\sigma,c)=(1.0,\,2.0)$, $D_\theta(x,\sigma,c_\emptyset)=(0.2,\,1.5)$라고 하자. `cfg=7`이면 guided 출력은 $(0.2,1.5)+7\cdot((1.0,2.0)-(0.2,1.5))=(5.8,\,5.0)$처럼 차이가 크게 증폭된다.",
            r"(쌍 조건 예) $\mathcal{C}=\mathbb{R}^2\times\mathbb{R}^2$로 두고 $c=(c^+,c^-)$. 예: $c^+=(1.0,0.0)$, $c^-=(0.0,1.0)$이면 $c=((1.0,0.0),(0.0,1.0))\in\mathcal{C}$.",
            r"(조건 스케줄 예) $N=4$일 때 $c_0=(1,0)$, $c_1=(1,0)$, $c_2=(0,1)$, $c_3=(0,1)$처럼 step에 따라 조건을 바꾸면 “절반은 A 조건, 절반은 B 조건” 같은 스케줄을 수학적으로 $k\mapsto c_k$로 쓸 수 있다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\mathcal{C}\ \text{(value space)},\qquad \mathcal{G}\subseteq 2^{\mathcal{C}}\ \text{($\sigma$-algebra)}",
                "explain": [
                    r"$\mathcal{C}$는 조건값이 들어갈 ‘값공간’이다. 텍스트 임베딩이라면 보통 $\mathbb{R}^n$, 범주형 라벨이면 유한 집합 $\{1,\dots,K\}$ 같은 형태가 된다.",
                    r"$\mathcal{G}$는 조건값에 대해 ‘사건’을 말하기 위한 집합족이다. $\mathcal{C}=\mathbb{R}^n$이면 대개 $\mathcal{G}=\mathcal{B}(\mathbb{R}^n)$로 둔다.",
                ],
                "intuition": [
                    r"조건도 결국 어떤 ‘데이터 타입’이다. $\mathcal{C}$는 그 타입의 수학적 이름이라고 보면 된다.",
                ],
                "examples": [
                    r"예: $\mathcal{C}=\mathbb{R}^2$, $c=(1.5,-0.3)$.",
                    r"예: $\mathcal{C}=\{1,2,3\}$ (3가지 라벨), $\mathcal{G}=2^{\mathcal{C}}$.",
                ],
            },
            {
                "stmt": r"c:(\Omega,\mathcal{F})\to(\mathcal{C},\mathcal{G})\ \text{measurable}",
                "explain": [
                    r"조건을 확률변수로 두면, 각 난수 시나리오 $\omega$에 대해 조건값 $c(\omega)$가 정해진다.",
                    r"가측성은 “조건에 대한 관측/사건이 표본공간 사건으로 옮겨진다”는 최소한의 일관성 조건이다.",
                ],
                "intuition": [
                    r"데이터셋에서 조건을 랜덤으로 뽑는다면, ‘그 랜덤 선택’ 자체가 $c(\omega)$다.",
                ],
                "examples": [
                    r"유한 예: $\Omega=\{1,2\}$, $\mathbb{P}(1)=\mathbb{P}(2)=0.5$. $c(1)=(0,0)$, $c(2)=(1,0)$ (조건공간 $\mathbb{R}^2$).",
                ],
            },
            {
                "stmt": r"c\in\mathcal{C}\ \text{(fixed parameter)}",
                "explain": [
                    r"추론/UI에서는 보통 조건을 고정한다. 이 경우 $c$는 확률변수가 아니라 단순한 상수 매개변수다.",
                    r"그래도 $(\mathcal{C},\mathcal{G})$를 정의해 두면 “조건을 바꾸면 다른 문제를 푼다”는 구조가 명확해진다.",
                ],
                "intuition": [
                    r"프롬프트를 바꾸는 건 $c$를 바꾸는 것이고, 같은 샘플러라도 다른 경로를 걷게 된다.",
                ],
                "examples": [
                    r"$c=(0.2,0.2)$를 고정해 두면 모든 step에서 같은 값을 넣는다.",
                ],
            },
        ],
        "code_map": [
            "`model(x, sigma, cond)`의 `cond` 인자가 여기에 해당한다.",
        ],
        "pitfalls": [
            "조건 공간/가측성 가정을 생략하면 조건부 표기 $p(x|c)$의 수학적 의미가 흐려진다.",
        ],
    },
    "Hk_Ak": {
        "summary": r"다단계(multistep) 샘플러는 현재 상태 $x_k$만으로 다음 값을 만들지 않고, 과거 값들을 ‘버퍼’로 들고 다닌다. 그 버퍼가 사는 공간을 $\mathcal{H}_k$로 적는다.",
        "why": [
            r"ComfyUI에서 `sampler_name`으로 고르는 것들 중 상당수(`lms`, `dpmpp_2m`, `dpmpp_3m_sde`, `deis`, `ipndm` 등)는 multistep 계열이다. 이런 방법은 $x_k$만으로 $x_{k+1}$를 만들지 않고, 이전 step들의 상태/모델출력(=과거 정보)을 함께 섞어 업데이트한다. 그 ‘과거 정보가 들어 있는 버퍼’의 수학적 타입이 $\mathcal{H}_k$다.",
            r"multistep의 핵심은 같은 모델 호출 수에서 더 높은 차수(더 작은 적분 오차)를 얻는 것이다. 대신 과거가 없으면 고차 공식을 쓸 수 없어서, 초기 몇 step은 저차로 시작하는 워밍업이 반드시 생긴다. $\mathcal{H}_k=\mathcal{X}^{m_k}$처럼 쓰면 이 워밍업(버퍼 길이 $m_k$ 증가)을 깔끔하게 한 틀로 설명할 수 있다.",
            r"생성/편집 파이프라인에서 “중간에 멈췄다가 이어서 샘플링”하거나 “일부 step만 추가로 더 돌리기” 같은 요구가 자주 생긴다. 이때 과거 버퍼를 함께 저장/복원하지 않으면, 이어서 돌리는 부분이 사실상 다른 solver(저차)로 바뀌어 결과가 미묘하게 달라질 수 있다. 재현성과 커스터마이징 관점에서 $\mathcal{H}_k$는 실무적으로도 중요하다.",
            r"영상 생성처럼 여러 프레임을 다룰 때는 ‘프레임별로’ history를 어떻게 유지할지(프레임마다 독립인지, 공유/상관을 줄지)를 결정해야 한다. 수학적으로는 “상태공간을 확장하거나(시간축 포함)”, “history의 정의를 프레임 단위로 분리”하는 문제이며, 이를 명확히 쓰는 기호가 $\mathcal{H}_k$다.",
            r"커스텀 solver를 만들 때도 “무엇을 과거로 저장하나”가 설계의 절반이다. 과거 $x$를 저장할 수도 있고, 과거 모델 출력(denoised/score)이나 잔차(residual)를 저장할 수도 있다. $\mathcal{H}_k$를 명시하면 이런 설계 차이를 타입 수준에서 비교할 수 있다.",
        ],
        "formal": [
            r"$\mathcal{H}_k=\mathcal{X}^{m_k}$, $\mathcal{A}_k=\mathcal{B}(\mathcal{X})^{\otimes m_k}$로 둔다.",
            r"$m_k=0$이면 한 점 공간(one-point space)으로 두어 1-step과 호환한다.",
        ],
        "formal_explain": [
            r"첫 문장은 “버퍼가 $m_k$개의 상태를 담는 튜플”이라는 뜻이다. 예를 들어 2-step이면 $h_k=(x_{k-1},x_k)$처럼 두 개를 들고 다닌다.",
            r"여기서 $\mathcal{A}_k$를 같이 적는 이유는, 확률론적으로 ‘버퍼도 확률변수’일 수 있기 때문이다. 하지만 대부분의 실무 독해에서는 $\mathcal{H}_k=\mathcal{X}^{m_k}$만 기억해도 충분하다.",
            r"두 번째 문장($m_k=0$)은 “과거가 아직 없는 초기 step도 같은 수식 틀 안에 넣겠다”는 장치다. 즉, 1-step 방법과 multistep 방법을 한 문장으로 쓰기 위한 통일 표기다.",
        ],
        "intuition": [
            r"코드 관점에서는 `deque`/리스트로 관리하는 ‘이전 값들의 묶음’이다.",
            r"이 버퍼가 있어야 “이번 step의 기울기뿐 아니라 지난 step의 기울기도 섞어서” 같은 문장이 가능해진다.",
            r"warmup 구간에서는 $m_k$가 0→1→…로 증가하므로, ‘처음 몇 step은 같은 이름의 sampler라도 실제로는 저차 공식’이 된다. ComfyUI에서 같은 `sampler_name`이라도 `steps`가 아주 작으면 성질이 바뀌는 이유 중 하나가 이 버퍼 부족(warmup)이다.",
            r"샘플링을 중간에 끊었다가 이어갈 때(재개/resume), $x_k$만 저장하면 정보가 손실될 수 있다. multistep이라면 $h_k$까지 같이 저장해야 “정확히 같은 경로”를 이어갈 수 있다(확률론적으로는 $\mathcal{F}_k$-적응성 유지).",
        ],
        "examples": [
            r"$\mathcal{X}=\mathbb{R}^2$, 2-step이라면 $x_{k-1}=(1.0,0.0)$, $x_k=(0.3,-0.2)$일 때 $h_k=(x_{k-1},x_k)=((1.0,0.0),(0.3,-0.2))\in\mathcal{H}_k$.",
            r"새 값이 $x_{k+1}=(0.1,0.0)$로 계산되면 버퍼 갱신은 $\Psi_k(h_k,x_{k+1})=((0.3,-0.2),(0.1,0.0))$처럼 ‘앞을 버리고 뒤를 붙이는’ 형태가 된다.",
            r"(warmup 수치 예) $\mathcal{X}=\mathbb{R}$, 목표 history 길이 $m=2$라 하자. 초기에는 $m_0=0$이라 $h_0=\ast$이고, $x_0=0.8\to x_1=0.3$가 되면 $m_1=1$로 $h_1=(0.3)$. 다음에 $x_2=-0.1$이면 $m_2=2$가 되어 $h_2=(x_1,x_2)=(0.3,-0.1)$처럼 ‘필요한 만큼 채워진다’.",
            r"(모델출력 history 예) $\mathcal{X}=\mathbb{R}$에서 과거 denoised를 저장한다고 하자. $d_{k-1}=0.20$, $d_k=0.12$이면 $h_k=(0.20,0.12)$이고, 새 값 $d_{k+1}=0.05$를 얻으면 $h_{k+1}=(0.12,0.05)$로 갱신된다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\mathcal{H}_k=\mathcal{X}^{m_k}",
                "explain": [
                    r"$\mathcal{X}$는 상태가 사는 공간(예: $\mathbb{R}^d$)이고, $\mathcal{X}^{m_k}$는 그 공간의 원소 $m_k$개를 한꺼번에 묶은 튜플들의 집합이다.",
                    r"즉 $h_k\in\mathcal{H}_k$는 “과거 상태들을 모아 둔 묶음”이며, multistep solver가 내부적으로 들고 다니는 메모리의 수학적 모델이다.",
                    r"$m_k$가 커질수록 더 많은 과거 정보를 사용할 수 있어, 같은 step 수에서도 더 고차의 근사식을 만들 수 있다(대신 초기 워밍업이 필요).",
                ],
                "intuition": [
                    r"코드에서 `history=[x_{k-1}, x_k]`처럼 리스트로 저장하는 것과 1:1로 대응된다.",
                    r"이 정의를 써두면 “업데이트는 $(x_k,h_k)$의 함수다”처럼 깔끔하게 말할 수 있다.",
                ],
                "examples": [
                    r"$\mathcal{X}=\mathbb{R}^2$, $m_k=2$라 두자. 그러면 $\mathcal{H}_k=\mathbb{R}^2\times\mathbb{R}^2$이고, $h_k=((1.0,0.0),(0.3,-0.2))\in\mathcal{H}_k$.",
                    r"$m_k=3$이면 $h_k=((1.0,0.0),(0.3,-0.2),(0.1,0.0))$처럼 3개를 들고 다닌다.",
                ],
            },
            {
                "stmt": r"\mathcal{A}_k=\mathcal{B}(\mathcal{X})^{\otimes m_k}",
                "explain": [
                    r"확률론을 엄밀히 쓰려면 “$h_k$가 어떤 사건들에 대해 가측인가”를 정해야 한다. $\mathcal{A}_k$는 그 사건들의 표준 선택이다(product $\sigma$-대수).",
                    r"실무적으로는 ‘버퍼도 $\mathcal{X}$값 확률변수들의 튜플’이라고 이해하면 되고, 측도론이 필요 없는 독해에서는 이 줄을 생략해도 무방하다.",
                ],
                "intuition": [
                    r"이 줄은 수치적 구현을 바꾸지 않는다. 다만 “확률변수/분포/기대값”을 논문 스타일로 쓸 때 문법을 맞춰 준다.",
                ],
                "examples": [
                    r"$\mathcal{X}=\mathbb{R}$, $m_k=2$면 $\mathcal{A}_k=\mathcal{B}(\mathbb{R})\otimes\mathcal{B}(\mathbb{R})$이다. 예를 들어 집합 $(-1,2)\times[0,1]$은 $\mathcal{A}_k$에 속한다.",
                ],
            },
            {
                "stmt": r"m_k=0 \Rightarrow \mathcal{H}_k \simeq \{\ast\}",
                "explain": [
                    r"초기에는 과거값이 없으므로 “버퍼가 비어 있다”는 상황이 생긴다. 이를 수학적으로는 ‘한 점만 있는 집합’ $\{\ast\}$로 모델링한다.",
                    r"이렇게 하면 $\Phi_k:\mathcal{X}\times\mathcal{H}_k\to\mathcal{X}$ 같은 전이 사상을 1-step/다단계 모두에 대해 같은 타입으로 쓸 수 있다.",
                ],
                "intuition": [
                    r"`history=[]`를 굳이 따로 처리하지 않고, “형식상은 항상 존재하는 값(더미)”이 있다고 생각하면 구현 분기와 수식 분기가 같은 방향으로 맞춰진다.",
                ],
                "examples": [
                    r"$m_k=0$이면 $h_k=\ast$ 하나뿐이다. 즉 ‘버퍼를 인자로 받지만 실제로는 쓰지 않는’ 1-step solver가 된다.",
                ],
            },
        ],
        "code_map": [
            r"`old_denoised`, `history`, `buffer` 류 상태가 $\mathcal{H}_k$ 역할을 한다.",
        ],
        "pitfalls": [
            "warmup 구간에서 $m_k$가 변한다는 점을 빼먹으면 초기 step 구현이 틀리기 쉽다.",
        ],
    },
    "K_set": {
        "summary": r"샘플러가 만든 상태 $x$가 반드시 만족해야 하는 제약을 모아 둔 ‘허용 집합’이다. 예를 들어 값의 범위를 제한하거나(clip), 마스크 영역을 고정하는(inpaint) 제약을 $\mathcal{K}\subset\mathcal{X}$로 표현한다.",
        "why": [
            r"ComfyUI의 inpaint/outpaint는 단순히 ‘노이즈를 다시 뿌리는’ 작업이 아니라, 마스크로 지정한 영역은 유지하고 나머지만 샘플링하는 제약 문제로 볼 수 있다. 이때 “유지해야 하는 영역을 만족하는 모든 latent의 집합”이 바로 $\mathcal{K}\subset\mathcal{X}$이고, step마다 마스크를 다시 덮어쓰는 동작은 $\Pi_{\mathcal{K}}$의 한 형태로 해석된다.",
            r"샘플러 업데이트는 때때로 값 폭주/클리핑이 필요한 상황을 만든다(특히 높은 `cfg`, 공격적인 스케줄, SDE/ancestral에서 큰 잡음 주입). $\mathcal{K}$를 두면 “합법 범위 밖으로 나가면 다시 안으로 넣는다”를 수학적으로 한 줄로 쓰고, 안정성 논의를 ‘집합을 벗어나는가’ 문제로 바꿀 수 있다.",
            r"img2img에서 원본을 얼마나 보존할지(`denoise` 강도)를 직관적으로 말하면 “초기 상태 근처에 머물게 한다”이다. 이는 $\|x-x_{\mathrm{init}}\|\le R$ 같은 허용집합(공)이나, 그에 대한 페널티/투영으로 모델링할 수 있다. 즉 $\mathcal{K}$ 관점은 생성과 편집을 같은 언어로 묶어 준다.",
            r"현실의 편집은 한 가지 제약만 있지 않다: 마스크 고정 + 값 범위 제한 + (원한다면) 특정 통계/색감 유지 같은 복합 제약이 겹친다. 이런 경우 $\mathcal{K}=\mathcal{K}_1\cap\mathcal{K}_2\cap\cdots$로 두면 ‘어떤 제약을 어떤 순서로 적용할지’가 명시되고, 구현상의 순서(마스크 적용 후 클립 등)와 수식이 대응된다.",
            r"영상 편집에서는 프레임마다 같은 제약을 적용하거나(배경 고정), 프레임 간 제약을 추가로 거는(시간적 부드러움) 경우가 있다. 이를 전부 “허용되는 상태들의 집합”으로 보면, 문제는 결국 $\mathcal{K}$를 어떻게 설계하느냐로 내려간다.",
            r"수치해석/최적화 관점에서 ‘업데이트 후 투영’은 projected method 또는 proximal step으로 해석된다. 그래서 $\mathcal{K}$를 명시하면 단순 클립을 넘어서, 더 정교한 제약(soft/hard, convex/nonconvex)에 대한 구현 아이디어가 자연스럽게 나온다.",
        ],
        "formal": [
            r"$\mathcal{K}\subset\mathcal{X}$를 허용 집합이라 하고, 보통 닫힌 집합으로 가정한다(투영의 존재성을 위해).",
            r"$\mathcal{K}$가 닫힌 볼록집합이면 metric projection은 유일하며, $\Pi_{\mathcal{K}}(x)=\arg\min_{y\in\mathcal{K}}\|x-y\|_2$로 정의된다.",
        ],
        "formal_explain": [
            r"첫 문장에서 ‘닫힘(closed)’을 강조하는 이유는, $\Pi_{\mathcal{K}}(x)$ 같은 최소화 문제가 해를 가지려면(적어도 존재) 집합이 닫혀 있는 게 안전한 가정이기 때문이다.",
            r"둘째 문장에서 ‘볼록(convex)’을 추가하면, 투영이 거의 항상 유일해지고(최소점이 하나), 구현도 “한 번의 규칙”으로 고정하기 쉬워진다. 반대로 비볼록이면 “가장 가까운 점”이 여러 개가 될 수 있다.",
            r"mask/inpaint 제약은 $\mathcal{K}$를 ‘일부 좌표가 고정된 아핀 부분공간’으로 보는 관점이 가장 직관적이다. 그러면 투영은 고정 좌표를 덮어쓰는 연산으로 떨어진다.",
        ],
        "intuition": [
            "규칙을 어긴 상태 $x$를 ‘가장 가까운 합법 상태’로 되돌리는 영역이라고 보면 된다.",
            r"좌표 일부를 고정하는 제약은 “그 좌표만 원래 값으로 되돌린다”는 직관으로 이해하면 된다.",
            r"ComfyUI inpaint에서 “마스크 영역은 원본을 유지”는 전형적으로 $\mathcal{K}$를 ‘일부 좌표가 고정된 아핀 부분공간’으로 두는 것과 같다. 그때 투영 $\Pi_\mathcal{K}$는 고정 좌표를 덮어쓰는 한 줄 연산으로 구현된다.",
            r"코드 구현 관점에서 $\mathcal{K}$가 박스(예: $[-a,a]^d$)이면 $\Pi_\mathcal{K}$는 좌표별 `clamp`(clip)과 동일하다. 즉 “클리핑은 투영”이라는 말이 정확히 수학 정의(최단거리)로 정리된다.",
            r"제약이 비볼록(non-convex)이면 ‘가장 가까운 점’이 여러 개일 수 있어 투영이 비유일해진다. 실무에서는 보통 한 가지 규칙을 택해 구현하지만, 그 선택이 결과를 바꿀 수 있다는 점이 커스터마이징에서 중요하다.",
        ],
        "examples": [
            r"$\mathcal{X}=\mathbb{R}^2$, $\mathcal{K}=[-1,1]^2$라 하자. $x=(1.4,-0.5)$이면 $\Pi_{\mathcal{K}}(x)=(1.0,-0.5)$이다(첫 좌표만 clip).",
            r"$\mathcal{X}=\mathbb{R}^3$, $\mathcal{K}=\{(0.2,y_2,y_3):y_2,y_3\in\mathbb{R}\}$ (첫 좌표 고정)라 하자. $x=(0.9,-1.0,0.1)$이면 투영은 $(0.2,-1.0,0.1)$이다.",
            r"(마스크 투영 예) $\mathcal{X}=\mathbb{R}^4$, 고정값 $x_{\mathrm{fix}}=(1,1,0,0)$, 마스크 $m=(1,1,0,0)$라 하자(1인 좌표는 고정). 현재 $x=(0.1,0.2,0.3,0.4)$이면 투영(덮어쓰기)은 $\Pi_\mathcal{K}(x)=m\odot x_{\mathrm{fix}}+(1-m)\odot x=(1,1,0.3,0.4)$.",
            r"(볼(ball) 제약 예) $\mathcal{K}=\{x:\|x-x_{\mathrm{init}}\|_2\le 0.5\}$, $x_{\mathrm{init}}=(0,0)$라 하자. $x=(1,0)$는 제약을 깨므로 투영은 $(0.5,0)$가 된다(원점에서 반지름 0.5 원판의 경계로).",
        ],
        "walkthrough": [
            {
                "stmt": r"\mathcal{K}\subset\mathcal{X}\ \text{(closed, optionally convex)}",
                "explain": [
                    r"$\mathcal{K}$는 ‘허용되는 상태들의 모음’이다. 즉 $x\in\mathcal{K}$이면 제약을 만족한다.",
                    r"닫힘 가정은 “최소거리 점이 존재한다”는 성질을 얻기 위한 흔한 안전장치다.",
                    r"볼록성 가정은 “최소거리 점이 유일하다”는 성질을 얻기 위한 흔한 안전장치다.",
                ],
                "intuition": [
                    r"상태가 이 영역을 벗어나면, 다시 안으로 집어넣는 보정 단계가 필요하다.",
                ],
                "examples": [
                    r"$\mathcal{X}=\mathbb{R}$에서 $\mathcal{K}=[-1,1]$은 닫히고 볼록이다.",
                    r"$\mathcal{X}=\mathbb{R}^2$에서 $\mathcal{K}=\{(x_1,x_2):x_1^2+x_2^2\le 1\}$ (원판)도 닫히고 볼록이다.",
                ],
            },
            {
                "stmt": r"\Pi_{\mathcal{K}}(x)=\arg\min_{y\in\mathcal{K}}\|x-y\|_2",
                "explain": [
                    r"투영은 ‘가장 가까운 합법 점’을 고르는 연산이다. 거리의 기준은 (여기서는) 유클리드 노름 $\|\cdot\|_2$다.",
                    r"볼록이면 목적함수 $y\mapsto\|x-y\|_2^2$가 강볼록이 되어 최소점이 하나로 정해지는 경우가 많다.",
                ],
                "intuition": [
                    r"clip은 투영의 가장 쉬운 예다: 상자(box) 제약이면 좌표별로 자르면 끝이다.",
                    r"마스크 고정은 ‘해당 좌표만 덮어쓰기’로 투영이 구현된다.",
                ],
                "examples": [
                    r"$\mathcal{K}=[-1,1]$에서 $x=2.3$이면 $\Pi_{\mathcal{K}}(x)=1.0$.",
                    r"$\mathcal{K}=\{(0.2,y_2,y_3)\}$에서 $x=(0.9,-1.0,0.1)$이면 $\Pi_{\mathcal{K}}(x)=(0.2,-1.0,0.1)$.",
                ],
            },
        ],
        "code_map": [
            r"`clamp`, `mask-merge`, `inpaint overwrite` 단계가 사실상 $\Pi_{\mathcal{K}}$에 해당한다.",
        ],
        "pitfalls": [
            r"비볼록 제약에서는 ‘가장 가까운 점’이 여러 개일 수 있어 $\Pi_{\mathcal{K}}$가 다가값이 된다(구현 규약이 필요).",
        ],
    },
    "M_manifold": {
        "summary": r"상태가 $\mathbb{R}^d$ 전체가 아니라 어떤 ‘곡면/제약집합’ 위에 있어야 하는 상황을 수학적으로 쓰기 위해 $\mathcal{M}\subset\mathcal{X}$를 둔다. 예를 들어 “항상 길이가 1인 벡터” 같은 제약은 다양체(구면)로 표현된다.",
        "why": [
            r"일부 제약은 단순한 박스/선형 제약(클립, 마스크 고정)처럼 ‘집합’으로만 보기엔 구조가 더 강하다. 예를 들어 항상 $\|x\|_2=1$을 유지하는 조건은 구면 $S^{d-1}$이고, 이는 매끄러운 다양체다. 이런 제약을 자연스럽게 다루려면 $\mathcal{M}\subset\mathcal{X}$라는 다양체 관점이 필요하다.",
            r"ComfyUI 작업에서도 ‘정규화’가 은근히 자주 등장한다(예: 어떤 특징/임베딩을 단위노름으로 정규화해서 쓰거나, 특정 방향 성분을 제거하는 후처리). 이런 조작을 step마다 반복하면, 사실상 상태를 어떤 다양체 근처에 붙잡아 두는 알고리즘이 된다. 이를 수학적으로 제대로 설명하는 언어가 다양체/접공간/리트랙션이다.",
            r"유클리드 공간에서의 단순 업데이트 $x\leftarrow x+\Delta$는 곧바로 제약을 깨뜨릴 수 있다. 다양체에서는 먼저 $\Delta\in T_x\mathcal{M}$ 같은 ‘허용 방향’에서 보정 벡터를 계산하고, 그 다음 $R_x(\Delta)$로 다시 $\mathcal{M}$ 위에 올려 보내는 두 단계 구조가 표준이다. 이 구조가 정리되면 구현도 “보정 계산”과 “제약 복귀”로 분해되어 안정해진다.",
            r"수치해석 관점에서 다양체는 ‘제약을 만족하는 상태들’의 국소 좌표계를 제공한다. 그래서 고차 solver를 쓰거나, 제약을 만족하는 채로 오차를 제어하려면(예: 장기 드리프트 억제) 다양체 언어가 도움이 된다.",
            r"확산/샘플링 자체를 다양체 위에서 정의하는 연구도 많다(회전/포즈, 방향 데이터, 정규화된 표현 등). 생성·편집을 넘어 새로운 모델/샘플러를 커스터마이즈하려면 $\mathcal{M}$을 기본 도구로 갖고 있는 편이 유리하다.",
        ],
        "formal": [
            r"$\mathcal{M}\subset\mathcal{X}$를 $C^r\ (r\ge1)$ 매장(embedded) 부분다양체로 둔다.",
            r"각 점 $x\in\mathcal{M}$에서 접공간 $T_x\mathcal{M}$이 정의되고, 작은 접벡터를 다시 $\mathcal{M}$ 위로 보내는 retraction $R_x:T_x\mathcal{M}\to\mathcal{M}$를 사용할 수 있다.",
        ],
        "formal_explain": [
            r"첫 문장은 $\mathcal{M}$이 ‘잘 behaved한’ 제약집합이라는 뜻이다. $C^r$ 매장 부분다양체라는 가정은 국소적으로는 $\mathbb{R}^m$처럼 보이고(좌표계가 존재), 미분기하적 도구(접공간, 미분, 곡선 등)를 쓸 수 있게 해 준다.",
            r"둘째 문장은 계산을 어디에서 하느냐를 분리한다: (1) 접공간 $T_x\mathcal{M}$에서 선형 연산으로 보정 벡터를 계산하고, (2) 그 결과를 retraction으로 다시 다양체 위 점으로 올려 보낸다.",
            r"실무에서는 ‘진짜’ 지오데식(exp map)을 쓰기보다, 계산이 싼 retraction(예: 정규화)을 쓰는 경우가 많다. 이때도 $R_x(0)=x$와 $dR_x(0)=\mathrm{Id}$ 같은 성질 덕분에 1차 정확도는 유지된다.",
        ],
        "intuition": [
            "전체 공간(평면/공간) 안에 있는 ‘곡면 위로만 이동’한다고 생각하면 된다.",
            r"업데이트를 하더라도 매번 $\mathcal{M}$ 위로 다시 올려 보내는 보정이 필요할 수 있다.",
            r"정의역이 유클리드 공간이라도 “실제로 의미 있는 상태”가 어떤 제약식을 만족한다면, 그 제약해 집합을 $\mathcal{M}$으로 본다(예: $f(x)=0$의 해집합). 다양체(manifold)라는 가정은 그 제약이 국소적으로 매끄럽게 풀린다는 뜻이다.",
            r"ComfyUI/코드에서는 `normalize`, `orthogonalize`, “특정 성분 제거” 같은 후처리가 자주 등장하는데, 이런 연산은 상태를 어떤 곡면/부분공간 근처로 되돌리는 ‘기하학적 보정’으로 해석할 수 있다. 즉 “매 step 보정”을 수학적으로는 $x\mapsto R_x(v)$ 또는 $x\mapsto \Pi_{\mathcal{M}}(x)$로 적는다.",
        ],
        "examples": [
            r"$\mathcal{X}=\mathbb{R}^2$, $\mathcal{M}=S^1=\{x:\|x\|_2=1\}$라 하자. $(0.6,0.8)$은 $\|x\|_2=1$이므로 $\mathcal{M}$ 위의 점이다.",
            r"$\mathcal{X}=\mathbb{R}^2$, $\mathcal{M}=\{(x_1,x_2):x_1+x_2=1\}$ (직선)이라 하자. $(0.3,0.7)\in\mathcal{M}$이다.",
            r"(3차원 구면 예) $\mathcal{X}=\mathbb{R}^3$, $\mathcal{M}=S^2=\{x:\|x\|_2=1\}$라 하자. $x=(1,0,0)$과 $y=(0,0.6,0.8)$는 $\mathcal{M}$ 위의 점이다.",
            r"(곡선 제약 예) $\mathcal{X}=\mathbb{R}^2$, $\mathcal{M}=\{(u,u^2):u\in\mathbb{R}\}$ (포물선)이라 하자. $u=2$이면 점 $(2,4)\in\mathcal{M}$이고, $(2,3)$은 $\mathcal{M}$ 위가 아니다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\mathcal{M}\subset\mathcal{X}\ \text{is an embedded }C^r\text{ submanifold}",
                "explain": [
                    r"‘부분다양체’라는 말은, $\mathcal{M}$이 어떤 제약식들로 잘 정의된(매끄러운) 집합이라는 뜻이다. 그래서 점 근처에서 접평면/좌표계를 쓸 수 있다.",
                    r"예를 들어 $S^1=\{x\in\mathbb{R}^2:\|x\|_2=1\}$은 1차원 다양체(원)이고, $S^{d-1}\subset\mathbb{R}^d$는 구면이다.",
                ],
                "intuition": [
                    r"제약을 ‘곡면’이라고 생각하면 된다. 곡면 위에서만 움직이게 하려면 곡면의 기하를 써야 한다.",
                ],
                "examples": [
                    r"$\mathbb{R}^2$에서 $\mathcal{M}=\{(x_1,x_2):x_1+x_2=1\}$는 직선(1차원 다양체)이고, 예를 들어 $(0.3,0.7)$이 그 위의 점이다.",
                    r"$\mathbb{R}^2$에서 $\mathcal{M}=S^1$이면 $(0.6,0.8)$은 그 위의 점이고, $(0.6,0.7)$은 아니다(노름이 1이 아님).",
                ],
            },
            {
                "stmt": r"v\in T_x\mathcal{M}\ \xrightarrow{\ R_x\ }\ R_x(v)\in\mathcal{M}",
                "explain": [
                    r"접공간에서 계산한 보정 벡터 $v$를 실제 제약집합 위의 점으로 바꾸려면 어떤 ‘올림(map)’이 필요하다. 그 역할이 retraction $R_x$다.",
                    r"retraction은 $v$가 작을 때 $x+v$와 비슷한 효과를 내되, 결과가 $\mathcal{M}$ 위에 있도록 만들어 준다.",
                ],
                "intuition": [
                    r"“접평면에서 한 걸음 → 곡면 위로 붙이기”라는 두 단계로 생각하면 직관적이다.",
                ],
                "examples": [
                    r"$\mathcal{M}=S^1$, $x=(0.6,0.8)$, $v=(0.1,-0.2)$면 $x+v=(0.7,0.6)$이고 정규화 retraction은 $R_x(v)=\frac{(0.7,0.6)}{\|(0.7,0.6)\|}\approx(0.759,0.651)$.",
                ],
            },
        ],
        "code_map": [
            "`normalize(x)` 같은 후처리가 다양체로의 복귀 역할을 할 수 있다.",
        ],
        "pitfalls": [
            "유클리드 보정벡터를 그대로 더하면 다양체를 이탈하므로 retraction/projection이 필요하다.",
        ],
    },
    "TxM": {
        "summary": r"$T_x\mathcal{M}$은 다양체 $\mathcal{M}$ 위의 한 점 $x$에서 ‘순간적으로 움직일 수 있는 방향들’을 모아 둔 선형공간이다. 제약을 깨지 않는 업데이트 방향을 말할 때 등장한다.",
        "why": [
            r"다양체 제약 $x\in\mathcal{M}$이 있을 때, ‘다음으로 가는 방향’은 아무 벡터나 될 수 없다. 허용되는 1차 방향은 접공간 $T_x\mathcal{M}$에 들어가며, 이것이 제약을 보존하는 업데이트/벡터장을 정의하는 기본 무대다.",
            r"샘플러는 본질적으로 $x$에 어떤 보정 벡터를 더하는 수치적분이다. 제약 하에서는 보정 벡터를 먼저 $T_x\mathcal{M}$로 투영한 뒤, retraction으로 올려 보내는 방식이 표준이며(리만 최적화, constrained ODE), 이때 $T_x\mathcal{M}$이 없으면 ‘올바른 방향’이라는 말 자체가 모호해진다.",
            r"ComfyUI에서 어떤 값/특징을 ‘정규화해서 쓰자’ 같은 커스터마이징을 하면, 업데이트 후 매번 정규화하는 식으로 구현하기 쉽다. 하지만 업데이트 방향에 접공간이 아닌 성분이 많이 섞여 있으면 불필요한 왜곡이 생길 수 있다. 접공간 투영을 명시하면 왜곡을 줄이는 방향으로 설계를 바꿀 수 있다.",
            r"확률적 샘플링(SDE)에서도 노이즈 벡터를 접공간에 제한하면 상태가 제약을 덜 깨뜨린다(예: 구면 위의 등방 잡음은 접공간에서 정의). 즉 $T_x\mathcal{M}$은 결정론/확률론 모두에서 ‘허용되는 미소 변동’의 공간이다.",
            r"결국 $T_x\mathcal{M}$을 도입하는 목적은, 비선형 제약(곡면) 위 문제를 ‘선형 공간에서의 계산’으로 바꾸는 것이다. 그래서 수치 구현(벡터 연산)과 기하학적 제약을 연결하는 다리 역할을 한다.",
        ],
        "formal": [
            r"곡선 $\gamma:(-\varepsilon,\varepsilon)\to\mathcal{M}$가 $\gamma(0)=x$를 만족할 때, $\gamma'(0)$로 얻어지는 모든 벡터들의 집합(적절히 동치류로 정리한 것)이 $T_x\mathcal{M}$이다.",
            r"특히 $\mathcal{M}=S^{d-1}\subset\mathbb{R}^d$이면 $T_x\mathcal{M}=\{v\in\mathbb{R}^d:x^\top v=0\}$로 쓸 수 있다.",
        ],
        "formal_explain": [
            r"첫 문장은 접공간의 ‘정의’다. 곡면 위에서 움직이는 모든 가능한 곡선 $\gamma$를 생각하고, 그 곡선의 순간 속도 $\gamma'(0)$를 모아 만든 선형공간이 접공간이다.",
            r"두 번째 문장은 구면의 특수한 경우로, 조건 $x^\top v=0$이 접공간을 아주 간단히 표현해 준다. 즉, 구면에서 움직일 수 있는 방향은 반지름 방향($x$ 방향)과 직교해야 한다.",
            r"구현에서는 임의의 벡터 $u$를 접공간으로 ‘투영’하는 연산 $u\mapsto u-(x^\top u)x$를 자주 쓴다. 이 연산은 $x^\top(u-(x^\top u)x)=0$을 보장하기 때문이다.",
        ],
        "intuition": [
            "곡면을 확대해 보면 ‘접평면’처럼 보이는데, 그 접평면이 바로 접공간이다.",
            r"접공간에서 계산한 작은 이동 $v$를 다시 $\mathcal{M}$으로 올려 보내는 것이 retraction/정규화 같은 단계다.",
            r"코드 구현에선 “업데이트 방향을 제약에 맞게 다듬는다”는 말이 곧 $u\mapsto \Pi_{T_x\mathcal{M}}(u)$를 뜻한다. 즉 임의 벡터(예: 모델이 준 방향)를 접공간으로 투영한 뒤에만 이동시키면 제약 위반/왜곡이 줄어든다.",
            r"ComfyUI에서 정규화 보정(`normalize`)을 step마다 넣는 경우, 접공간 투영을 같이 넣으면 ‘정규화로 인해 생기는 접선 방향의 왜곡’을 줄이는 설계가 가능하다(리만 최적화의 표준 패턴).",
        ],
        "examples": [
            r"$\mathcal{M}=S^1\subset\mathbb{R}^2$, $x=(0.6,0.8)$라 하자. $v=(0.8,-0.6)$는 $x^\top v=0.6\cdot0.8+0.8\cdot(-0.6)=0$이므로 $v\in T_x\mathcal{M}$이다.",
            r"같은 점에서 $v'=(1,0)$는 $x^\top v'=0.6\ne0$이므로 접공간 벡터가 아니다(구면/원 제약을 즉시 깨는 방향).",
            r"(다른 접벡터 예) 같은 $x=(0.6,0.8)$에서 $v''=(4,-3)$도 $0.6\cdot4+0.8\cdot(-3)=2.4-2.4=0$이므로 $v''\in T_x\mathcal{M}$이다.",
            r"(포물선 접선 예) $\mathcal{M}=\{(u,u^2)\}$에서 $u=2$인 점 $(2,4)$에서의 접선 방향은 $\frac{d}{du}(u,u^2)\big|_{u=2}=(1,4)$로 볼 수 있다(기울기 4).",
        ],
        "walkthrough": [
            {
                "stmt": r"T_x\mathcal{M}=\{\gamma'(0):\gamma:(-\varepsilon,\varepsilon)\to\mathcal{M},\ \gamma(0)=x\}",
                "explain": [
                    r"곡선 $\gamma$는 다양체 위를 따라 움직이는 ‘경로’다. $\gamma(0)=x$는 그 경로가 $t=0$에서 점 $x$를 지난다는 뜻이다.",
                    r"$\gamma'(0)$는 그 순간의 속도 벡터다. 가능한 모든 경로의 가능한 모든 순간 속도를 모으면, 그게 접공간이 된다.",
                ],
                "intuition": [
                    r"곡면 위에서 아주 짧게 움직이는 건 거의 직선 운동이다. 그 ‘직선 근사’가 접공간이다.",
                ],
                "examples": [
                    r"$\mathcal{M}=S^1$, $x=(0.6,0.8)$라 하고 $\gamma(t)=(\cos(\alpha t)\cdot0.6-\sin(\alpha t)\cdot0.8,\ \sin(\alpha t)\cdot0.6+\cos(\alpha t)\cdot0.8)$처럼 원 위를 회전시키면 $\gamma(0)=x$이고 $\gamma'(0)=\alpha(-0.8,0.6)$가 접벡터가 된다.",
                ],
            },
            {
                "stmt": r"\mathcal{M}=S^{d-1}\Rightarrow T_x\mathcal{M}=\{v\in\mathbb{R}^d:x^\top v=0\}",
                "explain": [
                    r"구면은 제약식 $\|x\|_2^2=1$로 정의된다. 이를 미분하면 $2x^\top v=0$이 되어 접벡터 조건 $x^\top v=0$이 나온다.",
                    r"따라서 접공간은 ‘반지름 방향과 직교한 모든 방향’의 집합이다.",
                ],
                "intuition": [
                    r"구면 위에서 앞으로 움직이면, 순간적으로는 반지름 방향이 아니라 ‘옆으로’ 움직인다.",
                ],
                "examples": [
                    r"$x=(0.6,0.8)$에서 $v=(0.8,-0.6)$는 $x^\top v=0$이므로 접벡터다.",
                    r"$u=(1,0)$를 접공간으로 투영하면 $u-(x^\top u)x=(1,0)-0.6(0.6,0.8)=(0.64,-0.48)$이고 실제로 $x^\top(0.64,-0.48)=0$이다.",
                ],
            },
        ],
        "code_map": [
            "보정 벡터를 `v = v - (x^T v)x`처럼 직교화하면 접공간 투영과 같다.",
        ],
        "pitfalls": [
            "접공간을 무시하면 제약 위상에서 장기 drift가 생긴다.",
        ],
    },
    "S_map": {
        "summary": r"step index $k=0,1,\dots,N$을 노이즈 스케일 $\sigma_k$로 바꿔 주는 함수(또는 룩업 테이블)를 $S$라고 쓴다. 코드에서는 `sigmas[k]`를 돌려주는 것과 같다.",
        "why": [
            r"ComfyUI에서 `scheduler` 선택은 곧 “step 인덱스 $k$를 어떤 $\sigma_k$로 매핑할지”를 고르는 일이다. 같은 `sampler_name`이라도 `scheduler`를 바꾸면 결과가 크게 달라지는 이유는, 실제로 모델을 평가하는 지점들이 바뀌기 때문이다.",
            r"수치해석 관점에서 $S$는 적분 그리드(mesh)다. Euler/Heun/DPM++ 같은 solver가 동일해도, 그리드가 달라지면 국소 오차가 어느 구간에 집중되는지(전역 구조 vs 미세 디테일)가 달라진다. 그래서 실무에서는 sampler와 scheduler를 분리된 옵션이 아니라 ‘한 쌍’으로 보는 편이 맞다.",
            r"`steps`는 정의역의 크기 $N$을 정하고, $S$는 각 step의 실질적인 step-size(예: $\Delta\sigma=\sigma_{k+1}-\sigma_k$)를 정한다. 따라서 “steps를 늘렸는데도 품질이 크게 안 좋아진다/오히려 나빠진다” 같은 현상도 $S$의 배치가 원인일 수 있다.",
            r"img2img/inpaint에서 `denoise`나 구간 옵션을 쓰면, 사실상 $S$의 일부만 사용한다(큰 $\sigma$ 구간을 생략하거나, 특정 구간만 반복). 이때 어떤 구간을 생략하느냐가 원본 보존/변형 정도를 결정하므로, $S$를 명시적으로 보는 습관이 편집 설계에 도움이 된다.",
            r"논문/코드에서는 시간 $t$와 $\sigma$를 오가며 표기하는데, 실제 구현은 결국 $k\mapsto$ (모델이 요구하는 스케일 변수) 변환이다. $S$를 중심에 두면 “우리 구현은 $\sigma$-매개변수인가, $t$-매개변수인가”가 혼동되지 않는다.",
            r"영상 생성에서는 프레임별로 동일한 $S$를 쓰는지, 혹은 일부 프레임에만 다른 구간을 쓰는지(예: 키프레임만 더 낮은 $\sigma$까지) 같은 설계가 생긴다. 이 역시 $S$를 조작하는 문제로 정리된다.",
        ],
        "formal": [
            r"$S:\{0,\dots,N\}\to\Sigma,\ k\mapsto\sigma_k$.",
            r"대부분 단조감소 $\sigma_{k+1}\le\sigma_k$를 가정한다.",
        ],
        "formal_explain": [
            r"첫 문장은 스케줄이 “유한한 룩업 테이블”이라는 뜻이다. 실제 구현에서는 길이 $N+1$인 배열 `sigmas`가 있고, $S(k)$는 그 배열의 $k$번째 원소다.",
            r"둘째 문장의 단조감소 가정은 ‘노이즈를 줄이며 정제한다’는 해석과 맞아떨어진다. 단조가 깨지면(중간에 $\sigma$가 올라가면) solver가 의도하지 않은 흔들림을 만들 수 있다.",
            r"수치해석 관점에서는, 같은 solver라 해도 mesh(그리드) 선택이 달라지면 오차가 어느 구간에 분배되는지가 바뀐다. 그래서 스케줄러를 ‘solver의 일부’로 보는 게 실무적으로 맞다.",
        ],
        "intuition": [
            r"‘$N$번 호출을 어떤 $\sigma$ 위치에 배치할지’를 정하는 시간표다.",
            r"같은 solver라도 스케줄을 바꾸면 체감 품질이 크게 바뀌는 이유가 여기에 있다.",
            r"ComfyUI에서는 `steps`가 정의역 크기 $N$을, `scheduler`가 $S$의 구체적인 형태를 정한다. 즉 “sampler는 유지하고 scheduler만 바꾼다”는 말은 $\Phi_k$는 고정하고 $\sigma_k=S(k)$만 바꾸는 것이다.",
            r"코드 구현 관점에선 $S$는 단순 배열 룩업(`sigmas[k]`)이지만, 수치해석 관점에선 $S$가 곧 step-size/오차/안정성을 좌우한다. 그래서 커스터마이징의 1순위가 되는 경우가 많다.",
        ],
        "examples": [
            r"$N=4$일 때 한 예로 $S(0)=10, S(1)=5, S(2)=2, S(3)=1, S(4)=0$처럼 둘 수 있다(즉 `sigmas=[10,5,2,1,0]`).",
            r"다른 예로 `sigmas=[10,4,1,0.2,0]`처럼 작은 $\sigma$ 쪽을 더 촘촘히 둘 수도 있다.",
            r"(ComfyUI img2img 예) `steps=4`, `sigmas=[10,5,2,1,0]`라고 하자. `denoise=0.5`를 “앞 절반 생략”으로 단순화하면 시작점을 $\sigma=5$로 잡아 $(5,2,1,0)$만 따라가는 것과 비슷하다(정확한 시작 인덱스는 구현에 따라 다를 수 있음).",
            r"(룩업 예) `sigmas=[10,4,1,0.2,0]`면 $S(2)=\sigma_2=1$이다. 즉 step 번호는 인덱스이고, 실제 물리량은 $\sigma_k$다.",
        ],
        "walkthrough": [
            {
                "stmt": r"S:\{0,\dots,N\}\to\Sigma,\qquad S(k)=\sigma_k",
                "explain": [
                    r"정의역 $\{0,\dots,N\}$는 이산 step 인덱스이고, 공역 $\Sigma$는 노이즈 스케일 값공간이다.",
                    r"즉 스케줄은 ‘인덱스 → 실수값’ 변환이다. 구현에서는 단순 배열 조회다.",
                ],
                "intuition": [
                    r"샘플러가 모델을 언제(어떤 $\sigma$에서) 평가하는지의 타임라인이다.",
                ],
                "examples": [
                    r"`sigmas=[10,5,2,1,0]`면 $S(2)=2$.",
                ],
            },
            {
                "stmt": r"\sigma_{k+1}\le\sigma_k",
                "explain": [
                    r"단조감소는 “점점 노이즈를 줄인다”는 뜻이다. 대부분의 diffusion sampling은 큰 노이즈에서 작은 노이즈로 내려가는 흐름을 전제한다.",
                    r"단조감소가 깨지면 어떤 step에서는 노이즈가 다시 커져, solver의 안정성/품질이 떨어질 수 있다(특히 저 step 수에서).",
                ],
                "intuition": [
                    r"사진을 점점 선명하게 만드는 과정에서, 중간에 다시 흐리게 만들면 이상해진다.",
                ],
                "examples": [
                    r"`sigmas=[10,4,1,0.2,0]`는 단조감소다.",
                    r"`sigmas=[10,4,6,1,0]`는 $4\to6$에서 단조감소가 깨진 예다(보통 의도하지 않음).",
                ],
            },
        ],
        "code_map": [
            "`get_sigmas_*` 함수들이 $S$를 구현한다.",
        ],
        "pitfalls": [
            "solver만 바꾸고 스케줄을 고정하면 기대한 차수/안정성 향상이 안 나올 수 있다.",
        ],
    },
    "D_theta": {
        "summary": r"신경망(예: U-Net)이 구현하는 함수다. 현재 상태 $x$, 노이즈 수준 $\sigma$, 조건 $c$를 넣으면 같은 차원의 벡터(또는 ‘denoised’ 추정치)를 반환한다. 샘플러는 이 함수를 step마다 여러 번 호출해서 업데이트 방향을 만든다.",
        "why": [
            r"ComfyUI에서 sampler는 본질적으로 `model(x, sigma, cond)`를 여러 번 호출하는 루프다. `model` 객체 안에는 U-Net뿐 아니라 CFG/컨트롤/LoRA/추가 모듈이 래핑되어 들어갈 수 있지만, 외부에서 보면 결국 하나의 함수 $D_\theta$로 행동한다. 이 관점을 잡아야 “샘플러를 바꾼다”와 “모델을 바꾼다”를 분리해서 사고할 수 있다.",
            r"Euler/Heun/DPM++/LMS 등 다양한 sampler는 수식은 달라도, 대부분 ‘한 step에서 $D_\theta$를 몇 번 평가하고 어떻게 조합하느냐’로 요약된다. 그래서 같은 `steps`라도 sampler마다 속도(모델 호출 수)와 품질(근사 차수)이 달라진다. $D_\theta$를 중심에 두면 이 차이가 명확해진다.",
            r"실무에서 가장 흔한 버그는 ‘모델 출력의 의미’를 잘못 해석하는 것이다. 어떤 모델은 $\epsilon$을, 어떤 모델은 $x_0$를, 어떤 모델은 $v$를 예측한다. ComfyUI는 내부에서 이런 파라미터화를 맞추는 래퍼를 두지만, 커스터마이징(자체 샘플러/모델 래핑)을 하려면 $D_\theta$가 무엇을 반환한다고 가정하는지부터 고정해야 한다.",
            r"CFG는 $D_\theta(x,\sigma,c)$와 $D_\theta(x,\sigma,c_{\emptyset})$를 둘 다 평가한 뒤 `cfg` 스케일로 섞는다. 즉 한 step의 비용과 동작이 조건 설계에 직접 의존한다. 프롬프트/컨트롤을 복잡하게 만들수록 ‘한 번의 $D_\theta$ 호출이 의미하는 것’이 중요해지고, 타입/입력 구조를 명확히 해 두면 디버깅이 쉬워진다.",
            r"커스텀 sampler를 만들 때는 $D_\theta$ 출력으로부터 drift/score/velocity 등을 계산해 적분한다. 이 과정에서 필요한 것은 “입력 $(x,\sigma,c)$에 대해 출력이 어떤 공간의 원소인가”라는 타입과, “출력 물리량이 무엇인가”라는 해석이다. 둘 다 $D_\theta$ 정의에 포함된다.",
            r"영상/조건 스케줄링에서는 step 또는 프레임에 따라 조건을 바꾸는 경우가 많다. 이때도 sampler 본체는 그대로 두고 $c$만 바꾸면 되도록 인터페이스를 설계하는 게 보통 유리하다. 결국 $D_\theta$를 ‘조건을 포함한 함수’로 보는 관점이 이런 확장을 가능하게 한다.",
        ],
        "formal": [
            r"$D_\theta:(\mathcal{X}\times\Sigma\times\mathcal{C})\to\mathcal{X}$.",
            r"수치해석에서는 보통 $\mathcal{X}$에서 국소 Lipschitz(또는 적절한 성장 조건)를 가정하고 step 안정성/오차를 논한다.",
            r"모델이 무엇을 예측하는지에 따라 $D_\theta$를 ‘denoised $\hat{x}_0$’, ‘score’, ‘velocity’ 등으로 해석한다(같은 출력도 스케일 변환이 다름).",
        ],
        "formal_explain": [
            r"첫 문장은 $D_\theta$가 ‘입력 3개를 받아 출력 1개를 내는 함수’라는 타입 선언이다. 이 타입이 맞으면 sampler가 어떤 식으로 조합하더라도 최소한 계산이 성립한다.",
            r"둘째 문장은 수치해석적 안전장치다. solver는 $D_\theta$를 여러 점에서 평가해 선형 결합을 만들기 때문에, $D_\theta$가 너무 요동치면(비연속/폭주) step이 불안정해진다. 그래서 보통 국소 Lipschitz 같은 가정을 깔고 논의를 시작한다.",
            r"셋째 문장은 실무에서 가장 자주 터지는 함정이다. 어떤 코드는 $D_\theta$가 $x_0$를 예측한다고 가정하고, 어떤 코드는 $\epsilon$을 예측한다고 가정한다. 같은 네트워크 출력이라도 ‘어떤 물리량으로 해석하느냐’가 달라지면 업데이트 식의 계수/스케일이 달라진다.",
        ],
        "intuition": [
            r"$D_\theta$는 ‘지금 상태가 너무 노이즈가 섞여 있는데, 깨끗한 쪽이 어디인지’ 힌트를 주는 함수라고 보면 된다.",
            r"반환값 자체가 곧바로 다음 $x$가 되는 것이 아니라, 보통은 그 반환값으로부터 drift/방향 벡터를 만들어 한 step 이동한다.",
            r"ComfyUI에서 sampler는 매 step마다 같은 형태로 $D_\theta$를 호출한다. 그래서 “sampler의 차이”는 대부분 (1) $D_\theta$를 몇 번 평가하는지, (2) 그 평가값을 어떻게 조합하는지의 차이다(수치해석 관점의 적분기 선택).",
            r"모델 파라미터화(parameterization)가 달라도 샘플러가 일관되게 동작하려면, 내부에서 출력 의미를 맞추는 변환이 필요하다. 예를 들어 $\epsilon$-예측이면 $\hat{x}_0=x-\sigma\hat\epsilon$로 바꿔 denoised를 만들 수 있고, $v$-예측도 적절한 선형변환으로 환산한다.",
            r"코드 구현에서 가장 중요한 불변식은 “입력과 출력 텐서의 shape가 같다”는 점이다. 즉 `model(x, sigma, cond)`는 보통 `x`와 같은 shape의 텐서를 반환해야 샘플러 업데이트가 닫힌다(closure).",
        ],
        "examples": [
            r"(숫자는 예시) $\mathcal{X}=\mathbb{R}^2$라 하고 $x=(0.5,-1.2)$, $\sigma=1.0$, $c=(1.5,-0.3)$를 넣었더니 $D_\theta(x,\sigma,c)=(0.42,-1.05)$ 같은 벡터가 나온다고 하자. 출력도 $\mathbb{R}^2$의 원소다.",
            r"또 다른 호출에서 $x=(1.0,0.0)$, $\sigma=5.0$, $c=(0.0,1.0)$일 때 $D_\theta(x,\sigma,c)=(0.2,-0.1)$처럼 전혀 다른 값이 나올 수 있다(조건/노이즈에 따라 달라짐).",
            r"($\epsilon$-예측 수치 예) 모델이 $\hat\epsilon$을 예측한다고 하자. $x=(1.0,-0.5)$, $\sigma=2.0$, $\hat\epsilon=(0.3,-0.1)$이면 denoised 추정치는 $\hat{x}_0=x-\sigma\hat\epsilon=(1.0,-0.5)-2.0(0.3,-0.1)=(0.4,-0.3)$이다.",
            r"(shape 예) `x.shape=(1,4,64,64)`인 latent를 넣으면 $D_\theta$의 출력도 보통 같은 `shape=(1,4,64,64)`다. 예를 들어 어떤 위치의 성분이 $x[0,0,0,0]=0.12$였다면 출력도 같은 인덱스에 실수 값을 준다(예: 0.08).",
            r"(CFG 연산 예) $D_\theta(x,\sigma,c)=(1,2)$, $D_\theta(x,\sigma,c_\emptyset)=(0.2,1.5)$, `cfg=2`면 guided 출력은 $(0.2,1.5)+2((1,2)-(0.2,1.5))=(1.8,2.5)$다(조건 차이를 적당히 증폭).",
        ],
        "walkthrough": [
            {
                "stmt": r"D_\theta:\mathcal{X}\times\Sigma\times\mathcal{C}\to\mathcal{X}",
                "explain": [
                    r"입력 $(x,\sigma,c)$에서 $x\in\mathcal{X}$는 현재 상태, $\sigma\in\Sigma$는 노이즈 수준, $c\in\mathcal{C}$는 조건이다.",
                    r"출력은 다시 $\mathcal{X}$의 원소다. 즉 상태와 같은 차원의 벡터가 나온다(예: denoised 추정치나 score/velocity).",
                    r"샘플러는 이 출력을 그대로 쓰지 않고, 보통 ‘방향 벡터’로 변환(to\_d 등)한 뒤 적분한다.",
                ],
                "intuition": [
                    r"모델 호출 한 번은 “지금 위치에서 나침반을 한 번 읽는다”에 가깝다. 2-stage solver는 중간 위치에서도 한 번 더 읽는다.",
                ],
                "examples": [
                    r"$\mathcal{X}=\mathbb{R}^2$에서 $D_\theta((0.5,-1.2),1.0,(1.5,-0.3))=(0.42,-1.05)$ 같은 출력이 나올 수 있다.",
                ],
            },
            {
                "stmt": r"\hat{x}_0=D_\theta(x,\sigma,c)\quad(\text{one possible parameterization})",
                "explain": [
                    r"$D_\theta$가 ‘깨끗한 샘플 $\hat{x}_0$’를 직접 예측한다고 해석하는 경우가 많다. 이때 샘플러는 $x$와 $\hat{x}_0$의 차이를 이용해 drift를 만든다.",
                    r"반대로 $\epsilon$-pred, v-pred에서는 $D_\theta$ 출력이 다른 물리량이므로, $\hat{x}_0$나 score로 바꾸는 변환이 필요하다.",
                ],
                "intuition": [
                    r"“모델이 무엇을 출력한다고 가정했는가”가 solver 식의 계수를 결정한다.",
                ],
                "examples": [
                    r"$x=(0.5,-1.2)$, $\hat{x}_0=(0.42,-1.05)$이면 차이는 $\hat{x}_0-x=(-0.08,0.15)$이다. 많은 업데이트 식은 이 차이를 스케일링해서 이동량으로 쓴다.",
                ],
            },
        ],
        "code_map": [
            "`model(x, sigma, cond)` 호출 결과가 직접 drift/보정항 계산에 들어간다.",
        ],
        "pitfalls": [
            r"파라미터화($\epsilon$-pred, $x_0$-pred, v-pred) 차이를 무시하면 업데이트 식이 달라진다.",
        ],
    },
    "b_theta": {
        "summary": r"연속시간 관점에서 샘플링을 ODE/SDE로 보면, $b_\theta$는 상태가 평균적으로 어떻게 변하는지(결정론적 변화율)를 주는 벡터장이다. 수치해석 관점에선 ‘적분해야 하는 미분방정식의 우변’이다.",
        "why": [
            r"ComfyUI에서 `sampler_name`을 바꾸는 것은 결국 “어떤 연속 동역학을 어떤 수치적분기로 근사할 것인가”를 바꾸는 것과 가깝다. 이때 연속 동역학의 ‘평균적인 이동 방향’이 바로 drift $b_\theta$이고, Euler/Heun/DPM 계열은 이를 이산 step으로 적분하는 방식으로 해석할 수 있다.",
            r"결정론 sampler(예: Euler, Heun, DPM++(non-ancestral) 계열)는 ‘노이즈를 새로 뽑지 않고’ 하나의 경로를 따라 내려오는 느낌이 강하다. 이런 성질은 $g\equiv 0$인 probability-flow ODE 관점에서 자연스럽고, 그 ODE의 우변이 $b_\theta$다. 즉 재현성(같은 seed면 같은 결과)을 설명하는 핵심 기호가 $b_\theta$다.",
            r"CFG/컨트롤은 “어느 방향으로 움직일지” 자체를 바꾼다. 수식으로는 $b_\theta(x,t,c)$에서 조건 $c$를 바꾸거나, $b$를 조건부/비조건부 출력의 조합으로 바꾸는 것과 같다. 그래서 `cfg`를 올리면 경로가 더 강하게 조건 쪽으로 휘는 이유를 drift 관점에서 설명할 수 있다.",
            r"SDE/ancestral 계열에서는 한 step이 drift(평균 이동)와 diffusion(무작위 흔들림)의 합으로 이루어진다. 이때 결과의 ‘형태(구성)’는 대체로 drift가, ‘질감/다양성’은 diffusion이 크게 좌우한다. $b_\theta$와 $g$를 분리해 두면 `eta/s_noise/s_churn` 같은 파라미터가 어느 항을 건드리는지 명확해져 튜닝이 쉬워진다.",
            r"영상 생성에서는 프레임 간 일관성이 중요한데, drift가 만들어 내는 구조적 이동과 diffusion이 만들어 내는 프레임별 랜덤 질감이 충돌하면 깜빡임이 생긴다. 따라서 “구조는 유지하고 질감만 조절” 같은 실무 목표도 결국 $b_\theta$와 $g$의 역할 분리로 정리된다.",
        ],
        "formal": [
            r"$b_\theta:\mathcal{X}\times[0,1]\times\mathcal{C}\to\mathcal{X}$.",
            r"ODE는 $\dot{x}_t=b_\theta(x_t,t,c)$로 쓴다.",
        ],
        "formal_explain": [
            r"첫 문장은 $b_\theta$의 타입 선언이다. 시간 변수 $t\in[0,1]$는 ‘연속시간 매개변수’이며, 구현에서는 종종 $\sigma$로 재매개화되어 나타난다(스케줄러가 하는 일).",
            r"둘째 문장은 ‘확률이 없는 경우’(probability-flow ODE 같은 결정론 경로)에서 상태가 어떻게 움직이는지의 정의다. 수치해석에서는 결국 이 ODE를 이산 step으로 적분한다.",
            r"실무에서는 $b_\theta$를 신경망이 직접 주는 경우는 드물고, 보통 $D_\theta$의 출력(예: $\hat{x}_0$)을 적절히 스케일링해 drift 형태로 바꾼다(`to_d` 같은 변환).",
        ],
        "intuition": [
            r"노이즈가 없다고 생각하면 $b_\theta$가 가리키는 방향으로 상태가 흘러간다.",
            r"샘플러는 이 흐름을 이산 step으로 잘라 근사하는 것이라 볼 수 있다.",
            r"ComfyUI의 결정론 sampler(euler/heun/lms/dpmpp_2m 등)는 ‘어떤 ODE를 적분한다’고 볼 수 있고, 그 ODE의 우변(벡터장)이 $b_\theta$다. 즉 sampler 커스터마이징은 사실상 $b_\theta$의 근사 방식(수치적분기)을 바꾸는 일이다.",
            r"코드에서는 보통 $D_\theta$ 출력(예: denoised $\hat{x}_0$)으로부터 $d=(x-\hat{x}_0)/\sigma$ 같은 ‘미분량’을 만들고, 이를 drift 근사로 쓴다. 즉 $b_\theta$는 신경망 출력 그 자체라기보다 “신경망 출력으로부터 구성된 벡터장”인 경우가 많다.",
        ],
        "examples": [
            r"(1차원 예) $b_\theta(x,t,c)=-x$라고 하자. $x=2.0$, $\Delta t=0.1$이면 Euler 한 step은 $x_{\text{new}}=2.0+0.1\cdot(-2.0)=1.8$이다.",
            r"(2차원 예) $b_\theta(x,t,c)=(-x_1,-x_2)$라면 $x=(1.0,-0.5)$에서 $b_\theta(x,\cdot,\cdot)=(-1.0,0.5)$이다.",
            r"(denoised→drift 예) $\sigma=2.0$, $x=(1.0,-0.5)$, $\hat{x}_0=(0.4,-0.3)$라 하자. 그러면 $d=(x-\hat{x}_0)/\sigma=((0.6,-0.2))/2=(0.3,-0.1)$이고, 많은 샘플러는 이 $d$를 ‘$\sigma$-축에서의 변화율’로 사용한다(표기만 다를 뿐 drift 역할).",
            r"(step-size 영향) 같은 $b(x)=-x$에서 $x_0=2.0$라도 $\Delta t=0.05$면 $x_1=1.9$, $\Delta t=0.5$면 $x_1=1.0$으로 이동량이 크게 달라진다. ComfyUI에서 `steps`/스케줄이 결과를 바꾸는 이유가 여기에 있다.",
        ],
        "walkthrough": [
            {
                "stmt": r"b_\theta:\mathcal{X}\times[0,1]\times\mathcal{C}\to\mathcal{X}",
                "explain": [
                    r"입력 $(x,t,c)$에서 $x$는 상태, $t$는 연속시간, $c$는 조건이다. 출력은 같은 공간 $\mathcal{X}$의 벡터(변화율)다.",
                    r"즉 $b_\theta(x,t,c)$는 “그 상태에서의 순간 속도”를 의미한다.",
                ],
                "intuition": [
                    r"‘바람(벡터장)’이 불면 점이 이동한다. $b_\theta$는 그 바람이다.",
                ],
                "examples": [
                    r"$\mathcal{X}=\mathbb{R}^2$에서 $b_\theta(x,t,c)=(-x_1,-x_2)$라면 $x=(1.0,-0.5)$에서 $b_\theta=(-1.0,0.5)$.",
                ],
            },
            {
                "stmt": r"\dot{x}_t=b_\theta(x_t,t,c)",
                "explain": [
                    r"이 미분방정식은 “아주 작은 시간 $\Delta t$ 동안 $x$가 $b_\theta$ 방향으로 $\Delta t$만큼 움직인다”는 뜻이다.",
                    r"Euler 적분은 $x_{k+1}=x_k+\Delta t_k\,b_\theta(x_k,t_k,c)$로 이를 근사한다. Heun은 중간점에서도 $b_\theta$를 한 번 더 평가해 2차 정확도를 얻는다.",
                ],
                "intuition": [
                    r"모든 sampler는 결국 이런 ‘연속 흐름’을 이산화한 것이라고 생각하면, 이름이 달라도 공통 구조가 보인다.",
                ],
                "examples": [
                    r"$b(x)=-x$, $x_0=2.0$, $\Delta t=0.1$이면 $x_1=1.8$ (위와 동일). $\Delta t=0.5$이면 $x_1=1.0$로 더 크게 이동한다.",
                ],
            },
        ],
        "code_map": [
            "코드에서는 종종 `d = to_d(x, sigma, denoised)`가 drift 근사량 역할을 한다.",
        ],
        "pitfalls": [
            "모델 출력을 drift로 변환하는 스케일 계수를 놓치면 전혀 다른 동역학이 된다.",
        ],
    },
    "g_func": {
        "summary": r"SDE에서 ‘난수 항’의 크기를 정하는 함수다. 같은 drift라도 $g$가 크면 경로가 더 흔들리고(분산 증가), $g=0$이면 완전히 결정론적 ODE로 바뀐다.",
        "why": [
            r"ComfyUI에서 `_ancestral`, `_sde`가 붙은 sampler나 `er_sde` 같은 계열은 step마다 ‘추가 노이즈’를 주입한다. 이 추가 노이즈의 크기를 연속시간 SDE로 표현하면 diffusion 계수 $g(t)$가 된다. 즉 “왜 ancestral이 질감이 더 거칠고 변동이 큰가”는 $g$가 0이 아니기 때문이다.",
            r"같은 drift라도 $g$를 키우면 경로가 더 흔들리고 결과 분포가 더 퍼진다(다양성↑, 그레인/랜덤 텍스처↑). 반대로 $g$를 줄이면 경로가 더 결정론적으로 되며(다양성↓), 특히 영상에서는 프레임 간 일관성이 좋아지는 경향이 있다. 실무의 ‘질감 vs 안정성’ 트레이드오프를 한 변수로 잡아 주는 게 $g$다.",
            r"이산화하면 노이즈 항은 대략 $g(t_k)\sqrt{\Delta t_k}\,\xi_k$ 꼴이므로, 같은 $g$라도 step-size가 크면(=적은 step) 한 번에 들어가는 랜덤 흔들림이 커진다. 그래서 `steps`/스케줄과 SDE 노이즈 파라미터(`eta/s_noise/s_churn`)가 서로 얽혀 동작한다.",
            r"일부 sampler는 $\sigma$를 잠깐 ‘올려서’ 탐색을 넓히는(churn) 방식으로 stochasticity를 추가한다. 이런 조작도 연속시간 관점에선 $g$를 구간적으로 키우는 것과 비슷한 효과로 해석할 수 있어, “어느 구간에서 랜덤성을 늘릴지” 설계를 수학적으로 말할 수 있다.",
            r"커스텀 sampler를 만들 때 가장 중요한 설계 중 하나가 “언제 새 노이즈를 뽑을지, 얼마나 섞을지”다. 이를 $g$로 분리해 두면 drift 설계와 독립적으로 stochastic 설계를 바꿀 수 있고, ComfyUI 파라미터 이름으로도 자연스럽게 매핑된다(`eta`, `s_noise`, `noise_scaler` 등).",
        ],
        "formal": [
            r"$g:[0,1]\to[0,\infty)$.",
            r"SDE는 $dX_t=b_\theta(X_t,t,c)\,dt+g(t)\,dW_t$ 형태다.",
        ],
        "formal_explain": [
            r"첫 문장은 $g$가 시간에 따라 변할 수 있는 ‘스칼라 함수’라는 뜻이다. (일반 SDE에서는 행렬도 가능하지만, 여기서는 단순화를 위해 스칼라로 적는다.)",
            r"둘째 문장에서 $W_t$는 브라운 운동이다. 따라서 $dW_t$는 평균 0, 분산 $\Delta t$를 갖는 무작위 증분으로 생각할 수 있다.",
            r"그래서 이산화하면 한 step에서 노이즈 항이 대략 $g(t_k)\sqrt{\Delta t_k}\,\xi_k$ 꼴로 나타난다($\xi_k\sim\mathcal{N}(0,1)$). 이 항이 질감/다양성(분산)을 좌우한다.",
        ],
        "intuition": [
            r"$b_\theta$는 ‘평균 이동’, $g$는 ‘랜덤 흔들림의 세기’다.",
            r"같은 $b_\theta$라도 $g$를 키우면 샘플이 더 다양해지고, 줄이면 더 결정론적으로 보인다.",
            r"ComfyUI에서 ancestral/SDE 계열의 `eta`, `s_noise`, `s_churn`은 구현 디테일은 달라도 결국 “추가 노이즈 항의 크기”를 조절하는 손잡이로 작동한다. 즉 사용자가 체감하는 ‘거칠어짐/다양성’ 변화는 $g$가 커졌다고 보면 된다.",
        ],
        "examples": [
            r"$g(t)=0$이면 확률항이 사라져 ODE만 남는다(완전 결정론).",
            r"예: $g(t)=0.5$, $\Delta t=0.04$이면 $g\sqrt{\Delta t}=0.5\cdot0.2=0.1$. 표준정규 난수 $\xi=-0.73$을 뽑으면 노이즈 항은 $0.1\cdot(-0.73)=-0.073$ 정도 크기로 더해진다.",
            r"(벡터 노이즈 예) $d=2$에서 $\xi=(0.5,-0.2)$, 스케일이 0.4이면 주입 노이즈는 $0.4\,\xi=(0.2,-0.08)$이다. 코드에선 `x = x + 0.4*randn_like(x)` 같은 한 줄로 구현된다.",
        ],
        "walkthrough": [
            {
                "stmt": r"g:[0,1]\to[0,\infty)",
                "explain": [
                    r"시간 $t$에 따라 ‘노이즈의 세기’를 정하는 함수다. $g(t)$가 0이면 그 순간에는 노이즈를 주입하지 않는다.",
                    r"일반 이론에서는 $g(t)$가 벡터/행렬일 수도 있지만(비등방성 확산), 많은 diffusion sampling 표기에서는 스칼라 계수로 정리해 쓴다.",
                ],
                "intuition": [
                    r"노이즈 볼륨을 조절하는 손잡이로 보면 된다.",
                ],
                "examples": [
                    r"예: $g(t)=0$ (항상 결정론).",
                    r"예: $g(t)=0.2+0.8t$ (시간이 갈수록 노이즈가 커짐).",
                ],
            },
            {
                "stmt": r"dX_t=b_\theta(X_t,t,c)\,dt+g(t)\,dW_t",
                "explain": [
                    r"첫 항 $b_\theta\,dt$는 평균 이동, 둘째 항 $g\,dW_t$는 무작위 이동이다.",
                    r"이산화하면 $X_{k+1}\approx X_k+b_\theta(X_k,t_k,c)\Delta t_k+g(t_k)\sqrt{\Delta t_k}\,\xi_k$ 꼴이 된다.",
                ],
                "intuition": [
                    r"같은 drift라도 노이즈를 더 섞으면 경로가 더 흔들리고, 결과 분포가 더 퍼진다.",
                ],
                "examples": [
                    r"1차원에서 $b(x)=-x$, $g=0.5$, $x_k=1.0$, $\Delta t=0.04$, $\xi_k=-0.73$이면 $x_{k+1}\approx 1.0+(-1.0)\cdot0.04+0.5\cdot0.2\cdot(-0.73)=1.0-0.04-0.073=0.887$.",
                ],
            },
        ],
        "code_map": [
            "`eta`, `s_noise`, `noise_scaler` 등이 이 계수의 이산 근사에 해당한다.",
        ],
        "pitfalls": [
            "drift를 잘 맞춰도 $g$가 크면 샘플 분산이 커져 품질이 흔들릴 수 있다.",
        ],
    },
    "Phi_k": {
        "summary": r"한 번의 step 업데이트 규칙 자체를 $\Phi_k$로 적는다. 즉, 코드에서 ‘한 반복’에 해당하는 함수(또는 연산자)를 수학 기호로 이름 붙인 것이다.",
        "why": [
            r"ComfyUI `KSampler`의 내부를 한 줄로 쓰면 결국 `x = step(x, sigma_k, ...)`의 반복이다. 이 ‘한 번의 반복’을 수학적으로 $\Phi_k$로 이름 붙이면, UI에서 보이는 다양한 sampler가 사실은 $\Phi_k$의 정의만 다를 뿐이라는 점이 드러난다.",
            r"sampler를 비교할 때 가장 헷갈리는 부분은 “모델을 몇 번 호출하나”, “중간점에서 무엇을 평가하나”, “노이즈를 언제 뽑아 섞나” 같은 구현 디테일이다. $\Phi_k$를 쓰면 이 모든 차이는 ‘$\Phi_k$의 내부’로 모이고, 외부에서는 $x_{k+1}=\Phi_k(\cdot)$ 한 문장으로 통일된다.",
            r"재현성도 $\Phi_k$로 깔끔하게 정리된다. 결정론 sampler는 $\omega_k$가 필요 없고(또는 $\Omega_k$가 한 점 공간), stochastic sampler는 $\omega_k$가 필요하다. ComfyUI에서 `seed`를 고정했는데도 sampler에 따라 결과가 조금씩 달라지는 이유가 바로 $\omega_k$의 유무/사용 방식 차이다.",
            r"사용자 파라미터가 어디에 들어가는지도 $\Phi_k$로 보면 명확해진다. `scheduler`는 $\sigma_k$를 제공하고, `cfg`는 $D_\theta$ 호출/조합 방식을 바꾸며, `eta/s_noise`는 $\omega_k$가 섞이는 스케일을 바꾼다. 커스터마이징할 때 “무엇을 건드리면 어떤 성질이 바뀌는가”를 설명하는 공용 언어가 된다.",
            r"영상/다단계 solver에서도 아이디어는 같다: 상태를 확장해 $\Phi_k$의 입력을 키우면(예: $h_k$ 포함, 여러 프레임 포함) 여전히 같은 형태의 반복으로 표현된다. 그래서 새로운 sampler를 설계할 때도 $\Phi_k$를 먼저 정하고 나머지를 붙이는 방식이 가장 깔끔하다.",
        ],
        "formal": [
            r"$\Phi_k:(\mathcal{X}\times\mathcal{H}_k\times\Omega_k)\to\mathcal{X}$.",
            r"적응성: $\Phi_k$는 시점 $k$에서 이용 가능한 정보에만 의존(비예견성)한다.",
        ],
        "formal_explain": [
            r"첫 문장은 ‘step 함수의 타입’을 적는다. 입력은 (현재 상태 $x_k$, 과거 버퍼 $h_k$, 그리고 난수 접두 $\omega_k$)이고 출력은 다음 상태 $x_{k+1}$다.",
            r"여기서 $\Omega_k$는 “이번 step에서 새로 쓰는 난수”를 담는 공간이라고 보면 된다(예: 표준정규 벡터의 공간). 결정론 solver면 $\Omega_k$가 사실상 한 점 공간으로 축약된다.",
            r"둘째 문장의 적응성(비예견성)은 확률 과정에서 아주 중요한 문법이다. 쉽게 말해 “아직 뽑지 않은 미래 난수는 보지 않는다”를 수식으로 적는 장치다.",
        ],
        "intuition": [
            r"‘이번 step에서 할 일’ 전체를 묶어 부르는 이름이다.",
            r"deterministic sampler면 난수 입력이 없어 $\Phi_k(x_k,h_k)$처럼 쓸 수도 있다.",
            r"ComfyUI에서 `sampler_name`이 달라진다는 것은 결국 $\Phi_k$의 내부(중간 평가점, 결합 계수, 노이즈 주입 방식)가 달라진다는 뜻이다. 밖에서 보면 항상 $x\leftarrow \Phi_k(\cdot)$ 반복이므로, 커스텀 샘플러도 $\Phi_k$부터 정의하는 게 가장 빠르다.",
            r"코드 구현 관점에선 $\Phi_k$는 함수 하나로 캡슐화할 수 있다: `x_next = step(x, sigma_k, sigma_k1, model, cond, rng, history)`처럼. 이렇게 쓰면 ComfyUI와 독립적인 “순수 파이썬 샘플러”로도 이식이 쉽다.",
        ],
        "examples": [
            r"일반형: $x_{k+1}=\Phi_k(x_k,h_k,\omega_k)$.",
            r"(숫자 예) 업데이트가 Euler 형태 $x_{k+1}=x_k+(\sigma_{k+1}-\sigma_k)\,d_k$라고 하자. $x_k=1.0$, $\sigma_k=5.0$, $\sigma_{k+1}=2.0$, $d_k=-0.2$이면 $x_{k+1}=1.0+(2.0-5.0)\cdot(-0.2)=1.6$이다.",
            r"(ancestral 수치 예) 추가 노이즈를 $x_{k+1}=x_k+(\sigma_{k+1}-\sigma_k)d_k+\sigma_{\mathrm{up}}\xi_k$로 넣는다고 하자. 위 예에서 $\sigma_{\mathrm{up}}=0.3$, $\xi_k=0.5$면 노이즈 항은 $0.15$이고 $x_{k+1}=1.6+0.15=1.75$가 된다(같은 drift라도 결과가 흔들림).",
            r"(제약 포함 예) 업데이트 후 클립 제약 $\mathcal{K}=[-1,1]$을 적용하면 $x'=\Phi_k(\cdot)$로 나온 값이 1.75여도 $\Pi_\mathcal{K}(x')=1.0$으로 되돌아간다. 즉 $\Phi_k$ 뒤에 투영을 합성하면 “제약 샘플러”가 된다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\Phi_k:(\mathcal{X}\times\mathcal{H}_k\times\Omega_k)\to\mathcal{X}",
                "explain": [
                    r"이 표기는 “입력 3개를 받는 함수”라는 뜻이다. $x_k\in\mathcal{X}$는 현재 상태, $h_k\in\mathcal{H}_k$는 과거값/과거기울기 버퍼, $\omega_k\in\Omega_k$는 이번 step의 난수다.",
                    r"출력은 다음 상태 $x_{k+1}\in\mathcal{X}$ 하나다.",
                    r"solver가 1-step이면 $\mathcal{H}_k$가 한 점 공간이라 사실상 영향이 없고, deterministic이면 $\Omega_k$도 한 점 공간이라 난수 입력이 사라진다.",
                ],
                "intuition": [
                    r"코드의 `for k in steps: x = step(x, ...)`에서 `step`이 바로 $\Phi_k$다.",
                ],
                "examples": [
                    r"결정론 1-step 예: $x_{k+1}=x_k+(\sigma_{k+1}-\sigma_k)\,d_k$는 $\omega_k$ 없이도 정의된다.",
                    r"확률 1-step 예: $x_{k+1}=m_k(x_k)+\alpha_k\,\xi_k$처럼 난수 $\xi_k$가 들어가면 $\omega_k$가 필요하다.",
                ],
            },
            {
                "stmt": r"\text{non-anticipative: }\Phi_k\ \text{uses only information up to step }k",
                "explain": [
                    r"비예견성은 “$x_{k+1}$을 계산할 때 미래 잡음 $\xi_{k+1},\xi_{k+2},\dots$를 미리 보지 않는다”는 뜻이다.",
                    r"엄밀히는 $x_k$가 $\mathcal{F}_k$-가측이고, $\Phi_k$가 $(x_k,h_k,\omega_k)$만으로 정의된다는 조건으로 표현된다.",
                ],
                "intuition": [
                    r"샘플러 루프는 보통 ‘난수 하나 뽑고 바로 쓰고 넘어간다’ 구조다. 그게 바로 비예견성이다.",
                ],
                "examples": [
                    r"예: $\xi_0=0.5$를 뽑아 $x_1$을 계산하고, 그 다음에야 $\xi_1=-1.2$를 뽑아 $x_2$를 계산한다. $x_1$을 계산할 때 $\xi_1$을 쓰지 않는다.",
                ],
            },
        ],
        "code_map": [
            r"`sample_*` 루프 내부의 한 반복 업데이트가 $\Phi_k$다.",
        ],
        "pitfalls": [
            "미래 난수를 참조하면(look-ahead) 확률해석에서 적응성이 깨진다.",
        ],
    },
    "Psi_k": {
        "summary": r"multistep 샘플러는 과거값/과거 기울기 등을 버퍼에 저장한다. $\Psi_k$는 ‘새로운 상태 $x_{k+1}$를 얻은 뒤 그 버퍼를 어떻게 갱신하는가’를 나타내는 함수다.",
        "why": [
            r"multistep sampler의 성능/정확도는 ‘과거 정보를 어떻게 들고 가는가’에 크게 좌우된다. 즉 solver 공식을 정확히 구현하려면, 버퍼의 내용뿐 아니라 갱신 규칙까지 명확해야 한다. $\Psi_k$는 그 갱신 규칙을 solver 본체에서 분리해 이름 붙인 것이다.",
            r"ComfyUI에서 `lms`, `dpmpp_2m`, `deis`, `ipndm` 같은 계열은 내부적으로 과거 denoised/derivative를 저장한다. 이때 버퍼를 shift하는 순서가 바뀌거나, 한 step 늦게/빠르게 넣으면 같은 이름의 sampler라도 완전히 다른 수치식을 실행하게 된다. $\Psi_k$를 명시하면 이런 실수를 문서/코드에서 바로 드러낼 수 있다.",
            r"초기 워밍업 구간에서는 과거가 부족해서 버퍼 길이 $m_k$가 커져 가는 경우가 많다. 이때 갱신은 단순 ‘한 칸 밀고 넣기’가 아니라 ‘길이를 늘리기 + 밀기’가 섞인다. $\Psi_k$에 이 규칙을 포함시키면 워밍업을 포함한 전체 알고리즘이 한 틀로 정리된다.",
            r"실무적으로는 ‘중간 상태 저장/재개’에서 $\Psi_k$가 결정적이다. $x_k$만 저장해 재개하면 multistep 버퍼가 초기화되어, 이어서 돌리는 구간이 사실상 저차 방법으로 바뀐다. 버퍼(= $h_k$)와 그 갱신 규약까지 함께 저장해야 동일한 궤적을 재현할 수 있다.",
            r"영상 생성/타일링처럼 여러 스트림을 병렬로 샘플링할 때도, 버퍼를 어느 단위로 유지할지(프레임별/타일별 독립) 설계해야 한다. 이는 결국 $\Psi_k$를 ‘어떤 상태 증강 공간’에서 정의할지의 문제로 귀결된다.",
        ],
        "formal": [
            r"$\Psi_k:(\mathcal{H}_k\times\mathcal{X})\to\mathcal{H}_{k+1}$.",
        ],
        "formal_explain": [
            r"이 타입 선언은 “버퍼 $h_k$와 새 상태 $x_{k+1}$를 받아서 다음 버퍼 $h_{k+1}$를 만든다”는 뜻이다.",
            r"다단계 solver를 구현할 때 자주 실수하는 지점이 버퍼의 순서/길이다. $\Psi_k$를 따로 떼어 생각하면 “버퍼 갱신 로직”이 solver 본체와 분리돼 검증하기 쉬워진다.",
            r"워밍업 구간에서는 $m_k$가 작다가 점점 최대 차수로 올라가는 경우가 많다. 이때 $\Psi_k$는 단순 shift뿐 아니라 “길이를 늘리는 규칙”까지 포함할 수 있다.",
        ],
        "intuition": [
            r"리스트/큐를 한 칸 민 뒤, 맨 뒤에 새 값을 넣는 연산이라고 보면 된다.",
            r"무엇을 저장하느냐(상태 $x$인지, denoised인지, derivative인지)에 따라 $\mathcal{H}_k$의 내용만 바뀌고 ‘갱신’이라는 구조는 같다.",
            r"코드에서는 보통 `append` 후 `pop(0)`(또는 `deque(maxlen=m)`)로 구현되는 간단한 연산이지만, 이 한 줄이 solver 차수/안정성에 직접 영향을 준다. 예를 들어 무엇을 저장하느냐(x, denoised, residual)가 달라지면 완전히 다른 multistep이 된다.",
        ],
        "examples": [
            r"$\mathcal{X}=\mathbb{R}^2$, 2-step 버퍼라 하자. $h_k=((1.0,0.0),(0.3,-0.2))$, $x_{k+1}=(0.1,0.0)$이면 $\Psi_k(h_k,x_{k+1})=((0.3,-0.2),(0.1,0.0))$.",
            r"1-step(버퍼 없음)으로 보면 $\mathcal{H}_k$가 한 점 공간이므로, 사실상 $\Psi_k$는 ‘아무 것도 안 함’에 가깝다.",
            r"(리스트 구현 예) history 길이를 2로 유지한다고 하자. `history=[0.20, 0.12]`이고 새 값이 0.05면, 갱신 후 `history=[0.12, 0.05]`가 된다(가장 오래된 값 제거).",
        ],
        "walkthrough": [
            {
                "stmt": r"\Psi_k:(\mathcal{H}_k\times\mathcal{X})\to\mathcal{H}_{k+1}",
                "explain": [
                    r"입력은 (현재 버퍼 $h_k$, 새 상태 $x_{k+1}$)이고 출력은 다음 버퍼 $h_{k+1}$이다.",
                    r"$\mathcal{H}_k$ 자체가 $\mathcal{X}^{m_k}$ 형태이므로, 결국 $\Psi_k$는 “튜플을 어떻게 업데이트하나”의 문제로 내려온다.",
                ],
                "intuition": [
                    r"코드에서는 보통 `append`/`pop(0)` 또는 `deque`의 `append`/`popleft` 조합으로 구현된다.",
                ],
                "examples": [
                    r"2-step에서 $h_k=(x_{k-1},x_k)$이면 $\Psi_k(h_k,x_{k+1})=(x_k,x_{k+1})$.",
                ],
            },
            {
                "stmt": r"\text{warmup: }m_k\ \text{can increase with }k",
                "explain": [
                    r"초기에는 과거값이 부족하므로 $m_0=0$ 또는 $m_0=1$에서 시작해 $m_k$를 점차 늘리는 구현이 흔하다.",
                    r"예를 들어 목표가 3-step solver이면, $k=0$에서는 1-step처럼 돌리고, $k=1$부터 2-step, $k=2$부터 3-step으로 전환하는 식이다.",
                ],
                "intuition": [
                    r"‘기억이 쌓여야 고차 방법을 쓸 수 있다’는 뜻이다.",
                ],
                "examples": [
                    r"$m_0=0$ (버퍼 없음) → $m_1=1$ (최근 1개 저장) → $m_2=2$ (최근 2개 저장)처럼 늘어날 수 있다.",
                ],
            },
        ],
        "code_map": [
            r"`old_denoised.append(...)`, `pop(0)` 같은 로직이 $\Psi_k$에 해당한다.",
        ],
        "pitfalls": [
            "버퍼 순서를 바꾸면 같은 solver 이름이라도 실제 수치식이 달라진다.",
        ],
    },
    "Pi_K": {
        "summary": r"임의의 점 $x\in\mathcal{X}$를 허용 집합 $\mathcal{K}$ 안으로 되돌리는 연산을 $\Pi_{\mathcal{K}}$로 적는다. ‘제약을 만족시키는 보정 단계’를 수학적으로 한 글자로 표기한 것이다.",
        "why": [
            r"ComfyUI의 편집 파이프라인은 종종 ‘샘플러 업데이트 후 보정’을 끼워 넣는다(마스크 고정, latent 합성, clamp 등). 이를 수학적으로는 $x\leftarrow \Pi_{\mathcal{K}}(x)$ 한 줄로 쓰며, 전체 알고리즘은 $\Phi_k$와 $\Pi_{\mathcal{K}}$의 합성으로 정리된다. 이 관점이 있어야 “제약이 언제/어디서 적용되는가”가 명확해진다.",
            r"같은 제약이라도 ‘어떤 거리(노름)’에 대한 투영인지에 따라 결과가 달라질 수 있다. 예를 들어 유클리드 노름에서는 좌표별 clip이 자연스럽지만, 채널별 가중치를 주면(가중 노름) 어떤 채널을 더 강하게 보정하는 투영이 된다. $\Pi_{\mathcal{K}}$ 표기는 이런 전제(거리 구조)를 드러낸다.",
            r"inpaint를 ‘허용집합으로의 투영’으로 보면, hard mask는 정확한 투영이고 soft mask는 투영의 완화(또는 proximal/relaxation)로 이해할 수 있다. 그래서 경계(seam) 문제를 줄이기 위해 마스크를 블러 처리하는 실무 팁도 “투영을 부드럽게 만든다”로 해석된다.",
            r"수치해석/최적화에서는 projected Euler, projected gradient descent, proximal point 같은 기법이 표준이다. 샘플러에 투영을 끼워 넣는 아이디어는 이런 계열과 구조적으로 같아서, 안정성(발산 방지)과 제약 만족을 동시에 얻는 이유를 설명해 준다.",
            r"커스텀 샘플러를 만들 때도 $\Pi_{\mathcal{K}}$를 명시적으로 분리해 두면, ‘기본 샘플러’는 그대로 두고 제약만 교체/추가하는 모듈식 설계가 가능해진다(예: 같은 sampler에 다른 inpaint 제약을 끼워 넣기).",
        ],
        "formal": [
            r"$\Pi_{\mathcal{K}}:\mathcal{X}\to\mathcal{K}$.",
            r"$\mathcal{K}$가 닫힌 볼록이면 $\Pi_{\mathcal{K}}(x)=\arg\min_{y\in\mathcal{K}}\|x-y\|_2$가 유일하다.",
        ],
        "formal_explain": [
            r"첫 문장은 “투영은 언제나 $\mathcal{K}$ 안의 점을 반환한다”는 뜻이다. 즉, 보정 후에는 제약을 만족한다($\Pi_{\mathcal{K}}(x)\in\mathcal{K}$).",
            r"둘째 문장은 투영을 ‘최소거리 문제’로 정의한다. 여기서 거리의 기준이 $\|\cdot\|_2$이므로, 다른 노름(가중 노름 등)을 쓰면 투영 연산도 달라진다.",
            r"볼록성은 유일성을 보장하는 전형적인 조건이다. 비볼록이면 최소점이 여러 개일 수 있어서, 구현이 어떤 점을 고르는지(타이브레이크 규칙)가 결과에 영향을 준다.",
        ],
        "intuition": [
            "규칙 위반한 점을 가장 가까운 합법 점으로 ‘스냅(snap)’시키는 연산이다.",
            r"$\mathcal{K}$가 ‘상자(box)’면 clip, ‘좌표 고정’이면 해당 좌표를 덮어쓰는 동작으로 생각하면 된다.",
            r"ComfyUI의 inpaint는 마스크로 고정할 좌표를 정한 뒤, 매 step에서 그 좌표를 다시 원본으로 덮어쓰는 형태가 흔하다. 이것이 바로 “아핀 부분공간으로의 투영”의 구현이다.",
            r"코드 구현에서는 $\Pi_{\mathcal{K}}$가 너무 비싸면(복잡한 제약) soft constraint(패널티/프로시멀)로 바꾸기도 한다. 하지만 ‘하드 제약이면 투영’이라는 원칙을 알면 커스터마이징 선택지가 명확해진다.",
        ],
        "examples": [
            r"$\mathcal{X}=\mathbb{R}^2$, $\mathcal{K}=[-1,1]^2$, $x=(1.4,-0.5)$이면 $\Pi_{\mathcal{K}}(x)=(1.0,-0.5)$.",
            r"$\mathcal{X}=\mathbb{R}^3$, $\mathcal{K}=\{(0.2,y_2,y_3)\}$, $x=(0.9,-1.0,0.1)$이면 $\Pi_{\mathcal{K}}(x)=(0.2,-1.0,0.1)$.",
            r"(clip 수치 예) $d=2$, $\mathcal{K}=[-1,1]^2$, $x=(1.3,-1.4)$면 $\Pi_{\mathcal{K}}(x)=(1.0,-1.0)$이다.",
            r"(마스크 수치 예) $x=(0.1,0.2,0.3,0.4)$, $x_{\mathrm{fix}}=(1,1,0,0)$, $m=(1,1,0,0)$이면 $\Pi_{\mathcal{K}}(x)=m\odot x_{\mathrm{fix}}+(1-m)\odot x=(1,1,0.3,0.4)$로 계산된다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\Pi_{\mathcal{K}}:\mathcal{X}\to\mathcal{K}",
                "explain": [
                    r"입력은 임의의 상태 $x\in\mathcal{X}$이고, 출력은 제약을 만족하는 상태 $\Pi_{\mathcal{K}}(x)\in\mathcal{K}$다.",
                    r"따라서 “solver가 만든 후보 $x$를 제약 안으로 되돌린다”를 함수 합성으로 $x\leftarrow \Pi_{\mathcal{K}}(x)$처럼 쓸 수 있다.",
                ],
                "intuition": [
                    r"항상 제약을 만족하는 ‘정정 단계’다.",
                ],
                "examples": [
                    r"$\mathcal{K}=[-1,1]$에서 $x=2.3$이면 $\Pi_{\mathcal{K}}(x)=1.0$.",
                ],
            },
            {
                "stmt": r"\Pi_{\mathcal{K}}(x)=\arg\min_{y\in\mathcal{K}}\|x-y\|_2",
                "explain": [
                    r"투영은 최소화 문제로 정의된다. 목적함수는 “$x$와 $y$의 거리”다.",
                    r"볼록이면 유일성이 확보되는 경우가 많고, 수치 구현도 명확해진다.",
                ],
                "intuition": [
                    r"상자 제약이면 ‘좌표별로 잘라내기’가 최소거리 투영과 일치한다.",
                ],
                "examples": [
                    r"$\mathcal{K}=[-1,1]^2$, $x=(1.4,-0.5)$이면 가장 가까운 점은 $(1.0,-0.5)$.",
                    r"$\mathcal{K}=\{(0.2,y_2,y_3)\}$, $x=(0.9,-1.0,0.1)$이면 첫 좌표만 바뀐 $(0.2,-1.0,0.1)$이 가장 가깝다.",
                ],
            },
        ],
        "code_map": [
            "`torch.clamp`, 마스크 기반 overwrite가 대표적인 구현이다.",
        ],
        "pitfalls": [
            "비유클리드 거리/가중 노름을 쓰면 projection 공식이 달라진다.",
        ],
    },
    "R_x": {
        "summary": r"접공간에서 계산한 작은 이동 $v\in T_x\mathcal{M}$을 실제 다양체 $\mathcal{M}$ 위의 점으로 되돌리는 지도를 retraction $R_x$라고 한다. 실무에서는 ‘더한 뒤 정규화’ 같은 간단한 형태로 구현되는 경우가 많다.",
        "why": [
            r"다양체 제약($x\in\mathcal{M}$)이 있는 상황에서 샘플러/보정은 보통 “접공간에서 벡터 연산으로 방향을 계산”하고 “그 결과를 다시 제약 위로 복귀”하는 두 단계로 구성된다. 이때 두 번째 단계를 담당하는 표준 도구가 retraction $R_x$다.",
            r"ComfyUI 커스터마이징에서도 ‘정규화’ 류 후처리를 step마다 넣는 경우가 생긴다(예: 어떤 특징/임베딩/상태를 단위노름으로 유지). 이런 구현은 수학적으로 $R_x(v)=\frac{x+v}{\|x+v\|}$ 같은 retraction으로 해석되며, “왜 제약이 유지되는가”를 설명해 준다.",
            r"retraction을 명시하면 누적 오차(장기 drift)를 다루기 쉬워진다. 단순히 $x+v$만 반복하면 작은 제약 위반이 step마다 쌓여 점점 제약에서 멀어질 수 있는데, $R_x$를 넣으면 매 step에서 제약 만족을 강제할 수 있다.",
            r"이론적으로는 exponential map(지오데식)이 ‘정확한’ 이동이지만, 계산이 비싸거나 닫힌형이 없는 경우가 많다. retraction은 $R_x(0)=x$, $dR_x(0)=\mathrm{Id}$ 같은 최소 조건만 만족하면서 계산이 싼 근사로, 실무 수치해석에서 널리 쓰인다.",
            r"확률적 업데이트(SDE)에서는 노이즈를 더한 뒤에도 제약을 유지해야 할 수 있다. 이때도 “노이즈를 접공간에서 주입하고(restrict), 결과를 $R_x$로 복귀”하는 패턴이 자연스럽고, 다양체 위의 확산을 구현하는 기본 골격이 된다.",
        ],
        "formal": [
            r"$R_x:T_x\mathcal{M}\to\mathcal{M},\ R_x(0)=x,\ dR_x(0)=\mathrm{Id}$.",
        ],
        "formal_explain": [
            r"첫 부분 $R_x:T_x\mathcal{M}\to\mathcal{M}$는 “접공간 벡터를 다양체의 점으로 보낸다”는 타입 선언이다. 입력은 ‘방향/보정’이고 출력은 ‘제약을 만족하는 새 점’이다.",
            r"$R_x(0)=x$는 “아무 것도 안 움직이면 그대로다”라는 최소한의 일관성 조건이다.",
            r"$dR_x(0)=\mathrm{Id}$는 “아주 작은 이동에서는 $R_x$가 거의 $x+v$처럼 행동한다”는 뜻이다. 그래서 1차(선형) 수준에서는 접공간 계산이 올바르게 반영된다.",
        ],
        "intuition": [
            "접평면에서 한 걸음 뗀 뒤, 다시 곡면 위로 ‘붙여 넣는’ 동작이다.",
            r"구면/원 제약에서는 “벡터를 더한 뒤 길이를 1로 맞춘다”가 가장 직관적인 retraction이다.",
            r"코드 구현에서 $R_x$는 보통 “더한 뒤 normalize” 같은 한두 줄로 구현된다. 하지만 수학적으로는 $R_x(0)=x$, $dR_x(0)=\mathrm{Id}$ 같은 조건 덕분에 작은 업데이트에서 1차 정확도를 보장하는 ‘올바른’ 제약 복귀 연산으로 분류된다.",
        ],
        "examples": [
            r"$\mathcal{M}=S^1\subset\mathbb{R}^2$, $x=(0.6,0.8)$라 하자. $v=(0.1,-0.2)$면 $x+v=(0.7,0.6)$이고 $\|x+v\|_2\approx0.922$이므로 $R_x(v)=\frac{(0.7,0.6)}{0.922}\approx(0.759,0.651)$.",
            r"더 작은 $v=(0.01,-0.02)$면 $x+v=(0.61,0.78)$, 정규화 후도 $(0.616,0.788)$처럼 크게 달라지지 않는다(‘작은 이동’ 가정).",
            r"(구면 수치 예) $\mathcal{M}=S^2\subset\mathbb{R}^3$, $x=(1,0,0)$, $v=(0,0.2,0.1)$이면 $x+v=(1,0.2,0.1)$, 노름은 $\sqrt{1+0.04+0.01}\approx1.025$이므로 $R_x(v)\approx(0.975,0.195,0.098)$이다.",
        ],
        "walkthrough": [
            {
                "stmt": r"R_x:T_x\mathcal{M}\to\mathcal{M},\quad R_x(0)=x,\quad dR_x(0)=\mathrm{Id}",
                "explain": [
                    r"retraction은 ‘접공간에서의 벡터 연산’과 ‘제약 위의 점’ 사이를 이어 주는 다리다.",
                    r"$dR_x(0)=\mathrm{Id}$는 $v$가 작을 때 $R_x(v)$가 $x+v$의 1차 근사와 일치하도록 해 준다.",
                ],
                "intuition": [
                    r"작은 스텝에서는 “그냥 더했다가 제약으로 복귀”와 거의 같다.",
                ],
                "examples": [
                    r"$v=0$이면 항상 $R_x(0)=x$.",
                ],
            },
            {
                "stmt": r"\mathcal{M}=S^{d-1}:\quad R_x(v)=\frac{x+v}{\|x+v\|_2}",
                "explain": [
                    r"구면 제약에서 가장 흔한 retraction은 정규화다. 어떤 벡터든 길이를 1로 맞추면 구면 위에 올라간다.",
                    r"$v$가 충분히 작다면, 정규화는 $x+v$와 매우 비슷하게 움직이면서도 제약을 지킨다.",
                ],
                "intuition": [
                    r"“더한 다음 길이를 1로 맞춘다”는 구현이 수학적으로는 retraction으로 해석된다.",
                ],
                "examples": [
                    r"$x=(0.6,0.8)$, $v=(0.1,-0.2)$면 $R_x(v)\approx(0.759,0.651)$ (위 계산).",
                ],
            },
        ],
        "code_map": [
            "`x = normalize(x + v)` 패턴이 가장 흔한 retraction 근사다.",
        ],
        "pitfalls": [
            "지오데식 exponential map과 retraction을 동일시하면 고차 오차 해석이 틀릴 수 있다.",
        ],
    },
    "Omega_F_P": {
        "summary": r"확률을 엄밀히 쓰려면 ‘무작위의 원천’이 되는 확률공간 $(\Omega,\mathcal{F},\mathbb{P})$부터 정한다. 샘플러에서 말하는 난수(가우시안 노이즈 등)는 결국 이 공간 위의 확률변수로 모델링된다.",
        "why": [
            r"ComfyUI에서 `seed`를 고정한다는 것은 사실상 “난수 경로 $\omega\in\Omega$를 하나 선택해 고정한다”는 것과 같다. 그래서 같은 모델/프롬프트/설정이면 결과가 재현되고, seed를 바꾸면 다른 $\omega$를 뽑아 다른 결과를 얻는다. 이 재현성의 ‘수학적 바닥’이 $(\Omega,\mathcal{F},\mathbb{P})$다.",
            r"샘플러는 결국 “난수 $\omega$를 입력받아 최종 latent/이미지를 출력하는 함수”로 볼 수 있다. 즉 $x_N:\Omega\to\mathcal{X}$ 같은 사상이며, sampler를 바꾸는 것은 이 사상을 바꾸는 것이다. deterministic sampler는 사실상 $\omega$ 의존성이 약하고, SDE/ancestral은 step마다 더 많은 난수를 사용하므로 $\omega$의 영향이 커진다.",
            r"배치 생성(여러 장 생성)은 서로 다른 $\omega$를 여러 개 샘플링하는 것과 같다. ‘variation seed’류의 개념은 $\omega$의 일부 성분만 바꾸거나(조건부 샘플링) 일부는 고정하는 것(공통 노이즈)으로도 해석할 수 있어, 확률공간 관점이 설계를 명확히 해 준다.",
            r"영상 생성에서 프레임 간 노이즈 상관을 주거나(같은 노이즈를 공유, 또는 상관 노이즈 사용) 특정 프레임만 다른 변동을 주는 설계는 결국 “어떤 공동분포 $\mathbb{P}$를 쓸 것인가”의 문제다. 이때 $\Omega$를 ‘프레임×step×채널’ 전체 노이즈 경로의 공간으로 잡으면 한 번에 정리된다.",
            r"분포 수준에서는 $\rho_t=(x_t)_\#\mathbb{P}$ (pushforward)로 ‘난수의 분포가 상태 분포로 어떻게 옮겨졌는가’를 표현한다. FPE/OT 같은 이론으로 연결하거나, 커스텀 sampler를 분포 관점에서 비교하려면 확률공간을 명시하는 게 결국 필요하다.",
        ],
        "formal": [
            r"$\Omega$는 표본공간, $\mathcal{F}$는 그 위 $\sigma$-대수, $\mathbb{P}$는 확률측도.",
        ],
        "formal_explain": [
            r"이 문장은 확률론의 기본 3요소를 한꺼번에 적는다. (1) $\Omega$는 가능한 모든 ‘난수 시나리오’의 집합이고, (2) $\mathcal{F}$는 그중에서 확률을 정의할 사건들의 모음이며, (3) $\mathbb{P}$는 각 사건에 확률을 부여하는 함수다.",
            r"샘플러에서 seed를 바꾼다는 것은 직관적으로는 $\omega$를 다시 뽑는 것과 같다. 같은 $\omega$를 쓰면 동일한 노이즈 경로가 재현된다.",
            r"실무 독해에서는 “난수는 어딘가에서 나온다” 정도로 넘어가도 되지만, 분포 수준(FPE/OT)으로 올라가면 $(\Omega,\mathcal{F},\mathbb{P})$가 문법을 지탱한다.",
        ],
        "intuition": [
            "‘가능한 모든 난수 시나리오’를 한 번에 담은 무대라고 보면 된다.",
            r"seed를 고정하면(실험적으로) 이 무대에서 사실상 한 경로만 계속 선택하는 것에 가깝다.",
            r"ComfyUI에서 `seed`가 같으면(같은 실행 경로라면) 같은 초기 노이즈와 같은 추가 노이즈 시퀀스가 생성된다. 즉 난수 생성기가 $(\Omega,\mathcal{F},\mathbb{P})$를 컴퓨터에서 구현한 구체물이라고 보면 된다(유한정밀도/의사난수이긴 하지만).",
            r"결정론 sampler는 추가 노이즈가 없어 $\Omega$를 한 점 공간으로 축약해도 된다. 반대로 ancestral/SDE 계열은 step마다 새 난수가 필요하므로 $\Omega$를 “여러 개 난수의 곱공간”으로 보는 게 자연스럽다.",
        ],
        "examples": [
            r"(유한 예) $\Omega=\{1,2,3\}$, $\mathcal{F}=2^\Omega$, $\mathbb{P}(1)=0.2,\ \mathbb{P}(2)=0.5,\ \mathbb{P}(3)=0.3$처럼 둘 수 있다.",
            r"(가우시안 예) $N$ step, $d$차원 표준정규를 쓰면 $\Omega=(\mathbb{R}^d)^N$, $\mathbb{P}=\bigotimes_{k=0}^{N-1}\mathcal{N}(0,I_d)$로 모델링한다.",
            r"(난수 경로 수치 예) $d=2$, $N=3$이면 한 $\omega\in\Omega$는 $\omega=(\xi_0,\xi_1,\xi_2)$로 쓸 수 있고, 예를 들어 $\xi_0=(0.3,-1.2)$, $\xi_1=(0.1,0.5)$, $\xi_2=(-0.7,0.0)$ 같은 값들의 묶음이 된다.",
        ],
        "walkthrough": [
            {
                "stmt": r"(\Omega,\mathcal{F},\mathbb{P})",
                "explain": [
                    r"$\Omega$는 ‘가능한 세계들(난수 시나리오)’의 집합이다.",
                    r"$\mathcal{F}$는 사건들의 모음이다. 사건은 $\Omega$의 부분집합이고, $\sigma$-대수 조건(여집합/가산합집합 닫힘)을 만족한다.",
                    r"$\mathbb{P}$는 각 사건 $A\in\mathcal{F}$에 확률 $\mathbb{P}(A)\in[0,1]$을 부여하고, $\mathbb{P}(\Omega)=1$이며 가산가법성을 만족한다.",
                ],
                "intuition": [
                    r"샘플러가 쓰는 모든 난수는 결국 $\omega\in\Omega$ 하나가 선택되면서 ‘구체 값’으로 실현된다.",
                ],
                "examples": [
                    r"유한 예: $\Omega=\{1,2,3\}$, $\mathcal{F}=2^\Omega$, $\mathbb{P}(1)=0.2,\mathbb{P}(2)=0.5,\mathbb{P}(3)=0.3$.",
                ],
            },
            {
                "stmt": r"\Omega=(\mathbb{R}^d)^N,\quad \mathbb{P}=\bigotimes_{k=0}^{N-1}\mathcal{N}(0,I_d)",
                "explain": [
                    r"step마다 $d$차원 가우시안 난수 벡터를 하나씩 뽑는다면, 전체 난수 경로는 길이 $N$의 벡터열 $(\xi_0,\dots,\xi_{N-1})$로 표현된다.",
                    r"독립(i.i.d.)을 가정하면 전체 분포는 곱측도(product measure)로 쓸 수 있다.",
                ],
                "intuition": [
                    r"“N번 난수를 뽑는다”를 수학적으로 한 번에 포장한 표현이다.",
                ],
                "examples": [
                    r"$d=1,N=3$이면 $\omega=(\xi_0,\xi_1,\xi_2)\in\mathbb{R}^3$이고, 예를 들어 $\omega=(0.5,-1.2,0.1)$ 같은 값이 한 경로다.",
                ],
            },
        ],
        "code_map": [
            "코드에서는 RNG state가 이 확률공간의 한 실현 경로를 선택한다.",
        ],
        "pitfalls": [
            "결정론 실험(seed 고정)과 확률모형 자체를 혼동하지 않아야 한다.",
        ],
    },
    "Xi_k": {
        "summary": r"$\xi_k$는 $k$번째 step에서 실제로 뽑히는 난수(노이즈) 확률변수다. 코드에서 `torch.randn_like(x)` 같은 호출로 생성되는 텐서를 수학적으로 부르면 $\xi_k$에 해당한다.",
        "why": [
            r"ComfyUI에서는 sampler에 따라 난수가 ‘초기 한 번’만 쓰이기도 하고(결정론 계열), step마다 추가로 주입되기도 한다(ancestral/SDE). 이때 step별 난수를 기호로 분리해 두면, “왜 어떤 sampler는 결과가 더 랜덤하게 흔들리는가”가 $\xi_k$의 사용 방식 차이로 정리된다.",
            r"`eta`, `s_noise`, `s_churn` 같은 파라미터는 실무적으로 “$\xi_k$를 얼마나 강하게 섞을지”를 조절하는 손잡이에 가깝다. 즉 질감/다양성/안정성(특히 영상의 깜빡임)을 튜닝할 때, 결국 $\xi_k$가 들어가는 항의 스케일을 만지는 셈이다.",
            r"inpaint에서는 보통 마스크 영역에만 노이즈를 주입하고 나머지는 고정하는데, 이는 $\xi_k$를 마스크로 곱해 부분적으로만 작동시키는 것과 같다. 따라서 ‘어디에 랜덤성을 남길지’는 $\xi_k$ 설계(및 $\mathcal{K}$ 제약)와 직접 연결된다.",
            r"기본 가정(i.i.d. 표준정규)을 바꾸면 결과 분포가 달라진다. 예를 들어 프레임 간 상관 노이즈(영상), 저주파/컬러드 노이즈, 또는 비정규 노이즈를 쓰면 질감과 안정성이 달라질 수 있고, 그 변화는 결국 $\xi_k$의 분포/상관 구조 변화로 설명된다.",
            r"재현성과 중간 재개(continue sampling)를 제대로 하려면, RNG state를 보존하거나 $\xi_0,\dots,\xi_{k-1}$의 소비 순서를 일관되게 유지해야 한다. 그렇지 않으면 같은 seed라도 “어느 step에서 어떤 $\xi_k$가 뽑혔는가”가 달라져 결과가 어긋난다.",
        ],
        "formal": [
            r"$\xi_k:(\Omega,\mathcal{F},\mathbb{P})\to(\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d))$.",
            r"표준 가정은 $\xi_k\sim\mathcal{N}(0,I_d)$, 서로 독립(i.i.d.).",
        ],
        "formal_explain": [
            r"첫 문장은 $\xi_k$가 “확률공간에서 값공간으로 가는 함수”라는 뜻이다. 즉 $\omega$가 정해지면(난수 시나리오가 하나 선택되면) $\xi_k(\omega)$라는 구체 난수 벡터가 정해진다.",
            r"둘째 문장은 분포 가정이다. 대부분의 diffusion sampling에서는 각 step에서 표준정규를 독립으로 뽑는다(평균 0, 공분산 $I_d$).",
            r"이 가정이 바뀌면(예: 상관 노이즈, 비정규 노이즈) 결과 경로의 분산 구조가 달라지고, 이론적 해석도 바뀐다.",
        ],
        "intuition": [
            r"각 step마다 새로 뽑아 넣는 ‘랜덤 흔들림 벡터’다.",
            r"같은 seed를 쓰면 $\xi_k$의 실현값이 재현돼 결과가 반복된다.",
            r"ComfyUI에서 deterministic sampler는 초기 노이즈만으로 경로가 결정되고, step별 추가 난수를 쓰지 않는다. 반면 ancestral/SDE sampler는 매 step에 $\xi_k$가 들어가므로, 같은 `seed`라도 ‘추가 난수 사용 방식’이 바뀌면 결과가 달라질 수 있다(예: churn 구간에서만 뽑기).",
        ],
        "examples": [
            r"(1차원) $\xi_k\sim\mathcal{N}(0,1)$에서 한 번 뽑았더니 $\xi_k=-0.73$이 나왔다고 하자(실현값).",
            r"(2차원) $\xi_k=(0.31,-1.24)\in\mathbb{R}^2$ 같은 벡터도 실현값이 될 수 있다.",
            r"(스케일링 예) $d=3$에서 $\xi_k=(0.2,-0.1,1.5)$를 뽑고 스케일 0.4를 곱하면 주입 노이즈는 $(0.08,-0.04,0.6)$이 된다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\xi_k:(\Omega,\mathcal{F},\mathbb{P})\to(\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d))",
                "explain": [
                    r"표본공간의 원소 $\omega$는 ‘난수의 전체 시나리오’다. $\xi_k(\omega)$는 그 시나리오에서 $k$번째로 뽑힌 난수 벡터다.",
                    r"값공간이 $\mathbb{R}^d$인 이유는 상태 $x_k\in\mathbb{R}^d$에 더해지려면 차원이 맞아야 하기 때문이다.",
                ],
                "intuition": [
                    r"코드에서 난수를 한 번 뽑아 텐서로 받는 그 값이 $\xi_k$의 실현값이다.",
                ],
                "examples": [
                    r"$d=2$에서 $\xi_k(\omega)=(0.31,-1.24)$ 같은 값이 가능하다.",
                ],
            },
            {
                "stmt": r"\xi_k\sim\mathcal{N}(0,I_d),\quad \text{i.i.d.}",
                "explain": [
                    r"표준정규는 평균 0, 공분산이 항등행렬이라는 뜻이다: $\mathbb{E}[\xi_k]=0$, $\mathrm{Cov}(\xi_k)=I_d$.",
                    r"i.i.d.는 서로 다른 step의 노이즈가 독립이고 동일한 분포를 따른다는 뜻이다.",
                ],
                "intuition": [
                    r"‘매 step 새로 뽑는 랜덤’이고, 과거 노이즈와 상관이 없다고 보는 것이 기본 모델이다.",
                ],
                "examples": [
                    r"$d=1$에서 $\xi_0=0.5$, $\xi_1=-1.2$, $\xi_2=0.1$처럼 서로 다른 값이 독립적으로 나온다고 가정한다.",
                ],
            },
        ],
        "code_map": [
            "`noise_sampler(...)`, `torch.randn_like(x)`로 생성되는 텐서.",
        ],
        "pitfalls": [
            "독립 가정을 깨는 상관 난수를 쓰면 이론적 분산식이 바뀐다.",
        ],
    },
    "Fk": {
        "summary": r"$\mathcal{F}_k$는 ‘$k$번째 step 직전까지 어떤 난수들을 이미 봤는가’를 나타내는 정보 집합(필트레이션)이다. “미래 난수를 미리 보지 않는다”를 엄밀히 쓰면 $\mathcal{F}_k$-가측성으로 표현된다.",
        "why": [
            r"확률 과정에서 ‘현재까지의 정보로만 다음을 계산한다’는 말(적응성, adaptedness)을 수식으로 쓰려면, 시간에 따라 정보가 늘어나는 구조 $\{\mathcal{F}_k\}$가 필요하다. 샘플러 루프가 온라인 알고리즘이라는 사실을 엄밀히 기록하는 장치다.",
            r"ComfyUI에서 seed 재현성이 깨지는 흔한 이유 중 하나가 “중간에 RNG를 한 번 더(혹은 덜) 소비하는 부수 작업”이다(예: 디버그 출력, preview, 조건 분기). 이런 변화는 결국 $\xi_k$의 소비 순서를 바꾸고, 그 결과 $\mathcal{F}_k$ 관점에서 ‘무엇이 언제 알려졌는가’가 달라진다.",
            r"look-ahead(미래 난수 참조) 같은 버그/변형이 있으면 $\mathcal{F}_k$-적응성이 깨지고, SDE 해석(독립 증분 가정, 오차 분석)이 달라질 수 있다. 특히 adaptive step-size(accept/reject) 같은 로직을 넣을 때 이 문제가 더 쉽게 발생한다.",
            r"deterministic sampler에서는 난수 유입이 거의 없어서 필트레이션이 사실상 단순해지지만, ancestral/SDE에서는 매 step 새 난수 증분이 들어가므로 $\mathcal{F}_k$ 구조가 의미를 갖는다. “이 sampler는 어디에서 랜덤성을 쓰는가”를 체계적으로 설명할 때 도움이 된다.",
            r"영상/병렬 샘플링에서는 프레임/타일마다 RNG를 어떤 순서로 소비할지 규약이 필요하다. 이를 ‘정보의 시간 순서’로 보고 $\mathcal{F}_k$를 의식하면, 구현이 바뀌어도 재현성 규약을 유지하는 설계를 하기가 쉬워진다.",
        ],
        "formal": [
            r"$\mathcal{F}_k=\sigma(\xi_0,\dots,\xi_{k-1})$.",
            r"적응성은 $x_k$가 $\mathcal{F}_k$-가측임을 뜻한다.",
        ],
        "formal_explain": [
            r"첫 문장은 “$k$번째 step까지 어떤 난수들이 공개됐는가”를 $\sigma$-대수로 포장한다. $k$가 커질수록 더 많은 난수를 보니 정보가 늘어나고, 그래서 보통 $\mathcal{F}_0\subseteq\mathcal{F}_1\subseteq\cdots$가 된다.",
            r"둘째 문장은 적응성(adaptedness)의 정의다. $x_k$가 $\mathcal{F}_k$-가측이라는 것은 $x_k$가 $\xi_0,\dots,\xi_{k-1}$만으로 결정된다는 뜻이며, 미래 난수 $\xi_k,\xi_{k+1},\dots$를 참조하지 않는다는 의미다.",
            r"이 문법을 써두면 $\Phi_k$의 비예견성 조건을 엄밀히 쓸 수 있고, SDE의 strong/weak error 논의 같은 이론으로 자연스럽게 연결된다.",
        ],
        "intuition": [
            "‘지금까지 공개된 난수와 계산 결과를 모두 모아 둔 장부’라고 보면 된다.",
            r"$k$가 커질수록 더 많은 난수를 봤으니 $\mathcal{F}_k\subseteq\mathcal{F}_{k+1}$로 정보가 늘어난다.",
            r"“미래 난수를 미리 보지 않는다(no peeking)”를 엄밀히 쓰면 $x_k$가 $\mathcal{F}_k$-가측이라는 말이 된다. 즉 샘플러는 현재까지 생성된 난수에만 의존해야 한다(확률과정의 적응성/adaptedness).",
            r"코드 구현에서 난수를 ‘미리 한 번에 뽑아 배열로 저장’해도 된다. 그 경우에도 $k$번째 업데이트가 그 배열의 앞부분(0..k-1)만 참조하도록 만들면 $\mathcal{F}_k$-적응성을 만족한다(필트레이션을 구현으로 옮긴 것).",
        ],
        "examples": [
            r"예를 들어 $\xi_0=0.5$, $\xi_1=-1.2$가 뽑혔다면 $k=2$ 시점의 정보는 “$\xi_0=0.5$와 $\xi_1=-1.2$를 이미 봤다”를 포함한다.",
            r"$x_2$가 $\mathcal{F}_2$-가측이라는 말은 $x_2$가 $\xi_0,\xi_1$만으로 계산되고, $\xi_2$ 이후 값은 보지 않았다는 뜻이다.",
            r"(수치 예) $\Omega=\mathbb{R}^2$에 좌표 $(\xi_0,\xi_1)$를 두고 $\mathcal{F}_1=\sigma(\xi_0)$라 하자. 사건 $A=\{\xi_0>0\}$는 $\mathcal{F}_1$에 속하지만, $B=\{\xi_1>0\}$는 $\mathcal{F}_1$에 속하지 않는다(미래 성분).",
        ],
        "walkthrough": [
            {
                "stmt": r"\mathcal{F}_k=\sigma(\xi_0,\dots,\xi_{k-1})",
                "explain": [
                    r"$\sigma(\xi_0,\dots,\xi_{k-1})$는 “이 난수들로 결정되는 모든 사건”의 모음이다. 즉 $\xi_0,\dots,\xi_{k-1}$를 알면 판정할 수 있는 질문들의 집합이다.",
                    r"$k$가 증가하면 더 많은 난수를 포함하므로 정보가 늘어난다: $\mathcal{F}_k\subseteq\mathcal{F}_{k+1}$.",
                ],
                "intuition": [
                    r"“지금까지 나온 난수 로그”가 커질수록, 할 수 있는 판단(사건)이 많아진다.",
                ],
                "examples": [
                    r"$\xi_0=0.5$를 이미 봤다면, 사건 “$\xi_0>0$”는 $\mathcal{F}_1$에 속한다.",
                    r"$k=2$면 $\xi_0,\xi_1$을 보았으니, 사건 “$\xi_0+\xi_1>0$”도 $\mathcal{F}_2$에 속한다.",
                ],
            },
            {
                "stmt": r"x_k\ \text{is }\mathcal{F}_k\text{-measurable}",
                "explain": [
                    r"이는 $x_k$가 과거 정보만으로 계산된다는 뜻이다. 즉 $x_k=f(\xi_0,\dots,\xi_{k-1})$ 꼴로 쓸 수 있다는 직관과 맞닿아 있다.",
                    r"샘플러가 미래 노이즈를 미리 뽑아서 섞어 쓰면 이 조건이 깨질 수 있다(look-ahead).",
                ],
                "intuition": [
                    r"“미래를 보지 않는” 정상적인 온라인 알고리즘이라는 의미다.",
                ],
                "examples": [
                    r"$x_1=x_0+\alpha\xi_0$라면 $x_1$은 $\xi_0$만으로 결정되므로 $\mathcal{F}_1$-가측이다.",
                ],
            },
        ],
        "code_map": [
            "루프에서 매 step 새 난수를 뽑아 즉시 쓰는 구조가 필트레이션 적응성과 대응된다.",
        ],
        "pitfalls": [
            "미래 노이즈를 재사용하는 look-ahead 구현은 확률모형 가정을 위반할 수 있다.",
        ],
    },
    "Rho_t": {
        "summary": r"$\rho_t$는 시간 $t$에서의 상태 $x_t$가 ‘어떤 값들을 얼마나 자주 가지는지’를 나타내는 확률분포(확률측도)다. 개별 샘플 하나가 아니라 샘플 집단 전체를 보는 관점에서 등장한다.",
        "why": [
            r"ComfyUI 사용자는 보통 한 번에 한 장의 결과를 보지만, ‘품질’이나 ‘다양성’은 사실상 여러 seed에 걸친 결과의 분포적 성질이다. 예를 들어 어떤 sampler가 아티팩트를 자주 내는지, 특정 스타일이 얼마나 안정적으로 나오는지는 결국 최종 분포 $\rho_{t_{\mathrm{end}}}$의 성질로 표현하는 게 더 정확하다.",
            r"scheduler/sampler/CFG 선택은 초기 가우시안 분포에서 데이터 분포로 가는 ‘분포 경로’ 자체를 바꾼다. $\rho_t$를 도입하면 “이 선택은 분포를 어떻게 이동시키는가”를 한 언어로 말할 수 있고, 특히 스케줄 설계(어느 구간에 오차를 배분하는가)와 연결된다.",
            r"편집(inpaint/img2img)에서는 사실상 조건부 분포 $\rho_t(\cdot\mid c)$를 샘플링한다. 마스크/참조/프롬프트가 바뀌면 조건이 바뀌고, 그에 따라 분포가 바뀐다. 이를 ‘조건이 바뀐 같은 샘플링 문제’로 묶어 주는 관점이 $\rho_t$다.",
            r"영상 생성에서는 프레임별 독립 분포가 아니라 ‘프레임들의 결합분포’(상관 구조)가 중요해진다. 깜빡임은 프레임 간 상관이 약한 분포에서 쉽게 나타나며, 이를 해결하는 방법(공통 노이즈, 조건 공유 등)은 결국 $\rho_t$의 결합 구조를 설계하는 문제로 볼 수 있다.",
            r"이론적으로는 FPE가 $\rho_t$의 시간 진화를 PDE로 기술하고, OT는 $\rho_0\to\rho_1$ 수송을 최소 작용으로 기술한다. 커스텀 sampler를 ‘경로 수준’이 아니라 ‘분포 수준’에서 이해하고 싶다면 $\rho_t$가 출발점이다.",
        ],
        "formal": [
            r"$\rho_t\in\mathcal{P}_2(\mathcal{X})$ (2차 모멘트 유한 확률측도).",
            r"SDE의 전진 방정식은 $\partial_t\rho_t=-\nabla\cdot(\rho_t b_\theta)+\frac12\nabla\cdot((gg^\top)\nabla\rho_t)$.",
        ],
        "formal_explain": [
            r"첫 문장에서 $\mathcal{P}_2(\mathcal{X})$는 “2차 모멘트 $\int\|x\|^2\,d\rho(x)$가 유한한 확률분포들의 집합”이다. OT에서 $W_2$ 거리(2-와서슈타인)를 쓰려면 이 조건이 자연스럽게 등장한다.",
            r"둘째 문장은 SDE가 유도하는 분포의 시간 진화를 나타내는 PDE(FPE)다. drift는 질량을 ‘운반’하고($-\nabla\cdot(\rho b)$), diffusion은 질량을 ‘퍼뜨린다’(라플라시안 항).",
            r"샘플러를 분포 수준에서 이해하면, 개별 경로의 미세한 오차보다 “분포가 어디로 흘러가나”가 핵심이 된다(weak 관점).",
        ],
        "intuition": [
            r"많은 샘플을 찍어 히스토그램을 그렸을 때, 그 히스토그램이 시간에 따라 어떻게 변하는지가 $\rho_t$다.",
            r"ODE(결정론)면 분포가 ‘쓸려 가듯’ 이동하고, SDE(확률)면 추가로 퍼지는(확산) 효과가 생긴다.",
            r"ComfyUI에서 “같은 설정으로 seed만 바꿔 여러 장을 뽑는다”는 것은 최종 분포 $\rho_{t_{\mathrm{end}}}$에서 i.i.d. 샘플을 여러 개 얻는 행위다. 한 장 생성만으로는 $\rho_t$를 볼 수 없고, 여러 seed에 대한 집단 통계가 필요하다.",
            r"코드 구현 관점에선 $\rho_t$를 직접 저장하지는 못하므로(고차원), 보통은 관측 함수 $f:\mathcal{X}\to\mathbb{R}$를 정해 $f(x_t)$의 히스토그램/평균을 본다(몬테카를로). 예: 픽셀 평균, 노름 $\|x\|_2$, 특정 채널 분산 등.",
        ],
        "examples": [
            r"가장 쉬운 예로 $\mathcal{X}=\mathbb{R}$에서 $\rho_t=\mathcal{N}(0,\sigma(t)^2)$라고 하자. 예를 들어 $\sigma(t)=1-t$면 $t=0$에서 분산 1, $t=0.9$에서 분산 $0.1^2=0.01$이다.",
            r"샘플을 3개만 보자: $t=0$에서 $x$가 (예시로) $-0.2, 0.5, 1.1$처럼 나왔다면, $\rho_0$는 ‘이런 값들이 나올 수 있다’는 분포를 요약한다.",
            r"(관측량 예) $\mathcal{X}=\mathbb{R}^2$에서 $f(x)=\|x\|_2$라 하자. seed 3개로 $x^{(1)}=(0.3,0.4)$, $x^{(2)}=(1.0,0.0)$, $x^{(3)}=(-0.6,0.8)$를 얻으면 $f$ 값은 $(0.5,1.0,1.0)$이고, 평균은 $(0.5+1.0+1.0)/3\approx0.833$이다(분포를 관측량으로 요약).",
        ],
        "walkthrough": [
            {
                "stmt": r"\rho_t\in\mathcal{P}_2(\mathcal{X})",
                "explain": [
                    r"$\rho_t$는 시간 $t$에서의 확률분포다. 확률분포는 “집합 $A$가 일어날 확률”을 주는 객체(측도)로 생각할 수 있다.",
                    r"$\mathcal{P}_2$ 조건은 OT(특히 $W_2$)를 쓰기 위한 기술적 조건으로, 분산이 무한대인 분포는 제외한다는 의미다.",
                ],
                "intuition": [
                    r"샘플 하나하나는 랜덤하지만, 많이 모으면 ‘분포의 모양’이 보인다. 그 모양이 $\rho_t$다.",
                ],
                "examples": [
                    r"$\mathbb{R}$에서 $\rho_t=\mathcal{N}(0,1)$이면 $\int x^2\,d\rho_t(x)=1$로 유한하므로 $\rho_t\in\mathcal{P}_2(\mathbb{R})$.",
                ],
            },
            {
                "stmt": r"\partial_t\rho_t=-\nabla\cdot(\rho_t b_\theta)+\frac12\nabla\cdot((gg^\top)\nabla\rho_t)",
                "explain": [
                    r"첫 항 $-\nabla\cdot(\rho b)$는 “속도장 $b$에 의해 질량이 이동한다”를 의미한다(연속방정식의 형태).",
                    r"둘째 항은 확산에 의한 퍼짐을 나타낸다. $g=0$이면 이 항이 사라져 결정론적 수송 방정식으로 줄어든다.",
                ],
                "intuition": [
                    r"drift는 흐름(flow), diffusion은 퍼짐(spread)이다. 두 효과가 합쳐져 분포가 진화한다.",
                ],
                "examples": [
                    r"1차원에서 $b(x)=-x$, $g=0$이면 분포는 원점으로 수축하는 흐름을 따른다.",
                    r"1차원에서 $b=0$, $g$ 상수면 분포는 열방정식처럼 점점 퍼진다.",
                ],
            },
        ],
        "code_map": [
            r"실험에서는 히스토그램/feature 통계로 $\rho_t$ 변화를 간접 추정한다.",
        ],
        "pitfalls": [
            "개별 trajectory 오차와 분포 오차(weak error)는 다른 개념이다.",
        ],
    },
    "V_t": {
        "summary": r"$v_t$는 시간 $t$에서 ‘분포 $\rho_t$가 어느 방향으로 흘러가는지’를 나타내는 속도장(벡터장)이다. 결정론적(OT/연속방정식) 관점에서 샘플링을 볼 때 자연스럽게 등장한다.",
        "why": [
            r"결정론적 샘플링(특히 probability-flow ODE 관점)에서는 ‘분포가 흐르는 속도’가 핵심이며, 그 속도장이 $v_t$다. ComfyUI의 많은 non-ancestral sampler를 “어떤 속도장을 어떤 수치적분기로 따라가는가”로 해석할 수 있어, solver 비교가 정리된다.",
            r"수치해석적으로는 $x_{k+1}\approx x_k+h\,v(t_k,x_k)$ 같은 형태가 기본이고, Euler/Heun/다단계 방법은 이 속도장을 어떻게 근사하고 어느 점에서 평가하느냐가 차이를 만든다. 즉 sampler의 ‘차수/안정성’은 결국 $v_t$의 적분 근사 품질로 환원된다.",
            r"CFG는 조건부/비조건부 출력을 섞어 벡터장을 바꾼다. 따라서 `cfg`를 조절하는 것은 사실상 $v_t$를 스케일링/변형하는 것과 같고, 과도한 `cfg`에서 생기는 과포화/왜곡을 “벡터장 과증폭”으로 해석할 수 있다(필요하면 CFG-rescale 같은 보정이 여기서 등장).",
            r"OT 관점에서는 $v_t$가 질량을 옮기는 ‘수송 정책’이고, Benamou-Brenier의 action $\int\|v_t\|^2\,d\rho_t\,dt$를 최소화하는 흐름이 가장 ‘부드러운 이동’이다. 영상에서 깜빡임을 줄이려면 프레임 간 분포 수송이 매끄러워야 하는데, 이 역시 $v_t$를 부드럽게 만드는 문제로 정리된다.",
            r"SDE/FPE 관점에서는 확산항이 있어 연속방정식에 추가 항이 붙지만, 여전히 drift/score로부터 유도되는 ‘유효한 흐름’(혹은 probability-flow ODE)을 통해 $v_t$를 정의해 볼 수 있다. 그래서 $v_t$는 ODE/SDE/OT를 한데 묶어 주는 공통 언어로 쓸 수 있다.",
        ],
        "formal": [
            r"$v_t\in L^2_{\rho_t}(\mathcal{X};\mathcal{X})$.",
            r"연속방정식 $\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0$를 만족한다.",
        ],
        "formal_explain": [
            r"첫 문장은 $v_t$가 ‘분포 $\rho_t$에 대해 제곱적분 가능한 벡터장’이라는 뜻이다. 즉 $\int\|v_t(x)\|^2\,d\rho_t(x)$가 유한하다고 가정한다.",
            r"둘째 문장은 질량 보존의 PDE다. $\rho_t$가 속도장 $v_t$에 의해 이동(수송)할 때 밀도가 만족해야 하는 방정식이 연속방정식이다.",
            r"OT 관점에서는 $v_t$를 선택해서 $\rho_0$에서 $\rho_1$로 질량을 옮기되, $\int\|v_t\|^2\,d\rho_t\,dt$ 비용을 최소화하는 문제가 핵심으로 등장한다.",
        ],
        "intuition": [
            r"공간의 각 위치 $x$에 대해 ‘그 자리의 밀도가 어느 쪽으로 움직이는가’를 주는 화살표장이다.",
            r"샘플러를 매우 작은 step으로 보면, 개별 샘플도 대략 $dx_t/dt=v_t(x_t)$를 따른다고 해석할 수 있다(정확한 동일시는 아님).",
            r"ComfyUI의 결정론 sampler를 아주 작은 step-size로 보면, 한 step 업데이트는 $x\mapsto x+\Delta t\,v_t(x)$ 같은 형태(Euler/Heun 등)로 해석할 수 있다. 즉 “sampler 수식”을 “속도장 적분”으로 읽으면 수치해석적 비교(차수/안정성)가 쉬워진다.",
            r"OT(optimal transport) 관점에서는 $v_t$가 질량을 옮기는 수송 정책이며, action $\int\|v_t\|^2\,d\rho_t\,dt$가 작을수록 ‘부드러운 이동’이다. 영상에서 프레임 간 깜빡임을 줄이는 목표는 결국 분포 수송을 부드럽게(작은 action) 만드는 문제로 연결된다.",
        ],
        "examples": [
            r"(1차원) $v_t(x)=-x$라 하자. $x=2.0$에선 속도 $v=-2.0$이므로 왼쪽으로 당겨진다. Euler로 $\Delta t=0.1$이면 $x_{\text{new}}=2.0+0.1\cdot(-2.0)=1.8$.",
            r"(OT 비용 예) 동적 OT에서 action은 $\int_0^1\int_{\mathcal{X}}\|v_t(x)\|_2^2\,d\rho_t(x)\,dt$로 주어진다.",
            r"(2차원 수치 예) $v(x)=(-x_1,-2x_2)$라 하자. $x=(1.0,0.5)$면 $v(x)=(-1.0,-1.0)$이고, Euler $\Delta t=0.2$면 $x_{\text{new}}=(1.0,0.5)+0.2(-1.0,-1.0)=(0.8,0.3)$이다.",
        ],
        "walkthrough": [
            {
                "stmt": r"\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0",
                "explain": [
                    r"이 식은 ‘질량 보존’을 의미한다. 어떤 영역 $A$에 들어 있는 확률질량은 경계로 흐르는 유량에 의해만 변한다.",
                    r"속도장 $v_t$가 주어지면, 분포 $\rho_t$는 이 PDE를 만족하며 이동한다(결정론적 수송).",
                ],
                "intuition": [
                    r"물(밀도)이 바람(속도장)을 따라 흐른다고 생각하면 된다.",
                ],
                "examples": [
                    r"1차원에서 $v(x)=-x$면 질량이 원점으로 모이는 흐름이다. $x=2$는 왼쪽으로, $x=-2$는 오른쪽으로 움직인다.",
                ],
            },
            {
                "stmt": r"\int_0^1\!\int_{\mathcal{X}}\|v_t(x)\|_2^2\,d\rho_t(x)\,dt",
                "explain": [
                    r"이 적분은 “분포를 옮기는 데 든 에너지(작용)”로 해석된다. OT의 Benamou-Brenier 정식화에서 이 비용을 최소화하는 $v_t$가 최적수송 경로를 만든다.",
                    r"$\rho_t$가 주어지면, $v_t$가 너무 크면 비용이 커진다. 그래서 OT는 ‘가능하면 짧고 부드럽게 옮기기’를 선호한다.",
                ],
                "intuition": [
                    r"질량을 빨리/세게 옮기면 에너지가 많이 든다. OT는 그 에너지를 최소화하는 이동을 고른다.",
                ],
                "examples": [
                    r"아주 단순히 $\mathcal{X}=\mathbb{R}$, $\rho_t$가 특정 점 근처에 몰려 있고 $v_t$가 상수 1이라면 비용은 대략 $1^2$을 시간에 대해 적분한 값($\approx 1$) 수준이 된다(정확한 값은 $\rho_t$에 의존).",
                ],
            },
        ],
        "code_map": [
            "deterministic sampler의 drift 근사량이 이 속도장에 대응된다.",
        ],
        "pitfalls": [
            "샘플 단위 업데이트와 분포 수준 속도장을 1:1 동일시하면 해석이 과도 단순화된다.",
        ],
    },
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8-sig")


def css_text() -> str:
    return """
:root{
  --bg:#f4f7ef;--paper:#fffefb;--ink:#161811;--muted:#5b6252;--line:#d2d9c3;
  --accent:#175747;--soft:#e8f3ee;--code:#edf2df;
}
*{box-sizing:border-box}
body{
  margin:0;color:var(--ink);line-height:1.72;
  font-family:"IBM Plex Sans KR","Pretendard","Noto Sans KR",sans-serif;
  background:
    radial-gradient(circle at 8% 8%,#deedd9 0%,transparent 34%),
    radial-gradient(circle at 92% 0%,#dcecf1 0%,transparent 36%),
    var(--bg);
}
.wrap{max-width:1180px;margin:26px auto;padding:0 16px}
.paper{
  background:var(--paper);border:1px solid var(--line);border-radius:16px;
  box-shadow:0 10px 24px rgba(0,0,0,.06);padding:22px;
}
h1,h2,h3{margin:0;font-family:"Source Serif 4","Noto Serif KR",serif;line-height:1.34}
h1{font-size:1.95rem;margin-bottom:10px}
h2{margin-top:22px;border-top:1px solid var(--line);padding-top:12px;font-size:1.36rem;color:#173e31}
h3{margin-top:16px;font-size:1.08rem;color:#1b3551}
p{margin:8px 0}
.muted{color:var(--muted)}
code{
  background:var(--code);border-radius:6px;padding:2px 6px;
  font-family:"Cascadia Code","Consolas",monospace;font-size:.92em;
}
.formula{
  border-left:4px solid var(--accent);background:#ecf7f2;border-radius:8px;
  padding:10px 12px;margin:10px 0;
}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.box{border:1px solid var(--line);border-radius:10px;background:#fff;padding:10px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.chip{
  border:1px solid #97b8a8;border-radius:999px;background:var(--soft);
  color:#104d3c;padding:3px 10px;font-size:.84rem;
}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{border:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}
th{background:#edf3df}
/* MathJax 간격 보정: 표 셀 안 display 수식의 과도한 여백을 줄임 */
td mjx-container[display="true"]{
  margin:.18em 0 !important;
  text-align:left !important;
}
td mjx-container{
  font-size:1em !important;
}
.topnav{
  display:flex;flex-wrap:wrap;gap:8px;margin:0 0 12px 0;
}
.btn{
  display:inline-block;text-decoration:none;color:#174836;background:#e8f3ec;
  border:1px solid #9fbbaa;border-radius:9px;padding:7px 10px;font-size:.89rem;
}
.btn:hover{filter:brightness(.98)}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.tab-btn{
  appearance:none;border:1px solid #9fbbaa;background:#eef5ea;color:#19453a;
  border-radius:10px;padding:7px 11px;font-size:.89rem;cursor:pointer;
  font-family:"IBM Plex Sans KR","Pretendard","Noto Sans KR",sans-serif;
}
.tab-btn:hover{filter:brightness(.98)}
.tab-btn.active{background:#175747;color:#fff;border-color:#175747}
.tabs.mini .tab-btn{padding:5px 9px;font-size:.82rem;border-radius:9px}
.tab-panel{display:none}
.tab-panel.active{display:block}
.path{font-size:.87rem;color:#2d4b40}
.small{font-size:.86rem;color:var(--muted)}
.codebox{
  border:1px solid var(--line);border-radius:10px;background:#f2f6e9;
  padding:10px;white-space:pre-wrap;font-family:"Cascadia Code","Consolas",monospace;
  font-size:.83rem;max-height:260px;overflow:auto;
}
.list{margin:8px 0 8px 20px;padding:0}
.list li{margin:4px 0}
.card-grid{display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:10px}
.card{border:1px solid var(--line);border-radius:10px;background:#fff;padding:10px}
.card h3{margin:0 0 6px 0}
.filter{
  display:grid;grid-template-columns:1fr 220px 170px;gap:8px;margin:10px 0;
}
.filter input,.filter select{
  border:1px solid #bac9af;border-radius:8px;padding:8px 9px;font-size:.92rem;background:#fdfefb;
}
.row-link{text-decoration:none;color:inherit}
@media (max-width:980px){
  .grid2,.card-grid{grid-template-columns:1fr}
  .filter{grid-template-columns:1fr}
  table{display:block;overflow-x:auto;white-space:nowrap}
}
"""


def html_head(title: str) -> str:
    return f"""<!doctype html>
<html lang="ko"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{esc(title)}</title>
<link rel="stylesheet" href="../assets/style.css"/>
<script>
window.MathJax={{
  tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}},
  options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}
}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head><body><div class="wrap"><article class="paper">"""


def html_tail() -> str:
    return "</article></div></body></html>"


def symbol_href(symbol_id: str, prefix: str = "") -> str:
    return f"{prefix}symbol/{symbol_id}.html"


def symbol_link(symbol_id: str, tex: str, prefix: str = "") -> str:
    return f'<a class="row-link" href="{esc(symbol_href(symbol_id, prefix))}">${tex}$</a>'


def symbol_tex(symbol_id: str) -> str:
    item = next((s for s in SYMBOL_WIKI if s["id"] == symbol_id), None)
    return item["tex"] if item is not None else symbol_id


def symbol_link_by_id(symbol_id: str, prefix: str = "") -> str:
    return symbol_link(symbol_id, symbol_tex(symbol_id), prefix)


def symbol_guide_entry(item: dict[str, Any]) -> dict[str, Any]:
    guide = SYMBOL_GUIDE.get(item["id"], {})
    return {
        "summary": guide.get("summary", item.get("name", "")),
        "why": guide.get("why", []),
        "formal": guide.get("formal", []),
        "formal_explain": guide.get("formal_explain", []),
        "intuition": guide.get("intuition", []),
        "examples": guide.get("examples", []),
        "walkthrough": guide.get("walkthrough", []),
        "code_map": guide.get("code_map", []),
        "pitfalls": guide.get("pitfalls", []),
    }


def render_inline_text(text: str) -> str:
    # backtick 구간만 code 태그로 감싸고, 나머지는 escape 처리한다.
    parts = text.split("`")
    chunks: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            chunks.append(f"<code>{esc(part)}</code>")
        else:
            chunks.append(esc(part))
    return "".join(chunks)


def render_ordered_list(items: list[str], empty_text: str = "내용 없음") -> str:
    if not items:
        return f"<p class='small'>{esc(empty_text)}</p>"
    rows = "".join(f"<li>{render_inline_text(x)}</li>" for x in items)
    return f"<ol class='list'>{rows}</ol>"


def render_paragraphs(items: list[str], empty_text: str = "내용 없음") -> str:
    if not items:
        return f"<p class='small'>{esc(empty_text)}</p>"
    return "".join(f"<p>{render_inline_text(x)}</p>" for x in items)


def formal_block_label(block: dict[str, Any], idx: int) -> str:
    kind = str(block.get("kind") or "정의")
    title = str(block.get("title") or "")
    label = f"{kind} {idx}"
    if title:
        label += f" ({title})"
    return label


def render_formal_only(blocks: list[dict[str, Any]]) -> str:
    """엄밀 정의 섹션: 수식(정의/정리 문장)만 출력한다.

    자연어 해설은 '정의 해설' 섹션으로 분리한다.
    """
    if not blocks:
        return "<p class='small'>엄밀 정의 메모 없음</p>"
    html: list[str] = []
    for idx, b in enumerate(blocks, start=1):
        stmt = str(b.get("stmt") or "")
        label = formal_block_label(b, idx)
        html.append("<div class='box'>")
        html.append(f"<p class='small'><strong>{esc(label)}.</strong></p>")
        if stmt:
            html.append(f"<div class='formula'>$$ {stmt} $$</div>")
        else:
            html.append("<p class='small'>정의식(수식)이 비어 있습니다.</p>")
        html.append("</div>")
    return "".join(html)


def render_definition_walkthrough(blocks: list[dict[str, Any]]) -> str:
    if not blocks:
        return "<p class='small'>정의 해설 없음</p>"
    html = []
    for idx, b in enumerate(blocks, start=1):
        stmt = b.get("stmt", "")
        explain = b.get("explain", [])
        intuition = b.get("intuition", [])
        examples = b.get("examples", [])
        html.append("<div class='box'>")
        html.append(f"<p class='small'><strong>{esc(formal_block_label(b, idx))}.</strong></p>")
        if stmt:
            html.append(f"<div class='formula'>$$ {stmt} $$</div>")
        if explain:
            html.append("<h3>해설</h3>")
            html.append(render_ordered_list(explain, ""))
        if intuition:
            html.append("<h3>직관(정의와 연결)</h3>")
            html.append(render_ordered_list(intuition, ""))
        if examples:
            html.append("<h3>수치 예시</h3>")
            html.append(render_ordered_list(examples, ""))
        html.append("</div>")
    return "".join(html)


def render_symbol_index_page() -> str:
    header = html_head("Symbol Wiki Index")
    cards = []
    for item in SYMBOL_WIKI:
        guide = symbol_guide_entry(item)
        cards.append(
            "<article class='card'>"
            f"<h3><a class='row-link' href='{esc(item['id'])}.html'>${item['tex']}$</a></h3>"
            f"<p><strong>{esc(item['name'])}</strong></p>"
            f"<p class='small'>{render_inline_text(str(guide['summary']))}</p>"
            "</article>"
        )
    body = (
        '<div class="topnav">'
        '<a class="btn" href="../index.html#notation">최상위 문서(기호 탭)</a>'
        "</div>"
        "<h1>Symbol Wiki</h1>"
        "<p class='muted'>기호별 상세 문서를 위키처럼 분리했습니다. "
        "각 페이지는 <strong>모티베이션 → 엄밀 정의(정의식) → 정의 해설 → 직관/예시 → 코드 대응</strong> 순서로 읽히게 구성했습니다.</p>"
        "<div class='card-grid'>"
        f"{''.join(cards)}"
        "</div>"
    )
    return header + body + html_tail()


def render_symbol_page(item: dict[str, Any], prev_id: str | None, next_id: str | None) -> str:
    header = html_head(f"Symbol Wiki: {item['id']}")
    guide = symbol_guide_entry(item)

    related = []
    for rid in item.get("related", []):
        target = next((s for s in SYMBOL_WIKI if s["id"] == rid), None)
        if target is None:
            continue
        related.append(
            f'<a class="btn" href="{esc(target["id"])}.html">${target["tex"]}$</a>'
        )

    motivation_intro = render_inline_text(str(guide["summary"]))
    motivation_rows = render_ordered_list(guide["why"], "모티베이션 메모 없음")
    # 엄밀 정의는 자연어 문장 대신, '정의식(수식)' 블록만 출력한다.
    formal_rows = render_formal_only(guide.get("walkthrough", []))
    formal_explain = render_paragraphs(
        guide["formal_explain"],
        "엄밀 정의의 문장을 직관/예시로 푸는 해설을 추가할 수 있습니다.",
    )
    intuition_rows = render_ordered_list(guide["intuition"], "직관 메모 없음")
    example_rows = render_ordered_list(guide["examples"], "구체 예시 메모 없음")
    walkthrough = render_definition_walkthrough(guide["walkthrough"])
    code_rows = render_ordered_list(guide["code_map"], "코드 대응 메모 없음")
    pitfalls_rows = render_ordered_list(guide["pitfalls"], "오해 포인트 메모 없음")

    rel_nav = ['<div class="topnav">']
    if prev_id is not None:
        prev_item = next(s for s in SYMBOL_WIKI if s["id"] == prev_id)
        rel_nav.append(f'<a class="btn" href="{esc(prev_id)}.html">이전: ${prev_item["tex"]}$</a>')
    if next_id is not None:
        next_item = next(s for s in SYMBOL_WIKI if s["id"] == next_id)
        rel_nav.append(f'<a class="btn" href="{esc(next_id)}.html">다음: ${next_item["tex"]}$</a>')
    rel_nav.append("</div>")

    body = (
        '<div class="topnav">'
        '<a class="btn" href="index.html">기호 위키 인덱스</a>'
        '<a class="btn" href="../index.html#notation">최상위 문서(기호 탭)</a>'
        "</div>"
        f"<h1>Symbol: ${item['tex']}$</h1>"
        f"<p class='muted'><strong>{esc(item['name'])}</strong></p>"
        "<h2>모티베이션 (Motivation)</h2>"
        f"<p>{motivation_intro}</p>"
        f"{motivation_rows}"
        "<h2>엄밀 정의 (Formal Definitions)</h2>"
        f"{formal_rows}"
        "<h2>정의 해설 (Commentary)</h2>"
        "<h3>해설(요약) (Summary)</h3>"
        f"{formal_explain}"
        f"{walkthrough}"
        "<h2>직관(요약) (Intuition)</h2>"
        f"{intuition_rows}"
        "<h2>구체 원소 예시(모음) (Concrete Examples)</h2>"
        f"{example_rows}"
        "<h2>코드 대응 (Code Mapping)</h2>"
        f"{code_rows}"
        "<h2>자주 헷갈리는 점 (Pitfalls)</h2>"
        f"{pitfalls_rows}"
        "<h2>관련 기호 (Related Symbols)</h2>"
        f"<div class='topnav'>{''.join(related) if related else '<span class=\"small\">관련 기호 없음</span>'}</div>"
        + "".join(rel_nav)
    )
    return header + body + html_tail()


def sampler_numeric_note(name: str, family: str, stochastic: str) -> str:
    if name in {"euler", "euler_ancestral", "euler_cfg_pp", "euler_ancestral_cfg_pp"}:
        return "Euler 계열은 1차 근사(국소 오차 O(h^2), 전역 O(h)) 해석이 기본이며, ancestral/SDE 변형은 통계적 분산 항이 추가된다."
    if name in {"heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp"}:
        return "2차 보정이 들어가는 계열로, 동일 step에서 Euler보다 drift 근사 bias가 작다. 다만 SDE 변형은 noise 항 영향으로 분산 trade-off가 발생한다."
    if name in {"dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu"}:
        return "2M 계열은 이전 step 정보를 활용하는 반-다단계 구조. solver_type(midpoint/heun)에 따라 correction 항의 안정성/정확도 특성이 달라진다."
    if name in {"dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "seeds_3"}:
        return "3-stage/3M 계열로 higher-order correction을 사용한다. 낮은 step에서도 구조 보존성이 좋지만 구현/튜닝 민감도가 높다."
    if name in {"lms", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "ipndm", "ipndm_v", "deis"}:
        return "이력 기반 다단계 적분. 초기 워밍업 구간은 저차로 시작하고, 이후 이력 버퍼가 쌓이면서 고차 근사가 활성화된다."
    if name in {"dpm_adaptive"}:
        return "적응형 step-size 제어(PID + local error test). rtol/atol과 accept_safety가 품질-시간 Pareto를 결정한다."
    if name in {"ddpm", "lcm"}:
        return "Markov 성분이 강한 확률 경로 해석이 유용하며, deterministic ODE 솔버와 다른 분산 거동을 보인다."
    if name in {"er_sde", "sa_solver", "sa_solver_pece", "seeds_2"}:
        return "SDE 특화 계열. drift 항보다 stochastic design(eta/s_noise/tau_func/noise_scaler)이 출력 분산과 질감에 큰 영향을 준다."
    if name in {"uni_pc", "uni_pc_bh2"}:
        return "UniPC 다항 보정 계열. k-diffusion 구현과 별개 경로이므로 내부 보정 로직과 step control 규약을 별도로 보는 것이 좋다."
    return f"{family} 계열 {'stochastic' if stochastic == 'yes' else 'deterministic'} 해석."


def fpe_ot_note(stochastic: str) -> str:
    if stochastic == "yes":
        return "FPE 관점에서 drift + diffusion가 모두 활성화된다: $\\partial_t\\rho=-\\nabla\\cdot(\\rho b)+\\frac12 g^2\\Delta\\rho$. OT 관점에서는 entropic regularization이 있는 bridge 해석이 자연스럽다."
    return "결정론 경로로 보면 확산항이 제거된 continuity equation 관점: $\\partial_t\\rho+\\nabla\\cdot(\\rho v)=0$. 동적 OT(Benamou-Brenier) 형태의 수송 해석이 용이하다."


def common_param_rows(name: str) -> list[tuple[str, str, str]]:
    if name in {"dpm_fast", "dpm_adaptive"}:
        return [
            ("model", "denoiser callable", "모델 함수"),
            ("x", "latent", "현재 상태"),
            ("sigma_min/sigma_max", "integration interval", "적분 구간 경계"),
            ("extra_args", "dict", "추가 인자"),
            ("callback", "callable", "스텝 콜백"),
            ("disable", "bool", "progress disable"),
        ]
    if name.startswith("uni_pc"):
        return [
            ("model", "denoiser callable", "모델 함수"),
            ("noise", "latent/noise", "초기 상태"),
            ("sigmas", "sigma schedule", "스케줄"),
            ("extra_args", "dict", "추가 인자"),
            ("callback", "callable", "스텝 콜백"),
            ("disable", "bool", "progress disable"),
        ]
    return [
        ("model", "denoiser callable", "모델 함수"),
        ("x", "latent", "현재 상태"),
        ("sigmas", "sigma schedule", "스텝별 노이즈 스케일"),
        ("extra_args", "dict", "seed/model_options 등 추가 인자"),
        ("callback", "callable", "진행 콜백"),
        ("disable", "bool", "progress disable"),
    ]


def sampler_order_profile(name: str) -> dict[str, str]:
    if name in {"euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp"}:
        return {
            "class": "단일 스텝 Euler/EM",
            "local": r"O(h^2)",
            "global": r"O(h)",
            "strong_weak": "SDE 변형에서는 strong order ~0.5, weak order ~1(전형적 EM 해석)",
            "stability": "작은 step에서는 안정적이나, 거친 스텝에서 bias가 빠르게 증가",
        }
    if name in {"heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp"}:
        return {
            "class": "2차 predictor-corrector",
            "local": r"O(h^3)",
            "global": r"O(h^2)",
            "strong_weak": "확률항이 있으면 강한 차수는 낮아질 수 있으나 평균 drift 근사 정확도는 우수",
            "stability": "Euler 대비 drift 오차가 작고, 중간 stage 평가 덕분에 곡률 변화에 상대적으로 강함",
        }
    if name in {"dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu"}:
        return {
            "class": "2-step multistep / exponential integrator",
            "local": r"O(h^3)\ \text{(smooth regime)}",
            "global": r"O(h^2)",
            "strong_weak": "SDE 항이 있으면 diffusion 분산은 유지, correction은 deterministic bias 완화에 기여",
            "stability": "history 기반 보정으로 저스텝에서도 비교적 안정적인 품질",
        }
    if name in {"dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "seeds_3"}:
        return {
            "class": "3-stage/3M high-order",
            "local": r"O(h^4)\ \text{(근사적)}",
            "global": r"O(h^3)\ \text{(근사적)}",
            "strong_weak": "고차 correction으로 drift 편향은 낮지만 step/스케줄 민감도가 높음",
            "stability": "충분한 step에서 구조 보존 우수, 거친 설정에서는 진동 가능성",
        }
    if name in {"lms", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "ipndm", "ipndm_v", "deis"}:
        return {
            "class": "history 기반 multistep",
            "local": r"O(h^{m+1})",
            "global": r"O(h^m)",
            "strong_weak": "초기 warm-up 구간에서는 유효 차수가 낮고 이후 history 축적으로 상승",
            "stability": "메쉬 불균일이 크면 계수 조건수가 악화될 수 있어 스케줄과 동시 튜닝 필요",
        }
    if name == "dpm_adaptive":
        return {
            "class": "adaptive DPM solver",
            "local": r"\|e_k\|\le \text{atol} + \text{rtol}\cdot\|x_k\|",
            "global": "허용오차 기반으로 자동 제어(고정 차수 표현보다 tolerance 해석이 적합)",
            "strong_weak": "accept/reject와 PID 계수가 계산량-오차 균형을 직접 제어",
            "stability": "accept_safety와 초기 step(h_init) 설정이 수렴/속도의 핵심",
        }
    if name in {"ddpm", "lcm"}:
        return {
            "class": "Markov chain 계열",
            "local": "고전 ODE 차수보다는 transition kernel 오차 관점이 적합",
            "global": "step 수 증가에 따라 KL/score mismatch 누적 감소",
            "strong_weak": "확률적 transition의 분포 근사 정확도가 핵심",
            "stability": "노이즈 주입 구조 덕분에 deterministic solver와 다른 형태의 안정성",
        }
    if name in {"er_sde", "sa_solver", "sa_solver_pece", "seeds_2"}:
        return {
            "class": "stochastic PC/SDE 특화",
            "local": r"O(h^{p+1})\ \text{(drift)} + O(h^{q+1/2})\ \text{(diffusion)}",
            "global": "weak error 중심으로 해석하는 것이 실용적",
            "strong_weak": "noise 설계(eta, s_noise, tau_func, noise_scaler)가 지배적",
            "stability": "확산항 스케일이 과하면 질감은 증가하지만 구조 안정성은 저하",
        }
    if name in {"uni_pc", "uni_pc_bh2"}:
        return {
            "class": "Unified predictor-corrector",
            "local": "단계/보정 차수에 따라 가변",
            "global": "저스텝 품질을 노린 고차 보정 경로",
            "strong_weak": "구현 경로가 k-diffusion과 별개라 내부 보정 항 해석을 독립적으로 보는 것이 안전",
            "stability": "step 수가 매우 작을 때의 안정화 목적이 강함",
        }
    return {
        "class": "generic",
        "local": r"O(h^2)",
        "global": r"O(h)",
        "strong_weak": "구체 구현에 의존",
        "stability": "스케줄/노이즈 설정에 의존",
    }


def param_symbol(param: str) -> tuple[str, str, str]:
    table = {
        "model": (r"\hat{x}_0(\cdot;\theta,c)", "drift 항", "denoiser/score 기반 추정기"),
        "x": (r"x_k", "상태 변수", "현재 latent 상태"),
        "noise": (r"x_0 \text{ or } \xi_0", "초기 조건", "초기 노이즈/초기 latent"),
        "sigmas": (r"\{\sigma_k\}_{k=0}^{N}", "시간 재매개화", "노이즈 스케줄 격자"),
        "sigma_min/sigma_max": (r"\sigma_{\min},\sigma_{\max}", "적분 구간", "경계값"),
        "extra_args": (r"c,\ \text{options}", "조건 벡터장", "conditioning/옵션 전달"),
        "callback": (r"\mathcal{C}_k", "관측 함수", "수치 궤적 모니터링"),
        "disable": ("-", "UI/로그 제어", "수학 항에는 직접 미참여"),
        "eta": (r"\eta", "diffusion 강도", "노이즈 주입 강도 및 drift 감쇠와 결합"),
        "s_noise": (r"s_{\text{noise}}", "noise 스케일", r"\xi_k \mapsto s_{\text{noise}}\xi_k"),
        "s_churn": (r"\gamma", "sigma inflation", r"\hat{\sigma}=(1+\gamma)\sigma"),
        "s_tmin": (r"\sigma_{\text{low}}", "churn 적용 구간", "노이즈 주입 하한"),
        "s_tmax": (r"\sigma_{\text{high}}", "churn 적용 구간", "노이즈 주입 상한"),
        "r": (r"r\in(0,1)", "중간 stage 위치", "2-stage 비율"),
        "r_1": (r"r_1", "중간 stage1", "SEEDS-3 첫 stage"),
        "r_2": (r"r_2", "중간 stage2", "SEEDS-3 둘째 stage"),
        "solver_type": (r"\psi\in\{\text{midpoint, heun}\}", "보정 연산자", "corrector 식 선택"),
        "order": (r"m", "다단계 차수", "히스토리 계수 차수"),
        "max_order": (r"m_{\max}", "최대 차수", "적응형/다단계 상한"),
        "rtol": (r"\varepsilon_{\text{rel}}", "오차 기준", "상대오차 허용치"),
        "atol": (r"\varepsilon_{\text{abs}}", "오차 기준", "절대오차 허용치"),
        "h_init": (r"h_0", "초기 스텝", "적응형 시작 크기"),
        "pcoeff": (r"K_P", "PID 제어", "오차 기반 step 조정"),
        "icoeff": (r"K_I", "PID 제어", "누적 오차 반영"),
        "dcoeff": (r"K_D", "PID 제어", "오차 변화율 반영"),
        "accept_safety": (r"s_{\text{acc}}", "수락 조건", "보수적 수락 계수"),
        "ge_gamma": (r"\gamma_{\text{GE}}", "gradient blending", "추정 gradient 혼합 강도"),
        "noise_scaler": (r"c_g", "diffusion 계수", "ER-SDE 노이즈 스케일"),
        "max_stage": (r"M", "stage 수", "다단계 SDE stage 상한"),
        "tau_func": (r"\tau(t)", "stochastic interval", "노이즈 적용 구간 함수"),
        "predictor_order": (r"p", "predictor 차수", "예측 단계 차수"),
        "corrector_order": (r"q", "corrector 차수", "보정 단계 차수"),
        "use_pece": (r"\mathbf{1}_{\mathrm{PECE}}", "PECE 토글", "predict-evaluate-correct-evaluate"),
        "simple_order_2": (r"\mathbf{1}_{2nd}", "2차 단순화", "order-2 단순 모드"),
        "deis_mode": (r"\mathcal{M}_{\mathrm{DEIS}}", "계수 선택", "DEIS coefficient 모드"),
    }
    return table.get(param, (param, "구현 의존", "sampler-specific tuning parameter"))


def render_parameter_symbol_table(common_rows: list[tuple[str, str, str]], param_rows: list[dict[str, str]]) -> str:
    names: list[str] = []
    for n, _, _ in common_rows:
        names.append(n)
    for p in param_rows:
        names.append(p["name"])
    seen: set[str] = set()
    uniq = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        uniq.append(n)

    rows = []
    for n in uniq:
        sym, where, note = param_symbol(n)
        rows.append(
            "<tr>"
            f"<td><code>{esc(n)}</code></td>"
            f"<td>$$ {sym} $$</td>"
            f"<td>{esc(where)}</td>"
            f"<td>{esc(note)}</td>"
            "</tr>"
        )
    return (
        "<h2>파라미터-수식 기호 대응</h2>"
        "<table><thead><tr><th>코드 파라미터</th><th>수식 기호</th><th>들어가는 항</th><th>해석</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def sampler_model_eval_hint(name: str) -> str:
    if name in {"euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "lms", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "ipndm", "ipndm_v", "deis", "ddpm", "lcm"}:
        return "대체로 step당 모델 평가 1회(히스토리 결합 비용은 별도)."
    if name in {"heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "exp_heun_2_x0", "exp_heun_2_x0_sde"}:
        return "대체로 step당 모델 평가 2회(예측+보정)."
    if name in {"dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu"}:
        return "history 재사용 구조라 step당 평가 횟수는 낮지만, 버퍼/계수 업데이트가 추가된다."
    if name in {"dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "seeds_3", "sa_solver", "sa_solver_pece"}:
        return "stage 기반 predictor/corrector 조합으로 평가 횟수가 증가할 수 있다."
    if name in {"dpm_adaptive", "dpm_fast"}:
        return "고정된 step당 횟수보다 accept/reject 및 내부 제어 루프에 의해 총 평가 횟수가 결정된다."
    if name in {"uni_pc", "uni_pc_bh2"}:
        return "구현 경로에 따라 1~2회 이상 변동 가능."
    return "구현/분기 조건에 따라 변동."


def sampler_history_hint(name: str) -> str:
    if name in {"dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu"}:
        return "직전 단계 history(보통 1-step memory)를 유지한다."
    if name in {"dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "seeds_3"}:
        return "다중 stage history(2~3단계 수준)를 유지한다."
    if name in {"lms", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "ipndm", "ipndm_v", "deis"}:
        return "차수 m에 비례하는 history 버퍼(최근 gradient/derivative)를 유지한다."
    return "명시적 history 버퍼 의존이 낮은 단일스텝 구조."


def sampler_controller_hint(name: str, stochastic: str) -> str:
    if name == "dpm_adaptive":
        return "오차 추정 + PID(step controller) 기반 accept/reject 제어."
    if stochastic in {"yes", "optional"}:
        return "고정 mesh 위에서 noise injection 파라미터(eta, s_noise 등)로 분산 제어."
    return "고정 mesh의 deterministic stepper 제어."


def sampler_storage_hint(name: str) -> str:
    if name.endswith("_gpu"):
        return "노이즈 샘플링/연산 일부가 GPU 경로로 최적화될 수 있다."
    if name in {"lms", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp", "ipndm", "ipndm_v", "deis"}:
        return "history 버퍼 메모리와 계수 연산(벡터화) 비용이 핵심."
    return "기본 latent + 중간 stage 텐서 저장 비용이 주된 메모리 사용처."


def render_sampler_math_deep(name: str, family: str, stochastic: str, cfg_pp: bool) -> str:
    prof = sampler_order_profile(name)
    cfg_note = (
        r"$$v_{\mathrm{cfg}}=v_u+w(v_c-v_u),\quad "
        r"v_{\mathrm{cfg++}}=\Pi_{\mathcal{T}_{\rho}}(v_{\mathrm{cfg}})$$"
        if cfg_pp
        else r"$$v_{\mathrm{cfg}}=v_u+w(v_c-v_u)$$"
    )
    stochastic_note = (
        r"$$x_{k+1}=m_k(x_k)+G_k\xi_k,\ \xi_k\sim\mathcal{N}(0,I),\ "
        r"\mathrm{Cov}[x_{k+1}|x_k]=G_kG_k^\top$$"
        if stochastic in {"yes", "optional"}
        else r"$$x_{k+1}=m_k(x_k),\quad \partial_t\rho+\nabla\cdot(\rho v)=0$$"
    )
    generator_note = (
        r"$$\mathcal{L}_t\varphi=b_t\cdot\nabla\varphi+\frac12 g_t^2\Delta\varphi,\quad "
        r"\partial_t\rho_t=\mathcal{L}_t^\star\rho_t$$"
        if stochastic in {"yes", "optional"}
        else r"$$\mathcal{L}_t\varphi=v_t\cdot\nabla\varphi,\quad "
        r"\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0$$"
    )
    transport_note = (
        r"$$\rho_{k+1}\approx\arg\min_\rho\left(\frac{W_2^2(\rho,\rho_k)}{2\tau_k}+\mathcal{F}(\rho)\right)$$"
        if stochastic in {"yes", "optional"}
        else r"$$\min_{\rho,v}\int_0^1\!\!\int \frac12\|v_t(x)\|^2\rho_t(x)\,dx\,dt,\quad "
        r"\partial_t\rho+\nabla\cdot(\rho v)=0$$"
    )
    impl_rows = (
        "<tr><td>스텝 커널 구조</td>"
        "<td>$$x_{k+1}=A_kx_k+B_k\\hat{x}_{0,k}+C_k(\\text{history})+D_k\\xi_k$$</td></tr>"
        f"<tr><td>모델 평가량(경향)</td><td>{esc(sampler_model_eval_hint(name))}</td></tr>"
        f"<tr><td>history 버퍼</td><td>{esc(sampler_history_hint(name))}</td></tr>"
        f"<tr><td>스텝 제어</td><td>{esc(sampler_controller_hint(name, stochastic))}</td></tr>"
        "<tr><td>메쉬 변수</td>"
        "<td>$$\\lambda=\\log\\alpha-\\log\\sigma,\\ h_k=|\\lambda_{k+1}-\\lambda_k|$$</td></tr>"
        f"<tr><td>저장/정밀도 메모</td><td>{esc(sampler_storage_hint(name))}</td></tr>"
    )
    return (
        "<h2>수학 심화 프로파일</h2>"
        "<h3>순수수학 관점</h3>"
        "<table><thead><tr><th>항목</th><th>내용</th></tr></thead><tbody>"
        f"<tr><td>method class</td><td>{esc(prof['class'])}</td></tr>"
        f"<tr><td>local truncation</td><td>$$ {prof['local']} $$</td></tr>"
        f"<tr><td>global error</td><td>$$ {prof['global']} $$</td></tr>"
        f"<tr><td>strong/weak 관점</td><td>{esc(prof['strong_weak'])}</td></tr>"
        f"<tr><td>stability 메모</td><td>{esc(prof['stability'])}</td></tr>"
        "</tbody></table>"
        f"<div class='formula'>{generator_note}</div>"
        f"<div class='formula'>{transport_note}</div>"
        "<div class='formula'>"
        r"$$\|x(t_{k+1})-x_{k+1}\|\le C h_k^{p+1},\quad "
        r"\|x(T)-x_N\|\le C\max_k h_k^p,\quad h_k:=|\lambda_{k+1}-\lambda_k|$$"
        "</div>"
        "<h3>수치해석/구현 관점</h3>"
        f"<table><thead><tr><th>구현 항목</th><th>내용</th></tr></thead><tbody>{impl_rows}</tbody></table>"
        "<div class='formula'>"
        r"$$\lambda=\log\alpha-\log\sigma,\quad "
        r"x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k$$"
        "</div>"
        f"<div class='formula'>{cfg_note}</div>"
        f"<div class='formula'>{stochastic_note}</div>"
        f"<p class='small'><strong>family:</strong> {esc(family)} / <strong>stochastic:</strong> {esc(stochastic)}</p>"
    )


def scheduler_mesh_profile(name: str) -> tuple[str, str, str]:
    if name == "karras":
        return (
            r"\sigma(u)=\left(\sigma_{\max}^{1/\rho}+u(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho})\right)^\rho,\ u\in[0,1]",
            "rho로 고/저 sigma 구간의 mesh 밀도를 조정한다.",
            "rho가 커질수록 저노이즈 구간 분해능이 증가해 세부 복원에 유리할 수 있다.",
        )
    if name == "exponential":
        return (
            r"\log\sigma(u)=(1-u)\log\sigma_{\max}+u\log\sigma_{\min}",
            "로그 공간 균등 분할로 배율 변화가 일정하다.",
            "멀티스텝 계열에서 step ratio가 안정적이라 계수 변동이 완만해진다.",
        )
    if name == "kl_optimal":
        return (
            r"\sigma(u)=\tan\!\left(u\arctan\sigma_{\min}+(1-u)\arctan\sigma_{\max}\right)",
            "KL 기반 휴리스틱으로 중간 노이즈 구간을 상대적으로 강조한다.",
            "score mismatch가 큰 구간의 누적 오차를 완화하려는 목적의 스케줄.",
        )
    if name in {"normal", "sgm_uniform"}:
        return (
            r"t_k\sim U,\ \sigma_k=\sigma(t_k)",
            "시간축 균등 분할 계열.",
            "모델 학습 시간축과의 정합성이 좋으면 기본선(baseline)으로 안정적이다.",
        )
    if name == "beta":
        return (
            r"t_k=F^{-1}_{\mathrm{Beta}(\alpha,\beta)}(u_k),\ \sigma_k=\sigma(t_k)",
            "Beta CDF 역함수로 특정 구간에 스텝을 집중한다.",
            "질감/구조 우선순위에 맞춰 노이즈 구간 가중을 설계할 때 유용하다.",
        )
    if name == "linear_quadratic":
        return (
            r"\sigma(u)\approx a_0+a_1u+a_2u^2\ (\text{piecewise})",
            "선형+이차 감쇠를 조합해 초반/후반 분해능을 다르게 배치한다.",
            "초반 큰 이동과 후반 미세 보정을 분리해 튜닝하기 좋다.",
        )
    if name in {"simple", "ddim_uniform"}:
        return (
            r"\sigma_k=\text{discrete ladder stride}(k)",
            "학습된 이산 ladder를 직접 stride 샘플링한다.",
            "학습 분포와 직접적으로 맞닿아 있고 구현 해석이 직관적이다.",
        )
    return (
        r"\sigma_k=S(k)",
        "mesh 생성 함수 기반 스케줄",
        "적분기와 결합 시 step 밀도 분포가 품질/안정성을 좌우한다.",
    )


def render_global_symbol_contract() -> str:
    l = symbol_link_by_id
    return (
        "<h2>0) 공통 기호 계약(측도론/함수해석학 기준)</h2>"
        "<p>기호를 추상 정의만 나열하지 않고, 탭별로 <strong>엄밀 정의 + 직관 + 구체 원소 예시</strong>를 같이 제공합니다.</p>"
        '<div class="topnav"><a class="btn" href="symbol/index.html">기호 위키 인덱스</a></div>'
        '<div class="tabs mini" data-tab-group="notation-sub">'
        '<button class="tab-btn active" data-tab-group="notation-sub" data-tab-target="notation-a">A 기본 공간</button>'
        '<button class="tab-btn" data-tab-group="notation-sub" data-tab-target="notation-b">B 사상</button>'
        '<button class="tab-btn" data-tab-group="notation-sub" data-tab-target="notation-c">C 확률구조</button>'
        '<button class="tab-btn" data-tab-group="notation-sub" data-tab-target="notation-d">D FPE/OT</button>'
        '<button class="tab-btn" data-tab-group="notation-sub" data-tab-target="notation-e">E 시그니처 독해</button>'
        '<button class="tab-btn" data-tab-group="notation-sub" data-tab-target="notation-f">F 직관+예시</button>'
        "</div>"
        '<section id="notation-a" class="tab-panel active" data-tab-group="notation-sub">'
        "<h3>A. 기본 공간(엄밀 정의)</h3>"
        "<table><thead><tr><th>기호</th><th>정의</th><th>비고</th></tr></thead><tbody>"
        f"<tr><td>{l('X')}</td><td>실수 힐베르트 공간. 구현 기본은 $\\mathcal{{X}}=\\mathbb{{R}}^d$, $d=C\\cdot H\\cdot W$.</td><td>내적은 $\\langle x,y\\rangle=x^\\top y$, 노름은 $\\|x\\|_2$.</td></tr>"
        f"<tr><td>{l('B_X')}</td><td>$\\mathcal{{X}}$ 위 Borel sigma-대수</td><td>가측성 판단의 표준 기준.</td></tr>"
        f"<tr><td>{l('Sigma')}</td><td>$\\Sigma=[\\sigma_{{\\min}},\\sigma_{{\\max}}]\\subset(0,\\infty)$</td><td>노이즈 스케일 공역.</td></tr>"
        f"<tr><td>{l('C_G')}</td><td>조건변수의 가측공간</td><td>보통 표준 Borel 공간으로 둔다.</td></tr>"
        f"<tr><td>{l('Hk_Ak')}</td><td>$\\mathcal{{H}}_k=\\mathcal{{X}}^{{m_k}}$, $\\mathcal{{A}}_k=\\mathcal{{B}}(\\mathcal{{X}})^{{\\otimes m_k}}$</td><td>$m_k=0$이면 한 점 공간으로 본다.</td></tr>"
        f"<tr><td>{l('K_set')}</td><td>닫힌 부분집합(필요 시 닫힌 볼록집합)</td><td>투영의 유일성은 볼록성에서 보장.</td></tr>"
        f"<tr><td>{l('M_manifold')}</td><td>$C^r$ 매장 부분다양체($r\\ge1$)</td><td>각 점에서 접공간이 선형공간으로 정의.</td></tr>"
        f"<tr><td>{l('TxM')}</td><td>$\\mathcal{{M}}$의 점 $x$에서의 접공간</td><td>보정 방향의 허용 선형화 공간.</td></tr>"
        "</tbody></table>"
        "</section>"
        '<section id="notation-b" class="tab-panel" data-tab-group="notation-sub">'
        "<h3>B. 사상(정의역/공역)</h3>"
        "<table><thead><tr><th>기호</th><th>사상</th><th>가정</th></tr></thead><tbody>"
        f"<tr><td>{l('S_map')}</td><td>$S:\\{{0,\\dots,N\\}}\\to\\Sigma$, $k\\mapsto\\sigma_k$</td><td>보통 단조감소($\\sigma_{{k+1}}\\le\\sigma_k$).</td></tr>"
        f"<tr><td>{l('D_theta')}</td><td>$D_\\theta:(\\mathcal{{X}}\\times\\Sigma\\times\\mathcal{{C}})\\to\\mathcal{{X}}$</td><td>가측 사상, 국소 Lipschitz 가정이 흔함.</td></tr>"
        f"<tr><td>{l('b_theta')}</td><td>$b_\\theta:\\mathcal{{X}}\\times[0,1]\\times\\mathcal{{C}}\\to\\mathcal{{X}}$</td><td>ODE/SDE drift 사상.</td></tr>"
        f"<tr><td>{l('g_func')}</td><td>$g:[0,1]\\to[0,\\infty)$</td><td>가측 함수(보통 piecewise $C^1$).</td></tr>"
        f"<tr><td>{l('Phi_k')}</td><td>$\\Phi_k:(\\mathcal{{X}}\\times\\mathcal{{H}}_k\\times\\Omega_k)\\to\\mathcal{{X}}$</td><td>이산시간 전이 사상.</td></tr>"
        f"<tr><td>{l('Psi_k')}</td><td>$\\Psi_k:(\\mathcal{{H}}_k\\times\\mathcal{{X}})\\to\\mathcal{{H}}_{{k+1}}$</td><td>다단계 과거값 갱신 사상.</td></tr>"
        f"<tr><td>{l('Pi_K')}</td><td>$\\Pi_{{\\mathcal{{K}}}}:\\mathcal{{X}}\\to\\mathcal{{K}}$</td><td>힐베르트 공간에서 metric projection.</td></tr>"
        f"<tr><td>{l('R_x')}</td><td>$R_x:T_x\\mathcal{{M}}\\to\\mathcal{{M}}$</td><td>$R_x(0)=x$, $dR_x(0)=\\mathrm{{Id}}$.</td></tr>"
        "</tbody></table>"
        "</section>"
        '<section id="notation-c" class="tab-panel" data-tab-group="notation-sub">'
        "<h3>C. 확률구조</h3>"
        "<table><thead><tr><th>기호</th><th>정의</th><th>의미</th></tr></thead><tbody>"
        f"<tr><td>{l('Omega_F_P')}</td><td>기저 확률공간</td><td>모든 확률변수의 공통 기반.</td></tr>"
        "<tr><td>이산 stochastic</td><td>$\\Omega=(\\mathbb{R}^d)^N$, $\\mathcal{F}=\\mathcal{B}(\\Omega)$, $\\mathbb{P}=\\bigotimes_{k=0}^{N-1}\\mathcal{N}(0,I_d)$</td><td>$N$ step 가우시안 잡음의 곱측도 모델.</td></tr>"
        "<tr><td>결정론</td><td>$\\Omega=\\{\\omega_0\\}$, $\\mathcal{F}=\\{\\varnothing,\\Omega\\}$, $\\mathbb{P}(\\Omega)=1$</td><td>확률구조가 자명하게 축약.</td></tr>"
        f"<tr><td>{l('Xi_k')}</td><td>$\\xi_k:(\\Omega,\\mathcal{{F}},\\mathbb{{P}})\\to\\mathbb{{R}}^d$</td><td>보통 i.i.d. $\\mathcal{{N}}(0,I_d)$.</td></tr>"
        f"<tr><td>{l('Fk')}</td><td>$\\mathcal{{F}}_k=\\sigma(\\xi_0,\\dots,\\xi_{{k-1}})$</td><td>$x_k$의 $\\mathcal{{F}}_k$-가측성이 적응성.</td></tr>"
        "<tr><td>연속시간 모델</td><td>$\\Omega=C([0,1],\\mathbb{R}^d)$</td><td>Wiener 공간 관점의 경로측도 모델.</td></tr>"
        "</tbody></table>"
        "</section>"
        '<section id="notation-d" class="tab-panel" data-tab-group="notation-sub">'
        "<h3>D. 분포 수준 기호(FPE/OT)</h3>"
        "<table><thead><tr><th>기호</th><th>정의</th><th>해석</th></tr></thead><tbody>"
        f"<tr><td>{l('Rho_t')}</td><td>$\\rho_t\\in\\mathcal{{P}}_2(\\mathcal{{X}})$</td><td>시간 $t$에서의 latent 분포.</td></tr>"
        f"<tr><td>{l('V_t')}</td><td>$v_t\\in L^2_{{\\rho_t}}(\\mathcal{{X}};\\mathcal{{X}})$</td><td>연속방정식의 속도장.</td></tr>"
        "<tr><td>FPE</td><td>$\\partial_t\\rho=-\\nabla\\cdot(b_\\theta\\rho)+\\frac12\\nabla\\cdot((gg^\\top)\\nabla\\rho)$</td><td>SDE의 분포 동역학.</td></tr>"
        "<tr><td>OT action</td><td>$\\int_0^1\\!\\int_{\\mathcal{X}}\\|v_t(x)\\|^2\\,d\\rho_t(x)\\,dt$</td><td>Benamou-Brenier 동적 정식화와 대응.</td></tr>"
        "</tbody></table>"
        "</section>"
        '<section id="notation-e" class="tab-panel" data-tab-group="notation-sub">'
        "<h3>E. 시그니처 읽는 법</h3>"
        "<table><thead><tr><th>표기</th><th>수학적 독해</th><th>코드 대응</th></tr></thead><tbody>"
        f"<tr><td>{l('Phi_k')} : $\\mathcal{{X}}\\times\\mathcal{{H}}_k\\times\\Omega_k\\to\\mathcal{{X}}$</td><td>입력 3개(상태, 과거값, 난수접두)에서 출력 1개로 가는 사상.</td><td><code>x_next = step(x, h_k, omega_prefix)</code></td></tr>"
        f"<tr><td>{l('Psi_k')} : $\\mathcal{{H}}_k\\times\\mathcal{{X}}\\to\\mathcal{{H}}_{{k+1}}$</td><td>현재 과거값과 새 상태를 받아 다음 과거값 벡터를 생성.</td><td><code>h_next = update_history(h_k, x_next)</code></td></tr>"
        f"<tr><td>{l('D_theta')} : $\\mathcal{{X}}\\times\\Sigma\\times\\mathcal{{C}}\\to\\mathcal{{X}}$</td><td>조건부 벡터장/denoiser를 반환하는 모형 사상.</td><td><code>denoised = model(x, sigma, cond)</code></td></tr>"
        "</tbody></table>"
        "</section>"
        '<section id="notation-f" class="tab-panel" data-tab-group="notation-sub">'
        "<h3>F. 직관 + 구체 원소 예시</h3>"
        "<table><thead><tr><th>기호</th><th>추상 정의</th><th>직관</th><th>구체 원소 예시</th></tr></thead><tbody>"
        f"<tr><td>{l('X')}</td><td>실수 힐베르트 공간</td><td>latent 벡터가 사는 좌표공간</td><td>$d=4$라면 $x=(0.2,-1.1,0.0,3.4)\\in\\mathbb{{R}}^4$</td></tr>"
        f"<tr><td>{l('Hk_Ak')}</td><td>2-step 과거값 곱공간</td><td>현재 step에서 직전 두 상태를 기억</td><td>$h_k=(x_{{k-1}},x_k)$</td></tr>"
        f"<tr><td>{l('Omega_F_P')}</td><td>$N$개 잡음벡터 경로의 집합</td><td>하나의 $\\omega$가 전체 난수 시나리오</td><td>$\\omega=(\\xi_0,\\xi_1,\\dots,\\xi_{{N-1}})$</td></tr>"
        f"<tr><td>{l('Xi_k')}</td><td>가우시안 잡음 확률변수</td><td>k번째 step에서 쓰는 난수벡터</td><td>$d=3$에서 $\\xi_k=(0.31,-1.24,0.08)$</td></tr>"
        f"<tr><td>{l('S_map')}</td><td>$S(k)=\\sigma_k$</td><td>step 번호를 노이즈 크기로 변환</td><td>$S(0)=14.6,\\ S(1)=9.2,\\dots,S(N)=0$</td></tr>"
        f"<tr><td>{l('Phi_k')}</td><td>step 전이 사상</td><td>한 번의 업데이트 규칙</td><td>Euler형 예시: $\\Phi_k(x_k,h_k,\\omega_k)=x_k+\\Delta\\sigma_k d_\\theta(x_k,\\sigma_k)$</td></tr>"
        f"<tr><td>{l('Psi_k')}</td><td>과거값 갱신 사상</td><td>슬라이딩 윈도우 갱신</td><td>$\\Psi_k((x_{{k-1}},x_k),x_{{k+1}})=(x_k,x_{{k+1}})$</td></tr>"
        f"<tr><td>{l('Pi_K')}</td><td>허용집합으로의 metric projection</td><td>제약을 벗어나면 가장 가까운 점으로 되돌림</td><td>$\\mathcal{{K}}=[-1,1]^d$이면 좌표별 clip과 동일</td></tr>"
        "</tbody></table>"
        "</section>"
        "<p class='small'>표기 규약: $k$는 이산 step index, $t$는 연속시간 변수. 곱공간의 sigma-대수는 product sigma-대수로 둡니다.</p>"
    )


def render_sampler_symbol_contract(
    name: str,
    stochastic: str,
    common_rows: list[tuple[str, str, str]],
    param_rows: list[dict[str, str]],
) -> str:
    pnames = {p["name"] for p in param_rows}
    quick_ids = ["X", "Sigma", "D_theta", "Phi_k", "Psi_k", "S_map", "Xi_k", "Fk", "Omega_F_P"]
    quick_links = []
    for sid in quick_ids:
        item = next((s for s in SYMBOL_WIKI if s["id"] == sid), None)
        if item is None:
            continue
        quick_links.append(
            f'<a class="btn" href="../symbol/{esc(sid)}.html">${item["tex"]}$</a>'
        )

    rows = [
        "<tr><td>상태 변수</td><td>$x_k:(\\Omega,\\mathcal{F}_k)\\to(\\mathcal{X},\\mathcal{B}(\\mathcal{X}))$</td><td>$\\mathcal{X}$는 보통 $\\mathbb{R}^d$ (유한차원 힐베르트 공간).</td></tr>",
        "<tr><td>조건 변수</td><td>$c:(\\Omega,\\mathcal{F})\\to(\\mathcal{C},\\mathcal{G})$ 또는 고정 매개변수 $c\\in\\mathcal{C}$</td><td>조건은 가측 사상 또는 상수 매개변수로 모델링.</td></tr>",
        "<tr><td>스케줄 사상</td><td>$S:\\{0,\\dots,N\\}\\to\\Sigma$, $k\\mapsto\\sigma_k$</td><td>단조감소 가정이 일반적이며 $h_k=|\\lambda_{k+1}-\\lambda_k|$가 오차를 지배.</td></tr>",
        "<tr><td>모형 사상</td><td>$D_\\theta:(\\mathcal{X}\\times\\Sigma\\times\\mathcal{C})\\to\\mathcal{X}$</td><td>측도론적으로는 $(\\mathcal{B}(\\mathcal{X})\\otimes\\mathcal{B}(\\Sigma)\\otimes\\mathcal{G},\\mathcal{B}(\\mathcal{X}))$-가측 사상.</td></tr>",
        "<tr><td>이산시간 전이 사상</td><td>$\\Phi_k:(\\mathcal{X}\\times\\mathcal{H}_k\\times\\Omega_k)\\to\\mathcal{X}$</td><td>시간지수 $k$에서 $k+1$로 가는 상태전이 사상.</td></tr>",
        "<tr><td>다단계 과거값 갱신 사상</td><td>$\\Psi_k:(\\mathcal{H}_k\\times\\mathcal{X})\\to\\mathcal{H}_{k+1}$</td><td>과거값 벡터를 다음 단계의 과거값 벡터로 옮기는 사상.</td></tr>",
        "<tr><td>적응성/가측성</td><td>$x_k$는 $\\mathcal{F}_k$-가측, $\\Phi_k$는 $(\\mathcal{B}(\\mathcal{X})\\otimes\\mathcal{A}_k\\otimes\\mathcal{F}_k,\\mathcal{B}(\\mathcal{X}))$-가측</td><td>미래 잡음 미참조(non-anticipative) 조건을 형식화.</td></tr>",
    ]
    if stochastic in {"yes", "optional"}:
        rows.append(
            "<tr><td>기저 확률공간</td><td>$\\Omega=(\\mathbb{R}^d)^N$, $\\mathcal{F}=\\mathcal{B}(\\Omega)$, $\\mathbb{P}=\\bigotimes_{k=0}^{N-1}\\mathcal{N}(0,I_d)$</td>"
            "<td>이산 stochastic sampler의 표준 곱측도 모델.</td></tr>"
        )
        rows.append(
            "<tr><td>잡음 확률변수</td><td>$\\xi_k:(\\Omega,\\mathcal{F},\\mathbb{P})\\to(\\mathbb{R}^d,\\mathcal{B}(\\mathbb{R}^d))$, $\\xi_k\\sim\\mathcal{N}(0,I_d)$</td>"
            "<td>$\\xi_k\\in L^2(\\Omega;\\mathbb{R}^d)$. 상관잡음이면 공분산 연산자를 명시해야 함.</td></tr>"
        )
        rows.append(
            "<tr><td>필트레이션</td><td>$\\mathcal{F}_k=\\sigma(\\xi_0,\\dots,\\xi_{k-1})$ 및 $x_k$의 $\\mathcal{F}_k$-가측성</td>"
            "<td>현재 상태가 과거 정보에만 의존한다는 적응성 조건.</td></tr>"
        )
        rows.append(
            "<tr><td>접두 경로공간</td><td>$\\Omega_k=(\\mathbb{R}^d)^k$</td>"
            "<td>k-step까지 사용한 난수 경로를 나타내는 부분공간.</td></tr>"
        )
    else:
        rows.append(
            "<tr><td>결정론적 모델</td><td>$\\Omega=\\{\\omega_0\\}$, $\\mathcal{F}=\\{\\varnothing,\\Omega\\}$, $\\mathbb{P}(\\Omega)=1$</td>"
            "<td>확률기호는 형식적으로만 남고 난수항은 제거된다.</td></tr>"
        )
        rows.append(
            "<tr><td>step 사상 축약</td><td>$\\Phi_k:\\mathcal{X}\\times\\mathcal{H}_k\\to\\mathcal{X}$</td>"
            "<td>난수 인자 $\\Omega_k$가 소거된 결정론적 사상으로 동작.</td></tr>"
        )
    if "order" in pnames or "max_order" in pnames:
        rows.append(
            "<tr><td>다단계 곱공간</td><td>$\\mathcal{H}_k=\\mathcal{X}^{m_k}$, $\\mathcal{A}_k=\\mathcal{B}(\\mathcal{X})^{\\otimes m_k}$, $m_k\\le m$</td><td>초기 구간에서는 $m_k$가 작고, 진행되며 최대 차수까지 증가.</td></tr>"
        )
    if "sigma_min/sigma_max" in {n for n, _, _ in common_rows}:
        rows.append(
            "<tr><td>적분 구간</td><td>$\\sigma\\in[\\sigma_{\\min},\\sigma_{\\max}]$</td><td>경계 선택이 해상도 보존과 계산량 균형에 직접 영향.</td></tr>"
        )

    constraints = [
        "<tr><td>mesh 단조성</td><td>$\\sigma_{k+1}\\le\\sigma_k$, $h_k:=|\\lambda_{k+1}-\\lambda_k|>0$</td><td>역적분 안정성 및 오차 분석의 기본 가정.</td></tr>",
        "<tr><td>drift 정칙성</td><td>$\\|b_\\theta(x,t)-b_\\theta(y,t)\\|\\le L\\|x-y\\|$</td><td>존재/유일성과 수치해석 수렴률에 필요한 대표 가정.</td></tr>",
    ]
    if "eta" in pnames:
        constraints.append("<tr><td>확률강도</td><td>$\\eta\\ge0$</td><td>noise 주입 강도/드리프트 감쇠 결합.</td></tr>")
    if "s_noise" in pnames:
        constraints.append("<tr><td>노이즈 배율</td><td>$s_{noise}\\ge0$</td><td>분산 스케일 파라미터.</td></tr>")
    if "r" in pnames:
        constraints.append("<tr><td>중간 stage 비율</td><td>$0<r<1$</td><td>2-stage 보간 위치.</td></tr>")
    if "r_1" in pnames:
        constraints.append("<tr><td>중간 stage1</td><td>$0<r_1<1$</td><td>SEEDS-3 첫 중간점.</td></tr>")
    if "r_2" in pnames:
        constraints.append("<tr><td>중간 stage2</td><td>$0<r_2<1$</td><td>SEEDS-3 둘째 중간점.</td></tr>")
    if "rtol" in pnames:
        constraints.append("<tr><td>상대오차 허용치</td><td>$rtol>0$</td><td>adaptive accept/reject 기준.</td></tr>")
    if "atol" in pnames:
        constraints.append("<tr><td>절대오차 허용치</td><td>$atol>0$</td><td>adaptive accept/reject 기준.</td></tr>")
    if "accept_safety" in pnames:
        constraints.append("<tr><td>안전계수</td><td>$0<accept\\_safety\\le1$</td><td>수락 조건의 보수성 제어.</td></tr>")
    if "solver_type" in pnames:
        constraints.append("<tr><td>보정자 선택</td><td>$solver\\_type\\in\\{\\mathrm{midpoint},\\mathrm{heun}\\}$</td><td>보정식 계열 전환.</td></tr>")
    if "tau_func" in pnames:
        constraints.append("<tr><td>간격 함수</td><td>$\\tau:[0,1]\\to[0,1]$</td><td>측정가능성/유계성 가정.</td></tr>")

    ex_rows = [
        "<tr><td>$x_k\\in\\mathcal{X}$</td><td>$d=4$ 예시에서 $x_k=(0.12,-0.34,1.08,0.00)$</td><td>현재 latent 상태의 한 점.</td></tr>",
        "<tr><td>$h_k\\in\\mathcal{H}_k$</td><td>2-step이면 $h_k=(x_{k-1},x_k)$</td><td>다단계 solver의 과거값 벡터.</td></tr>",
        "<tr><td>$\\Phi_k$</td><td>$x_{k+1}=\\Phi_k(x_k,h_k,\\omega_k)$</td><td>한 step에서 상태를 다음 상태로 보내는 사상.</td></tr>",
        "<tr><td>$\\Psi_k$</td><td>$\\Psi_k((x_{k-1},x_k),x_{k+1})=(x_k,x_{k+1})$</td><td>슬라이딩 윈도우 형태의 과거값 갱신.</td></tr>",
    ]
    if stochastic in {"yes", "optional"}:
        ex_rows.append(
            "<tr><td>$\\omega\\in\\Omega$</td><td>$\\omega=(\\xi_0,\\xi_1,\\dots,\\xi_{N-1})$</td><td>전체 샘플링 과정에서 소비될 난수 경로 하나.</td></tr>"
        )
        ex_rows.append(
            "<tr><td>$\\xi_k$</td><td>$d=3$ 예시: $\\xi_k=(0.31,-1.24,0.08)$</td><td>k번째 step의 가우시안 잡음 벡터.</td></tr>"
        )
    else:
        ex_rows.append(
            "<tr><td>결정론적 경우</td><td>$\\Omega=\\{\\omega_0\\}$</td><td>난수 경로가 하나뿐이라 잡음항이 사라짐.</td></tr>"
        )

    return (
        "<h2>기호 계약(정의역/공역/조건)</h2>"
        "<p class='small'>기호별 상세 위키: "
        "<a class='row-link' href='../symbol/index.html'><strong>Symbol Wiki Index</strong></a></p>"
        f"<div class='topnav'>{''.join(quick_links)}</div>"
        "<table><thead><tr><th>항목</th><th>수식</th><th>설명</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p class='small'>해석 팁: <code>k</code>는 이산 step index, <code>t</code>는 연속시간 변수로 구분한다. "
        "또한 $\\mathcal{X}\\times\\mathcal{H}_k\\times\\Omega_k$ 위에서 정의된 $\\Phi_k$의 가측성은 "
        "코드에서 난수 소비 순서(시드 재현성)와 직접 연결된다.</p>"
        "<h3>직관/구체 원소 예시</h3>"
        "<table><thead><tr><th>기호</th><th>원소 예시</th><th>직관</th></tr></thead>"
        f"<tbody>{''.join(ex_rows)}</tbody></table>"
        "<h3>해당 sampler의 추가 제약</h3>"
        "<table><thead><tr><th>제약</th><th>조건</th><th>의미</th></tr></thead>"
        f"<tbody>{''.join(constraints)}</tbody></table>"
    )


def render_scheduler_symbol_contract(name: str) -> str:
    specific = []
    if name == "karras":
        specific.append("<tr><td>형상 파라미터</td><td>$$\\rho>0$$</td><td>mesh 밀도 곡률을 제어.</td></tr>")
    if name == "beta":
        specific.append("<tr><td>분포 파라미터</td><td>$$\\alpha>0,\\ \\beta>0$$</td><td>구간 집중도 조절.</td></tr>")
    if name in {"simple", "ddim_uniform"}:
        specific.append("<tr><td>이산 ladder</td><td>$$k\\mapsto i_k\\in\\{0,\\dots,T\\}$$</td><td>학습된 이산 sigma 테이블 stride.</td></tr>")
    if not specific:
        specific.append("<tr><td>일반 조건</td><td>$$\\sigma_k>0,\\ \\sigma_{k+1}\\le\\sigma_k$$</td><td>역적분용 단조성.</td></tr>")

    return (
        "<h2>기호 계약(정의역/공역/조건)</h2>"
        "<table><thead><tr><th>항목</th><th>수식</th><th>설명</th></tr></thead><tbody>"
        "<tr><td>스케줄 맵</td><td>$$S:\\{0,\\dots,N\\}\\to\\Sigma\\subset(0,\\infty)$$</td><td>step index를 sigma로 매핑.</td></tr>"
        "<tr><td>mesh 변수</td><td>$$h_k=|\\lambda_{k+1}-\\lambda_k|,\\ \\lambda=\\log\\alpha-\\log\\sigma$$</td><td>오차식에 직접 들어가는 유효 step.</td></tr>"
        "<tr><td>오차 스케일</td><td>$$\\|e_{global}\\|\\approx C\\max_k h_k^p$$</td><td>동일 solver에서 scheduler 품질 차이의 핵심.</td></tr>"
        f"{''.join(specific)}"
        "</tbody></table>"
    )


def render_sampler_derivation_block(name: str, family: str) -> str:
    if name == "heun":
        pure = (
            "<div class='formula'>$$d_k=\\frac{x_k-\\hat{x}_{0,k}}{\\sigma_k},\\quad "
            r"\tilde{x}_{k+1}=x_k+d_k\Delta\sigma_k,\quad "
            r"d_{k+1}=d(\tilde{x}_{k+1},\sigma_{k+1})$$</div>"
            "<div class='formula'>"
            r"$$x_{k+1}=x_k+\frac12(d_k+d_{k+1})\Delta\sigma_k,\quad "
            r"e_{loc}=O(\Delta\sigma_k^3)$$"
            "</div>"
            "<p class='small'>주요 오차원천: 중간 곡률(2차 미분) 항 절단, 모델 추정 오차의 누적.</p>"
        )
        num = (
            "<ol class='list'>"
            "<li>step당 모델 평가 2회(예측/보정)를 수행한다.</li>"
            "<li><code>s_churn</code> 활성 시 sigma를 일시적으로 inflate 후 동일 보정 절차를 적용한다.</li>"
            "<li>고정 step 환경에서 안정성은 좋지만 계산량은 Euler 대비 증가한다.</li>"
            "</ol>"
        )
    elif name == "dpmpp_2m_sde":
        pure = (
            "<div class='formula'>"
            r"$$\lambda=\log\alpha-\log\sigma,\ h=\lambda_t-\lambda_s,\ h_\eta=\eta h$$"
            "</div>"
            "<div class='formula'>"
            r"$$x_t=\frac{\sigma_t}{\sigma_s}e^{-h_\eta}x_s+\alpha_t(1-e^{-h_\eta})\hat{x}_{0,s}"
            r"+\mathrm{corr}_{2M}(h,\hat{x}_{0,s},\hat{x}_{0,s-1})"
            r"+\sigma_t\sqrt{1-e^{-2h_\eta}}\,s_{noise}\xi$$"
            "</div>"
            "<p class='small'>주요 오차원천: history 보정항 근사, 모델 출력의 비선형 변화, 난수항 샘플링 분산.</p>"
        )
        num = (
            "<ol class='list'>"
            "<li>직전 denoised 결과를 버퍼링해 2M correction을 계산한다.</li>"
            "<li><code>solver_type</code>(midpoint/heun)에 따라 보정 계수를 분기한다.</li>"
            "<li><code>expm1</code> 기반 계산으로 작은 step에서 수치 취소오차를 줄인다.</li>"
            "</ol>"
        )
    elif name == "dpm_adaptive":
        pure = (
            "<div class='formula'>"
            r"$$x_{k+1}^{[p]},\ x_{k+1}^{[p-1]}\ \text{를 동시 계산},\quad "
            r"err_k=\frac{\|x_{k+1}^{[p]}-x_{k+1}^{[p-1]}\|}{atol+rtol\|x_{k+1}^{[p]}\|}$$"
            "</div>"
            "<div class='formula'>"
            r"$$err_k\le1\ \Rightarrow\ accept,\quad "
            r"h_{new}=h\cdot s\cdot err_k^{-\beta}\cdot err_{k-1}^{\gamma}$$"
            "</div>"
            "<p class='small'>주요 오차원천: embedded error estimator 편향, 과도한 reject 반복, 모델 비매끄러움.</p>"
        )
        num = (
            "<ol class='list'>"
            "<li><code>rtol</code>/<code>atol</code>를 기준으로 accept/reject 루프를 돈다.</li>"
            "<li><code>pcoeff</code>/<code>icoeff</code>/<code>dcoeff</code>로 PID step 제어를 수행한다.</li>"
            "<li><code>accept_safety</code>가 작을수록 보수적(안정)이나 step 수가 증가한다.</li>"
            "</ol>"
        )
    elif name == "er_sde":
        pure = (
            "<div class='formula'>"
            r"$$x_{k+1}=x_k+\sum_{j=1}^{M}a_j b_\theta(x_{k,j},t_{k,j})\Delta t_j"
            r"+\sum_{j=1}^{M}c_j g(t_{k,j})\sqrt{\Delta t_j}\,\xi_{k,j}$$"
            "</div>"
            "<p class='small'>주요 오차원천: stage 결합 계수 근사, diffusion 계수 스케일링, 다중 노이즈 샘플 분산.</p>"
        )
        num = (
            "<ol class='list'>"
            "<li><code>max_stage</code>만큼 stage를 누적해 drift/noise를 합성한다.</li>"
            "<li><code>noise_scaler</code>, <code>s_noise</code>로 확률항 크기를 제어한다.</li>"
            "<li>구조 보존과 질감 다양성은 stage 수와 noise 스케일의 trade-off다.</li>"
            "</ol>"
        )
    elif name == "seeds_2":
        pure = (
            "<div class='formula'>"
            r"$$\sigma_m=\sigma_k^{1-r}\sigma_{k+1}^{r},\quad "
            r"x_m=\Phi(x_k,\sigma_k\to\sigma_m),\quad "
            r"x_{k+1}=\Phi(x_m,\sigma_m\to\sigma_{k+1})+\Gamma_k\xi_k$$"
            "</div>"
            "<p class='small'>주요 오차원천: 중간 stage 비율 $$r$$ 선택, solver_type 보정 선택, 노이즈 주입 분산.</p>"
        )
        num = (
            "<ol class='list'>"
            "<li>중간 sigma를 생성해 2-stage 적분을 수행한다.</li>"
            "<li><code>r</code>과 <code>solver_type</code>이 품질/안정성 균형을 좌우한다.</li>"
            "<li><code>eta</code>, <code>s_noise</code>는 다양성-구조 보존 균형 파라미터다.</li>"
            "</ol>"
        )
    else:
        pure = (
            "<div class='formula'>"
            r"$$x_{k+1}=x_k+\mathcal{I}_k^{(drift)}+\mathcal{C}_k^{(history)}+\mathcal{N}_k^{(noise)}$$"
            "</div>"
            "<p class='small'>이 항 분해에서 drift/correction/noise를 어떤 차수로 근사하는지가 sampler family의 본질이다.</p>"
        )
        num = (
            "<ol class='list'>"
            "<li>scheduler로 mesh를 고정한 뒤 stepper를 선택한다.</li>"
            "<li>history 버퍼와 모델 평가 횟수의 비용-정확도 균형을 맞춘다.</li>"
            "<li>필요 시 stochastic 항을 조절해 분산과 구조 보존을 트레이드오프한다.</li>"
            "</ol>"
        )

    return (
        "<h2>유도 스케치(순수수학) / 구현 절차(수치해석)</h2>"
        f"<p class='small'><strong>대상:</strong> <code>{esc(name)}</code> / <strong>family:</strong> {esc(family)}</p>"
        "<h3>순수수학 유도 스케치</h3>"
        f"{pure}"
        "<h3>수치해석 구현 절차</h3>"
        f"{num}"
    )


def render_sampler_page(
    order: list[str],
    details: dict[str, Any],
    name: str,
    prev_name: str | None,
    next_name: str | None,
) -> str:
    d = details[name]
    param_rows = d["param_rows"]
    common_rows = common_param_rows(name)

    header = html_head(f"Sampler: {name}")
    nav = (
        '<div class="topnav">'
        '<a class="btn" href="../index.html#top">최상위 문서</a>'
        '<a class="btn" href="../index.html#integrated-model">통합 관점</a>'
        '<a class="btn" href="../index.html#bridge">KSampler 브리지</a>'
        '<a class="btn" href="../symbol/index.html">기호 위키</a>'
        '<a class="btn" href="../index.html#sampler-catalog">Sampler Catalog</a>'
        "</div>"
    )
    title = f"<h1>Sampler: <code>{esc(name)}</code></h1>"
    meta = (
        '<div class="chips">'
        f'<span class="chip">family: {esc(d["family"])}</span>'
        f'<span class="chip">stochastic: {esc(d["stochastic"])}</span>'
        f'<span class="chip">cfg_pp: {"yes" if d["cfg_pp"] else "no"}</span>'
        f'<span class="chip">gpu_variant: {"yes" if d["gpu_variant"] else "no"}</span>'
        f'<span class="chip">standalone: {"yes" if d["standalone_supported"] else "no"}</span>'
        "</div>"
    )
    sig = f"<p><strong>ComfyUI 함수 시그니처</strong><br><code>{esc(d['signature'])}</code></p>"
    doc = f"<p class='small'><strong>docstring:</strong> {esc(d['doc'])}</p>" if d["doc"] else ""
    eq = f"<div class='formula'>$$ {d['equation']} $$</div>"

    role = f"<p>{sampler_numeric_note(name, d['family'], d['stochastic'])}</p>"
    fpe = f"<p>{fpe_ot_note(d['stochastic'])}</p>"
    math_deep = render_sampler_math_deep(name, d["family"], d["stochastic"], d["cfg_pp"])
    derivation_block = render_sampler_derivation_block(name, d["family"])
    path = (
        f"<p class='path'><strong>ComfyUI 경로:</strong> <code>{esc(d['comfy_path'])}</code></p>"
        f"<p class='path'><strong>독립 구현 전략:</strong> {esc(d['strategy'])}</p>"
    )
    note = f"<p class='small'><strong>참고:</strong> {esc(d['note'])}</p>" if d["note"] else ""

    common_table_rows = "".join(
        f"<tr><td><code>{esc(n)}</code></td><td>{esc(t)}</td><td>{esc(r)}</td></tr>"
        for n, t, r in common_rows
    )
    common_table = (
        "<h2>공통 인자(시그니처 공통부)</h2>"
        "<table><thead><tr><th>인자</th><th>타입/의미</th><th>역할</th></tr></thead><tbody>"
        f"{common_table_rows}</tbody></table>"
    )
    symbol_table = render_parameter_symbol_table(common_rows, param_rows)
    contract_block = render_sampler_symbol_contract(name, d["stochastic"], common_rows, param_rows)

    if param_rows:
        specific_rows = "".join(
            f"<tr><td><code>{esc(p['name'])}</code></td><td><code>{esc(p['default'])}</code></td>"
            f"<td>{esc(p['role'])}</td><td>{esc(p['exposure'])}</td></tr>"
            for p in param_rows
        )
        specific_table = (
            "<h2>sampler 고유 파라미터 상세</h2>"
            "<table><thead><tr><th>파라미터</th><th>기본값</th><th>수학/알고리즘 역할</th><th>KSampler 노출 경로</th></tr></thead>"
            f"<tbody>{specific_rows}</tbody></table>"
        )
    else:
        specific_table = "<h2>sampler 고유 파라미터 상세</h2><p>추가 파라미터가 거의 없거나 공통 인자 중심으로 동작합니다.</p>"

    preview = (
        "<h2>원본 구현 스니펫</h2>"
        f"<div class='codebox'>{esc(d['preview'])}</div>"
    )

    rel_links = ['<div class="topnav">']
    if prev_name is not None:
        rel_links.append(f'<a class="btn" href="{esc(prev_name)}.html">이전: {esc(prev_name)}</a>')
    if next_name is not None:
        rel_links.append(f'<a class="btn" href="{esc(next_name)}.html">다음: {esc(next_name)}</a>')
    rel_links.append("</div>")
    rel = "".join(rel_links)

    body = (
        nav
        + title
        + meta
        + sig
        + doc
        + eq
        + role
        + fpe
        + math_deep
        + derivation_block
        + path
        + note
        + contract_block
        + common_table
        + specific_table
        + symbol_table
        + preview
        + rel
    )
    return header + body + html_tail()


def render_scheduler_page(name: str, formula: str, effect: str) -> str:
    header = html_head(f"Scheduler: {name}")
    mesh_formula, mesh_note, mesh_tuning = scheduler_mesh_profile(name)
    contract_block = render_scheduler_symbol_contract(name)
    body = (
        '<div class="topnav">'
        '<a class="btn" href="../index.html#top">최상위 문서</a>'
        '<a class="btn" href="../index.html#integrated-model">통합 관점</a>'
        '<a class="btn" href="../index.html#bridge">KSampler 브리지</a>'
        '<a class="btn" href="../symbol/index.html">기호 위키</a>'
        '<a class="btn" href="../index.html#scheduler-catalog">Scheduler Catalog</a>'
        "</div>"
        f"<h1>Scheduler: <code>{esc(name)}</code></h1>"
        "<p class='muted'>Scheduler는 solver 식 자체보다 sigma grid(시간 재매개화)를 바꾸어 오차 분포와 경로 성향을 조정한다.</p>"
        f"<div class='formula'>$$ {formula} $$</div>"
        f"<p>{esc(effect)}</p>"
        f"{contract_block}"
        "<h2>메쉬(시간 재매개화) 수학</h2>"
        f"<div class='formula'>$$ {mesh_formula} $$</div>"
        "<div class='formula'>"
        r"$$h_k:=|\lambda_{k+1}-\lambda_k|,\quad "
        r"\lambda=\log\alpha-\log\sigma,\quad "
        r"\|e_{\mathrm{global}}\|\approx C\max_k h_k^p$$"
        "</div>"
        f"<p>{esc(mesh_note)}</p>"
        f"<p class='small'>{esc(mesh_tuning)}</p>"
        "<h2>Sampler와의 결합 해석</h2>"
        "<p>같은 sampler라도 scheduler가 바뀌면 고 sigma 구간과 저 sigma 구간의 스텝 밀도가 달라지고, 결과적으로 세부 질감/구조 보존/수렴 속도 균형이 달라진다.</p>"
        "<div class='formula'>"
        r"$$x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k,\quad "
        r"A_k,B_k,C_k,D_k \text{는 스케줄 메쉬에 의해 간접적으로 변한다}$$"
        "</div>"
    )
    return header + body + html_tail()


def render_index(entries: list[dict[str, str]], details: dict[str, Any]) -> str:
    by_family: dict[str, list[dict[str, str]]] = {}
    for e in entries:
        by_family.setdefault(e["family"], []).append(e)

    family_sections = []
    for fam in sorted(by_family):
        cards = []
        for e in by_family[fam]:
            name = e["name"]
            cards.append(
                "<article class='card'>"
                f"<h3><a class='row-link' href='sampler/{esc(name)}.html'><code>{esc(name)}</code></a></h3>"
                f"<p class='small'>stochastic={esc(e['stochastic'])}, cfg_pp={esc(e['cfg_pp'])}, gpu={esc(e['gpu_variant'])}, standalone={esc(e['standalone'])}</p>"
                f"<p class='small'><code>{esc(e['extra_params'])}</code></p>"
                "</article>"
            )
        family_sections.append(f"<h2>{esc(fam)}</h2><div class='card-grid'>{''.join(cards)}</div>")

    rows = []
    for e in entries:
        rows.append(
            "<tr class='catalog-row' "
            f"data-name='{esc(e['name'])}' data-family='{esc(e['family'])}' data-stochastic='{esc(e['stochastic'])}' data-supported='{esc(e['standalone'])}'>"
            f"<td><a class='row-link' href='sampler/{esc(e['name'])}.html'><code>{esc(e['name'])}</code></a></td>"
            f"<td>{esc(e['family'])}</td>"
            f"<td>{esc(e['stochastic'])}</td>"
            f"<td>{esc(e['cfg_pp'])}</td>"
            f"<td>{esc(e['gpu_variant'])}</td>"
            f"<td><code>{esc(e['extra_params'])}</code></td>"
            f"<td>{esc(e['standalone'])}</td>"
            "</tr>"
        )

    srows = []
    for s in scheduler_rows():
        srows.append(
            "<tr>"
            f"<td><a class='row-link' href='scheduler/{esc(s['name'])}.html'><code>{esc(s['name'])}</code></a></td>"
            f"<td>$$ {s['formula']} $$</td>"
            f"<td>{esc(s['effect'])}</td>"
            "</tr>"
        )

    families = sorted({e["family"] for e in entries})
    fam_opts = "".join(f"<option value='{esc(f)}'>{esc(f)}</option>" for f in families)
    notation_block = render_global_symbol_contract()

    body = f"""
<div id="top"></div>
<p class="small">문서 계층: <code>sampler_site/index.html</code> (최상위) → 개별 sampler/scheduler 페이지</p>
<h1>ComfyUI Sampler Master Docs</h1>
<p class="muted">선택한 탭만 보이도록 구성했습니다. 수학 기호는 추상 정의와 함께 직관/원소 예시를 제공합니다.</p>
<div class="chips">
  <span class="chip">총 sampler: {len(entries)}</span>
  <span class="chip">총 scheduler: {len(SCHEDULER_NAMES_ALL)}</span>
  <span class="chip">함수 시그니처/기본값: ComfyUI 소스 자동 추출</span>
</div>
<div class="topnav">
  <a class="btn" href="symbol/index.html">기호 위키 인덱스</a>
</div>

<div class="tabs" data-tab-group="master">
  <button class="tab-btn active" data-tab-group="master" data-tab-target="notation">0) 기호 계약</button>
  <button class="tab-btn" data-tab-group="master" data-tab-target="integrated-model">1) 통합 모델</button>
  <button class="tab-btn" data-tab-group="master" data-tab-target="bridge">2) KSampler 브리지</button>
  <button class="tab-btn" data-tab-group="master" data-tab-target="reading-flow">3) 읽기 경로</button>
  <button class="tab-btn" data-tab-group="master" data-tab-target="sampler-catalog">4) Sampler Catalog</button>
  <button class="tab-btn" data-tab-group="master" data-tab-target="scheduler-catalog">5) Scheduler Catalog</button>
  <button class="tab-btn" data-tab-group="master" data-tab-target="family-overview">6) Family Overview</button>
</div>

<section id="notation" class="tab-panel active" data-tab-group="master">
{notation_block}
</section>

<section id="integrated-model" class="tab-panel" data-tab-group="master">
<h2>1) 통합 상태천이 모델</h2>
<p>개별 sampler를 서로 다른 도구로 보지 않고, 동일한 상태천이식에서 <code>drift / correction / noise</code> 항을 교체하는 방식으로 통합합니다.</p>
<h3>1-1) 순수수학 관점</h3>
<div class="formula">$$x_{{k+1}}=\\Phi_{{\\mathrm{{drift}}}}(x_k,\\sigma_k,\\sigma_{{k+1}})+\\Phi_{{\\mathrm{{corr}}}}(\\text{{history}})+\\Phi_{{\\mathrm{{noise}}}}(\\eta,s_{{noise}},\\xi_k)$$</div>
<div class="formula">$$d(x,\\sigma)=\\frac{{x-\\hat{{x}}_0}}{{\\sigma}}\\quad(\\text{{ComfyUI }}\\texttt{{to\\_d}})$$</div>
<div class="grid2">
  <div class="box">
    <h3>연속시간 동역학</h3>
    <div class="formula">$$dx=f_\\theta(x,t)dt+g(t)d\\bar W_t$$</div>
    <div class="formula">$$dx=\\left(f_\\theta(x,t)-\\frac12 g(t)^2\\nabla_x\\log p_t(x)\\right)dt$$</div>
    <p class="small">확산항 포함 시 역시간 SDE, 제거 시 probability-flow ODE.</p>
  </div>
  <div class="box">
    <h3>밀도/수송 해석</h3>
    <div class="formula">$$\\partial_t\\rho_t=-\\nabla\\cdot(\\rho_t b_t)+\\frac12 g_t^2\\Delta\\rho_t$$</div>
    <div class="formula">$$\\partial_t\\rho_t+\\nabla\\cdot(\\rho_t v_t)=0$$</div>
    <div class="formula">$$\\rho_{{k+1}}\\approx\\arg\\min_\\rho\\left(\\frac{{W_2^2(\\rho,\\rho_k)}}{{2\\tau_k}}+\\mathcal{{F}}(\\rho)\\right)$$</div>
    <p class="small">SDE는 FPE/entropic bridge, ODE는 continuity equation/OT 관점.</p>
  </div>
</div>

<h3>1-2) 수치해석/구현 관점</h3>
<div class="formula">$$x_{{k+1}}=A_kx_k+B_k\\hat{{x}}_{{0,k}}+C_k(\\text{{history}})+D_k\\xi_k$$</div>
<div class="formula">$$\\lambda=\\log\\alpha-\\log\\sigma,\\quad h_k:=|\\lambda_{{k+1}}-\\lambda_k|$$</div>
<div class="formula">$$\\|x(T)-x_N\\|\\lesssim C\\max_k h_k^p,\\quad \\|x(t_{{k+1}})-x_{{k+1}}\\|\\lesssim C h_k^{{p+1}}$$</div>
<table>
  <thead><tr><th>구현 레이어</th><th>핵심 객체</th><th>수치해석 역할</th><th>튜닝 포인트</th></tr></thead>
  <tbody>
    <tr><td>Model</td><td><code>model(x, sigma, **extra_args)</code></td><td>drift/denoised 추정</td><td>cond, adapter, cfg</td></tr>
    <tr><td>Schedule</td><td><code>calculate_sigmas</code></td><td>mesh(<code>h_k</code>) 분포 결정</td><td>karras, exp, kl_optimal</td></tr>
    <tr><td>Stepper</td><td><code>sample_*</code></td><td>local/global error 및 stability 결정</td><td>family, order, solver_type</td></tr>
    <tr><td>Noise</td><td><code>eta, s_noise, seed</code></td><td>분산/재현성 제어</td><td>diversity vs structure</td></tr>
    <tr><td>Controller</td><td><code>rtol/atol/PID</code></td><td>accept/reject 기반 오차 제어</td><td>dpm_adaptive 계열</td></tr>
  </tbody>
</table>
<div class="codebox"># 통합 stepper 스켈레톤\nfor k in range(N):\n    sigma, sigma_next = sigmas[k], sigmas[k+1]\n    den = model(x, sigma, **extra_args)\n    x = drift_update(x, den, sigma, sigma_next)\n    x = correction_update(x, history, sigma, sigma_next)\n    x = noise_update(x, eta, s_noise, seed, k)\n    history = update_history(history, den)\n</div>
</section>

<section id="bridge" class="tab-panel" data-tab-group="master">
<h2>2) KSampler 파라미터 브리지</h2>
<p>UI 파라미터를 수식 항에 매핑해야 커스터마이징이 가능합니다. 아래 표를 기준으로 코드를 분해하면 sampler 교체 실험이 쉬워집니다.</p>
<table>
  <thead><tr><th>UI 파라미터</th><th>내부 코드</th><th>수식 관점</th><th>효과</th></tr></thead>
  <tbody>
  {''.join("<tr><td><code>"+esc(r['ui'])+"</code></td><td>"+esc(r['internal'])+"</td><td>$$ "+r['math']+" $$</td><td>"+esc(r['effect'])+"</td></tr>" for r in bridge_rows())}
  </tbody>
</table>
<h3>브리지 확장: 구현 레이어 분해</h3>
<table>
  <thead><tr><th>레이어</th><th>핵심 함수/객체</th><th>수학 역할</th><th>커스터마이즈 포인트</th></tr></thead>
  <tbody>
    <tr><td>Model Layer</td><td><code>model(x, sigma, **extra_args)</code></td><td>$$\\hat{{x}}_0\\ \\/\\ v_\\theta\\ \\/\\ \\epsilon_\\theta$$ 추정</td><td>조건/어댑터/로라 주입</td></tr>
    <tr><td>Schedule Layer</td><td><code>calculate_sigmas</code></td><td>$$\\{{\\sigma_k\\}}$$ mesh 생성</td><td>karras/exponential/kl_optimal 선택</td></tr>
    <tr><td>Stepper Layer</td><td><code>sample_*</code></td><td>drift/correction/noise 적분기</td><td>family/solver_type/eta 조정</td></tr>
    <tr><td>Guidance Layer</td><td><code>cfg_function</code></td><td>조건 벡터장 외삽</td><td>cfg/CFG++/rescale 설계</td></tr>
  </tbody>
</table>
</section>

<section id="reading-flow" class="tab-panel" data-tab-group="master">
<h2>3) 읽기 경로</h2>
<ol class="list">
  <li>먼저 family를 고르고(아래 Family Overview),</li>
  <li>Sampler Catalog에서 후보를 좁힌 뒤,</li>
  <li>개별 페이지에서 수식/파라미터/코드 스니펫을 보고 최종 선택합니다.</li>
</ol>
</section>

<section id="sampler-catalog" class="tab-panel" data-tab-group="master">
<h2>4) Sampler Catalog</h2>
<div class="filter">
  <input id="searchInput" placeholder="sampler 검색: dpmpp, ancestral, er_sde ..."/>
  <select id="familyFilter"><option value="all">family 전체</option>{fam_opts}</select>
  <select id="supportFilter"><option value="all">standalone 전체</option><option value="yes">standalone: yes</option><option value="no">standalone: no</option></select>
</div>
<table>
  <thead><tr><th>sampler</th><th>family</th><th>stochastic</th><th>cfg_pp</th><th>gpu</th><th>extra_params</th><th>standalone</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
</section>

<section id="scheduler-catalog" class="tab-panel" data-tab-group="master">
<h2>5) Scheduler Catalog</h2>
<p>Scheduler는 적분기의 종류가 아니라, 동일 적분기에서 시간축/노이즈축 분할을 바꾸는 층입니다.</p>
<table>
  <thead><tr><th>scheduler</th><th>핵심 수식</th><th>효과</th></tr></thead>
  <tbody>{''.join(srows)}</tbody>
</table>
</section>

<section id="family-overview" class="tab-panel" data-tab-group="master">
<h2>6) Family Overview</h2>
{''.join(family_sections)}
</section>

<script>
function activateTab(group, targetId, updateHash){{
  const btns = [...document.querySelectorAll(`.tab-btn[data-tab-group="${{group}}"]`)];
  const panels = [...document.querySelectorAll(`.tab-panel[data-tab-group="${{group}}"]`)];
  btns.forEach(b => b.classList.toggle('active', b.dataset.tabTarget === targetId));
  panels.forEach(p => p.classList.toggle('active', p.id === targetId));
  if (updateHash && group === 'master') {{
    history.replaceState(null, '', `#${{targetId}}`);
  }}
  if (window.MathJax && window.MathJax.typesetPromise) {{
    window.MathJax.typesetPromise();
  }}
}}

const tabGroups = [...new Set(
  [...document.querySelectorAll('.tab-btn[data-tab-group]')].map(b => b.dataset.tabGroup)
)];
tabGroups.forEach(group => {{
  const btns = [...document.querySelectorAll(`.tab-btn[data-tab-group="${{group}}"]`)];
  if (!btns.length) return;
  btns.forEach(b => {{
    b.addEventListener('click', () => activateTab(group, b.dataset.tabTarget, group === 'master'));
  }});
  const initial = btns.find(b => b.classList.contains('active')) || btns[0];
  activateTab(group, initial.dataset.tabTarget, false);
}});

if (location.hash) {{
  const id = location.hash.slice(1);
  const masterBtn = document.querySelector(`.tab-btn[data-tab-group="master"][data-tab-target="${{id}}"]`);
  if (masterBtn) {{
    activateTab('master', id, false);
  }}
}}

const catalogRows = [...document.querySelectorAll('.catalog-row')];
const q = document.getElementById('searchInput');
const ff = document.getElementById('familyFilter');
const sf = document.getElementById('supportFilter');
function applyFilter(){{
  if (!q || !ff || !sf) return;
  const t = q.value.trim().toLowerCase();
  const f = ff.value;
  const s = sf.value;
  catalogRows.forEach(r => {{
    const okQ = !t || r.dataset.name.toLowerCase().includes(t);
    const okF = f === 'all' || r.dataset.family === f;
    const okS = s === 'all' || r.dataset.supported === s;
    r.style.display = (okQ && okF && okS) ? '' : 'none';
  }});
}}
if (q && ff && sf) {{
  q.addEventListener('input', applyFilter);
  ff.addEventListener('change', applyFilter);
  sf.addEventListener('change', applyFilter);
  applyFilter();
}}
</script>
"""
    return (
        "<!doctype html><html lang='ko'><head>"
        "<meta charset='UTF-8'/><meta name='viewport' content='width=device-width, initial-scale=1.0'/>"
        "<title>ComfyUI Sampler Site Index</title>"
        "<link rel='stylesheet' href='assets/style.css'/>"
        "<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script>"
        "<script async src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js'></script>"
        "</head><body><div class='wrap'><article class='paper'>"
        + body
        + "</article></div></body></html>"
    )


def render_root_alias() -> str:
    return """<!doctype html>
<html lang="ko"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ComfyUI Sampler Docs Hub</title>
<meta http-equiv="refresh" content="0; url=sampler_site/index.html"/>
<style>
body{margin:0;font-family:"IBM Plex Sans KR","Pretendard","Noto Sans KR",sans-serif;background:#f4f7ef;color:#161811}
.wrap{max-width:920px;margin:48px auto;padding:0 16px}
.card{background:#fffefb;border:1px solid #d2d9c3;border-radius:16px;padding:22px;box-shadow:0 10px 24px rgba(0,0,0,.06)}
h1{margin:0 0 10px 0;font-family:"Source Serif 4","Noto Serif KR",serif}
p{margin:8px 0;line-height:1.68;color:#5b6252}
a.btn{display:inline-block;text-decoration:none;margin-top:12px;border:1px solid #9fbbaa;background:#e8f3ec;color:#174836;border-radius:10px;padding:8px 12px}
</style>
</head><body><div class="wrap"><div class="card">
<h1>ComfyUI Sampler Docs Hub</h1>
<p>최상위 문서는 <code>sampler_site/index.html</code>로 통합되었습니다.</p>
<p>자동으로 이동하지 않으면 아래 버튼을 사용하세요.</p>
<a class="btn" href="sampler_site/index.html">sampler_site/index.html 열기</a>
</div></div></body></html>
"""


def main() -> None:
    entries, details = build_data()
    write(ASSET_DIR / "style.css", css_text())

    order = [e["name"] for e in entries]
    for i, name in enumerate(order):
        prev_name = order[i - 1] if i > 0 else None
        next_name = order[i + 1] if i < len(order) - 1 else None
        write(SAMPLER_DIR / f"{name}.html", render_sampler_page(order, details, name, prev_name, next_name))

    for s in scheduler_rows():
        write(SCHED_DIR / f"{s['name']}.html", render_scheduler_page(s["name"], s["formula"], s["effect"]))

    symbol_order = [s["id"] for s in SYMBOL_WIKI]
    write(SYMBOL_DIR / "index.html", render_symbol_index_page())
    for i, sid in enumerate(symbol_order):
        prev_sid = symbol_order[i - 1] if i > 0 else None
        next_sid = symbol_order[i + 1] if i < len(symbol_order) - 1 else None
        item = next(s for s in SYMBOL_WIKI if s["id"] == sid)
        write(SYMBOL_DIR / f"{sid}.html", render_symbol_page(item, prev_sid, next_sid))

    write(SITE / "index.html", render_index(entries, details))
    write(
        ROOT / "legacy" / "comfyui_sampler_docs_hub_ko.html",
        render_root_alias(),
    )
    print(f"Wrote site: {SITE}")
    print(
        f"Sampler pages: {len(order)}, Scheduler pages: {len(scheduler_rows())}, "
        f"Symbol pages: {len(symbol_order)} (+index)"
    )


if __name__ == "__main__":
    main()

