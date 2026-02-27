# standalone_sampler_lab

ComfyUI 의존을 최소화하면서 sampler/scheduler를 독립적으로 실험하기 위한 모듈입니다.  
파라미터 이름은 ComfyUI KSampler와 최대한 맞췄습니다 (`sampler_name`, `scheduler_name`, `eta`, `s_noise`, `solver_type` 등).

## 포함 기능

- scheduler
  - `karras`, `exponential`, `kl_optimal`, `linear_quadratic`
  - `simple`, `ddim_uniform`, `normal`, `sgm_uniform`, `beta`
- sampler
  - `euler`
  - `heun`
  - `euler_ancestral`
  - `dpmpp_2m_sde` / `dpmpp_2m_sde_heun` (`dpmpp_2m_sde_gpu` alias 포함)
- 통합 엔진
  - `UnifiedKSampler` (`standalone` / `comfy_native` backend)
- taxonomy
  - ComfyUI sampler/scheduler 목록, 패밀리 분류, 추가 파라미터 메타데이터

## 주요 파일

- `standalone_sampler_lab/schedulers.py`
- `standalone_sampler_lab/samplers.py`
- `standalone_sampler_lab/taxonomy.py`
- `standalone_sampler_lab/unified_ksampler.py`
- `standalone_sampler_lab/generate_sampler_homepage.py`
- `standalone_sampler_lab/generate_sampler_site.py`
- `standalone_sampler_lab/demo_minimal.py`
- `standalone_sampler_lab/demo_unified.py`

## 사용 예시

```python
import torch
from standalone_sampler_lab.schedulers import calculate_sigmas
from standalone_sampler_lab.samplers import sample


def denoiser(x, sigma):
    s = sigma.view(-1, 1, 1, 1)
    return x / (1 + s * s)


x = torch.randn(1, 4, 32, 32)
sigmas = calculate_sigmas(
    scheduler_name="karras",
    steps=20,
    sigma_min=0.0291675,
    sigma_max=14.614642,
)

out = sample(
    denoiser=denoiser,
    x=x,
    sigmas=sigmas,
    sampler_name="euler_ancestral",
    eta=1.0,
    s_noise=1.0,
    seed=0,
)
```

## 통합 엔진 예시

```python
from standalone_sampler_lab.unified_ksampler import UnifiedKSampler

engine = UnifiedKSampler(comfy_root="ComfyUI-0.13.0/ComfyUI-0.13.0")
specs = engine.list_sampler_specs()
print(specs[0]["name"])
```

- `backend="standalone"`: 독립 구현 경로 실행
- `backend="comfy_native"`: 로컬 ComfyUI `comfy.sample.sample` 경로 위임

## 문서 생성

### 1) 통합 허브(레거시 단일 페이지)

```powershell
python standalone_sampler_lab\generate_sampler_homepage.py
```

생성 파일(레거시):

- `legacy/comfyui_sampler_docs_hub_ko.html`

### 2) 샘플러 사이트(권장, 현재 최상위 문서 체계)

```powershell
python standalone_sampler_lab\generate_sampler_site.py
```

생성 경로:

- `sampler_site/index.html` (최상위 마스터 문서: 통합 관점 + 카탈로그)
- `sampler_site/sampler/*.html` (개별 sampler 문서)
- `sampler_site/scheduler/*.html` (개별 scheduler 문서)
- `sampler_site/assets/style.css`

참고: `legacy/comfyui_sampler_docs_hub_ko.html`는 호환용 엔트리이며 `sampler_site/index.html`로 이동합니다.

## 참고

- 본 모듈은 연구/프로토타입 목적의 최소 구현입니다.
- ComfyUI 전체 기능(모델 패치, hooks, nested latent, ControlNet, inpaint wrapper 등)은 완전 복제하지 않습니다.
- 실행 환경에 `torch`가 없으면 데모 실행은 실패할 수 있습니다.
