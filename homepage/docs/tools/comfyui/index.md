# ComfyUI

<span class="version-badge">ComfyUI v0.14.2</span>

!!! quote "분석 기준 소스"
    **리포지토리**: [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)  
    **분석 버전**: `0.14.2`  
    **로컬 경로**: `Reference/ComfyUI-0.14.2/`

## 아키텍처 개요

ComfyUI의 샘플링은 3계층 구조:

1. **`nodes.py`** → `common_ksampler(...)` — UI 노드
2. **`comfy/samplers.py`** → `KSampler` + `CFGGuider` — 오케스트레이션
3. **`comfy/k_diffusion/sampling.py`** → `sample_*` — 실제 수치적분

!!! tip "핵심 포인트"
    UI 노드가 직접 알고리즘을 돌리는 구조가 아니라, `KSampler` + `CFGGuider` + `k_diffusion sample_*`의 3계층 구조.

## 주요 분석 페이지

- [샘플러 구현](samplers.md) — `sample_*` 함수들의 코드 분석, 파라미터 ↔ 수식 매핑

## KSampler 파라미터 매핑

| UI 파라미터 | 내부 코드 | 수식 관점 | 이론 참조 |
|---|---|---|---|
| `seed` | `prepare_noise(seed)` | $\xi_0 \sim \mathcal{N}(0,I)$ | — |
| `steps` | `calculate_sigmas(steps)` | $N = \mathrm{len}(\sigma) - 1$ | — |
| `sampler_name` | `ksampler(...)` | $\Phi_\text{drift/corr/noise}$ 선택 | [→ 이론/샘플러](../../theory/samplers/index.md) |
| `scheduler` | `SCHEDULER_HANDLERS` | $\sigma_0, \dots, \sigma_N$ 재배치 | [→ 이론/샘플러](../../theory/samplers/index.md) |
| `cfg` | `cfg_function` | $v_\text{cfg} = v_u + w(v_c - v_u)$ | — |
| `denoise` | tail slice | $\sigma$ 경로 일부만 사용 | — |
