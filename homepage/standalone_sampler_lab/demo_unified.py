"""
통합 엔진 데모:
- sampler taxonomy 조회
- standalone backend 샘플링

실행:
python standalone_sampler_lab\\demo_unified.py
"""

import torch

from standalone_sampler_lab.unified_ksampler import UnifiedKSampler


def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while x.ndim < target_ndim:
        x = x.unsqueeze(-1)
    return x


def toy_denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    s = append_dims(sigma, x.ndim)
    return x / (1.0 + s * s)


def main() -> None:
    engine = UnifiedKSampler()
    specs = engine.list_sampler_specs()
    print("sampler count:", len(specs))
    print("first 5 samplers:", [x["name"] for x in specs[:5]])

    x = torch.randn(1, 4, 32, 32)
    sigmas = engine.calculate_sigmas_standalone(
        scheduler_name="karras",
        steps=20,
        sigma_min=0.0291675,
        sigma_max=14.614642,
        rho=7.0,
    )
    y = engine.sample(
        backend="standalone",
        denoiser=toy_denoiser,
        x=x,
        sigmas=sigmas,
        sampler_name="dpmpp_2m_sde",
        eta=1.0,
        s_noise=1.0,
        solver_type="midpoint",
        seed=42,
    )
    print("output mean/std:", float(y.mean()), float(y.std()))


if __name__ == "__main__":
    main()

