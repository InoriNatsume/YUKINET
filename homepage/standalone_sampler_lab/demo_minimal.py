"""
ComfyUI 독립형 샘플러 최소 데모.

실행 (Windows PowerShell):
python standalone_sampler_lab\\demo_minimal.py
"""

import torch

from standalone_sampler_lab.samplers import sample
from standalone_sampler_lab.schedulers import calculate_sigmas


def append_dims(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while x.ndim < target_ndim:
        x = x.unsqueeze(-1)
    return x


def toy_denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    실제 UNet 대신 쓰는 장난감 denoiser.
    sigma가 클수록 x를 더 강하게 축소한다.
    """
    s = append_dims(sigma, x.ndim)
    return x / (1.0 + s * s)


def main() -> None:
    device = "cpu"
    batch, channels, height, width = 1, 4, 32, 32
    x = torch.randn(batch, channels, height, width, device=device)

    sigma_min = 0.0291675
    sigma_max = 14.614642
    steps = 20

    sigmas = calculate_sigmas(
        scheduler_name="karras",
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=7.0,
        device=device,
    )

    out = sample(
        denoiser=toy_denoiser,
        x=x,
        sigmas=sigmas,
        sampler_name="dpmpp_2m_sde",
        eta=1.0,
        s_noise=1.0,
        solver_type="midpoint",
        seed=1234,
    )
    print("output mean/std:", float(out.mean()), float(out.std()))


if __name__ == "__main__":
    main()

