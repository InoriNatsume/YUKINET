import math
from typing import Callable, Optional

import torch

Tensor = torch.Tensor
DenoiserFn = Callable[[Tensor, Tensor], Tensor]
NoiseSamplerFn = Callable[[Tensor, Tensor], Tensor]
CallbackFn = Callable[[dict], None]


def append_dims(x: Tensor, target_ndim: int) -> Tensor:
    while x.ndim < target_ndim:
        x = x.unsqueeze(-1)
    return x


def to_d(x: Tensor, sigma: Tensor, denoised: Tensor) -> Tensor:
    """ComfyUI k-diffusion과 동일한 기본 변환식."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from: Tensor, sigma_to: Tensor, eta: float = 1.0) -> tuple[Tensor, Tensor]:
    """
    ComfyUI get_ancestral_step와 동일한 핵심 식.
    반환: (sigma_down, sigma_up)
    """
    if not eta:
        return sigma_to, torch.zeros_like(sigma_to)
    sigma_up = torch.minimum(
        sigma_to,
        torch.tensor(eta, dtype=sigma_to.dtype, device=sigma_to.device)
        * torch.sqrt((sigma_to**2) * (sigma_from**2 - sigma_to**2) / (sigma_from**2)),
    )
    sigma_down = torch.sqrt(torch.clamp(sigma_to**2 - sigma_up**2, min=0.0))
    return sigma_down, sigma_up


def default_noise_sampler(x: Tensor, seed: Optional[int] = None) -> NoiseSamplerFn:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    def _sample(_sigma: Tensor, _sigma_next: Tensor) -> Tensor:
        return torch.randn(x.size(), dtype=x.dtype, device=x.device, generator=generator)

    return _sample


def _make_sigma_batch(x: Tensor, sigma: Tensor) -> Tensor:
    return x.new_ones([x.shape[0]]) * sigma


@torch.no_grad()
def sample_euler(
    denoiser: DenoiserFn,
    x: Tensor,
    sigmas: Tensor,
    callback: Optional[CallbackFn] = None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
) -> Tensor:
    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i]
        if s_churn > 0:
            gamma = min(s_churn / max(len(sigmas) - 1, 1), math.sqrt(2.0) - 1.0) if s_tmin <= sigma_i <= s_tmax else 0.0
            sigma_hat = sigma_i * (gamma + 1.0)
        else:
            gamma = 0.0
            sigma_hat = sigma_i

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * torch.sqrt(torch.clamp(sigma_hat**2 - sigma_i**2, min=0.0))

        denoised = denoiser(x, _make_sigma_batch(x, sigma_hat))
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt

        if callback is not None:
            callback({"i": i, "x": x, "sigma": sigma_i, "sigma_hat": sigma_hat, "denoised": denoised})
    return x


@torch.no_grad()
def sample_heun(
    denoiser: DenoiserFn,
    x: Tensor,
    sigmas: Tensor,
    callback: Optional[CallbackFn] = None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
) -> Tensor:
    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i]
        if s_churn > 0:
            gamma = min(s_churn / max(len(sigmas) - 1, 1), math.sqrt(2.0) - 1.0) if s_tmin <= sigma_i <= s_tmax else 0.0
            sigma_hat = sigma_i * (gamma + 1.0)
        else:
            gamma = 0.0
            sigma_hat = sigma_i

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * torch.sqrt(torch.clamp(sigma_hat**2 - sigma_i**2, min=0.0))

        denoised = denoiser(x, _make_sigma_batch(x, sigma_hat))
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, _make_sigma_batch(x_2, sigmas[i + 1]))
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            x = x + ((d + d_2) / 2.0) * dt

        if callback is not None:
            callback({"i": i, "x": x, "sigma": sigma_i, "sigma_hat": sigma_hat, "denoised": denoised})
    return x


@torch.no_grad()
def sample_euler_ancestral(
    denoiser: DenoiserFn,
    x: Tensor,
    sigmas: Tensor,
    callback: Optional[CallbackFn] = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Optional[NoiseSamplerFn] = None,
    seed: Optional[int] = None,
) -> Tensor:
    if noise_sampler is None:
        noise_sampler = default_noise_sampler(x, seed=seed)

    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i]
        denoised = denoiser(x, _make_sigma_batch(x, sigma_i))
        sigma_down, sigma_up = get_ancestral_step(sigma_i, sigmas[i + 1], eta=eta)
        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigma_i, denoised)
            dt = sigma_down - sigma_i
            x = x + d * dt + noise_sampler(sigma_i, sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({"i": i, "x": x, "sigma": sigma_i, "sigma_hat": sigma_i, "denoised": denoised})
    return x


def sigma_to_half_log_snr(sigma: Tensor, mode: str = "vp") -> Tensor:
    """
    ComfyUI의 sigma_to_half_log_snr 단순화 버전.
    - vp: lambda = -log(sigma)
    - const: lambda = -logit(sigma)
    """
    sigma = torch.clamp(sigma, min=1e-12, max=1.0 - 1e-6)
    if mode == "const":
        return -torch.logit(sigma)
    return -torch.log(sigma)


@torch.no_grad()
def sample_dpmpp_2m_sde(
    denoiser: DenoiserFn,
    x: Tensor,
    sigmas: Tensor,
    callback: Optional[CallbackFn] = None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    noise_sampler: Optional[NoiseSamplerFn] = None,
    seed: Optional[int] = None,
    solver_type: str = "midpoint",
    snr_mode: str = "vp",
) -> Tensor:
    """
    ComfyUI sample_dpmpp_2m_sde의 독립형 축약 구현.
    """
    if len(sigmas) <= 1:
        return x
    if solver_type not in {"midpoint", "heun"}:
        raise ValueError("solver_type must be 'midpoint' or 'heun'")

    if noise_sampler is None:
        noise_sampler = default_noise_sampler(x, seed=seed)

    old_denoised = None
    h_last = None

    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = denoiser(x, _make_sigma_batch(x, sigma_i))

        if sigma_next == 0:
            x = denoised
        else:
            lambda_s = sigma_to_half_log_snr(sigma_i, mode=snr_mode)
            lambda_t = sigma_to_half_log_snr(sigma_next, mode=snr_mode)
            h = lambda_t - lambda_s
            h_eta = h * (eta + 1.0)
            alpha_t = sigma_next * torch.exp(lambda_t)

            x = (sigma_next / sigma_i) * torch.exp(-h * eta) * x + alpha_t * torch.neg(torch.expm1(-h_eta)) * denoised

            if old_denoised is not None and h_last is not None:
                r = h_last / h
                if solver_type == "heun":
                    corr = alpha_t * (torch.neg(torch.expm1(-h_eta)) / (-h_eta) + 1.0) * (1.0 / r) * (denoised - old_denoised)
                else:
                    corr = 0.5 * alpha_t * torch.neg(torch.expm1(-h_eta)) * (1.0 / r) * (denoised - old_denoised)
                x = x + corr

            if eta > 0 and s_noise > 0:
                noise_coeff = sigma_next * torch.sqrt(torch.neg(torch.expm1(-2.0 * h * eta)))
                x = x + noise_sampler(sigma_i, sigma_next) * noise_coeff * s_noise

            h_last = h

        old_denoised = denoised
        if callback is not None:
            callback({"i": i, "x": x, "sigma": sigma_i, "sigma_hat": sigma_i, "denoised": denoised})
    return x


def sample(
    denoiser: DenoiserFn,
    x: Tensor,
    sigmas: Tensor,
    sampler_name: str,
    **kwargs,
) -> Tensor:
    """
    ComfyUI sampler_name 스타일의 독립형 dispatcher.
    """
    name = sampler_name.lower()
    if name == "euler":
        return sample_euler(denoiser, x, sigmas, **kwargs)
    if name == "heun":
        return sample_heun(denoiser, x, sigmas, **kwargs)
    if name == "euler_ancestral":
        return sample_euler_ancestral(denoiser, x, sigmas, **kwargs)
    if name in {"dpmpp_2m_sde", "dpmpp_2m_sde_gpu"}:
        return sample_dpmpp_2m_sde(denoiser, x, sigmas, **kwargs)
    if name == "dpmpp_2m_sde_heun":
        return sample_dpmpp_2m_sde(denoiser, x, sigmas, solver_type="heun", **kwargs)
    raise ValueError(f"Unsupported sampler_name: {sampler_name}")

