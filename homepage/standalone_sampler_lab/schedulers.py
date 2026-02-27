import math
from typing import Optional

import torch


def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(
    n: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: str = "cpu",
) -> torch.Tensor:
    """ComfyUI `karras` 스케줄과 동일한 형태."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


def get_sigmas_exponential(
    n: int,
    sigma_min: float,
    sigma_max: float,
    device: str = "cpu",
) -> torch.Tensor:
    """ComfyUI `exponential` 스케줄과 동일한 형태."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def kl_optimal_scheduler(
    n: int,
    sigma_min: float,
    sigma_max: float,
    device: str = "cpu",
) -> torch.Tensor:
    """ComfyUI `kl_optimal` 스케줄과 동일한 핵심 식."""
    adj_idxs = torch.arange(n, dtype=torch.float32, device=device).div_(max(n - 1, 1))
    sigmas = torch.zeros(n + 1, dtype=torch.float32, device=device)
    sigmas[:-1] = torch.tan(adj_idxs * math.atan(sigma_min) + (1.0 - adj_idxs) * math.atan(sigma_max))
    return sigmas


def linear_quadratic_schedule(
    steps: int,
    sigma_max: float,
    threshold_noise: float = 0.025,
    linear_steps: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """ComfyUI `linear_quadratic` 스케줄의 독립형 구현."""
    if steps <= 1:
        return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)

    if linear_steps is None:
        linear_steps = steps // 2
    linear_steps = max(1, min(linear_steps, steps - 1))

    linear_sigma = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * steps
    quadratic_steps = steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * (quadratic_steps**2))
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)

    quadratic_sigma = [
        quadratic_coef * (i**2) + linear_coef * i + const
        for i in range(linear_steps, steps)
    ]
    sigma_schedule = linear_sigma + quadratic_sigma + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.tensor(sigma_schedule, dtype=torch.float32, device=device) * sigma_max


def make_sigma_ladder(
    sigma_min: float,
    sigma_max: float,
    num_train_sigmas: int = 1000,
    device: str = "cpu",
) -> torch.Tensor:
    """
    discrete model_sampling.sigmas를 대체할 log-spaced ladder.

    ComfyUI의 simple/ddim/beta 류 스케줄은 본래 모델 내부의 discrete sigma 테이블을 사용하므로,
    독립형에서는 이 ladder를 surrogate로 사용한다.
    """
    if sigma_min <= 0.0:
        sigma_min = 1e-4
    ladder = torch.linspace(math.log(sigma_min), math.log(sigma_max), num_train_sigmas, device=device).exp()
    return ladder


def simple_scheduler_from_ladder(ladder: torch.Tensor, steps: int) -> torch.Tensor:
    sigs = []
    ss = len(ladder) / max(steps, 1)
    for i in range(steps):
        sigs.append(float(ladder[-(1 + int(i * ss))]))
    sigs.append(0.0)
    return torch.tensor(sigs, dtype=torch.float32, device=ladder.device)


def ddim_uniform_from_ladder(ladder: torch.Tensor, steps: int) -> torch.Tensor:
    sigs = []
    x = 1
    if x < len(ladder) and math.isclose(float(ladder[x]), 0.0, abs_tol=1e-5):
        steps += 1
    else:
        sigs = [0.0]

    ss = max(len(ladder) // max(steps, 1), 1)
    while x < len(ladder):
        sigs.append(float(ladder[x]))
        x += ss
    sigs = sigs[::-1]
    return torch.tensor(sigs, dtype=torch.float32, device=ladder.device)


def normal_scheduler_from_range(
    sigma_min: float,
    sigma_max: float,
    steps: int,
    sgm: bool = False,
    device: str = "cpu",
) -> torch.Tensor:
    """
    ComfyUI normal_scheduler의 독립형 근사.

    ComfyUI 원본은 model_sampling.sigma(timestep)로 계산한다.
    독립형에서는 log-sigma 선형 보간으로 근사한다.
    """
    append_zero_term = True
    if sgm:
        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)[:-1]
    else:
        ts = torch.linspace(0.0, 1.0, steps, device=device)
    log_s = math.log(max(sigma_max, 1e-12))
    log_e = math.log(max(sigma_min, 1e-12))
    sigs = torch.exp(log_s + (log_e - log_s) * ts)
    if append_zero_term:
        sigs = append_zero(sigs)
    return sigs.to(torch.float32)


def beta_scheduler_from_ladder(
    ladder: torch.Tensor,
    steps: int,
    alpha: float = 0.6,
    beta: float = 0.6,
) -> torch.Tensor:
    """
    ComfyUI beta_scheduler의 독립형 구현.

    주의: torch.distributions.Beta.icdf가 필요하다.
    """
    if steps <= 0:
        return torch.zeros(1, dtype=torch.float32, device=ladder.device)

    dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32, device=ladder.device),
        torch.tensor(beta, dtype=torch.float32, device=ladder.device),
    )

    # ComfyUI와 동일하게 endpoint=False인 grid를 사용.
    u = 1.0 - torch.arange(steps, dtype=torch.float32, device=ladder.device) / steps
    u = u.clamp(1e-6, 1.0 - 1e-6)
    q = dist.icdf(u)
    indices = torch.round(q * (len(ladder) - 1)).to(torch.long).clamp(0, len(ladder) - 1)

    sigs = []
    last_idx = -1
    for idx in indices.tolist():
        if idx != last_idx:
            sigs.append(float(ladder[idx]))
        last_idx = idx
    sigs.append(0.0)
    return torch.tensor(sigs, dtype=torch.float32, device=ladder.device)


def calculate_sigmas(
    scheduler_name: str,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    sigma_ladder: Optional[torch.Tensor] = None,
    device: str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """
    ComfyUI scheduler 이름과 동일한 인터페이스를 노린 독립형 진입점.
    """
    if steps <= 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    name = scheduler_name.lower()
    if sigma_ladder is None:
        sigma_ladder = make_sigma_ladder(sigma_min, sigma_max, device=device)
    else:
        sigma_ladder = sigma_ladder.to(device=device, dtype=torch.float32)

    if name == "karras":
        return get_sigmas_karras(
            n=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=float(kwargs.get("rho", 7.0)),
            device=device,
        )
    if name == "exponential":
        return get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    if name == "kl_optimal":
        return kl_optimal_scheduler(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    if name == "linear_quadratic":
        return linear_quadratic_schedule(
            steps=steps,
            sigma_max=sigma_max,
            threshold_noise=float(kwargs.get("threshold_noise", 0.025)),
            linear_steps=kwargs.get("linear_steps"),
            device=device,
        )
    if name == "simple":
        return simple_scheduler_from_ladder(sigma_ladder, steps)
    if name == "ddim_uniform":
        return ddim_uniform_from_ladder(sigma_ladder, steps)
    if name == "normal":
        return normal_scheduler_from_range(sigma_min, sigma_max, steps, sgm=False, device=device)
    if name == "sgm_uniform":
        return normal_scheduler_from_range(sigma_min, sigma_max, steps, sgm=True, device=device)
    if name == "beta":
        return beta_scheduler_from_ladder(
            sigma_ladder,
            steps=steps,
            alpha=float(kwargs.get("alpha", 0.6)),
            beta=float(kwargs.get("beta", 0.6)),
        )

    raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")

