from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ComfyUI comfy.samplers.KSAMPLER_NAMES + extra sampler names
SAMPLER_NAMES_ALL = [
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

SCHEDULER_NAMES_ALL = [
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
]


STANDALONE_SUPPORTED = {
    "euler",
    "heun",
    "euler_ancestral",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
}


@dataclass(frozen=True)
class SamplerSpec:
    name: str
    family: str
    stochastic: str  # "no" | "yes" | "optional"
    cfg_pp: bool
    gpu_variant: bool
    extra_params: str
    standalone_supported: bool
    note: str


def _family_of(name: str) -> str:
    if name.startswith("euler"):
        return "Euler"
    if name.startswith("heun") or "exp_heun" in name:
        return "Heun/ExpHeun"
    if name.startswith("dpmpp"):
        return "DPM++"
    if name.startswith("dpm_2") or name.startswith("dpm_fast") or name.startswith("dpm_adaptive"):
        return "DPM"
    if name in {"lms", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp"}:
        return "Linear/Residual Multistep"
    if name in {"ipndm", "ipndm_v", "deis"}:
        return "N-DM / DEIS"
    if name in {"ddpm", "lcm"}:
        return "DDPM/LCM"
    if name.startswith("gradient_estimation"):
        return "Gradient Estimation"
    if name == "er_sde":
        return "ER-SDE"
    if name.startswith("seeds"):
        return "SEEDS"
    if name.startswith("sa_solver"):
        return "SA-Solver"
    if name.startswith("uni_pc"):
        return "UniPC"
    if name == "ddim":
        return "DDIM(alias)"
    return "Other"


def _stochastic_of(name: str) -> str:
    if any(
        token in name
        for token in ["_ancestral", "_sde", "ddpm", "seeds", "sa_solver", "er_sde", "lcm"]
    ):
        return "yes"
    if name in {"dpm_fast", "dpm_adaptive"}:
        return "optional"
    return "no"


def _extra_params_of(name: str) -> str:
    if name in {"euler", "heun", "heunpp2", "dpm_2"}:
        return "s_churn, s_tmin, s_tmax, s_noise"
    if name in {"euler_ancestral", "euler_ancestral_cfg_pp", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp"}:
        return "eta, s_noise"
    if name in {"dpmpp_sde", "dpmpp_sde_gpu"}:
        return "eta, s_noise, r"
    if name in {"dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu"}:
        return "eta, s_noise, solver_type"
    if name in {"dpmpp_3m_sde", "dpmpp_3m_sde_gpu"}:
        return "eta, s_noise"
    if name == "dpm_fast":
        return "eta, s_noise"
    if name == "dpm_adaptive":
        return "order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise"
    if name == "lms":
        return "order"
    if name in {"ipndm", "ipndm_v"}:
        return "max_order"
    if name == "deis":
        return "max_order, deis_mode"
    if name.startswith("gradient_estimation"):
        return "ge_gamma"
    if name == "er_sde":
        return "s_noise, noise_scaler, max_stage"
    if name == "seeds_2":
        return "eta, s_noise, r, solver_type"
    if name == "seeds_3":
        return "eta, s_noise, r_1, r_2"
    if name in {"sa_solver", "sa_solver_pece"}:
        return "tau_func, s_noise, predictor_order, corrector_order, use_pece, simple_order_2"
    return "-"


def _note_of(name: str) -> str:
    if name in {"ddim"}:
        return "ComfyUI 내부에서 euler+random inpaint option alias로 취급"
    if name in {"uni_pc", "uni_pc_bh2"}:
        return "k_diffusion 경로가 아니라 comfy.extra_samplers.uni_pc 경로"
    if name.endswith("_gpu"):
        return "노이즈 샘플링 경로 GPU 변형"
    if "_cfg_pp" in name:
        return "CFG++ 계열"
    return ""


def make_sampler_specs(names: Iterable[str] = SAMPLER_NAMES_ALL) -> list[SamplerSpec]:
    specs: list[SamplerSpec] = []
    for n in names:
        specs.append(
            SamplerSpec(
                name=n,
                family=_family_of(n),
                stochastic=_stochastic_of(n),
                cfg_pp="_cfg_pp" in n,
                gpu_variant=n.endswith("_gpu"),
                extra_params=_extra_params_of(n),
                standalone_supported=n in STANDALONE_SUPPORTED,
                note=_note_of(n),
            )
        )
    return specs

