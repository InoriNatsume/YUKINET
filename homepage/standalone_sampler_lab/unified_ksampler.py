from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict
from typing import Any, Callable, Optional

import torch

from . import samplers as standalone_samplers
from . import schedulers as standalone_schedulers
from .taxonomy import SCHEDULER_NAMES_ALL, SAMPLER_NAMES_ALL, make_sampler_specs


class UnifiedKSampler:
    """
    통합 샘플러 엔진.

    backend:
    - "standalone": standalone_sampler_lab 구현 사용 (ComfyUI 비의존)
    - "comfy_native": 로컬 ComfyUI 코드를 import해 원본 KSampler 경로 사용
    """

    def __init__(self, comfy_root: Optional[str] = None):
        self.comfy_root = comfy_root
        self._comfy_sample = None
        self._comfy_loaded = False

    def list_sampler_specs(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in make_sampler_specs()]

    def list_sampler_names(self) -> list[str]:
        return list(SAMPLER_NAMES_ALL)

    def list_scheduler_names(self) -> list[str]:
        return list(SCHEDULER_NAMES_ALL)

    def calculate_sigmas_standalone(
        self,
        scheduler_name: str,
        steps: int,
        sigma_min: float,
        sigma_max: float,
        **kwargs,
    ) -> torch.Tensor:
        return standalone_schedulers.calculate_sigmas(
            scheduler_name=scheduler_name,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            **kwargs,
        )

    def sample_standalone(
        self,
        denoiser: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        sigmas: torch.Tensor,
        sampler_name: str,
        **kwargs,
    ) -> torch.Tensor:
        return standalone_samplers.sample(
            denoiser=denoiser,
            x=x,
            sigmas=sigmas,
            sampler_name=sampler_name,
            **kwargs,
        )

    def _ensure_comfy_native_loaded(self) -> None:
        if self._comfy_loaded:
            return

        if not self.comfy_root:
            raise RuntimeError(
                "comfy_native backend를 쓰려면 comfy_root가 필요합니다. "
                "예: ComfyUI-0.13.0/ComfyUI-0.13.0"
            )

        comfy_root = os.path.abspath(self.comfy_root)
        if comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)

        try:
            self._comfy_sample = importlib.import_module("comfy.sample")
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"ComfyUI import 실패: {e}\n"
                f"comfy_root 경로와 ComfyUI 의존성(torch 등)을 확인하세요."
            ) from e

        self._comfy_loaded = True

    def sample_comfy_native(
        self,
        model: Any,
        noise: torch.Tensor,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: torch.Tensor,
        denoise: float = 1.0,
        disable_noise: bool = False,
        start_step: Optional[int] = None,
        last_step: Optional[int] = None,
        force_full_denoise: bool = False,
        noise_mask: Optional[torch.Tensor] = None,
        sigmas: Optional[torch.Tensor] = None,
        callback: Optional[Callable[..., Any]] = None,
        disable_pbar: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        ComfyUI `comfy.sample.sample(...)` 경로를 그대로 호출한다.

        주의:
        - model은 ComfyUI MODEL 객체(ModelPatcher 기반)여야 함
        - positive/negative는 ComfyUI CONDITIONING 타입 구조여야 함
        """
        self._ensure_comfy_native_loaded()
        assert self._comfy_sample is not None

        return self._comfy_sample.sample(
            model=model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            sigmas=sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

    def sample(
        self,
        backend: str,
        **kwargs,
    ) -> torch.Tensor:
        """
        통합 호출 엔트리.

        backend="standalone"이면 sample_standalone kwargs를,
        backend="comfy_native"이면 sample_comfy_native kwargs를 넘기면 된다.
        """
        mode = backend.lower()
        if mode == "standalone":
            return self.sample_standalone(**kwargs)
        if mode == "comfy_native":
            return self.sample_comfy_native(**kwargs)
        raise ValueError(f"Unknown backend: {backend}")

