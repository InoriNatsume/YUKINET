# 🧩 모델

각 모델이 [이론](../theory/index.md)의 부품을 **어떻게 조합했는지** 분석합니다.

| 모델 | 아키텍처 | 확산 방식 | 예측 유형 | 상태 | HuggingFace |
|---|---|---|---|---|---|
| SD 1.5 | UNet | DDPM (VP-SDE) | ε-prediction | 예정 | — |
| SDXL | UNet (대형) | DDPM | ε/v-prediction | 예정 | — |
| SD3 | MMDiT | Flow Matching | velocity | 예정 | — |
| [Flux](flux.md) | DiT | Flow Matching | velocity | ✅ | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Wan | — | Flow Matching | velocity | 예정 | [Wan2.1-T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| [Anima](anima.md) | DiT (Cosmos) | Flow Matching | — | ✅ | [Anima](https://huggingface.co/circlestone-labs/Anima) |
| [Z-Image](z_image.md) | DiT (Flux) | Flow Matching | — | ✅ | [z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo) |
