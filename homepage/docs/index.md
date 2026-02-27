---
hide:
  - navigation
  - toc
---

<div class="hero-section" markdown>

# {{ site.title }}

**AI 딸깍질**이 **수학**과 **코드**를 만날 때, 이야기는 시작된다.

</div>

<div class="card-grid">

<a class="card" href="theory/">
<span class="card-icon">📐</span>
<h3>이론 (Theory)</h3>
<p>확산 수학, 샘플러, 옵티마이저, 훈련 이론, 아키텍처.
<strong>모델과 도구에 무관한 보편적 수학/알고리즘.</strong>
SDE/ODE, Flow Matching, FPE, 최적수송까지.</p>
</a>

<a class="card" href="models/">
<span class="card-icon">🧩</span>
<h3>모델 (Models)</h3>
<p>SD 1.5, SDXL, Flux, SD3, Wan 등 개별 모델이
<strong>이론의 부품을 어떻게 조합하는지</strong> 분석.
아키텍처 선택, 확산 방식, 예측 유형 비교.</p>
</a>

<a class="card" href="tools/">
<span class="card-icon">🔧</span>
<h3>도구 (Tools)</h3>
<p>ComfyUI, kohya(sd-scripts), HuggingFace, DiffSynth.
<strong>각 도구가 이론/모델을 어떻게 구현하는지</strong> 코드 수준 분석.
파라미터 ↔ 수식 매핑.</p>
</a>

<a class="card" href="lab/">
<span class="card-icon">🧪</span>
<h3>실험 (Lab)</h3>
<p>실제 훈련/추론 실험 기록.
세팅, TensorBoard/W&amp;B 그래프, 결과 비교, 분석.
<strong>검증된 레시피 모음.</strong></p>
</a>

</div>

## 분석 대상 코드베이스

| 코드베이스 | 버전 | 역할 |
|---|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | `{{ ver.comfyui }}` | 추론 (노드 기반) |
| [sd-scripts (kohya)](https://github.com/kohya-ss/sd-scripts) | `{{ ver.sdscripts }}` | 훈련 (CLI 기반) |
| [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) | `{{ ver.diffusion_pipe }}` | 훈련 (Pipeline Parallel) |
| [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | `{{ ver.diffsynth }}` | 훈련+추론 (통합) |
| [HuggingFace diffusers](https://github.com/huggingface/diffusers) | `{{ ver.diffusers }}` | 훈련+추론 (Pipeline) |
| [HuggingFace transformers](https://github.com/huggingface/transformers) | `{{ ver.transformers }}` | 모델 아키텍처 |
