# ComfyUI Sampler 동작 개발자 가이드 (ComfyUI 0.13.0 기준)

이 문서는 ComfyUI에서 sampler가 실제로 어떻게 동작하는지, 그리고 스크립트 의존을 줄이고 자체 추론 코드를 만들거나 커스터마이징할 때 어떤 지점을 봐야 하는지를 코드 기준으로 정리한 문서다.

기준 코드 경로:
- `ComfyUI-0.13.0/ComfyUI-0.13.0/nodes.py`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/sample.py`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/samplers.py`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/k_diffusion/sampling.py`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy_extras/nodes_custom_sampler.py`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/comfy/model_sampling.py`
- `ComfyUI-0.13.0/ComfyUI-0.13.0/latent_preview.py`

---

## 1) 전체 실행 흐름

기본 KSampler 노드 기준 호출 체인은 다음과 같다.

1. `nodes.py`의 `common_ksampler(...)`
- latent 채널/해상도 보정: `comfy.sample.fix_empty_latent_channels(...)`
- noise 준비:
  - 일반: `comfy.sample.prepare_noise(latent, seed, batch_index)`
  - `disable_noise=True`: 0 noise
- `noise_mask`(inpaint mask) 추출
- preview callback 준비: `latent_preview.prepare_callback(...)`
- `comfy.sample.sample(...)` 호출

2. `comfy/sample.py`의 `sample(...)`
- `comfy.samplers.KSampler(...)` 객체 생성
- 내부 `KSampler.sample(...)` 실행

3. `comfy/samplers.py`의 `KSampler.sample(...)`
- scheduler로 sigma 시퀀스 생성
- `denoise/start_step/last_step/force_full_denoise` 반영
- 실제 sampler 객체(`ksampler(...)`, `uni_pc`, `ddim`) 선택
- `sample(...)` -> `CFGGuider` 경유 실행

4. `comfy/samplers.py`의 `CFGGuider.sample(...)`
- cond/negative cond 정리 (`process_conds`)
- hook/ControlNet/extra cond 반영
- `KSAMPLER.sample(...)` 실행

5. `KSAMPLER.sample(...)`
- inpaint wrapper(`KSamplerX0Inpaint`) 적용
- model별 `noise_scaling`/`inverse_noise_scaling` 적용
- 최종적으로 `k_diffusion/sampling.py`의 `sample_*` 함수 호출

요약하면, UI 노드가 직접 알고리즘을 돌리는 구조가 아니라 `KSampler` + `CFGGuider` + `k_diffusion sample_*`의 3계층 구조다.

---

## 2) KSampler/KSamplerAdvanced 파라미터의 실제 의미

### KSampler (일반)
입력은 `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise` 중심이다.

- `steps`
  - 최종 sampling step 수.
  - 내부적으로는 `sigmas` 길이가 `steps + 1`이 되도록 생성된다.

- `cfg`
  - `cfg_function(...)`에서 `uncond + (cond - uncond) * cfg`로 합성된다.
  - model option hook으로 pre/post CFG 개입 가능.

- `sampler_name`
  - 실제 solver(`sample_euler`, `sample_dpmpp_2m_sde` 등) 선택.

- `scheduler`
  - sigma sequence 생성 방식 선택.

- `denoise`
  - 1.0이면 full denoise.
  - 0~1이면 내부적으로 `new_steps = int(steps / denoise)`를 계산한 뒤 tail 구간의 sigma만 잘라서 사용한다.
  - `<= 0`이면 sigma가 빈 텐서가 되고, 결과적으로 입력 latent를 그대로 반환하는 경로로 들어간다.

### KSamplerAdvanced (고급)
`KSamplerAdvanced.sample(...)`는 UI 파라미터를 내부 플래그로 번역해서 `common_ksampler(...)`를 호출한다.

- `add_noise = disable` -> `disable_noise = True`
- `return_with_leftover_noise = enable` -> `force_full_denoise = False`
- `start_at_step` -> `start_step`
- `end_at_step` -> `last_step`

핵심 포인트:
- `force_full_denoise=True`는 `last_step`로 중간 컷한 경우 마지막 sigma를 강제로 0으로 만들어 완전 복원 쪽으로 당긴다.
- `start_step`이 sigma 길이보다 크면 입력 latent(없으면 zero tensor) 반환으로 종료된다.

---

## 3) CFG/Conditioning/Inpaint 처리 구조

### CFG 처리
`comfy/samplers.py`의 핵심 함수:
- `sampling_function(...)`
- `cfg_function(...)`

동작 요점:
- `cond_scale == 1.0`이고 최적화 조건이 맞으면 uncond branch를 생략하는 최적화가 있다.
- `sampler_pre_cfg_function`, `sampler_cfg_function`, `sampler_post_cfg_function` hook으로 커스터마이징 가능.
- CFG++ 계열 sampler는 post-cfg hook으로 uncond denoised를 따로 캡처해서 업데이트에 사용한다.

### Conditioning 처리
`process_conds(...)`에서:
- area/mask 해상도 정렬
- `start_percent`, `end_percent`를 sigma/timestep 범위로 변환
- model별 `extra_conds` 인코딩
- ControlNet/GLIGEN 등 cond 균형 맞춤

### Inpaint 처리
`KSamplerX0Inpaint.__call__(...)`에서:
- `denoise_mask`가 있으면 마스크 영역만 denoise하고 나머지는 latent 원본 유지
- `model_options["denoise_mask_function"]` hook으로 mask를 step별 수정 가능

---

## 4) Sampler 목록과 계열별 동작

`comfy/samplers.py` 기준 sampler name:
- `KSAMPLER_NAMES`: `euler`, `euler_cfg_pp`, `euler_ancestral`, `euler_ancestral_cfg_pp`, `heun`, `heunpp2`, `exp_heun_2_x0`, `exp_heun_2_x0_sde`, `dpm_2`, `dpm_2_ancestral`, `lms`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2s_ancestral`, `dpmpp_2s_ancestral_cfg_pp`, `dpmpp_sde`, `dpmpp_sde_gpu`, `dpmpp_2m`, `dpmpp_2m_cfg_pp`, `dpmpp_2m_sde`, `dpmpp_2m_sde_gpu`, `dpmpp_2m_sde_heun`, `dpmpp_2m_sde_heun_gpu`, `dpmpp_3m_sde`, `dpmpp_3m_sde_gpu`, `ddpm`, `lcm`, `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `res_multistep_cfg_pp`, `res_multistep_ancestral`, `res_multistep_ancestral_cfg_pp`, `gradient_estimation`, `gradient_estimation_cfg_pp`, `er_sde`, `seeds_2`, `seeds_3`, `sa_solver`, `sa_solver_pece`
- `SAMPLER_NAMES`는 위 목록에 `ddim`, `uni_pc`, `uni_pc_bh2`가 추가된다.

### 4-1) 결정론(ODE 계열, 기본적으로 추가 랜덤 노이즈 없음)
대표:
- `euler`, `heun`, `dpm_2`, `lms`, `dpmpp_2m`, `ipndm`, `ipndm_v`, `deis`, `res_multistep`, `gradient_estimation`, `exp_heun_2_x0`

특징:
- seed를 고정하면 재현성이 높다.
- 스텝 수/스케줄 영향이 직접적으로 나타난다.

### 4-2) 확률론(ancestral/SDE 계열, 노이즈 재주입)
대표:
- `euler_ancestral`, `dpm_2_ancestral`, `dpmpp_2s_ancestral`, `res_multistep_ancestral`
- `dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_2m_sde_heun`, `dpmpp_3m_sde`
- `er_sde`, `seeds_2`, `seeds_3`, `sa_solver`, `sa_solver_pece`
- `exp_heun_2_x0_sde`

공통 파라미터 패턴:
- `eta`: stochastic 강도
- `s_noise`: 재주입 noise multiplier

중요:
- 기본 `KSampler` 노드에서는 이 파라미터를 직접 조절하지 못하고 함수 기본값을 사용한다.
- 정밀 튜닝하려면 `sampling/custom_sampling` 노드(`SamplerDPMPP_*`, `SamplerSASolver`, `SamplerSEEDS2` 등)로 가야 한다.

### 4-3) CFG++ 변형 계열
대표:
- `euler_cfg_pp`, `euler_ancestral_cfg_pp`, `dpmpp_2s_ancestral_cfg_pp`, `dpmpp_2m_cfg_pp`, `res_multistep_cfg_pp`, `res_multistep_ancestral_cfg_pp`, `gradient_estimation_cfg_pp`

특징:
- 내부적으로 post-CFG hook을 써서 uncond denoised를 별도로 사용한다.
- 일반 CFG와 업데이트 식이 달라 결과 성향이 달라질 수 있다.

### 4-4) 특수 케이스
- `dpm_fast`, `dpm_adaptive`
  - `ksampler(...)`에서 `sigma_min/sigma_max`를 별도로 계산해 호출한다.
  - adaptive는 `order`, `rtol`, `atol`, `h_init`, PID 계수 등 추가 하이퍼파라미터를 가진다.

- `ddim`
  - `sampler_object("ddim")`은 전통적 DDIM 함수를 직접 호출하는 구조가 아니라 내부적으로 `ksampler("euler", inpaint_options={"random": True})`를 만든다.
  - 즉 ComfyUI 내부 구현 관점에서는 별도 solver 구현명이 아니라 alias에 가깝다.

- `uni_pc`, `uni_pc_bh2`
  - `k_diffusion/sampling.py` 경로가 아니라 `comfy/extra_samplers/uni_pc.py` 경로를 탄다.

- `ModelSampling.CONST` 분기
  - `euler_ancestral`, `dpm_2_ancestral`는 모델 sampling 타입이 `CONST`인 경우 RF 전용 분기(`*_RF`)를 사용한다.
  - flow/rectified-flow 계열 모델에서 sampler 동작이 달라지는 핵심 포인트다.

---

## 5) Scheduler와 sigma 생성 로직

`SCHEDULER_HANDLERS`:
- `simple`
- `sgm_uniform`
- `karras`
- `exponential`
- `ddim_uniform`
- `beta`
- `normal`
- `linear_quadratic`
- `kl_optimal`

핵심 메커니즘:
- `calculate_sigmas(model_sampling, scheduler_name, steps)`가 공통 진입점.
- scheduler마다 호출 방식이 두 가지다.
  - `handler(model_sampling, steps)`
  - `handler(n, sigma_min, sigma_max)`

KSampler 보정 규칙:
- 일부 sampler(`dpm_2`, `dpm_2_ancestral`, `uni_pc`, `uni_pc_bh2`)는 penultimate sigma를 버리는 특수 처리(`DISCARD_PENULTIMATE_SIGMA_SAMPLERS`)가 있다.
- `denoise < 1`이면 더 긴 sigma를 만든 뒤 tail slice만 사용한다.

---

## 6) Custom Sampling 노드의 의미

`comfy_extras/nodes_custom_sampler.py`는 기본 KSampler UI에서 숨겨진 제어권을 노드 그래프로 노출한다.

구성요소:
- Scheduler 생성 노드
  - `BasicScheduler`, `KarrasScheduler`, `ExponentialScheduler`, `PolyexponentialScheduler`, `LaplaceScheduler`, `VPScheduler`, `SDTurboScheduler`, `BetaSamplingScheduler`
- Sigma 가공 노드
  - `SplitSigmas`, `SplitSigmasDenoise`, `FlipSigmas`, `SetFirstSigma`, `ExtendIntermediateSigmas`, `ManualSigmas`
- Sampler 객체 노드
  - `KSamplerSelect` + 파라미터 노출형 sampler 노드(`SamplerDPMPP_*`, `SamplerDPMAdaptative`, `SamplerER_SDE`, `SamplerSASolver`, `SamplerSEEDS2`, `SamplerEulerAncestral`, `SamplerLMS` 등)
- Guider/Noise 노드
  - `CFGGuider`, `BasicGuider`, `RandomNoise`, `DisableNoise`
- 실행 노드
  - `SamplerCustom`, `SamplerCustomAdvanced`

핵심 차이:
- 기본 `KSampler`는 고정된 인터페이스(steps/cfg/sampler/scheduler/denoise)만 제공.
- `SamplerCustom*`는 `sampler 객체`, `sigmas`, `noise`, `guider`를 분리해서 완전히 조합 가능.
- 즉 자체 엔진 프로토타이핑에는 `custom_sampling` 경로가 훨씬 적합하다.

---

## 7) 개발자가 자체 추론 코드를 만들 때의 최소 재현 순서

ComfyUI 동작을 가장 비슷하게 복제하려면 아래 순서를 그대로 따르는 것이 안전하다.

```python
# 1) latent 정규화
latent = fix_empty_latent_channels(model, latent)

# 2) noise 준비 (seed + batch_index 지원)
noise = prepare_noise(latent, seed, noise_inds=batch_index)

# 3) sigma 생성
sigmas = calculate_sigmas(model_sampling, scheduler, steps)
if denoise < 1:
    sigmas = denoise_tail_slice(sigmas, steps, denoise)
if sampler in DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
    sigmas = drop_penultimate(sigmas)

# 4) start/end step 컷
sigmas = apply_start_last_step(sigmas, start_step, last_step, force_full_denoise)

# 5) cond 처리 + cfg guider 준비
conds = process_conds(...)
guider = CFGGuider(model)
guider.set_conds(positive, negative)
guider.set_cfg(cfg)

# 6) sampler 실행
samples = guider.sample(noise, latent, sampler_obj, sigmas, denoise_mask=noise_mask, seed=seed)
```

실무적으로 중요한 재현 포인트:
- `noise_scaling`/`inverse_noise_scaling`을 model sampling 타입별로 맞춰야 결과가 맞는다.
- inpaint mask 처리(`KSamplerX0Inpaint`)를 빼면 ComfyUI inpaint 결과와 달라진다.
- cond의 `start_percent/end_percent` 변환을 빼면 ControlNet/영역 조건 결과가 어긋난다.

---

## 8) 커스터마이징 포인트 (코드 수정 기준)

### 새 sampler 추가
1. `comfy/k_diffusion/sampling.py`에 `sample_my_sampler(...)` 구현
2. `comfy/samplers.py`의 `KSAMPLER_NAMES`에 이름 추가
3. 필요 시 `comfy_extras/nodes_custom_sampler.py`에 전용 파라미터 노드 추가

### 새 scheduler 추가
1. scheduler 함수 구현
2. `SCHEDULER_HANDLERS`에 등록
3. 필요 시 custom scheduler 노드 추가

### CFG/denoise 동작 커스터마이징
- `model_options` hook 사용:
  - `sampler_pre_cfg_function`
  - `sampler_cfg_function`
  - `sampler_post_cfg_function`
  - `denoise_mask_function`

### 모델 타입 대응
- `comfy/model_sampling.py`의 타입(`EPS`, `V_PREDICTION`, `EDM`, `CONST`, Flow 계열)을 확인해야 한다.
- 동일 sampler라도 model sampling 타입에 따라 체감 동작이 달라질 수 있다.

---

## 9) 자주 헷갈리는 포인트

- `comfy.sample.sample(...)`의 `disable_noise` 인자는 함수 본문에서 직접 쓰이지 않는다.
- 실제 무노이즈 동작은 상위(`common_ksampler`)에서 zero noise를 만들어 전달하는 방식으로 구현된다.

- `KSampler`만 쓰면 `eta`, `s_noise`, `order`, `rtol` 같은 solver 하이퍼파라미터를 직접 제어하기 어렵다.
- 이런 튜닝이 필요하면 `SamplerCustom`/`SamplerCustomAdvanced` 그래프가 사실상 정답이다.

- `denoised_output`은 `latent_preview.prepare_callback(...)`이 저장한 `x0`를 사용해 구성된다.
- 최종 샘플(`output`)과 `denoised_output`은 의미가 다르다.

---

## 10) 결론

ComfyUI sampler 구조의 핵심은 다음 3가지다.

1. `KSampler`는 "간단 UI", 실제 엔진은 `comfy.samplers` + `k_diffusion`에 있다.
2. sampler 품질/성향은 `sampler_name`만이 아니라 `scheduler + sigma slicing(denoise/start/end)`의 결합으로 결정된다.
3. 자체 코드/고급 튜닝 목표라면 기본 KSampler보다 `custom_sampling` 노드 구조를 기준으로 설계하는 것이 구현 재현성과 확장성이 높다.

---

## 11) 수학적 관점: ComfyUI sampler를 방정식으로 읽기

이 섹션은 ComfyUI 구현을 수학적으로 해석한 요약이다.

### 11-1) 기본 확률미분방정식(SDE) 틀

score-based diffusion의 표준 표현은 다음 꼴이다.

- 순방향(노이즈 추가):
  - `d x = f(x,t) dt + g(t) dW_t`
- 역방향(샘플링):
  - `d x = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dW_bar_t`
  - 여기서 `score(x,t) = ∇_x log p_t(x)`

ComfyUI의 많은 sampler는 이 역방향 방정식의 수치해석기라고 보면 된다.

### 11-2) 확률흐름 ODE와 결정론 sampler

같은 주변분포를 갖는 확률흐름 ODE는 보통 다음 꼴로 쓴다.

- `d x / dt = f(x,t) - 0.5 * g(t)^2 * score(x,t)`

`euler`, `heun`, `lms`, `dpmpp_2m` 같은 결정론 계열은 본질적으로 이 ODE를 적분하는 해석으로 이해하면 된다.

### 11-3) ComfyUI의 sigma-파라미터화

ComfyUI는 시간 `t`보다 `sigma`를 중심으로 움직인다.

- `k_diffusion/sampling.py`의 `to_d(x, sigma, denoised)`:
  - `d = (x - denoised) / sigma`
  - 이 `d`가 ODE 업데이트의 벡터장 역할을 한다.

- 한 스텝은 일반적으로:
  - `x_{i+1} = x_i + d_i * Δsigma`
  - 고차 sampler는 중간점/과거 스텝 정보를 추가해 `d`를 보정한다.

즉 코드 관점에서 `denoised` 예측기 + `to_d` + sigma schedule이 미분방정식의 수치적분기를 구성한다.

### 11-4) Ancestral/SDE 샘플러의 노이즈 주입

`get_ancestral_step(sigma_from, sigma_to, eta)`는
- deterministic 하강량 `sigma_down`
- stochastic 주입량 `sigma_up`
을 분해한다.

업데이트를 단순화하면:
- `x <- x + drift_term + sigma_up * N(0, I)`

그래서 `eta=0`이면 ODE에 가까워지고, `eta>0`이면 reverse SDE 성격이 강해진다.

---

## 12) 수치해석 관점: sampler별 근사 성질

엄밀한 전역오차는 스케줄/모델/파라미터화에 따라 달라지지만, 실무 분류는 다음이 유용하다.

- 1차 단일스텝:
  - `euler`, `euler_ancestral`
  - 빠르지만 step 수가 낮을 때 bias가 크기 쉽다.

- 2차 단일스텝:
  - `heun`, `dpm_2`, `dpmpp_2s_ancestral`, `exp_heun_2_x0`
  - 같은 step에서 1차보다 drift 근사가 안정적인 편.

- 다단계(multistep):
  - `lms`, `dpmpp_2m`, `deis`, `ipndm`, `res_multistep`
  - 과거 기울기/모델 출력을 써서 효율을 높인다.
  - 초기 몇 step은 저차로 시작하고 이후 고차로 올라가는 워밍업 구간이 생긴다.

- 적응 스텝:
  - `dpm_adaptive`
  - 내부적으로 오차 제어(`rtol`, `atol`)로 step 크기를 조정한다.

SDE 계열(`*_ancestral`, `*_sde`, `er_sde`, `sa_solver`, `seeds_*`)은 위 오차 개념에 더해 분산 주입 항의 통계적 오차가 같이 작동한다.

---

## 13) 미분기하 관점: latent 공간에서의 해석

### 13-1) 벡터장 관점

latent 공간을 다양체(또는 국소적으로 유클리드 공간)로 보면 sampler는 시간의존 벡터장 `V_sigma(x)`의 적분곡선을 추적한다.

- `V_sigma(x)`에 해당하는 코드 객체가 사실상 `to_d(...)`의 출력이다.
- scheduler는 같은 궤적을 어떤 파라미터 속도로 따라갈지를 정한다(재매개화).

### 13-2) CFG의 기하학적 의미

`cfg_function`의 기본식:
- `u + s(c - u) = (1-s)u + s c`

여기서 `u`(uncond)와 `c`(cond)는 두 벡터장(혹은 두 예측점)이고, CFG는 그 affine 결합이다.

기하적으로 보면:
- `s=1`이면 조건/무조건의 중간 보정이 사라지고,
- `s>1`은 cond 방향으로 외삽(extrapolation)한다.

이 외삽이 커질수록 곡률이 큰 영역에서 수치적 불안정(과도한 saturation, artifact)이 증가할 수 있다.

### 13-3) Inpaint를 제약 다양체로 보기

`KSamplerX0Inpaint`의 mask 결합은 자유영역/고정영역으로 상태를 분할한 제약 동역학으로 볼 수 있다.

- 자유 영역: 벡터장 따라 업데이트
- 고정 영역: 원래 latent에 대한 투영(projection)

즉 각 step마다 `x <- Π_M( update(x) )` 형태의 투영-적분 스킴에 가깝다.

### 13-4) 스케줄러를 메트릭 선택으로 보는 해석

`karras`, `exponential`, `normal` 등은 동일한 시작/끝 sigma라도 step 밀도를 다르게 배치한다.

이는 기하적으로:
- 곡률/강성(stiffness)이 큰 구간에 샘플링 점을 더 촘촘히 둘지,
- 평탄한 구간에 점을 아낄지
를 정하는 시간-메트릭 선택 문제로 해석할 수 있다.

---

## 14) 수학 전공자용 실험 프레임

ComfyUI를 연구 실험 플랫폼처럼 쓰려면 다음 조합이 좋다.

1. 동일 seed/동일 sigma에서 solver만 교체
- `SamplerCustom` + `KSamplerSelect` 또는 전용 sampler 노드 사용
- 순수 적분기 차이(ODE solver bias) 분리 가능

2. 동일 solver에서 sigma schedule만 교체
- `BasicScheduler`, `KarrasScheduler`, `ExponentialScheduler`, `BetaSamplingScheduler`
- 재매개화 효과(속도장 샘플링 밀도) 분리 가능

3. SDE 강도 스윕
- `eta`, `s_noise`를 grid로 스윕
- 다양성-충실도 trade-off를 통계적으로 계량

4. CFG 기하 분석
- `cfg`를 연속적으로 증가시키며 norm 폭주 구간 체크
- 필요 시 `sampler_pre/post_cfg_function` hook으로 안정화 항 추가

---

## 15) 핵심 요약 (수학 관점)

1. ComfyUI sampler는 본질적으로 `sigma`-파라미터화된 ODE/SDE 적분기 모음이다.
2. `deterministic sampler + schedule`은 벡터장 적분 문제, `ancestral/SDE`는 여기에 확률항이 더해진 문제다.
3. CFG/inpaint/scheduler는 각각 벡터장 결합, 제약 투영, 시간 재매개화로 해석하면 코드 커스터마이징 방향이 명확해진다.

---

## 16) 코드 함수 파라미터와 수식 항의 직접 매핑

아래 표는 ComfyUI 코드 파라미터가 수식의 어느 항을 제어하는지 정리한 것이다.

### 16-1) 공통 핵심

| 코드 요소 | 수식 해석 | 의미 |
|---|---|---|
| `to_d(x, sigma, denoised)` | `d = (x - denoised) / sigma` | sigma-좌표계에서의 드리프트(벡터장) 근사 |
| `sigmas[i], sigmas[i+1]` | `Delta sigma` 또는 `h` | 적분 step 크기/시간 재매개화 |
| `noise_sampler(...)` | `z ~ N(0, I)` 또는 브라운 운동 증분 | 확률항 샘플 |
| `scheduler` | `sigma(t)` 경로 | 같은 시작/끝이라도 step 밀도(해상도)를 바꿈 |

### 16-2) 대표 sampler별 파라미터 매핑

| 함수(코드) | 주요 파라미터 | 수식 대응 | 해석 |
|---|---|---|---|
| `sample_euler` | `s_churn, s_tmin, s_tmax, s_noise` | `sigma_hat = (1+gamma)sigma`, `x <- x + eps * sqrt(sigma_hat^2 - sigma^2)` | 일시적 sigma inflation + 랜덤 perturbation |
| `sample_euler_ancestral` | `eta, s_noise` | `x <- x + d*(sigma_down-sigma) + s_noise*sigma_up*z` | `eta`가 `sigma_up`를 통해 확률항 세기를 조절 |
| `sample_dpm_2_ancestral` | `eta, s_noise` | 2차 drift + `sigma_up z` | 2차 정확도 + ancestral 재노이징 |
| `sample_dpmpp_sde` | `eta, s_noise, r` | 중간 스테이지 `lambda_s1 = lambda_s + r h` + SDE 노이즈 | `r`은 중간점 위치(2-stage 분할 비율) |
| `sample_dpmpp_2m_sde` | `eta, s_noise, solver_type` | 2M 보정식 + `sqrt(expm1(...)) z` | `solver_type`이 midpoint/heun 보정 형태를 선택 |
| `sample_dpm_adaptive` | `order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety` | local error test + PID step-size update | 적응형 스텝 제어기(오차 기반) |
| `sample_seeds_2` | `eta, s_noise, r, solver_type` | 지수적분(`phi_1/phi_2`) + 중간점 `r` | 확률형 exp-integrator 2단 |
| `sample_seeds_3` | `eta, s_noise, r_1, r_2` | 3-stage 지수적분 + 확률항 | 내부 stage 배치를 `r_1, r_2`가 결정 |
| `sample_sa_solver` | `tau_func, predictor_order, corrector_order, use_pece, s_noise` | PC/PECE 다단계 + `tau_t` 구간 확률항 | 확률적 Adams 계열, 구간별 stochastic 강도 제어 |
| `sample_er_sde` | `s_noise, noise_scaler, max_stage` | ER-SDE 다단계 drift + scaled noise | `noise_scaler`가 확률항 스케일 함수를 직접 바꿈 |

### 16-3) scheduler 파라미터 매핑

| 함수(코드) | 파라미터 | 수식/분포 해석 | 의미 |
|---|---|---|---|
| `get_sigmas_karras` | `rho` | power-law 곡률 | 고 sigma/저 sigma 구간의 점밀도 재배치 |
| `get_sigmas_exponential` | - | log-sigma 선형 | 지수형 감쇠 경로 |
| `beta_scheduler` | `alpha, beta` | Beta 분포 분위수 기반 timestep 샘플링 | 특정 구간 step 집중 |
| `linear_quadratic_schedule` | `threshold_noise, linear_steps` | 선형+이차 혼합 스케줄 | 초기/후반 감쇠 속도 분리 |
| `kl_optimal_scheduler` | `sigma_min, sigma_max` | atan/tan 재매개화 | KL 관점 휴리스틱 스케줄 |

### 16-4) 실전 해석 포인트

- `eta`와 `s_noise`는 대부분 `z` 항의 계수에 곱해져 확률항 분산을 키운다.
- `denoise`는 sigma 경로의 tail slice를 바꿔 사실상 적분 구간 길이를 줄인다.
- `cfg`는 드리프트를 `v_cfg = v_u + w(v_c - v_u)`로 바꾸는 제어 입력처럼 작동한다.

---

## 17) FPE(Fokker-Planck) 관점에서 본 ComfyUI sampler

### 17-1) 기본 FPE

SDE를
- `dX_t = b_t(X_t) dt + g_t dW_t`
로 쓰면 밀도 `rho_t`는
- `partial_t rho_t = - div(rho_t b_t) + 0.5 * g_t^2 * Delta rho_t`
를 따른다.

해석:
- `-div(rho b)`는 수송(transport) 항
- `0.5 g^2 Delta rho`는 확산(diffusion) 항

### 17-2) ComfyUI 코드와 FPE 항의 대응

- drift 항 `b_t` 대응:
  - `to_d(...)`로 계산되는 업데이트 방향
  - CFG 적용 시 `cfg_function`이 drift를 재정의

- diffusion 항 `g_t` 대응:
  - ancestral/SDE sampler에서 `noise_sampler(...)`가 생성하는 가우시안 증분
  - `eta`, `s_noise`가 유효 확산계수(분산) 크기를 조절

- schedule 대응:
  - `sigmas`는 `t`의 재매개화이므로, 같은 벡터장이라도 수치적으로는 다른 FPE 근사를 만든다.

### 17-3) probability-flow ODE와 FPE의 관계

확률흐름 ODE는
- `partial_t rho_t + div(rho_t v_t) = 0`
형태의 연속방정식을 만든다(확산항 없음).

즉 결정론 sampler는 FPE의 확산항을 제거한 수송 문제를 푸는 관점이고,
ancestral/SDE sampler는 확산항을 유지한 근사라고 볼 수 있다.

---

## 18) 최적수송(OT) / 엔트로피 정규화 관점

### 18-1) ODE sampler와 동적 OT

Benamou-Brenier 형태의 동적 OT는
- `min ∫_0^1 ∫ 0.5 * ||v_t(x)||^2 rho_t(x) dx dt`
- subject to `partial_t rho_t + div(rho_t v_t) = 0`

결정론 sampler를 이 프레임으로 보면:
- sampler는 `v_t`를 score 네트워크로 근사한 수송 경로 적분기
- scheduler는 시간축 가중(어느 구간에 계산량을 배치할지) 선택

### 18-2) SDE sampler와 entropic OT (Schrodinger bridge 해석)

노이즈 항이 있으면 문제는 순수 OT보다 엔트로피 정규화된 브리지 문제에 가깝다.

직관적으로:
- `eta`, `s_noise` 증가 -> 확산/엔트로피 가중 증가
- 결과 분포는 더 넓게 탐색(다양성 증가), 대신 프롬프트 충실도/선명도 trade-off가 발생

### 18-3) CFG와 제어(control) 관점

CFG는 drift에 제어 입력을 더하는 형태로 볼 수 있다.

- `v_cfg = v_u + w(v_c - v_u)`
- `w`가 커질수록 경로 에너지(`||v||`)가 증가해 고주파 artifact/과포화가 늘 수 있다.

OT/제어 관점에서는:
- 목표 분포로 가는 비용 최소화와 제어 강도 사이의 균형 문제다.

### 18-4) Inpaint의 OT 제약 해석

inpaint mask는 일부 좌표를 고정하는 하드 제약이다.

- 자유 영역만 수송
- 고정 영역은 투영(`projection`) 유지

즉 unconstrained OT가 아니라 제약된 상태공간에서의 수송 문제로 보는 편이 정확하다.

---

## 19) 수학 실험용 체크리스트 (파라미터-수식 연결 검증)

1. `eta` 스윕(0 -> 1 -> 2)과 출력 분산 측정
- 표본 간 LPIPS/CLIP score 분산으로 diffusion 항 영향 계량

2. `scheduler` 고정 vs 변경
- solver 고정 후 `karras/normal/beta`만 교체하여 재매개화 영향 분리

3. `cfg` 스윕과 drift norm 모니터링
- step별 `||d||` 또는 latent norm 추적으로 제어 에너지 폭주 구간 찾기

4. `dpm_adaptive`에서 `rtol/atol` 변화
- 허용오차와 실제 NFE(함수평가횟수), 품질의 Pareto 곡선 확인
