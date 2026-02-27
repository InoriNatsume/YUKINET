# 스케줄러 카탈로그

Scheduler는 **sigma grid(시간 재매개화)** 를 결정하여 동일한 sampler라도 수렴 속도와 품질을 다르게 만듭니다.

$$
\|e_{\text{global}}\| \approx C \max_k h_k^p, \quad h_k := |\lambda_{k+1} - \lambda_k|
$$

## 스케줄러 목록

| Scheduler | 수식 특징 | 특징 |
|---|---|---|
| **karras** | power-law $\rho$ 제어 | 저노이즈 구간 밀도 증가 |
| **exponential** | log 공간 균등 분할 | step ratio 안정 |
| **beta** | Beta 분포 샘플링 | 유연한 밀도 조정 |
| **normal** | 정규 분포 기반 | 중간 timestep 집중 |
| **kl_optimal** | KL Optimal | 이론 최적 분할 |
| **linear_quadratic** | 선형+2차 혼합 | 구간별 비율 조정 |
| **sgm_uniform** | SGM 균등 | 안정적 기준선 |
| **ddim_uniform** | DDIM 균등 | 고전적 균등 분할 |
| **simple** | 단순 균등 | 기본값 |
