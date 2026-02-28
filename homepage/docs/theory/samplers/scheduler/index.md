# 스케줄러 카탈로그

Scheduler는 **sigma grid(시간 재매개화)** 를 결정하여 동일한 sampler라도 수렴 속도와 품질을 다르게 만듭니다.

## 0) 공통 기호

| 기호 | 타입 | 의미 |
|---|---|---|
| $S$ | $S:\{0,\dots,N\}\to(0,\infty)$ | scheduler 사상 |
| $\sigma_k$ | $\sigma_k=S(k)$ | k번째 sigma |
| $\lambda_k$ | $\lambda_k=\log\alpha_k-\log\sigma_k$ | 오차 분석용 좌표 |
| $h_k$ | $h_k=|\lambda_{k+1}-\lambda_k|$ | 유효 step 크기 |

\[
\|e_{\text{global}}\| \approx C \max_k h_k^p
\]

즉 scheduler는 solver의 식을 바꾸지 않고 $h_k$ 분포를 바꿉니다.

## 1) 구체 예시 (원소 나열)

\[
K=\{0,1,2,3\},\quad S(K)=\{\sigma_0,\sigma_1,\sigma_2,\sigma_3\}
\]

같은 $K$라도 $S$를 다르게 두면
$(\sigma_0,\sigma_1,\sigma_2,\sigma_3)$가 바뀌고,
결과적으로 구조/디테일/안정성의 체감이 달라집니다.

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
