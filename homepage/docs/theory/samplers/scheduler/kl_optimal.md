

# Scheduler: kl_optimal

Scheduler는 solver 식 자체보다 sigma grid(시간 재매개화)를 바꾸어 오차 분포와 경로 성향을 조정한다.
$$ \sigma=\tan(a\arctan\sigma_{\min}+(1-a)\arctan\sigma_{\max}) $$
KL 휴리스틱

## 기호 계약(정의역/공역/조건)

| 항목 | 수식 | 설명 |
|---|---|---|
| 스케줄 맵 | $S:\{0,\dots,N\}\to\Sigma\subset(0,\infty)$ | step index를 sigma로 매핑. |
| mesh 변수 | $h_k=\|\lambda_{k+1}-\lambda_k\|,\ \lambda=\log\alpha-\log\sigma$ | 오차식에 직접 들어가는 유효 step. |
| 오차 스케일 | $\\|e_{global}\\|\approx C\max_k h_k^p$ | 동일 solver에서 scheduler 품질 차이의 핵심. |
| 일반 조건 | $\sigma_k>0,\ \sigma_{k+1}\le\sigma_k$ | 역적분용 단조성. |

## 메쉬(시간 재매개화) 수학
$$ \sigma(u)=\tan\!\left(u\arctan\sigma_{\min}+(1-u)\arctan\sigma_{\max}\right) $$$$h_k:=|\lambda_{k+1}-\lambda_k|,\quad \lambda=\log\alpha-\log\sigma,\quad \|e_{\mathrm{global}}\|\approx C\max_k h_k^p$$
KL 기반 휴리스틱으로 중간 노이즈 구간을 상대적으로 강조한다.

score mismatch가 큰 구간의 누적 오차를 완화하려는 목적의 스케줄.

## Sampler와의 결합 해석

같은 sampler라도 scheduler가 바뀌면 고 sigma 구간과 저 sigma 구간의 스텝 밀도가 달라지고, 결과적으로 세부 질감/구조 보존/수렴 속도 균형이 달라진다.
$$x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k,\quad A_k,B_k,C_k,D_k \text{는 스케줄 메쉬에 의해 간접적으로 변한다}$$
