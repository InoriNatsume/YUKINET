

# Scheduler: beta

Scheduler는 solver 식 자체보다 sigma grid(시간 재매개화)를 바꾸어 오차 분포와 경로 성향을 조정한다.
$$ t_i=F^{-1}_{\mathrm{Beta}(\alpha,\beta)}(u_i) $$
구간 집중

## 기호 계약(정의역/공역/조건)

| 항목 | 수식 | 설명 |
|---|---|---|
| 스케줄 맵 | $S:\{0,\dots,N\}\to\Sigma\subset(0,\infty)$ | step index를 sigma로 매핑. |
| mesh 변수 | $h_k=\|\lambda_{k+1}-\lambda_k\|,\ \lambda=\log\alpha-\log\sigma$ | 오차식에 직접 들어가는 유효 step. |
| 오차 스케일 | $\\|e_{global}\\|\approx C\max_k h_k^p$ | 동일 solver에서 scheduler 품질 차이의 핵심. |
| 분포 파라미터 | $\alpha>0,\ \beta>0$ | 구간 집중도 조절. |

## 메쉬(시간 재매개화) 수학
$$ t_k=F^{-1}_{\mathrm{Beta}(\alpha,\beta)}(u_k),\ \sigma_k=\sigma(t_k) $$$$h_k:=|\lambda_{k+1}-\lambda_k|,\quad \lambda=\log\alpha-\log\sigma,\quad \|e_{\mathrm{global}}\|\approx C\max_k h_k^p$$
Beta CDF 역함수로 특정 구간에 스텝을 집중한다.

질감/구조 우선순위에 맞춰 노이즈 구간 가중을 설계할 때 유용하다.

## Sampler와의 결합 해석

같은 sampler라도 scheduler가 바뀌면 고 sigma 구간과 저 sigma 구간의 스텝 밀도가 달라지고, 결과적으로 세부 질감/구조 보존/수렴 속도 균형이 달라진다.
$$x_{k+1}=A_kx_k+B_k\hat{x}_{0,k}+C_k(\text{history})+D_k\xi_k,\quad A_k,B_k,C_k,D_k \text{는 스케줄 메쉬에 의해 간접적으로 변한다}$$
