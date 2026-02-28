# Muon / GenericOptim — 직교화 기반 업데이트

> 분류: 실험적 최적화 계열 · diffusion-pipe ✓

## 핵심 아이디어

gradient 방향을 직교화(whitening/orthogonalization)해서
업데이트 조건수를 개선하려는 접근입니다.

## Newton-Schulz 반복 (대표식)

\[
X_{k+1}=\frac{15}{8}X_k-\frac{5}{4}X_k^3+\frac{3}{8}X_k^5
\]

이 반복은 특정 스케일 조건에서 직교화에 가까운 변환을 빠르게 근사합니다.

## Subspace Momentum (선택)

고차원에서 비용을 낮추기 위해 투영 공간에서 momentum을 계산하는 변형:

\[
G_{\text{proj}}=P_r(G),\qquad
\theta \leftarrow \theta-\eta\,P_r^{-1}(m_t)
\]

## 장단점

| 항목 | 장점 | 주의점 |
|---|---|---|
| 방향 정규화 | 특정 문제에서 빠른 수렴 | 구현/튜닝 난이도 높음 |
| 확장성 | 투영 기반으로 비용 제어 가능 | 하이퍼파라미터 민감 |
| 실험성 | 연구 가치 높음 | 안정성은 AdamW 계열보다 가변적 |

## 코드 매핑

```toml
# diffusion-pipe config 예시
[optimizer]
type = "muon"        # 구현명은 환경에 따라 다를 수 있음
lr = 1e-4
# rank, automagic 등은 구현 옵션에 따라 추가
```
