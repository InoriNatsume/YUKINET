# 확산 수학

확산 수학은 **훈련과 추론이 공유하는 공통 기반**입니다.

## 적용 범위

| 단계 | 확산 수학이 하는 일 |
|---|---|
| 훈련 | `x_t` 생성 규칙, target(`epsilon/v/velocity`)의 정의 근거 제공 |
| 추론 | 역시간 ODE/SDE 적분의 벡터장 의미 제공 |

## 핵심 객체

| 기호 | 타입 | 의미 |
|---|---|---|
| $x_0$ | $x_0 \in \mathcal{X}$ | 데이터 상태 |
| $x_t$ | $x_t \in \mathcal{X}$ | 시간 $t$의 상태 |
| $p_t$ | 분포 | 시간 $t$의 상태 분포 |
| $v_t$ / score | 벡터장 | 분포 이동 방향 |

- [정방향 확산](forward.md) — DDPM 마르코프 체인, VP-SDE, Flow Matching 보간
- [Flow Matching](flow-matching.md) — OT 경로, velocity matching, timestep 분포 전략
