# 샘플러 카탈로그

ComfyUI `k_diffusion/sampling.py` 에 구현된 **44개 샘플러** 전체 분석 모음.

각 페이지는 동일한 구조를 따릅니다:

- **수식**: 업데이트 규칙 (ODE/SDE 관점)
- **파라미터-기호 대응**: 코드 파라미터 ↔ 수학 표기
- **수학 심화 프로파일**: 수치 오차 차수, 안정성 특성
- **구현 스니펫**: ComfyUI 원본 코드

## 계열(Family) 분류

| Family | 포함 샘플러 | 특징 |
|---|---|---|
| **Euler / EM** | euler, euler\_ancestral, euler\_cfg\_pp 등 | 1차 ODE, 가장 단순 |
| **Heun / ExpHeun** | heun, heunpp2 | 2차 predictor-corrector |
| **DPM-Solver** | dpm\_2, dpm\_fast, dpm\_adaptive | 고차 정확도 |
| **DPM-Solver++** | dpmpp\_2m, dpmpp\_2s\_, dpmpp\_3m 등 | 다단계, SDE 변형 포함 |
| **DDIM / DDPM** | ddim, ddpm | 조기 클래식 |
| **LMS / IPNDM** | lms, ipndm, ipndm\_v | 선형 다단계 |
| **SA-Solver** | sa\_solver, sa\_solver\_pece | 예측-수정 적응형 |
| **기타** | lcm, er\_sde, seeds\_2, seeds\_3, deis 등 | 특화 구조 |
