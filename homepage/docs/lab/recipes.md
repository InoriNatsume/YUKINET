# 레시피

검증된 훈련/추론 세팅 모음. 실험을 거쳐 **재현 가능**하게 정리합니다.

!!! tip "레시피 vs 실험"
    **실험**: 시행착오 과정 전체 기록 (실패 포함)  
    **레시피**: 실험에서 확인된 **좋은 세팅**만 요약

## 훈련 레시피

*추가 예정*

### 템플릿

```
## [모델] + [도구] + [방법]

**대상**: 캐릭터 LoRA / 스타일 LoRA / Full FT 등
**모델**: SDXL / Flux / ...
**도구**: kohya / diffusion-pipe / ...
**하드웨어**: RTX 3050 8GB / T4 16GB / ...

### 세팅

​```bash
# 전체 CLI 명령어
accelerate launch train_network.py \
  --network_module=networks.lora \
  --network_dim=16 \
  --network_alpha=8 \
  --optimizer_type=Prodigy \
  --learning_rate=1.0 \
  ...
​```

### 데이터셋
- 이미지 수: N장
- 해상도: 1024x1024
- 반복: 10
- 캡션: 방식

### 결과
- 훈련 시간: ~X분
- VRAM 사용: ~XGB
- 품질 평가: (샘플 이미지)

### 이론 연결
- 옵티마이저 선택 이유: [→ 이론/옵티마이저/Prodigy](../theory/optimizers/prodigy.md)
- LoRA rank 선택 이유: [→ 이론/훈련이론/LoRA](../theory/training/lora.md)
```

## 추론 레시피

*추가 예정*
