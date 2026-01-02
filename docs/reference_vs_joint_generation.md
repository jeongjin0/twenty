# Layer-wise Inpainting: Reference Generation vs Joint Generation

## Overview

두 가지 접근법 모두 **pretrained diffusion model (PixArt)**을 활용하여 layer-wise inpainting을 수행합니다.

---

## 1. Reference Generation (Sequential Approach)

### 구조
```
Input: Text prompt for layer i
       Previously generated layers (0 to i-1) as reference

Process:
  Layer 0: Generate from text only
  Layer 1: Generate conditioned on text + layer 0
  Layer 2: Generate conditioned on text + layers 0,1
  ...
  Layer N: Generate conditioned on text + layers 0,1,...,N-1
```

### 동작 방식
- **순차적 생성**: 한 번에 하나의 layer만 생성
- **Reference conditioning**: 이전 layers를 ControlNet 방식으로 conditioning
- **Pretrained model 활용**: PixArt를 freeze하고 ControlNet adapter만 학습

### 문제점

#### 1. **Error Accumulation (오류 누적)**
```
Layer 0: 생성 (약간의 오류)
  ↓
Layer 1: Layer 0를 reference로 사용 (Layer 0의 오류 포함)
  ↓
Layer 2: Layers 0,1을 reference로 사용 (누적된 오류)
  ↓
...
  ↓
Layer N: 모든 이전 layer의 오류가 누적됨
```
- 초기 layer의 작은 실수가 후반 layer로 전파
- 마지막 layer는 품질이 매우 낮아질 수 있음

#### 2. **Context Limitation**
```
Layer i를 생성할 때:
  - 이전 layers (0 to i-1)만 볼 수 있음
  - 이후 layers (i+1 to N)는 볼 수 없음

문제:
  - 전체 composition을 고려하지 못함
  - 후반 layer가 전반 layer와 조화롭지 않을 수 있음
```

#### 3. **Inference Inefficiency**
```
N개 layers 생성 = N번의 diffusion process
각 diffusion: 50 steps (DDIM)
총 inference: 50 × N steps

예: 6 layers → 300 steps (매우 느림)
```

#### 4. **Training-Inference Mismatch**
```
Training: Teacher forcing
  - Ground truth layers를 reference로 사용
  - 완벽한 reference 제공

Inference: Auto-regressive
  - 생성된 layers를 reference로 사용
  - 불완전한 reference 제공

→ Distribution shift 발생
```

---

## 2. Joint Generation (Channel Concatenation - 현재 방식)

### 구조
```
Input: All layers concatenated (30 channels)
       - Visible layers: clean VAE latents
       - Masked layer: noisy VAE latent
       - Layer masks: which layer is masked (6 channels)

Architecture:
  Input Projection: 30ch → 4ch
       ↓
  Pretrained PixArt: 4ch diffusion
       ↓
  Output Projection: 4ch → 24ch (6 layers × 4 channels)
```

### 동작 방식
- **병렬 처리**: 모든 layers를 한 번에 처리
- **Conditional inpainting**: Visible layers를 conditioning으로 사용
- **Single diffusion pass**: 하나의 denoising process로 masked layer 생성

### 왜 이 방식으로 변경했는가?

#### 해결 1: **No Error Accumulation**
```
모든 layers를 동시에 처리:
  - Visible layers는 항상 ground truth (clean)
  - Masked layer만 denoising
  - 이전 layer의 오류가 전파되지 않음
```

#### 해결 2: **Full Context**
```
Masked layer를 생성할 때:
  - 모든 visible layers를 볼 수 있음
  - 전체 composition 고려 가능
  - Layer 간 일관성 보장
```

#### 해결 3: **Efficient Inference**
```
N개 layers 중 1개 masked → 1번의 diffusion process
각 diffusion: 50 steps (DDIM)
총 inference: 50 steps

예: 6 layers → 50 steps (6배 빠름!)
```

#### 해결 4: **No Training-Inference Mismatch**
```
Training & Inference 동일:
  - 항상 ground truth visible layers 사용
  - Distribution shift 없음
  - 안정적인 성능
```

### 왜 이 방식이 작동하는가?

#### 1. **Pretrained Knowledge 활용**
```
PixArt는 이미 natural image diffusion을 학습함:
  - Input projection: layers → single latent representation
  - PixArt: latent diffusion (pretrained knowledge 사용)
  - Output projection: single latent → per-layer predictions

Pretrained PixArt가 이미 "이미지를 denoising하는 방법"을 알고 있음
→ Projection만 학습하면 됨
```

#### 2. **Inpainting과 유사**
```
Traditional inpainting:
  - Visible region: clean pixels
  - Masked region: noisy pixels
  - Task: denoise masked region given visible region

Layer-wise inpainting:
  - Visible layers: clean latents
  - Masked layer: noisy latent
  - Task: denoise masked layer given visible layers

구조적으로 동일 → Pretrained model이 자연스럽게 적용 가능
```

#### 3. **Efficient Learning**
```
Stage 1 (Projection 학습):
  - PixArt frozen (pretrained knowledge 보존)
  - Input/Output projection만 학습
  - "Layer를 합치는 방법" 학습

Stage 2 (Fine-tuning):
  - PixArt unfrozen
  - Layer-specific patterns 학습
  - 전체 모델 최적화
```

---

## 비교 요약

| 특징 | Reference Generation | Joint Generation |
|------|---------------------|------------------|
| **생성 방식** | 순차적 (sequential) | 병렬 (parallel) |
| **Inference 속도** | 느림 (N×50 steps) | 빠름 (50 steps) |
| **Error propagation** | 있음 (누적) | 없음 |
| **Context** | 이전 layers만 | 모든 visible layers |
| **Training-Inference gap** | 있음 (distribution shift) | 없음 |
| **Pretrained 활용** | ControlNet adapter | Channel concatenation |
| **Architecture** | PixArt + ControlNet | Input/Output projections |
| **학습 난이도** | 쉬움 (adapter만) | 중간 (projections 학습) |

---

## 결론

**Joint Generation이 더 나은 이유:**

1. ✅ **안정성**: Error accumulation 없음
2. ✅ **품질**: Full context로 일관성 보장
3. ✅ **속도**: N배 빠른 inference
4. ✅ **일관성**: Training-inference mismatch 없음

**Trade-off:**
- Projection architecture 설계가 중요 (현재 개선 중)
- Initial training이 조금 더 어려움 (2-stage training으로 해결)

하지만 장점이 단점을 압도하므로 **Joint Generation을 선택**했습니다.
