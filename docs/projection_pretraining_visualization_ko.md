# Projection Pretraining 시각화

## 개요

Projection pretraining 중에 자동으로 생성되는 시각화를 통해 학습이 제대로 되고 있는지 확인할 수 있습니다.

## 생성되는 파일

각 에폭이 끝날 때마다 `output/projection_pretrain/visualizations/epoch_XX/` 폴더에 4개의 샘플에 대한 시각화가 저장됩니다.

### 샘플별 파일

각 샘플(`sample_00`, `sample_01`, ...) 폴더에는:

#### 1. `merged_comparison.png`

**Input Projection 평가**

```
[GT Merged Image]        [Predicted Merged Image]
(Alpha Blending)         (Input Projection Output)
```

- **왼쪽**: Ground truth merged image (6개 레이어를 alpha blending으로 합친 이미지)
- **오른쪽**: Input projection이 예측한 merged image

**기대 결과**:
- 두 이미지가 거의 동일해야 함
- Input projection이 6개 레이어를 받아서 올바르게 합쳐진 이미지를 출력하는지 확인
- 이게 바로 **PixArt가 학습한 실제 이미지 latent space**입니다!

#### 2. `layers_comparison.png`

**Output Projection 평가**

```
Row 1: [GT Layer 0] [GT Layer 1] [GT Layer 2] ... [GT Layer 5]
Row 2: [Recon 0]    [Recon 1]    [Recon 2]    ... [Recon 5]
       [MASKED]     [visible]    [visible]        [visible]
```

- **위쪽 행**: Ground truth 6개 레이어 (원본)
- **아래쪽 행**: Output projection이 재구성한 6개 레이어

**기대 결과**:
- Visible layers (마스크 안 된 레이어): GT와 거의 동일해야 함
- Masked layer: 재구성 품질이 학습 진행에 따라 향상되어야 함
- Output projection이 merged image를 받아서 6개 레이어로 제대로 분해하는지 확인

#### 3. `stats.txt`

**손실 통계**

```
Sample 0 - image_id_12345
============================================================

Merge Loss (MSE): 0.002341

Decompose Loss per Layer:
  Layer 0 [MASKED]: 0.005234
  Layer 1 [visible]: 0.001234
  Layer 2 [visible]: 0.001456
  Layer 3 [visible]: 0.001123
  Layer 4 [visible]: 0.001345
```

- **Merge Loss**: Input projection의 merged image 예측 오차
- **Decompose Loss**: Output projection의 각 레이어별 재구성 오차

**기대 결과**:
- 학습이 진행됨에 따라 모든 loss가 감소해야 함
- Visible layers의 loss가 특히 낮아야 함 (입력 그대로 보존해야 하므로)

## 학습 진행 체크리스트

### ✓ 잘 학습되고 있는 경우

1. **Merged comparison**:
   - GT와 예측이 시각적으로 거의 구분 안 됨
   - 색상, 구조, 디테일이 일치

2. **Layers comparison**:
   - Visible layers가 GT와 거의 동일
   - Masked layer도 점점 GT와 비슷해짐
   - 패딩 레이어는 검은색 (정상)

3. **Stats**:
   - Merge loss < 0.01 (잘 수렴)
   - Visible layers의 decompose loss < 0.005
   - Epoch가 증가할수록 loss 감소

### ✗ 문제가 있는 경우

1. **Merged comparison**:
   - 예측이 흐릿하거나 아티팩트 많음
   - 색상이 이상함
   - 구조가 무너짐
   → Input projection이 제대로 학습 안 됨

2. **Layers comparison**:
   - Visible layers가 GT와 크게 다름
   - 모든 레이어가 비슷하게 보임 (평균화됨)
   - 이상한 패턴 생성
   → Output projection이 제대로 분해 못 함

3. **Stats**:
   - Loss가 감소하지 않음
   - Loss가 발산
   - NaN 발생
   → 학습률, weight 조정 필요

## 시각화 빈도

- **기본**: 매 에폭마다 4개 샘플 시각화
- **샘플 수 변경**: `train_scripts/pretrain_projections.py`의 `num_samples` 파라미터 조정

```python
vis_dir = visualize_predictions(
    model_unwrapped,
    train_dataloader,
    vae,
    epoch=epoch+1,
    output_dir=args.output_dir,
    num_samples=8  # 여기 변경
)
```

## 의미

### Input Projection: Merge (합성)

6개 레이어 → 합쳐진 이미지

- **학습 목표**: Alpha blending처럼 레이어들을 자연스럽게 합치기
- **출력 공간**: PixArt가 이미 학습한 실제 이미지의 latent space
- **이점**: Diffusion 모델이 이미 아는 공간에서 동작

### Output Projection: Decompose (분해)

합쳐진 이미지 → 6개 레이어

- **학습 목표**: 이미지를 개별 레이어로 분해
- **의미**: 역과정, compositing의 반대
- **이점**: 생성된 이미지를 다시 레이어 구조로 변환

## 전체 파이프라인 확인

```
Input: 6 layers (GT)
  ↓
[Input Projection] → Merged latent (should match GT merged image)
  ↓
[Diffusion Model] → (이후 main training에서 사용)
  ↓
[Output Projection] → 6 layers (should match GT layers)
  ↓
Output: 6 reconstructed layers
```

Pretraining에서는 이 전체 파이프라인이 **재구성 문제**로 학습됩니다:
- Clean input → Merge → Decompose → Should reconstruct input
- Merge/Decompose가 서로 역함수 관계가 되도록 학습

Main training에서는:
- Visible layers (clean) → Merge
- Diffusion adds masked layer
- Decompose → All 6 layers with generated masked layer
