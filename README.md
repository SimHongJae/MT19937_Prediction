# MT19937 Prediction with Machine Learning

머신러닝을 이용한 메르센 트위스터(MT19937) 의사난수 생성기 예측 및 복원 프로젝트

## 프로젝트 개요

이 프로젝트는 Transformer 아키텍처를 사용하여 MT19937 난수 생성기의 출력을 예측합니다. 하이브리드 접근 방식을 사용하여:

1. **역템퍼링 네트워크**: 템퍼링된 출력에서 내부 상태 복원
2. **상태 전이 네트워크**: 624개의 내부 상태에서 다음 상태 예측

## 주요 특징

- ✅ MT19937 완전 구현 (내부 상태 접근 가능)
- ✅ 비트 레벨 표현 (32비트 → 32차원 벡터)
- ✅ Transformer 기반 상태 전이 학습
- ✅ Binary Cross-Entropy 손실 함수
- ✅ GTX 1060 3GB GPU 최적화

## 파일 구조

```
MT19937_Prediction/
├── mt19937.py                     # MT19937 구현
├── generate_dataset.py            # 데이터셋 생성
├── model.py                       # 신경망 모델 정의
├── train_inverse_tempering.py    # 역템퍼링 학습
├── train_transition.py            # 상태 전이 학습
├── test.py                        # 평가 스크립트
└── README.md                      # 이 파일
```

## 설치

필요한 패키지:
```bash
pip install torch numpy tqdm
```

## 사용 방법

### 1. 데이터셋 생성

```bash
python generate_dataset.py
```

생성되는 데이터:
- 템퍼링 데이터셋: 100,000 샘플
- 상태 전이 데이터셋: 500,000 샘플
- 각각 train/val/test로 80/10/10 분할

### 2. 역템퍼링 네트워크 학습

```bash
python train_inverse_tempering.py
```

목표: >95% 비트 정확도 (쉬운 작업)
예상 시간: 10-30분

### 3. 상태 전이 네트워크 학습

```bash
python train_transition.py
```

목표: >80% 비트 정확도 (주요 작업)
예상 시간: 2-4시간 (GTX 1060 3GB)

### 4. 평가

```bash
python test.py
```

테스트 세트에서 두 모델 평가 및 상세 결과 출력

## 모델 아키텍처

### 역템퍼링 네트워크
```
Input: (batch, 32) bits
  ↓
MLP: Linear(32, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 32)
  ↓
Output: (batch, 32) bits (logits)
```
파라미터: ~8,352개

### 상태 전이 네트워크
```
Input: (batch, 624, 32) bit sequences
  ↓
Embedding: Linear(32, 128)
  ↓
Positional Encoding
  ↓
Transformer Encoder (4 layers, 4 heads)
  ↓
Output Head: Linear(128, 32)
  ↓
Output: (batch, 32) bits (logits)
```
파라미터: ~801,440개

## 하이퍼파라미터

### 역템퍼링
- Hidden dimension: 64
- Batch size: 256
- Learning rate: 1e-3
- Epochs: 20

### 상태 전이
- d_model: 128
- nhead: 4
- num_layers: 4
- dim_feedforward: 512
- Batch size: 16 (GPU 메모리 고려)
- Learning rate: 5e-5
- Epochs: 50

## 예상 성능

연구 논문 기반 예상 결과:

| 모델 | 비트 정확도 | 정확히 일치 |
|------|------------|------------|
| 역템퍼링 | >95% | >80% |
| 상태 전이 | 70-85% | 1-10% |

**목표**: 상태 전이 모델에서 80% 비트 정확도 달성

## 핵심 설계 원칙

1. **비트 레벨 표현**: 32비트 정수를 32개의 0/1 비트로 분해
2. **BCE 손실**: Binary Cross-Entropy (각 비트별 분류)
3. **Transformer**: 624 스텝 장기 의존성 학습
4. **하이브리드**: 역템퍼링 + 상태 전이 분리 학습

## 데이터셋 정보

### 템퍼링 데이터셋
- **입력**: 템퍼링 후 32비트
- **출력**: 템퍼링 전 내부 상태 32비트
- **샘플 수**: 100,000

### 상태 전이 데이터셋
- **입력**: 624개 내부 상태 시퀀스 (624, 32)
- **출력**: 다음 내부 상태 (32)
- **샘플 수**: 500,000

## 참고 자료

- MT19937 알고리즘: [Wikipedia](https://en.wikipedia.org/wiki/Mersenne_Twister)
- 연구 보고서: 머신러닝을 이용한 메르센 트위스터 예측 및 복원

## 라이센스

교육 및 연구 목적

## 작성자

3학년 통섭연구 프로젝트
박채연 교수님 지도
