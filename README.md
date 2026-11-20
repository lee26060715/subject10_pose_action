# subject10_pose_action
# Pose-based Human Action Recognition

> YOLO Pose 기반 관절 데이터 추출부터 데이터 품질 검증, Subject-wise 분할, 동작 분류 및 실시간 추론까지 구현한 학습 프로젝트

**Repository:** https://github.com/lee26060715/subject10_pose_action

## 1. 프로젝트 개요

영상 속 사람의 자세를 **COCO 17개 Keypoint**로 추출하고, 관절 좌표 및 각도를 활용하여 **6가지 동작(Action)**을 분류하는 프로젝트입니다. 정적 이미지와 동영상 분석에서 시작하여, 데이터셋 구축·검증, Random Forest 기반 분류, 저장 영상 및 IP Webcam 실시간 추론까지 단계별 파이프라인을 구성했습니다.

이 프로젝트는 **모델 성능뿐 아니라 데이터 품질(QA), 데이터 누수 방지, 재현 가능한 평가, 실시간 예측 안정화**에 초점을 맞춥니다.

## 2. 기술 스택

| 구분 | 기술 |
| --- | --- |
| 언어 | Python |
| 영상 처리 | OpenCV |
| Pose 추정 | Ultralytics YOLO Pose (COCO 17 Keypoints) |
| 데이터 처리 | NumPy, Pandas |
| 동작 분류 | scikit-learn RandomForestClassifier |
| 평가 및 시각화 | scikit-learn metrics, Matplotlib |
| 모델 저장 | joblib, JSON |
| 프로젝트 환경 | `pyproject.toml`, `uv.lock` |

## 3. 인식 대상 동작

| Action ID | Label | 동작 설명 |
| --- | --- | --- |
| A001 | `bend_return` | 몸을 굽혔다가 돌아오기 |
| A002 | `leg_raise_lower` | 다리 들었다 내리기 |
| A003 | `walk_turn_walk` | 걷기·회전·다시 걷기 |
| A004 | `drink_return` | 음료를 마시는 동작 후 돌아오기 |
| A005 | `sit_stand` | 앉았다 일어나기 |
| A006 | `wave` | 손 흔들기 |

## 4. 전체 파이프라인

```text
이미지 / MP4 / IP Webcam
          │
          ▼
     YOLO Pose 추정
     (COCO 17 Keypoints)
          │
          ▼
  관절 좌표 정규화 · 관절각도 계산
          │
          ▼
  Frame Feature (34 좌표 + 4 각도 = 38-D)
          │
          ├── 데이터셋 구축 및 QA
          │    ├── 영상/파일명 검증
          │    ├── Keypoint CSV 생성
          │    ├── 자동 QA + Human Review
          │    └── 승인 Clip Manifest
          │
          ▼
   Subject-wise Train / Val / Test
          │
          ▼
   시간축 Resampling (30 Steps)
          │
          ▼
   Sequence Feature (30 × 38 = 1,140-D)
          │
          ▼
    Random Forest 학습
          │
          ├── Validation / Test 평가
          ├── 오분류 및 Feature 구성 비교
          │
          ▼
    저장 영상 / IP Webcam 실시간 추론
          │
          ▼
 Sliding Window → 예측 → Confidence 필터
          │
          ▼
    Prediction Stabilization · CSV 로그 분석
```

## 5. 데이터 구성 및 누수 방지

기본 실습 데이터는 **3명의 Subject(S01~S03) × 6개 동작 × 1 Take = 18개 Clip**으로 구성합니다.

| Split | Subject | Clip 수 | 용도 |
| --- | --- | ---: | --- |
| Train | S01 | 6 | 모델 학습 |
| Validation | S02 | 6 | 모델 및 특징 구성 검토 |
| Test | S03 | 6 | 최종 성능 평가 |

동일 인물의 프레임이 학습·평가 세트에 섞이지 않도록 **Subject-wise Split**을 적용합니다. 단순한 Frame 단위 무작위 분할의 데이터 누수 위험도 별도 스크립트로 확인합니다.

> **데이터 규모 관련 주의:** 이 구성은 파이프라인 검증을 위한 소규모 실습 데이터입니다. Subject별 샘플 수가 적으므로 측정된 정확도를 일반 환경에서의 성능으로 해석해서는 안 됩니다.

## 6. 주요 구현 내용

### 6.1 Pose 추출 및 Feature 생성

- 이미지·영상·IP Webcam에서 Pose Keypoint 추출
- Keypoint Confidence 및 검출 품질 확인
- 어깨 너비를 기준으로 좌표 정규화
- 17개 관절의 x/y 좌표 **34개**와 관절각도 **4개**를 결합한 **38차원 Frame Feature** 생성
- 정적 특징과 시간 흐름을 반영한 Sequence 특징 비교

### 6.2 데이터 품질 관리(QA)

- 입력 영상 수, 파일명 규칙 및 Action 정보 확인
- Keypoint CSV 생성 및 Clip별 품질 검사
- 자동 QA 결과를 바탕으로 사람이 `PASS`, `RESHOOT`, `EXCLUDE` 판정
- 승인된 Clip만 Manifest에 포함
- Split별 Action 누락 및 Subject 중복 여부 검사

### 6.3 모델 학습 및 평가

- Frame Feature를 시간축으로 **30 Step**에 맞춰 Resampling
- Clip당 **1,140차원 Sequence Feature** 생성
- Random Forest Baseline 학습 및 모델·Feature 순서 저장
- Accuracy, Macro F1, Classification Report, Confusion Matrix로 평가
- 오분류 Clip 목록과 원본 영상 경로 연결
- **Static vs. Sequence**, **좌표+각도 vs. 좌표만** 비교 실험

### 6.4 실시간 동작 인식

- 저장 MP4 또는 IP Webcam 스트림 입력
- 일정 시간의 Frame Feature를 Sliding Window에 누적
- Window 데이터를 학습 시 사용한 Sequence 형식으로 변환하여 예측
- Confidence 임계값 미달 시 `UNKNOWN` 처리
- 최근 예측의 투표를 이용해 Label 표시 안정화
- 처리 FPS, 예측 Label, Confidence 등을 CSV로 기록하고 후속 분석

## 7. 주요 스크립트 안내

| 단계 | 스크립트 | 역할 |
| --- | --- | --- |
| 기본 입력·Pose | `00`~`11` | 실행 환경, 이미지·영상 입력, Pose 시각화 및 품질 점검 |
| Feature | `12`~`19` | 관절각도, 정규화, 좌우 반전, Frame Feature 및 규칙 기반 분석 |
| Action Sequence | `20`~`24` | Action 목록 확인, Keyframe 추출, Sequence 생성 및 요약 |
| Dataset & QA | `25`~`29` | 18개 영상 확인, Keypoint Dataset 추출, 자동·수동 QA, 승인 Manifest |
| Split 검증 | `30`~`32` | Subject-wise 분리, 누수 검사, Frame Split 위험 시연 |
| Baseline | `33`~`37` | 입력 검사, Sequence Dataset, Random Forest 학습, Validation 평가, 오분류 분석 |
| 비교 실험 | `38`~`39` | Static/Sequence 및 Feature 구성 비교 |
| Test | `40_evaluate_test.py` | 학습에 사용하지 않은 S03 데이터로 최종 평가 |
| 실시간 준비 | `41_check_day07_inputs.py` | 모델·설정·영상 입력 호환성 검사 |
| 저장 영상 추론 | `42_realtime_action_video.py` | MP4 기반 동작 추론 및 결과 기록 |
| 실행 로그 요약 | `43_summarize_realtime_runs.py` | 저장 영상 실험의 예측 변화 및 UNKNOWN 비율 요약 |
| IP Webcam 추론 | `44_ip_webcam_action_realtime.py` | IP 스트림 기반 동작 인식 및 런타임 로그 저장 |
| IP Webcam 분석 | `45_analyze_ip_webcam_log.py` | 실시간 로그의 Action별 성능·안정성 분석 |

스크립트 전체는 [`scripts/`](./scripts)에서 확인할 수 있습니다.

## 8. 프로젝트 구조

```text
subject10_pose_action/
├── configs/             # 프로젝트, Action, Split, Baseline, 실시간 설정
├── scripts/             # 00~45 단계별 실행 스크립트
├── src/                 # Pose / Feature / Sequence / Realtime 공통 함수
├── reports/             # QA, 평가, Confusion Matrix, 실행 로그 등
├── data/                # 로컬 입력 영상 및 생성 데이터 (환경별 구성)
├── artifacts/           # 로컬 학습 모델 및 Feature 메타데이터
├── pyproject.toml
├── uv.lock
└── README.md
```

`data/`와 `artifacts/`는 실행에 필요한 로컬 경로를 설명한 것으로, GitHub에 모든 원본 데이터나 학습 산출물이 포함되어 있다는 뜻은 아닙니다.

## 9. 실행 방법

프로젝트 루트에서 실행합니다. 실행 전 `configs/` 설정, 원본 영상, 필요한 학습 모델 파일을 준비해야 합니다.

```bash
git clone https://github.com/lee26060715/subject10_pose_action.git
cd subject10_pose_action
uv sync
```

**데이터셋 구축 → 학습 → 평가**

```bash
python scripts/25_check_day05_data.py
python scripts/26_extract_keypoint_dataset.py
python scripts/27_keypoint_qa.py
# Human Review CSV에서 PASS / RESHOOT / EXCLUDE 판정 후
python scripts/28_validate_human_review.py
python scripts/29_build_approved_manifest.py
python scripts/30_build_subject_split.py
python scripts/31_check_split_leakage.py
python scripts/33_check_day06_inputs.py
python scripts/34_build_sequence_dataset.py
python scripts/35_train_rf_baseline.py
python scripts/36_evaluate_validation.py
python scripts/40_evaluate_test.py
```

**저장 영상 기반 추론**

```bash
python scripts/41_check_day07_inputs.py
python scripts/42_realtime_action_video.py --source data/day05/videos/S02/S02_A006_T01.mp4
python scripts/43_summarize_realtime_runs.py
```

**IP Webcam 기반 추론**

```bash
python scripts/44_ip_webcam_action_realtime.py --url "http://<IP-ADDRESS>:8080/video"
python scripts/45_analyze_ip_webcam_log.py
```

IP Webcam URL은 실제 스마트폰 스트림 주소로 변경해야 합니다. 실시간 추론에는 `artifacts/day06/`의 학습 모델 및 관련 메타데이터가 필요합니다. `uv`가 없다면 저장소의 `pyproject.toml`에 명시된 의존성에 맞춰 환경을 구성하세요.

## 10. 평가 결과 및 해석

평가 스크립트는 다음 결과를 생성하도록 구성되어 있습니다.

- `reports/day06/metrics/validation_metrics.txt`
- `reports/day06/metrics/test_metrics.txt`
- `reports/day06/confusion/test_confusion_matrix.png`
- `reports/day06/misclassified/validation_misclassified.csv`
- `reports/day06/experiments/`의 비교 실험 CSV
- `reports/day07/`의 실시간 실행 로그 및 요약 CSV

**정확도·F1 수치는 실제 실행 결과를 확인한 뒤 기재해야 합니다.** 스크립트에 지표 계산 로직이 있다는 사실만으로 특정 성능을 달성했다고 주장하지 않습니다.

## 11. 학습 포인트 및 개선 방향

이 프로젝트를 통해 영상에서 추출한 Pose 데이터를 모델 학습에 적합한 구조로 변환하고, **데이터 품질 검증 → 인물별 분리 → Baseline 평가 → 실시간 추론**까지 연결하는 과정을 다룹니다.

향후에는 더 많은 Subject와 촬영 환경을 확보하여 일반화 성능을 확인하고, Sequence 길이·Confidence 기준·예측 안정화 설정을 체계적으로 비교하며, 실제 사용자 동작에서 발생하는 오분류 사례를 개선할 수 있습니다.

---

**작성 기준:** 프로젝트의 단계별 Python 스크립트 및 공개 저장소 구조. 수치 성능과 실행 성공 여부는 별도 검증 전까지 단정하지 않습니다.
