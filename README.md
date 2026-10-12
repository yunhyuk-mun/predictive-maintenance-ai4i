# Predictive Maintenance — XGBoost & SHAP Analysis

> Failure-prediction model for industrial equipment on the AI4I 2020 dataset. Handles severe
> class imbalance (3.4% failure rate) and uses SHAP to turn model output into interpretable,
> root-cause-style signals rather than a black-box score.

> 코드·분석은 Claude Code로 작성했고, 문제 재정의·결정·채택은 작성자가 했다.

**TL;DR (English):** Binary classification (machine failure vs. normal) on the public
[AI4I 2020 Predictive Maintenance dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
(10,000 rows, 3.4% positive rate). SMOTE is applied only to the training split (after the
train/test split, not before) to avoid leaking synthetic samples into the test set. The five
failure-subtype columns (TWF/HDF/PWF/OSF/RNF) are dropped from the features since they encode
the failure mode itself and would leak target information. Recall was chosen as the primary
metric over accuracy because missed failures are far more costly than false alarms. SHAP
(TreeExplainer) is used to attribute predictions to specific process variables.

## 결과

| 지표 | 값 |
|---|---|
| Recall (고장 클래스) | 79% |
| F1-score (고장 클래스) | 0.67 |
| 위험 기여 상위 변수 (SHAP) | Tool wear, Torque |

## 방법론에서 신경 쓴 것

- **SMOTE는 train/test 분리 이후, train에만 적용.** 분리 전에 적용하면 test셋에 합성 샘플과
  거의 동일한 실제 샘플이 함께 들어가 성능이 낙관적으로 부풀려질 수 있다.
- **실패 세부유형 컬럼(TWF, HDF, PWF, OSF, RNF) 제외.** 이 컬럼들은 `Machine failure`가 어떤
  방식으로 발생했는지를 기록한 것이라, 남겨두면 사실상 타겟 정보가 피처로 새어 들어간다.
- **재현율(Recall)을 우선 지표로 설정.** 정확도만 보면 전부 정상으로 예측해도 96.6%가 나오는
  불균형 문제이기 때문에, 검출 실패 비용이 오경보 비용보다 큰 실제 상황을 반영해 recall을
  목표로 삼음.
- **SHAP으로 모델 출력을 원인 분석 가능한 형태로 전환.** 어떤 공정 변수가 고장 위험을
  끌어올렸는지 개별 예측 단위로 분해함.

## 추가: 예측에서 조치로 (2026-10)

재현율 79%였지만 테스트셋 알림 96건 중 41건이 헛알림이었다(96/41은 재현 코드 실행 기준이라 이 실행의 재현율은 55/68 ≈ 81%). 그래서 문제를 "더 정확한 모델"에서
**"알림을 받은 사람이 무엇을 하는가"**로 다시 정의했다. 자세한 내용은 [action_design/REPORT.md](action_design/REPORT.md).

| 한 일 | 결과 (합성 데이터, 효과 크기는 현장에서 다시 재야 함) |
|---|---|
| 온도차·출력(토크×각속도) 같은 물리 변수 추가 | 헛알림 35.3 → 19.0 (10회 분할 평균) |
| 놓친 고장 분석 | 59%가 공구 마모 고장 → 모델이 아닌 공구 수명 관리로 분리 |
| 알림 단계·수신자·점검 기록 설계 | 확률 하나로 만든 중간 구간은 93%가 헛알림 → 사람 호출 대신 PM 목록 |
| 시간축·정비 기록이 있는 Azure PdM 데이터로 확장 | 높은 점수(0.999)는 생성기 흔적으로 판정, 정비 기록을 넣자 센서만(0.29) → 0.92 |

## 구성

```
predictive_maintenance_xgboost_shap.ipynb
action_design/
  REPORT.md                 조치 설계 v1.1 보고서
  ai4i_action_design.py     물리 파생변수 비교 + 조치 정책 시뮬레이션
  azure_pdm_v11.py          Azure PdM 시간 분할 적용
  azure_pdm_checks.py       누수·새 설비·기간·시드 점검
README.md
```

## 실행

```bash
git clone https://github.com/yunhyuk-mun/predictive-maintenance-ai4i.git
cd predictive-maintenance-ai4i
pip install numpy pandas scikit-learn imbalanced-learn xgboost shap matplotlib
jupyter notebook predictive_maintenance_xgboost_shap.ipynb
```

데이터셋은 [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)에서
받아 `ai4i2020.csv`로 두면 그대로 돌아간다. 렌더링된 결과는
[Kaggle 노트북](https://www.kaggle.com/code/ericwittjr/predictive-maintenance-xgboost-shap-analysis)(본인 Kaggle 계정)에서 바로 볼 수 있다.
