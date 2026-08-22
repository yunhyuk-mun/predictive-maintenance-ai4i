# Predictive Maintenance — XGBoost & SHAP Analysis

> Failure-prediction model for industrial equipment on the AI4I 2020 dataset. Handles severe
> class imbalance (3.4% failure rate) and uses SHAP to turn model output into interpretable,
> root-cause-style signals rather than a black-box score.

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
| 핵심 고장 원인 (SHAP) | Tool wear, Torque |

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

## 구성

```
predictive_maintenance_xgboost_shap.ipynb
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
[Kaggle 노트북](https://www.kaggle.com/code/ericwittjr/predictive-maintenance-xgboost-shap-analysis)에서 바로 볼 수 있다.
