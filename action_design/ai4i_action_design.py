"""AI4I 2020: 물리 파생변수 비교 + 조치 설계 v1.1 시뮬레이션 (10회 분할 평균)

사용법: python ai4i_action_design.py ai4i2020.csv
설정은 원본 노트북과 같다: test 0.2, stratify, 학습셋에만 표준화·SMOTE, XGBoost 기본값,
고장 유형 5컬럼(TWF/HDF/PWF/OSF/RNF)은 입력에서 제외.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

d = pd.read_csv(sys.argv[1])
d['type_code'] = d['Type'].map({'L': 0, 'M': 1, 'H': 2})
d['temp_diff'] = d['Process temperature [K]'] - d['Air temperature [K]']
d['power_W'] = d['Torque [Nm]'] * d['Rotational speed [rpm]'] * 2 * np.pi / 60   # 출력 = 토크 x 각속도
d['wear_x_torque'] = d['Tool wear [min]'] * d['Torque [Nm]']
base = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
SETS = {'A 기존': base,
        'B +Type': base + ['type_code'],
        'C +물리 파생': base + ['temp_diff', 'power_W', 'wear_x_torque'],
        'D +Type+물리': base + ['type_code', 'temp_diff', 'power_W', 'wear_x_torque']}
y = d['Machine failure']

# 물리 조건 (UCI 데이터 설명의 고장 규칙). 주의: 이 데이터에서는 곧 라벨 생성 규칙이다.
lim = d['Type'].map({'L': 11000, 'M': 12000, 'H': 13000})
heat = (d.temp_diff < 8.6) & (d['Rotational speed [rpm]'] < 1380)
power = (d.power_W < 3500) | (d.power_W > 9000)
over = d.wear_x_torque > lim
wear = d['Tool wear [min]'] >= 200
phys = heat | power | over | wear
A_TH, B_TH = 0.3, 0.5   # 이전 분석 값을 그대로 사용. 이 10회 분할로 다시 조정하지 않음.

def proba(cols, tr, te, seed):
    sc = StandardScaler(); a = sc.fit_transform(d.loc[tr, cols]); b = sc.transform(d.loc[te, cols])
    Xs, ys = SMOTE(random_state=seed).fit_resample(a, y.loc[tr])
    m = XGBClassifier(random_state=seed, eval_metric='logloss').fit(Xs, ys)
    return pd.Series(m.predict_proba(b)[:, 1], index=te)

feat, pol = {k: [] for k in SETS}, []
twf_miss = [0, 0]
for seed in range(10):
    tr, te = train_test_split(d.index, test_size=0.2, random_state=seed, stratify=y)
    yt = y.loc[te]
    P = {k: proba(c, tr, te, seed) for k, c in SETS.items()}
    for k, p in P.items():
        al = p >= 0.5
        feat[k].append(((al & (yt == 1)).sum(), ((~al) & (yt == 1)).sum(), (al & (yt == 0)).sum()))
    p = P['D +Type+물리']
    miss = te[((p < 0.5) & (yt == 1)).values]
    twf_miss[0] += d.loc[miss, 'TWF'].sum(); twf_miss[1] += len(miss)
    high = (p >= B_TH) & phys.loc[te]
    mid = (p >= A_TH) & ~high
    rule = wear.loc[te] & ~high & ~mid
    allv = high | mid | rule
    k = int(allv.sum()); top = p.sort_values(ascending=False).index[:k]
    pol.append({'high': high.sum(), 'high_hit': (high & (yt == 1)).sum(),
                'mid': mid.sum(), 'mid_hit': (mid & (yt == 1)).sum(),
                'rule': rule.sum(), 'rule_hit': (rule & (yt == 1)).sum(),
                'v1_checks': k, 'v1_miss': ((~allv) & (yt == 1)).sum(),
                'same_budget_threshold_only_miss': int(yt.sum() - yt.loc[top].sum()),
                'to_engineer': ((high | mid) & (heat | power).loc[te]).sum(),
                'to_worker_first': ((high | mid) & ~(heat | power).loc[te]).sum()})

print('== 1. 입력 변수 비교 (임계값 0.5, 10회 평균) ==')
for k, v in feat.items():
    v = np.array(v); print(f'{k:14s} 찾음 {v[:,0].mean():5.1f}  놓침 {v[:,1].mean():5.1f}  헛알림 {v[:,2].mean():5.1f} (범위 {v[:,2].min()}~{v[:,2].max()})')
print(f'D 모델이 놓친 고장 중 공구 마모 고장(TWF): {twf_miss[0]}/{twf_miss[1]}')
print('\n== 2. 조치 설계 v1 시뮬레이션 (10회 평균) ==')
print(pd.DataFrame(pol).mean().round(1).to_string())
print('\n물리 조건만으로 전체 1만 행 알림:', int(phys.sum()), '/ 포착 고장', int((phys & (y == 1)).sum()), '/', int(y.sum()), '(순환성 점검)')
