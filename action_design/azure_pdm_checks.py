# Azure PdM 추가 점검: 규칙 기준선, 라벨 섞기(누수), 처음 보는 설비, 기간·시드 변경
# 사용법: python azure_pdm_checks.py <PdM_*.csv 폴더> azure_pdm_v11.py
# (특성·라벨 구성은 azure_pdm_v11.py 앞부분을 그대로 실행해 재사용한다)
import sys; sys.stdout.reconfigure(encoding='utf-8')
src=open(sys.argv[2],encoding='utf-8').read().split('\n')
cut=[i for i,l in enumerate(src) if l.startswith('tr=snap')][0]
exec('\n'.join(src[:cut]))          # 특성·라벨 구성부만 재사용 (snap, F, fl 생성)
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from xgboost import XGBClassifier
import numpy as np, pandas as pd
def fit(tr,cols,seed=0,y=None):
    y=tr.y if y is None else y; spw=(y==0).sum()/(y==1).sum()
    return XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=spw,random_state=seed,eval_metric='logloss').fit(tr[cols],y)
def ev(name,y,p):
    print(f'{name}: AUC {roc_auc_score(y,p):.3f}  PR-AUC {average_precision_score(y,p):.3f}  (양성비율 {y.mean():.3f})')
F2=[c for c in F if not c.startswith('error') and c!='err_any_24']
tr=snap[snap.datetime<'2015-08-01']; te=snap[snap.datetime>='2015-09-02']
print('== 1. 모델 없는 규칙: 최근 24h 오류 있으면 알림 ==')
r=(te.err_any_24>0).astype(int); tp=((r==1)&(te.y==1)).sum()
print(f'알림 스냅샷 {r.sum()}, 정밀도 {tp/r.sum():.3f}, 재현율 {tp/te.y.sum():.3f}')
ev('규칙(오류 개수 점수)',te.y,te.err_any_24)
ev('모델 전체 특성',te.y,fit(tr,F).predict_proba(te[F])[:,1])
ev('모델 센서+교체이력',te.y,fit(tr,F2).predict_proba(te[F2])[:,1])
F3=[c for c in F2 if not c.startswith('days_')]
ev('모델 센서만(교체이력 없음)',te.y,fit(tr,F3).predict_proba(te[F3])[:,1])
print('\n== 2. 라벨 섞기 (누수 점검): 성능이 0.5 근처로 떨어져야 정상 ==')
rng=np.random.default_rng(0); ys=pd.Series(rng.permutation(tr.y.values),index=tr.index)
ev('섞은 라벨로 학습, 전체 특성',te.y,fit(tr,F,y=ys).predict_proba(te[F])[:,1])
print('\n== 3. 처음 보는 설비 (설비 30대 제외 학습, 같은 기간) ==')
rng=np.random.default_rng(1); hold=set(rng.choice(np.arange(1,101),30,replace=False))
trm=snap[(~snap.machineID.isin(hold))&(snap.datetime<'2015-08-01')]; tem=snap[(snap.machineID.isin(hold))&(snap.datetime>='2015-09-02')]
ev('새 설비, 전체 특성',tem.y,fit(trm,F).predict_proba(tem[F])[:,1])
ev('새 설비, 센서+교체이력',tem.y,fit(trm,F2).predict_proba(tem[F2])[:,1])
ev('새 설비, 센서만',tem.y,fit(trm,F3).predict_proba(tem[F3])[:,1])
print('\n== 4. 기간 바꾸기 (학습 구간 → 다음 2개월) ==')
for a,b,c in [('2015-04-01','2015-04-02','2015-06-01'),('2015-06-01','2015-06-02','2015-08-01'),('2015-10-01','2015-10-02','2015-12-01')]:
    t1=snap[snap.datetime<a]; t2=snap[(snap.datetime>=b)&(snap.datetime<c)]
    ev(f'~{a} 학습 → {b}~{c}, 센서+교체이력',t2.y,fit(t1,F2).predict_proba(t2[F2])[:,1])
print('\n== 5. 시드 바꾸기 (센서+교체이력, 원래 분할) ==')
for s in [1,2,3]: ev(f'seed {s}',te.y,fit(tr,F2,seed=s).predict_proba(te[F2])[:,1])
