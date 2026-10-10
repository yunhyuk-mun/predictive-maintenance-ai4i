# Azure PdM에 조치 설계 v1.1 구조 적용: 시간 분할, 24시간 앞 예측, 알림 에피소드, 리드타임
# 사용법: python azure_pdm_v11.py <PdM_*.csv가 있는 폴더>
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
P=sys.argv[1]; H=24
tel=pd.read_csv(f'{P}/PdM_telemetry.csv',parse_dates=['datetime'])
err=pd.read_csv(f'{P}/PdM_errors.csv',parse_dates=['datetime'])
mnt=pd.read_csv(f'{P}/PdM_maint.csv',parse_dates=['datetime'])
fl=pd.read_csv(f'{P}/PdM_failures.csv',parse_dates=['datetime'])
mac=pd.read_csv(f'{P}/PdM_machines.csv')
S=['volt','rotate','pressure','vibration']
tel=tel.sort_values(['machineID','datetime'])
g=tel.groupby('machineID')[S]
for w in [3,24]:
    m=g.rolling(w,min_periods=1).mean().reset_index(level=0,drop=True); s=g.rolling(w,min_periods=2).std().reset_index(level=0,drop=True)
    for c in S: tel[f'{c}_m{w}']=m[c]; tel[f'{c}_s{w}']=s[c]
# 오류 24시간 누적
e=pd.get_dummies(err.set_index(['datetime','machineID']).errorID).groupby(level=[0,1]).sum().reset_index()
tel=tel.merge(e,on=['datetime','machineID'],how='left'); EC=['error1','error2','error3','error4','error5']
tel[EC]=tel[EC].fillna(0)
for c in EC: tel[c+'_24']=tel.groupby('machineID')[c].transform(lambda x:x.rolling(24,min_periods=1).sum())
tel['err_any_24']=tel[[c+'_24' for c in EC]].sum(1)
# 부품별 마지막 교체 후 경과일 (t 시점까지의 정비 기록만)
for c in ['comp1','comp2','comp3','comp4']:
    mm=mnt[mnt.comp==c][['datetime','machineID']].rename(columns={'datetime':'last_'+c}).sort_values('last_'+c)
    tel=pd.merge_asof(tel.sort_values('datetime'),mm,left_on='datetime',right_on='last_'+c,by='machineID',direction='backward')
    tel['days_'+c]=(tel.datetime-tel['last_'+c]).dt.total_seconds()/86400; tel.drop(columns='last_'+c,inplace=True)
tel=tel.merge(mac,on='machineID'); tel=pd.concat([tel,pd.get_dummies(tel.model,dtype=int)],axis=1)
# 라벨: (t, t+24h] 안에 고장
tel=tel.sort_values(['machineID','datetime']).reset_index(drop=True)
fset=fl.groupby('machineID').datetime.apply(lambda x:np.sort(x.unique()))
def label(row_m, times):
    f=fset.get(row_m,np.array([],dtype='datetime64[ns]'))
    if len(f)==0: return pd.Series(pd.NaT,index=times.index)
    idx=np.searchsorted(f,times.values,side='right')
    nxt=np.where(idx<len(f),f[np.minimum(idx,len(f)-1)],np.datetime64('NaT'))
    return pd.Series(nxt,index=times.index)
tel['next_fail']=pd.concat([label(m,grp.datetime) for m,grp in tel.groupby('machineID')])
tel['y']=((tel.next_fail-tel.datetime).dt.total_seconds()/3600<=H).astype(int)
snap=tel[tel.datetime.dt.hour%3==0].copy()
F=[c for c in snap.columns if any(c.startswith(s+'_') for s in S)]+[c+'_24' for c in EC]+['err_any_24','days_comp1','days_comp2','days_comp3','days_comp4','age','model1','model2','model3','model4']
tr=snap[snap.datetime<'2015-08-01']; va=snap[(snap.datetime>='2015-08-02')&(snap.datetime<'2015-09-01')]; te=snap[snap.datetime>='2015-09-02'].copy()
spw=(tr.y==0).sum()/(tr.y==1).sum()
mdl=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=spw,random_state=0,eval_metric='logloss').fit(tr[F],tr.y)
va_p=mdl.predict_proba(va[F])[:,1]; te['p']=mdl.predict_proba(te[F])[:,1]
# 검증셋에서 기준 결정: B = F1 최대, A = 재현율 0.95 이상 되는 가장 높은 값
from sklearn.metrics import precision_recall_curve
pr,rc,th=precision_recall_curve(va.y,va_p); f1=2*pr*rc/(pr+rc+1e-9)
B=float(th[np.argmax(f1[:-1])]); A=float(th[np.where(rc[:-1]>=0.95)[0].max()])
print(f'thresholds from validation: A={A:.3f} B={B:.3f}  | train {len(tr)} val {len(va)} test {len(te)} pos_rate_test {te.y.mean():.4f}')
# 에피소드: 같은 설비에서 연속(<=24h 간격) 알림을 하나로 묶음
def episodes(df,mask):
    d=df[mask].sort_values(['machineID','datetime']); out=[]
    for m,gg in d.groupby('machineID'):
        start=prev=None
        for t in gg.datetime:
            if start is None or (t-prev).total_seconds()/3600>H: 
                if start is not None: out.append((m,start,prev))
                start=t
            prev=t
        if start is not None: out.append((m,start,prev))
    return pd.DataFrame(out,columns=['machineID','start','end'])
tf=fl[fl.datetime>='2015-09-03'].drop_duplicates(['machineID','datetime'])   # 테스트 구간 고장 이벤트 (설비·시각 단위)
nonfail_maint=mnt.merge(fl,on=['datetime','machineID'],how='left',indicator=True); nonfail_maint=nonfail_maint[nonfail_maint._merge=='left_only']
def outcome(ep):
    res=[]
    for _,r in ep.iterrows():
        f=fl[(fl.machineID==r.machineID)&(fl.datetime>r.start)&(fl.datetime<=r.end+pd.Timedelta(hours=H))]
        if len(f): res.append('이상 확인'); continue
        mm=nonfail_maint[(nonfail_maint.machineID==r.machineID)&(nonfail_maint.datetime>r.start)&(nonfail_maint.datetime<=r.end+pd.Timedelta(hours=H))]
        res.append('예방 조치' if len(mm) else '이상 없음')
    return pd.Series(res,dtype=object)
def report(name,mask):
    ep=episodes(te,mask); oc=outcome(ep) if len(ep) else pd.Series([],dtype=object)
    caught=[];lead=[]
    for _,f in tf.iterrows():
        a=te[(te.machineID==f.machineID)&mask&(te.datetime<f.datetime)&(te.datetime>=f.datetime-pd.Timedelta(hours=H))]
        caught.append(len(a)>0)
        if len(a): lead.append((f.datetime-a.datetime.min()).total_seconds()/3600)
    months=(te.datetime.max()-te.datetime.min()).days/30
    vc=oc.value_counts().to_dict()
    print(f'{name}: 에피소드 {len(ep)} (설비당 월 {len(ep)/100/months:.2f}) | 고장 {sum(caught)}/{len(tf)} 사전포착 | 리드타임 중앙값 {np.median(lead) if lead else 0:.0f}h | 결과 {vc} | 에피소드 정밀도 {vc.get("이상 확인",0)/max(len(ep),1):.2f}')
    return ep
hi=te.p>=B; mid=(te.p>=A)&~hi
report('단일 기준 B',hi)
report('높음 (B)',hi)
report('중간 (A~B)',mid)
report('높음+중간',hi|mid)
report('높음 & 최근24h 오류',hi&(te.err_any_24>0))
# 계획 교체 근거: 부품 교체 후 경과일별 고장률
rows=[]
for c in ['comp1','comp2','comp3','comp4']:
    d=snap[['days_'+c]].copy(); d['f']=0
    q=fl[fl.failure==c].copy(); q['t0']=q.datetime-pd.Timedelta(hours=3)
    ff=q.merge(snap[['datetime','machineID','days_'+c]].rename(columns={'datetime':'t0'}),on=['t0','machineID'],how='left'); ff['days_'+c]=ff['days_'+c]+3/24
    bins=[0,30,60,90,120,180,400]
    tot=pd.cut(d['days_'+c],bins).value_counts().sort_index(); ev=pd.cut(ff['days_'+c],bins).value_counts().sort_index()
    rows.append((ev/ (tot*3/24)).rename(c))   # 스냅샷 3h 간격 → 일 단위 노출
print('\n부품 교체 후 경과일별 하루당 고장률(x1000):'); print((pd.concat(rows,axis=1)*1000).round(2).to_string())
imp=pd.Series(mdl.feature_importances_,index=F).sort_values(ascending=False).head(8); print('\n상위 변수:', ', '.join(imp.index))

F2=[c for c in F if not c.startswith('error') and c!='err_any_24']
m2=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=spw,random_state=0,eval_metric='logloss').fit(tr[F2],tr.y)
v2=m2.predict_proba(va[F2])[:,1]; te['p']=m2.predict_proba(te[F2])[:,1]
pr,rc,th=precision_recall_curve(va.y,v2); f1=2*pr*rc/(pr+rc+1e-9); B2=float(th[np.argmax(f1[:-1])]); A2=float(th[np.where(rc[:-1]>=0.95)[0].max()])
print(f'\n[오류 기록 없이 센서+교체이력만] A={A2:.3f} B={B2:.3f}')
hi=te.p>=B2; mid=(te.p>=min(A2,B2))&~hi
report('높음 (B)',hi); report('중간 (A~B)',mid); report('높음+중간',hi|mid)
# 고장 직전 마지막 오류까지 시간
fe=[]
for _,f in fl.iterrows():
    e2=err[(err.machineID==f.machineID)&(err.datetime<f.datetime)]
    fe.append((f.datetime-e2.datetime.max()).total_seconds()/3600 if len(e2) else np.nan)
fe=pd.Series(fe); print('\n고장 전 마지막 오류까지 시간(h): 중앙값',fe.median(),' 24h 이내 비율',round((fe<=24).mean(),3))
