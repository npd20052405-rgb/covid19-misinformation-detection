"""Post-lock artifact export, current app registry and complete final metrics. No fitting."""
import json,sys
import numpy as np
import pandas as pd
import unified_coursework as u
ROOT=u.ROOT;OUT=u.OUT;APP=ROOT/'apps/coursework_v1'
MAIN_COMBINERS=['B1__weighted_mean','B1__learned_choquet','B3__weighted_mean','B3__learned_choquet']
LABELS={'word_lr':'Logistic (từ)','word_svm':'SVM (từ)','word_nb':'Naive Bayes (từ)','char_lr':'Logistic (ký tự)','bert_tiny':'BERT-Tiny','distilbert':'DistilBERT','gcn':'GCN (đồ thị từ)','gat':'GAT (đồ thị từ)','bert':'BERT-Base','roberta':'RoBERTa-Base','gpt2':'GPT-2 (phân loại)',
    'B1__weighted_mean':'SVM + Logistic ký tự · Trọng số','B1__learned_choquet':'SVM + Logistic ký tự · Choquet','B3__weighted_mean':'SVM + DistilBERT · Trọng số','B3__learned_choquet':'SVM + DistilBERT · Choquet'}
def update_app():
    u.verify(u.read(OUT/'EVALUATION_COMPLETE.json'));u.verify(u.read(OUT/'global_development_lock.json'))
    params=u.read(OUT/'combiner_parameters.json');reg={'bases':{},'models':{}}
    for key in u.KEYS:
        d=u.read(OUT/key/'base_lock.json');u.verify(d);c=u.read(OUT/key/'calibration_lock.json');u.verify(c)
        paths=[ROOT/p for p in d['files']]+[OUT/key/'base_lock.json',OUT/key/'calibrator.joblib',OUT/'global_development_lock.json']
        reg['bases'][key]={'kind':'unified','unified_key':key,'calibrator':str((OUT/key/'calibrator.joblib').relative_to(ROOT)),'files':{str(p.relative_to(ROOT)):u.old.sha(p) for p in paths}}
        reg['models'][key+'_cal']={'kind':'base','base':key,'calibrated':True,'label':LABELS[key],'protocol':'Unified inner-fit3792 / C calibration / M ensemble'}
    for key in MAIN_COMBINERS:
        appid=key.replace('__','_');reg['models'][appid]={'kind':'ensemble','members':params[key]['members'],'weights':params[key]['weights'],'label':LABELS[key],'protocol':'Unified inner-fit3792 / C calibration / M ensemble','metric_model':key}
    (APP/'model_registry.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2),encoding='utf-8')
    release={'registry_sha256':u.old.sha(APP/'model_registry.json'),'trusted_local_models_only':True,'no_automatic_download':True,'official_test_access':True,'protocol':'single unified protocol; evaluation after lock; Test was previously seen','global_lock_sha256':u.old.sha(OUT/'global_development_lock.json')}
    (APP/'release.json').write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf-8')
    final=pd.read_csv(OUT/'final_metrics.csv');final.to_csv(APP/'metrics_unified.csv',index=False)
    if not (OUT/'app_export.json').exists():u.save(OUT/'app_export.json',dict(models=len(reg['models']),base_models=11,ensemble_choices=4,registry_sha256=release['registry_sha256'],metric_source_sha256=u.old.sha(OUT/'final_metrics.csv')))
    return reg

def audit():
    from sklearn.metrics import f1_score
    u.verify(u.read(OUT/'EVALUATION_COMPLETE.json'));u.verify(u.read(OUT/'global_development_lock.json'))
    final=pd.read_csv(OUT/'final_metrics.csv');maxdiff=0.;checked=0
    for _,row in final.iterrows():
        f=pd.read_csv(OUT/'evaluation'/f'{row.model}_{row.split}_predictions.csv');assert f.sample_id.is_unique and len(f)==row.n
        p=f[['prob_fake','prob_real']].to_numpy();assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-10)
        met,_=u.pm.evaluate(f)
        for k,v in met.items():
            if isinstance(v,(int,float)) and pd.notna(row.get(k)):
                delta=abs(v-row[k]);maxdiff=max(maxdiff,delta);assert delta<1e-10,(row.model,k,delta)
        checked+=1
    assert len(final.model.unique())==26 and set(final.split)=={'Validation2139','Test2140_reused'}
    result=dict(metric_rows_verified=checked,models=11,combiner_variants=15,maximum_metric_difference=maxdiff,Validation_n=2139,Test_n=2140,Test_status='previously seen Test; reused exploratory evaluation',seed=42)
    if not (OUT/'integrity_audit.json').exists():u.save(OUT/'integrity_audit.json',result)
    return result
