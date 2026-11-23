"""One unified protocol, exact Val/Test pairing from frozen saved results."""
from pathlib import Path
from numbers import Number
import pandas as pd
import inference as engine
APP=Path(__file__).resolve().parent
PRIMARY_COHORT='Validation2139'
PROPOSED_ID='ctbert_url_emoji_cal'
COHORTS={'Validation2139':'Validation · 2.139 mẫu','Test2140_reused':'Test · 2.140 mẫu (đã xem trước)'}
MEMBER_NAMES={'ctbert':'CT-BERT','word_svm':'SVM (từ)','char_lr':'Logistic (ký tự)','distilbert':'DistilBERT','word_nb':'Naive Bayes (từ)'}
CATALOG={}
for key,spec in engine.registry()['models'].items():
    base=spec.get('base')
    kind='Kết hợp' if spec['kind']=='ensemble' else ('Graph' if base in ['gcn','gat'] else ('Học máy' if base in ['word_lr','word_svm','word_nb','char_lr'] else 'Transformer'))
    CATALOG[key]=(spec['label'],kind,spec.get('metric_model',base),'calibrated')
def available():
    choices,missing=engine.options()
    return {k:v[0] for k,v in CATALOG.items() if k in choices},{k:missing[k] for k in CATALOG if k in missing}
def overview(protocol=None,cohort=None):
    final=pd.read_csv(APP/'metrics_unified.csv');rows=[]
    supplement=APP/'metrics_ctbert.csv'
    if supplement.is_file():
        import json
        release=json.loads((APP/'release.json').read_text(encoding='utf-8'))
        if engine.digest(supplement)!=release['ctbert_metrics_sha256']:
            raise ValueError('Bảng metric CT-BERT không khớp checksum.')
        final=pd.concat([final,pd.read_csv(supplement)],ignore_index=True)
    for key,(name,kind,model,rep) in CATALOG.items():
        val=final[(final.model==model)&(final.split=='Validation2139')]
        test=final[(final.model==model)&(final.split=='Test2140_reused')]
        if len(val)!=1 or len(test)!=1:raise ValueError('Metric thiếu hoặc không khớp đúng checkpoint/giao thức.')
        row=val.iloc[0].to_dict();row.update(display_name=name,kind=kind,app_id=key,
            role="Đề xuất" if key==PROPOSED_ID else ("Đối chứng ensemble" if kind=="Kết hợp" else "Baseline"))
        for col,value in test.iloc[0].items():
            if isinstance(value,Number):row['test_'+col]=value
        rows.append(row)
    result=pd.DataFrame(rows)
    assert set(result.n)=={2139} and set(result.test_n)=={2140} and result.app_id.is_unique
    return result.sort_values('weighted_f1',ascending=False,kind='stable').reset_index(drop=True)
