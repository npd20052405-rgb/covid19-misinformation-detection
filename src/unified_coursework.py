"""Notebook18: one fixed inner-fit protocol, calibration C, pooling M, frozen Val/Test."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json,time,gc,traceback,warnings,sys
from functools import lru_cache
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,log_loss
from scipy.optimize import minimize
import expanded_coursework as old
import gap_controlled_graphs as graph
import pilot_h1_h2 as pm
from text_preprocessing import normalize_text

ROOT=old.ROOT;RUN='20261022_unified_coursework_v1';OUT=ROOT/'results'/RUN;MODELS=ROOT/'models'/RUN;CFG=ROOT/'configs'/(RUN+'.json')
KEYS=['word_lr','word_svm','word_nb','char_lr','bert_tiny','distilbert','gcn','gat','bert','roberta','gpt2']
NAMES={'word_lr':'Word TF-IDF + Logistic Regression','word_svm':'Word TF-IDF + LinearSVC','word_nb':'Word TF-IDF + MultinomialNB','char_lr':'Char TF-IDF + Logistic Regression','bert_tiny':'BERT-Tiny','distilbert':'DistilBERT','gcn':'GCN','gat':'GAT','bert':'BERT-Base','roberta':'RoBERTa-Base','gpt2':'GPT-2'}
GROUPS={'B0':['word_nb','char_lr'],'B1':['word_svm','char_lr'],'B2':['word_nb','char_lr','word_svm'],'B3':['word_svm','distilbert']}
PRETRAINED={'bert_tiny':ROOT/'models/pretrained/google_bert_tiny_30b0a37ccaaa32f332884b96992754e246e48c5f','distilbert':ROOT/'models/pretrained/distilbert_12040accade4e8a0f71eabdb258fecc2e7e948be'}
def save(p,v):old.save(p,v)
def read(p):return old.read(p)
def log(s):
    OUT.mkdir(parents=True,exist_ok=True);print(s,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+s+'\n')
def verify(d):
    assert all(old.sha(ROOT/p)==h for p,h in d['files'].items())
def inner():
    a=old.partition('A');ix=pd.read_csv(ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv').set_index('sample_id').loc[a.sample_id]
    return a,np.flatnonzero(ix.inner_role.to_numpy()=='fit'),np.flatnonzero(ix.inner_role.to_numpy()=='dev')
def prepare():
    import psutil,shutil,importlib.metadata as md
    graph.prepare();assert psutil.virtual_memory().available>1_000_000_000
    assert shutil.disk_usage(ROOT).free>8_000_000_000
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0]>2_500_000_000
    probe=np.array([[[.8,.2],[.3,.7]]])
    assert np.allclose(pool(probe,[.5,.5]),[[.55,.45]])
    assert np.allclose(pool(probe,[1.,0.]),probe[:,0])
    assert np.allclose(pool(probe,[0.,0.,1.]).sum(1),1)
    cfg=dict(run=RUN,seed=42,models=KEYS,groups=GROUPS,fit_n=3792,inner_dev_n=702,C_n=642,M_n=642,
        assignment_sha256=old.sha(old.PREP/'assignment.csv'),inner_sha256=old.sha(ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv'),
        reused_models=['gcn','gat','bert','roberta','gpt2'],train_new=['word_lr','word_svm','word_nb','char_lr','bert_tiny','distilbert'],
        classical=read(ROOT/'configs/pilot_h1_h2_seed42_v1.json'),
        neural={'bert_tiny':{'epochs':4,'lr':5e-5,'batch':16},'distilbert':{'epochs':3,'lr':1e-5,'batch':4}},
        neural_common={'dropout':.2,'weight_decay':.1,'max_length':128,'warmup':.1,'gradient_clip':1.,'eval_batch':4},
        selection='raw unsmoothed Train/inner-Validation CE and weighted F1 eval mode; both absolute gaps<0.1, then maximum inner-Validation F1, lower loss, earliest epoch; if no eligible report failure, never loosen',
        ML_selection='fixed original hyperparameters, no epoch selection; report raw/calibrated Train-innerDev gaps; SVM raw CE is N/A; gap is diagnostic, no fabricated eligibility',
        calibration='all 11 base models: LogisticRegression C=1 max_iter2000 seed42 on C score only; LR/NB/neural logit P(real) clip1e-7; LinearSVC real margin; no tuning',
        combiner='B0/B1/B2/B3; Equal Mean, Weighted Mean, nonnegative 2-additive Choquet; normalized pair-min for pairs; learn only on calibrated M; SLSQP log loss bounds0..1 sum1 maxiter300 ftol1e-10; equal singleton initial',
        evaluation='lock every checkpoint/calibrator/combiner before Official Validation2139 and Test2140; no tuning on either; report all fixed candidates',
        test='user explicitly approved reevaluation of already-seen Test mirror; reused Test is not a new independent confirmatory test; source pinned commit030deba36d118c124709fd72d7d5f14c7bc69053',
        no_test_before_lock=True,threshold='argmax fake then real; ties fake',new_training_budget_per_model_seconds=3600,total_training_budget_seconds=14400,multi_seed=False,new_folds=0)
    if CFG.exists():assert read(CFG)==cfg;assert read(OUT/'preflight.json')['code_sha256']==old.sha(__file__)
    else:
        save(CFG,cfg);save(OUT/'preflight.json',dict(code_sha256=old.sha(__file__),config_sha256=old.sha(CFG),ram_available=psutil.virtual_memory().available,disk_free=shutil.disk_usage(ROOT).free,gpu=torch.cuda.get_device_name(),gpu_free_total=list(torch.cuda.mem_get_info()),versions={p:md.version(p) for p in ['torch','transformers','numpy','scikit-learn','scipy','streamlit']}))
    log('UNIFIED PROTOCOL LOCKED; no Val/Test read before all parameter locks');return cfg

def reuse():
    prepare()
    for key in ['gcn','gat','bert','roberta','gpt2']:
        target=OUT/key/'base_lock.json'
        if target.exists():verify(read(target));continue
        src=ROOT/'results'/('20261022_gap_controlled_graphs_v1' if key in ['gcn','gat'] else '20261022_gap_controlled_transformers_v2')/key
        d=read(src/'COMPLETE.json');verify(d);assert d['status']=='PASS_GAPS';epoch=d['selected']['epoch']
        if key in ['gcn','gat']:
            checkpoint=ROOT/d['checkpoint'];vocab=checkpoint.parent/'vocab.json'
            spec=dict(kind='graph',key=key,checkpoint=str(checkpoint.relative_to(ROOT)),vocab=str(vocab.relative_to(ROOT)),files={str(p.relative_to(ROOT)):old.sha(p) for p in [checkpoint,vocab]})
        else:
            checkpoint=ROOT/'models'/src.parent.name/key/f'epoch_{epoch:02d}'
            spec=dict(kind='transformer',key=key,checkpoint=str(checkpoint.relative_to(ROOT)),files={str(p.relative_to(ROOT)):old.sha(p) for p in checkpoint.glob('*') if p.is_file()})
        spec.update(selected=d['selected'],fit_n=3792,reused=True,source=str(src.relative_to(ROOT)),base_training_seconds=d.get('end_to_end_seconds',d.get('training_seconds')))
        save(target,spec)
    log('REUSED 5 valid inner-fit checkpoints; no retraining')

def fit_classical():
    cfg=prepare();a,fi,di=inner()
    from threadpoolctl import threadpool_limits
    from sklearn.exceptions import ConvergenceWarning
    for key in ['word_lr','word_svm','word_nb','char_lr']:
        target=OUT/key/'base_lock.json'
        if target.exists():verify(read(target));continue
        model=pm._base_pipeline(key,cfg['classical']);tick=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
            warnings.simplefilter('always',ConvergenceWarning);model.fit(a.tweet.iloc[fi],a.label.iloc[fi])
        if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Classifier did not converge; no silent tuning')
        path=MODELS/key/'pipeline.joblib';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();joblib.dump(model,path)
        save(target,dict(kind='sklearn',key=key,pipeline=str(path.relative_to(ROOT)),files={str(path.relative_to(ROOT)):old.sha(path)},fit_n=3792,base_training_seconds=time.perf_counter()-tick,epoch_history='N/A',selected=None,reused=False))
        log(key+' fit3792 complete')

def fit_neural(key):
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_linear_schedule_with_warmup
    cfg=prepare();assert key in PRETRAINED
    target=OUT/key/'base_lock.json'
    if target.exists():verify(read(target));log(key+' COMPLETE reused');return
    out=OUT/key;assert not out.exists(),'Partial failed attempt preserved; inspect before retry'
    out.mkdir(parents=True);start=time.perf_counter();old.seed();a,fi,di=inner();source=PRETRAINED[key]
    prov=read(source/'provenance.json');files=prov['files'];files={r['name']:r['sha256'] for r in files} if isinstance(files,list) else files
    assert all(old.sha(source/p)==h for p,h in files.items())
    tok=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False)
    enc=tok(a.tweet.map(normalize_text).tolist(),padding='max_length',truncation=True,max_length=128,return_tensors='pt');y=torch.tensor((a.label=='real').astype(int).to_numpy())
    model=None
    try:
        model=AutoModelForSequenceClassification.from_pretrained(source,local_files_only=True,trust_remote_code=False,use_safetensors=True,num_labels=2,id2label={0:'fake',1:'real'},label2id={'fake':0,'real':1},attn_implementation='eager',low_cpu_mem_usage=False).cuda()
        for m in model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=.2
        for attr in ['hidden_dropout_prob','attention_probs_dropout_prob','classifier_dropout','dropout','attention_dropout','seq_classif_dropout']:
            if hasattr(model.config,attr):setattr(model.config,attr,.2)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        settings=cfg['neural'][key]
        class Dataset(torch.utils.data.Dataset):
            def __init__(self,ids):self.ids=list(ids)
            def __len__(self):return len(self.ids)
            def __getitem__(self,k):
                i=self.ids[k];return {**{n:v[i] for n,v in enc.items()},'labels':y[i]}
        def loader(ids,shuffle=False):return torch.utils.data.DataLoader(Dataset(ids),batch_size=settings['batch'] if shuffle else 4,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(42))
        trainloader=loader(fi,True);opt=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=.1)
        steps=len(trainloader)*settings['epochs'];sched=get_linear_schedule_with_warmup(opt,int(.1*steps),steps);scaler=torch.amp.GradScaler('cuda')
        def forward(b):
            b={k:v.cuda() for k,v in b.items()}
            with torch.autocast('cuda',dtype=torch.float16):z=model(**b).logits.float()
            loss=torch.nn.functional.cross_entropy(z,b['labels']);assert torch.isfinite(z).all() and torch.isfinite(loss)
            return loss,z,b['labels']
        def evaluate(ids):
            model.eval();total=0.;pred=[]
            with torch.no_grad():
                for b in loader(ids):
                    loss,z,t=forward(b);total+=loss.item()*len(t);pred.extend(z.argmax(1).cpu().tolist())
            return total/len(ids),float(f1_score(y[ids],pred,average='weighted',zero_division=0))
        hist=[];best=None;torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,settings['epochs']+1):
            tick=time.perf_counter();model.train()
            for step,b in enumerate(trainloader,1):
                if time.perf_counter()-start>3600:raise TimeoutError('60 minute cap exceeded')
                opt.zero_grad(set_to_none=True);loss,z,t=forward(b);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                before=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()>=before:sched.step()
                if step%200==0:log(f'{key} epoch{epoch} step{step}/{len(trainloader)}')
            tl,tf=evaluate(fi);dl,df=evaluate(di);row=dict(epoch=epoch,train_loss=tl,dev_loss=dl,train_f1=tf,dev_f1=df,loss_gap=abs(tl-dl),f1_gap=abs(tf-df),seconds=time.perf_counter()-tick);row['eligible']=graph.eligible(row);hist.append(row);save(out/f'epoch_{epoch:02d}.json',row)
            if row['eligible'] and (best is None or graph.rank(row)>graph.rank(best)):
                best=row.copy();dest=MODELS/key/f'epoch_{epoch:02d}';model.save_pretrained(dest,safe_serialization=True);tok.save_pretrained(dest)
            log(key+' '+json.dumps(row))
        pd.DataFrame(hist).to_csv(out/'history.csv',index=False,mode='x')
        if best is None:raise RuntimeError('No eligible checkpoint under the frozen two-gap rule; do not relax')
        dest=MODELS/key/f'epoch_{best["epoch"]:02d}'
        save(target,dict(kind='transformer',key=key,checkpoint=str(dest.relative_to(ROOT)),files={str(p.relative_to(ROOT)):old.sha(p) for p in dest.glob('*') if p.is_file()},fit_n=3792,selected=best,reused=False,base_training_seconds=time.perf_counter()-start,peak_vram_bytes=torch.cuda.max_memory_allocated()))
        log(key+' BASE COMPLETE')
    except Exception:
        save(out/'FAILED.json',dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start));raise
    finally:
        if model is not None:del model
        gc.collect();torch.cuda.empty_cache()

@lru_cache(maxsize=1)
def load(key):
    d=read(OUT/key/'base_lock.json');verify(d)
    if d['kind']=='sklearn':return joblib.load(ROOT/d['pipeline']),None
    if d['kind']=='graph':
        from safetensors.torch import load_file
        vocab=read(ROOT/d['vocab']);model=graph.Graph(key,len(vocab)+2);model.load_state_dict(load_file(str(ROOT/d['checkpoint'])));return model.eval(),vocab
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    dest=ROOT/d['checkpoint'];tok=AutoTokenizer.from_pretrained(dest,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(dest,local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager').cuda().eval();return model,tok

def raw(key,texts):
    from threadpoolctl import threadpool_limits
    d=read(OUT/key/'base_lock.json');model,aux=load(key);tick=time.perf_counter();p=None
    if d['kind']=='sklearn':
        with threadpool_limits(limits=2):p,score=pm._prob_and_score(model,texts)
    elif d['kind']=='graph':
        from torch_geometric.loader import DataLoader
        blocks=[]
        with torch.inference_mode():
            for b in DataLoader([old.graph_data(t,0,aux) for t in texts],batch_size=32,shuffle=False):blocks.append(model(b).softmax(1).numpy())
        p=np.concatenate(blocks)
    else:
        blocks=[];norm=[normalize_text(t) for t in texts]
        with torch.inference_mode():
            for start in range(0,len(texts),4):
                b=aux(norm[start:start+4],padding='max_length',truncation=True,max_length=128,return_tensors='pt').to('cuda')
                with torch.autocast('cuda',dtype=torch.float16):z=model(**b).logits.float()
                blocks.append(z.softmax(1).cpu().numpy())
        p=np.concatenate(blocks)
    if p is not None:
        p=np.asarray(p,dtype=float);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6);p/=p.sum(1,keepdims=True)
        v=np.clip(p[:,1],1e-7,1-1e-7);score=np.log(v/(1-v)).reshape(-1,1)
    return p,score,time.perf_counter()-tick

def frame(data,p):
    f=data[['sample_id','label']].rename(columns={'label':'true_label'}).copy();f['predicted_label']=pm.CLASSES[p.argmax(1)];f['prob_fake']=p[:,0];f['prob_real']=p[:,1];return f

def calibrate():
    prepare();a,fi,di=inner();c=old.partition('C');m=old.partition('M');rows=[]
    for key in KEYS:
        dest=OUT/key
        if (dest/'calibration_lock.json').exists():verify(read(dest/'calibration_lock.json'));continue
        load.cache_clear();tick=time.perf_counter();_,score,ci=raw(key,c.tweet.tolist());cal=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(score,c.label);fit_time=time.perf_counter()-tick-ci
        assert list(cal.classes_)==['fake','real'];path=dest/'calibrator.joblib';assert not path.exists();joblib.dump(cal,path)
        diagnostics=[]
        for name,data in [('M',m),('train_inner',a.iloc[fi]),('validation_inner',a.iloc[di])]:
            p,s,secs=raw(key,data.tweet.tolist());pc=cal.predict_proba(s);f=frame(data,pc);f.to_csv(dest/f'{name}_calibrated_predictions.csv',index=False,mode='x');met,bins=pm.evaluate(f);met.update(model=key,cohort=name,representation='calibrated',base_inference_seconds=secs);diagnostics.append(met);bins.to_csv(dest/f'{name}_calibration_bins.csv',index=False,mode='x')
        pd.DataFrame(diagnostics).to_csv(dest/'internal_metrics.csv',index=False,mode='x')
        lock=dict(fit_role='C only',calibrator=str(path.relative_to(ROOT)),calibration_fit_seconds=fit_time,C_inference_seconds=ci,files={str(p.relative_to(ROOT)):old.sha(p) for p in dest.glob('*') if p.is_file()})
        save(dest/'calibration_lock.json',lock);log(key+' CALIBRATION COMPLETE; internal diagnostics only')
        load.cache_clear();gc.collect();torch.cuda.empty_cache()

def pool(p,w):
    n=p.shape[1];basis=[p[:,i] for i in range(n)]
    if len(w)>n:basis.extend(np.minimum(p[:,i],p[:,j]) for i,j in combinations(range(n),2))
    scores=np.einsum('nkc,k->nc',np.stack(basis,axis=1),np.asarray(w));z=scores.sum(1,keepdims=True)
    return np.divide(scores,z,out=np.full_like(scores,.5),where=z>1e-15)

def meta_and_freeze():
    if (OUT/'global_development_lock.json').exists():return read(OUT/'global_development_lock.json')
    params={}
    for key in KEYS:verify(read(OUT/key/'base_lock.json'));verify(read(OUT/key/'calibration_lock.json'))
    for group,members in GROUPS.items():
        frames=[pd.read_csv(OUT/k/'M_calibrated_predictions.csv') for k in members];ref=frames[0]
        assert all(np.array_equal(f.sample_id,ref.sample_id) and np.array_equal(f.true_label,ref.true_label) for f in frames)
        p=np.stack([f[['prob_fake','prob_real']].to_numpy() for f in frames],axis=1);n=len(members);y=(ref.true_label=='real').astype(int).to_numpy()
        methods={'equal_mean':dict(weights=[1/n]*n,fit_seconds=0.)}
        if n==2:methods['normalized_pair_min']=dict(weights=[0.,0.,1.],fit_seconds=0.)
        for name,nw in [('weighted_mean',n),('learned_choquet',n+n*(n-1)//2)]:
            tick=time.perf_counter();x0=np.array([1/n]*n+[0.]*(nw-n))
            result=minimize(lambda w:log_loss(y,pool(p,w),labels=[0,1]),x0,method='SLSQP',bounds=[(0,1)]*nw,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300,'ftol':1e-10})
            assert result.success,(group,name,result.message);w=np.clip(result.x,0,1);w/=w.sum()
            assert (w>=0).all() and np.isclose(w.sum(),1)
            methods[name]=dict(weights=w.tolist(),fit_seconds=time.perf_counter()-tick,objective_M_fit=float(log_loss(y,pool(p,w),labels=[0,1])))
        for name,value in methods.items():params[group+'__'+name]=dict(members=members,**value)
    save(OUT/'combiner_parameters.json',params)
    locks=[CFG,OUT/'preflight.json',OUT/'combiner_parameters.json']+[OUT/k/name for k in KEYS for name in ['base_lock.json','calibration_lock.json']]
    frozen=dict(status='ALL_PARAMETERS_FROZEN',models=KEYS,combiners=list(params),seed=42,fit_n=3792,inner_dev_n=702,Val_n=2139,Test_n=2140,Test_previously_seen=True,no_new_independence_claim=True,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files={str(p.relative_to(ROOT)):old.sha(p) for p in locks},code_sha256=old.sha(__file__))
    save(OUT/'global_development_lock.json',frozen);log('GLOBAL LOCK COMPLETE before reading Val/Test for this unified round');return frozen

def evaluate_final():
    lock=read(OUT/'global_development_lock.json');verify(lock)
    assert lock['status']=='ALL_PARAMETERS_FROZEN' and lock['code_sha256']==old.sha(__file__)
    if (OUT/'EVALUATION_COMPLETE.json').exists():verify(read(OUT/'EVALUATION_COMPLETE.json'));return pd.read_csv(OUT/'final_metrics.csv')
    assert not (OUT/'evaluation').exists(),'Partial evaluation preserved; inspect before retry'
    dest=OUT/'evaluation';dest.mkdir();valpath=pm.RAW/'Constraint_Val.csv';assert old.sha(valpath)==pm.VAL_SHA
    val=pd.read_csv(valpath,dtype=str);val['sample_id']='validation:'+val.id;val=val[val.sample_id!='validation:910'];assert len(val)==2139
    # Access the already-approved mirror only AFTER the new global lock. No download or tuning.
    testroot=ROOT/'data/raw/constraint2021_test_mirror/030deba36d118c124709fd72d7d5f14c7bc69053';provenance=read(testroot/'provenance.json')
    expected={r['name']:r for r in provenance['files']}
    import hashlib
    for name in ['Constraint_Test.csv','english_test_with_labels.csv']:
        payload=(testroot/name).read_bytes();assert old.sha(testroot/name)==expected[name]['sha256']
        assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==expected[name]['git_blob_sha1']
    test=pd.read_csv(testroot/'english_test_with_labels.csv',dtype=str);unlabeled=pd.read_csv(testroot/'Constraint_Test.csv',dtype=str)
    assert len(test)==2140 and test.id.is_unique and unlabeled.id.is_unique
    assert np.array_equal(unlabeled.set_index('id').loc[test.id].tweet.to_numpy(),test.tweet.to_numpy())
    test['sample_id']='test:'+test.id;rows=[];timings=[];params=read(OUT/'combiner_parameters.json')
    save(OUT/'test_source.json',dict(source='user-approved mirror; previously evaluated Test; not independent new test',provenance_sha256=old.sha(testroot/'provenance.json'),global_lock_sha256=old.sha(OUT/'global_development_lock.json'),n=2140))
    for split,data in [('Validation2139',val),('Test2140_reused',test)]:
        probabilities={}
        for key in KEYS:
            load.cache_clear();_,s,seconds=raw(key,data.tweet.tolist());cal=joblib.load(OUT/key/'calibrator.joblib');tick=time.perf_counter();p=cal.predict_proba(s);cal_seconds=time.perf_counter()-tick;probabilities[key]=p
            f=frame(data,p);f.to_csv(dest/f'{key}_{split}_predictions.csv',index=False,mode='x');scores,bins=pm.evaluate(f);scores.update(model=key,split=split,protocol='Unified inner-fit3792 + C calibration',representation='calibrated',evaluation_status='development; previously seen evaluation data');rows.append(scores);bins.to_csv(dest/f'{key}_{split}_calibration_bins.csv',index=False,mode='x');timings.append(dict(model=key,split=split,n=len(data),base_inference_seconds=seconds,calibrator_seconds=cal_seconds));log(key+' '+split+' evaluated; no tuning')
            load.cache_clear();gc.collect();torch.cuda.empty_cache()
        for key,spec in params.items():
            tick=time.perf_counter();p=pool(np.stack([probabilities[k] for k in spec['members']],axis=1),spec['weights']);seconds=time.perf_counter()-tick;f=frame(data,p);f.to_csv(dest/f'{key}_{split}_predictions.csv',index=False,mode='x');scores,bins=pm.evaluate(f);scores.update(model=key,split=split,protocol='Unified inner-fit3792 + C calibration + M meta-fit',representation='calibrated',evaluation_status='development; previously seen evaluation data');rows.append(scores);bins.to_csv(dest/f'{key}_{split}_calibration_bins.csv',index=False,mode='x');timings.append(dict(model=key,split=split,n=len(data),pooling_only_seconds=seconds))
    pd.DataFrame(rows).to_csv(OUT/'final_metrics.csv',index=False,mode='x');pd.DataFrame(timings).to_csv(OUT/'inference_timing.csv',index=False,mode='x')
    save(OUT/'EVALUATION_COMPLETE.json',dict(status='COMPLETE',models=11,combiners=len(params),files={str(p.relative_to(ROOT)):old.sha(p) for p in [OUT/'final_metrics.csv',OUT/'inference_timing.csv',*dest.glob('*')]}));log('UNIFIED FINAL EVALUATION COMPLETE; no post-evaluation parameter changes');return pd.DataFrame(rows)
