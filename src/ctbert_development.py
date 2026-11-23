"""Notebook19: one pinned CT-BERT run; reuse frozen baseline predictions."""
from unified_coursework import *
import unified_coursework as unified
import hashlib
import shutil

RUN='20261110_ctbert_development_v1'
OUT=ROOT/'results'/RUN
MODELS=ROOT/'models'/RUN
CFG=ROOT/'configs'/(RUN+'.json')
KEY='ctbert'
SOURCE=ROOT/'models/pretrained/ctbert_b113bc3'
PRETRAINED={KEY:SOURCE}
MEMBERS=(KEY,'distilbert')
BASE_SHA='ffc8cfeada7b96b152c088eb154d2a724fd68045bc09cc34a389dc2f20e12179'
SMALL={'config.json':'5582c435586e6c82b4413ae6ad7182f28b2b9158','vocab.txt':'fb140275c155a9c7c5a3b3e0e77a9e839594a938','tokenizer_config.json':'18d63e03c656615691d7e3b5e75c217d4088c357','special_tokens_map.json':'e7b0375001f109a6b8873d756ad4f7bbb15fbaa5'}

def log(s):
    OUT.mkdir(parents=True,exist_ok=True)
    print(s,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+s+'\n')

def prepare():
    prior=read(unified.CFG)
    lock=read(unified.OUT/'global_development_lock.json')
    assert lock['files'][unified.CFG.relative_to(ROOT).as_posix()]==old.sha(unified.CFG) if unified.CFG.relative_to(ROOT).as_posix() in lock['files'] else lock['files'][str(unified.CFG.relative_to(ROOT))]==old.sha(unified.CFG)
    assert old.sha(old.PREP/'assignment.csv')==prior['assignment_sha256']
    assert old.sha(ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv')==prior['inner_sha256']
    a,fi,di=inner();assert (len(a),len(fi),len(di))==(4494,3792,702) and a.sample_id.is_unique
    assert set(fi).isdisjoint(di)
    assert torch.cuda.is_available() and shutil.disk_usage(ROOT).free>8_000_000_000
    cfg=dict(run=RUN,seed=42,neural={KEY:prior['neural']['distilbert']},neural_common=prior['neural_common'],
        selection=prior['selection'],source='digitalepidemiologylab/covid-twitter-bert-v2',revision='b113bc3c2590d7b32ed62603fe1ebe32e1e5beee',
        optimizer_foreach=False,model_budget_seconds=3600,assignment_sha256=prior['assignment_sha256'],inner_sha256=prior['inner_sha256'],
        calibration='fixed sigmoid C only',meta='M only, frozen SLSQP log loss from prior protocol',
        members=MEMBERS,methods=['equal_mean','normalized_pair_min','weighted_mean','learned_choquet'],
        evaluation='Validation2139 development only, after lock; no Test',code_sha256=old.sha(__file__),historical_code_sha256=old.sha(unified.__file__))
    cfg=json.loads(json.dumps(cfg))
    if CFG.exists():assert read(CFG)==cfg
    else:
        save(CFG,cfg)
        save(OUT/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),vram_free_total=list(torch.cuda.mem_get_info())))
    for name,expected in SMALL.items():
        payload=(SOURCE/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==expected,name
    assert old.sha(SOURCE/'pytorch_model.bin')==BASE_SHA
    converted=SOURCE/'model.safetensors'
    if not (SOURCE/'provenance.json').exists():
        from safetensors.torch import save_model
        from transformers import AutoModelForMaskedLM
        assert not converted.exists()
        assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,6)
        from transformers import AutoConfig
        config=AutoConfig.from_pretrained(SOURCE,local_files_only=True,trust_remote_code=False)
        model=AutoModelForMaskedLM.from_config(config)
        state=torch.load(SOURCE/'pytorch_model.bin',map_location='cpu',weights_only=True)
        # Validate encoder coverage; MLM-only key differences are irrelevant to classification.
        incompatible=model.load_state_dict(state,strict=False)
        assert not [k for k in incompatible.missing_keys if k.startswith('bert.')],incompatible
        save_model(model,str(converted),metadata={'format':'pt'})
        save(SOURCE/'provenance.json',dict(source=cfg['source'],revision=cfg['revision'],original_sha256=BASE_SHA,files={p:old.sha(SOURCE/p) for p in [*SMALL,converted.name]}))
    log('Protocol locked; CT-BERT checkpoint verified. No Validation/Test access.')
    return cfg

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
        trainloader=loader(fi,True);opt=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=.1,foreach=False)
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
                    if time.perf_counter()-start>3600:raise TimeoutError('60 minute cap exceeded')
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


def historical(key, split):
    if split=='M':
        path=unified.OUT/key/'M_calibrated_predictions.csv'
        files=read(unified.OUT/key/'calibration_lock.json')['files']
    else:
        path=unified.OUT/'evaluation'/f'{key}_Validation2139_predictions.csv'
        files=read(unified.OUT/'EVALUATION_COMPLETE.json')['files']
    assert old.sha(path)==files[str(path.relative_to(ROOT))]
    return pd.read_csv(path)

def aligned(data,f):
    ref=data.set_index('sample_id').label
    assert f.sample_id.is_unique and set(f.sample_id)==set(ref.index)
    f=f.set_index('sample_id').loc[ref.index]
    assert np.array_equal(f.true_label,ref)
    return f[['prob_fake','prob_real']].to_numpy()

def lock_parameters():
    prepare()
    target=OUT/'development_lock.json'
    if target.exists():
        lock=read(target);verify(lock);return lock
    verify(read(OUT/KEY/'base_lock.json'))
    c,m=old.partition('C'),old.partition('M')
    assert len(c)==len(m)==642 and set(c.sample_id).isdisjoint(m.sample_id)
    _,score,ctime=raw(KEY,c.tweet.tolist())
    tick=time.perf_counter()
    cal=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(score,c.label)
    fit_seconds=time.perf_counter()-tick
    assert list(cal.classes_)==['fake','real']
    cp=OUT/'calibrator.joblib'
    assert not cp.exists()
    joblib.dump(cal,cp)
    _,s,mtime=raw(KEY,m.tweet.tolist())
    mc=cal.predict_proba(s)
    mp=OUT/'M_ctbert_predictions.csv'
    frame(m,mc).to_csv(mp,index=False,mode='x')
    p=np.stack([mc,aligned(m,historical('distilbert','M'))],axis=1)
    y=(m.label=='real').astype(int)
    params={'equal_mean':dict(weights=[.5,.5],fit_seconds=0.),'normalized_pair_min':dict(weights=[0.,0.,1.],fit_seconds=0.)}
    for name,n in [('weighted_mean',2),('learned_choquet',3)]:
        tick=time.perf_counter()
        result=minimize(lambda w:log_loss(y,pool(p,w),labels=[0,1]),[.5,.5]+[0.]*(n-2),method='SLSQP',
            bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300,'ftol':1e-10})
        assert result.success,result.message
        w=np.clip(result.x,0,1);w/=w.sum()
        params[name]=dict(weights=w.tolist(),fit_seconds=time.perf_counter()-tick)
    lock=dict(status='FROZEN',members=list(MEMBERS),params=params,C_inference_seconds=ctime,M_inference_seconds=mtime,
        calibration_fit_seconds=fit_seconds,files={str(q.relative_to(ROOT)):old.sha(q) for q in [Path(__file__),CFG,cp,mp,OUT/KEY/'base_lock.json']})
    save(target,lock);log('Frozen checkpoint, calibration and all pooling parameters before Validation.')
    return lock

def evaluate_validation():
    lock=read(OUT/'development_lock.json');verify(lock);assert lock['status']=='FROZEN'
    verify(read(OUT/KEY/'base_lock.json'))
    completed=OUT/'COMPLETE.json'
    if completed.exists():verify(read(completed));return pd.read_csv(OUT/'metrics.csv')
    valpath=pm.RAW/'Constraint_Val.csv';assert old.sha(valpath)==pm.VAL_SHA
    val=pd.read_csv(valpath,dtype=str);val['sample_id']='validation:'+val.id
    val=val[val.sample_id!='validation:910'];assert len(val)==2139 and val.sample_id.is_unique
    p,s,seconds=raw(KEY,val.tweet.tolist())
    tick=time.perf_counter();pc=joblib.load(OUT/'calibrator.joblib').predict_proba(s);cal_seconds=time.perf_counter()-tick
    scores={k:aligned(val,historical(k,'Val')) for k in ['distilbert','bert','roberta','B3__weighted_mean','B3__learned_choquet']}
    scores.update(ctbert=pc,ctbert_raw=p)
    pair=np.stack([pc,scores['distilbert']],axis=1)
    timings={}
    for name,spec in lock['params'].items():
        tick=time.perf_counter();scores['ctbert_distilbert__'+name]=pool(pair,spec['weights'])
        timings[name]=time.perf_counter()-tick
    rows=[];files=[]
    for name,pred in scores.items():
        f=frame(val,pred);metrics,bins=pm.evaluate(f)
        rows.append(dict(model=name,**metrics))
        for suffix,table in [('predictions',f),('calibration_bins',bins)]:
            q=OUT/f'{name}_{suffix}.csv';table.to_csv(q,index=False,mode='x');files.append(q)
    result=pd.DataFrame(rows)
    result['delta_weighted_f1_vs_distilbert']=result.weighted_f1-result.loc[result.model=='distilbert','weighted_f1'].iloc[0]
    q=OUT/'metrics.csv';result.to_csv(q,index=False,mode='x');files.append(q)
    save(completed,dict(status='COMPLETE',n=len(val),base_inference_seconds=seconds,calibration_seconds=cal_seconds,pooling_seconds=timings,
        test_access=False,files={str(q.relative_to(ROOT)):old.sha(q) for q in files}))
    log('Validation2139 evaluation complete; no Test.')
    return result
