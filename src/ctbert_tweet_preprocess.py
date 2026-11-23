"""Notebook 21: one seed-42 CT-BERT preprocessing comparison."""
from ctbert_development import *
import ctbert_development as prior
import re
import emoji
from text_preprocessing import normalize_text as original_normalize

RUN='20261110_ctbert_tweet_preprocess_v2'
OUT=ROOT/'results'/RUN
MODELS=ROOT/'models'/RUN
CFG=ROOT/'configs'/(RUN+'.json')

def normalize_text(text):
    text=re.sub(r'https?://[^\s<>]+|www\.[^\s<>]+',lambda m: ' $URL$ '+m[0][len(m[0].rstrip('.,!?;:)]}')):],text,flags=re.I)
    return original_normalize(emoji.demojize(text,delimiters=(' ', ' ')))

def log(s):
    OUT.mkdir(parents=True,exist_ok=True)
    print(s,flush=True)
    with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+s+'\n')

def prepare():
    source=read(prior.CFG)
    prior.verify(prior.read(prior.OUT/'development_lock.json'))
    prior.verify(prior.read(prior.OUT/'COMPLETE.json'))
    assert old.sha(old.PREP/'assignment.csv')==source['assignment_sha256']
    assert old.sha(ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv')==source['inner_sha256']
    assert source['seed']==42
    assert normalize_text('Not safe! See https://t.co/AB.')=='not safe! see $url$ .'
    assert normalize_text('Works? '+chr(0x1f44d))=='works? thumbs_up'
    assert normalize_text('@a #COVID not 12.5%')=='@a #covid not 12.5%'
    assert shutil.disk_usage(ROOT).free>6_000_000_000
    cfg=dict(source,run=RUN,code_sha256=old.sha(__file__),reference_protocol_sha256=old.sha(prior.CFG),
        preprocessing='URL placeholder, English emoji aliases, original NFKC/lowercase/whitespace; preserve mentions and hashtags',
        emoji_version=emoji.__version__,evaluation='Validation2139 only; compare to locked CT-BERT seed42; no Test')
    if CFG.exists():assert read(CFG)==cfg
    else:save(CFG,cfg)
    if not (OUT/KEY/'base_lock.json').exists():
        assert torch.cuda.is_available()
        import subprocess
        free=torch.cuda.mem_get_info()[0]
        need=prior.read(prior.OUT/KEY/'base_lock.json')['peak_vram_bytes']
        # On Windows CUDA's budget and NVIDIA's physical free-memory counter can differ.
        # Preserve the exact training settings; a CUDA OOM is recorded as a failed run.
        physical=None
        if free<need:
            try:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()
                if len(rows)==1:physical=int(rows[0].strip())*1024**2
            except (OSError,ValueError,subprocess.SubprocessError):
                pass
        log(f'GPU preflight: CUDA free={free/1024**2:.0f} MiB; NVIDIA physical free={None if physical is None else physical//1024**2} MiB; prior peak={need/1024**2:.0f} MiB')
        if max(free,physical or 0)<need:
            raise RuntimeError('Insufficient GPU memory against prior measured peak; training not started.')
    log('Seed 42 and preprocessing protocol locked; no Test.')
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



def finish():
    target=OUT/'COMPLETE.json'
    if target.exists():verify(read(target));return pd.read_csv(OUT/'metrics.csv')
    prepare()
    base=OUT/KEY/'base_lock.json'
    verify(read(base))
    c=old.partition('C')
    assert len(c)==642 and c.sample_id.is_unique
    _,s,_=raw(KEY,c.tweet.tolist())
    cal=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(s,c.label)
    assert list(cal.classes_)==['fake','real']
    path=OUT/'calibrator.joblib';assert not path.exists();joblib.dump(cal,path)
    locks=[Path(__file__),CFG,base,path]
    save(OUT/'evaluation_lock.json',dict(status='FROZEN_BEFORE_VALIDATION',fit_roles='3792 train / 702 inner selection / C calibration',
        files={str(q.relative_to(ROOT)):old.sha(q) for q in locks}))
    valpath=pm.RAW/'Constraint_Val.csv';assert old.sha(valpath)==pm.VAL_SHA
    v=pd.read_csv(valpath,dtype=str);v['sample_id']='validation:'+v.id;v=v[v.id!='910']
    assert len(v)==2139 and v.sample_id.is_unique
    baseline=pd.read_csv(prior.OUT/'ctbert_predictions.csv')
    previous=prior.aligned(v,baseline)
    _,s,secs=raw(KEY,v.tweet.tolist())
    prediction=cal.predict_proba(s)
    rows=[]
    for name,prob in [('original',previous),('url_emoji',prediction)]:
        f=frame(v,prob);met,bins=pm.evaluate(f);rows.append(dict(model=name,**met))
        f.to_csv(OUT/f'{name}_predictions.csv',index=False,mode='x')
        bins.to_csv(OUT/f'{name}_calibration_bins.csv',index=False,mode='x')
    result=pd.DataFrame(rows);result.to_csv(OUT/'metrics.csv',index=False,mode='x')
    save(target,dict(status='COMPLETE',n=len(v),inference_seconds=secs,test_access=False,
        files={str(q.relative_to(ROOT)):old.sha(q) for q in OUT.glob('*') if q.is_file() and q.name!='run.log'}))
    return result
