"""Bounded coursework stages, called visibly by Notebook12. No I/O at import."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import time
import warnings
import platform
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/20261020_coursework_completion_v1'
CFG = ROOT / 'configs/coursework_completion_v1.json'
PILOT = ROOT / 'results/20261018_h1_h2_pilot_seed42_01'
PREP = PILOT / '00_prepare_attempt_a984c403'
PRETRAINED = ROOT / 'models/pretrained/distilbert_12040accade4e8a0f71eabdb258fecc2e7e948be'
MODELS = ROOT / 'models/20261020_coursework_completion_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def log(message):
    print(message, flush=True)
    with (OUT / 'run.log').open('a', encoding='utf-8') as f:
        f.write(datetime.now().astimezone().isoformat() + ' ' + message + '\n')


def old(stage):
    marker = read(PILOT / (stage + '_COMPLETE.json'))
    for name, expected in marker['files'].items():
        assert 'test' not in name.lower() and sha(ROOT / name) == expected, name
    return ROOT / marker['folder']


def preflight():
    import psutil
    import shutil
    import pandas as pd
    if (OUT / 'preflight.json').exists():
        assert read(OUT / 'preflight.json')['config_sha256'] == sha(CFG)
        return read(OUT / 'hardware.json')
    OUT.mkdir(exist_ok=False)
    config = read(CFG)
    assert not config['test_access'] and config['seed'] == 42
    prepared = old('00_prepare')
    a = pd.read_csv(prepared / 'assignment.csv')
    assert len(a) == 6420 and a.sample_id.is_unique
    # Structural coverage is validated without fitting or selecting on held-out data.
    inventory = []
    for d in ['docs', 'prompts', 'notebooks']:
        for p in sorted((ROOT / d).glob('*')):
            if p.suffix in ['.md', '.ipynb'] and 'test' not in p.name.lower():
                body = p.read_text(encoding='utf-8-sig')
                item = {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}
                if p.suffix == '.ipynb':
                    nb = json.loads(body)
                    item.update(code_cells=sum(c['cell_type']=='code' for c in nb['cells']),
                                errors=sum(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[])))
                inventory.append(item)
    hardware = {'python': platform.python_version(), 'cpu_count': psutil.cpu_count(),
                'ram_total': psutil.virtual_memory().total, 'ram_available': psutil.virtual_memory().available,
                'disk_free': shutil.disk_usage(ROOT).free,
                'versions': {p:importlib.metadata.version(p) for p in ['torch','transformers','streamlit','numpy','pandas','scikit-learn','scipy']}}
    save(OUT / 'hardware.json', hardware)
    save(OUT / 'inventory.json', inventory)
    save(OUT / 'preflight.json', {'config_sha256': sha(CFG), 'assignment_sha256': sha(prepared/'assignment.csv'),
                                'code_sha256': sha(Path(__file__)), 'test_access': False})
    save(OUT / 'state_prepared.json', {'status':'PREPARED','next':'download pinned public checkpoint, then visible Notebook12'})
    log('PREFLIGHT PASS: inherited assignment checked; no training/Test')
    return hardware


def fetch():
    """Data-only pinned safetensors download; never import remote model code."""
    import urllib.request
    cfg = read(CFG)
    if (PRETRAINED / 'provenance.json').exists():
        for name, h in read(PRETRAINED/'provenance.json')['files'].items():
            assert sha(PRETRAINED/name) == h
        return str(PRETRAINED)
    PRETRAINED.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ['config.json','vocab.txt','tokenizer_config.json','tokenizer.json','model.safetensors']:
        p = PRETRAINED/name
        if not p.exists():
            url = f"https://huggingface.co/{cfg['model_id']}/resolve/{cfg['revision']}/{name}"
            # Incomplete attempts remain; a new transfer never overwrites one.
            tmp = PRETRAINED / (name + '.part_' + str(time.time_ns()))
            with urllib.request.urlopen(url,timeout=90) as response, tmp.open('xb') as f:
                total = 0
                while True:
                    block = response.read(1024*1024)
                    if not block: break
                    total += len(block)
                    assert total <= 300_000_000
                    f.write(block)
            if name == 'model.safetensors': assert sha(tmp) == cfg['model_sha256']
            tmp.rename(p)
        files[name] = sha(p)
        log('CHECKPOINT ' + name + ' ' + str(p.stat().st_size) + ' bytes')
    assert files['model.safetensors'] == cfg['model_sha256']
    save(PRETRAINED/'provenance.json', {'model_id':cfg['model_id'],'revision':cfg['revision'],
        'license':cfg['license'],'files':files,'trust_remote_code':False,'source':'publisher HF model repository'})
    return str(PRETRAINED)


def partition(role):
    import pandas as pd
    assert role in ['A','C','M','H']
    return pd.read_csv(PREP/f'partition_{role}.csv')


def train():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    import gc
    import random
    import shutil
    import numpy as np
    import pandas as pd
    import psutil
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import f1_score
    from text_preprocessing import normalize_text
    cfg = read(CFG)
    assert read(OUT/'preflight.json')['config_sha256'] == sha(CFG)
    if (OUT/'base_lock.json').exists():
        lock = read(OUT/'base_lock.json')
        for p,h in lock['files'].items(): assert sha(ROOT/p)==h
        log('BASE already complete; reused')
        return lock
    assert not MODELS.exists(), 'Partial training preserved; inspect saved epoch/state before any restart'
    assert torch.cuda.is_available()
    free,total = torch.cuda.mem_get_info()
    assert free >= cfg['minimum_free_vram_bytes']
    assert psutil.virtual_memory().available >= cfg['minimum_available_ram_bytes']
    assert shutil.disk_usage(ROOT).free >= cfg['minimum_disk_free_bytes']
    for p,h in read(PRETRAINED/'provenance.json')['files'].items(): assert sha(PRETRAINED/p)==h
    MODELS.mkdir()
    save(OUT/'gpu_before_training.json', {'name':torch.cuda.get_device_name(0),'free':free,'total':total,
        'ram_available':psutil.virtual_memory().available})
    a = partition('A')
    assert len(a)==4494
    assert 'group_id' in a.columns
    itrain,idev = next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=42).split(a,groups=a.group_id))
    assert not set(a.iloc[itrain].group_id)&set(a.iloc[idev].group_id)
    assert abs((a.iloc[idev].label=='fake').mean()-(a.label=='fake').mean())<.05
    assignment = a[['sample_id','group_id','label']].copy()
    assignment['inner_role']='fit';assignment.loc[idev,'inner_role']='dev'
    assignment.to_csv(OUT/'A_inner_assignment.csv',index=False,mode='x')
    tok = AutoTokenizer.from_pretrained(PRETRAINED,local_files_only=True,trust_remote_code=False)
    tok.save_pretrained(MODELS/'tokenizer')
    texts = a.tweet.map(normalize_text).tolist()
    enc = tok(texts,truncation=True,max_length=cfg['max_length'],padding='max_length',return_tensors='pt')
    labels = torch.tensor((a.label=='real').astype(int).to_numpy(),dtype=torch.long)
    class Data(torch.utils.data.Dataset):
        def __init__(self, ix): self.ix=list(ix)
        def __len__(self): return len(self.ix)
        def __getitem__(self,k):
            i=self.ix[k];return {**{n:t[i] for n,t in enc.items()},'labels':labels[i]}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    def seed():
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    start=time.perf_counter(); history=[]; peakrss=0
    def evaluate(model, ix):
        model.eval();values=[];loss=0
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(Data(ix),batch_size=cfg['eval_batch_size']):
                b={k:v.cuda() for k,v in b.items()}
                with torch.autocast('cuda',dtype=torch.float16):o=model(**b)
                loss+=o.loss.item()*len(b['labels']);values.extend(o.logits.argmax(1).cpu().tolist())
        return loss/len(ix),float(f1_score(labels[list(ix)].numpy(),values,average='weighted',zero_division=0))
    def phase(name, fitix, epochs, devix=None):
        nonlocal peakrss
        seed(); torch.cuda.reset_peak_memory_stats()
        model=AutoModelForSequenceClassification.from_pretrained(PRETRAINED,local_files_only=True,
            trust_remote_code=False,use_safetensors=True,num_labels=2,id2label={0:'fake',1:'real'},
            label2id={'fake':0,'real':1},attn_implementation='eager').cuda()
        loader=torch.utils.data.DataLoader(Data(fitix),batch_size=cfg['batch_size'],shuffle=True,
            generator=torch.Generator().manual_seed(42),num_workers=0)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=get_linear_schedule_with_warmup(opt,int(len(loader)*epochs*cfg['warmup_ratio']),len(loader)*epochs)
        scaler=torch.amp.GradScaler('cuda')
        for ep in range(1,epochs+1):
            tick=time.perf_counter(); model.train();total_loss=0
            for step,b in enumerate(loader,1):
                if time.perf_counter()-start>cfg['max_training_seconds']: raise TimeoutError('60 minute training cap')
                b={k:v.cuda() for k,v in b.items()};opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):loss=model(**b).loss
                assert torch.isfinite(loss)
                scaler.scale(loss).backward();scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'])
                prev=scaler.get_scale();scaler.step(opt);scaler.update()
                if scaler.get_scale()>=prev:scheduler.step()
                total_loss+=loss.item()*len(b['labels'])
                if name=='selection' and ep==1 and step==20:
                    elapsed=time.perf_counter()-tick
                    estimate=elapsed/20*len(loader)*cfg['selection_epochs']*2
                    save(OUT/'training_probe.json',{'measured_20_steps_seconds':elapsed,
                        'estimated_selection_plus_refit_seconds_excludes_eval_save':estimate,
                        'cuda_allocated':torch.cuda.memory_allocated(),'cuda_peak_allocated':torch.cuda.max_memory_allocated()})
                    if estimate>cfg['max_training_seconds']*.85:
                        raise TimeoutError('Probe estimates training exceeds safe 60min budget')
                if step%100==0:log(f'{name} epoch{ep} step{step}/{len(loader)} loss{loss.item():.4f}')
            tl,tf=evaluate(model,fitix)
            dl,df=evaluate(model,devix) if devix is not None else (None,None)
            checkpoint=MODELS/f'{name}_epoch_{ep:02d}';model.save_pretrained(checkpoint,safe_serialization=True)
            torch.save({'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'epoch':ep,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint/'training_state.pt')
            peakrss=max(peakrss,psutil.Process().memory_info().rss)
            row={'phase':name,'epoch':ep,'optimization_loss':total_loss/len(fitix),'train_loss':tl,
                 'train_weighted_f1':tf,'inner_dev_loss':dl,'inner_dev_weighted_f1':df,
                 'epoch_seconds':time.perf_counter()-tick,'peak_cuda_allocated':torch.cuda.max_memory_allocated()}
            history.append(row);save(OUT/f'{name}_epoch_{ep:02d}.json',row)
            log(f'{name} epoch{ep} train_loss={tl:.5f} train_F1={tf:.6f} inner_dev_F1={df}')
        del model,opt,scheduler,scaler;gc.collect();torch.cuda.empty_cache()
    phase('selection',itrain,cfg['selection_epochs'],idev)
    best=max((h for h in history if h['phase']=='selection'),key=lambda r:(r['inner_dev_weighted_f1'],-r['epoch']))['epoch']
    save(OUT/'epoch_selection_lock.json',{'selected_epochs':best,'selection_data':'inner-A only','config_sha256':sha(CFG)})
    phase('refit_A',range(len(a)),best)
    pd.DataFrame(history).to_csv(OUT/'history.csv',index=False,mode='x')
    checkpoint=MODELS/f'refit_A_epoch_{best:02d}'
    files={str(p.relative_to(ROOT)):sha(p) for d in [checkpoint,MODELS/'tokenizer'] for p in d.glob('*') if p.suffix!='.pt'}
    lock={'checkpoint':str(checkpoint.relative_to(ROOT)),'tokenizer':str((MODELS/'tokenizer').relative_to(ROOT)),
          'selected_epochs':best,'training_seconds_selection_plus_refit':time.perf_counter()-start,
          'peak_process_rss_observed_epoch_boundaries':peakrss,'files':files,'test_access':False}
    save(OUT/'base_lock.json',lock)
    log('BASE LOCKED before C/M/H/Validation evaluation')
    return lock


def finish():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize
    from sklearn.metrics import log_loss
    from text_preprocessing import normalize_text
    from validation_cohorts import select_validation_cohorts
    import pilot_h1_h2 as pilot
    cfg=read(CFG)
    if (OUT/'COMPLETE.json').exists(): return pd.read_csv(OUT/'metrics.csv')
    lock=read(OUT/'base_lock.json')
    for p,h in lock['files'].items():assert sha(ROOT/p)==h
    assert not (OUT/'calibrator.joblib').exists(), 'Partial evaluation retained; inspect before retry'
    tok=AutoTokenizer.from_pretrained(ROOT/lock['tokenizer'],local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(ROOT/lock['checkpoint'],local_files_only=True,
        trust_remote_code=False,use_safetensors=True,attn_implementation='eager').cuda().eval()
    timers=[]
    def proba(data,role):
        tick=time.perf_counter();p=[]
        with torch.no_grad():
            for start in range(0,len(data),cfg['eval_batch_size']):
                x=tok(data.tweet.iloc[start:start+cfg['eval_batch_size']].map(normalize_text).tolist(),
                    padding='max_length',truncation=True,max_length=cfg['max_length'],return_tensors='pt').to('cuda')
                with torch.autocast('cuda',dtype=torch.float16):z=model(**x).logits
                p.append(torch.softmax(z.float(),dim=1).cpu().numpy())
        timers.append({'stage':'base_inference_including_normalize_tokenize_transfer','cohort':role,
                       'n':len(data),'seconds':time.perf_counter()-tick,'device':'CUDA fp16'})
        return np.concatenate(p).astype(float)
    def scores(p):
        v=np.clip(p[:,1],1e-7,1-1e-7);return np.log(v/(1-v)).reshape(-1,1)
    c,m=partition('C'),partition('M')
    pc=proba(c,'C');pm=proba(m,'M')
    tick=time.perf_counter();cal=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(scores(pc),c.label)
    assert list(cal.classes_)==['fake','real'];cal_seconds=time.perf_counter()-tick
    joblib.dump(cal,OUT/'calibrator.joblib')
    pcal=cal.predict_proba(scores(pm))
    svm_m=pd.read_csv(old('02_calibration_word_svm')/'calibrated_M_predictions.csv').set_index('sample_id').loc[m.sample_id]
    assert np.array_equal(svm_m.true_label,m.label)
    stacked=np.stack([svm_m[['prob_fake','prob_real']].to_numpy(),pcal],axis=1)
    params={'equal_mean':{'weights':[.5,.5],'fit_seconds':0.},
            'normalized_pair_min':{'weights':[0.,0.,1.],'fit_seconds':0.}}
    for name,k in [('weighted_mean',2),('learned_choquet',3)]:
        tick=time.perf_counter()
        res=minimize(lambda w:log_loss(m.label,pilot.pool(stacked,w),labels=['fake','real']),
            np.array([.5,.5]+([0.] if k==3 else [])),method='SLSQP',bounds=[(0,1)]*k,
            constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300,'ftol':1e-10})
        assert res.success,res.message
        w=np.maximum(res.x,0);w/=w.sum();params[name]={'weights':w.tolist(),'fit_seconds':time.perf_counter()-tick}
    save(OUT/'meta_lock.json',{'B3':params,'members':cfg['B3_members'],'calibration_seconds':cal_seconds,
        'calibrator_sha256':sha(OUT/'calibrator.joblib'),'fit_partitions':'C calibration,M meta only',
        'base_lock_sha256':sha(OUT/'base_lock.json'),'config_sha256':sha(CFG)})
    log('CALIBRATION/META LOCKED before H/Validation labels read for evaluation')
    rows=[];allbins=[]
    def record(data,p,name,cohort):
        f=data[['sample_id','label']].rename(columns={'label':'true_label'}).copy()
        f['predicted_label']=np.array(['fake','real'])[p.argmax(1)];f['prob_fake']=p[:,0];f['prob_real']=p[:,1]
        f.to_csv(OUT/f'{name}__{cohort}_predictions.csv',index=False,mode='x')
        met,bins=pilot.evaluate(f);rows.append({**met,'model':name,'cohort':cohort,'protocol':'A-fit','evaluation_status':'exploratory/development'})
        bins['model']=name;bins['cohort']=cohort;allbins.append(bins)
    record(m,pm,'distilbert_raw','M_FIT_DIAGNOSTIC');record(m,pcal,'distilbert_calibrated','M_FIT_DIAGNOSTIC')
    svm=joblib.load(old('01_base_word_svm')/'pipeline.joblib')
    svmcal=joblib.load(old('02_calibration_word_svm')/'calibrator.joblib')
    h=partition('H')
    raw=ROOT/'data/raw/constraint2021/030deba36d118c124709fd72d7d5f14c7bc69053/Constraint_Val.csv'
    assert sha(raw)=='c610931d0641fe1d9b9495c453f99ae7aa45bf294c7db6333ec84ecebb0b6c3c'
    v=pd.read_csv(raw,dtype=str);v['sample_id']='validation:'+v.id
    for role,data in [('H_exploratory',h),('benchmark_original',v)]:
        p=proba(data,role);tick=time.perf_counter();pcal=cal.predict_proba(scores(p))
        timers.append({'stage':'sigmoid_inference','cohort':role,'n':len(data),'seconds':time.perf_counter()-tick,'device':'CPU'})
        tick=time.perf_counter();svmp=svmcal.predict_proba(pilot._prob_and_score(svm,data.tweet)[1])
        timers.append({'stage':'SVM_pipeline_plus_calibration','cohort':role,'n':len(data),'seconds':time.perf_counter()-tick,'device':'CPU'})
        probs={'distilbert_raw':p,'distilbert_calibrated':pcal}
        stack=np.stack([svmp,pcal],axis=1)
        for name,item in params.items():
            tick=time.perf_counter();probs['B3_'+name]=pilot.pool(stack,item['weights'])
            timers.append({'stage':'aggregation_'+name,'cohort':role,'n':len(data),'seconds':time.perf_counter()-tick,'device':'CPU cached member probabilities'})
        if role=='H_exploratory':
            for name,q in probs.items():record(data,q,name,role)
        else:
            ref=data[['sample_id','label']].rename(columns={'label':'true_label'})
            for name,q in probs.items():
                f=ref.copy();f['prob_fake']=q[:,0];f['prob_real']=q[:,1];f['predicted_label']=np.array(['fake','real'])[q.argmax(1)]
                for cohort,subset in select_validation_cohorts(f,pilot.REVIEW,approve_proposed=True).items():
                    ix=f.set_index('sample_id').index.get_indexer(subset.sample_id)
                    record(data.set_index('sample_id').loc[subset.sample_id].reset_index(),q[ix],name,cohort)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False,mode='x')
    pd.DataFrame(timers).to_csv(OUT/'timings.csv',index=False,mode='x')
    pd.concat(allbins).to_csv(OUT/'calibration_bins.csv',index=False,mode='x')
    plots()
    save(OUT/'COMPLETE.json',{'status':'COMPLETE','test_access':False,'config_sha256':sha(CFG),
        'files':{str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='run.log'}})
    log('COMPLETE: DistilBERT and fixed B3 comparisons; no Test; no winner selected')
    del model;torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def plots():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    h=pd.read_csv(OUT/'history.csv');s=h[h.phase=='selection']
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for col,dev,title,a in [('train_loss','inner_dev_loss','Loss',ax[0]),('train_weighted_f1','inner_dev_weighted_f1','Weighted-F1',ax[1])]:
        a.plot(s.epoch,s[col],'-o',label='inner-A fit');a.plot(s.epoch,s[dev],'-o',label='inner-A dev')
        a.set(xlabel='Epoch',ylabel=title,title='DistilBERT: '+title);a.legend()
    fig.tight_layout();fig.savefig(OUT/'loss_f1_inner_A.png',dpi=150);plt.close(fig)
    metrics=pd.read_csv(OUT/'metrics.csv')
    for cohort in ['H_exploratory','benchmark_original','sensitivity_exact_approved','sensitivity_verified_proposed']:
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for ax,(_,r) in zip(axes.flat,metrics[metrics.cohort==cohort].iterrows()):
            cm=[[r.cm_fake_fake,r.cm_fake_real],[r.cm_real_fake,r.cm_real_real]]
            ConfusionMatrixDisplay(np.asarray(cm,dtype=int),display_labels=['fake','real']).plot(ax=ax,colorbar=False,cmap='Blues')
            ax.set_title(r.model)
        fig.suptitle(cohort+' — exploratory/development');fig.tight_layout()
        fig.savefig(OUT/f'cm_{cohort}.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    import sys
    assert len(sys.argv)==2 and sys.argv[1] in ['preflight','fetch'], 'Training runs only visibly in Notebook12'
    print(globals()[sys.argv[1]]())
