"""Notebook16: bounded development-only graph retraining; no external evaluation."""
import json, time, sys, platform, traceback
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss, classification_report, confusion_matrix
from safetensors.torch import save_file, load_file
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from expanded_coursework import ROOT, PREP, partition, seed, word_tokens, graph_data, sha

RUN = '20261022_gap_controlled_graphs_v1'
OUT = ROOT / 'results' / RUN
MODEL = ROOT / 'models' / RUN
CONFIG = ROOT / 'configs' / (RUN + '.json')

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def eligible(row):
    return abs(row['train_loss']-row['dev_loss']) < .1 and abs(row['train_f1']-row['dev_f1']) < .1

def rank(row):
    return row['dev_f1'], -row['dev_loss'], -row['epoch']

def prepare():
    import psutil, shutil, importlib.metadata as md
    assignment = PREP / 'assignment.csv'
    inner_path = ROOT / 'results/20261020_coursework_completion_v1/A_inner_assignment.csv'
    assert sha(assignment) == 'cb457e35c0eb68fc8fb95ec8096f0a8db11d0ae4a9c8a7f25a93d77479e8a7b9'
    assert sha(inner_path) == '9e0165ff21cecc6b6888d77ea8d157e5504315bdbbd9fac6e1dfd228b657bd5c'
    full = pd.read_csv(assignment)
    assert len(full) == 6420 and full.sample_id.is_unique
    for i, role in enumerate('ACMH'):
        for other in 'ACMH'[i+1:]:
            x, y = full[full.role==role], full[full.role==other]
            assert not set(x.sample_id)&set(y.sample_id)
            assert not set(x.group_id)&set(y.group_id)
    a = partition('A'); inner = pd.read_csv(inner_path)
    assert set(a.sample_id) == set(inner.sample_id) and inner.sample_id.is_unique
    fit, dev = inner[inner.inner_role=='fit'], inner[inner.inner_role=='dev']
    assert len(fit)==3792 and len(dev)==702
    assert not set(fit.group_id)&set(dev.group_id)
    assert psutil.virtual_memory().available > 1_000_000_000
    assert shutil.disk_usage(ROOT).free > 500_000_000
    cfg = dict(run=RUN, models=['gcn','gat'], seed=42, hidden=32, embedding=32,
        dropout=.5, weight_decay=.05, lr=.0005, batch_size=32, epochs=20,
        patience=5, max_seconds_per_model=3600, total_seconds=7200,
        assignment_sha256=sha(assignment), inner_sha256=sha(inner_path),
        preprocessing='inherited normalize_text and word_tokens; max128; vocabulary fit only on inner-fit',
        selection='both absolute gaps <0.1, then maximum dev weighted-F1, lower dev loss, earliest epoch',
        stopping='five epochs without eligible checkpoint improvement; before first eligible use unconstrained rank for stopping only',
        evaluation='same eval mode, dropout off, sample-mean cross entropy and weighted F1; fake=0 real=1',
        scope='inner-fit checkpoint diagnostic only; no refit A, calibration, ensemble or app replacement',
        test_status='Test seen in prior round; never accessed by this run; no new independent Test claims',
        stability='single seed only; no between-run stability inference')
    if CONFIG.exists():
        assert json.loads(CONFIG.read_text(encoding='utf-8')) == cfg
        assert json.loads((OUT/'preflight.json').read_text())['code_sha256'] == sha(__file__)
    else:
        save(CONFIG,cfg)
        save(OUT/'preflight.json',dict(code_sha256=sha(__file__), config_sha256=sha(CONFIG),
            python=sys.version, platform=platform.platform(), counts={'fit':3792,'dev':702},
            ram_available=psutil.virtual_memory().available,disk_free=shutil.disk_usage(ROOT).free,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            gpu_free_total=list(torch.cuda.mem_get_info()) if torch.cuda.is_available() else None,
            versions={p:md.version(p) for p in ['torch','torch-geometric','numpy','scikit-learn']},
            id_overlap=0,group_overlap=0))
    # Boundary checks: strict '<', absolute gap, and F1-first ranking.
    assert not eligible(dict(train_loss=.1,dev_loss=.2,train_f1=.8,dev_f1=.8))
    assert not eligible(dict(train_loss=.3,dev_loss=.1,train_f1=.8,dev_f1=.8))
    assert eligible(dict(train_loss=.2,dev_loss=.25,train_f1=.85,dev_f1=.8))
    print('PROTOCOL LOCKED — fit 3792 / internal validation 702; no H/Validation/Test read',flush=True)
    return cfg

class Graph(torch.nn.Module):
    def __init__(self,key,n):
        super().__init__()
        self.embedding=torch.nn.Embedding(n,32,padding_idx=0)
        self.c1=GCNConv(32,32) if key=='gcn' else GATConv(32,8,heads=4,dropout=.5)
        self.c2=GCNConv(32,32) if key=='gcn' else GATConv(32,8,heads=4,dropout=.5)
        self.head=torch.nn.Linear(32,2)
    def forward(self,b):
        x=torch.relu(self.c1(self.embedding(b.x),b.edge_index))
        x=torch.nn.functional.dropout(x,.5,self.training)
        x=torch.relu(self.c2(x,b.edge_index))
        return self.head(global_mean_pool(x,b.batch))

def train(key):
    cfg=prepare(); assert key in cfg['models']
    out=OUT/key; dest=MODEL/key
    if (out/'COMPLETE.json').exists():
        done=json.loads((out/'COMPLETE.json').read_text())
        assert all(sha(ROOT/p)==h for p,h in done['files'].items())
        print(key,'already complete; not rerun'); return done
    assert not out.exists(), 'Partial run preserved. Inspect before any retry.'
    out.mkdir(parents=True); dest.mkdir(parents=True)
    start=time.perf_counter(); seed()
    try:
        a=partition('A'); inner=pd.read_csv(ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv').set_index('sample_id').loc[a.sample_id]
        fi=np.flatnonzero(inner.inner_role.to_numpy()=='fit'); di=np.flatnonzero(inner.inner_role.to_numpy()=='dev')
        y=(a.label=='real').astype(int).to_numpy()
        counts=Counter(w for i in fi for w in word_tokens(a.tweet.iloc[i]))
        vocab={w:i+2 for i,w in enumerate(sorted(counts,key=lambda w:(-counts[w],w))[:15000])}
        save(dest/'vocab.json',vocab)
        ds=[graph_data(t,int(label),vocab) for t,label in zip(a.tweet,y)]
        def loader(ids,shuffle=False):
            return DataLoader([ds[i] for i in ids],batch_size=32,shuffle=shuffle,num_workers=0,generator=torch.Generator().manual_seed(42))
        model=Graph(key,len(vocab)+2)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        def evaluate(ids):
            model.eval(); probs=[]; total_loss=0.
            with torch.no_grad():
                for b in loader(ids):
                    z=model(b)
                    total_loss+=torch.nn.functional.cross_entropy(z,b.y,reduction='sum').item()
                    probs.append(z.softmax(1).numpy())
            p=np.concatenate(probs)
            assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
            loss=total_loss/len(ids); f1=float(f1_score(y[ids],p.argmax(1),average='weighted',zero_division=0))
            return loss,f1,p
        history=[]; best=None; unconstrained=None; wait=0
        for epoch in range(1,cfg['epochs']+1):
            tick=time.perf_counter(); model.train()
            for b in loader(fi,True):
                if time.perf_counter()-start>cfg['max_seconds_per_model']: raise TimeoutError('Locked budget exceeded')
                opt.zero_grad(set_to_none=True); z=model(b)
                loss=torch.nn.functional.cross_entropy(z,b.y)
                assert torch.isfinite(loss) and torch.isfinite(z).all()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            tl,tf,_=evaluate(fi); dl,df,_=evaluate(di)
            row=dict(epoch=epoch,train_loss=tl,dev_loss=dl,train_f1=tf,dev_f1=df,loss_gap=abs(tl-dl),f1_gap=abs(tf-df),seconds=time.perf_counter()-tick)
            row['eligible']=eligible(row); history.append(row); save(out/f'epoch_{epoch:02d}.json',row)
            improved=False
            if row['eligible'] and (best is None or rank(row)>rank(best)):
                best=row.copy(); improved=True
                # Each improving checkpoint has a unique name; never overwrite artifacts.
                save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(dest/f'epoch_{epoch:02d}.safetensors'))
            if unconstrained is None or rank(row)>rank(unconstrained):
                unconstrained=row.copy()
                if best is None: improved=True
            wait=0 if improved else wait+1
            print(f'{key} epoch {epoch}: loss {tl:.4f}/{dl:.4f}; F1 {tf:.4f}/{df:.4f}; eligible={row["eligible"]}',flush=True)
            with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(key+' '+json.dumps(row)+'\n')
            if wait>=cfg['patience']: break
        pd.DataFrame(history).to_csv(out/'history.csv',index=False,mode='x')
        result=dict(model=key,status='PASS_GAPS' if best else 'NO_ELIGIBLE_CHECKPOINT',selected=best,
            best_unconstrained_diagnostic=unconstrained,training_seconds=time.perf_counter()-start,
            protocol='inner-fit 3792; dev702 used for selection; not independent evaluation',config_sha256=sha(CONFIG))
        if best:
            checkpoint=dest/f'epoch_{best["epoch"]:02d}.safetensors'
            model.load_state_dict(load_file(str(checkpoint)))
            for name,ids in [('train_inner',fi),('validation_inner',di)]:
                loss,f1,p=evaluate(ids)
                assert abs(loss-best['train_loss' if name=='train_inner' else 'dev_loss'])<1e-7
                frame=pd.DataFrame(dict(sample_id=a.sample_id.iloc[ids].to_numpy(),true_label=a.label.iloc[ids].to_numpy(),predicted_label=np.array(['fake','real'])[p.argmax(1)],prob_fake=p[:,0],prob_real=p[:,1]))
                frame.to_csv(out/(name+'_predictions.csv'),index=False,mode='x')
                save(out/(name+'_metrics.json'),dict(loss=loss,weighted_f1=f1,n=len(ids),report=classification_report(y[ids],p.argmax(1),labels=[0,1],target_names=['fake','real'],output_dict=True,zero_division=0),confusion_matrix=confusion_matrix(y[ids],p.argmax(1),labels=[0,1]).tolist()))
            result['checkpoint']=str(checkpoint.relative_to(ROOT))
        result['files']={str(p.relative_to(ROOT)):sha(p) for folder in [out,dest] for p in folder.glob('*') if p.is_file()}
        save(out/'COMPLETE.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return result
    except Exception:
        save(out/'FAILED.json',dict(traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)); raise

def report():
    import matplotlib.pyplot as plt
    rows=[]
    for key in ['gcn','gat']:
        d=json.loads((OUT/key/'COMPLETE.json').read_text()); rows.append({k:v for k,v in d.items() if k not in ['files','best_unconstrained_diagnostic']})
        h=pd.read_csv(OUT/key/'history.csv'); fig,axes=plt.subplots(1,2,figsize=(10,3.4))
        for ax,metric in zip(axes,['loss','f1']):
            ax.plot(h.epoch,h['train_'+metric],label='Inner Train'); ax.plot(h.epoch,h['dev_'+metric],label='Inner Validation')
            if d['selected']:ax.axvline(d['selected']['epoch'],ls='--',color='black',label='Selected epoch')
            ax.set(xlabel='Epoch',ylabel='Cross-entropy' if metric=='loss' else 'Weighted-F1',ylim=(0,1),title=key.upper()); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); path=OUT/key/'learning_curves.png'
        if not path.exists():fig.savefig(path,dpi=650)
        plt.show(); plt.close(fig)
    if not (OUT/'summary.json').exists():save(OUT/'summary.json',rows)
    return pd.DataFrame([dict(model=r['model'],status=r['status'],seconds=r['training_seconds'],**(r['selected'] or {})) for r in rows])
