"""Local-only inference from pinned, trusted project checkpoints. No user paths/network."""
from pathlib import Path
from itertools import combinations
from functools import lru_cache
import hashlib
import json
import os
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
CLASSES = np.array(['fake', 'real'])
MAX_CHARS = 20000


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def registry():
    path = APP / 'model_registry.json'
    release = json.loads((APP / 'release.json').read_text(encoding='utf-8'))
    if digest(path) != release['registry_sha256']:
        raise ValueError('Danh mục mô hình không khớp checksum bản bàn giao.')
    return json.loads(path.read_text(encoding='utf-8'))


def checked_path(relative, hashes):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or relative not in hashes:
        raise ValueError('Artifact nằm ngoài danh mục được phép.')
    if not path.is_file() or digest(path) != hashes[relative]:
        raise ValueError('Checkpoint thiếu hoặc không khớp checksum; mô hình đã bị vô hiệu hóa.')
    return path


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Hãy nhập văn bản tiếng Anh không rỗng.')
    if len(text) > MAX_CHARS:
        raise ValueError(f'Văn bản vượt giới hạn {MAX_CHARS:,} ký tự; hãy chia nhỏ trước khi dự đoán.')
    if '\x00' in text:
        raise ValueError('Văn bản chứa ký tự điều khiển không hợp lệ.')
    return text


@lru_cache(maxsize=12)
def load_base(model_id):
    reg = registry()
    spec = reg['bases'][model_id]
    hashes = spec['files']
    for p in hashes: checked_path(p, hashes)
    if spec['kind'] == 'sklearn':
        import joblib
        model = joblib.load(checked_path(spec['pipeline'], hashes))
        assert list(model.classes_) == list(CLASSES)
        cal = joblib.load(checked_path(spec['calibrator'], hashes)) if spec.get('calibrator') else None
        if cal is not None: assert list(cal.classes_) == list(CLASSES)
        return model, cal, None
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import joblib
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(ROOT/spec['tokenizer'],local_files_only=True,trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(ROOT/spec['checkpoint'],
        local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager').eval()
    assert {int(k):v for k,v in model.config.id2label.items()} == {0:'fake',1:'real'}
    cal = joblib.load(checked_path(spec['calibrator'], hashes)) if spec.get('calibrator') else None
    return model,cal,tok


def probability_ok(p):
    p = np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1):
        raise ValueError('Mô hình trả xác suất không hợp lệ.')
    return p


@lru_cache(maxsize=16)
def checked_unified(model_id):
    spec=registry()['bases'][model_id]
    for p in spec['files']:checked_path(p,spec['files'])


def base_predict(model_id, texts, calibrated):
    from threadpoolctl import threadpool_limits
    reg = registry(); spec=reg['bases'][model_id]
    if spec['kind'] in {'ctbert_original','ctbert_url_emoji'}:
        import joblib
        if spec['kind']=='ctbert_original':
            import ctbert_development as pipeline
        else:
            import ctbert_tweet_preprocess as pipeline
        other_name='ctbert_tweet_preprocess' if spec['kind']=='ctbert_original' else 'ctbert_development'
        other=sys.modules.get(other_name)
        if other is not None and other.load.cache_info().currsize:
            import torch
            other.load.cache_clear()
            torch.cuda.empty_cache()
        checked_unified(model_id)
        count=len(texts)
        inputs=list(texts)+[texts[-1]]*((-count)%4)
        raw,score,_=pipeline.raw(pipeline.KEY,inputs)
        if calibrated:
            cal=joblib.load(checked_path(spec['calibrator'],spec['files']))
            assert list(cal.classes_)==list(CLASSES)
            raw=cal.predict_proba(score)
        raw=probability_ok(raw[:count])
        tok=pipeline.load(pipeline.KEY)[1]
        truncated=[len(tok.encode(pipeline.normalize_text(t),truncation=False))>128 for t in texts]
        return {'labels':CLASSES[raw.argmax(1)],'probability':raw,'truncated':truncated}
    if spec['kind']=='unified':
        import unified_coursework as unified
        import joblib,re
        checked_unified(model_id)
        key=spec['unified_key']
        source=unified.read(unified.OUT/key/'base_lock.json')
        count=len(texts)
        inputs=list(texts)
        # Frozen evaluation used FP16 batches of four. Keep that numerical shape
        # for singleton/partial requests; discard repeated padding outputs.
        if source['kind']=='transformer' and count%4:
            inputs.extend([inputs[-1]]*(4-count%4))
        raw,score,_=unified.raw(key,inputs)
        raw=None if raw is None else raw[:count]
        score=score[:count]
        if calibrated:
            cal=joblib.load(checked_path(spec['calibrator'],spec['files']))
            raw=probability_ok(cal.predict_proba(score))
        elif raw is None:raise ValueError('LinearSVC raw margin không phải probability.')
        if source['kind']=='transformer':
            tok=unified.load(key)[1]
            truncated=[len(tok.encode(unified.normalize_text(t),truncation=False))>128 for t in texts]
        elif source['kind']=='graph':truncated=[len(re.findall(r'\b\w+\b',unified.normalize_text(t)))>128 for t in texts]
        else:truncated=[False]*len(texts)
        return {'labels':CLASSES[raw.argmax(1)],'probability':raw,'truncated':truncated}
    if spec['kind']=='gap_controlled':
        if calibrated:raise ValueError('Checkpoint mới chưa có calibration; không dùng calibrator A-fit cũ.')
        return gap_predict(model_id,texts)
    if spec['kind']=='expanded':
        import joblib
        import expanded_coursework as expanded
        model,aux=load_expanded(model_id)
        raw=probability_ok(expanded.raw_predict(spec['expanded_key'],texts,(model,aux)))
        if calibrated:
            cal=joblib.load(checked_path(spec['calibrator'],spec['files']))
            raw=probability_ok(cal.predict_proba(expanded.logits(raw)))
        if spec['expanded_key'] in ['gcn','gat']:
            import re
            truncated=[len(re.findall(r'\b\w+\b',expanded.normalize_text(t)))>128 for t in texts]
        else:truncated=[len(aux.encode(expanded.normalize_text(t),truncation=False))>128 for t in texts]
        return {'labels':CLASSES[raw.argmax(1)],'probability':raw,'truncated':truncated}
    model,cal,tok = load_base(model_id)
    with threadpool_limits(limits=2):
        if spec['kind']=='sklearn':
            if hasattr(model,'predict_proba'):
                raw=probability_ok(model.predict_proba(texts));raw/=raw.sum(1,keepdims=True)
                v=np.clip(raw[:,1],1e-7,1-1e-7);score=np.log(v/(1-v)).reshape(-1,1)
            else:
                raw=None;score=np.asarray(model.decision_function(texts)).reshape(-1,1)
            labels=model.predict(texts)
            truncated=[False]*len(texts)
        else:
            import torch
            from text_preprocessing import normalize_text
            norm=[normalize_text(t) for t in texts]
            lengths=[len(tok.encode(t,add_special_tokens=True,truncation=False)) for t in norm]
            truncated=[n>spec['max_length'] for n in lengths]
            blocks=[]
            with torch.inference_mode():
                for i in range(0,len(texts),8):
                    inputs=tok(norm[i:i+8],padding='max_length',truncation=True,
                               max_length=spec['max_length'],return_tensors='pt')
                    blocks.append(torch.softmax(model(**inputs).logits.float(),1).numpy())
            raw=probability_ok(np.concatenate(blocks));labels=CLASSES[raw.argmax(1)]
            v=np.clip(raw[:,1],1e-7,1-1e-7);score=np.log(v/(1-v)).reshape(-1,1)
        if calibrated:
            if cal is None: raise ValueError('Chưa có calibrator hợp lệ cho mô hình này.')
            p=probability_ok(cal.predict_proba(score));labels=CLASSES[p.argmax(1)]
        else:p=raw
    return {'labels':np.asarray(labels),'probability':p,'truncated':truncated}


@lru_cache(maxsize=1)
def load_expanded(model_id):
    """Keep one new neural model in memory; no concurrent GPU model collection."""
    import expanded_coursework as expanded
    spec=registry()['bases'][model_id]
    for p in spec['files']:checked_path(p,spec['files'])
    return expanded.load_new(spec['expanded_key'])


@lru_cache(maxsize=1)
def load_gap(model_id):
    """Load exactly one verified inner-fit model; no legacy calibrator reuse."""
    import torch
    spec=registry()['bases'][model_id]
    for p in spec['files']:checked_path(p,spec['files'])
    torch.set_num_threads(2)
    if spec['model_key'] in ['gcn','gat']:
        from gap_controlled_graphs import Graph
        from safetensors.torch import load_file
        vocab=json.loads((ROOT/spec['vocab']).read_text(encoding='utf-8'))
        model=Graph(spec['model_key'],len(vocab)+2)
        model.load_state_dict(load_file(str(ROOT/spec['checkpoint'])))
        return model.eval(),vocab
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    path=ROOT/spec['checkpoint']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager')
    assert {int(k):v for k,v in model.config.id2label.items()}=={0:'fake',1:'real'}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(),tok


def gap_predict(model_id,texts):
    import torch
    from text_preprocessing import normalize_text
    spec=registry()['bases'][model_id];model,aux=load_gap(model_id);blocks=[]
    if spec['model_key'] in ['gcn','gat']:
        from expanded_coursework import graph_data
        from torch_geometric.loader import DataLoader
        import re
        data=[graph_data(t,0,aux) for t in texts]
        with torch.inference_mode():
            for batch in DataLoader(data,batch_size=32,shuffle=False):blocks.append(model(batch).softmax(1).numpy())
        truncated=[len(re.findall(r'\b\w+\b',normalize_text(t)))>128 for t in texts]
    else:
        norm=[normalize_text(t) for t in texts];device=next(model.parameters()).device
        truncated=[len(aux.encode(t,truncation=False))>128 for t in norm]
        with torch.inference_mode():
            for i in range(0,len(texts),4):
                inputs=aux(norm[i:i+4],padding='max_length',truncation=True,max_length=128,return_tensors='pt').to(device)
                with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):z=model(**inputs).logits
                blocks.append(z.float().softmax(1).cpu().numpy())
    raw=probability_ok(np.concatenate(blocks))
    return {'labels':CLASSES[raw.argmax(1)],'probability':raw,'truncated':truncated}


def pool(probability, weights):
    p=np.asarray(probability,dtype=float);w=np.asarray(weights,dtype=float)
    assert p.ndim==3 and p.shape[2]==2
    n=p.shape[1];basis=[p[:,i] for i in range(n)]
    if len(w)>n:basis += [np.minimum(p[:,i],p[:,j]) for i,j in combinations(range(n),2)]
    assert len(w)==len(basis) and (w>=-1e-12).all() and np.isclose(w.sum(),1)
    score=np.einsum('nkc,k->nc',np.stack(basis,axis=1),w)
    z=score.sum(1,keepdims=True)
    return probability_ok(np.divide(score,z,out=np.full_like(score,.5),where=z>1e-15))


def options():
    reg=registry();choices={};unavailable={}
    for model_id,spec in reg['models'].items():
        try:
            members=spec.get('members',[spec.get('base')])
            for member in members:
                b=reg['bases'][member]
                for p in b['files']:checked_path(p,b['files'])
            choices[model_id]=spec['label']
        except (OSError,ValueError,KeyError):
            unavailable[model_id]='Artifact chưa sẵn sàng hoặc checksum không hợp lệ.'
    return choices,unavailable


def predict(model_id, texts):
    if not isinstance(texts,list) or not 1<=len(texts)<=64:raise ValueError('Batch phải có 1–64 văn bản.')
    texts=[validate_text(t) for t in texts]
    reg=registry()
    if model_id not in reg['models']:raise ValueError('Mô hình không nằm trong danh mục cho phép.')
    spec=reg['models'][model_id];tick=time.perf_counter()
    if spec['kind']=='base':
        result=base_predict(spec['base'],texts,spec['calibrated']);members={}
    else:
        members={m:base_predict(m,texts,True) for m in spec['members']}
        p=pool(np.stack([members[m]['probability'] for m in spec['members']],axis=1),spec['weights'])
        result={'labels':CLASSES[p.argmax(1)],'probability':p,
                'truncated':[any(v['truncated'][i] for v in members.values()) for i in range(len(texts))]}
    result.update(members=members,seconds=time.perf_counter()-tick,protocol=spec['protocol'],
                  label=spec['label'],weights=spec.get('weights'))
    return result
