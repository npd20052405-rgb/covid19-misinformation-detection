"""Visible Notebook13 only: inductive document graphs and bounded A-fit models."""
from pathlib import Path
import os,json,time,gc,random,traceback,hashlib
import numpy as np
import pandas as pd
import torch
from expanded_model_download import SPECS,sha
from coursework_completion_v1 import partition,ROOT,PREP
from text_preprocessing import normalize_text

RUN='20261021_expanded_coursework_v1'
OUT=ROOT/'results'/RUN
MODELS=ROOT/'models'/RUN
CFG=ROOT/'configs'/f'{RUN}.json'
NAMES={'gcn':'GCN · đồ thị văn bản','gat':'GAT · đồ thị văn bản','bert':'BERT-Base','roberta':'RoBERTa-Base','gpt2':'GPT-2 · phân loại','t5':'T5-Small · phân loại','tinyllama':'TinyLlama-1.1B · QLoRA'}
CLASSES=np.array(['fake','real'])

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def log(s):
 print(s,flush=True)
 with (OUT/'run.log').open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S ')+s+'\n')
def seed():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 # Scatter reductions on CUDA can be nondeterministic; graph uses CPU for repeatability.
 torch.use_deterministic_algorithms(True)

def prepare():
 import psutil,shutil,importlib.metadata as md
 OUT.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
 if CFG.exists():return read(CFG)
 a=pd.read_csv(PREP/'assignment.csv');inner=pd.read_csv(ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv')
 assert sha(PREP/'assignment.csv')=='cb457e35c0eb68fc8fb95ec8096f0a8db11d0ae4a9c8a7f25a93d77479e8a7b9'
 assert len(a)==6420 and a.sample_id.is_unique
 rolecol=next(c for c in ['partition','role','split'] if c in a)
 roles={r:a[a[rolecol]==r] for r in 'ACMH'}
 assert [len(roles[r]) for r in 'ACMH']==[4494,642,642,642]
 for i,r in enumerate('ACMH'):
  for s in 'ACMH'[i+1:]:
   assert not set(roles[r].sample_id)&set(roles[s].sample_id)
   assert not set(roles[r].group_id)&set(roles[s].group_id)
 assert set(inner.sample_id)==set(roles['A'].sample_id)
 assert not set(inner[inner.inner_role=='fit'].group_id)&set(inner[inner.inner_role=='dev'].group_id)
 config={'run':RUN,'seed':42,'models':list(NAMES),'selection':'max inner-A weighted F1; earliest tie; fresh refit entire A',
 'inner_assignment':str((ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv').relative_to(ROOT)),
 'assignment_sha256':sha(PREP/'assignment.csv'),'inner_sha256':sha(ROOT/'results/20261020_coursework_completion_v1/A_inner_assignment.csv'),
 'preprocessing':'normalize_text NFKC lower whitespace, same inherited pipeline','max_length':128,'max_seconds_per_model':3600,'total_training_cap_seconds':14400,
 'transformer':{'selection_epochs':3,'batch_size':4,'eval_batch_size':8,'lr':2e-5,'weight_decay':.01,'warmup':.1,'gradient_clip':1.,'gradient_checkpointing':True},
 't5':{'variant':'T5ForSequenceClassification, full encoder-decoder EOS hidden state; discriminative 2-class head; not text generation','lr':2e-5},
 'tinyllama':{'variant':'AutoModelForSequenceClassification + NF4 double-quantized frozen base + SEQ_CLS LoRA','selection_epochs':2,'batch_size':2,'lr':2e-4,'r':8,'alpha':16,'dropout':.05,'target_modules':['q_proj','v_proj'],'compute_dtype':'float16'},
 'graph':{'variant':'inductive per-document token-position graph; bidirectional window 2 plus self-loops; no cross-document edges','vocab_max':15000,'max_length':128,'embedding_dim':64,'hidden_dim':64,'layers':2,'dropout':.3,'gat_heads':4,'selection_epochs':12,'batch_size':32,'lr':.001,'device':'CPU deterministic'},
 'calibration':'sigmoid LogisticRegression(C=1,max_iter=2000) on C logit(p_real) only',
 'new_ensemble':'none; preserve pre-existing locked B0/B1/B2/B3; no model/combiner selection on H or Validation',
 'evaluation':'H and Val2139 exploratory; Official Test only after global lock; once; no tuning afterward',
 'stability':'single-seed pilot; epoch trajectory and numeric checks, not proof of multi-seed stability',
 'pretrained':{k:{'repo':s[0],'revision':s[1],'model_sha256':s[2]} for k,s in SPECS.items()}}
 save(CFG,config)
 save(OUT/'preflight.json',{'config_sha256':sha(CFG),'code_sha256':sha(__file__),'partition_counts':{r:len(d) for r,d in roles.items()},'group_overlap':0,'id_overlap':0,
 'gpu':torch.cuda.get_device_name(),'gpu_free_total':torch.cuda.mem_get_info(),'ram_available':psutil.virtual_memory().available,'disk_free':shutil.disk_usage(str(ROOT)).free,
 'versions':{p:md.version(p) for p in ['torch','transformers','peft','bitsandbytes','torch-geometric','numpy','scikit-learn']},'official_test_read':False})
 log('PROTOCOL LOCKED: A/C/M/H and inner-A verified; Test unopened')
 return config

def word_tokens(t):
 import re
 return re.findall(r"\b\w+\b",normalize_text(t))[:128] or ['<empty>']

class GraphClassifier(torch.nn.Module):
 def __init__(self,key,nv):
  super().__init__()
  from torch_geometric.nn import GCNConv,GATConv
  self.embedding=torch.nn.Embedding(nv,64,padding_idx=0)
  self.c1=GCNConv(64,64) if key=='gcn' else GATConv(64,16,heads=4,dropout=.3)
  self.c2=GCNConv(64,64) if key=='gcn' else GATConv(64,16,heads=4,dropout=.3)
  self.head=torch.nn.Linear(64,2)
 def forward(self,b):
  from torch_geometric.nn import global_mean_pool
  x=self.embedding(b.x);x=torch.relu(self.c1(x,b.edge_index));x=torch.nn.functional.dropout(x,.3,self.training)
  x=torch.relu(self.c2(x,b.edge_index));return self.head(global_mean_pool(x,b.batch))

def graph_data(text,label,vocab):
 from torch_geometric.data import Data
 words=word_tokens(text);n=len(words)
 edges=[(i,j) for i in range(n) for j in range(max(0,i-2),min(n,i+3)) if i!=j]
 edge=torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long)
 return Data(x=torch.tensor([vocab.get(w,1) for w in words]),edge_index=edge,y=torch.tensor([label]))

def tokenizer_for(key):
 from transformers import AutoTokenizer
 p=ROOT/'models/pretrained'/f'{key}_{SPECS[key][1]}'
 tok=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 tok.padding_side='right'
 return tok,p

def transformer_for(key,pretrained):
 from transformers import AutoModelForSequenceClassification,BitsAndBytesConfig
 args={'local_files_only':True,'trust_remote_code':False,'use_safetensors':True,'num_labels':2,
       'id2label':{0:'fake',1:'real'},'label2id':{'fake':0,'real':1},'low_cpu_mem_usage':True,'attn_implementation':'eager'}
 if key=='tinyllama':
  from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
  args.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),device_map={'':0},torch_dtype=torch.float16)
  model=AutoModelForSequenceClassification.from_pretrained(pretrained,**args)
  model=prepare_model_for_kbit_training(model)
  model=get_peft_model(model,LoraConfig(task_type='SEQ_CLS',r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],modules_to_save=['score']))
 else:
  model=AutoModelForSequenceClassification.from_pretrained(pretrained,**args)
  # Transformers 4.46 can leave a newly added classification head on meta.
  # Materialize only missing parameters; never reinitialize pretrained weights.
  for module in model.modules():
   missing={n:p for n,p in module._parameters.items() if p is not None and p.is_meta}
   if not missing:continue
   existing={n:p.detach().clone() for n,p in module._parameters.items() if p is not None and not p.is_meta}
   for n,p in missing.items():module._parameters[n]=torch.nn.Parameter(torch.empty(p.shape,dtype=p.dtype),requires_grad=p.requires_grad)
   model._init_weights(module)
   with torch.no_grad():
    for n,p in existing.items():module._parameters[n].copy_(p)
  assert not any(p.is_meta for p in model.parameters())
  model=model.cuda()
 model.config.use_cache=False
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 return model

def train_one(key):
 import psutil
 from sklearn.metrics import f1_score
 from safetensors.torch import save_file
 from transformers import get_linear_schedule_with_warmup
 cfg=read(CFG);assert sha(CFG)==read(OUT/'preflight.json')['config_sha256']
 out=OUT/key;out.mkdir(exist_ok=True);modeldir=MODELS/key
 if (out/'base_lock.json').exists():
  d=read(out/'base_lock.json');assert all(sha(ROOT/p)==h for p,h in d['files'].items());return d
 assert not modeldir.exists() or not any(modeldir.iterdir()),'Partial checkpoint preserved; inspect before resuming'
 used=sum(read(p)['training_seconds'] for p in OUT.glob('*/base_lock.json'))+sum(read(p).get('elapsed_seconds',0) for p in OUT.glob('*/FAILED*.json'))
 assert used<cfg['total_training_cap_seconds']
 a=partition('A');inner=pd.read_csv(ROOT/cfg['inner_assignment']).set_index('sample_id').loc[a.sample_id]
 fi=np.flatnonzero(inner.inner_role.to_numpy()=='fit');di=np.flatnonzero(inner.inner_role.to_numpy()=='dev')
 labels=torch.tensor((a.label=='real').astype(int).to_numpy())
 graph=key in ['gcn','gat'];device='cpu' if graph else 'cuda'
 if not graph:
  assert torch.cuda.mem_get_info()[0]>2_500_000_000
  assert psutil.virtual_memory().available>1_000_000_000,'Insufficient available RAM before model load'
  tok,pretrained=tokenizer_for(key)
  prov=read(pretrained/'provenance.json');assert sha(pretrained/'model.safetensors')==SPECS[key][2]
  enc=tok(a.tweet.map(normalize_text).tolist(),padding='max_length',truncation=True,max_length=128,return_tensors='pt')
 class TextData(torch.utils.data.Dataset):
  def __init__(self,ix):self.ix=list(ix)
  def __len__(self):return len(self.ix)
  def __getitem__(self,k):
   i=self.ix[k];return {**{n:v[i] for n,v in enc.items()},'labels':labels[i]}
 history=[];start=time.perf_counter();modeldir.mkdir(exist_ok=True)
 budget=min(3600,cfg['total_training_cap_seconds']-used)
 def phase(name,ix,epochs,dev=None):
  seed();vocab=None
  if graph:
   from collections import Counter
   from torch_geometric.loader import DataLoader
   count=Counter(w for i in ix for w in word_tokens(a.tweet.iloc[i]))
   words=sorted(count,key=lambda w:(-count[w],w))[:15000]
   vocab={w:i+2 for i,w in enumerate(words)}
   ds=[graph_data(t,int(y),vocab) for t,y in zip(a.tweet,labels)]
   def loader(ids,shuffle=False):return DataLoader([ds[i] for i in ids],batch_size=32,shuffle=shuffle,generator=torch.Generator().manual_seed(42),num_workers=0)
   model=GraphClassifier(key,len(vocab)+2)
  else:
   model=transformer_for(key,pretrained);model.config.pad_token_id=tok.pad_token_id
   bs=2 if key=='tinyllama' else 4
   def loader(ids,shuffle=False):return torch.utils.data.DataLoader(TextData(ids),batch_size=bs if shuffle else 8,shuffle=shuffle,generator=torch.Generator().manual_seed(42),num_workers=0)
  lr=.001 if graph else (2e-4 if key=='tinyllama' else 2e-5)
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01)
  trainloader=loader(ix,True);sched=get_linear_schedule_with_warmup(opt,int(.1*len(trainloader)*epochs),len(trainloader)*epochs)
  scaler=torch.amp.GradScaler('cuda',enabled=not graph)
  def forward(batch):
   batch=batch.to(device) if graph else {k:v.to(device) for k,v in batch.items()}
   if graph:
    z=model(batch);y=batch.y;loss=torch.nn.functional.cross_entropy(z,y)
   else:
    with torch.autocast('cuda',dtype=torch.float16,enabled=key!='t5'):output=model(**batch)
    z=output.logits;y=batch['labels'];loss=output.loss
   assert torch.isfinite(loss) and torch.isfinite(z).all(),'Nonfinite model outputs'
   return loss,z,y
  def evaluate(ids):
   model.eval();total=0;pred=[]
   with torch.no_grad():
    for b in loader(ids):
     loss,z,y=forward(b);total+=loss.item()*len(y);pred.extend(z.argmax(1).cpu().tolist())
   return total/len(ids),float(f1_score(labels[list(ids)],pred,average='weighted',zero_division=0))
  for ep in range(1,epochs+1):
   tick=time.perf_counter();model.train()
   for step,b in enumerate(trainloader,1):
    if time.perf_counter()-start>budget:raise TimeoutError('Locked training time budget exceeded')
    opt.zero_grad(set_to_none=True);loss,z,y=forward(b)
    scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    previous=scaler.get_scale();scaler.step(opt);scaler.update()
    if scaler.get_scale()>=previous:sched.step()
    if step%200==0:log(f'{key} {name} epoch={ep} step={step}/{len(trainloader)} loss={loss.item():.4f}')
    if ep==1 and step==30:
     estimate=(time.perf_counter()-tick)/30*len(trainloader)*epochs*2
     save(out/f'{name}_probe.json',{'estimated_selection_plus_refit_seconds':estimate,'budget_seconds':budget})
     if name=='selection' and estimate>budget*.90:raise TimeoutError('Measured throughput exceeds locked budget; no truncation masquerading as full training')
   tl,tf=evaluate(ix);dl,df=evaluate(dev) if dev is not None else (None,None)
   row={'phase':name,'epoch':ep,'train_loss':tl,'train_weighted_f1':tf,'inner_dev_loss':dl,'inner_dev_weighted_f1':df,'seconds':time.perf_counter()-tick}
   history.append(row);save(out/f'{name}_epoch_{ep:02d}.json',row)
   log(f'{key} {name} epoch {ep}: train loss {tl:.5f}, F1 {tf:.6f}; inner dev loss {dl}, F1 {df}')
   if name=='refit_A' and ep==epochs:
    if graph:
     save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()},str(modeldir/'model.safetensors'))
     save(modeldir/'vocab.json',vocab)
    else:
     model.save_pretrained(modeldir/'checkpoint',safe_serialization=True);tok.save_pretrained(modeldir/'tokenizer')
  del model,opt,sched,scaler;gc.collect();torch.cuda.empty_cache()
 epochs=12 if graph else (2 if key=='tinyllama' else 3)
 try:
  phase('selection',fi,epochs,di)
  best=max(history,key=lambda r:(r['inner_dev_weighted_f1'],-r['epoch']))['epoch']
  save(out/'epoch_selection_lock.json',{'epochs':best,'selection_data':'inner-A only','config_sha256':sha(CFG)})
  phase('refit_A',range(len(a)),best)
  pd.DataFrame(history).to_csv(out/'history.csv',index=False,mode='x')
  d={'key':key,'name':NAMES[key],'selected_epochs':best,'training_seconds':time.perf_counter()-start,'config_sha256':sha(CFG),
     'files':{str(p.relative_to(ROOT)):sha(p) for p in modeldir.rglob('*') if p.is_file()},'official_test_read':False}
  save(out/'base_lock.json',d);return d
 except Exception as e:
  if history:pd.DataFrame(history).to_csv(out/'partial_history.csv',index=False,mode='x')
  fail=out/'FAILED.json'
  if fail.exists():fail=out/f'FAILED_{time.time_ns()}.json'
  save(fail,{'error':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-start,'artifacts_preserved':True})
  log(key+' FAILED: '+str(e));raise

def load_new(key):
 modeldir=MODELS/key
 d=read(OUT/key/'base_lock.json');assert all(sha(ROOT/p)==h for p,h in d['files'].items())
 if key in ['gcn','gat']:
  from safetensors.torch import load_file
  vocab=read(modeldir/'vocab.json');model=GraphClassifier(key,len(vocab)+2)
  model.load_state_dict(load_file(str(modeldir/'model.safetensors')));return model.eval(),vocab
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 tok=AutoTokenizer.from_pretrained(modeldir/'tokenizer',local_files_only=True,trust_remote_code=False)
 if key=='tinyllama':
  from peft import PeftModel
  base=transformer_for(key,ROOT/'models/pretrained'/f'{key}_{SPECS[key][1]}')
  # Reload adapter onto the original frozen base, rather than nesting adapters.
  base=base.unload()
  model=PeftModel.from_pretrained(base,modeldir/'checkpoint',local_files_only=True)
 else:model=AutoModelForSequenceClassification.from_pretrained(modeldir/'checkpoint',local_files_only=True,trust_remote_code=False,use_safetensors=True,attn_implementation='eager').cuda()
 model.config.pad_token_id=tok.pad_token_id;return model.eval(),tok

def raw_predict(key,texts,loaded=None):
 model,aux=loaded or load_new(key);parts=[]
 with torch.inference_mode():
  for start in range(0,len(texts),8):
   ts=list(texts[start:start+8])
   if key in ['gcn','gat']:
    from torch_geometric.data import Batch
    b=Batch.from_data_list([graph_data(t,0,aux) for t in ts]);z=model(b)
   else:
    x=aux([normalize_text(t) for t in ts],padding='max_length',truncation=True,max_length=128,return_tensors='pt').to('cuda')
    with torch.autocast('cuda',dtype=torch.float16,enabled=key!='t5'):z=model(**x).logits
   p=torch.softmax(z.float(),1).cpu().numpy();assert np.isfinite(p).all();parts.append(p)
 return np.concatenate(parts).astype(float)

def logits(p):
 v=np.clip(p[:,1],1e-7,1-1e-7);return np.log(v/(1-v)).reshape(-1,1)

def evaluate_one(key):
 import joblib
 from sklearn.linear_model import LogisticRegression
 import pilot_h1_h2 as pilot
 out=OUT/key
 if (out/'COMPLETE.json').exists():return pd.read_csv(out/'metrics.csv')
 loaded=load_new(key);c=partition('C');tick=time.perf_counter();pc=raw_predict(key,c.tweet.tolist(),loaded);c_inf=time.perf_counter()-tick
 tick=time.perf_counter();cal=LogisticRegression(C=1,max_iter=2000,random_state=42).fit(logits(pc),c.label);cal_time=time.perf_counter()-tick
 assert list(cal.classes_)==list(CLASSES)
 assert not (out/'calibrator.joblib').exists();joblib.dump(cal,out/'calibrator.joblib')
 save(out/'calibration_lock.json',{'fit_role':'C only','sha256':sha(out/'calibrator.joblib'),'calibration_fit_seconds':cal_time,'C_inference_seconds':c_inf})
 rows=[];bins=[];timings=[]
 val=ROOT/'data/raw/constraint2021/030deba36d118c124709fd72d7d5f14c7bc69053/Constraint_Val.csv'
 assert sha(val)=='c610931d0641fe1d9b9495c453f99ae7aa45bf294c7db6333ec84ecebb0b6c3c'
 v=pd.read_csv(val,dtype=str);v['sample_id']='validation:'+v.id
 masks=pd.read_csv(pilot.REVIEW/'validation_evaluation_masks.csv')
 for role,data in [('H_exploratory',partition('H')),('benchmark_original',v)]:
  tick=time.perf_counter();raw=raw_predict(key,data.tweet.tolist(),loaded);secs=time.perf_counter()-tick
  for rep,p in [('raw',raw),('calibrated',cal.predict_proba(logits(raw)))]:
   f=data[['sample_id','label']].rename(columns={'label':'true_label'}).copy();f['predicted_label']=CLASSES[p.argmax(1)];f['prob_fake']=p[:,0];f['prob_real']=p[:,1]
   cohorts={role:f}
   if role=='benchmark_original':
    from validation_cohorts import select_validation_cohorts
    cohorts=select_validation_cohorts(f,pilot.REVIEW,approve_proposed=True)
   for cohort,frame in cohorts.items():
    frame.to_csv(out/f'{key}_{rep}__{cohort}_predictions.csv',index=False,mode='x')
    met,b=pilot.evaluate(frame);rows.append({**met,'model':key,'representation':rep,'cohort':cohort,'protocol':'Pilot A-fit','evaluation_status':'exploratory/development'})
    b['representation']=rep;b['cohort']=cohort;bins.append(b)
  timings.append({'cohort':role,'n':len(data),'base_inference_seconds':secs,'scope':'normalization/tokenization+model forward+transfer; excludes load'})
 pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False,mode='x');pd.concat(bins).to_csv(out/'calibration_bins.csv',index=False,mode='x');pd.DataFrame(timings).to_csv(out/'timings.csv',index=False,mode='x')
 save(out/'COMPLETE.json',{'official_test_read':False,'files':{str(p.relative_to(ROOT)):sha(p) for p in out.glob('*') if p.is_file()}})
 del loaded;gc.collect();torch.cuda.empty_cache();return pd.DataFrame(rows)

def run_model(key):
 """Notebook wrapper reports a failed model and proceeds to the next bounded task."""
 try:
  train_one(key);return evaluate_one(key)
 except Exception as exc:
  p=OUT/key/'FAILED.json'
  if not p.exists():save(p,{'error':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'stage':'preflight/load/evaluation','artifacts_preserved':True})
  traceback.print_exc();gc.collect();torch.cuda.empty_cache();return {'model':key,'status':'FAILED; preserved logs, not counted as trained'}

