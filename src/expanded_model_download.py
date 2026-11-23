"""Download only publisher data files at locked revisions; no model execution."""
from pathlib import Path
import hashlib,json,time,urllib.request

ROOT=Path(__file__).resolve().parents[1]
SPECS={
 'bert':('google-bert/bert-base-uncased','86b5e0934494bd15c9632b12f734a8a67f723594','68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3'),
 'roberta':('FacebookAI/roberta-base','e2da8e2f811d1448a5b465c236feacd80ffbac7b','5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81'),
 'gpt2':('openai-community/gpt2','607a30d783dfa663caf39e06633721c8d4cfcd7e','248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'),
 't5':('google-t5/t5-small','df1b051c49625cf57a3d0d8d3863ed4d13564fe4','bd944e5f1b3ad9b70dd9d00010a517059e19265671076b8b0a4a58d9491842bc'),
 'tinyllama':('TinyLlama/TinyLlama-1.1B-Chat-v1.0','fe8a4ea1ffedaf415f4da2f062534de366a451e6','6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933')}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def fetch(key):
 repo,rev,expected=SPECS[key]; folder=ROOT/'models/pretrained'/f'{key}_{rev}'
 folder.mkdir(parents=True,exist_ok=True)
 if (folder/'provenance.json').exists():
  d=json.loads((folder/'provenance.json').read_text());assert all(sha(folder/n)==v for n,v in d['files'].items());return
 with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{rev}',timeout=60) as r:meta=json.load(r)
 names=[s['rfilename'] for s in meta['siblings'] if '/' not in s['rfilename'] and (s['rfilename'] in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','vocab.json','merges.txt','spiece.model','tokenizer.model','model.safetensors'])]
 for name in names:
  p=folder/name
  if not p.exists():
   tmp=folder/(name+'.part_'+str(time.time_ns()))
   with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',timeout=120) as r,tmp.open('xb') as f:
    size=0
    while block:=r.read(8*1024*1024):
     size+=len(block);assert size<2_300_000_000;f.write(block)
   if name=='model.safetensors':assert sha(tmp)==expected
   tmp.rename(p)
  if name=='model.safetensors':assert sha(p)==expected
  print(key,name,p.stat().st_size,flush=True)
 d={'repo':repo,'revision':rev,'files':{n:sha(folder/n) for n in names},'source':'publisher Hugging Face repository','trust_remote_code':False}
 with (folder/'provenance.json').open('x') as f:json.dump(d,f,indent=2)

if __name__=='__main__':
 import sys
 for k in sys.argv[1:]:fetch(k)
