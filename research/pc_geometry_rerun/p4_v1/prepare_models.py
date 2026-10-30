"""Fetch public configuration/tokenizer metadata only; never model weights."""
from pathlib import Path
import hashlib,json,urllib.request
HERE=Path(__file__).resolve().parent
DEST=HERE.parent/'design_inputs/p4_v1'
MODELS=[('gpt2','openai-community/gpt2','607a30d783dfa663caf39e06633721c8d4cfcd7e',768,12,50257,'LayerNorm'),('opt','facebook/opt-1.3b','3f5c25d0bc631cb57ac65913f76e22c2dfb61d62',2048,24,50272,'LayerNorm'),('qwen','Qwen/Qwen2.5-1.5B','8faed761d45a263340a0528343f099c05c9a4323',1536,28,151936,'RMSNorm')]
def get(url):
 with urllib.request.urlopen(url,timeout=90) as r:return r.read()
def put(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
def main():
 output={}
 for key,repo,rev,hidden,layers,vocab,norm in MODELS:
  raw=get(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true');data=json.loads(raw);assert data['sha']==rev
  put(DEST/key/'hub_metadata.json',raw)
  names={'config.json','generation_config.json','merges.txt','vocab.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'}
  files={}
  for s in data['siblings']:
   n=s['rfilename']
   if n in names:
    b=get(f'https://huggingface.co/{repo}/resolve/{rev}/{n}');put(DEST/key/n,b)
    files[n]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
   elif n==('pytorch_model.bin' if key=='opt' else 'model.safetensors'):
    files[n]={'bytes':s['size'],'sha256':s['lfs']['sha256']}
  assert sum(n.endswith(('.bin','.safetensors')) for n in files)==1
  output[key]=dict(repo=repo,revision=rev,hidden=hidden,layers=layers,mid=layers//2,vocab=vocab,norm=norm,files=files,use_safetensors=key!='opt')
  print(key,rev,'tokenizer/config only; weight bytes planned',sum(f['bytes'] for n,f in files.items() if n.endswith(('.bin','.safetensors'))),flush=True)
 put(HERE/'models.json',(json.dumps(output,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())
if __name__=='__main__':main()
