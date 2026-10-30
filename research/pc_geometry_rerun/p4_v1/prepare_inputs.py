"""Freeze 64 training + 64 evaluation articles, shared exact text across models.
Only tokenizers run locally. No model weight load or inference.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'design_inputs/p1_v1'
def sha(b):return hashlib.sha256(b).hexdigest()
def grams(ids):return {tuple(ids[i:i+5]) for i in range(len(ids)-4)}
def main():
 models=json.loads((HERE/'models.json').read_text())
 ts={k:AutoTokenizer.from_pretrained(ROOT/'design_inputs/p4_v1'/k,local_files_only=True,use_fast=True,trust_remote_code=False) for k in models}
 downloads=json.loads((SOURCE/'downloads.json').read_text())
 assert sha((SOURCE/'train.parquet').read_bytes())==downloads['train.parquet']['sha256']
 rows=pq.read_table(SOURCE/'train.parquet',columns=['text'])['text'].to_pylist()
 articles=[]; current=None
 for ix,row in enumerate(rows):
  heading=re.fullmatch(r'\s*=\s+([^=].*?)\s+=\s*',row)
  if heading and not heading.group(1).strip().endswith('='):
   if current is not None:articles.append(current)
   current=dict(title=heading.group(1).strip(),row_start=ix,rows=[])
  elif current is not None and row.strip() and not row.strip().startswith('='):current['rows'].append(row.strip())
 if current is not None:articles.append(current)
 old=sum([json.loads((ROOT/f'p{j}_v1/fixtures.json').read_text()) for j in [1,2,3]],[])
 old_ids={d['id'] for d in old};oldtitles={' '.join(d['title'].casefold().split()) for d in old}
 checks=[(d['id'],grams(d['token_ids'])) for d in old]
 p0=json.loads((ROOT/'p0_v2/fixtures.json').read_text());p0ids=[ts['gpt2'].encode(d['text'],add_special_tokens=False)[:256] for d in p0]
 candidates=[];excluded=[]
 for a in articles:
  full='\n\n'.join(a['rows']);h=sha(full.encode());a.update(id=f"wt2_train_{a['row_start']:05d}",full_article_body_sha256=h,text=full[:8192],selection_key=sha(('pcgeom-p4-20261020:'+h).encode()))
  candidates.append(a)
 selected=[]
 for a in sorted(candidates,key=lambda a:(a['selection_key'],a['id'])):
  reason=None;title=' '.join(a['title'].casefold().split())
  if a['id'] in old_ids or title in oldtitles:reason='previous_article_or_title'
  if reason:excluded.append(dict(id=a['id'],reason=reason));continue
  # Offset maps only determine a common character cutoff. Re-encode the exact
  # shared text afterward; never assume that slicing text preserves token IDs.
  enc={k:t(a['text'],add_special_tokens=False,return_offsets_mapping=True) for k,t in ts.items()}
  if min(len(v['input_ids']) for v in enc.values())<256:
   excluded.append(dict(id=a['id'],reason='short_article'));continue
  cutoff=min(v['offset_mapping'][min(512,len(v['input_ids']))-1][1] for v in enc.values())
  text=a['text'][:cutoff]
  # A word boundary avoids chopped words and duplicate byte-fallback offsets.
  ends=[m.end() for m in re.finditer(r'\s+',text)]
  if ends:text=text[:ends[-1]].rstrip()
  ids={k:t.encode(text,add_special_tokens=False) for k,t in ts.items()}
  while max(map(len,ids.values()))>512:
   text=text.rsplit(' ',1)[0];ids={k:t.encode(text,add_special_tokens=False) for k,t in ts.items()}
  if min(map(len,ids.values()))<256:
   excluded.append(dict(id=a['id'],reason='common_span_shorter_than_256'));continue
  if any(ids['gpt2'][:256]==p for p in p0ids):
   excluded.append(dict(id=a['id'],reason='P0_exact_prefix'));continue
  g=grams(ids['gpt2']);scores=[(len(g&o)/len(g|o),oid) for oid,o in checks];score,against=max(scores)
  if score>=.5:
   excluded.append(dict(id=a['id'],reason='gpt2_token_5gram_jaccard_ge_0.5',against=against,score=score));continue
  d={k:a[k] for k in ['id','title','row_start','full_article_body_sha256','selection_key']}
  d.update(text=text,text_sha256=sha(text.encode()),split='train' if len(selected)<64 else 'test',token_ids=ids,input_ids_sha256={k:sha(np.array(v,dtype='<i8').tobytes()) for k,v in ids.items()},max_jaccard_against_previous_and_selected=score)
  selected.append(d);checks.append((a['id'],g));oldtitles.add(title)
  if len(selected)==128:break
 assert len(selected)==128
 order=sorted([d['id'] for d in selected if d['split']=='train'],key=lambda s:sha(('pcgeom-p4-fold:'+s).encode()))
 for d in selected:d['fold']=order.index(d['id'])%4 if d['split']=='train' else None
 manifest=dict(dataset='Salesforce/wikitext',revision='b08601e04326c79dfdd32d625aee71d232d685c3',configuration='wikitext-2-raw-v1',split='train',source=downloads['train.parquet'],article_count=len(articles),selected=128,train=64,test=64,old_article_ids=sorted(old_ids),excluded=excluded,rule='Remove headings; first 8192 Unicode chars; shared text cut at last whitespace before earliest model token-512 end; reencode; require 256..512 for every tokenizer; no special tokens. SHA256 pcgeom-p4-20261020:+full-body SHA order; first64 train, next64 test.',model_inference_performed=False)
 manifest['token_counts']={k:{s:dict(total=sum(len(d['token_ids'][k]) for d in selected if d['split']==s),min=min(len(d['token_ids'][k]) for d in selected if d['split']==s),max=max(len(d['token_ids'][k]) for d in selected if d['split']==s)) for s in ['train','test']} for k in models}
 manifest['raw_hidden_bytes']=sum(sum(len(d['token_ids'][k]) for d in selected)*v['hidden']*2*4 for k,v in models.items())
 for n,v in [('fixtures.json',selected),('input_manifest.json',manifest)]:
  body=json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+'\n';p=HERE/n
  if p.exists():assert p.read_text()==body
  else:p.write_text(body)
 print(json.dumps({k:manifest[k] for k in ['article_count','selected','token_counts','raw_hidden_bytes']},indent=2))
if __name__=='__main__':main()
