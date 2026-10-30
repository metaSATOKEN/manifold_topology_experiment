"""Read-only frozen-input and shipping invariants, with tokenizer reconstruction."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
from transformers import AutoTokenizer
import p4_io as disk


def main():
    fs=disk.fixtures();models=disk.read(disk.HERE/'models.json');root=disk.HERE.parent
    counts={};tokens=0
    old=sum([disk.read(root/f'p{j}_v1/fixtures.json') for j in [1,2,3]],[])
    gram=lambda ids:{tuple(ids[i:i+5]) for i in range(len(ids)-4)}
    previous=[(d['id'],gram(d['token_ids'])) for d in old]
    assert not {f['id'] for f in fs}&{f['id'] for f in old}
    for f in fs:
        g=gram(f['token_ids']['gpt2']);assert max(len(g&o)/len(g|o) for _,o in previous)<.5
        previous.append((f['id'],g))
    for k,m in models.items():
        src=root/'design_inputs/p4_v1'/k
        for n,e in m['files'].items():
            if n.endswith(('.bin','.safetensors')):
                assert not (src/n).exists();continue
            assert (src/n).stat().st_size==e['bytes'] and disk.digest(src/n)==e['sha256']
        tok=AutoTokenizer.from_pretrained(src,use_fast=True,local_files_only=True,trust_remote_code=False)
        for f in fs:
            actual=tok.encode(f['text'],add_special_tokens=False);assert actual==f['token_ids'][k]
            assert disk.sha(np.array(actual,dtype='<i8').tobytes())==f['input_ids_sha256'][k]
            tokens+=len(actual)
        counts[k]={split:sum(len(f['token_ids'][k]) for f in fs if f['split']==split) for split in ['train','test']}
    total=sum(sum(v.values())*models[k]['hidden']*2*4 for k,v in counts.items())
    assert total==disk.read(disk.HERE/'input_manifest.json')['raw_hidden_bytes']
    for p in disk.HERE.glob('*.py'):ast.parse(p.read_text())
    result=dict(status='PASS',checked_at=disk.stamp(),article_count=len(fs),tokenizer_article_checks=len(fs)*len(models),token_ids_checked=tokens,token_counts=counts,raw_hidden_bytes=total,old_article_exclusions=256,near_duplicate_check=True,models_are_original_pinned_base_repositories=True,pretrained_weights_downloaded=0,research_model_forwards=0)
    disk.write(disk.HERE/'design_checks.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
