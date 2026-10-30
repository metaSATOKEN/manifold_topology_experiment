"""Audit the release without executing its cell or altering historical evidence."""
from pathlib import Path
import ast,base64,datetime,hashlib,json,re,subprocess,sys,zlib
import p4_io as disk


def main():
    p=disk.HERE;root=p.parents[2];project=p.parent
    before=disk.read(p/'preservation_before.json');release=disk.read(p/'release_manifest.json')
    mutable={'AGENTS.md',*[str((project/n).relative_to(root)) for n in ['README.md','PLAN.md','REQUIREMENTS.md','prereg/README.md','audits/ledger_2026-10.md']]}
    preserved={n:h for n,h in before['sha256'].items() if n not in mutable}
    mismatch=[n for n,h in preserved.items() if not (root/n).is_file() or disk.digest(root/n)!=h]
    disk.require(not mismatch,'Old files changed: '+repr(mismatch))
    ledger=(project/'audits/ledger_2026-10.md').read_text();disk.require(ledger.startswith(before['ledger_text']),'Ledger prefix changed')
    req=(project/'REQUIREMENTS.md').read_text();section=lambda s:s[s.index('## 2.'):s.index('## 4.')]
    disk.require(section(req)==section(before['requirements_text']),'Requirements sections 2/3 changed')
    disk.require(disk.source_hashes()==release['sources'] and disk.capsule()==release['capsule'],'Frozen source/capsule changed')
    cell=(p/'pcgeom_p4_v1_onecell.py').read_text();nb=disk.read(p/'pcgeom_p4_v1_onecell.ipynb')
    disk.require(len(nb['cells'])==1 and ''.join(nb['cells'][0]['source'])==cell,'Notebook cell mismatch')
    disk.require(disk.digest(p/'pcgeom_p4_v1_onecell.py')==release['onecell_sha256'],'Cell SHA')
    disk.require(disk.digest(p/'pcgeom_p4_v1_onecell.ipynb')==release['notebook_sha256'],'Notebook SHA')
    tree=ast.parse(cell);values={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:values[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    raw=zlib.decompress(base64.b64decode(values['_PAYLOAD'],validate=True))
    disk.require(disk.sha(raw)==release['payload_sha256']==values['_PAYLOAD_SHA256'],'Payload hash')
    src=json.loads(raw);disk.require(set(src)==set(disk.SOURCE_NAMES),'Embedded source inventory')
    for n,encoded in src.items():disk.require(base64.b64decode(encoded,validate=True)==(p/n).read_bytes(),'Embedded source mismatch: '+n)
    disk.require(values['RESTORE_ZIPS']=='' and values['AUTO_DOWNLOAD'] is True,'Shipping defaults')
    disk.require((p/'experiment_card.md').read_bytes()==(project/'prereg/EXP-0004_v1_three_models.md').read_bytes(),'Card copy mismatch')
    files=[*p.glob('*.md'),project/'prereg/EXP-0004_v1_three_models.md',project/'README.md',project/'PLAN.md',project/'prereg/README.md',project/'HANDOFF/2026-10-20_zzzzzzzzzzzz_p4_v1.md']
    links=0;pending={p/'final_checks.json',project/'evidence/p4_release_checks_2026-10-20.json'}
    for f in files:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',f.read_text()):
            if '://' in target or target.startswith('#'):continue
            target=target.split('#')[0];dest=(f.parent/target).resolve()
            disk.require(dest.exists() or dest in pending,'Broken link: '+str(f)+' -> '+target);links+=1
    baseline=subprocess.run([sys.executable,str(project/'check_baseline.py')],cwd=root,capture_output=True,text=True,check=True)
    result=dict(status='PASS',checked_at=disk.stamp(),preserved_files_checked=len(preserved),preservation_mismatches=[],requirements_sections_2_3_unchanged=True,ledger_previous_prefix_unchanged=True,
        source_files_embedded=len(src),cell_and_notebook_identical=True,payload_roundtrip=True,local_links_checked=links,
        local_tests=disk.read(p/'test_results.json'),adapter_tests=disk.read(p/'adapter_test_results.json'),design_checks=disk.read(p/'design_checks.json'),
        release=release,baseline=json.loads(baseline.stdout),research_model_weights_downloaded=0,research_model_forward_calls=0,
        unverified=['Pinned torch 2.8 CUDA runtime','Pretrained model extraction','Real Colab serial download/upload','New VM restore with real observations','Actual three-model runtime and ZIP sizes'],
        comparison_end='After this fixed three-model comparison; no automatic intervention or model expansion')
    disk.write(p/'final_checks.json',result)
    artifacts=[*disk.files(p),*disk.files(project/'design_inputs/p4_v1'),project/'prereg/EXP-0004_v1_three_models.md',project/'HANDOFF/2026-10-20_zzzzzzzzzzzz_p4_v1.md',*[root/n for n in mutable]]
    artifact_sha256={str(f.relative_to(root)):disk.digest(f) for f in artifacts if '__pycache__' not in f.parts and '.pytest_cache' not in f.parts}
    disk.write(project/'evidence/p4_release_checks_2026-10-20.json',dict(status='PASS',checked_at=disk.stamp(),previous_anchor_files_at_start=len(before['sha256']),preserved_files_checked=len(preserved),preservation_mismatches=[],requirements_sections_2_3_unchanged=True,ledger_previous_prefix_unchanged=True,tests=26,synthetic_adapter_models=3,research_model_forward_calls=0,release_capsule=release['capsule'],preserved_sha256=preserved,artifact_sha256=artifact_sha256))
    print(json.dumps({k:result[k] for k in ['status','preserved_files_checked','source_files_embedded','local_links_checked','research_model_forward_calls','unverified']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
