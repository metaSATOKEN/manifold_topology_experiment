"""GPU-only extraction. Importing this module does not import torch or load a model."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import p4_io as disk


def hardware(torch):
    props = torch.cuda.get_device_properties(0)
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            mem[key] = int(value.split()[0])*1024
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    return dict(gpu_name=props.name, compute_capability=[props.major, props.minor],
                gpu_total_bytes=props.total_memory, gpu_free_bytes=torch.cuda.mem_get_info()[0],
                host_ram=mem, disk_free_bytes=shutil.disk_usage("/content").free,
                python=sys.version, machine=platform.machine(), torch=torch.__version__,
                cuda_build=torch.version.cuda, nvidia_smi=smi.stdout, driver_query_exit=smi.returncode,
                packages=sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()),
                tf32=False, dtype="float32", attention="eager")


def compatibility(env):
    selected = {k: env[k] for k in ("gpu_name", "compute_capability", "python", "torch",
                                "cuda_build", "nvidia_smi", "driver_query_exit",
                                "packages", "tf32", "dtype", "attention")}
    # JSON persists tuples as lists. Use the persisted representation on both sides.
    return json.loads(disk.canonical(selected))


def require_compatible(previous, current):
    left, right = compatibility(previous), compatibility(current)
    changed = [key for key in left if left[key] != right[key]]
    disk.require(not changed, "GPU/environment changed; do not mix runs. Changed fields: " + ", ".join(changed))


def record_model_config(root, attempt, key, config):
    """A restored cache directory may move; model settings must remain identical."""
    target = Path(root) / "models" / key / "loaded_model_config.json"
    if target.exists():
        previous = disk.read(target)
        semantic = lambda c: {k: v for k, v in c.items() if k != "_name_or_path"}
        disk.require(disk.canonical(semantic(previous)) == disk.canonical(semantic(config)),
                     "Loaded model configuration changed")
    else:
        disk.write(target, config, immutable=True)
    disk.write(Path(root) / "attempts" / attempt / "loaded_model_config.json", config, immutable=True)


def modules(model, key):
    if key=="gpt2":return model.transformer.h,model.transformer.ln_f
    if key=="opt":
        disk.require(model.model.decoder.project_out is None, "OPT output projection changed")
        return model.model.decoder.layers,model.model.decoder.final_layer_norm
    if key=="qwen":return model.model.layers,model.model.norm
    raise ValueError("Unknown model adapter")


def extract(model, tokenizer, fixture, key, torch, np, device="cuda"):
    cfg=disk.model(key)
    ids=tokenizer(fixture["text"],add_special_tokens=False,return_tensors="pt")["input_ids"]
    disk.require(ids[0].tolist()==fixture["token_ids"][key], "Frozen IDs differ")
    layers,norm=modules(model,key)
    disk.require(len(layers)==cfg["layers"] and norm is not None, "Layer structure changed")
    captures={};handles=[]
    def capture(name):
        def hook(_module,_inputs,output):
            value=output[0] if isinstance(output,(tuple,list)) else output
            captures[name]=value.detach().clone()
        return hook
    def before(_module,inputs):captures["pre_norm"]=inputs[0].detach().clone()
    handles.append(layers[cfg["mid"]-1].register_forward_hook(capture("middle")))
    handles.append(norm.register_forward_pre_hook(before))
    handles.append(norm.register_forward_hook(capture("final_norm")))
    try:
        with torch.inference_mode():
            output=model(input_ids=ids.to(device),attention_mask=torch.ones_like(ids).to(device),
                         use_cache=False,output_hidden_states=True,return_dict=True)
            pre=captures["pre_norm"]
            if key=="qwen":
                direct=norm.weight*(pre.float()*torch.rsqrt(pre.float().pow(2).mean(-1,keepdim=True)+norm.variance_epsilon))
            else:
                direct=torch.nn.functional.layer_norm(pre,norm.normalized_shape,norm.weight,norm.bias,norm.eps)
            checks=dict(middle_matches=bool(torch.equal(captures["middle"],output.hidden_states[cfg["mid"]])),
                final_matches=bool(torch.equal(captures["final_norm"],output.hidden_states[-1])),
                normalization_max_relative_error=float((direct-captures["final_norm"]).abs().max()/max(1.,float(direct.abs().max()))))
            logits=output.logits[0,:-1].double();targets=ids[0,1:].to(device)
            target=logits.gather(1,targets[:,None]).squeeze(1);logz=torch.logsumexp(logits,dim=1)
            ce=torch.nn.functional.cross_entropy(logits,targets,reduction="none")
            checks["surprisal_ce_max_abs"]=float((ce-(logz-target)).abs().max())
            arrays={"h_"+name:captures[name][0].float().cpu().numpy().copy() for name in disk.spec()["points"]}
            for name,v in [("surprisal",logz-target),("target_logit",target),("logsumexp",logz)]:
                arrays[name]=np.r_[np.nan,v.cpu().numpy()].astype(np.float64)
    finally:
        for h in handles:h.remove()
    tid=ids[0].numpy().astype(np.int64);n=len(tid)
    decoded=[tokenizer.decode([int(i)],clean_up_tokenization_spaces=False) for i in tid]
    arrays.update(token_id=tid,position=np.arange(n,dtype=np.int64),
        token_chars=np.array([len(s) for s in decoded],np.int64),
        token_utf8_bytes=np.array([len(s.encode()) for s in decoded],np.int64),
        logit_source_index=np.arange(-1,n-1,dtype=np.int64))
    return arrays,dict(tokens=tokenizer.convert_ids_to_tokens(tid.tolist()),decoded_tokens=decoded,
        extraction_checks=checks,input_ids_sha256=disk.sha(tid.astype("<i8").tobytes()))


def main(root,key,split):
    root=Path(root);attempt=uuid.uuid4().hex[:12];started=time.monotonic()
    disk.require(key in disk.spec()["models"] and split in ("train","test"), "Worker scope")
    run=disk.check_run(root);done=disk.committed(root)
    wanted=[f for f in disk.fixtures() if f["split"]==split]
    if all(disk.doc_key(key,f["id"]) in done for f in wanted):
        disk.event(root,attempt,"NOOP",model_key=key,split=split,forward_calls=0)
        print(f"P4 {key}/{split}: committed documents verified; zero new forwards.",flush=True)
        return 0
    fit_sha=None
    if split=="test":
        disk.check_fit(root);fit_sha=disk.digest(root/"fit/COMMIT.json")
    else:
        disk.require(not (root/"fit/COMMIT.json").exists(), "Training is already sealed")
    try:
        disk.require(sys.platform=="linux" and Path("/content").is_dir(), "Colab GPU only")
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from huggingface_hub import snapshot_download
        from p4_numeric import save_npz,validate_raw
        cfg=disk.spec(); m=disk.model(key)
        for name,version in cfg["pins"].items():disk.require(importlib.metadata.version(name)==version,"Pin mismatch: "+name)
        disk.require(torch.cuda.is_available(),"CUDA unavailable; no CPU fallback")
        torch.cuda.init();torch.set_num_threads(2);torch.manual_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest")
        env=hardware(torch);disk.write(root/"attempts"/attempt/"environment.json",env)
        disk.require("A100" in env["gpu_name"] or ("RTX PRO 6000" in env["gpu_name"] and "Blackwell" in env["gpu_name"]),"Expected A100 or RTX PRO 6000 Blackwell")
        disk.require(env["gpu_free_bytes"]>=cfg["minimum_gpu_free_bytes"],"Need 14 GiB free VRAM")
        disk.require(env["host_ram"]["MemAvailable"]>=cfg["minimum_host_available_bytes"],"Need 12 GiB available host RAM")
        previous=root/"environment_contract.json"
        if previous.exists():require_compatible(disk.read(previous),env)
        else:disk.write(previous,compatibility(env),immutable=True)
        x=torch.eye(16,device="cuda",dtype=torch.float32)
        disk.require(torch.equal(x@x,x),"GPU preflight");torch.cuda.synchronize()
        disk.event(root,attempt,"GPU_PREFLIGHT_PASS",model_key=key,scientific_forward_calls=0)
        folder=Path(snapshot_download(m["repo"],revision=m["revision"],allow_patterns=list(m["files"]),token=False))
        inventory={n:dict(bytes=(folder/n).stat().st_size,sha256=disk.digest(folder/n)) for n in m["files"]}
        disk.require(inventory==m["files"],"Model/tokenizer SHA mismatch")
        disk.write(root/"models"/key/"files.json",inventory,immutable=True)
        tok=AutoTokenizer.from_pretrained(folder,use_fast=True,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
            use_safetensors=m["use_safetensors"],weights_only=True,dtype=torch.float32,attn_implementation="eager").to("cuda").eval()
        model.requires_grad_(False)
        layers,norm=modules(model,key)
        hidden=model.config.n_embd if key=="gpt2" else model.config.hidden_size
        disk.require(len(layers)==m["layers"] and hidden==m["hidden"] and model.config.vocab_size==m["vocab"],"Architecture mismatch")
        disk.require(all(p.dtype==torch.float32 and p.device.type=="cuda" for p in model.parameters()),"Precision/device")
        record_model_config(root,attempt,key,model.config.to_dict())
        for f in wanted:
            dk=disk.doc_key(key,f["id"])
            if dk in done:
                validate_raw(root,key,f["id"]);continue
            disk.require(shutil.disk_usage(root).free>6*1024**3,"Need 6 GiB free for evidence/recovery")
            disk.require(sum(p.stat().st_size for p in disk.files(root))<disk.RAW_LIMIT-64*1024**2,"Evidence budget")
            dest=root/"docs"/dk
            if dest.exists():
                orphan=root/"attempts"/attempt/"incomplete_before_retry"/dk
                orphan.parent.mkdir(parents=True,exist_ok=True);os.replace(dest,orphan)
            disk.event(root,attempt,"FORWARD_STARTED",model_key=key,document_id=f["id"],split=split,fit_commit_sha256=fit_sha)
            t=time.monotonic();torch.cuda.reset_peak_memory_stats()
            arrays,details=extract(model,tok,f,key,torch,np);torch.cuda.synchronize()
            save_npz(dest/"raw.npz",arrays)
            disk.write(dest/"doc.json",dict(document_id=f["id"],model_key=key,split=split,fit_commit_sha256=fit_sha,
                run_id=run["run_id"],attempt=attempt,text=f["text"],text_sha256=f["text_sha256"],model=m["repo"],revision=m["revision"],
                specification_sha256=run["specification_sha256"],forward_seconds=time.monotonic()-t,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),**details),immutable=True)
            validate_raw(root,key,f["id"]);disk.seal_doc(dest);done[dk]=disk.read(dest/"COMMIT.json")
            disk.event(root,attempt,"DOCUMENT_COMMITTED",model_key=key,document_id=f["id"],split=split)
            n=sum(disk.doc_key(key,x["id"]) in done for x in wanted)
            print(f"P4 {key}/{split}: {n}/{len(wanted)} articles saved; all models {len(done)}/384.",flush=True)
        disk.write(root/"attempts"/attempt/"result.json",dict(status="PHASE_COMPLETE",model_key=key,split=split,seconds=time.monotonic()-started))
        return 0
    except BaseException as exc:
        disk.event(root,attempt,"WORKER_FAILURE",model_key=key,split=split,error_type=type(exc).__name__,error=str(exc))
        disk.write(root/"attempts"/attempt/"error.json",dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        traceback.print_exc();return 2


if __name__=="__main__":sys.exit(main(*sys.argv[1:]))
