"""Acquire each fixed article once per device, using explicit GPT-2 hooks."""
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import numpy as np
import p5_io as disk


def hardware(torch):
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        driver = type(exc).__name__
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    meminfo = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value, _ = line.split()
                meminfo[key.rstrip(":")] = int(value)*1024
    info = dict(python=platform.python_version(), machine=platform.machine(),
        platform=platform.platform(), torch_cuda=torch.version.cuda, nvidia_smi=driver,
        gpu_name=gpu.name if gpu else None, gpu_memory=gpu.total_memory if gpu else None,
        compute_capability=[gpu.major, gpu.minor] if gpu else None,
        packages={x: importlib.metadata.version(x) for x in disk.spec()["pins"]},
        dtype="float32", attention="eager", tf32=False, deterministic=True,
        cpu_threads=torch.get_num_threads(), host_memory=meminfo,
        gpu_free_bytes=torch.cuda.mem_get_info()[0] if gpu else None)
    return json.loads(disk.canonical(info))


def compatibility(info):
    # Normalize the persisted and in-memory types before equality (P0 v1 lesson).
    return json.loads(disk.canonical({k: v for k, v in info.items()
                                     if k not in ("platform", "host_memory", "gpu_free_bytes")}))


def collect(model, ids, tokenizer):
    """Shared by production and random-model adapter tests; no additional forward."""
    import torch
    captured, calls, handles = {}, {}, []

    def save(name, value):
        calls[name] = calls.get(name, 0)+1
        captured[name] = value.detach().clone()

    def hook(name):
        def inner(_, __, output):
            save(name, output[0] if isinstance(output, tuple) else output)
        return inner

    handles.append(model.transformer.h[5].register_forward_hook(hook("block06")))
    handles.append(model.transformer.h[6].register_forward_hook(hook("block07")))
    handles.append(model.transformer.ln_f.register_forward_pre_hook(
        lambda _, args: save("block12_preln", args[0])))
    handles.append(model.transformer.ln_f.register_forward_hook(hook("final_ln")))
    try:
        with torch.no_grad():
            # Same call shape as historical extraction; config fixes hidden states/cache.
            output = model(ids)
        disk.require(all(calls.get(p) == 1 for p in disk.spec()["points"]), "Hook count differs")
        for p, index in [("block06", 6), ("block07", 7), ("final_ln", -1)]:
            torch.testing.assert_close(captured[p], output.hidden_states[index], rtol=0, atol=0)
        norm = model.transformer.ln_f
        reference = torch.nn.functional.layer_norm(captured["block12_preln"],
            norm.normalized_shape, norm.weight, norm.bias, norm.eps)
        torch.testing.assert_close(reference, captured["final_ln"], rtol=1e-5, atol=1e-5)
        final_hidden = captured["final_ln"]
        # Independently reconstruct the tied readout to verify the point before logits.
        logits_check = torch.nn.functional.linear(final_hidden, model.lm_head.weight)
        torch.testing.assert_close(logits_check, output.logits, rtol=1e-5, atol=1e-5)
        logits = output.logits[0, :-1].double()
        targets = ids[0, 1:]
        logz = torch.logsumexp(logits, dim=1)
        selected = logits.gather(1, targets[:, None]).squeeze(1)
        ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        torch.testing.assert_close(logz-selected, ce, rtol=1e-10, atol=1e-10)
        n = ids.shape[1]
        arrays = {p: captured[p][0].float().cpu().numpy() for p in disk.spec()["points"]}
        token_ids = ids[0].cpu().numpy().astype(np.int64)
        pieces = [tokenizer.decode([int(i)], clean_up_tokenization_spaces=False) for i in token_ids]
        arrays.update(token_id=token_ids, position=np.arange(n, dtype=np.int64),
            token_chars=np.array([len(x) for x in pieces], dtype=np.int64),
            token_bytes=np.array([len(x.encode()) for x in pieces], dtype=np.int64),
            token_text=np.array(pieces),
            surprisal=np.r_[np.nan, (logz-selected).cpu().numpy()],
            target_logit=np.r_[np.nan, selected.cpu().numpy()],
            logsumexp=np.r_[np.nan, logz.cpu().numpy()],
            logit_source_index=np.arange(n, dtype=np.int64)-1,
            norm_weight=norm.weight.detach().float().cpu().numpy(),
            norm_bias=norm.bias.detach().float().cpu().numpy(),
            norm_eps=np.array(norm.eps, dtype=np.float64))
        return arrays, dict(hooks=calls, hidden_state_alignment=True,
            norm_reconstruction=True, logit_reconstruction=True, cross_entropy_alignment=True)
    finally:
        for handle in handles:
            handle.remove()


def main(root, device):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = Path(root)
    cfg = disk.spec()
    disk.require(device in cfg["devices"], "Unknown device")
    prior = disk.committed(root)
    needed = [e for e in disk.entries() if device+"_"+e["id"] not in prior]
    if not needed:
        print("NOOP: all "+device+" units already committed.", flush=True)
        return
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    env = hardware(torch)
    disk.require(env["packages"] == cfg["pins"], "Pinned dependencies differ")
    compatible = root/"environment_contract.json"
    if compatible.exists():
        disk.require(disk.read(compatible) == compatibility(env), "GPU/environment changed; retain separate runs")
    else:
        disk.write(compatible, compatibility(env), immutable=True)
    disk.write(root/"environments"/(uuid.uuid4().hex+".json"), dict(at=disk.stamp(), **env), immutable=True)
    if device == "cuda":
        disk.require(torch.cuda.is_available(), "CUDA unavailable; CPU checkpoint retained")
        disk.require(torch.cuda.mem_get_info()[0] >= 2*1024**3, "Need at least 2 GiB free GPU memory")
    model_info = disk.read(disk.HERE/"model.json")
    local = Path(snapshot_download(repo_id=model_info["repo"], revision=model_info["revision"],
                allow_patterns=list(model_info["files"]), token=False, max_workers=2))
    for name, expected in model_info["files"].items():
        disk.require((local/name).stat().st_size == expected["bytes"] and
            disk.digest(local/name) == expected["sha256"], "Model/tokenizer checksum differs: "+name)
    disk.write(root/"model_verified.json", model_info, immutable=True)
    tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float32, attn_implementation="eager",
        output_hidden_states=True)
    disk.require(model.config.n_layer == 12 and model.config.n_embd == 768 and
                 model.config.n_positions == 1024 and model.config.use_cache is True,
                 "GPT-2 structure/default cache differs")
    model.eval().to(device)
    disk.require(all(p.dtype == torch.float32 for p in model.parameters()), "Model dtype differs")
    for entry in needed:
        key = device+"_"+entry["id"]
        ids = tokenizer.encode(entry["text"], return_tensors="pt")
        disk.require(ids[0].tolist() == entry["token_ids"], "Frozen token IDs differ")
        disk.require(ids.shape[1] <= model.config.n_positions, "Unexpected truncation")
        event = root/"forward_attempts"/(uuid.uuid4().hex+".json")
        disk.write(event, dict(at=disk.stamp(), key=key, status="STARTED"), immutable=True)
        started = time.monotonic()
        arrays, checks = collect(model, ids.to(device), tokenizer)
        disk.save_unit(root, key, arrays, dict(device=device, text_id=entry["id"],
            text_sha256=entry["utf8_sha256"], seconds=time.monotonic()-started, checks=checks))
        print("P5 committed "+key+" ("+str(len(ids[0]))+" tokens)", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
