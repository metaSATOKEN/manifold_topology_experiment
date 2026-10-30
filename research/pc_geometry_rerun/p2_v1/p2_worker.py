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
import p2_io as disk


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


def record_model_config(root, attempt, config):
    """A restored cache directory may move; model settings must remain identical."""
    target = Path(root) / "loaded_model_config.json"
    if target.exists():
        previous = disk.read(target)
        semantic = lambda c: {k: v for k, v in c.items() if k != "_name_or_path"}
        disk.require(disk.canonical(semantic(previous)) == disk.canonical(semantic(config)),
                     "Loaded model configuration changed")
    else:
        disk.write(target, config, immutable=True)
    disk.write(Path(root) / "attempts" / attempt / "loaded_model_config.json", config, immutable=True)


def extract(model, tokenizer, fixture, torch, np):
    """Explicit output hooks disambiguate final block from final layer norm."""
    cfg = disk.spec()
    encoded = tokenizer(fixture["text"], add_special_tokens=False, truncation=True,
                        max_length=cfg["tokens"], return_tensors="pt")
    ids = encoded["input_ids"]
    disk.require(ids.shape == (1, cfg["tokens"]), "Fixture contains fewer than 512 tokens")
    disk.require(ids[0].tolist() == fixture["token_ids"], "Tokenizer differs from frozen input IDs")
    captures = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captures[name] = value.detach().clone()
        return hook

    for index, name in ((0, "block01"), (5, "block06"), (11, "block12_preln")):
        handles.append(model.transformer.h[index].register_forward_hook(capture(name)))
    handles.append(model.transformer.ln_f.register_forward_hook(capture("final_ln")))
    try:
        with torch.inference_mode():
            output = model(input_ids=ids.to("cuda"), attention_mask=torch.ones_like(ids).to("cuda"),
                           use_cache=False, output_hidden_states=True, return_dict=True)
            preln = torch.nn.functional.layer_norm(captures["block12_preln"],
                model.transformer.ln_f.normalized_shape, model.transformer.ln_f.weight,
                model.transformer.ln_f.bias, model.transformer.ln_f.eps)
            checks = dict(block01_matches=bool(torch.equal(captures["block01"], output.hidden_states[1])),
                          block06_matches=bool(torch.equal(captures["block06"], output.hidden_states[6])),
                          final_ln_matches=bool(torch.equal(captures["final_ln"], output.hidden_states[-1])),
                          preln_to_final_max_abs=float((preln-captures["final_ln"]).abs().max()))
            # logit[t-1] predicts token[t]; record both terms used in the subtraction.
            logits = output.logits[0, :-1].double()
            targets = ids[0, 1:].to("cuda")
            target = logits.gather(1, targets[:, None]).squeeze(1)
            logz = torch.logsumexp(logits, dim=1)
            ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
            checks["surprisal_ce_max_abs"] = float((ce - (logz-target)).abs().max())
            arrays = {"h_"+name: value[0].float().cpu().numpy().copy() for name, value in captures.items() if name in cfg["points"]}
            for key, values in (("target_logit", target), ("logsumexp", logz), ("surprisal", logz-target)):
                arrays[key] = np.r_[np.nan, values.cpu().numpy()].astype(np.float64)
    finally:
        for handle in handles:
            handle.remove()
    token_ids = ids[0].numpy().astype(np.int64)
    tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist())
    decoded = [tokenizer.decode([int(i)], clean_up_tokenization_spaces=False) for i in token_ids]
    arrays.update(token_id=token_ids, position=np.arange(len(token_ids), dtype=np.int64),
                  mask=np.ones(len(token_ids), np.int64),
                  token_chars=np.array([len(s) for s in decoded], np.int64),
                  token_utf8_bytes=np.array([len(s.encode()) for s in decoded], np.int64),
                  logit_source_index=np.arange(-1, len(token_ids)-1, dtype=np.int64))
    return arrays, dict(tokens=tokens, decoded_tokens=decoded, extraction_checks=checks,
                        input_ids_sha256=disk.sha(token_ids.tobytes()))


def main(root, stop_after=128):
    root = Path(root)
    attempt = uuid.uuid4().hex[:12]
    started = time.monotonic()
    disk.require(sys.platform == "linux" and Path("/content").is_dir(), "Extraction requires Colab GPU")
    m = disk.check_run(root)
    done = disk.committed(root)
    cfg = disk.spec()
    fit_sha = None
    if stop_after > cfg["train_documents"]:
        from p2_fit import check_frozen_fit
        check_frozen_fit(root, done=done)
        fit_sha = disk.digest(root/"fit"/"COMMIT.json")
    if len(done) >= stop_after:
        disk.event(root, attempt, "NOOP", completed=len(done), forward_calls=0)
        print("P2: no new forward calls; committed documents verified.", flush=True)
        return 0
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        from p2_numeric import save_npz, validate_raw
        cfg = disk.spec()
        for n, version in cfg["pins"].items():
            disk.require(importlib.metadata.version(n) == version, "Dependency pin mismatch: " + n)
        disk.require(torch.cuda.is_available(), "CUDA is unavailable; no CPU fallback")
        torch.cuda.init()
        torch.set_num_threads(2)
        torch.manual_seed(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        env = hardware(torch)
        disk.write(root / "attempts" / attempt / "environment.json", env)
        disk.require("A100" in env["gpu_name"] or
                     ("RTX PRO 6000" in env["gpu_name"] and "Blackwell" in env["gpu_name"]),
                     "This release is scoped to A100 or RTX PRO 6000 Blackwell")
        previous = root / "environment_contract.json"
        if previous.exists():
            require_compatible(disk.read(previous), env)
        else:
            disk.write(previous, compatibility(env), immutable=True)
        # Capability must actually execute; cuda.is_available alone is insufficient on a new architecture.
        x = torch.eye(16, device="cuda", dtype=torch.float32)
        disk.require(torch.equal(x @ x, x), "GPU matmul preflight")
        torch.cuda.synchronize()
        disk.event(root, attempt, "GPU_PREFLIGHT_PASS", scientific_forward_calls=0)
        wanted = ["config.json", "generation_config.json", "model.safetensors",
                  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
        folder = Path(snapshot_download(cfg["model"], revision=cfg["revision"],
                      allow_patterns=wanted, token=False))
        inventory = {p.name: dict(bytes=p.stat().st_size, sha256=disk.digest(p))
                     for p in sorted(folder.iterdir()) if p.is_file() and p.name in wanted}
        disk.require(set(inventory) == set(wanted), "Model/tokenizer file inventory")
        disk.write(root / "model_files.json", inventory, immutable=True)
        tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True, local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, dtype=torch.float32,
                    attn_implementation="eager").to("cuda").eval()
        model.requires_grad_(False)
        disk.require(model.config.n_layer == 12 and model.config.n_embd == 768 and
                     model.config.vocab_size == 50257, "Unexpected GPT-2 architecture")
        disk.require(all(p.dtype == torch.float32 and p.device.type == "cuda" for p in model.parameters()),
                     "Model precision/device")
        record_model_config(root, attempt, model.config.to_dict())
        for fixture in disk.fixtures():
            doc_id = fixture["id"]
            if doc_id in done:
                validate_raw(root, doc_id)
                disk.event(root, attempt, "SKIPPED_COMMITTED", document_id=doc_id)
                continue
            if len(done) >= stop_after:
                break
            disk.require(sum(p.stat().st_size for p in disk.files(root)) < disk.RAW_LIMIT-32*1024**2,
                         "Evidence budget reached; no new document started")
            disk.require(shutil.disk_usage(root).free > 2*1024**3,
                         "Less than 2 GiB free for evidence and ZIP recovery")
            dest = root / "docs" / doc_id
            if dest.exists():
                orphan = root / "attempts" / attempt / "incomplete_before_retry" / doc_id
                orphan.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dest, orphan)
            disk.require(fixture["split"] == "train" or fit_sha is not None, "Test extraction requires frozen fit")
            disk.event(root, attempt, "FORWARD_STARTED", document_id=doc_id, split=fixture["split"], fit_commit_sha256=fit_sha if fixture["split"] == "test" else None)
            t = time.monotonic()
            arrays, details = extract(model, tokenizer, fixture, torch, np)
            torch.cuda.synchronize()
            save_npz(dest / "raw.npz", arrays)
            meta = dict(document_id=doc_id, split=fixture["split"], fit_commit_sha256=fit_sha if fixture["split"] == "test" else None, run_id=m["run_id"], attempt=attempt, text=fixture["text"],
                        text_sha256=disk.sha(fixture["text"].encode()), model=cfg["model"],
                        revision=cfg["revision"], specification_sha256=m["specification_sha256"],
                        forward_seconds=time.monotonic()-t,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), **details)
            disk.write(dest / "doc.json", meta, immutable=True)
            validate_raw(root, doc_id)
            disk.seal_doc(dest)
            done[doc_id] = disk.read(dest / "COMMIT.json")
            disk.event(root, attempt, "DOCUMENT_COMMITTED", document_id=doc_id)
            print(f"P2: {len(done)}/{cfg['documents']} documents saved ({doc_id}).", flush=True)
        disk.write(root / "attempts" / attempt / "result.json",
                   dict(status="CHUNK_COMPLETE" if len(done) < cfg["documents"] else "EXTRACTION_COMPLETE",
                        completed=len(done), seconds=time.monotonic()-started))
        return 0
    except BaseException as exc:
        disk.event(root, attempt, "WORKER_FAILURE", error_type=type(exc).__name__, error=str(exc))
        disk.write(root / "attempts" / attempt / "error.json",
                   dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2])))
