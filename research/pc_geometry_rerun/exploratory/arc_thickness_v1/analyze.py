"""One fixed saved-data set; no forward pass."""
import os
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "2"
import sys
import json
import time
import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import numeric as n

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
PROJECT = BASE.parents[1]
sys.path.insert(0, str(BASE/"p4_v1"))
import p4_numeric as io


def stamp():
    return datetime.now(timezone.utc).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    return x


def write(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main():
    start = time.monotonic()
    spec = json.loads((OUT/"spec.json").read_text())
    freeze = json.loads((OUT/"FREEZE.json").read_text())
    for p, h in freeze["files"].items():
        assert hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() == h
    assert not (OUT/"execution_start.json").exists()
    write(OUT/"execution_start.json", {"at": stamp(), "source_sha256": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/"numeric.py", OUT/"analyze.py"]}})
    raw = BASE/"receipts/4b013b35152b_p4_v1_final/raw"
    rows, failed = [], []
    for model in spec["models"]:
        frequency = io.load_npz(raw/"fit"/model/"frequency.npz")["counts"]
        for di, doc in enumerate(spec["article_ids"]):
            arrays = io.load_npz(raw/"docs"/(model+"__"+doc)/"raw.npz")
            for point in spec["points"]:
                original = arrays["h_"+point].astype(np.float64)
                fullnorm = np.linalg.norm(original, axis=1)
                assert np.min(fullnorm) > 1e-12
                for condition in spec["conditions"]:
                    key = dict(model=model, point=point, condition=condition, article_id=doc)
                    dest = OUT/"results"/model/point/condition
                    dest.mkdir(parents=True, exist_ok=True)
                    try:
                        positions = np.arange(len(original))
                        x = original
                        if condition == "C1":
                            positions, x = positions[1:], x[1:]
                        if condition == "C4":
                            x = x/fullnorm[:, None]
                        radius = fullnorm[positions]
                        p, y = n.pca(x)
                        folds = np.minimum(4, (positions*5)//len(original))
                        curves = n.curve_analysis(y, p, folds)
                        features = n.feature_analysis(arrays, positions, curves["z"],
                                                      curves["thickness"], radius, frequency)
                        m = n.metrics(p, curves, features, positions)
                        ss = np.sum(y*y, axis=1)/p["total"]
                        m.update(max_centered_share=float(ss.max()),
                                 max_centered_position=int(positions[ss.argmax()]),
                                 first_centered_share=float(ss[positions == 0][0]) if condition != "C1" else None)
                        saved = {k: p[k] for k in ["mean", "components", "scores", "eigenvalues"]}
                        saved.update(curves)
                        saved.update(features)
                        saved.update(positions=positions, folds=folds, radius=radius,
                                     centered_ss_fraction=ss, total=np.array(p["total"]))
                        np.savez_compressed(dest/(doc+".npz"), **saved)
                        row = key | dict(status="OK", n=len(x), metrics=m,
                                        orth_error=p["orth_error"], eigen_error=p["eigen_error"])
                        write(dest/(doc+".json"), row)
                        rows.append(row)
                    except Exception as e:
                        failure = key | dict(status="TECHNICAL_NA", reason=str(e), traceback=traceback.format_exc())
                        failed.append(failure)
                        write(dest/(doc+"_FAILURE.json"), failure)
                    assert time.monotonic()-start < spec["limits"]["cpu_seconds"]
            if di % 16 == 15:
                print(model, di+1, "/64 saved; seconds", round(time.monotonic()-start, 1), flush=True)
    write(OUT/"article_results.json", rows)
    write(OUT/"failures.json", failed)
    summaries = {}
    for model in spec["models"]:
        for point in spec["points"]:
            for condition in spec["conditions"]:
                sub = [r for r in rows if (r["model"], r["point"], r["condition"]) == (model, point, condition)]
                group = {}
                if sub:
                    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
                    ix = rng.integers(0, len(sub), (spec["bootstrap"], len(sub)))
                    for metric in sub[0]["metrics"]:
                        values = [r["metrics"][metric] for r in sub if r["metrics"][metric] is not None]
                        if not values:
                            group[metric] = {"n": 0}
                            continue
                        v = np.array(values)
                        if v.dtype == bool:
                            group[metric] = dict(n=len(v), count=int(v.sum()), planned=64)
                        else:
                            assert len(v) == len(sub)
                            boot = np.median(v[ix], axis=1)
                            group[metric] = dict(n=len(v), median=float(np.median(v)), mean=float(v.mean()),
                                q25=float(np.quantile(v,.25)), q75=float(np.quantile(v,.75)),
                                min=float(v.min()), max=float(v.max()), positive=int((v>0).sum()),
                                ci95=np.quantile(boot,[.025,.975]).tolist())
                summaries["/".join([model,point,condition])] = group
    write(OUT/"summary.json", summaries)
    write(OUT/"execution_end.json", dict(at=stamp(), seconds=time.monotonic()-start,
            planned=1152, analyzed=len(rows), failures=len(failed), status="COMPUTED_NEEDS_VERIFICATION"))


if __name__ == "__main__":
    main()
