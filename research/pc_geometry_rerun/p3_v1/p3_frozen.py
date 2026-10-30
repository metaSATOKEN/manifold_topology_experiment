"""Read immutable P2/EDA-0001 models. No fitting operation exists here."""
from pathlib import Path
import numpy as np
import p3_io as disk
import p3_numeric as num


def split_ids(split):
    return [d["id"] for d in disk.fixtures() if d["split"] == split]


def check_frozen_fit(root):
    disk.check_run(root)
    commit = disk.read(Path(root)/"fit"/"COMMIT.json")
    manifest = disk.read(disk.HERE/"frozen_manifest.json")
    disk.require(commit["new_training_performed"] is False and
        commit["source_manifest_sha256"] == disk.digest(disk.HERE/"frozen_manifest.json") and
        commit["files"] == manifest["files"], "Fixed model reference differs")
    for name, sha in commit["files"].items():
        disk.require(disk.digest(disk.HERE/name) == sha and
                     disk.digest(Path(root)/"sources"/name) == sha, "Frozen model changed: "+name)
    return commit


def models(root):
    check_frozen_fit(root)
    result = {}
    for point in disk.spec()["points"]:
        m = num.load_npz(disk.HERE/f"frozen_{point}_C1.npz")
        n = len(m["vocab"])
        expected = {"beta": (22, 9), "token_effect": (22, n), "theta": (21, 8),
                    "mean": (768,), "components": (8, 768), "random_components": (20, 8, 768),
                    "feature_mean": (8,), "feature_scale": (8,), "feature_active": (8,),
                    "score_mean": (168,), "score_scale": (168,), "score_active": (168,),
                    "vocab": (n,), "gammas": (21,)}
        disk.require(set(m) == set(expected), "Frozen model field inventory")
        for k, shape in expected.items():
            disk.require(m[k].shape == shape and np.isfinite(m[k]).all(), "Frozen parameter: "+k)
        disk.require(np.all(np.diff(m["vocab"]) > 0) and
                     np.all(m["feature_scale"] > 0) and np.all(m["score_scale"] > 0),
                     "Frozen vocabulary/scales")
        result[(point, "C1")] = m
    counts = num.load_npz(disk.HERE/"frozen_frequency.npz")["counts"]
    disk.require(counts.shape == (50257,) and counts.sum() == 64*512 and np.all(counts >= 0),
                 "Frozen training frequency")
    return result, counts


def predict(raw, point, m, counts):
    """Return fixed readout predictions and scores; evaluation labels are not read."""
    cfg = disk.spec()
    idx = np.arange(1, cfg["tokens"])
    x = raw["h_"+point][idx].astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    t = raw["position"][idx].astype(float)
    p = t/(cfg["tokens"]-1)
    token = raw["token_id"][idx]
    rarity = -np.log((counts[token]+1)/(counts.sum()+len(counts)))
    b = np.column_stack([p, p*p, p**3, np.log1p(t)/np.log(cfg["tokens"]),
                         np.log1p(raw["token_chars"][idx]),
                         np.log1p(raw["token_utf8_bytes"][idx]), rarity, np.log1p(norms)])
    z = np.column_stack([np.ones(len(idx)),
        (b-m["feature_mean"])/m["feature_scale"]*m["feature_active"]])
    bases = np.concatenate([m["components"], m["random_components"].reshape(-1, cfg["hidden"])])
    score = (x-m["mean"]) @ bases.T
    w = (score-m["score_mean"])/m["score_scale"]*m["score_active"]
    ix = np.searchsorted(m["vocab"], token)
    known = ix < len(m["vocab"])
    ix = np.minimum(ix, len(m["vocab"])-1)
    known &= m["vocab"][ix] == token
    prediction = z @ m["beta"].T + m["token_effect"][:, ix].T*known[:, None]
    for group in range(21):
        prediction[:, group+1] += w[:, group*8:group*8+8] @ m["theta"][group]
    disk.require(np.isfinite(prediction).all(), "Nonfinite fixed predictions")
    return prediction.T, score[:, :8], known
