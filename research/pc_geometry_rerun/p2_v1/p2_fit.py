"""Train-only PCA and probes. Never fits, selects or standardizes on test data."""
import os
from pathlib import Path
import uuid
import numpy as np
import p2_io as disk
import p2_numeric as num

HEADS = ("surprisal", "log_chars")


def split_ids(split):
    return [d["id"] for d in disk.fixtures() if d["split"] == split]


def prepare(raw, point, condition):
    index = np.arange(1 if condition == "C1" else 0, len(raw["position"]))
    x = raw["h_" + point][index].astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    if condition == "C4":
        if np.any(norms == 0):
            return None, index, norms, "zero_norm_unit_conversion"
        x = x / norms[:, None]
    return x, index, norms, None


def frequency(raws):
    ids = np.concatenate([r["token_id"] for r in raws])
    return np.bincount(ids, minlength=disk.spec()["vocabulary_size"]).astype(np.int64)


def controls(raw, index, norms, counts, head):
    cfg = disk.spec()
    t = raw["position"][index].astype(float)
    pos = t / (cfg["tokens"] - 1)
    freq = -np.log((counts[raw["token_id"][index]] + cfg["frequency_smoothing"]) /
                  (counts.sum() + cfg["frequency_smoothing"] * len(counts)))
    columns = [pos, pos**2, pos**3, np.log1p(t)/np.log(cfg["tokens"])]
    if head == "surprisal":
        columns += [np.log1p(raw["token_chars"][index]),
                    np.log1p(raw["token_utf8_bytes"][index])]
        y = raw["surprisal"][index]
    else:
        y = np.log1p(raw["token_chars"][index])
    return np.column_stack(columns + [freq, np.log1p(norms)]), y


def random_bases(d, k, point, condition):
    result = []
    for rep in range(disk.spec()["random_subspaces"]):
        seed = int(disk.sha(f"{disk.spec()['random_seed']}:{point}:{condition}:{rep}".encode())[:16], 16)
        q, _ = np.linalg.qr(np.random.Generator(np.random.PCG64(seed)).normal(size=(d, k)),
                            mode="reduced")
        basis = q.T.copy()
        for v in basis:
            if v[np.argmax(abs(v))] < 0:
                v *= -1
        result.append(basis)
    return np.stack(result)


def ridge(x, y):
    mu = x.mean(0)
    sd = x.std(0)
    active = sd > 1e-12
    sd = np.where(active, sd, 1.)
    z = (x-mu)/sd * active
    ym = float(y.mean())
    coef = np.linalg.solve(z.T@z/len(y) + disk.spec()["ridge_alpha"]*np.eye(z.shape[1]),
                           z.T@(y-ym)/len(y))
    return mu, sd, active.astype(np.int64), coef, ym


def predict(x, mu, sd, active, coef, ym):
    return ((x-mu)/sd * active) @ coef + ym


def select_position(correlations):
    # [groups, articles, components]. A test article can never enter this array.
    selections = []
    for group in correlations:
        medians, ns = [], []
        for column in group.T:
            v = column[np.isfinite(column)]
            ns.append(len(v))
            medians.append(float(np.median(v)) if len(v) >= disk.spec()["minimum_valid_documents"] else None)
        candidates = [i for i, v in enumerate(medians) if v is not None]
        if not candidates:
            selections.append(dict(component=None, sign=None, training_median=None,
                                   valid_training_articles=0))
        else:
            i = max(candidates, key=lambda j: abs(medians[j]))
            selections.append(dict(component=i+1, sign=1 if medians[i]>=0 else -1,
                                   training_median=abs(medians[i]), valid_training_articles=ns[i]))
    return selections


def fit_condition(raws, counts, point, condition, audit_arrays=None):
    cfg = disk.spec()
    prepared = [prepare(r, point, condition) for r in raws]
    reason = next((z[3] for z in prepared if z[3]), None)
    if reason:
        return {}, dict(valid=False, reason=reason, point=point, condition=condition)
    x = np.concatenate([z[0] for z in prepared])
    mean = x.mean(0)
    centered = x-mean
    cov = centered.T@centered/(len(x)-1)
    total = float(np.trace(cov))
    if total <= 0:
        return {}, dict(valid=False, reason="zero_training_variance", point=point, condition=condition)
    if audit_arrays is None:
        eigenvalues, vectors = np.linalg.eigh(cov)
        eigenvalues = eigenvalues[::-1].copy()
        basis = vectors[:, ::-1][:, :cfg["components"]].T.copy()
        for v in basis:
            if v[np.argmax(abs(v))] < 0:
                v *= -1
    else:
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        basis = audit_arrays["components"]
        np.testing.assert_allclose(audit_arrays["mean"], mean, atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(audit_arrays["eigenvalues"], eigenvalues, atol=1e-8, rtol=1e-7)
        np.testing.assert_allclose(basis@basis.T, np.eye(cfg["components"]), atol=1e-8, rtol=1e-7)
        residual = cov@basis.T - basis.T*eigenvalues[:cfg["components"]]
        disk.require(np.max(abs(residual))/max(total,1) < 1e-8, "Training PCA eigen-equation failed")
        disk.require(all(v[np.argmax(abs(v))] >= 0 for v in basis), "PCA sign rule changed")
    rank = int(np.sum(eigenvalues > total*1e-12))
    if rank < cfg["components"]:
        return {}, dict(valid=False, reason="training_rank_below_components", rank=rank,
                        point=point, condition=condition)
    random = random_bases(x.shape[1], cfg["components"], point, condition)
    bases = np.concatenate([basis[None], random])
    # One BLAS projection covers the PCA and all twenty random subspaces.
    z = (centered @ bases.reshape(-1,x.shape[1]).T).reshape(len(x),len(bases),cfg["components"])
    arrays = dict(mean=mean, components=basis, eigenvalues=eigenvalues, random_components=random)
    lengths = [len(v[1]) for v in prepared]
    offsets = np.r_[0, np.cumsum(lengths)]
    corr = np.full((len(bases),len(raws),cfg["components"]),np.nan)
    for j, raw in enumerate(raws):
        position = raw["position"][prepared[j][1]]
        for group in range(len(bases)):
            for pc in range(cfg["components"]):
                r = num.correlation(z[offsets[j]:offsets[j+1],group,pc], position)
                if r is not None:
                    corr[group,j,pc] = r
    arrays["training_position_correlations"] = corr
    meta = dict(valid=True, point=point, condition=condition, rank=rank,
                training_rows=len(x), total_variance=total, position_selection=select_position(corr),
                heads={}, pooling="Training-global mean; no article-wise PCA or test centering")
    for head in HEADS:
        data = [controls(raw,index,norms,counts,head)
                for raw,(_,index,norms,_) in zip(raws,prepared)]
        b = np.concatenate([v[0] for v in data])
        y = np.concatenate([v[1] for v in data])
        ok = np.isfinite(y)
        if ok.sum()<3 or np.ptp(y[ok]) == 0:
            meta["heads"][head] = dict(valid=False, reason="constant_or_insufficient_training_target",
                                       rows=int(ok.sum()))
            continue
        fitted=[]
        # First model is controls only; zero PC slots preserve a rectangular artifact.
        for model in range(len(bases)+1):
            scores = np.zeros((len(x),cfg["components"])) if model==0 else z[:,model-1,:]
            fitted.append(ridge(np.column_stack([b,scores])[ok],y[ok]))
        for j,name in enumerate(("x_mean","x_scale","active","coef","y_mean")):
            arrays[head+"__"+name] = np.asarray([m[j] for m in fitted])
        meta["heads"][head] = dict(valid=True, rows=int(ok.sum()), controls=b.shape[1],
                                   model_order="baseline, PCA, random_0..random_19")
    return arrays, meta


def check_frozen_fit(root, done=None):
    root=Path(root)
    if done is None:
        done=disk.committed(root)
    folder=root/"fit"
    disk.require((folder/"COMMIT.json").is_file(), "Test extraction requires sealed training fit")
    commit=disk.read(folder/"COMMIT.json")
    actual={str(p.relative_to(folder)) for p in disk.files(folder)}-{"COMMIT.json"}
    disk.require(actual==set(commit["files"]), "Fitted artifact inventory differs")
    for name,h in commit["files"].items():
        disk.require(disk.digest(folder/disk.safe_name(name))==h, "Fitted artifact changed: "+name)
    meta=disk.read(folder/"meta.json")
    disk.require(meta["capsule"]==disk.capsule() and meta["train_ids"]==split_ids("train"), "Fit identity")
    disk.require(meta["train_commits"]=={d:done.get(d) for d in split_ids("train")},
                 "Training observations changed or missing")
    disk.require(all(v is not None for v in meta["train_commits"].values()), "Incomplete training")
    return meta


def seal_fit(root):
    root=Path(root)
    done=disk.committed(root)
    if (root/"fit"/"COMMIT.json").exists():
        return check_frozen_fit(root,done)
    disk.require(not (set(done)&set(split_ids("test"))), "Cannot fit after test extraction")
    disk.require(set(split_ids("train"))<=set(done), "All training documents required")
    attempt="fit_"+uuid.uuid4().hex[:12]
    folder=root/"fit"
    if folder.exists():
        dest=root/"attempts"/attempt/"incomplete_fit_before_retry"
        dest.parent.mkdir(parents=True,exist_ok=True)
        os.replace(folder,dest)
    folder.mkdir()
    raws=[num.validate_raw(root,d)[0] for d in split_ids("train")]
    counts=frequency(raws)
    num.save_npz(folder/"frequency.npz",dict(counts=counts))
    for point in disk.spec()["points"]:
        for c in disk.spec()["conditions"]:
            arrays,meta=fit_condition(raws,counts,point,c)
            dest=folder/(point+"__"+c)
            num.save_npz(dest/"arrays.npz",arrays)
            disk.write(dest/"meta.json",meta,immutable=True)
            print(f"P2 training fit saved: {point}/{c}, valid={meta['valid']}",flush=True)
    meta=dict(capsule=disk.capsule(),train_ids=split_ids("train"),
              train_commits={d:done[d] for d in split_ids("train")},completed_at=disk.stamp(),
              fitted_before_test_extraction=True, test_documents_read=0)
    disk.write(folder/"meta.json",meta,immutable=True)
    disk.write(folder/"COMMIT.json",dict(sealed_at=disk.stamp(),files={
        str(p.relative_to(folder)):disk.digest(p) for p in disk.files(folder)}),immutable=True)
    disk.event(root,attempt,"FIT_SEALED",fit_commit_sha256=disk.digest(folder/"COMMIT.json"),
               training_documents=len(raws),test_documents_read=0)
    return meta


def audit_fit(root):
    root=Path(root)
    check_frozen_fit(root)
    raws=[num.validate_raw(root,d)[0] for d in split_ids("train")]
    counts=frequency(raws)
    np.testing.assert_array_equal(counts,num.load_npz(root/"fit"/"frequency.npz")["counts"])
    for point in disk.spec()["points"]:
        for c in disk.spec()["conditions"]:
            folder=root/"fit"/(point+"__"+c)
            stored=num.load_npz(folder/"arrays.npz")
            oldmeta=disk.read(folder/"meta.json")
            expected,meta=fit_condition(raws,counts,point,c,audit_arrays=stored if oldmeta["valid"] else None)
            num.compare_tree(oldmeta,meta)
            disk.require(set(stored)==set(expected),"Fit array inventory")
            for k in expected:
                np.testing.assert_allclose(stored[k],expected[k],atol=1e-8,rtol=1e-7,equal_nan=True)
            print(f"P2 training fit verified: {point}/{c}",flush=True)
