import numpy as np
import numeric as n
from analyze import OUT, write

rng = np.random.default_rng(20261030)
t = np.linspace(-1, 1, 400)
q = np.linalg.qr(rng.normal(size=(16, 16)))[0]
folds = np.arange(400)*5//400
out = {}
for name in ["line", "arc", "outlier_cloud", "cloud"]:
    x = rng.normal(scale=.01 if name in ["line","arc"] else 1, size=(400,16))
    if name in ["line","arc"]:
        x[:,0] += t*4
    if name == "arc":
        x[:,1] += 2*t*t
    if name == "outlier_cloud":
        x[0,0] += 500
    x = x @ q
    p,y = n.pca(x)
    c = n.curve_analysis(y,p,folds)
    occ,gap = n.spacing(p["scores"][:,0])
    share = np.max(p["scores"][:,0]**2)/np.sum(p["scores"][:,0]**2)
    match = occ >= .8 and gap <= .1 and share <= .5 and c["curve_q"][1,0]>=.9
    out[name] = dict(occupancy=occ,gap=gap,max_pc1_share=share,
                    q=c["curve_q"],graph=bool(match))
assert out["line"]["graph"] and out["arc"]["graph"]
assert not out["outlier_cloud"]["graph"] and not out["cloud"]["graph"]
assert out["arc"]["q"][1,0]-out["arc"]["q"][0,0]>.01
assert out["line"]["q"][1,0]-out["line"]["q"][0,0]<.01
write(OUT/"synthetic_checks.json",dict(status="PASS",cases=out))
print("Synthetic fixed cases PASS")
