"""Independent augmented least-squares checks for the new analysis algebra."""
import numpy as np
import analyze as a

rng = np.random.default_rng(20261027)
n = 137
token = rng.integers(0, 13, n)
y = rng.normal(size=n)
z = rng.normal(size=(n, 8))
w = rng.normal(size=(n, 168))
alpha, tau = .1, 10.
gammas = np.geomspace(.0001, 1., 21)
m = a.ordinary_fit(z, w, y, token, alpha, tau, gammas)
u = np.eye(len(m["vocab"]))[np.searchsorted(m["vocab"], token)]
errors = {}
for j in range(22):
    extra = w[:, (j-1)*8:j*8] if j else np.empty((n, 0))
    design = np.column_stack([np.ones(n), z, extra, u])
    penalty = np.r_[0., np.repeat(n*alpha, 8),
                    np.repeat(n*gammas[j-1], 8) if j else [],
                    np.repeat(tau, u.shape[1])]
    ref = np.linalg.lstsq(np.vstack([design, np.diag(np.sqrt(penalty))]),
                         np.r_[y, np.zeros(len(penalty))], rcond=None)[0]
    actual = np.r_[m["beta"][j], m["theta"][j-1] if j else [], m["token_effect"][j]]
    errors[f"ordinary_{j}"] = float(abs(ref-actual).max())
    np.testing.assert_allclose(ref, actual, atol=1e-9, rtol=1e-9)
values = np.column_stack([y, z[:, a.CONTROLS], w])
c, mask, groups = a.center(token, values)
refc = values.copy()
for t in np.unique(token):
    ix = token == t
    refc[ix] -= values[ix].mean(0)
np.testing.assert_allclose(c, refc, atol=1e-12)
errors["centering"] = float(abs(c-refc).max())
yc, zc, wc = c[:, 0], c[:, 1:6], c[:, 6:]
wm = a.solve_within(a.stats(zc, wc, yc), alpha, gammas)
for j in range(22):
    design = zc if j == 0 else np.column_stack([zc, wc[:, (j-1)*8:j*8]])
    penalty = np.r_[np.repeat(n*alpha, 5),
                    np.repeat(n*gammas[j-1], 8) if j else []]
    ref = np.linalg.lstsq(np.vstack([design, np.diag(np.sqrt(penalty))]),
                         np.r_[yc, np.zeros(len(penalty))], rcond=None)[0]
    actual = np.r_[wm["beta"][j], wm["theta"][j-1] if j else []]
    errors[f"within_{j}"] = float(abs(ref-actual).max())
    np.testing.assert_allclose(ref, actual, atol=1e-9, rtol=1e-9)
q = np.linalg.qr(rng.normal(size=(768, 8)))[0].T
random = np.array([np.linalg.qr(rng.normal(size=(768, 8)))[0].T for _ in range(20)])
bases, singular = a.complement(dict(components=q, random_components=random))
errors["complement_projection"] = float(abs(bases @ q.T).max())
assert errors["complement_projection"] < 1e-12
singleton = np.array([1, 1, 2, 3, 3])
v = np.array([0., 2., 8., 1., 5.])
vc, mask, groups = a.center(singleton, v)
np.testing.assert_allclose(vc, [-1, 1, 0, -2, 2])
np.testing.assert_array_equal(mask, [1, 1, 0, 1, 1])
assert groups == 2
a.write(a.HERE/"test_results.json", dict(status="PASS", augmented_lstsq_models=44,
         centering_and_singleton_checks=True, complement_orthogonality=True,
         maximum_difference=max(errors.values()), details=errors))
print("PASS: 44 augmented least-squares comparisons, centering, singleton, complement", max(errors.values()))
