"""Small meaningful numeric controls before touching P4 derived outcomes."""
import numpy as np
import numeric as n


def test_group_center_and_equal_id():
    ids = np.array([7, 2, 7, 9, 2, 2])
    x = np.array([4., 1., 8., 50., 2., 9.])
    c, repeat, counts = n.center(x, ids)
    np.testing.assert_allclose(c, [-2, -3, 2, 0, -2, 5])
    assert repeat.tolist() == [True, True, True, False, True, True]
    for i in [2, 7]:
        assert abs((1 / counts[ids == i]).sum() - 1) < 1e-12


def test_variable_dimension_cv_against_direct_search():
    rng = np.random.default_rng(12)
    z, w = rng.normal(size=(84, 3)), rng.normal(size=(84, 2, 7))
    y = z[:, 1] + w[:, 0, 4] * .7 + rng.normal(size=84) * .2
    folds = np.arange(84) % 4
    model, cv = n.cv_block(z, w, y, folds, np.repeat(.01, 3))
    direct = np.zeros((5, 2))
    for g, gamma in enumerate(n.GRID):
        for j in range(2):
            x = np.column_stack([z, w[:, j]])
            for fold in range(4):
                va = folds == fold
                pen = np.r_[np.repeat(.01, 3), np.repeat(gamma, 7)]
                a = np.vstack([x[~va], np.diag(np.sqrt((~va).sum() * pen))])
                b = np.r_[y[~va], np.zeros(10)]
                beta = np.linalg.lstsq(a, b, rcond=None)[0]
                direct[g, j] += np.square(x[va] @ beta - y[va]).sum() / len(y)
    np.testing.assert_allclose(direct, cv["mse"], atol=1e-11)
    np.testing.assert_array_equal(model["gamma"], n.GRID[np.argmin(direct, axis=0)])


def test_random_errors_are_not_ensemble_prediction_error():
    y = np.array([1., -1.])
    pred = {k: np.column_stack([y, np.tile([2., -2.], (2, 10))]) for k in n.FAMILIES}
    pred.update({k: np.zeros((2, 1)) for k in ["P", "B", "N"]})
    deltas = n.token_contrasts(n.errors(y, pred))
    assert np.all(deltas["A8_adv"] == 5.)  # Mean squared error, not MSE of mean prediction (=1).
    metrics, _ = n.score(y, pred)
    assert abs(deltas["A8_adv"].sum() / (y @ y) - metrics["A8_adv"]) < 1e-12


def test_unicode_partition():
    assert [n.token_type(s) for s in ["  ", "12.3", "abc42", "!!", "\u0301"]] == [0, 1, 2, 3, 4]


if __name__ == "__main__":
    for f in [test_group_center_and_equal_id, test_variable_dimension_cv_against_direct_search,
              test_random_errors_are_not_ensemble_prediction_error, test_unicode_partition]:
        f()
        print(f.__name__, "PASS")
