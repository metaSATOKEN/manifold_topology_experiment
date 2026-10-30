"""Nontrivial checks against a direct joint dense ridge solution."""
import unittest
import numpy as np
from analyze import Context, Features


class NumericalTests(unittest.TestCase):
    def test_joint_direct_solution_and_unknown_ids(self):
        rng = np.random.default_rng(90)
        x, xv = rng.normal(size=(240, 7)), rng.normal(size=(31, 7))
        x[:, -1] = 2
        xv[:, -1] = 20  # training constant must stay disabled
        ids, idsv = rng.integers(0, 20, 240), rng.integers(0, 25, 31)
        s, sv = rng.normal(size=(240, 168)), rng.normal(size=(31, 168))
        y = 3 + x[:, 0]*2 + np.sin(ids) + s[:, 0] + rng.normal(size=240)
        for use_id in [False, True]:
            ctx = Context(x, y, ids, xv, idsv, use_id)
            ctx.fit_base(.01, 10)
            ctx.prepare_extra(s, sv)
            pred, arrays = ctx.extra_prediction([.001]*21)
            for group in [0, 4, 20]:
                sl = slice(group*8, group*8+8)
                w = (s[:, sl]-ctx.sm[sl])/ctx.ss[sl]
                wv = (sv[:, sl]-ctx.sm[sl])/ctx.ss[sl]
                u = np.eye(20)[ids] if use_id else np.empty((240, 0))
                uv = np.eye(25)[idsv, :20] if use_id else np.empty((31, 0))
                design = np.column_stack([ctx.z, w, u])
                valid = np.column_stack([ctx.zv, wv, uv])
                penalty = np.r_[0, np.repeat(240*.01, 7),
                                np.repeat(240*.001, 8), np.repeat(10, u.shape[1])]
                coef = np.linalg.solve(design.T @ design+np.diag(penalty), design.T @ y)
                np.testing.assert_allclose(pred[:, group+1], valid @ coef, atol=2e-12, rtol=2e-12)
            self.assertLess(arrays["normal_residual"].max(), 1e-12)
            self.assertFalse(ctx.active[-1])

    def test_no_validation_fitting_and_training_only_knots(self):
        rng = np.random.default_rng(14)
        b = rng.normal(size=(100, 8))
        f = Features(b, "NL_ID")
        original = f.metadata()
        x = f.transform(b, np.zeros((100, 12)))
        self.assertEqual(x.shape, (100, 200))
        f.transform(b*1e5, np.ones((100, 12)))
        self.assertEqual(original, f.metadata())
        y, ids = rng.normal(size=100), rng.integers(0, 10, 100)
        c1 = Context(b, y, ids, b[:10], ids[:10], True)
        c2 = Context(b, y, ids, b[:10]*100, ids[:10]+100, True)
        c1.fit_base(.1, 1)
        c2.fit_base(.1, 1)
        np.testing.assert_array_equal(c1.beta, c2.beta)
        np.testing.assert_array_equal(c1.effects, c2.effects)
        self.assertTrue(np.isfinite(c2.base_prediction).all())

    def test_lexical_signal_shrinks_and_unseen_falls_back(self):
        ids = np.tile(np.arange(4), 40)
        y = np.array([-3, -1, 2, 4])[ids].astype(float)
        x = np.zeros((len(y), 1))
        c = Context(x, y, ids, np.zeros((5, 1)), np.arange(5), True)
        p = c.fit_base(.01, 1)
        self.assertLess(np.mean((p[:4]-[-3, -1, 2, 4])**2), .01)
        self.assertAlmostEqual(p[4], y.mean())
        p2 = c.fit_base(.01, 10000)
        self.assertLess(np.ptp(p2[:4]), np.ptp(p[:4]))


if __name__ == "__main__":
    unittest.main()
