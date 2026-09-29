"""
Regression tests for the screened explosive-root check.

``is_explosive`` must take the same decision as the eigenvalue check
``_is_explosive_eig`` on every input, so that seeded Gibbs runs draw the same
candidates in the same order and produce identical chains.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

import MBFVAR
import MBFVAR._estimation as estimation
from MBFVAR.mfbvar_funcs import _is_explosive_eig, is_explosive


@contextlib.contextmanager
def silence_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


@contextlib.contextmanager
def reference_check():
    original = estimation.is_explosive
    estimation.is_explosive = _is_explosive_eig
    try:
        yield
    finally:
        estimation.is_explosive = original


def near_unit_root_data(seed=7, n_months=120):
    """Random walks in levels: almost every coefficient draw is explosive by a
    hair, which is the case the screen is built for."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    walk = np.cumsum(rng.normal(scale=0.3, size=(n_months, 2)), axis=0)
    monthly = pd.DataFrame({"m_1": walk[:, 0], "m_2": 0.5 * walk[:, 0] + walk[:, 1]}, index=index)
    quarterly = monthly["m_1"].groupby(pd.PeriodIndex(monthly.index, freq="Q")).mean().to_timestamp(how="end")
    quarterly = pd.DataFrame({"q_1": quarterly + rng.normal(scale=0.05, size=len(quarterly))})
    return MBFVAR.mbfvar_data([quarterly, monthly], [np.array([1]), np.array([1, 1])], ["Q", "M"])


class TestExplosiveCheck(unittest.TestCase):
    def test_matches_eigenvalue_check(self):
        rng = np.random.default_rng(3)
        n, p = 4, 3
        cases = []
        for scale in (0.0005, 0.002, 0.01, 0.05, 0.3):
            for _ in range(80):
                phi = rng.normal(0, scale, (n * p + 1, n))
                phi[:n, :] += np.eye(n)  # unit roots perturbed in both directions
                cases.append(phi)
        decisions = [(bool(_is_explosive_eig(c, n, p)), bool(is_explosive(c, n, p))) for c in cases]
        self.assertTrue(all(a == b for a, b in decisions))
        self.assertTrue(0 < sum(a for a, _ in decisions) < len(decisions))

    def test_complex_and_negative_roots_fall_through_to_eigenvalues(self):
        n, p = 4, 1
        rotation = np.zeros((n * p + 1, n))
        theta, radius = 0.4, 1.001  # complex pair just outside the unit circle
        rotation[:2, :2] = radius * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotation[2, 2] = rotation[3, 3] = 0.5
        negative = np.zeros((n * p + 1, n))
        negative[:n, :] = np.diag([-1.002, 0.3, 0.2, 0.1])  # real root below -1
        stable = np.zeros((n * p + 1, n))
        stable[:n, :] = np.diag([0.999, 0.5, -0.4, 0.2])
        for phi, expected in ((rotation, True), (negative, True), (stable, False)):
            self.assertEqual(bool(is_explosive(phi, n, p)), expected)
            self.assertEqual(bool(_is_explosive_eig(phi, n, p)), expected)

    def test_seeded_chain_is_unchanged(self):
        data = near_unit_root_data()
        hyp = [[0.09, 4.3, 1, 2.7, 4.3]]
        draws = {}
        for name in ("screened", "reference"):
            model = MBFVAR.MixedFrequencyBVAR(12, 0.5, [3], 1)
            ctx = reference_check() if name == "reference" else contextlib.nullcontext()
            with ctx, silence_output():
                model.fit(data, hyp, seed=11)
            draws[name] = (model.Phip_list[-1].copy(), model.lstate_list[-1].copy())
        self.assertTrue(np.array_equal(draws["screened"][0], draws["reference"][0]))
        self.assertTrue(np.array_equal(draws["screened"][1], draws["reference"][1]))


if __name__ == "__main__":
    unittest.main()
