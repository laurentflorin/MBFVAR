"""
The backward step of the simulation smoothers (MBFVAR 0.9.4).

The one-step-ahead state covariance that the Carter-Kohn backward step
inverts is singular by construction: the state stacks lagged copies of the
latent series, and the temporal-aggregation constraint is observed without
error. ``smoother_pinv`` inverts it as a Moore-Penrose inverse. These tests
check that it is the ordinary inverse when the matrix is regular, and that
with the singular covariance of an exactly observed four-week mean the
backward step gives the exact Gaussian conditional, computed here
independently from a full-rank parametrisation of the state.
"""
import unittest

import numpy as np

from MBFVAR.mfbvar_funcs import smoother_pinv


def companion(phi, extra_lags=1):
    """Transition of the state [x_t, ..., x_{t-p}] for an AR(p) latent series
    carried with ``extra_lags`` more lags than the AR order, as the blocks'
    states are (p+1 entries for p lags)."""
    p = len(phi)
    k = p + extra_lags
    G = np.zeros((k, k))
    G[0, :p] = phi
    G[1:, :-1] = np.eye(k - 1)
    return G


class TestSmootherPinv(unittest.TestCase):
    def test_is_the_inverse_of_a_regular_matrix(self):
        rng = np.random.default_rng(0)
        A = rng.normal(size=(6, 6))
        P = A @ A.T + 0.1 * np.eye(6)
        np.testing.assert_allclose(smoother_pinv(P), np.linalg.inv(P), rtol=1e-9, atol=1e-12)

    def test_is_the_moore_penrose_inverse_of_a_singular_matrix(self):
        rng = np.random.default_rng(1)
        L = rng.normal(size=(6, 3))
        P = L @ L.T
        np.testing.assert_allclose(smoother_pinv(P), np.linalg.pinv(P, hermitian=True),
                                   rtol=1e-8, atol=1e-10)

    def test_backward_step_is_the_exact_conditional_under_an_exact_aggregate(self):
        """x follows a weekly AR(4) with a tiny innovation variance (the
        scale of the growth-rate runs); its four-week mean is observed
        exactly at t, which removes one dimension from the filtered
        covariance and, through the lag copies, from the one-step-ahead
        covariance. The backward step's mean and covariance must equal the
        exact conditional of alpha_t given alpha_{t+1}."""
        rng = np.random.default_rng(2)
        phi = np.array([0.26, 1.2, -0.29, -0.31])        # weekly GDP equation of the failed run
        G = companion(phi)
        k = G.shape[0]
        s2 = 2.2e-5                                      # innovation variance
        u = np.zeros(k)
        u[0] = 1.0
        c = np.zeros(k)
        c[0] = 6.5e-4

        # filtered moments at t: a regular prior updated with the exact
        # observation h' alpha_t = y (mean of the four current weeks)
        B = rng.normal(scale=3e-3, size=(k, k))
        P0 = B @ B.T + 1e-6 * np.eye(k)
        a0 = rng.normal(scale=1e-2, size=k)
        h = np.r_[np.full(4, 0.25), np.zeros(k - 4)]
        y = 0.004
        g = P0 @ h / (h @ P0 @ h)
        a = a0 + g * (y - h @ a0)
        P = P0 - np.outer(g, h @ P0)
        P = 0.5 * (P + P.T)

        # exact representation: alpha_t = a + L z with z ~ N(0, I_r)
        w, V = np.linalg.eigh(P)
        keep = w > 1e-12 * w.max()
        L = V[:, keep] * np.sqrt(w[keep])
        r = L.shape[1]
        self.assertEqual(r, k - 1)

        # a consistent alpha_{t+1}
        z = rng.standard_normal(r)
        e = np.sqrt(s2) * rng.standard_normal()
        alpha_next = G @ (a + L @ z) + c + u * e

        # exact conditional of (z, e) given alpha_{t+1}, from the whitened map
        K = np.column_stack((G @ L, u * np.sqrt(s2)))    # alpha_{t+1} - G a - c = K (z, e/sqrt(s2))
        K_pinv = np.linalg.pinv(K)
        rhs = alpha_next - G @ a - c
        np.testing.assert_allclose(K @ (K_pinv @ rhs), rhs, atol=1e-12)   # consistent draw
        mean_w = K_pinv @ rhs
        cov_w = np.eye(r + 1) - K_pinv @ K
        exact_mean = a + L @ mean_w[:r]
        exact_cov = L @ cov_w[:r, :r] @ L.T

        # the smoothers' backward step, on the filtered covariance as a long
        # filter recursion leaves it: exact up to round-off of 1e-14
        # relative, which turns its zero eigenvalue into a tiny non-zero one
        # (the run that failed had relative eigenvalues of about 1e-14 there)
        E = rng.normal(size=(k, k))
        P_rec = P + 1e-14 * np.abs(P).max() * (E + E.T)
        Phat = G @ P_rec @ G.T + s2 * np.outer(u, u)
        Phat = 0.5 * (Phat + Phat.T)
        ev = np.linalg.eigvalsh(Phat)
        self.assertLess(abs(ev[0]) / ev[-1], 1e-12)      # singular up to round-off
        inv_Phat = smoother_pinv(Phat)
        temp = P_rec @ G.T
        Amean = a + temp @ inv_Phat @ (alpha_next - G @ a - c)
        Pmean = P_rec - temp @ inv_Phat @ temp.T

        scale = np.sqrt(np.max(np.diag(P)))
        np.testing.assert_allclose(Amean, exact_mean, atol=1e-8 * scale)
        np.testing.assert_allclose(Pmean, exact_cov, atol=1e-8 * scale ** 2)
        # and the conditional honours the lag copies exactly: alpha_t's first
        # k-1 entries are alpha_{t+1}'s last k-1
        np.testing.assert_allclose(Amean[:-1], alpha_next[1:], atol=1e-10 * scale)
        self.assertLess(np.abs(Pmean[:-1, :-1]).max(), 1e-10 * scale ** 2)


if __name__ == "__main__":
    unittest.main()
