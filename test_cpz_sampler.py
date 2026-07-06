"""
Lightweight checks for the CPZ mixed-frequency sampler helpers.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_cpz_funcs():
    spec = importlib.util.spec_from_file_location(
        "cpz_funcs", ROOT / "MBFVAR" / "_cpz_funcs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCPZSamplerHelpers(unittest.TestCase):
    def test_construct_minnesota_returns_prior_precision(self):
        cpz = load_cpz_funcs()

        invVbeta = cpz.construct_minnesota(
            AR_s2=np.array([1.0]),
            n=1,
            lag=1,
            theta=(0.04, 0.25, 100.0, 2.0),
        )

        self.assertTrue(np.allclose(invVbeta, np.array([25.0, 0.01])))

    def test_missing_high_frequency_entry_is_latent_and_samples_finitely(self):
        cpz = load_cpz_funcs()
        np.random.seed(123)

        n, Nm, Nq, T = 2, 1, 1, 4
        lag, r, nQ = 1, 2, 2
        Y_obs = np.array([[1.0], [2.0], [3.0], [np.nan]])

        M_o, M_u, obs_idx, unobs_idx = cpz.build_selection_matrices(
            n, Nm, T, Y_obs=Y_obs)

        missing_full_idx = (T - 1) * n
        self.assertNotIn(missing_full_idx, set(obs_idx))
        self.assertIn(missing_full_idx, set(unobs_idx))

        M_a, con_index = cpz.build_intertemporal_constraint(
            n, Nm, Nq, T, r, nQ, unobs_idx, temp_agg="mean")
        Y = cpz.sample_latent_states(
            Y_obs=Y_obs,
            beta=np.zeros((n * lag + 1, n)),
            invSig=np.eye(n),
            h=np.zeros(T - lag),
            n=n,
            Nm=Nm,
            Nq=Nq,
            lag=lag,
            r=r,
            nQ=nQ,
            Y_con=np.zeros(len(con_index)),
            M_o=M_o,
            M_u=M_u,
            M_a=M_a,
            obs_idx=obs_idx,
        )

        self.assertTrue(np.isfinite(Y).all())
        self.assertEqual(Y[0, 0], 1.0)
        self.assertEqual(Y[1, 0], 2.0)
        self.assertEqual(Y[2, 0], 3.0)

    def test_cpz_estimator_stores_mh_correction_state(self):
        source = (ROOT / "MBFVAR" / "_estimation_cpz.py").read_text()

        self.assertIn("mh_accept_counts", source)
        self.assertIn("mh_total_counts", source)
        self.assertIn("mh_accept_rates", source)
        self.assertIn("_cpz_downstream_loglik", source)
        self.assertIn("for _m in range(M - 1)", source)


if __name__ == "__main__":
    unittest.main()
