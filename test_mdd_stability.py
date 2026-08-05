"""
Regression tests for marginal-data-density numerical stability.
"""
import contextlib
import importlib.util
import io
import math
import random
import unittest

import numpy as np
import pandas as pd
from scipy.special import loggamma
from scipy.stats import uniform

import MBFVAR
from MBFVAR.mfbvar_funcs import mdd_, varprior
from MBFVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse


class DummyData:
    frequencies = ["Q", "M"]


class NonFiniteMDDModel(MBFVAR.MixedFrequencyBVAR):
    def fit(self, *args, **kwargs):
        return np.nan


@contextlib.contextmanager
def silence_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def make_mixed_frequency_data(seed=123, n_months=72):
    rng = np.random.default_rng(seed)
    monthly_index = pd.date_range("2000-01-31", periods=n_months, freq="ME")

    monthly_base = np.zeros(n_months)
    for idx in range(1, n_months):
        monthly_base[idx] = 0.7 * monthly_base[idx - 1] + rng.normal(scale=0.4)

    monthly = pd.DataFrame(
        {
            "m_1": monthly_base + rng.normal(scale=0.1, size=n_months),
            "m_2": 0.5 * monthly_base + rng.normal(scale=0.1, size=n_months),
        },
        index=monthly_index,
    )
    quarterly = monthly["m_1"].groupby(pd.PeriodIndex(monthly.index, freq="Q")).mean().to_timestamp(how="end")
    quarterly = pd.DataFrame(
        {"q_1": quarterly + rng.normal(scale=0.05, size=len(quarterly))}
    )

    return MBFVAR.mbfvar_data(
        [quarterly, monthly],
        [np.array([1]), np.array([1, 1])],
        ["Q", "M"],
    )


def legacy_mdd(hyp, YY, spec):
    nlags_ = int(spec[0])
    T0 = int(spec[1])
    nex_ = int(spec[2])
    nv = int(spec[3])
    nobs = int(spec[4])

    YY0 = YY[:int(T0 + 16), :]
    ybar = np.mean(YY0, axis=0)[:, np.newaxis]
    sbar = np.std(YY0, axis=0, ddof=1)[:, np.newaxis]
    premom = np.hstack((ybar, sbar))

    YYdum, XXdum = varprior(nv, nlags_, nex_, hyp, premom)
    YYact = YY[T0:T0 + nobs, :]
    XXact = np.zeros((nobs, nv * nlags_))

    for idx in range(nlags_):
        XXact[:, idx * nv:(idx + 1) * nv] = YY[T0 - 1 - idx:T0 + nobs - (idx + 1)]

    XXact = np.hstack((XXact, np.ones((nobs, 1))))
    valid = np.isfinite(YYact).all(axis=1) & np.isfinite(XXact).all(axis=1)
    YYact = YYact[valid]
    XXact = XXact[valid]

    YY_full = np.transpose(np.hstack((YYdum.T, YYact.T)))
    XX_full = np.transpose(np.hstack((XXdum.T, XXact.T)))

    n_total = YY_full.shape[0]
    n_dummy = YYdum.shape[0]
    k = XX_full.shape[1]

    XXdum_cross = XXdum.T @ XXdum
    XXfull_cross = XX_full.T @ XX_full
    S0 = (YYdum.T @ YYdum) - ((YYdum.T @ XXdum) @ calculate_pseudo_inverse(XXdum_cross)) @ XXdum.T @ YYdum
    S1 = (YY_full.T @ YY_full) - ((YY_full.T @ XX_full) @ calculate_pseudo_inverse(XXfull_cross)) @ XX_full.T @ YY_full

    gam0 = 0
    gam1 = 0

    for idx in range(nv):
        gam0 += loggamma(0.5 * (n_dummy - k + 1 - (idx + 1)))
        gam1 += loggamma(0.5 * (n_total - k + 1 - (idx + 1)))

    lnpY0 = (
        -nv * (n_dummy - k) * 0.5 * np.log(math.pi)
        - (nv / 2) * np.log(np.absolute(np.linalg.det(XXdum_cross)))
        - (n_dummy - k) * 0.5 * np.log(np.absolute(np.linalg.det(S0)))
        + nv * (nv - 1) * 0.25 * np.log(math.pi)
        + gam0
    )
    lnpY1 = (
        -nv * (n_total - k) * 0.5 * np.log(math.pi)
        - (nv / 2) * np.log(np.absolute(np.linalg.det(XXfull_cross)))
        - (n_total - k) * 0.5 * np.log(np.absolute(np.linalg.det(S1)))
        + nv * (nv - 1) * 0.25 * np.log(math.pi)
        + gam1
    )
    return lnpY1 - lnpY0


class TestMDDStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_in = make_mixed_frequency_data()

    def test_mdd_slogdet_matches_legacy_when_finite(self):
        rng = np.random.default_rng(0)
        YY = rng.normal(size=(120, 8))
        spec = np.array([4, 40, 1, 8, 80])
        hyps = [
            np.array([0.1, 0.5, 1, 1.0, 1.0]),
            np.array([0.5, 2.0, 1, 1.0, 1.0]),
            np.array([5.0, 10.0, 1, 10.0, 10.0]),
            np.array([20.0, 10.0, 1, 10.0, 10.0]),
        ]

        for hyp in hyps:
            new_mdd, *_ = mdd_(hyp, YY, spec)
            old_mdd = legacy_mdd(hyp, YY, spec)

            self.assertTrue(np.isfinite(old_mdd), msg=f"Legacy formula unexpectedly non-finite for {hyp.tolist()}")
            self.assertTrue(np.isfinite(new_mdd), msg=f"Patched MDD unexpectedly non-finite for {hyp.tolist()}")
            np.testing.assert_allclose(new_mdd, old_mdd, rtol=1e-10, atol=1e-10)

    def test_fit_return_mdd_is_finite_across_loose_priors(self):
        hyp_grid = [
            [0.001, 0.01, 1, 0.01, 0.01],
            [0.09, 4.3, 1, 2.7, 4.3],
            [5.0, 8.0, 1, 5.0, 2.0],
            [20.0, 10.0, 1, 10.0, 10.0],
        ]

        for hyp in hyp_grid:
            model = MBFVAR.MixedFrequencyBVAR(nsim=8, nburn_perc=0.25, nlags=[3], thining=1)
            with silence_output():
                mdd = model.fit(
                    self.data_in,
                    hyp=[hyp],
                    var_of_interest=["q_1"],
                    temp_agg="mean",
                    return_mdd=True,
                    check_explosive=False,
                )

            self.assertTrue(np.isfinite(mdd), msg=f"Non-finite MDD for hyperparameters {hyp}: {mdd}")

    @unittest.skipUnless(importlib.util.find_spec("mango") is not None, "mango is required for this test")
    def test_update_hyperparameters_mango_completes_for_mdd(self):
        np.random.seed(0)
        random.seed(0)
        model = MBFVAR.MixedFrequencyBVAR(nsim=8, nburn_perc=0.25, nlags=[3], thining=1)
        param_space = dict(
            lambda1_1=uniform(0.001, 19.999),
            lambda2_1=uniform(0.01, 9.99),
            lambda4_1=uniform(0.01, 9.99),
            lambda5_1=uniform(0.01, 9.99),
        )

        with silence_output():
            hyp = model.update_hyperparameters_mango(
                self.data_in,
                param_space,
                init_points=1,
                n_iter=1,
                nsim=8,
                njobs=1,
                var_of_interest=["q_1"],
                temp_agg="mean",
                save=False,
            )

        self.assertEqual(len(hyp), 1)
        self.assertEqual(len(hyp[0]), 5)
        self.assertEqual(hyp[0][2], 1)
        self.assertTrue(np.isfinite(np.asarray(hyp[0], dtype=float)).all())

    @unittest.skipUnless(
        importlib.util.find_spec("mango") is not None and importlib.util.find_spec("bayes_opt") is not None,
        "mango and bayes_opt are required for this test",
    )
    def test_nonfinite_mdd_is_penalized_in_both_optimizers(self):
        np.random.seed(0)
        random.seed(0)
        model = NonFiniteMDDModel(nsim=4, nburn_perc=0.25, nlags=[3], thining=1)
        param_space = dict(
            lambda1_1=uniform(0.001, 19.999),
            lambda2_1=uniform(0.01, 9.99),
            lambda4_1=uniform(0.01, 9.99),
            lambda5_1=uniform(0.01, 9.99),
        )
        pbounds = {
            "lambda1_1": (0.001, 20.0),
            "lambda2_1": (0.01, 10.0),
            "lambda4_1": (0.01, 10.0),
            "lambda5_1": (0.01, 10.0),
        }

        with silence_output():
            mango_hyp = model.update_hyperparameters_mango(
                DummyData(),
                param_space,
                init_points=1,
                n_iter=1,
                nsim=4,
                njobs=1,
            )
            bayes_hyp = model.update_hyperparameters(
                DummyData(),
                pbounds,
                init_points=1,
                n_iter=0,
                nsim=4,
            )

        self.assertEqual(len(mango_hyp), 1)
        self.assertEqual(len(mango_hyp[0]), 5)
        self.assertEqual(mango_hyp[0][2], 1)
        self.assertEqual(len(bayes_hyp), 1)
        self.assertEqual(len(bayes_hyp[0]), 5)
        self.assertEqual(bayes_hyp[0][2], 1)


if __name__ == "__main__":
    unittest.main()