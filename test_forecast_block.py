"""
Forecasts of a lower block at its own frequency (MBFVAR 0.9.5).

With ``var_of_interest`` set, the monthly indicators of the quarterly-monthly
block are not passed on to the weekly block, so ``forecast()`` -- which
simulates the last block only -- produces no forecast for them. 0.9.5
stores, for every kept draw, the end of each lower block's completed data
(last row and its lag regressors) and adds ``forecast_block(m, H)``, which
iterates that block's VAR forward the way ``forecast()`` does for the last
block. These tests pin the stored state against the data, the regressor
layout against ``calc_yyact``, the date index, and that the new method
leaves the last block's forecasts unchanged.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

import MBFVAR
from MBFVAR.mfbvar_funcs import calc_yyact
from test_prior_mean import three_frequency_data

HYP = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]
HYP_CPZ = [[0.2, 0.25, 100, 2], [0.2, 0.25, 100, 2]]


def fit(method="schorfheide_song", nsim=16, seed=3, mh_proposal="palindromic"):
    model = MBFVAR.MixedFrequencyBVAR(nsim, 0.5, [3, 4], 1, seed=seed)
    kw = {} if method == "chan_poon_zhu" else {"mh_proposal": mh_proposal}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.fit(three_frequency_data(), HYP_CPZ if method == "chan_poon_zhu" else HYP,
                  var_of_interest=["q_1"], method=method, **kw)
    return model


def forecast_draws(model, H=8):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.forecast(H)
    return np.array(model.forecast_draws_list[0])


class _Checks:
    """Shared by both estimators. Block 0 is the monthly VAR [m_1, q_1]
    (p = 3); m_1 is observed in every month, so its stored end-of-sample
    values must be the data (model units: the fixture's /100)."""

    def test_stored_end_of_sample_is_the_data(self):
        m = self.model
        yy, xx = m.block_end[0]
        v = m.valid_draws
        months = three_frequency_data().YM0_list[0][:, 0]   # already /100
        np.testing.assert_allclose(yy[v, 0], months[-1], atol=1e-12)
        # regressors [y_{T-1}, y_{T-2}, y_{T-3}, 1] with nv = 2: m_1 at 0, 2, 4
        np.testing.assert_allclose(xx[v][:, [0, 2, 4]],
                                   np.broadcast_to(months[[-2, -3, -4]], (len(v), 3)), atol=1e-12)
        np.testing.assert_array_equal(xx[v, -1], 1.0)
        self.assertTrue(np.all(np.isfinite(yy[v])) and np.all(np.isfinite(xx[v])))

    def test_regressor_layout_matches_calc_yyact(self):
        m = self.model
        yy, xx = m.block_end[0]
        jj = m.valid_draws[-1]
        # Rebuild a data matrix whose last four rows are the stored ones and
        # let calc_yyact construct the regressors of its last row.
        lags = xx[jj, :-1].reshape(3, 2)              # y_{T-1}, y_{T-2}, y_{T-3}
        YY = np.vstack([np.zeros((20, 2)), lags[::-1], yy[jj]])
        spec = np.hstack((3, 3, 1, 2, YY.shape[0] - 3))
        _, _, XXact, _ = calc_yyact(HYP[0], YY, spec)
        np.testing.assert_allclose(XXact[-1], xx[jj], atol=1e-14)

    def test_forecast_block_index_names_and_shapes(self):
        m = self.model
        out = m.forecast_block(0, 6)
        self.assertEqual(out["draws"].shape, (len(m.valid_draws), 6, 2))
        self.assertEqual(out["names"], ["m_1", "q_1"])
        last = pd.Timestamp(m.index_list[0][-1])
        self.assertEqual(out["index"][0], last + pd.offsets.MonthBegin())
        self.assertEqual(len(out["index"]), 6)
        self.assertTrue(np.all(np.isfinite(out["draws"])))
        # back-transformation of a /100 series: levels are 100 x model units
        np.testing.assert_allclose(out["mean"].values, 100 * out["draws"].mean(axis=0), rtol=1e-12)
        self.assertIs(m.block_forecasts[0], out)

    def test_first_step_is_the_var_on_the_stored_state(self):
        """Draw by draw, h = 1 equals x_T' Phi plus a shock whose covariance
        is Sigma: the standardised shocks must be exactly reproducible."""
        m = self.model
        out = m.forecast_block(0, 1, seed=11)
        yy, xx = m.block_end[0]
        rng = np.random.default_rng([11, 95, 0])
        for idx, jj in enumerate(m.valid_draws):
            x = np.concatenate((yy[jj], xx[jj, :-3], [1.0]))
            mean = x @ m.Phip_list[0][jj]
            shock = np.linalg.cholesky(m.Sigmap_list[0][jj]) @ rng.standard_normal(2)
            np.testing.assert_allclose(out["draws"][idx, 0], mean + shock, atol=1e-12)

    def test_last_block_forecasts_are_unchanged(self):
        m = self.model
        before = forecast_draws(m)
        m.forecast_block(0, 12)
        after = forecast_draws(m)
        np.testing.assert_array_equal(before, after)
        twin = fit(self.method)
        twin.forecast_block(0, 12)          # before forecast(), on a twin fit
        np.testing.assert_array_equal(forecast_draws(twin), before)

    def test_the_last_block_is_refused(self):
        with self.assertRaises(ValueError):
            self.model.forecast_block(1, 4)


class TestForecastBlockSS(_Checks, unittest.TestCase):
    method = "schorfheide_song"

    @classmethod
    def setUpClass(cls):
        cls.model = fit(cls.method)


class TestForecastBlockSSSystematic(unittest.TestCase):
    """The systematic-scan MH correction stores its own accepted draws."""

    def test_stored_state_is_the_data(self):
        m = fit("schorfheide_song", mh_proposal="systematic")
        yy, _ = m.block_end[0]
        months = three_frequency_data().YM0_list[0][:, 0]
        np.testing.assert_allclose(yy[m.valid_draws, 0], months[-1], atol=1e-12)


class TestForecastBlockPalindromicAccept(unittest.TestCase):
    """The palindromic proposal is rarely accepted (0 of 60 sweeps on the
    paper's 2015-03-27 vintage), so its storage line is exercised by forcing
    acceptance (u = 1e-300 in the MH test). Every accepted sweep must store
    the proposal's end-of-sample state, and the stored state must still
    hold the data: an exception there would be swallowed by the block's
    numerical-failure guard and silently skip the downstream update."""

    def test_accepted_proposals_store_their_state(self):
        from unittest import mock
        import MBFVAR._estimation as est
        calls = []
        real = est._store_block_end

        def spy(block_end, m, j_temp, YY, p, n_store):
            calls.append((m, j_temp))
            return real(block_end, m, j_temp, YY, p, n_store)

        with mock.patch.object(est, "_store_block_end", spy), \
                mock.patch("numpy.random.uniform", return_value=1e-300):
            model = fit("schorfheide_song", nsim=8)
        accepted = model.mh_accept_counts[0]
        self.assertGreater(accepted, 0)
        # one store per sweep in the forward pass, one more per acceptance
        self.assertEqual(len(calls), 8 + accepted)
        yy, xx = model.block_end[0]
        months = three_frequency_data().YM0_list[0][:, 0]
        np.testing.assert_allclose(yy[model.valid_draws, 0], months[-1], atol=1e-12)
        self.assertTrue(np.all(np.isfinite(xx[model.valid_draws])))


class TestForecastBlockCPZ(_Checks, unittest.TestCase):
    method = "chan_poon_zhu"

    @classmethod
    def setUpClass(cls):
        cls.model = fit(cls.method)


if __name__ == "__main__":
    unittest.main()
