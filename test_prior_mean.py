"""
A per-variable prior mean for the Schorfheide-Song prior (MBFVAR 0.9.2).

The SS prior's dummy observations centre every variable's own first lag on
a random walk and add sum-of-coefficients dummies that push towards unit
roots. That suits levels and year-on-year growth rates. For period-on-period
growth rates the usual centring is white noise for the growth series, with
persistent ones such as interest rates kept at a random walk (Banbura,
Giannone and Reichlin, 2010). ``prior_mean`` provides that, by variable name
across the blocks; ``None`` must leave the prior, and every existing run,
exactly as it was. The same option exists in SBFVAR 0.2.3, with the same
tests, so the two models can be given the same prior.
"""
import contextlib
import io
import unittest

import numpy as np
import pandas as pd

import MBFVAR
from MBFVAR.mfbvar_funcs import prior_mean_vector, resolve_prior_mean_blocks, varprior


@contextlib.contextmanager
def silence_output():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def varprior_091(nv, nlags, nex, hyp, premom):
    """The prior of MBFVAR 0.9.1, verbatim, as the regression reference."""
    lambda1, lambda2, lambda3, lambda4, lambda5 = hyp[0], hyp[1], int(hyp[2]), hyp[3], hyp[4]
    dsize = nex + (nlags + lambda3 + 1) * nv
    breakss = np.zeros((5, 1))
    ydu = np.zeros((int(dsize), int(nv)))
    xdu = np.zeros((int(dsize), int(nv * nlags + nex)))
    sig = np.diag(premom[:, 1])
    ydu[range(nv), :] = lambda1 * sig
    xdu[:nv, :sig.shape[1]] = lambda1 * sig
    breakss[0] = nv
    if nlags > 1:
        ydu[int(breakss[0, 0]):(nv * nlags), :] = np.zeros(((nlags - 1) * nv, nv))
        j = 1
        while j <= nlags - 1:
            xdu[int(breakss[0, 0]) + (j - 1) * nv:int(breakss[0, 0]) + j * nv] = np.hstack((
                np.zeros((nv, j * nv)), lambda1 * sig * ((j + 1) ** lambda2),
                np.zeros((nv, (nlags - 1 - j) * nv + nex))))
            j = j + 1
        breakss[1, 0] = breakss[0, 0] + (nlags - 1) * nv
    else:
        breakss[1, 0] = breakss[0, 0]
    ydu[int(breakss[1, 0]):int(breakss[1, 0]) + lambda3 * nv, :] = np.kron(np.ones((lambda3, 1)), sig)
    breakss[2, 0] = breakss[1, 0] + lambda3 * nv
    lammean = lambda4 * premom[:, 0]
    ydu[int(breakss[2, 0]), :] = lammean
    xdu[int(breakss[2, 0]), :] = np.hstack((np.squeeze(np.kron(np.ones((1, nlags)), lammean)), lambda4))
    breakss[3] = breakss[2, 0] + 1
    mumean = np.diag(lambda5 * premom[:, 0])
    ydu[int(breakss[3, 0]):int(breakss[3, 0]) + nv, :] = mumean
    xdu[int(breakss[3, 0]):int(breakss[3, 0]) + nv, :] = np.hstack((
        np.squeeze(np.kron(np.ones((1, nlags)), mumean)), np.zeros((nv, nex))))
    return ydu, xdu


def three_frequency_data(seed=7, n_months=96):
    """Quarterly, monthly and weekly series (four weeks a month): two blocks,
    a monthly VAR [m_1, q_1] and a weekly VAR [w_1, m_1, q_1]."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    weeks = pd.DatetimeIndex([m - pd.Timedelta(days=7 * (3 - k)) for m in months for k in range(4)])
    path = np.cumsum(rng.normal(scale=0.3, size=(4 * n_months, 3)), axis=0) * 0.1
    path += rng.normal(scale=0.3, size=path.shape)
    weekly = pd.DataFrame({"w_1": path[:, 0]}, index=weeks)
    monthly = pd.DataFrame({"m_1": path[:, 1].reshape(n_months, 4).mean(axis=1)}, index=months)
    quarterly = pd.DataFrame({"q_1": path[:, 2].reshape(n_months // 3, 12).mean(axis=1)},
                             index=months[2::3])
    return MBFVAR.mbfvar_data([quarterly, monthly, weekly],
                              [np.array([1]), np.array([1]), np.array([1])], ["Q", "M", "W"])


class TestVarpriorPriorMean(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.nv, self.p = 5, 4
        self.hyp = [0.7, 1.3, 1, 2.1, 0.9]
        self.premom = np.column_stack((rng.normal(size=self.nv), rng.uniform(0.5, 2, self.nv)))

    def test_default_is_the_091_prior_bit_for_bit(self):
        y0, x0 = varprior_091(self.nv, self.p, 1, self.hyp, self.premom)
        for pm in (None, np.ones(self.nv)):
            y, x = varprior(self.nv, self.p, 1, self.hyp, self.premom, prior_mean=pm)
            self.assertTrue(np.array_equal(y, y0) and np.array_equal(x, x0))

    def test_first_lag_dummy_encodes_the_requested_mean(self):
        delta = np.array([0.0, 1.0, 0.0, 0.5, 1.0])
        y, x = varprior(self.nv, self.p, 1, self.hyp, self.premom, prior_mean=delta)
        np.testing.assert_allclose(np.diag(y[:self.nv]) / np.diag(x[:self.nv, :self.nv]), delta)

    def test_sum_of_coefficients_dummy_drops_out_for_white_noise_series(self):
        delta = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        y, x = varprior(self.nv, self.p, 1, self.hyp, self.premom, prior_mean=delta)
        y0, x0 = varprior_091(self.nv, self.p, 1, self.hyp, self.premom)
        soc = slice(y.shape[0] - self.nv, y.shape[0])
        for i in range(self.nv):
            if delta[i] == 0:
                self.assertEqual(np.count_nonzero(y[soc][i]) + np.count_nonzero(x[soc][i]), 0)
            else:
                np.testing.assert_array_equal(y[soc][i], y0[soc][i])
                np.testing.assert_array_equal(x[soc][i], x0[soc][i])
        mid = slice(self.nv, y.shape[0] - self.nv)
        np.testing.assert_array_equal(y[mid], y0[mid])
        np.testing.assert_array_equal(x[mid], x0[mid])

    def test_vector_checks(self):
        with self.assertRaises(ValueError):
            prior_mean_vector([1.0, 0.0], 3)


class TestBlocks(unittest.TestCase):
    blocks = [["AWHI", "INDPRO", "UNRATE", "GDPC1"],
              ["WGS10YR", "FEDFUNDS", "AWHI", "INDPRO", "UNRATE", "GDPC1"]]

    def test_one_name_applies_in_every_block_it_appears_in(self):
        v = resolve_prior_mean_blocks({"AWHI": 0, "INDPRO": 0, "GDPC1": 0}, self.blocks)
        np.testing.assert_array_equal(v[0], [0, 0, 1, 0])
        np.testing.assert_array_equal(v[1], [1, 1, 0, 0, 1, 0])

    def test_a_weekly_name_is_not_unknown_to_the_monthly_block(self):
        v = resolve_prior_mean_blocks({"FEDFUNDS": 1, "GDPC1": 0}, self.blocks)
        np.testing.assert_array_equal(v[0], [1, 1, 1, 0])

    def test_a_misspelt_name_is_refused(self):
        with self.assertRaises(ValueError):
            resolve_prior_mean_blocks({"GDP": 0}, self.blocks)

    def test_none_stays_none(self):
        self.assertIsNone(resolve_prior_mean_blocks(None, self.blocks))


class TestFitWithPriorMean(unittest.TestCase):
    def fit(self, **kw):
        model = MBFVAR.MixedFrequencyBVAR(12, 0.5, [3, 4], 1)
        with silence_output():
            model.fit(three_frequency_data(), [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]],
                      seed=11, **kw)
        return model

    def test_default_and_explicit_none_draw_the_same_chain(self):
        a, b = self.fit(), self.fit(prior_mean=None)
        self.assertTrue(np.array_equal(a.Phip_list[-1], b.Phip_list[-1]))
        self.assertTrue(np.array_equal(a.lstate_list[-1], b.lstate_list[-1]))

    def test_white_noise_centring_changes_the_chain_and_is_recorded(self):
        a, b = self.fit(), self.fit(prior_mean={"m_1": 0, "q_1": 0})
        self.assertFalse(np.array_equal(a.Phip_list[-1], b.Phip_list[-1]))
        np.testing.assert_array_equal(b.prior_mean_by_block[0], [0, 0])     # [m_1, q_1]
        np.testing.assert_array_equal(b.prior_mean_by_block[1], [1, 0, 0])  # [w_1, m_1, q_1]

    def test_blocks_narrowed_by_var_of_interest_get_their_own_vector(self):
        """With var_of_interest the weekly block carries only the variables
        of interest ([w_1, q_1] here), not every low-frequency series; the
        paper's production runs pass var_of_interest=['GDPC1']."""
        b = self.fit(prior_mean={"m_1": 0, "q_1": 0}, var_of_interest=["q_1"])
        np.testing.assert_array_equal(b.prior_mean_by_block[0], [0, 0])     # [m_1, q_1]
        np.testing.assert_array_equal(b.prior_mean_by_block[1], [1, 0])     # [w_1, q_1]

    def test_refused_on_the_cpz_path(self):
        model = MBFVAR.MixedFrequencyBVAR(12, 0.5, [3, 4], 1)
        with self.assertRaises(ValueError):
            model.fit(three_frequency_data(), [[0.09, 4.3, 1, 2.7]] * 2,
                      method="chan_poon_zhu", prior_mean={"q_1": 0})


if __name__ == "__main__":
    unittest.main()
