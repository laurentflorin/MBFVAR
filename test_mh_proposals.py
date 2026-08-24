"""
Tests for the sampler/proposal flags of fit():

* ``sampler={'exact','cut'}`` -- the cut variant must skip the
  Metropolis-within-Gibbs cross-block correction entirely;
* ``mh_proposal={'palindromic','systematic'}`` -- the reversible
  palindromic proposal (default) and the legacy systematic-scan proposal
  must both run end-to-end and be reproducible under a fixed seed.

The distributional correctness of the palindromic sampler is validated by
the Geweke (2004) getting-it-right test in the paper repository
(``tests/test_sampler_validity.py``); the tests here are structural.
"""
import unittest

import numpy as np
import pandas as pd

import MBFVAR


def make_small_dataset(seed=0, n_weeks=240):
    """Simulate a tiny stationary weekly VAR(1) and aggregate to Q/M/W."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1, 0.0],
                  [0.05, 0.4, 0.1],
                  [0.0, 0.1, 0.3]])
    x = np.zeros((n_weeks, 3))
    for t in range(1, n_weeks):
        x[t] = A @ x[t-1] + 0.1 * rng.standard_normal(3)

    idx_w = pd.date_range("2005-01-07", periods=n_weeks, freq="W-FRI")
    df_w = pd.DataFrame({"w_1": x[:, 2]}, index=idx_w)

    m_groups = np.arange(n_weeks) // 4          # 4 weeks per month
    df_m = pd.DataFrame({"m_1": pd.Series(x[:, 1]).groupby(m_groups).mean().values})
    df_m.index = idx_w[3::4][: len(df_m)]

    q_groups = np.arange(n_weeks) // 12         # 12 weeks per quarter
    df_q = pd.DataFrame({"q_1": pd.Series(x[:, 0]).groupby(q_groups).mean().values})
    df_q.index = idx_w[11::12][: len(df_q)]

    data = [df_q, df_m, df_w]
    trans = [np.array((1,)), np.array((1,)), np.array((1,))]
    return MBFVAR.mbfvar_data(data, trans, ["Q", "M", "W"])


HYP = [[0.09, 4.3, 1, 2.7, 4.3], [0.09, 4.3, 1, 2.7, 4.3]]
NLAGS = [3, 4]


def small_fit(**kwargs):
    data_in = make_small_dataset()
    model = MBFVAR.MixedFrequencyBVAR(8, 0.5, NLAGS, 1)
    model.fit(data_in, hyp=HYP, seed=kwargs.pop("seed", 42), **kwargs)
    return model


class TestSamplerFlags(unittest.TestCase):
    def test_invalid_flags_raise(self):
        data_in = make_small_dataset()
        model = MBFVAR.MixedFrequencyBVAR(2, 0.5, NLAGS, 1)
        with self.assertRaises(ValueError):
            model.fit(data_in, hyp=HYP, sampler="nope")
        with self.assertRaises(ValueError):
            model.fit(data_in, hyp=HYP, mh_proposal="nope")

    def test_cut_sampler_skips_mh(self):
        model = small_fit(sampler="cut")
        self.assertEqual(model.sampler, "cut")
        self.assertEqual(model.mh_total_counts, [0])
        self.assertEqual(model.mh_accept_counts, [0])
        self.assertTrue(np.isnan(model.mh_accept_rates[0]))
        self.assertTrue(np.isfinite(model.Phip_list[-1]).all())

    def test_palindromic_default_runs_and_counts(self):
        model = small_fit()
        self.assertEqual(model.sampler, "exact")
        self.assertEqual(model.mh_proposal, "palindromic")
        self.assertEqual(model.mh_total_counts, [8])
        self.assertTrue(0 <= model.mh_accept_counts[0] <= 8)
        self.assertTrue(np.isfinite(model.Phip_list[-1]).all())
        self.assertTrue(np.isfinite(model.Sigmap_list[-1]).all())

    def test_systematic_legacy_runs(self):
        model = small_fit(mh_proposal="systematic")
        self.assertEqual(model.mh_total_counts, [8])
        self.assertTrue(np.isfinite(model.Phip_list[-1]).all())

    def test_seed_reproducibility_palindromic(self):
        m1 = small_fit()
        m2 = small_fit()
        np.testing.assert_array_equal(m1.Phip_list[-1], m2.Phip_list[-1])
        np.testing.assert_array_equal(m1.lstate_list[0], m2.lstate_list[0])

    def test_cut_and_exact_differ(self):
        m_exact = small_fit()
        m_cut = small_fit(sampler="cut")
        # With at least one MH acceptance the upstream block's stored draws
        # must differ between exact and cut; the shapes stay identical.
        self.assertEqual(m_exact.Phip_list[0].shape, m_cut.Phip_list[0].shape)
        if m_exact.mh_accept_counts[0] > 0:
            self.assertFalse(
                np.array_equal(m_exact.Phip_list[0], m_cut.Phip_list[0]))


class TestGewekeSupportKnobs(unittest.TestCase):
    """Structural tests of the opt-in knobs the Geweke test needs."""

    PREMOM = [np.hstack((np.zeros((2, 1)), np.ones((2, 1)))),
              np.hstack((np.zeros((3, 1)), np.ones((3, 1))))]

    def test_fixed_premom_changes_the_prior(self):
        m_legacy = small_fit()
        m_fixed = small_fit(prior_premom=self.PREMOM)
        self.assertFalse(np.array_equal(m_legacy.Phip_list[-1],
                                        m_fixed.Phip_list[-1]))
        self.assertTrue(np.isfinite(m_fixed.Phip_list[-1]).all())

    def test_premom_rejected_with_return_mdd(self):
        data_in = make_small_dataset()
        model = MBFVAR.MixedFrequencyBVAR(2, 0.5, NLAGS, 1)
        with self.assertRaises(ValueError):
            model.fit(data_in, hyp=HYP, prior_premom=self.PREMOM,
                      return_mdd=True)

    def test_fixed_kf_init_runs(self):
        model = small_fit(kf_init="fixed")
        self.assertEqual(model.kf_init, "fixed")
        self.assertTrue(np.isfinite(model.Phip_list[-1]).all())

    def test_init_params_warm_start_is_one_transition(self):
        ref = small_fit()
        init = [{"Phi": ref.Phip_list[b][-1], "sigma": ref.Sigmap_list[b][-1]}
                for b in range(2)]
        kw = dict(kf_init="fixed", prior_premom=self.PREMOM, init_params=init)
        data_in = make_small_dataset()
        m1 = MBFVAR.MixedFrequencyBVAR(1, 0.0, NLAGS, 1)
        m1.fit(data_in, hyp=HYP, seed=7, **kw)
        # the transition moved theta and is deterministic under the seed
        self.assertFalse(np.allclose(m1.Phip_list[0][0], init[0]["Phi"]))
        m2 = MBFVAR.MixedFrequencyBVAR(1, 0.0, NLAGS, 1)
        m2.fit(make_small_dataset(), hyp=HYP, seed=7, **kw)
        np.testing.assert_array_equal(m1.Phip_list[0][0], m2.Phip_list[0][0])
        np.testing.assert_array_equal(m1.Phip_list[1][0], m2.Phip_list[1][0])

    def test_init_params_wrong_length_raises(self):
        data_in = make_small_dataset()
        model = MBFVAR.MixedFrequencyBVAR(1, 0.0, NLAGS, 1)
        with self.assertRaises(ValueError):
            model.fit(data_in, hyp=HYP,
                      init_params=[{"Phi": np.zeros((1, 1)),
                                    "sigma": np.eye(1)}])


class TestThreeBlockData(unittest.TestCase):
    def test_lf_counts_accumulate_along_the_chain(self):
        """B >= 3 blocks: each block's LF input is the previous block's
        COMPLETED panel, so the LF variable count must accumulate
        recursively (q; q+m; q+m+w). The old formula (base quarterly +
        immediate HF block) mis-sized every block from the third on."""
        idx_d = pd.bdate_range("2010-01-04", periods=400)
        df_d = pd.DataFrame({"d_1": np.random.default_rng(0).standard_normal(400)},
                            index=idx_d)
        df_w = pd.DataFrame({"w_1": np.zeros(80)}, index=idx_d[4::5][:80])
        df_m = pd.DataFrame({"m_1": np.zeros(20)}, index=idx_d[19::20][:20])
        df_q = pd.DataFrame({"q_1": np.zeros(6)}, index=idx_d[59::60][:6])
        trans = [np.array((1,))] * 4
        din = MBFVAR.mbfvar_data([df_q, df_m, df_w, df_d], trans,
                                 ["Q", "M", "W", "D"])
        self.assertEqual(list(din.Nq_list), [1, 2, 3])
        self.assertEqual(list(din.nv_list), [2, 3, 4])
        self.assertEqual([len(s) for s in din.select_q], [1, 2, 3])


class TestSamplerFlagsCPZ(unittest.TestCase):
    def test_cpz_palindromic_runs(self):
        model = small_fit(method="chan_poon_zhu")
        self.assertEqual(model.mh_total_counts, [8])
        self.assertTrue(np.isfinite(model.Phip_list[-1]).all())

    def test_cpz_cut_skips_mh(self):
        model = small_fit(method="chan_poon_zhu", sampler="cut")
        self.assertEqual(model.mh_total_counts, [0])

    def test_cpz_systematic_runs(self):
        model = small_fit(method="chan_poon_zhu", mh_proposal="systematic")
        self.assertEqual(model.mh_total_counts, [8])


if __name__ == "__main__":
    unittest.main()
