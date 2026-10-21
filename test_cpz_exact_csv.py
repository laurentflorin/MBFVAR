"""
Which CPZ volatility draws the MBF-VAR forces (MBFVAR 0.9.8).

By default the forward pass and the systematic-scan proposal force
acceptance of the accept/reject draw on every sweep, as the replication
driver MFVAR.m does and as MBFVAR did up to 0.9.6. The palindromic proposal
never forces it. ``_cpz_funcs.CSV_EXACT`` switches to the exact step, which
forces only the first forward sweep, where h starts at zero. 0.9.7 had made
that the default; on the weekly block the exact step rejects almost every
proposal, so 0.9.8 restored the old default (see _cpz_funcs.CSV_EXACT). The
exactness of the step itself is tested in SBFVAR
(tests/test_cpz_exact_csv.py), which has the same function.
"""
import contextlib
import io
import unittest
from unittest import mock

import MBFVAR
import MBFVAR._estimation_cpz as E
from MBFVAR import _cpz_funcs as C
from test_prior_mean import three_frequency_data


class TestForcedOnlyOnTheFirstSweep(unittest.TestCase):

    def run_fit(self, mh_proposal, exact=False):
        calls = []
        real = E.sample_CSV

        def spy(s2, rho, sigh2, h, n, is_forced_accept=False, *a, **k):
            calls.append(bool(is_forced_accept))
            return real(s2, rho, sigh2, h, n, is_forced_accept, *a, **k)

        model = MBFVAR.MixedFrequencyBVAR(12, 0.5, [3, 4], 1, seed=2)
        with mock.patch.object(E, "sample_CSV", spy), mock.patch.object(C, "CSV_EXACT", exact), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(three_frequency_data(), [[0.2, 0.25, 100, 2]] * 2, var_of_interest=["q_1"],
                      method="chan_poon_zhu", mh_proposal=mh_proposal)
        return calls, model

    def test_default_systematic_forces_every_forward_and_proposal_draw(self):
        self.assertFalse(C.CSV_EXACT)
        calls, model = self.run_fit("systematic")
        self.assertGreater(len(calls), 2)
        self.assertTrue(all(calls))
        self.assertEqual(model.csv_accept_rates, [1.0, 1.0])

    def test_default_palindromic_forces_the_forward_pass_only(self):
        calls, _ = self.run_fit("palindromic")
        n_fwd = 2 * 12                       # two blocks, twelve sweeps
        self.assertEqual(sum(calls), n_fwd)  # palindromic proposals pass False

    def test_exact_mode_forces_only_the_first_sweep(self):
        calls, model = self.run_fit("systematic", exact=True)
        self.assertEqual(calls[:2], [True, True])
        self.assertEqual(sum(calls), 2)
        self.assertTrue(all(0.0 <= r <= 1.0 for r in model.csv_accept_rates))


if __name__ == "__main__":
    unittest.main()
