"""
The exact accept-reject Metropolis-Hastings volatility step (MBFVAR 0.9.7).

Up to 0.9.6 the CPZ forward pass forced acceptance of the volatility proposal
on every sweep, as the replication driver MFVAR.m does, skipping the MH
correction of Chan (2020, Online Appendix A). Since 0.9.7 it is forced only
on the first sweep, where h starts at zero, and the cross-block proposals
never force it. The exactness of the step itself is tested in SBFVAR
(tests/test_cpz_exact_csv.py), which has the same function.
"""
import contextlib
import io
import unittest
from unittest import mock

import MBFVAR
import MBFVAR._estimation_cpz as E
from test_prior_mean import three_frequency_data


class TestForcedOnlyOnTheFirstSweep(unittest.TestCase):

    def run_fit(self, mh_proposal):
        calls = []
        real = E.sample_CSV

        def spy(s2, rho, sigh2, h, n, is_forced_accept=False, *a, **k):
            calls.append(bool(is_forced_accept))
            return real(s2, rho, sigh2, h, n, is_forced_accept, *a, **k)

        model = MBFVAR.MixedFrequencyBVAR(12, 0.5, [3, 4], 1, seed=2)
        with mock.patch.object(E, "sample_CSV", spy), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model.fit(three_frequency_data(), [[0.2, 0.25, 100, 2]] * 2, var_of_interest=["q_1"],
                      method="chan_poon_zhu", mh_proposal=mh_proposal)
        return calls, model

    def test_systematic(self):
        calls, model = self.run_fit("systematic")
        self.assertEqual(calls[:2], [True, True])          # both blocks, first sweep
        self.assertEqual(sum(calls), 2)
        self.assertTrue(all(0.0 <= r <= 1.0 for r in model.csv_accept_rates))
        self.assertEqual(len(model.csv_accept_rates), 2)

    def test_palindromic(self):
        calls, _ = self.run_fit("palindromic")
        self.assertEqual(calls[:2], [True, True])
        self.assertEqual(sum(calls), 2)


if __name__ == "__main__":
    unittest.main()
