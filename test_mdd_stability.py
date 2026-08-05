import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import loggamma
from scipy.stats import uniform

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import MBFVAR
import MBFVAR._estimation as estimation_module
from MBFVAR.mfbvar_funcs import mdd_ as stable_mdd
from MBFVAR.pseudo_inverse.pseudo_inverse import calculate_pseudo_inverse


def _load_qm_data():
    hist_file = REPO_ROOT / "examples" / "hist.xlsx"
    data = [
        pd.read_excel(hist_file, sheet_name=freq, index_col=0)
        for freq in ("Q", "M")
    ]
    trans = [np.array([1, 1]), np.array([1, 1, 1])]
    return MBFVAR.mbfvar_data(data, trans, ["Q", "M"])


def _make_model(nsim=1):
    return MBFVAR.MixedFrequencyBVAR(nsim, 0.0, [3], 1)


def _mdd_from_var_matrices(YYact, YYdum, XXact, XXdum, use_slogdet):
    YY_full = np.vstack((YYdum, YYact))
    XX_full = np.vstack((XXdum, XXact))

    n_total = YY_full.shape[0]
    n_dummy = YYdum.shape[0]
    nv = YY_full.shape[1]
    k = XX_full.shape[1]

    xxdum = XXdum.T @ XXdum
    xxfull = XX_full.T @ XX_full
    S0 = (YYdum.T @ YYdum) - ((YYdum.T @ XXdum) @ calculate_pseudo_inverse(xxdum)) @ XXdum.T @ YYdum
    S1 = (YY_full.T @ YY_full) - ((YY_full.T @ XX_full) @ calculate_pseudo_inverse(xxfull)) @ XX_full.T @ YY_full

    gam0 = 0.0
    gam1 = 0.0
    for i in range(nv):
        gam0 += loggamma(0.5 * (n_dummy - k + 1 - (i + 1)))
        gam1 += loggamma(0.5 * (n_total - k + 1 - (i + 1)))

    if use_slogdet:
        logdet_xxdum = np.linalg.slogdet(xxdum)[1]
        logdet_s0 = np.linalg.slogdet(S0)[1]
        logdet_xxfull = np.linalg.slogdet(xxfull)[1]
        logdet_s1 = np.linalg.slogdet(S1)[1]
    else:
        logdet_xxdum = np.log(np.absolute(np.linalg.det(xxdum)))
        logdet_s0 = np.log(np.absolute(np.linalg.det(S0)))
        logdet_xxfull = np.log(np.absolute(np.linalg.det(xxfull)))
        logdet_s1 = np.log(np.absolute(np.linalg.det(S1)))

    lnpY0 = (-nv * (n_dummy - k) * 0.5 * np.log(math.pi) - (nv / 2) * logdet_xxdum -
             (n_dummy - k) * 0.5 * logdet_s0 + nv * (nv - 1) * 0.25 * np.log(math.pi) + gam0)
    lnpY1 = (-nv * (n_total - k) * 0.5 * np.log(math.pi) - (nv / 2) * logdet_xxfull -
             (n_total - k) * 0.5 * logdet_s1 + nv * (nv - 1) * 0.25 * np.log(math.pi) + gam1)
    return lnpY1 - lnpY0


def _assert_valid_hyperparameter_list(hyp):
    assert isinstance(hyp, list) and hyp, f"Expected non-empty hyperparameter list, got {hyp!r}"
    for block in hyp:
        assert len(block) == 5, f"Expected 5 hyperparameters per block, got {block!r}"
        assert block[2] == 1, f"Expected fixed lambda3=1, got {block!r}"
        assert np.isfinite(np.asarray(block, dtype=float)).all(), f"Expected finite hyperparameters, got {block!r}"


def test_fit_return_mdd_is_finite_across_loose_priors():
    data_in = _load_qm_data()
    hyperparameters = [
        [0.09, 4.3, 1, 2.7, 4.3],
        [0.001, 10.0, 1, 0.01, 0.01],
        [20.0, 10.0, 1, 10.0, 10.0],
    ]

    for seed, hyp_block in enumerate(hyperparameters, start=1):
        model = _make_model()
        np.random.seed(seed)
        mdd = model.fit(data_in, hyp=[hyp_block], return_mdd=True, check_explosive=False)
        assert mdd is not None, f"Expected finite MDD for {hyp_block}, got None"
        assert np.isfinite(mdd), f"Expected finite MDD for {hyp_block}, got {mdd}"


def test_slogdet_matches_previous_formula_when_det_is_finite():
    data_in = _load_qm_data()
    model = _make_model()
    hyp = [[0.09, 4.3, 1, 2.7, 4.3]]
    captured = {}
    original_mdd = estimation_module.mdd_

    def capture_mdd(hyp_in, YY, spec):
        result = stable_mdd(hyp_in, YY, spec)
        captured["result"] = result
        return result

    estimation_module.mdd_ = capture_mdd
    try:
        np.random.seed(1234)
        model_mdd = model.fit(data_in, hyp=hyp, return_mdd=True, check_explosive=False)
    finally:
        estimation_module.mdd_ = original_mdd

    stable_value, YYact, YYdum, XXact, XXdum = captured["result"]
    old_value = _mdd_from_var_matrices(YYact, YYdum, XXact, XXdum, use_slogdet=False)
    new_value = _mdd_from_var_matrices(YYact, YYdum, XXact, XXdum, use_slogdet=True)

    assert np.isfinite(old_value), f"Expected finite legacy determinant formula, got {old_value}"
    assert np.isfinite(new_value), f"Expected finite slogdet formula, got {new_value}"
    np.testing.assert_allclose(new_value, old_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(float(stable_value), new_value, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(float(model_mdd), new_value, rtol=1e-10, atol=1e-10)


def test_mdd_hyperparameter_optimizers_penalize_nonfinite_objectives():
    data_in = _load_qm_data()
    model = _make_model()
    original_fit = model.fit

    def fit_nan(self, *args, **kwargs):
        return np.nan

    model.fit = types.MethodType(fit_nan, model)
    pbounds = {
        "lambda1_1": (0.001, 20.0),
        "lambda2_1": (0.01, 10.0),
        "lambda4_1": (0.01, 10.0),
        "lambda5_1": (0.01, 10.0),
    }
    param_space = {
        "lambda1_1": uniform(0.001, 20.0),
        "lambda2_1": uniform(0.01, 10.0),
        "lambda4_1": uniform(0.01, 10.0),
        "lambda5_1": uniform(0.01, 10.0),
    }

    try:
        np.random.seed(2024)
        _assert_valid_hyperparameter_list(
            model.update_hyperparameters(data_in, pbounds, init_points=1, n_iter=1, nsim=1)
        )
        np.random.seed(2025)
        _assert_valid_hyperparameter_list(
            model.update_hyperparameters_mango(data_in, param_space, init_points=1, n_iter=1, nsim=1, njobs=1)
        )
    finally:
        model.fit = original_fit


def test_update_hyperparameters_mango_completes_end_to_end():
    data_in = _load_qm_data()
    model = _make_model()
    param_space = {
        "lambda1_1": uniform(0.001, 20.0),
        "lambda2_1": uniform(0.01, 10.0),
        "lambda4_1": uniform(0.01, 10.0),
        "lambda5_1": uniform(0.01, 10.0),
    }

    np.random.seed(99)
    hyp = model.update_hyperparameters_mango(
        data_in,
        param_space,
        init_points=1,
        n_iter=1,
        nsim=1,
        njobs=1,
    )

    _assert_valid_hyperparameter_list(hyp)


def run_all_tests():
    tests = [
        test_fit_return_mdd_is_finite_across_loose_priors,
        test_slogdet_matches_previous_formula_when_det_is_finite,
        test_mdd_hyperparameter_optimizers_penalize_nonfinite_objectives,
        test_update_hyperparameters_mango_completes_end_to_end,
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print("  ✓ passed")


if __name__ == "__main__":
    run_all_tests()
