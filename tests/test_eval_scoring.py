import numpy as np

from ecsfm.fm.eval_classical import calculate_metrics


def test_robust_scenario_score_prefers_better_trace():
    gt = np.linspace(-1.0, 1.0, 200)
    pred_good = gt + 0.01
    pred_bad = gt * 5.0 + 2.0

    gtox = np.tile(gt[:20], (2, 1))
    gtred = np.tile(-gt[:20], (2, 1))
    genox_good = gtox + 0.01
    genred_good = gtred - 0.01
    genox_bad = gtox * 4.0
    genred_bad = gtred * -3.0

    good = calculate_metrics(gt, pred_good, gtox, genox_good, gtred, genred_good)
    bad = calculate_metrics(gt, pred_bad, gtox, genox_bad, gtred, genred_bad)

    assert 0.0 <= good["scenario_score"] <= 100.0
    assert 0.0 <= bad["scenario_score"] <= 100.0
    assert good["scenario_score"] > bad["scenario_score"]
    assert good["nrmse_range"] < bad["nrmse_range"]
    assert good["nmae_rms"] < bad["nmae_rms"]
