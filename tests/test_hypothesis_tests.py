import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'evaluation', 'hypothesis_tests')))

import common
import hac
import bootstrap
import multiple_testing

def test_holm_bonferroni():
    pvals = [0.01, 0.04, 0.03, 0.20]
    adj = multiple_testing.holm_bonferroni(pvals)
    assert len(adj) == 4
    # Check bounds
    assert np.all(adj >= 0.0) and np.all(adj <= 1.0)
    # Check monotonicity of adjustment
    order = np.argsort(pvals)
    sorted_adj = adj[order]
    assert np.all(np.diff(sorted_adj) >= 0)

def test_hac_wald():
    # Synthetic data under H0: mean = 0
    np.random.seed(42)
    X = np.random.normal(0, 1, size=(500, 3))
    omega = hac.newey_west_covariance(X, bandwidth=5)
    assert omega.shape == (3, 3)
    assert np.all(np.linalg.eigvals(omega) > 0)
    
    # Wald test for H0: theta = 0
    res = hac.wald_test(np.mean(X, axis=0), omega / 500)
    assert res['df'] == 3
    assert 0.0 <= res['pvalue'] <= 1.0

def test_bootstrap_dimensions():
    X = np.random.normal(0, 1, size=(100, 5))
    samples = list(bootstrap.paired_block_bootstrap_matrix(X, block_length=10, n_boot=5, seed=123))
    assert len(samples) == 5
    for s in samples:
        assert s.shape == (100, 5)

def test_normalized_loss_differential_reproduces_r2():
    df_preds = common.load_unified_predictions('test')
    df_diff, hm_mse = common.compute_loss_differentials(df_preds)
    
    # Check TSMixer 120D
    tsm = df_diff[(df_diff['Model'] == 'TSMixer') & (df_diff['Horizon'] == '120D')]
    r2_from_diff = tsm['d_t'].mean()
    
    # Compute manual R2
    act = tsm['Actual'].values
    pred = tsm['Prediction'].values
    hm_pred = df_preds[(df_preds['Model'] == 'Historical_Mean') & (df_preds['Horizon'] == '120D')]['Prediction'].values
    
    ss_res = np.sum((act - pred) ** 2)
    ss_tot = np.sum((act - hm_pred) ** 2)
    r2_manual = 1.0 - ss_res / ss_tot
    
    assert np.isclose(r2_from_diff, r2_manual, atol=1e-5)
    assert np.isclose(r2_from_diff, 0.1753, atol=1e-3)
