import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Baseline_C2"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Baseline C2 — Regime-Shrunk Inverse-MSE Weighting
    - Base horizon weights B2 shrunk with regime sample weights via N_0.
    - If tune=False (Lan04): Fixed default shrinkage N_0 = 50.0.
    - If tune=True (Lan05): Optuna optimizes shrinkage parameter N_0 and inverse power gamma.
    """
    return run_ensemble('C2', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
