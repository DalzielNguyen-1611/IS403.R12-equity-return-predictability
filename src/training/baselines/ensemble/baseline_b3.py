import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Baseline_B3"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Baseline B3 — NNLS / Quadratic Programming with Simplex Constraints
    - Solves: min_w ||y - P*w||_2^2 subject to w_m >= 0 and sum(w_m) = 1 across Core 4.
    - If tune=False (Lan04): Exact SLSQP Quadratic Programming.
    - If tune=True (Lan05): Optuna TPE Simplex optimization.
    """
    return run_ensemble('B3', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
