import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Baseline_B2"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Baseline B2 — Normalized Inverse-MSE Weighting
    - Base weights proportional to (1 / MSE_m)^gamma across Core 4 models.
    - If tune=False (Lan04): Classical Inverse-MSE (gamma = 1.0).
    - If tune=True (Lan05): Optuna tunes power gamma in [0.2, 5.0].
    """
    return run_ensemble('B2', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
