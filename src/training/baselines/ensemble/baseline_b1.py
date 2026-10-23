import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Baseline_B1"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Baseline B1 — Best Single Model per Horizon
    - Evaluates argmin MSE(m, h) on prior validation folds across all 13 regression baselines.
    - If tune=False (Lan04): Deterministic selection of best model per horizon.
    - If tune=True (Lan05): Optuna convex combination among top candidate models.
    """
    return run_ensemble('B1', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
