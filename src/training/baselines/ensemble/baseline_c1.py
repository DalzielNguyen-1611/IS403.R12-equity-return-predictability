import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Baseline_C1"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Baseline C1 — Best Single Model per Horizon and Regime
    - Evaluates argmin MSE(m, h, s) on prior validation history.
    - If tune=False (Lan04): Deterministic selection of best model per regime.
    - If tune=True (Lan05): Optuna adaptive optimization across candidate models.
    """
    return run_ensemble('C1', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
