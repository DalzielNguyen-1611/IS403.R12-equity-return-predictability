import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Ensemble_A3"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Experiment A3 — Ridge + SARIMA + Historical Mean
    - tune=False: Equal weights w_m = 1/3 ≈ 33.33% (Lan04)
    - tune=True:  Optuna-tuned simplex weights (Lan05)
    """
    return run_ensemble('A3', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
