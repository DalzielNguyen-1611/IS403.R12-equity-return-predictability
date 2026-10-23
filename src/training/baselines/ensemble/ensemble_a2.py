import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Ensemble_A2"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Experiment A2 — Core 4 Models Ensemble (Ridge, SVR, LightGBM, SARIMA)
    - tune=False: Equal weights w_m = 1/4 = 25.0% (Lan04)
    - tune=True:  Optuna-tuned simplex weights (Lan05)
    """
    return run_ensemble('A2', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
