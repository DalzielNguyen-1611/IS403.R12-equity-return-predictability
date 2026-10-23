import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(CURRENT_DIR)
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR)

import config
from ensemble.ensemble_models import run_ensemble

MODEL_NAME = "Baseline_C3"

def run_model(df=None, horizons=config.HORIZONS, tune=False, logger=None):
    """
    Runner for Baseline C3 — Defensive Volatility Switch
    - Low/Normal volatility: A2 (Core 4 Ensemble)
    - High volatility: Historical Mean (Defensive benchmark)
    - If tune=False (Lan04): Strict rule switch.
    - If tune=True (Lan05): Optuna optimizes defensive weight alpha in High volatility and Core weights in Low/Normal.
    """
    return run_ensemble('C3', tune=tune, horizons=horizons, logger=logger)

if __name__ == "__main__":
    run_model()
