# QUANTITATIVE RESEARCH EVALUATION REPORT: SARIMA
**Research Project:** *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*
**Model Family:** Statistical Models | **Model Category:** Regression
**Evaluation Status:** Strict Walk-Forward Out-of-Sample & Independent Hold-out Verification

---

## SECTION 1 – Model Overview
- **Model Identifier:** `SARIMA`
- **Architectural Classification:** Statistical Models (Regression)
- **Primary Hyperparameter Configuration:** `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}`
- **Validation Period (Split 2):** 2015-06-10 to 2020-10-30 (5 walk-forward expanding folds, 1,360 total trading sessions)
- **Final Hold-out Test Period (Split 3):** 2020-11-02 to 2026-04-07 (1,362 out-of-sample trading sessions)
- **Evaluated Forecast Horizons:** 1D, 5D, 20D, 60D, 120D
- **Evaluated Volatility Regimes:** Low, Normal, High (Walk-forward expanding 33% & 67% quantiles)

---

## SECTION 2 – Overall Performance Across Evaluation Stages
The table below synthesizes predictive performance across the three evaluation stages. Note that Part A and Part B evaluate on Walk-Forward Validation (Split 2), whereas Part C evaluates on the strictly independent hold-out test set (Split 3).

| Evaluation Stage                | OOS_R2   | RMSE   | MAE    | DA (%)   | Naive Positive DA (%)   | Excess DA (pp)   | DM Statistic   | DM p-value   | Sign Test p-value   | N    |
|:--------------------------------|:---------|:-------|:-------|:---------|:------------------------|:-----------------|:---------------|:-------------|:--------------------|:-----|
| Part A – Before Tuning          | N/A      | N/A    | N/A    | N/A      | 54.00%                  | 0.00             | N/A            | N/A          | N/A                 | N/A  |
| Part B – After Tuning           | -0.1072  | 0.0456 | 0.0347 | 61.49%   | N/A                     | N/A              | N/A            | N/A          | N/A                 | 1360 |
| Part C – Final Independent Test | -0.0842  | 0.0476 | 0.0363 | 61.26%   | 66.61%                  | -0.05            | -1.5713        | 0.8418       | 0.1252              | 1362 |

## SECTION 3 – Forecast Horizon Analysis
Analysis of predictive decay across horizons from ultra-short term (1D) to long-term macroeconomic horizon (120D).

| Horizon   |   OOS_R2 |   RMSE |    MAE |     DA |   Naive_Positive_DA |   Excess_DA |   DM_Stat |   DM_pvalue |   Sign_pvalue |   Observations |
|:----------|---------:|-------:|-------:|-------:|--------------------:|------------:|----------:|------------:|--------------:|---------------:|
| 1D        |  -0.0049 | 0.0106 | 0.0076 | 0.5073 |              0.5374 |     -0.0301 |    0      |      1      |        0.3033 |           1362 |
| 5D        |  -0.2612 | 0.0251 | 0.0185 | 0.5066 |              0.594  |     -0.0874 |   -4.5464 |      1      |        0.3225 |           1362 |
| 20D       |  -0.127  | 0.0442 | 0.0338 | 0.591  |              0.6689 |     -0.0778 |   -2.617  |      0.9955 |        0      |           1362 |
| 60D       |  -0.0719 | 0.0678 | 0.0527 | 0.6946 |              0.7496 |     -0.0551 |   -1.0514 |      0.8534 |        0      |           1362 |
| 120D      |   0.0442 | 0.0905 | 0.0687 | 0.7636 |              0.7805 |     -0.0169 |    0.3585 |      0.36   |        0      |           1362 |

![Forecast Horizon Performance](figures/horizon_performance.png)

## SECTION 4 – Volatility Regime Analysis
Performance breakdown conditioned on market volatility regimes (Low, Normal, High) constructed using expanding quantiles.

| Regime   |   OOS_R2 |    RMSE |     MAE |      DA |
|:---------|---------:|--------:|--------:|--------:|
| Low      | -0.04334 | 0.03552 | 0.0273  | 0.64276 |
| Normal   | -0.05694 | 0.041   | 0.03166 | 0.66746 |
| High     | -0.10738 | 0.06184 | 0.04948 | 0.5132  |

![Volatility Regime Performance](figures/regime_performance.png)

## SECTION 5 – Horizon × Regime Matrices
Multi-dimensional matrix displaying performance across the 5 × 3 Horizon × Regime state space.

### Out-of-Sample R² Matrix (5 × 3):

| Horizon   |     Low |   Normal |    High |
|:----------|--------:|---------:|--------:|
| 1D        | -0.0001 |  -0.0053 | -0.0056 |
| 5D        | -0.1334 |  -0.2598 | -0.2832 |
| 20D       | -0.0667 |  -0.0599 | -0.1715 |
| 60D       |  0.0128 |  -0.0784 | -0.0909 |
| 120D      | -0.0293 |   0.1187 |  0.0143 |

![Horizon x Regime OOS R2](figures/horizon_regime_heatmap_oos_r2.png)

![Horizon x Regime DA](figures/horizon_regime_heatmap_da.png)

## SECTION 6 – Before vs After Tuning Comparison
> [!NOTE]
> **Single Tuned Workflow Execution:** Traditional models in this research suite were tuned directly during Walk-Forward Validation (Lan01) using cross-validated Grid Search and Optuna TPE. A separate un-tuned baseline run was not persisted in the repository archives.

## SECTION 7 – Final Independent Hold-out Test Verification
The Final Independent Hold-out Test (Split 3: 2020–2026, 1,362 days) serves as the definitive out-of-sample empirical evidence. All preprocessing steps and hyperparameter choices were completely frozen prior to test inference.
- **Total Test Days:** 1,362 trading sessions
- **Lead Out-of-Sample R²:** 0.0442 (achieved at horizon `120D`)
- **Max Directional Accuracy:** 76.36%

## SECTION 8 – Walk-Forward Fold-Level Analysis
Stability evaluation across the 5 expanding walk-forward folds (Split 2):

| Horizon   |   Mean_OOS_R2 |   Std_OOS_R2 |   Min_OOS_R2 |   Max_OOS_R2 |   Mean_RMSE |   Std_RMSE |   Mean_MAE |   Mean_DA |
|:----------|--------------:|-------------:|-------------:|-------------:|------------:|-----------:|-----------:|----------:|
| 1D        |        0.0072 |       0.012  |      -0.009  |       0.021  |      0.0108 |     0.006  |     0.0072 |    0.539  |
| 5D        |       -0.2589 |       0.0805 |      -0.3603 |      -0.1394 |      0.0246 |     0.0115 |     0.0172 |    0.5346 |
| 20D       |       -0.1183 |       0.0516 |      -0.1713 |      -0.0464 |      0.0455 |     0.029  |     0.0328 |    0.6066 |
| 60D       |       -0.1634 |       0.0957 |      -0.2939 |      -0.0407 |      0.0707 |     0.0418 |     0.0553 |    0.6375 |
| 120D      |       -0.0027 |       0.3273 |      -0.5112 |       0.3958 |      0.0765 |     0.0362 |     0.061  |    0.7566 |

![Fold Trajectory](figures/fold_level_performance.png)


## SECTION 9 – Quantitative Executive Summary
- **Optimal Horizon:** `120D` (OOS R² = 0.0442)
- **Sub-optimal Horizon:** `5D` (OOS R² = -0.2612)
- **Most Favorable Volatility Regime:** `Low`
- **Most Challenging Volatility Regime:** `High`
- **Maximum Directional Accuracy:** 76.36%
- **Test Set Observations:** 1362 days