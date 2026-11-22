# Volatility Regime Robustness Report: 10D, 20D, and 60D Window Comparison

## 1. Executive Summary

This report examines whether the empirical findings for **Research Question 2 (RQ2)** are robust to the choice of the historical volatility window used to define market volatility regimes. The baseline specification employs a **20-day window (20D)**. This evaluation compares the baseline against two alternative configurations: a shorter **10-day window (10D)** and a longer **60-day window (60D)**.

- **Methodology:** Expanding terciles (q33, q67) with `min_periods = 252` trading days, strictly avoiding forward-looking leakage (`.bfill()` omitted).
- **Data Scope:** 1,362 out-of-sample holdout trading days (2020-11-02 to 2026-04-07).
- **Model Scope:** 12 models across 5 forecasting horizons (1D, 5D, 20D, 60D, 120D).

## 2. Regime Distribution on the Holdout Sample

| Window   |   Low_N |   Low_Pct |   Normal_N |   Normal_Pct |   High_N |   High_Pct |   Total_N |
|:---------|--------:|----------:|-----------:|-------------:|---------:|-----------:|----------:|
| 10D      |     354 |     25.99 |        567 |        41.63 |      441 |      32.38 |      1362 |
| 20D      |     332 |     24.38 |        599 |        43.98 |      431 |      31.64 |      1362 |
| 60D      |     315 |     23.13 |        662 |        48.6  |      385 |      28.27 |      1362 |

## 3. Horizon-Regime Interaction Test Robustness (RQ2 Joint Test)

The table below reports the Wald statistic and Holm-adjusted p-values for the 8 linear interaction restrictions across windows:

| Model         |   Wald_10D |   pHolm_10D | Sig_10D   |   Wald_20D |   pHolm_20D | Sig_20D   |   Wald_60D |   pHolm_60D | Sig_60D   |
|:--------------|-----------:|------------:|:----------|-----------:|------------:|:----------|-----------:|------------:|:----------|
| ARIMA         |     9.6958 |      1      | False     |    17.118  |      0.289  | False     |    14.9189 |      0.6074 | False     |
| ARIMAX        |     5.6008 |      1      | False     |     9.2826 |      1      | False     |     4.3868 |      1      | False     |
| Ridge         |     7.4499 |      1      | False     |     6.4439 |      1      | False     |     5.6312 |      1      | False     |
| Elastic_Net   |     1.713  |      1      | False     |     3.4982 |      1      | False     |     4.0259 |      1      | False     |
| SVR           |     7.159  |      1      | False     |     9.4479 |      1      | False     |    12.9613 |      0.9055 | False     |
| Random_Forest |    21.5823 |      0.0633 | False     |    21.724  |      0.06   | False     |    22.9803 |      0.0373 | True      |
| XGBoost       |    11.0306 |      1      | False     |    14.1973 |      0.6909 | False     |    12.2046 |      0.9961 | False     |
| LSTM          |    12.3635 |      1      | False     |    12.1211 |      1      | False     |     6.8883 |      1      | False     |
| Transformer   |    11.3793 |      1      | False     |    10.2816 |      1      | False     |    11.0534 |      1      | False     |
| TCN           |    22.0775 |      0.0573 | False     |    30.0223 |      0.0025 | True      |    23.5759 |      0.0324 | True      |
| PatchTST      |     9.0452 |      1      | False     |     9.5971 |      1      | False     |    13.7078 |      0.8074 | False     |
| TSMixer       |    14.9077 |      0.6097 | False     |     8.6626 |      1      | False     |     9.2267 |      1      | False     |

## 4. Giacomini-White CPA Test Significance Comparison

| Window   | Significant_Models_at_5pct   |
|:---------|:-----------------------------|
| 10D      | 8 / 60                       |
| 20D      | 9 / 60                       |
| 60D      | 6 / 60                       |

## 5. Detailed OOS R² Comparison Across Regimes

| Model       | Horizon   |   Low_10D |   Low_20D |   Low_60D |   Normal_10D |   Normal_20D |   Normal_60D |   High_10D |   High_20D |   High_60D |
|:------------|:----------|----------:|----------:|----------:|-------------:|-------------:|-------------:|-----------:|-----------:|-----------:|
| ARIMA       | 1D        |      0.93 |      0.05 |      0.08 |         0.01 |        -0.47 |        -0.6  |      -1.02 |      -0.59 |      -0.54 |
| ARIMA       | 5D        |     -7.14 |     -2.74 |     -1.83 |        -5.43 |        -7.6  |        -6.52 |      -9.94 |      -9.49 |     -11.13 |
| ARIMA       | 20D       |     -6.56 |     -6.67 |     -3.08 |        -6.98 |        -5.99 |        -6.82 |     -17.63 |     -17.15 |     -18.84 |
| ARIMA       | 60D       |     -4.04 |     -1.3  |    -14.77 |        -6.79 |       -10.76 |        -7.15 |     -15.22 |     -12.41 |     -12.58 |
| ARIMA       | 120D      |     -3.04 |     -7.43 |     -2.68 |        10.9  |        11.61 |         5.95 |      -4.41 |      -1.99 |      -1.69 |
| ARIMAX      | 1D        |      0.85 |     -0.01 |      0.16 |         0.35 |         0.85 |         0.64 |       1.67 |       1.55 |       1.73 |
| ARIMAX      | 5D        |    -12.11 |     -9.19 |     -5.86 |       -11.36 |        -6.26 |        -4.17 |      -2    |      -5.18 |      -7.34 |
| ARIMAX      | 20D       |     -7.77 |     -4.38 |      1.71 |       -12.11 |        -8.76 |        -8.82 |       0.85 |      -2.47 |      -3.18 |
| ARIMAX      | 60D       |     -4.02 |     -0.53 |      3.49 |        -0.31 |        -4.13 |        -0.5  |       6.99 |       8.09 |       5.49 |
| ARIMAX      | 120D      |     -8.09 |    -15.27 |     -4.21 |         0.35 |         1.28 |        -1.34 |       2.1  |       3.99 |       2.46 |
| Ridge       | 1D        |      0.56 |     -1.06 |     -0.35 |         0.06 |         0.53 |         0.07 |       2.41 |       2.41 |       2.75 |
| Ridge       | 5D        |     -0.64 |     -1.3  |     -0.34 |         1.43 |         1.09 |        -0.72 |       2.35 |       2.64 |       4.33 |
| Ridge       | 20D       |     -2.64 |     -1.95 |     -4.96 |         0.25 |         0.52 |         2.66 |       3.44 |       2.81 |       2.47 |
| Ridge       | 60D       |      0.96 |      1.12 |     -2.27 |         3.62 |         4.23 |         4.75 |      -1.56 |      -1.51 |      -2.47 |
| Ridge       | 120D      |    -12.86 |    -17.17 |    -18.95 |        -4.57 |        -3.51 |        -1.43 |      -2.13 |      -1.47 |      -4.35 |
| Elastic_Net | 1D        |      0.48 |      0.15 |     -0.49 |         0.22 |         0.11 |         0.24 |       0.01 |       0.13 |       0.19 |
| Elastic_Net | 5D        |      0.92 |     -0.19 |     -0.31 |         0.3  |         0.7  |         0.52 |       0.73 |       0.7  |       0.9  |
| Elastic_Net | 20D       |      0    |      0    |      0    |         0    |         0    |         0    |       0    |       0    |       0    |
| Elastic_Net | 60D       |      0    |      0    |      0    |         0    |         0    |         0    |       0    |       0    |       0    |
| Elastic_Net | 120D      |      0    |      0    |      0    |         0    |         0    |         0    |       0    |       0    |       0    |
| SVR         | 1D        |      0.04 |     -0.21 |      0.74 |         0.14 |         0.43 |         0.35 |       3.3  |       3.1  |       3.23 |
| SVR         | 5D        |     -0.56 |     -1.18 |     -4.07 |         0.61 |         0.89 |         1.88 |       2.26 |       2.21 |       2.42 |
| SVR         | 20D       |     -1.49 |      0.12 |     -8.55 |         2.72 |         7.08 |         8.3  |       4.18 |       1.39 |       2.1  |
| SVR         | 60D       |      5.76 |      3.49 |      6.78 |         9.54 |        16.54 |        11.88 |      -0.11 |      -3.49 |      -3.42 |
| SVR         | 120D      |    -34.11 |    -33.02 |    -14.62 |       -29.12 |       -33.19 |       -21.28 |      -3.71 |      -2.98 |     -17.85 |

*(Table truncated to first 25 rows for readability; see `comparison_summary/regime_oos_r2_comparison.csv` for the full table)*

