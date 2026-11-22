# Benchmark Robustness Evaluation: Historical Mean vs. Zero-Return Benchmark

## 1. Executive Summary

- **Evaluation Period**: 2020-11-02 to 2026-04-07 (Holdout Test, 1,362 trading days).
- **Models Evaluated**: 12 models across 5 forecasting horizons (60 model-horizon pairs).
- **Average OOS $R^2$ under Historical Mean Benchmark**: `-10.545%`
- **Average OOS $R^2$ under Zero-Return Benchmark**: `0.073%`
- **Mean Difference ($\Delta R_{OOS}^2 = R_{Zero}^2 - R_{HM}^2$)**: `+10.618%`
- **Statistically Significant Pairs (DM Test at 5%)**:
  - Against Historical Mean: `1 / 60` (1.7%)
  - Against Zero-Return Benchmark: `8 / 60` (13.3%)

## 2. Theoretical Econometric Consistency

According to Welch & Goyal (2008) and Campbell & Thompson (2008), for an equity market with a positive historical equity risk premium (drift $\bar{y}_{\text{train}} > 0$):

$$SSE_{\text{Zero}} = \sum_{t} y_t^2 > SSE_{\text{HM}} = \sum_{t} (y_t - \bar{y}_{\text{train}})^2$$

Consequently, the denominator of $R_{OOS}^2$ under the Zero benchmark is larger, which systematically implies:

$$R_{OOS, \text{Zero}}^2 \ge R_{OOS, \text{HM}}^2$$

This confirms that the Historical Mean benchmark employed as the primary baseline in this study is strictly more conservative and harder to outperform than the Zero-return random walk benchmark.

## 3. Comparison Matrices

### Panel A: Out-of-Sample $R^2$ (%) under Historical Mean Benchmark

| Model         |      1D |      5D |      20D |     60D |     120D |
|:--------------|--------:|--------:|---------:|--------:|---------:|
| ARIMA         |  -0.487 |  -8.248 |  -12.701 | -10.263 |    1.95  |
| ARIMAX        |   1.181 |  -5.904 |   -4.61  |   2.518 |   -0.349 |
| Ridge         |   1.498 |   1.786 |    1.657 |   0.906 |   -4.947 |
| Elastic_Net   |   0.127 |   0.609 |    0     |   0     |    0     |
| SVR           |   1.985 |   1.479 |    3.048 |   4.639 |  -19.097 |
| Random_Forest |   1.771 |  -1.297 |   -0.393 |   0.331 |   -6.417 |
| XGBoost       |   1.478 |   0.109 |    1.374 |   0.13  |   -0.141 |
| LSTM          |  -0.651 |  -0.505 |    1.475 |  -0.959 |   -3.337 |
| Transformer   | -15.28  | -39.717 |  -11.637 | -21.282 |  -23.024 |
| TCN           | -14.156 | -64.91  | -123.428 | -87.232 | -142.548 |
| PatchTST      |  -4.251 |  -6.853 |  -12.171 | -21.863 |  -12.215 |
| TSMixer       |  -2.719 |  -5.472 |    3.795 |   4.982 |   17.535 |

### Panel B: Out-of-Sample $R^2$ (%) under Zero-Return Benchmark

| Model         |      1D |      5D |      20D |     60D |    120D |
|:--------------|--------:|--------:|---------:|--------:|--------:|
| ARIMA         |  -0.286 |  -7.102 |   -7.414 |   6.408 |  27.033 |
| ARIMAX        |   1.379 |  -4.782 |    0.298 |  17.257 |  25.322 |
| Ridge         |   1.695 |   2.826 |    6.27  |  15.888 |  21.9   |
| Elastic_Net   |   0.327 |   1.662 |    4.691 |  15.119 |  25.582 |
| SVR           |   2.181 |   2.523 |    7.596 |  19.057 |  11.37  |
| Random_Forest |   1.968 |  -0.224 |    4.316 |  15.4   |  20.806 |
| XGBoost       |   1.676 |   1.167 |    6.001 |  15.229 |  25.476 |
| LSTM          |  -0.449 |   0.56  |    6.097 |  14.305 |  23.098 |
| Transformer   | -15.049 | -38.237 |   -6.4   |  -2.945 |   8.447 |
| TCN           | -13.927 | -63.163 | -112.947 | -58.924 | -80.5   |
| PatchTST      |  -4.042 |  -5.722 |   -6.909 |  -3.438 |  16.491 |
| TSMixer       |  -2.513 |  -4.355 |    8.308 |  19.348 |  38.631 |

### Panel C: Difference $\Delta R_{OOS}^2$ (% points, Zero - HM)

| Model         |    1D |    5D |    20D |    60D |   120D |
|:--------------|------:|------:|-------:|-------:|-------:|
| ARIMA         | 0.201 | 1.146 |  5.287 | 16.671 | 25.083 |
| ARIMAX        | 0.198 | 1.122 |  4.907 | 14.738 | 25.671 |
| Ridge         | 0.197 | 1.04  |  4.613 | 14.982 | 26.847 |
| Elastic_Net   | 0.2   | 1.053 |  4.691 | 15.119 | 25.582 |
| SVR           | 0.196 | 1.043 |  4.548 | 14.418 | 30.467 |
| Random_Forest | 0.197 | 1.073 |  4.709 | 15.069 | 27.223 |
| XGBoost       | 0.197 | 1.058 |  4.627 | 15.1   | 25.618 |
| LSTM          | 0.202 | 1.064 |  4.622 | 15.264 | 26.435 |
| Transformer   | 0.231 | 1.48  |  5.237 | 18.337 | 31.472 |
| TCN           | 0.229 | 1.746 | 10.481 | 28.308 | 62.048 |
| PatchTST      | 0.209 | 1.132 |  5.262 | 18.425 | 28.706 |
| TSMixer       | 0.206 | 1.117 |  4.513 | 14.366 | 21.096 |

## 4. Volatility Regime Cross-Model Average

| Horizon   | Regime   |   OOS_R2_HM |   OOS_R2_Zero |   Delta_OOS_R2 |
|:----------|:---------|------------:|--------------:|---------------:|
| 1D        | Low      |     -0.0404 |       -0.0398 |         0.0007 |
| 1D        | Normal   |     -0.0441 |       -0.0407 |         0.0034 |
| 1D        | High     |     -0.0126 |       -0.011  |         0.0017 |
| 5D        | Low      |     -0.1073 |       -0.0891 |         0.0182 |
| 5D        | Normal   |     -0.0857 |       -0.0663 |         0.0193 |
| 5D        | High     |     -0.1181 |       -0.1113 |         0.0068 |
| 20D       | Low      |     -0.0801 |        0.0003 |         0.0804 |
| 20D       | Normal   |     -0.035  |        0.0771 |         0.112  |
| 20D       | High     |     -0.1841 |       -0.1763 |         0.0078 |
| 60D       | Low      |     -0.0235 |        0.1728 |         0.1963 |
| 60D       | Normal   |      0.0052 |        0.2654 |         0.2602 |
| 60D       | High     |     -0.2094 |       -0.1667 |         0.0428 |
| 120D      | Low      |     -0.0763 |        0.2316 |         0.3079 |
| 120D      | Normal   |     -0.0411 |        0.3382 |         0.3793 |
| 120D      | High     |     -0.2843 |       -0.1215 |         0.1628 |

---
*Note: This report is generated programmatically by `run_benchmark_robustness.py` with zero model retraining.*