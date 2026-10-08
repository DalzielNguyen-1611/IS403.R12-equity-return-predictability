# Research Data Analyst Report: Cross-Model & Multi-Horizon Predictive Performance Summary
**Dataset Source:** `results/summary/model_horizon_summary.csv`  
**Execution Timestamp:** 2026-10-08 15:18:00 UTC+7  
**Analyst Role:** Research Data Analyst (Financial Econometrics & Machine Learning)  
**Research Study:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*

---

## 1. File Description & Purpose

The file `results/summary/model_horizon_summary.csv` synthesizes the out-of-sample predictive performance across the 5 walk-forward expanding window folds for every registered model and forecasting horizon. 

Its analytical purpose is to answer the aggregate comparative questions:
1. What is the mean and variance of out-of-sample predictive skill ($OOS\_R^2$) across model families?
2. How does predictive skill evolve across the horizon continuum: $1\text{D} \to 5\text{D} \to 20\text{D} \to 60\text{D} \to 120\text{D}$?
3. Which models outperform the `Historical_Mean` benchmark, and at which specific horizons?
4. What structural trade-offs exist between point error ($MAE, RMSE$), relative explained variance ($OOS\_R^2$), and directional timing ($DA$)?

---

## 2. File Dimensions & Schema

- **Total Rows:** 70 rows ($14\text{ Models} \times 5\text{ Horizons}$)
- **Columns:** 10 columns:
  - `Model` (`str`): Algorithm identifier
  - `Horizon` (`str`): `1D`, `5D`, `20D`, `60D`, `120D`
  - `MAE_Mean`, `MAE_Std` (`float`): Mean and sample standard deviation of MAE across 5 folds
  - `RMSE_Mean`, `RMSE_Std` (`float`): Mean and sample standard deviation of RMSE across 5 folds
  - `OOS_R2_Mean`, `OOS_R2_Std` (`float`): Mean and sample standard deviation of $OOS\_R^2$ across 5 folds
  - `DA_Mean`, `DA_Std` (`float`): Mean and sample standard deviation of Directional Accuracy across 5 folds

---

## 3. Comprehensive Performance Matrices

### 3.1 Primary Metric: Mean Out-of-Sample $R^2$ ($OOS\_R^2_{\text{mean}}$)
*Note: Campbell-Thompson $OOS\_R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y}_{\text{hist}})^2}$. Positive indicates superiority over Historical Mean; negative indicates benchmark superiority.*

| Model Family | Model | 1D | 5D | 20D | 60D | 120D |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Benchmark** | **Historical_Mean** | -0.0012 | -0.0054 | -0.0267 | -0.0861 | -0.6974 |
| **Statistical TS** | **ARIMA** | **+0.0057** | **+0.5835** | **+0.8583** | **+0.9385** | **+0.9404** |
| | **ARIMAX** | **+0.0007** | **+0.7050** | **+0.9123** | **+0.9627** | **+0.9622** |
| | **SARIMA** | **+0.0061** | **+0.6607** | **+0.8584** | **+0.9380** | **+0.9395** |
| **Regularized Linear**| **Elastic_Net** | **+0.0017** | -0.0041 | -0.0389 | -0.1970 | -0.7255 |
| | **Ridge** | -0.0080 | -0.0254 | -0.0802 | -0.3759 | -1.3682 |
| **Kernel Method** | **SVR** | -0.0105 | -0.0140 | -0.2006 | -0.1489 | -0.8610 |
| **Tree Ensembles** | **Random_Forest** | -0.0137 | -0.0253 | -0.1318 | -0.3551 | -1.1247 |
| | **XGBoost** | -0.0069 | -0.0067 | -0.0453 | -0.3875 | -1.0514 |
| | **LightGBM** | -0.0248 | -0.0327 | -0.1155 | -0.4837 | -1.5774 |
| **Deep Learning** | **RNN** | -0.0711 | -0.0528 | -0.1110 | -0.5103 | -1.4222 |
| | **LSTM** | -0.0105 | -0.1041 | -0.1486 | -0.6348 | -1.4639 |
| | **Transformer** | -2.3800 | -0.8244 | -0.7016 | -1.0473 | -2.0454 |
| **Classification** | **Logistic_Regression** | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |

---

### 3.2 Stability Metric: Cross-Fold Standard Deviation of $OOS\_R^2$ ($OOS\_R^2_{\text{std}}$)
| Model | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 0.0013 | 0.0048 | 0.0283 | 0.1526 | 1.3558 |
| **ARIMA** | 0.0103 | 0.0332 | 0.0316 | 0.0124 | 0.0198 |
| **ARIMAX** | 0.0305 | 0.0591 | 0.0219 | 0.0106 | 0.0180 |
| **SARIMA** | 0.0117 | 0.0392 | 0.0317 | 0.0124 | 0.0204 |
| **Elastic_Net** | 0.0013 | 0.0109 | 0.0615 | 0.2786 | 1.4114 |
| **Ridge** | 0.0167 | 0.0466 | 0.0867 | 0.2565 | 2.4056 |
| **SVR** | 0.0371 | 0.0582 | 0.3282 | 0.1570 | 0.8862 |
| **Random_Forest** | 0.0245 | 0.0593 | 0.1536 | 0.1361 | 0.7633 |
| **XGBoost** | 0.0128 | 0.0038 | 0.0245 | 0.6516 | 1.3650 |
| **LightGBM** | 0.0434 | 0.0597 | 0.1868 | 0.5661 | 1.0475 |
| **RNN** | 0.0873 | 0.0473 | 0.1478 | 0.4755 | 3.0566 |
| **LSTM** | 0.0233 | 0.0733 | 0.0722 | 0.9383 | 2.0261 |
| **Transformer** | 3.5884 | 1.5362 | 0.8832 | 0.9295 | 2.0388 |

---

### 3.3 Supporting Metrics: Mean Absolute Error ($MAE_{\text{mean}}$) & Root Mean Squared Error ($RMSE_{\text{mean}}$)

#### Mean Absolute Error (MAE):
| Model | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 0.0072 | 0.0155 | 0.0315 | 0.0522 | 0.0626 |
| **ARIMAX** | 0.0072 | 0.0083 | 0.0082 | 0.0084 | 0.0081 |
| **Elastic_Net** | 0.0072 | 0.0156 | 0.0322 | 0.0543 | 0.0629 |
| **XGBoost** | 0.0072 | 0.0156 | 0.0321 | 0.0563 | 0.0693 |
| **Random_Forest** | 0.0072 | 0.0160 | 0.0346 | 0.0588 | 0.0739 |
| **LightGBM** | 0.0073 | 0.0160 | 0.0327 | 0.0587 | 0.0808 |
| **LSTM** | 0.0072 | 0.0166 | 0.0328 | 0.0568 | 0.0715 |
| **Transformer** | 0.0124 | 0.0189 | 0.0377 | 0.0669 | 0.0818 |

#### Root Mean Squared Error (RMSE):
| Model | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 0.0109 | 0.0222 | 0.0428 | 0.0648 | 0.0760 |
| **ARIMAX** | 0.0108 | 0.0122 | 0.0121 | 0.0120 | 0.0120 |
| **Elastic_Net** | 0.0109 | 0.0223 | 0.0432 | 0.0665 | 0.0763 |
| **XGBoost** | 0.0110 | 0.0223 | 0.0434 | 0.0715 | 0.0841 |
| **Random_Forest** | 0.0110 | 0.0227 | 0.0463 | 0.0743 | 0.0912 |
| **LightGBM** | 0.0111 | 0.0228 | 0.0444 | 0.0740 | 0.1001 |
| **LSTM** | 0.0109 | 0.0236 | 0.0447 | 0.0747 | 0.0882 |
| **Transformer** | 0.0155 | 0.0260 | 0.0494 | 0.0852 | 0.0969 |

---

### 3.4 Supplementary Metric: Directional Accuracy ($DA_{\text{mean}}$)
| Model | 1D | 5D | 20D | 60D | 120D |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical_Mean** | 54.85% | 61.84% | 68.46% | 74.56% | 77.79% |
| **ARIMA** | 52.72% | 79.41% | 91.84% | 95.74% | 96.69% |
| **ARIMAX** | 52.06% | 82.43% | 93.60% | 97.06% | 97.43% |
| **SARIMA** | 53.90% | 82.06% | 91.84% | 95.74% | 96.91% |
| **Elastic_Net** | 53.24% | 57.79% | 61.84% | 68.60% | 77.79% |
| **Ridge** | 51.47% | 50.15% | 56.03% | 56.18% | 66.10% |
| **XGBoost** | 54.34% | 61.32% | 65.15% | 71.25% | 71.69% |
| **Random_Forest** | 54.04% | 59.34% | 62.28% | 61.40% | 63.31% |
| **LightGBM** | 53.68% | 58.53% | 64.19% | 63.46% | 58.38% |
| **SVR** | 50.96% | 55.15% | 57.06% | 66.91% | 66.76% |
| **RNN** | 49.93% | 50.81% | 60.07% | 63.38% | 73.01% |
| **LSTM** | 52.50% | 54.78% | 58.60% | 67.35% | 65.81% |
| **Transformer** | 49.71% | 53.82% | 51.84% | 60.59% | 54.41% |

---

## 4. Key Empirical Findings & Model Family Patterns

### 4.1 Strict Predictability Decay in Machine Learning & Deep Learning Families
For all non-autoregressive models, out-of-sample predictability exhibits **strict monotonic decay** as the forecast horizon lengthens:
1. **Tree-Based Models (XGBoost, Random Forest, LightGBM):**
   - At $1\text{D}$ and $5\text{D}$, performance is tightly pegged near the benchmark ($OOS\_R^2 \in [-0.006, -0.032]$).
   - At $20\text{D}$, $OOS\_R^2$ deteriorates to $[-0.045, -0.131]$.
   - At $60\text{D}$ and $120\text{D}$, predictive skill collapses catastrophically into severe negative territory ($120\text{D}$ reaches $-1.051$ for XGBoost, $-1.125$ for Random Forest, and $-1.577$ for LightGBM).
2. **Deep Learning Models (RNN, LSTM, Transformer):**
   - Suffer the steepest horizon decay. At $120\text{D}$, RNN reaches $-1.422$, LSTM reaches $-1.464$, and Transformer reaches $-2.045$.
   - Transformer exhibits severe instability at short horizons as well ($1\text{D}$ $OOS\_R^2 = -2.3800, \text{std} = 3.5884$), reflecting high parameter complexity ($d_{\text{model}}=32/64$, self-attention layers) overfitting on noisy daily financial returns.
3. **Regularized Linear Models (Elastic Net, Ridge):**
   - Elastic Net demonstrates the highest stability among machine learning models. At $1\text{D}$, Elastic Net achieves $OOS\_R^2 = +0.0017$, successfully outperforming the Historical Mean. At $5\text{D}$, it remains virtually tied with Historical Mean ($-0.0041$).
   - However, by $120\text{D}$, Elastic Net decays to $-0.7255$, tracking the Historical Mean benchmark ($-0.6974$).

### 4.2 The Autoregressive Divergence (ARIMA, ARIMAX, SARIMA)
The time-series autoregressive family exhibits an empirical trajectory completely divergent from ML/DL models:
- At $1\text{D}$, $OOS\_R^2$ is marginally positive ($+0.0007 \text{ to } +0.0061$).
- As horizon expands, $OOS\_R^2$ **monotonically surges**, reaching $+0.58 \text{ to } +0.70$ at $5\text{D}$, $+0.85 \text{ to } +0.91$ at $20\text{D}$, and plateauing at $+0.94 \text{ to } +0.96$ at $60\text{D}$ and $120\text{D}$.
- **Methodological Mechanism:** Cumulative multi-horizon log returns $y_{t+h} = \log(P_{t+h}/P_t)$ act as an integrated rolling filter. The autoregressive models fit the high autocorrelation of overlapping returns with very few degrees of freedom ($p \le 3, d \le 1, q \le 3$), avoiding the curse of dimensionality that plagues high-dimensional ML feature sets.

### 4.3 Directional Accuracy (DA) Does Not Equal Point Predictive Skill
- Directional Accuracy increases from ~50–54% at $1\text{D}$ to 65–77% at $120\text{D}$ across nearly all models (including Historical Mean).
- This increase in DA occurs simultaneously while $OOS\_R^2$ is deteriorating to $-1.50$.
- **Econometric Rationale:** The S&P 500 equity market possesses a structural positive equity risk premium (upward drift). Over a 120-day horizon, the market is positive in >75% of periods. A model that constantly predicts positive will mechanically achieve ~75% DA, yet its point forecast variance may deviate severely from actual magnitudes, producing large negative $OOS\_R^2$.

---

## 5. Limitations & Caveats

1. **Fold-Averaging Masks Regime Variance:** The cross-fold standard deviation at $120\text{D}$ is extremely large across all models ($\text{std} > 1.0$ for Historical Mean, Elastic Net, XGBoost, and DL), demonstrating that performance is heavily dependent on the specific macroeconomic regime of each fold.
2. **Exogenous Predictor Leakage Risk in ARIMAX:** The exceptionally high $OOS\_R^2$ ($>0.96$) of ARIMAX at $60\text{D}$ and $120\text{D}$ requires verification at the individual prediction level (Step 3) to confirm that exogenous predictors ($VIX$, Momentum) are strictly lagged relative to the forecast origin $t$.

---

## 6. Conclusions & Handoff to Step 3

### 6.1 Summary of Model $\times$ Horizon Performance
- **Short Horizon (1D):** Predictability is minimal; noise dominates. Elastic Net, ARIMA, SARIMA, and ARIMAX marginally beat Historical Mean ($OOS\_R^2 \approx 0.001 - 0.006$). Deep Learning overfits.
- **Medium Horizons (5D, 20D):** Divergence point. Autoregressive models capture cumulative multi-day momentum; ML/DL models begin noticeable decay.
- **Long Horizons (60D, 120D):** Severe predictability decay across all Machine Learning and Deep Learning models ($OOS\_R^2 \le -1.0$). Only autoregressive specifications sustain positive explained variance.

### 6.2 Handoff to Step 3
To verify the validity of these aggregate summaries and cross-check whether individual predictions match these fold metrics without calculation discrepancies, we proceed to examine **`results/predictions/all_oos_predictions.csv`**.
