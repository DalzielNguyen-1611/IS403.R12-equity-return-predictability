# Statistical Hypothesis Testing Report (H1, H2, H3)

## 1. Overview
This report provides rigorous statistical inference for the three research hypotheses evaluating the S&P 500 forecasting models.

## 2. Hypothesis H1: Multi-Horizon Predictive Performance
Joint equality null hypothesis: $H_0: \mu_{1D} = \mu_{5D} = \mu_{20D} = \mu_{60D} = \mu_{120D}$.

| Model         |   Wald_Stat |   df |     pvalue |   pvalue_holm | Significant_05   |
|:--------------|------------:|-----:|-----------:|--------------:|:-----------------|
| ARIMA         |    15.7438  |    4 | 0.00338289 |     0.0405947 | True             |
| ARIMAX        |     7.3218  |    4 | 0.119829   |     0.838801  | False            |
| Ridge         |     5.56163 |    4 | 0.234365   |     0.93746   | False            |
| Elastic_Net   |     2.74794 |    2 | 0.2531     |     0.93746   | False            |
| SVR           |     9.75346 |    4 | 0.0447918  |     0.403126  | False            |
| Random_Forest |     6.70055 |    4 | 0.152585   |     0.838801  | False            |
| XGBoost       |     7.81635 |    4 | 0.0985421  |     0.788337  | False            |
| LSTM          |     4.61763 |    4 | 0.328826   |     0.93746   | False            |
| Transformer   |     6.9395  |    4 | 0.13912    |     0.838801  | False            |
| TCN           |    11.1244  |    4 | 0.0252007  |     0.252007  | False            |
| PatchTST      |     4.61942 |    4 | 0.328622   |     0.93746   | False            |
| TSMixer       |    12.3816  |    4 | 0.0147282  |     0.162011  | False            |

## 3. Hypothesis H2: Volatility-Regime Interaction
Joint interaction null hypothesis: $H_0: \text{All Horizon } \times \text{ Regime Interaction Effects} = 0$.

| Model         |   Interaction_Wald_Stat |   df |      pvalue |   pvalue_holm | Significant_05   |
|:--------------|------------------------:|-----:|------------:|--------------:|:-----------------|
| ARIMA         |                17.118   |    8 | 0.0289037   |    0.289037   | False            |
| ARIMAX        |                 9.28261 |    8 | 0.319019    |    1          | False            |
| Ridge         |                 6.44389 |    8 | 0.597638    |    1          | False            |
| Elastic_Net   |                 3.49819 |    4 | 0.478153    |    1          | False            |
| SVR           |                 9.44793 |    8 | 0.305929    |    1          | False            |
| Random_Forest |                21.724   |    8 | 0.00545365  |    0.0599901  | False            |
| XGBoost       |                14.1973  |    8 | 0.0767663   |    0.690897   | False            |
| LSTM          |                12.1211  |    8 | 0.14588     |    1          | False            |
| Transformer   |                10.2816  |    8 | 0.245815    |    1          | False            |
| TCN           |                30.0223  |    8 | 0.000209466 |    0.00251359 | True             |
| PatchTST      |                 9.59712 |    8 | 0.294449    |    1          | False            |
| TSMixer       |                 8.6626  |    8 | 0.371553    |    1          | False            |

## 4. Hypothesis H3: Temporal Instability across Hold-out Blocks
Block equality null hypothesis: $H_0: \mu_{2021-2022} = \mu_{2023-2024} = \mu_{2025-2026}$.

| Model         |   Wald_Stat |   df |       pvalue |   pvalue_holm | Significant_05   |
|:--------------|------------:|-----:|-------------:|--------------:|:-----------------|
| ARIMA         |    4.40465  |    2 |   0.110546   |    0.884369   | False            |
| ARIMAX        |    2.91953  |    2 |   0.232291   |    1          | False            |
| Ridge         |    1.29513  |    2 |   0.523319   |    1          | False            |
| Elastic_Net   |    0        |    0 | nan          |    1          | False            |
| SVR           |    4.79514  |    2 |   0.0909388  |    0.818449   | False            |
| Random_Forest |    1.76876  |    2 |   0.412969   |    1          | False            |
| XGBoost       |    1.95996  |    2 |   0.375318   |    1          | False            |
| LSTM          |    9.26276  |    2 |   0.00974132 |    0.107154   | False            |
| Transformer   |    1.49021  |    2 |   0.474684   |    1          | False            |
| TCN           |   15.6699   |    2 |   0.00039566 |    0.00474791 | True             |
| PatchTST      |    8.34135  |    2 |   0.0154418  |    0.154418   | False            |
| TSMixer       |    0.665269 |    2 |   0.717032   |    1          | False            |
