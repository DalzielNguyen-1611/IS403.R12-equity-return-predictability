# ABLATION STUDY: FEATURE GROUP MARGINAL CONTRIBUTIONS
*Generated automatically on:* `2026-11-13 00:28:16`

> [!NOTE]
> $\Delta R^2_{\mathrm{OOS}} = R^2_{\mathrm{OOS}}(\text{Ablated}) - R^2_{\mathrm{OOS}}(\text{Full})$.
> A **negative $\Delta R^2$** indicates that removing the feature group hurts predictive performance (i.e. the feature group has positive marginal predictive value).

## 1. Model: LSTM

### $\Delta R^2_{\mathrm{OOS}}$ Comparison (Drop vs Full)
| Configuration             |    120D |     1D |     20D |     5D |    60D |
|:--------------------------|--------:|-------:|--------:|-------:|-------:|
| Full_Model                |  0      | 0      |  0      | 0      | 0      |
| w/o Lagged Returns        |  0.0394 | 0.0128 | -0.0295 | 0.0092 | 0.0309 |
| w/o Momentum              |  0.0308 | 0.0075 | -0.0092 | 0.0044 | 0.0491 |
| w/o Historical Volatility |  0.0619 | 0.0215 | -0.0033 | 0.0162 | 0.0199 |
| w/o VIX                   |  0.0964 | 0.0096 | -0.0152 | 0.0099 | 0.0056 |
| w/o Intraday              |  0.0644 | 0.0157 | -0.006  | 0.0201 | 0.0208 |
| w/o Trading Volume        | -0.0286 | 0.0169 | -0.0173 | 0.0162 | 0.0118 |


### Raw $R^2_{\mathrm{OOS}}$ Metrics
| Configuration             |    120D |      1D |     20D |      5D |     60D |
|:--------------------------|--------:|--------:|--------:|--------:|--------:|
| Full_Model                | -0.0313 | -0.0062 |  0.0148 | -0.0058 | -0.0096 |
| w/o Lagged Returns        |  0.0081 |  0.0066 | -0.0147 |  0.0034 |  0.0213 |
| w/o Momentum              | -0.0005 |  0.0013 |  0.0056 | -0.0014 |  0.0396 |
| w/o Historical Volatility |  0.0306 |  0.0153 |  0.0115 |  0.0104 |  0.0104 |
| w/o VIX                   |  0.0651 |  0.0034 | -0.0004 |  0.0041 | -0.0039 |
| w/o Intraday              |  0.0331 |  0.0095 |  0.0088 |  0.0143 |  0.0112 |
| w/o Trading Volume        | -0.0599 |  0.0107 | -0.0025 |  0.0104 |  0.0022 |

---

## 1. Model: RIDGE

### $\Delta R^2_{\mathrm{OOS}}$ Comparison (Drop vs Full)
| Configuration             |    120D |      1D |     20D |      5D |     60D |
|:--------------------------|--------:|--------:|--------:|--------:|--------:|
| Full_Model                |  0      |  0      |  0      |  0      |  0      |
| w/o Lagged Returns        |  0.0005 |  0.001  | -0.0003 | -0.0004 | -0.0001 |
| w/o Momentum              | -0.0076 |  0.0004 | -0.0008 |  0.0029 | -0.0127 |
| w/o Historical Volatility |  0.043  | -0.0037 | -0.0055 | -0.0038 |  0.0182 |
| w/o VIX                   |  0.0653 | -0.005  | -0.0113 | -0.0044 |  0.0093 |
| w/o Intraday              | -0.0002 | -0.0007 | -0.0007 | -0.002  | -0.0025 |
| w/o Trading Volume        | -0.0001 | -0.0007 |  0      | -0      | -0      |


### Raw $R^2_{\mathrm{OOS}}$ Metrics
| Configuration             |    120D |     1D |    20D |     5D |     60D |
|:--------------------------|--------:|-------:|-------:|-------:|--------:|
| Full_Model                | -0.0501 | 0.0149 | 0.0166 | 0.0179 |  0.009  |
| w/o Lagged Returns        | -0.0495 | 0.016  | 0.0163 | 0.0175 |  0.0089 |
| w/o Momentum              | -0.0577 | 0.0153 | 0.0158 | 0.0208 | -0.0037 |
| w/o Historical Volatility | -0.0071 | 0.0112 | 0.0111 | 0.0141 |  0.0272 |
| w/o VIX                   |  0.0152 | 0.0099 | 0.0053 | 0.0136 |  0.0183 |
| w/o Intraday              | -0.0502 | 0.0143 | 0.0159 | 0.0159 |  0.0065 |
| w/o Trading Volume        | -0.0502 | 0.0143 | 0.0166 | 0.0179 |  0.009  |

---

## 1. Model: XGBOOST

### $\Delta R^2_{\mathrm{OOS}}$ Comparison (Drop vs Full)
| Configuration             |    120D |      1D |    20D |      5D |   60D |
|:--------------------------|--------:|--------:|-------:|--------:|------:|
| Full_Model                |  0      |  0      | 0      |  0      |     0 |
| w/o Lagged Returns        | -0.0005 | -0.0032 | 0.0067 |  0.0006 |     0 |
| w/o Momentum              | -0.0005 |  0.0005 | 0.0051 | -0.0011 |     0 |
| w/o Historical Volatility | -0.0005 | -0.0067 | 0.012  | -0.0054 |     0 |
| w/o VIX                   |  0.0018 |  0.0001 | 0.0065 |  0.0053 |     0 |
| w/o Intraday              | -0.0005 | -0.012  | 0.0075 |  0.0009 |     0 |
| w/o Trading Volume        |  0      |  0.006  | 0.0037 |  0.0021 |     0 |


### Raw $R^2_{\mathrm{OOS}}$ Metrics
| Configuration             |    120D |     1D |    20D |      5D |    60D |
|:--------------------------|--------:|-------:|-------:|--------:|-------:|
| Full_Model                | -0.0014 | 0.0153 | 0.0052 |  0.0011 | 0.0013 |
| w/o Lagged Returns        | -0.0019 | 0.0121 | 0.0118 |  0.0017 | 0.0013 |
| w/o Momentum              | -0.0019 | 0.0157 | 0.0103 | -0      | 0.0013 |
| w/o Historical Volatility | -0.0019 | 0.0086 | 0.0172 | -0.0043 | 0.0013 |
| w/o VIX                   |  0.0004 | 0.0154 | 0.0117 |  0.0063 | 0.0013 |
| w/o Intraday              | -0.0019 | 0.0032 | 0.0127 |  0.002  | 0.0013 |
| w/o Trading Volume        | -0.0014 | 0.0213 | 0.0089 |  0.0032 | 0.0013 |

---
