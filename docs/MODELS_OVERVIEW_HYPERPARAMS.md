# 📋 BẢNG TỔNG HỢP TOÀN BỘ MÔ HÌNH VÀ BỘ THAM SỐ TỐI ƯU SAU KHI TUNING

> **Đề tài nghiên cứu:** *When Does Equity Return Predictability Decay? Multi-Horizon & Regime Analysis*  
> **Quy tắc thể hiện:** Toàn bộ 46 đối tượng thực nghiệm qua 5 đợt (Lan01 → Lan05). Mô hình nào **đã qua Tuning** (Optuna, Grid Search, Fixed Search) sẽ hiển thị đầy đủ bộ tham số tốt nhất; mô hình nào **chưa Tuning** hoặc **không dùng Tuning** sẽ được **để trống hoàn toàn** theo đúng yêu cầu.

---

> **Cập nhật 26/10/2026:** Đã có đủ 18 trường hợp Lan04–Lan05: 9 cấu hình mỗi đợt × 5 chân trời × 5 fold = **450 lượt đánh giá**. Tham số/điểm trong bảng master là cấu hình đại diện **1D / Fold 5**, không phải bộ tối ưu chung cho mọi chân trời. Lan04 giữ trống tham số tuning vì dùng quy tắc mặc định.

## 1. BẢNG OVERVIEW TỔNG THỂ (MASTER MODEL & TUNING MATRIX)

| STT | Đợt Chạy | Tên Mô Hình | Họ Kiến Trúc | Phương Pháp Tuning | Không Gian / Trials | Trạng Thái Tuning | Bộ Tham Số Tốt Nhất Sau Tuning | Điểm Đánh Giá Tốt Nhất (Val Score) |
| :-: | :---: | :--- | :--- | :--- | :-: | :--- | :--- | :-: |
| 1 | **Lan01** | `Historical_Mean` | Benchmark | None | - | Không Tuning (Benchmark) |  |  |
| 2 | **Lan01** | `Ridge` | Linear Model | Grid Search | 6 alphas | Đã hoàn tất Tuning (Grid) | `{'alpha': 100}` | `0.012054` |
| 3 | **Lan01** | `Logistic_Regression` | Classification | Grid Search | 12 combos | Đã hoàn tất Tuning (Grid) | `{'C': 0.001, 'class_weight': 'balanced'}` | `0.513765` |
| 4 | **Lan01** | `Elastic_Net` | Linear Model | Grid Search | 30 combos | Đã hoàn tất Tuning (Grid) | `{'alpha': 0.001, 'l1_ratio': 0.5}` | `0.012006` |
| 5 | **Lan01** | `ARIMA` | Statistical TS | Grid Search | 3 orders | Đã hoàn tất Tuning (AIC) | `{'order': (1, 0, 1)}` | `-45970.15` |
| 6 | **Lan01** | `ARIMAX` | Statistical TS | Fixed Order | Fixed | Đã hoàn tất Tuning (Fixed) | `{'order': (1, 0, 1)}` | `-45979.01` |
| 7 | **Lan01** | `SARIMA` | Statistical TS | Fixed Order | Fixed | Đã hoàn tất Tuning (Fixed) | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` | `-45969.88` |
| 8 | **Lan01** | `SVR` | Kernel Method | Grid Search | 48 combos | Đã hoàn tất Tuning (Grid) | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` | `0.012081` |
| 9 | **Lan01** | `Random_Forest` | Tree Ensemble | Optuna TPE | 30 trials | Đã hoàn tất Tuning (Optuna) | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` | `0.008629` |
| 10 | **Lan01** | `XGBoost` | Tree Ensemble | Optuna TPE | 50 trials | Đã hoàn tất Tuning (Optuna) | `{'n_estimators': 200, 'learning_rate': 0.09346777377493276, 'max_depth': 9, 'min_child_weight': 1, 'subsample': 0.7219192643841313, 'colsample_bytree': 0.6558289950203557, 'gamma': 0.0015032555243577606, 'reg_alpha': 0.14113536657860465, 'reg_lambda': 2.879747417492425}` | `0.008624` |
| 11 | **Lan01** | `LightGBM` | Tree Ensemble | Optuna TPE | 50 trials | Đã hoàn tất Tuning (Optuna) | `{'n_estimators': 500, 'learning_rate': 0.017709597800863956, 'num_leaves': 56, 'max_depth': 2, 'min_child_samples': 22, 'subsample': 0.8439440638588244, 'colsample_bytree': 0.962726780479557, 'reg_alpha': 0.03669978237679096, 'reg_lambda': 3.5016679319694397}` | `0.008612` |
| 12 | **Lan01** | `RNN` | Deep Learning | Optuna TPE | 30 trials | Đã hoàn tất Tuning (Optuna) | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3017249991009523, 'learning_rate': 0.000305485208547517, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` | `0.008670` |
| 13 | **Lan01** | `LSTM` | Deep Learning | Optuna TPE | 30 trials | Đã hoàn tất Tuning (Optuna) | `{'lookback': 20, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3538584597651602, 'learning_rate': 0.0007662150068129557, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` | `0.008680` |
| 14 | **Lan01** | `Transformer` | Deep Learning | Optuna TPE | 30 trials | Đã hoàn tất Tuning (Optuna) | `{'lookback': 60, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 1, 'dim_feedforward': 128, 'dropout': 0.2833143475684623, 'learning_rate': 0.00020235315003022192, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` | `0.009152` |
| 15 | **Lan02** | `TCN` | Advanced DL (Temporal CNN) | None | - | Không Tuning (Mặc định) |  |  |
| 16 | **Lan02** | `N-BEATS` | Advanced DL (Basis Expansion) | None | - | Không Tuning (Mặc định) |  |  |
| 17 | **Lan02** | `N-HiTS` | Advanced DL (Hierarchical MLP) | None | - | Không Tuning (Mặc định) |  |  |
| 18 | **Lan02** | `PatchTST` | Advanced DL (Patch Transformer) | None | - | Không Tuning (Mặc định) |  |  |
| 19 | **Lan02** | `TSMixer` | Advanced DL (All-MLP) | None | - | Không Tuning (Mặc định) |  |  |
| 20 | **Lan02** | `iTransformer` | Advanced DL (Inverted Attn) | None | - | Không Tuning (Mặc định) |  |  |
| 21 | **Lan02** | `CNN` | Advanced DL (1D Conv Baseline) | None | - | Không Tuning (Mặc định) |  |  |
| 22 | **Lan03** | `TCN` | Advanced DL (Temporal CNN) | Optuna TPE | 30 trials | Chưa Tuning (Đang chờ chạy) |  |  |
| 23 | **Lan03** | `N-BEATS` | Advanced DL (Basis Expansion) | Optuna TPE | 30 trials | Chưa Tuning (Đang chờ chạy) |  |  |
| 24 | **Lan03** | `N-HiTS` | Advanced DL (Hierarchical MLP) | Optuna TPE | 30 trials | Chưa Tuning (Đang chờ chạy) |  |  |
| 25 | **Lan03** | `PatchTST` | Advanced DL (Patch Transformer) | Optuna TPE | 30 trials | Chưa Tuning (Đang chờ chạy) |  |  |
| 26 | **Lan03** | `TSMixer` | Advanced DL (All-MLP) | Optuna TPE | 30 trials | Chưa Tuning (Đang chờ chạy) |  |  |
| 27 | **Lan03** | `iTransformer` | Advanced DL (Inverted Attn) | Optuna TPE | 30 trials | Chưa Tuning (Đang chờ chạy) |  |  |
| 28 | **Lan03** | `CNN` | Advanced DL (1D Conv Baseline) | Optuna TPE | 30 trials | Chưa Tuning (Đang chờ chạy) |  |  |
| 29 | **Lan04** | `Ensemble_A1` | Ensemble (All 13 Models) | None | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 30 | **Lan04** | `Ensemble_A2` | Ensemble (Core 4 Models) | None | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 31 | **Lan04** | `Ensemble_A3` | Ensemble (Ridge, SARIMA, HM) | None | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 32 | **Lan04** | `Baseline_B1` | Benchmark Selection (Best Single Model) | Deterministic ArgMin | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 33 | **Lan04** | `Baseline_B2` | Weighting Strategy (Inverse-MSE) | Deterministic Analytical | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 34 | **Lan04** | `Baseline_B3` | Optimization (NNLS Simplex) | SciPy SLSQP | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 35 | **Lan04** | `Baseline_C1` | Regime Strategy (Best Model) | None | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 36 | **Lan04** | `Baseline_C2` | Regime Strategy (Shrunk Inv-MSE) | None | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 37 | **Lan04** | `Baseline_C3` | Regime Strategy (Defensive Switch) | None | - | Đã đánh giá mặc định (25/25 fold) |  |  |
| 38 | **Lan05** | `Ensemble_A1` | Ensemble (All 13 Models) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Historical_Mean': 0.043862, 'Ridge': 0.018383, 'Elastic_Net': 0.112595, 'ARIMA': 0.077588, 'ARIMAX': 0.065237, 'SARIMA': 0.122116, 'LightGBM': 0.108744, 'SVR': 0.004925, 'Random_Forest': 0.086254, 'XGBoost': 0.144366, 'RNN': 0.003685, 'LSTM': 0.12604, 'Transformer': 0.086206}}` | `0.008688` |
| 39 | **Lan05** | `Ensemble_A2` | Ensemble (Core 4 Models) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.010263, 'SVR': 0.003301, 'LightGBM': 0.210192, 'SARIMA': 0.776244}}` | `0.008698` |
| 40 | **Lan05** | `Ensemble_A3` | Ensemble (Ridge, SARIMA, HM) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.022671, 'SARIMA': 0.447036, 'Historical_Mean': 0.530294}}` | `0.008694` |
| 41 | **Lan05** | `Baseline_B1` | Benchmark Selection (Best Single Model) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'ARIMA', 'Historical_Mean'], 'blended_weights': {'Elastic_Net': 0.661789, 'ARIMA': 0.336706, 'Historical_Mean': 0.001505}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 7.554e-05, 'ARIMA': 7.563e-05, 'Historical_Mean': 7.566e-05, 'XGBoost': 7.566e-05, 'SARIMA': 7.569e-05}}` | `0.00869` |
| 42 | **Lan05** | `Baseline_B2` | Weighting Strategy (Inverse-MSE) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.249016, 'SVR': 0.23499, 'LightGBM': 0.254671, 'SARIMA': 0.261322}, 'mses': {'Ridge': 7.643e-05, 'SVR': 7.732e-05, 'LightGBM': 7.608e-05, 'SARIMA': 7.569e-05}}` | `0.00872` |
| 43 | **Lan05** | `Baseline_B3` | Optimization (NNLS Simplex) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.010263, 'SVR': 0.003301, 'LightGBM': 0.210192, 'SARIMA': 0.776244}, 'optuna_trials': 50}` | `0.008698` |
| 44 | **Lan05** | `Baseline_C1` | Regime Strategy (Best Model) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'SARIMA', 'Normal': 'ARIMAX', 'High': 'ARIMA'}, 'optuna_weights': {'ARIMAX': 0.21272, 'SARIMA': 0.305154, 'ARIMA': 0.482127}, 'fallback_model': 'Elastic_Net'}` | `0.008694` |
| 45 | **Lan05** | `Baseline_C2` | Regime Strategy (Shrunk Inv-MSE) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.249462, 'SVR': 0.241028, 'LightGBM': 0.252807, 'SARIMA': 0.256703}, 'lambda_shrinkage': {'Low': 0.9892, 'Normal': 0.9711, 'High': 0.9762}, 'final_regime_weights': {'Low': {'Ridge': 0.241976, 'SVR': 0.239936, 'LightGBM': 0.257267, 'SARIMA': 0.260822}, 'Normal': {'Ridge': 0.255126, 'SVR': 0.246888, 'LightGBM': 0.253119, 'SARIMA': 0.244867}, 'High': {'Ridge': 0.25109, 'SVR': 0.23948, 'LightGBM': 0.250525, 'SARIMA': 0.258904}}}` | `0.00872` |
| 46 | **Lan05** | `Baseline_C3` | Regime Strategy (Defensive Switch) | Optuna TPE | 50 trials/fold có lịch sử | Đã hoàn tất (25/25 fold; fold 1 khởi tạo) | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.609598013624639, 'low_normal_core_weights': {'Ridge': 0.038403, 'SVR': 0.005898, 'LightGBM': 0.40204, 'SARIMA': 0.55366}}` | `0.008693` |

### 1.1. KẾT QUẢ HUẤN LUYỆN 18 TRƯỜNG HỢP — LAN04 & LAN05

Mỗi đợt đã có đủ 45 dòng summary, 225 dòng fold metrics và 225 dòng nhật ký tham số. Chỉ số dưới đây là trung bình qua 5 fold; DA hiển thị theo %. Δ R² = Lan05 − Lan04, số dương là cải thiện. Đây là kết quả walk-forward OOS; mean/std và kết quả từng fold có trong CSV nguồn.

Lan05 chọn tham số từ lịch sử OOS của các fold trước. Fold 1 chưa có lịch sử nên dùng prior/fallback; điểm 0.0 trong nhật ký khởi tạo không phải RMSE validation bằng 0. Tuning không bảo đảm cải thiện hiệu năng OOS.

| Mô hình | Chân trời | RMSE Lan04 | R² Lan04 | DA Lan04 | RMSE Lan05 | R² Lan05 | DA Lan05 | Δ R² |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Ensemble_A1` | `1D` | 0.010882 | 0.003477 | 51.76% | 0.010896 | 0.000547 | 53.75% | -0.002929 |
| `Ensemble_A1` | `5D` | 0.022376 | -0.016828 | 55.07% | 0.022475 | -0.037798 | 50.96% | -0.020970 |
| `Ensemble_A1` | `20D` | 0.043700 | -0.044275 | 59.78% | 0.043804 | -0.051762 | 58.38% | -0.007487 |
| `Ensemble_A1` | `60D` | 0.068722 | -0.150749 | 67.43% | 0.069553 | -0.209882 | 67.21% | -0.059133 |
| `Ensemble_A1` | `120D` | 0.078402 | -0.110023 | 70.51% | 0.080181 | -0.140967 | 73.16% | -0.030944 |
| `Ensemble_A2` | `1D` | 0.010933 | -0.004953 | 50.59% | 0.010890 | 0.000950 | 52.06% | +0.005903 |
| `Ensemble_A2` | `5D` | 0.022630 | -0.060279 | 51.99% | 0.022972 | -0.106449 | 49.04% | -0.046169 |
| `Ensemble_A2` | `20D` | 0.044329 | -0.067158 | 57.57% | 0.045075 | -0.111862 | 51.40% | -0.044704 |
| `Ensemble_A2` | `60D` | 0.071035 | -0.187521 | 62.28% | 0.072762 | -0.256583 | 60.44% | -0.069062 |
| `Ensemble_A2` | `120D` | 0.083129 | -0.243642 | 62.87% | 0.083280 | -0.243946 | 69.78% | -0.000304 |
| `Ensemble_A3` | `1D` | 0.010866 | 0.003794 | 52.72% | 0.010878 | 0.001535 | 54.19% | -0.002259 |
| `Ensemble_A3` | `5D` | 0.022528 | -0.032272 | 53.97% | 0.022428 | -0.010880 | 54.56% | +0.021391 |
| `Ensemble_A3` | `20D` | 0.043712 | -0.025121 | 62.35% | 0.043053 | -0.005534 | 61.32% | +0.019588 |
| `Ensemble_A3` | `60D` | 0.068728 | -0.105666 | 66.99% | 0.066043 | -0.056013 | 72.94% | +0.049653 |
| `Ensemble_A3` | `120D` | 0.077336 | -0.015495 | 76.84% | 0.077755 | -0.056608 | 77.79% | -0.041114 |
| `Baseline_B1` | `1D` | 0.010918 | -0.007223 | 54.04% | 0.010892 | -0.001576 | 53.46% | +0.005648 |
| `Baseline_B1` | `5D` | 0.022255 | 0.000401 | 58.75% | 0.022408 | -0.018718 | 54.63% | -0.019119 |
| `Baseline_B1` | `20D` | 0.044747 | -0.236997 | 56.10% | 0.044240 | -0.166166 | 56.25% | +0.070831 |
| `Baseline_B1` | `60D` | 0.067066 | -0.179645 | 65.88% | 0.067917 | -0.221705 | 64.26% | -0.042059 |
| `Baseline_B1` | `120D` | 0.066791 | 0.108285 | 75.74% | 0.070053 | 0.082539 | 75.74% | -0.025746 |
| `Baseline_B2` | `1D` | 0.010932 | -0.004881 | 50.74% | 0.010931 | -0.004605 | 50.74% | +0.000276 |
| `Baseline_B2` | `5D` | 0.022631 | -0.062166 | 52.06% | 0.022680 | -0.073178 | 51.84% | -0.011012 |
| `Baseline_B2` | `20D` | 0.044382 | -0.070727 | 56.54% | 0.044555 | -0.081281 | 55.74% | -0.010554 |
| `Baseline_B2` | `60D` | 0.071268 | -0.194903 | 63.01% | 0.072225 | -0.233312 | 61.91% | -0.038409 |
| `Baseline_B2` | `120D` | 0.082549 | -0.223008 | 65.66% | 0.083408 | -0.263503 | 70.81% | -0.040495 |
| `Baseline_B3` | `1D` | 0.010918 | -0.002209 | 52.28% | 0.010890 | 0.000950 | 52.06% | +0.003159 |
| `Baseline_B3` | `5D` | 0.023223 | -0.174208 | 49.85% | 0.022972 | -0.106449 | 49.04% | +0.067760 |
| `Baseline_B3` | `20D` | 0.045144 | -0.123382 | 49.04% | 0.045075 | -0.111862 | 51.40% | +0.011520 |
| `Baseline_B3` | `60D` | 0.073352 | -0.292686 | 58.38% | 0.072762 | -0.256583 | 60.44% | +0.036103 |
| `Baseline_B3` | `120D` | 0.084384 | -0.307486 | 69.26% | 0.083280 | -0.243946 | 69.78% | +0.063540 |
| `Baseline_C1` | `1D` | 0.010937 | -0.028851 | 53.24% | 0.010937 | -0.028851 | 53.24% | 0.000000 |
| `Baseline_C1` | `5D` | 0.022581 | -0.067342 | 52.87% | 0.022581 | -0.067342 | 52.87% | 0.000000 |
| `Baseline_C1` | `20D` | 0.045220 | -0.237274 | 57.13% | 0.045220 | -0.237274 | 57.13% | 0.000000 |
| `Baseline_C1` | `60D` | 0.067314 | -0.109854 | 66.76% | 0.067314 | -0.109854 | 66.76% | 0.000000 |
| `Baseline_C1` | `120D` | 0.068638 | 0.026059 | 74.85% | 0.068638 | 0.026059 | 74.85% | 0.000000 |
| `Baseline_C2` | `1D` | 0.010932 | -0.004860 | 50.74% | 0.010931 | -0.004667 | 50.59% | +0.000193 |
| `Baseline_C2` | `5D` | 0.022644 | -0.061618 | 51.54% | 0.022696 | -0.065660 | 51.84% | -0.004042 |
| `Baseline_C2` | `20D` | 0.044394 | -0.069991 | 56.62% | 0.044523 | -0.076279 | 57.06% | -0.006288 |
| `Baseline_C2` | `60D` | 0.071351 | -0.194969 | 62.50% | 0.071839 | -0.209738 | 62.06% | -0.014769 |
| `Baseline_C2` | `120D` | 0.082756 | -0.225685 | 65.88% | 0.083963 | -0.273163 | 68.60% | -0.047478 |
| `Baseline_C3` | `1D` | 0.010901 | -0.000890 | 53.01% | 0.010896 | 0.000186 | 53.60% | +0.001075 |
| `Baseline_C3` | `5D` | 0.022432 | -0.042680 | 55.81% | 0.022393 | -0.024810 | 53.53% | +0.017870 |
| `Baseline_C3` | `20D` | 0.043029 | -0.019616 | 66.99% | 0.043361 | -0.058447 | 62.21% | -0.038831 |
| `Baseline_C3` | `60D` | 0.065538 | -0.017908 | 73.75% | 0.065452 | -0.016750 | 74.71% | +0.001158 |
| `Baseline_C3` | `120D` | 0.075776 | 0.022286 | 73.97% | 0.075426 | 0.027746 | 74.56% | +0.005460 |

Lan05 cải thiện OOS R² ở **17/45** cặp mô hình–chân trời. `Baseline_C1` có hiệu năng bằng Lan04 ở cả 5 chân trời theo summary.

- **Lan04:** [summary](../results/Lan04/summary/3.0.3_model_horizon_summary.csv), [fold metrics](../results/Lan04/fold_metrics/3.0.3_all_fold_metrics.csv), [tham số](../results/Lan04/tuning/3.0.3_best_hyperparameters.csv), [OOS predictions](../results/Lan04/predictions/3.0.3_all_oos_predictions.csv), [execution log](../results/Lan04/lan04_execution.log).
- **Lan05:** [summary](../results/Lan05/summary/3.0.3_model_horizon_summary.csv), [fold metrics](../results/Lan05/fold_metrics/3.0.3_all_fold_metrics.csv), [tham số](../results/Lan05/tuning/3.0.3_best_hyperparameters.csv), [OOS predictions](../results/Lan05/predictions/3.0.3_all_oos_predictions.csv), [execution log](../results/Lan05/lan05_execution.log).


---

## 2. CHI TIẾT TỪNG CHÂN TRỜI & FOLD CHO 13 MÔ HÌNH ĐÃ TUNING (LẦN 01 — LAN01)

Ở Lần 01, toàn bộ 13 mô hình (ngoại trừ Historical_Mean) đã hoàn thành 100% quá trình tinh chỉnh tham số ngoại mẫu (Zero-Leakage Walk-Forward CV). Dưới đây là bảng tra cứu chi tiết:

### 🔹 Mô Hình: `Ridge` (Grid Search / TimeSeriesSplit — Thước đo: `CV_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.012512` | `{'alpha': 100}` |
| `1D` | Fold 2 | `0.012466` | `{'alpha': 100}` |
| `1D` | Fold 3 | `0.012257` | `{'alpha': 100}` |
| `1D` | Fold 4 | `0.012093` | `{'alpha': 100}` |
| `1D` | Fold 5 | `0.012054` | `{'alpha': 100}` |
| `5D` | Fold 1 | `0.025658` | `{'alpha': 100}` |
| `5D` | Fold 2 | `0.025528` | `{'alpha': 100}` |
| `5D` | Fold 3 | `0.025088` | `{'alpha': 100}` |
| `5D` | Fold 4 | `0.024705` | `{'alpha': 100}` |
| `5D` | Fold 5 | `0.024742` | `{'alpha': 100}` |
| `20D` | Fold 1 | `0.048044` | `{'alpha': 100}` |
| `20D` | Fold 2 | `0.047581` | `{'alpha': 100}` |
| `20D` | Fold 3 | `0.046810` | `{'alpha': 100}` |
| `20D` | Fold 4 | `0.045928` | `{'alpha': 100}` |
| `20D` | Fold 5 | `0.046098` | `{'alpha': 100}` |
| `60D` | Fold 1 | `0.082740` | `{'alpha': 100}` |
| `60D` | Fold 2 | `0.080978` | `{'alpha': 100}` |
| `60D` | Fold 3 | `0.079019` | `{'alpha': 100}` |
| `60D` | Fold 4 | `0.078521` | `{'alpha': 100}` |
| `60D` | Fold 5 | `0.080125` | `{'alpha': 100}` |
| `120D` | Fold 1 | `0.136700` | `{'alpha': 100}` |
| `120D` | Fold 2 | `0.128370` | `{'alpha': 100}` |
| `120D` | Fold 3 | `0.125450` | `{'alpha': 100}` |
| `120D` | Fold 4 | `0.123198` | `{'alpha': 100}` |
| `120D` | Fold 5 | `0.125941` | `{'alpha': 100}` |

### 🔹 Mô Hình: `Logistic_Regression` (Grid Search / TimeSeriesSplit — Thước đo: `CV_ROC_AUC`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.512674` | `{'C': 0.001, 'class_weight': None}` |
| `1D` | Fold 2 | `0.511822` | `{'C': 100, 'class_weight': None}` |
| `1D` | Fold 3 | `0.508415` | `{'C': 0.001, 'class_weight': 'balanced'}` |
| `1D` | Fold 4 | `0.513490` | `{'C': 0.001, 'class_weight': 'balanced'}` |
| `1D` | Fold 5 | `0.513765` | `{'C': 0.001, 'class_weight': 'balanced'}` |
| `5D` | Fold 1 | `0.517275` | `{'C': 1, 'class_weight': None}` |
| `5D` | Fold 2 | `0.511467` | `{'C': 0.1, 'class_weight': None}` |
| `5D` | Fold 3 | `0.506517` | `{'C': 0.1, 'class_weight': None}` |
| `5D` | Fold 4 | `0.507858` | `{'C': 0.001, 'class_weight': None}` |
| `5D` | Fold 5 | `0.508239` | `{'C': 0.001, 'class_weight': 'balanced'}` |
| `20D` | Fold 1 | `0.514533` | `{'C': 100, 'class_weight': None}` |
| `20D` | Fold 2 | `0.512207` | `{'C': 100, 'class_weight': None}` |
| `20D` | Fold 3 | `0.519422` | `{'C': 100, 'class_weight': None}` |
| `20D` | Fold 4 | `0.518032` | `{'C': 100, 'class_weight': None}` |
| `20D` | Fold 5 | `0.511635` | `{'C': 100, 'class_weight': None}` |
| `60D` | Fold 1 | `0.476098` | `{'C': 0.001, 'class_weight': None}` |
| `60D` | Fold 2 | `0.468297` | `{'C': 100, 'class_weight': None}` |
| `60D` | Fold 3 | `0.494456` | `{'C': 100, 'class_weight': None}` |
| `60D` | Fold 4 | `0.496201` | `{'C': 100, 'class_weight': None}` |
| `60D` | Fold 5 | `0.480035` | `{'C': 100, 'class_weight': None}` |
| `120D` | Fold 1 | `0.509926` | `{'C': 0.001, 'class_weight': None}` |
| `120D` | Fold 2 | `0.496398` | `{'C': 0.001, 'class_weight': None}` |
| `120D` | Fold 3 | `0.522890` | `{'C': 0.001, 'class_weight': None}` |
| `120D` | Fold 4 | `0.549946` | `{'C': 0.001, 'class_weight': None}` |
| `120D` | Fold 5 | `0.500244` | `{'C': 0.001, 'class_weight': None}` |

### 🔹 Mô Hình: `LightGBM` (Optuna TPE — Thước đo: `Internal_Val_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.008778` | `{'n_estimators': 400, 'learning_rate': 0.11049890000613266, 'num_leaves': 49, 'max_depth': 10, 'min_child_samples': 34, 'subsample': 0.7840602059905246, 'colsample_bytree': 0.7428464775168512, 'reg_alpha': 0.44766655633145197, 'reg_lambda': 1.1488808987955952}` |
| `1D` | Fold 2 | `0.008290` | `{'n_estimators': 400, 'learning_rate': 0.165733299290735, 'num_leaves': 30, 'max_depth': 9, 'min_child_samples': 5, 'subsample': 0.6845692314699523, 'colsample_bytree': 0.688110778739281, 'reg_alpha': 0.4965633317958376, 'reg_lambda': 0.20238896974979587}` |
| `1D` | Fold 3 | `0.007695` | `{'n_estimators': 600, 'learning_rate': 0.1969736343909609, 'num_leaves': 72, 'max_depth': 6, 'min_child_samples': 27, 'subsample': 0.6121469995581234, 'colsample_bytree': 0.7974379566652847, 'reg_alpha': 0.4378329885291274, 'reg_lambda': 0.1387268048826412}` |
| `1D` | Fold 4 | `0.007911` | `{'n_estimators': 700, 'learning_rate': 0.010438976266606979, 'num_leaves': 26, 'max_depth': 6, 'min_child_samples': 19, 'subsample': 0.9108722672825401, 'colsample_bytree': 0.8346201924133981, 'reg_alpha': 0.49362649027987576, 'reg_lambda': 0.07709027767066044}` |
| `1D` | Fold 5 | `0.008612` | `{'n_estimators': 500, 'learning_rate': 0.017709597800863956, 'num_leaves': 56, 'max_depth': 2, 'min_child_samples': 22, 'subsample': 0.8439440638588244, 'colsample_bytree': 0.962726780479557, 'reg_alpha': 0.03669978237679096, 'reg_lambda': 3.5016679319694397}` |
| `5D` | Fold 1 | `0.018366` | `{'n_estimators': 400, 'learning_rate': 0.010662261314439397, 'num_leaves': 23, 'max_depth': 4, 'min_child_samples': 29, 'subsample': 0.8361720804787783, 'colsample_bytree': 0.9998584075938027, 'reg_alpha': 0.005020113928178385, 'reg_lambda': 0.8564809986032604}` |
| `5D` | Fold 2 | `0.017634` | `{'n_estimators': 700, 'learning_rate': 0.01891151745738583, 'num_leaves': 38, 'max_depth': 2, 'min_child_samples': 25, 'subsample': 0.8767269828137504, 'colsample_bytree': 0.8496972384669649, 'reg_alpha': 2.3994701946377655, 'reg_lambda': 1.8252664137017665}` |
| `5D` | Fold 3 | `0.016532` | `{'n_estimators': 700, 'learning_rate': 0.014453948235254758, 'num_leaves': 11, 'max_depth': 1, 'min_child_samples': 27, 'subsample': 0.9600310271441853, 'colsample_bytree': 0.7803963996491828, 'reg_alpha': 1.9963290824599744, 'reg_lambda': 42.78787106330723}` |
| `5D` | Fold 4 | `0.016863` | `{'n_estimators': 500, 'learning_rate': 0.02877739429459395, 'num_leaves': 83, 'max_depth': 4, 'min_child_samples': 15, 'subsample': 0.7445966184537816, 'colsample_bytree': 0.881339976681622, 'reg_alpha': 1.200948429008149, 'reg_lambda': 0.0069006108134769435}` |
| `5D` | Fold 5 | `0.018369` | `{'n_estimators': 700, 'learning_rate': 0.028986070520949335, 'num_leaves': 67, 'max_depth': 6, 'min_child_samples': 9, 'subsample': 0.9310084734098819, 'colsample_bytree': 0.9572203681943565, 'reg_alpha': 3.2553314288692103, 'reg_lambda': 1.7758898941908814}` |
| `20D` | Fold 1 | `0.029328` | `{'n_estimators': 400, 'learning_rate': 0.01214955896190583, 'num_leaves': 65, 'max_depth': 4, 'min_child_samples': 18, 'subsample': 0.8242531495618246, 'colsample_bytree': 0.8810419894379805, 'reg_alpha': 1.4639356450546328, 'reg_lambda': 0.0063168317738846085}` |
| `20D` | Fold 2 | `0.030998` | `{'n_estimators': 600, 'learning_rate': 0.06028678008705474, 'num_leaves': 8, 'max_depth': 7, 'min_child_samples': 10, 'subsample': 0.6892516865568443, 'colsample_bytree': 0.9202322459304487, 'reg_alpha': 7.6026855684463985, 'reg_lambda': 18.837447804130463}` |
| `20D` | Fold 3 | `0.029056` | `{'n_estimators': 600, 'learning_rate': 0.06028678008705474, 'num_leaves': 8, 'max_depth': 7, 'min_child_samples': 10, 'subsample': 0.6892516865568443, 'colsample_bytree': 0.9202322459304487, 'reg_alpha': 7.6026855684463985, 'reg_lambda': 18.837447804130463}` |
| `20D` | Fold 4 | `0.029018` | `{'n_estimators': 600, 'learning_rate': 0.06028678008705474, 'num_leaves': 8, 'max_depth': 7, 'min_child_samples': 10, 'subsample': 0.6892516865568443, 'colsample_bytree': 0.9202322459304487, 'reg_alpha': 7.6026855684463985, 'reg_lambda': 18.837447804130463}` |
| `20D` | Fold 5 | `0.033074` | `{'n_estimators': 900, 'learning_rate': 0.10177869850495169, 'num_leaves': 32, 'max_depth': 7, 'min_child_samples': 35, 'subsample': 0.7579133537718414, 'colsample_bytree': 0.9060140859477831, 'reg_alpha': 8.126418990015443, 'reg_lambda': 1.7706866414426934}` |
| `60D` | Fold 1 | `0.048711` | `{'n_estimators': 100, 'learning_rate': 0.011447637234827451, 'num_leaves': 18, 'max_depth': 11, 'min_child_samples': 15, 'subsample': 0.6412331914319542, 'colsample_bytree': 0.9701531023238978, 'reg_alpha': 0.19466938677429282, 'reg_lambda': 1.3857077573471899}` |
| `60D` | Fold 2 | `0.050691` | `{'n_estimators': 400, 'learning_rate': 0.03942388855847337, 'num_leaves': 20, 'max_depth': 1, 'min_child_samples': 12, 'subsample': 0.6313082533923459, 'colsample_bytree': 0.6185295031183613, 'reg_alpha': 9.235247253786006e-08, 'reg_lambda': 9.180886253913398}` |
| `60D` | Fold 3 | `0.048620` | `{'n_estimators': 700, 'learning_rate': 0.0920005366021435, 'num_leaves': 30, 'max_depth': 8, 'min_child_samples': 10, 'subsample': 0.6286841848378677, 'colsample_bytree': 0.9751650458273082, 'reg_alpha': 9.728822668045288, 'reg_lambda': 12.453069977180352}` |
| `60D` | Fold 4 | `0.048954` | `{'n_estimators': 800, 'learning_rate': 0.03792111591542901, 'num_leaves': 13, 'max_depth': 2, 'min_child_samples': 8, 'subsample': 0.6566363569879685, 'colsample_bytree': 0.9381751464075414, 'reg_alpha': 8.056272616412961, 'reg_lambda': 3.915869667236098}` |
| `60D` | Fold 5 | `0.063051` | `{'n_estimators': 700, 'learning_rate': 0.041743971025763715, 'num_leaves': 12, 'max_depth': 5, 'min_child_samples': 20, 'subsample': 0.6017148426091806, 'colsample_bytree': 0.7586408931727296, 'reg_alpha': 9.441872300725256, 'reg_lambda': 0.7468611502423925}` |
| `120D` | Fold 1 | `0.062615` | `{'n_estimators': 100, 'learning_rate': 0.020835711470139073, 'num_leaves': 20, 'max_depth': 12, 'min_child_samples': 9, 'subsample': 0.6152743051323472, 'colsample_bytree': 0.9620574151865015, 'reg_alpha': 0.05910391124433443, 'reg_lambda': 68.1202138216823}` |
| `120D` | Fold 2 | `0.061196` | `{'n_estimators': 600, 'learning_rate': 0.055729791256839245, 'num_leaves': 20, 'max_depth': 1, 'min_child_samples': 7, 'subsample': 0.6126230169411723, 'colsample_bytree': 0.6882584356649754, 'reg_alpha': 2.6811421426403608e-08, 'reg_lambda': 10.575715041714826}` |
| `120D` | Fold 3 | `0.065023` | `{'n_estimators': 600, 'learning_rate': 0.056608710147619944, 'num_leaves': 7, 'max_depth': 1, 'min_child_samples': 5, 'subsample': 0.6628601831216637, 'colsample_bytree': 0.6971592734655775, 'reg_alpha': 3.328753294660225e-08, 'reg_lambda': 28.371002381003297}` |
| `120D` | Fold 4 | `0.069924` | `{'n_estimators': 600, 'learning_rate': 0.15826541904647565, 'num_leaves': 17, 'max_depth': 1, 'min_child_samples': 9, 'subsample': 0.7301321323053057, 'colsample_bytree': 0.7554709158757928, 'reg_alpha': 2.7678419414850017e-06, 'reg_lambda': 13.921548533046511}` |
| `120D` | Fold 5 | `0.093769` | `{'n_estimators': 600, 'learning_rate': 0.15826541904647565, 'num_leaves': 17, 'max_depth': 1, 'min_child_samples': 9, 'subsample': 0.7301321323053057, 'colsample_bytree': 0.7554709158757928, 'reg_alpha': 2.7678419414850017e-06, 'reg_lambda': 13.921548533046511}` |

### 🔹 Mô Hình: `SVR` (Grid Search / TimeSeriesSplit — Thước đo: `CV_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.012524` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `1D` | Fold 2 | `0.012480` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `1D` | Fold 3 | `0.012273` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `1D` | Fold 4 | `0.012099` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `1D` | Fold 5 | `0.012081` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `5D` | Fold 1 | `0.026566` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `5D` | Fold 2 | `0.026414` | `{'C': 0.1, 'epsilon': 0.05, 'gamma': 0.1, 'kernel': 'rbf'}` |
| `5D` | Fold 3 | `0.025924` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `5D` | Fold 4 | `0.025540` | `{'C': 0.1, 'epsilon': 0.05, 'gamma': 0.1, 'kernel': 'rbf'}` |
| `5D` | Fold 5 | `0.025525` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `20D` | Fold 1 | `0.049667` | `{'C': 0.1, 'epsilon': 0.05, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `20D` | Fold 2 | `0.049155` | `{'C': 0.1, 'epsilon': 0.05, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `20D` | Fold 3 | `0.048325` | `{'C': 0.1, 'epsilon': 0.05, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `20D` | Fold 4 | `0.047516` | `{'C': 0.1, 'epsilon': 0.05, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `20D` | Fold 5 | `0.047779` | `{'C': 0.1, 'epsilon': 0.01, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `60D` | Fold 1 | `0.084939` | `{'C': 0.1, 'epsilon': 0.05, 'gamma': 0.1, 'kernel': 'rbf'}` |
| `60D` | Fold 2 | `0.083400` | `{'C': 0.1, 'epsilon': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `60D` | Fold 3 | `0.081080` | `{'C': 0.1, 'epsilon': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `60D` | Fold 4 | `0.080024` | `{'C': 0.1, 'epsilon': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `60D` | Fold 5 | `0.080300` | `{'C': 0.1, 'epsilon': 0.001, 'gamma': 0.001, 'kernel': 'rbf'}` |
| `120D` | Fold 1 | `0.123376` | `{'C': 0.1, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}` |
| `120D` | Fold 2 | `0.121015` | `{'C': 0.1, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}` |
| `120D` | Fold 3 | `0.120761` | `{'C': 0.1, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}` |
| `120D` | Fold 4 | `0.116060` | `{'C': 0.1, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}` |
| `120D` | Fold 5 | `0.120148` | `{'C': 0.1, 'epsilon': 0.1, 'gamma': 0.1, 'kernel': 'rbf'}` |

### 🔹 Mô Hình: `Random_Forest` (Optuna TPE — Thước đo: `Internal_Val_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.008776` | `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 0.5}` |
| `1D` | Fold 2 | `0.008310` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` |
| `1D` | Fold 3 | `0.007708` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 0.5}` |
| `1D` | Fold 4 | `0.007916` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` |
| `1D` | Fold 5 | `0.008629` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` |
| `5D` | Fold 1 | `0.018290` | `{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'}` |
| `5D` | Fold 2 | `0.017686` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 0.5}` |
| `5D` | Fold 3 | `0.016561` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 0.5}` |
| `5D` | Fold 4 | `0.016868` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 0.5}` |
| `5D` | Fold 5 | `0.018323` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 0.5}` |
| `20D` | Fold 1 | `0.029999` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 0.5}` |
| `20D` | Fold 2 | `0.031575` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` |
| `20D` | Fold 3 | `0.029593` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt'}` |
| `20D` | Fold 4 | `0.029576` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` |
| `20D` | Fold 5 | `0.033369` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt'}` |
| `60D` | Fold 1 | `0.049179` | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_features': 'sqrt'}` |
| `60D` | Fold 2 | `0.051905` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}` |
| `60D` | Fold 3 | `0.046291` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt'}` |
| `60D` | Fold 4 | `0.047338` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt'}` |
| `60D` | Fold 5 | `0.058608` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt'}` |
| `120D` | Fold 1 | `0.061748` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_features': 'sqrt'}` |
| `120D` | Fold 2 | `0.060722` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 'sqrt'}` |
| `120D` | Fold 3 | `0.066237` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 'sqrt'}` |
| `120D` | Fold 4 | `0.065075` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` |
| `120D` | Fold 5 | `0.084580` | `{'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 'sqrt'}` |

### 🔹 Mô Hình: `XGBoost` (Optuna TPE — Thước đo: `Internal_Val_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.008802` | `{'n_estimators': 200, 'learning_rate': 0.16482980845075787, 'max_depth': 9, 'min_child_weight': 2, 'subsample': 0.7299728847064147, 'colsample_bytree': 0.67498834754586, 'gamma': 0.6907781506070934, 'reg_alpha': 0.012006821319579775, 'reg_lambda': 12.464577340414984}` |
| `1D` | Fold 2 | `0.008312` | `{'n_estimators': 200, 'learning_rate': 0.17523144542136695, 'max_depth': 10, 'min_child_weight': 4, 'subsample': 0.7592183523018007, 'colsample_bytree': 0.7085139748885433, 'gamma': 1.656517344950464, 'reg_alpha': 0.07154889881109874, 'reg_lambda': 51.43471326516465}` |
| `1D` | Fold 3 | `0.007712` | `{'n_estimators': 600, 'learning_rate': 0.01894718686730443, 'max_depth': 4, 'min_child_weight': 4, 'subsample': 0.7460007086545342, 'colsample_bytree': 0.8194907321784571, 'gamma': 3.4919271037301725, 'reg_alpha': 0.05800421679197067, 'reg_lambda': 0.2601420495541158}` |
| `1D` | Fold 4 | `0.007930` | `{'n_estimators': 200, 'learning_rate': 0.16782678532623518, 'max_depth': 10, 'min_child_weight': 2, 'subsample': 0.6456429521023989, 'colsample_bytree': 0.6052767590737163, 'gamma': 1.6547883477496947, 'reg_alpha': 0.00923809201010153, 'reg_lambda': 9.274458155705567}` |
| `1D` | Fold 5 | `0.008624` | `{'n_estimators': 200, 'learning_rate': 0.09346777377493276, 'max_depth': 9, 'min_child_weight': 1, 'subsample': 0.7219192643841313, 'colsample_bytree': 0.6558289950203557, 'gamma': 0.0015032555243577606, 'reg_alpha': 0.14113536657860465, 'reg_lambda': 2.879747417492425}` |
| `5D` | Fold 1 | `0.018698` | `{'n_estimators': 500, 'learning_rate': 0.028344287522300348, 'max_depth': 7, 'min_child_weight': 2, 'subsample': 0.7104564182996586, 'colsample_bytree': 0.6709711642982997, 'gamma': 2.1902205055674906, 'reg_alpha': 0.06127605976015246, 'reg_lambda': 0.008997160419110447}` |
| `5D` | Fold 2 | `0.017770` | `{'n_estimators': 500, 'learning_rate': 0.18631744759265406, 'max_depth': 7, 'min_child_weight': 4, 'subsample': 0.7370473031808337, 'colsample_bytree': 0.763209333043121, 'gamma': 3.0558835894405076, 'reg_alpha': 0.0055460853975703855, 'reg_lambda': 0.23369533360494577}` |
| `5D` | Fold 3 | `0.016644` | `{'n_estimators': 500, 'learning_rate': 0.044714059136592404, 'max_depth': 8, 'min_child_weight': 5, 'subsample': 0.6362631858694583, 'colsample_bytree': 0.6200168792436461, 'gamma': 2.2026531547166743, 'reg_alpha': 0.06431622766171038, 'reg_lambda': 0.030432893559130553}` |
| `5D` | Fold 4 | `0.016946` | `{'n_estimators': 100, 'learning_rate': 0.08180523768303569, 'max_depth': 9, 'min_child_weight': 7, 'subsample': 0.6012847095667525, 'colsample_bytree': 0.7081499923198863, 'gamma': 4.941757519175194, 'reg_alpha': 0.0013465026146775535, 'reg_lambda': 0.007604515339664316}` |
| `5D` | Fold 5 | `0.018391` | `{'n_estimators': 500, 'learning_rate': 0.044940141673554874, 'max_depth': 5, 'min_child_weight': 4, 'subsample': 0.6940483727591255, 'colsample_bytree': 0.7802963569765408, 'gamma': 0.038205855949074774, 'reg_alpha': 0.008288295439113044, 'reg_lambda': 0.15621495477525174}` |
| `20D` | Fold 1 | `0.029793` | `{'n_estimators': 400, 'learning_rate': 0.15715176059861033, 'max_depth': 6, 'min_child_weight': 4, 'subsample': 0.6944645257579334, 'colsample_bytree': 0.8316834274806193, 'gamma': 0.0696901095246975, 'reg_alpha': 5.102519242452782e-05, 'reg_lambda': 6.87459055877633}` |
| `20D` | Fold 2 | `0.030976` | `{'n_estimators': 400, 'learning_rate': 0.17254716573280354, 'max_depth': 8, 'min_child_weight': 6, 'subsample': 0.6624074561769746, 'colsample_bytree': 0.662397808134481, 'gamma': 0.2904180608409973, 'reg_alpha': 0.6245760287469893, 'reg_lambda': 1.0129197956845732}` |
| `20D` | Fold 3 | `0.029017` | `{'n_estimators': 400, 'learning_rate': 0.15259144921853043, 'max_depth': 8, 'min_child_weight': 4, 'subsample': 0.6694216580478584, 'colsample_bytree': 0.7238927997467867, 'gamma': 0.13540197046038513, 'reg_alpha': 0.09176443641650511, 'reg_lambda': 2.0471994650509857}` |
| `20D` | Fold 4 | `0.029001` | `{'n_estimators': 400, 'learning_rate': 0.1468508767495767, 'max_depth': 9, 'min_child_weight': 7, 'subsample': 0.618437445799012, 'colsample_bytree': 0.7682131007965295, 'gamma': 3.022916152642125, 'reg_alpha': 8.391317246554213e-06, 'reg_lambda': 22.57092094687673}` |
| `20D` | Fold 5 | `0.032908` | `{'n_estimators': 300, 'learning_rate': 0.19145311972695062, 'max_depth': 9, 'min_child_weight': 4, 'subsample': 0.642220926251284, 'colsample_bytree': 0.6249733157038994, 'gamma': 0.09899401413215997, 'reg_alpha': 0.0001588239743053469, 'reg_lambda': 16.5549592009347}` |
| `60D` | Fold 1 | `0.048786` | `{'n_estimators': 800, 'learning_rate': 0.022540810038385727, 'max_depth': 8, 'min_child_weight': 5, 'subsample': 0.6094829317041391, 'colsample_bytree': 0.8543455381942271, 'gamma': 0.1423205970673974, 'reg_alpha': 0.001292333659266177, 'reg_lambda': 0.0027733304000431974}` |
| `60D` | Fold 2 | `0.045863` | `{'n_estimators': 1000, 'learning_rate': 0.14292633363708662, 'max_depth': 3, 'min_child_weight': 2, 'subsample': 0.6603318011208913, 'colsample_bytree': 0.6587378205373504, 'gamma': 1.8789623380436749, 'reg_alpha': 1.6580076322804724e-07, 'reg_lambda': 0.0017271138670197513}` |
| `60D` | Fold 3 | `0.041288` | `{'n_estimators': 900, 'learning_rate': 0.027277488229260405, 'max_depth': 4, 'min_child_weight': 4, 'subsample': 0.7509113070790233, 'colsample_bytree': 0.9366221073016855, 'gamma': 1.1362948899887748, 'reg_alpha': 1.2495317043749369e-05, 'reg_lambda': 0.3989291906083396}` |
| `60D` | Fold 4 | `0.041579` | `{'n_estimators': 900, 'learning_rate': 0.18503961523646953, 'max_depth': 3, 'min_child_weight': 4, 'subsample': 0.6931045457943362, 'colsample_bytree': 0.7564582363703096, 'gamma': 1.323160289395684, 'reg_alpha': 3.1750194505136243e-07, 'reg_lambda': 32.79808643165651}` |
| `60D` | Fold 5 | `0.050570` | `{'n_estimators': 900, 'learning_rate': 0.09017195441448847, 'max_depth': 3, 'min_child_weight': 6, 'subsample': 0.6630780084659571, 'colsample_bytree': 0.8072261917689363, 'gamma': 4.258905611621719, 'reg_alpha': 0.0469076970012189, 'reg_lambda': 0.27651975491831876}` |
| `120D` | Fold 1 | `0.061261` | `{'n_estimators': 500, 'learning_rate': 0.016007036385129125, 'max_depth': 9, 'min_child_weight': 7, 'subsample': 0.7527609886492506, 'colsample_bytree': 0.9789493472034575, 'gamma': 0.5879430556568598, 'reg_alpha': 0.41234594019696547, 'reg_lambda': 19.53765853018537}` |
| `120D` | Fold 2 | `0.056076` | `{'n_estimators': 200, 'learning_rate': 0.013777908322294695, 'max_depth': 7, 'min_child_weight': 8, 'subsample': 0.9312912015547016, 'colsample_bytree': 0.6134070516929855, 'gamma': 2.7244100752053093, 'reg_alpha': 5.018787839540521e-07, 'reg_lambda': 0.0018687920767718185}` |
| `120D` | Fold 3 | `0.053375` | `{'n_estimators': 1000, 'learning_rate': 0.1676063284982292, 'max_depth': 3, 'min_child_weight': 9, 'subsample': 0.6301886180051994, 'colsample_bytree': 0.7784937067777397, 'gamma': 3.7380722715536923, 'reg_alpha': 0.017760974213777727, 'reg_lambda': 0.005238005902748139}` |
| `120D` | Fold 4 | `0.050294` | `{'n_estimators': 600, 'learning_rate': 0.1643886641403159, 'max_depth': 8, 'min_child_weight': 4, 'subsample': 0.9026679863634834, 'colsample_bytree': 0.7167342905159535, 'gamma': 4.515708653381637, 'reg_alpha': 0.08495816127682672, 'reg_lambda': 0.17019369048371924}` |
| `120D` | Fold 5 | `0.058386` | `{'n_estimators': 500, 'learning_rate': 0.04027448760837812, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.7722240644985903, 'colsample_bytree': 0.9320474141643308, 'gamma': 3.62270958106528, 'reg_alpha': 0.12601284472593474, 'reg_lambda': 0.017065132603293003}` |

### 🔹 Mô Hình: `RNN` (Optuna TPE — Thước đo: `Internal_Val_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.009098` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 2 | `0.008416` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.4202626021016229, 'learning_rate': 0.0006840117725936527, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 3 | `0.007747` | `{'lookback': 10, 'hidden_size': 32, 'num_layers': 1, 'dropout': 0.3765588646777447, 'learning_rate': 0.0003595904554814919, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 4 | `0.007951` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 5 | `0.008670` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3017249991009523, 'learning_rate': 0.000305485208547517, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 1 | `0.018641` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 2 | `0.017616` | `{'lookback': 60, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.40584330029256915, 'learning_rate': 0.00035647284199131167, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 3 | `0.016416` | `{'lookback': 10, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.21504356496939736, 'learning_rate': 0.0008033164262520415, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 4 | `0.016891` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 5 | `0.018311` | `{'lookback': 20, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.2622779137308282, 'learning_rate': 0.000786477284633486, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 1 | `0.031431` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 2 | `0.031480` | `{'lookback': 60, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.4731981861368065, 'learning_rate': 0.0006691549927872443, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 3 | `0.029471` | `{'lookback': 10, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.4341837766925426, 'learning_rate': 0.0003878135088262649, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 4 | `0.028867` | `{'lookback': 10, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.42071945169752456, 'learning_rate': 0.0008241548024289433, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 5 | `0.032777` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 1 | `0.046386` | `{'lookback': 60, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.41732338388357026, 'learning_rate': 0.0008018561438811223, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 2 | `0.049564` | `{'lookback': 40, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.1982550298750246, 'learning_rate': 0.0005905320703537384, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 3 | `0.041860` | `{'lookback': 10, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.4698342584388917, 'learning_rate': 7.07162866259956e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 4 | `0.040772` | `{'lookback': 60, 'hidden_size': 32, 'num_layers': 3, 'dropout': 0.38488174403271147, 'learning_rate': 0.0006389541022107687, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 5 | `0.050872` | `{'lookback': 60, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3747383957604866, 'learning_rate': 0.0001837145924804805, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 1 | `0.072124` | `{'lookback': 20, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.07030322127719271, 'learning_rate': 1.3798608769259343e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 2 | `0.063604` | `{'lookback': 60, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.47292089152908906, 'learning_rate': 0.0005879003019755277, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 3 | `0.060521` | `{'lookback': 20, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.07030322127719271, 'learning_rate': 1.3798608769259343e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 4 | `0.051831` | `{'lookback': 20, 'hidden_size': 32, 'num_layers': 3, 'dropout': 0.4151470041640507, 'learning_rate': 0.00024483133322562205, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 5 | `0.059427` | `{'lookback': 10, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.25029088451059456, 'learning_rate': 0.0001604025443901114, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |

### 🔹 Mô Hình: `LSTM` (Optuna TPE — Thước đo: `Internal_Val_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.008767` | `{'lookback': 40, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.4871079271090035, 'learning_rate': 9.140136883051477e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 2 | `0.008403` | `{'lookback': 40, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.4758452397629151, 'learning_rate': 1.0519095028133849e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 3 | `0.007684` | `{'lookback': 10, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.28352723331727736, 'learning_rate': 5.6962967627719115e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 4 | `0.007955` | `{'lookback': 10, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.4698500673596681, 'learning_rate': 7.075788555862365e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 5 | `0.008680` | `{'lookback': 20, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3538584597651602, 'learning_rate': 0.0007662150068129557, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 1 | `0.018191` | `{'lookback': 20, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.16259166101337352, 'learning_rate': 0.0002878805718308924, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 2 | `0.017653` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.29568020673805173, 'learning_rate': 3.0726061518413326e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 3 | `0.016460` | `{'lookback': 20, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.4633049076651924, 'learning_rate': 0.0002084591201914449, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 4 | `0.016843` | `{'lookback': 60, 'hidden_size': 128, 'num_layers': 1, 'dropout': 0.46988503396576825, 'learning_rate': 0.0004369385579297834, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 5 | `0.018464` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 1 | `0.031748` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3861223846483287, 'learning_rate': 2.497073714505272e-05, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 2 | `0.031124` | `{'lookback': 60, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.4967641276397784, 'learning_rate': 0.0009103215068356808, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 3 | `0.029007` | `{'lookback': 10, 'hidden_size': 32, 'num_layers': 3, 'dropout': 0.22006422423046346, 'learning_rate': 0.00013735354452187095, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 4 | `0.028668` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.49075197409523963, 'learning_rate': 0.0008242110033509204, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 5 | `0.033487` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3861223846483287, 'learning_rate': 2.497073714505272e-05, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 1 | `0.046367` | `{'lookback': 40, 'hidden_size': 32, 'num_layers': 3, 'dropout': 0.424088876607049, 'learning_rate': 1.2759988367099719e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 2 | `0.045775` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.35145911280645464, 'learning_rate': 1.1875435143461654e-05, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 3 | `0.040386` | `{'lookback': 60, 'hidden_size': 32, 'num_layers': 3, 'dropout': 0.3889581090581174, 'learning_rate': 0.0008242519241846763, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 4 | `0.038366` | `{'lookback': 60, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.38968086282287073, 'learning_rate': 0.0005812645577060852, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 5 | `0.051930` | `{'lookback': 10, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.4202685958641559, 'learning_rate': 5.341629700497827e-05, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 1 | `0.057729` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.49728716450229293, 'learning_rate': 1.390025375411674e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 2 | `0.058174` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3861223846483287, 'learning_rate': 2.497073714505272e-05, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 3 | `0.054988` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3861223846483287, 'learning_rate': 2.497073714505272e-05, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 4 | `0.053046` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3037616521132426, 'learning_rate': 6.742465968587722e-05, 'weight_decay': 0.0001, 'batch_size': 64, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 5 | `0.059614` | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.21355615281666593, 'learning_rate': 0.0002883724954669892, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |

### 🔹 Mô Hình: `ARIMA` (Grid Search / TimeSeriesSplit — Thước đo: `AIC`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `-38850.09` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 2 | `-40547.93` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 3 | `-42437.92` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 4 | `-44254.53` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 5 | `-45970.15` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 1 | `-34974.17` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 2 | `-36517.95` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 3 | `-38241.62` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 4 | `-39890.26` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 5 | `-41444.79` | `{'order': (1, 0, 1)}` |
| `20D` | Fold 1 | `-34610.19` | `{'order': (1, 0, 0)}` |
| `20D` | Fold 2 | `-36158.38` | `{'order': (1, 0, 0)}` |
| `20D` | Fold 3 | `-37865.54` | `{'order': (1, 0, 0)}` |
| `20D` | Fold 4 | `-39497.08` | `{'order': (1, 0, 0)}` |
| `20D` | Fold 5 | `-41018.33` | `{'order': (1, 0, 0)}` |
| `60D` | Fold 1 | `-34168.91` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 2 | `-35701.34` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 3 | `-37386.24` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 4 | `-39024.09` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 5 | `-40569.65` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 1 | `-33706.06` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 2 | `-35216.03` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 3 | `-36884.03` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 4 | `-38538.77` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 5 | `-40076.66` | `{'order': (1, 0, 1)}` |

### 🔹 Mô Hình: `ARIMAX` (Grid Search / TimeSeriesSplit — Thước đo: `AIC`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `-38857.24` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 2 | `-40556.89` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 3 | `-42445.58` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 4 | `-44263.35` | `{'order': (1, 0, 1)}` |
| `1D` | Fold 5 | `-45979.01` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 1 | `-37857.80` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 2 | `-39524.67` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 3 | `-41364.99` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 4 | `-43138.73` | `{'order': (1, 0, 1)}` |
| `5D` | Fold 5 | `-44808.96` | `{'order': (1, 0, 1)}` |
| `20D` | Fold 1 | `-37826.90` | `{'order': (1, 0, 1)}` |
| `20D` | Fold 2 | `-39515.44` | `{'order': (1, 0, 1)}` |
| `20D` | Fold 3 | `-41358.14` | `{'order': (1, 0, 1)}` |
| `20D` | Fold 4 | `-43134.25` | `{'order': (1, 0, 1)}` |
| `20D` | Fold 5 | `-44803.89` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 1 | `-37679.02` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 2 | `-39357.86` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 3 | `-41204.40` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 4 | `-42961.62` | `{'order': (1, 0, 1)}` |
| `60D` | Fold 5 | `-44653.86` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 1 | `-37235.55` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 2 | `-38914.48` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 3 | `-40750.88` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 4 | `-42512.97` | `{'order': (1, 0, 1)}` |
| `120D` | Fold 5 | `-44214.99` | `{'order': (1, 0, 1)}` |

### 🔹 Mô Hình: `Elastic_Net` (Grid Search / TimeSeriesSplit — Thước đo: `CV_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.012482` | `{'alpha': 0.001, 'l1_ratio': 0.7}` |
| `1D` | Fold 2 | `0.012440` | `{'alpha': 0.001, 'l1_ratio': 0.5}` |
| `1D` | Fold 3 | `0.012219` | `{'alpha': 0.001, 'l1_ratio': 0.5}` |
| `1D` | Fold 4 | `0.012057` | `{'alpha': 0.001, 'l1_ratio': 0.5}` |
| `1D` | Fold 5 | `0.012006` | `{'alpha': 0.001, 'l1_ratio': 0.5}` |
| `5D` | Fold 1 | `0.025528` | `{'alpha': 0.005, 'l1_ratio': 0.3}` |
| `5D` | Fold 2 | `0.025425` | `{'alpha': 0.001, 'l1_ratio': 0.7}` |
| `5D` | Fold 3 | `0.024948` | `{'alpha': 0.001, 'l1_ratio': 0.7}` |
| `5D` | Fold 4 | `0.024590` | `{'alpha': 0.001, 'l1_ratio': 0.7}` |
| `5D` | Fold 5 | `0.024578` | `{'alpha': 0.001, 'l1_ratio': 0.9}` |
| `20D` | Fold 1 | `0.047960` | `{'alpha': 0.01, 'l1_ratio': 0.7}` |
| `20D` | Fold 2 | `0.047492` | `{'alpha': 0.0005, 'l1_ratio': 0.9}` |
| `20D` | Fold 3 | `0.046668` | `{'alpha': 0.001, 'l1_ratio': 0.7}` |
| `20D` | Fold 4 | `0.045805` | `{'alpha': 0.001, 'l1_ratio': 0.5}` |
| `20D` | Fold 5 | `0.045874` | `{'alpha': 0.005, 'l1_ratio': 0.9}` |
| `60D` | Fold 1 | `0.082260` | `{'alpha': 0.05, 'l1_ratio': 0.3}` |
| `60D` | Fold 2 | `0.080820` | `{'alpha': 0.005, 'l1_ratio': 0.3}` |
| `60D` | Fold 3 | `0.078900` | `{'alpha': 0.005, 'l1_ratio': 0.3}` |
| `60D` | Fold 4 | `0.078101` | `{'alpha': 0.005, 'l1_ratio': 0.3}` |
| `60D` | Fold 5 | `0.077896` | `{'alpha': 0.05, 'l1_ratio': 0.3}` |
| `120D` | Fold 1 | `0.125341` | `{'alpha': 0.05, 'l1_ratio': 0.5}` |
| `120D` | Fold 2 | `0.122534` | `{'alpha': 0.05, 'l1_ratio': 0.5}` |
| `120D` | Fold 3 | `0.120742` | `{'alpha': 0.05, 'l1_ratio': 0.5}` |
| `120D` | Fold 4 | `0.118428` | `{'alpha': 0.05, 'l1_ratio': 0.5}` |
| `120D` | Fold 5 | `0.117128` | `{'alpha': 0.05, 'l1_ratio': 0.5}` |

### 🔹 Mô Hình: `SARIMA` (Grid Search / TimeSeriesSplit — Thước đo: `AIC`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `-38849.61` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `1D` | Fold 2 | `-40545.76` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `1D` | Fold 3 | `-42436.01` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `1D` | Fold 4 | `-44253.40` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `1D` | Fold 5 | `-45969.88` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `5D` | Fold 1 | `-36758.85` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `5D` | Fold 2 | `-38366.29` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `5D` | Fold 3 | `-40160.08` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `5D` | Fold 4 | `-41880.49` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `5D` | Fold 5 | `-43504.61` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `20D` | Fold 1 | `-34610.19` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `20D` | Fold 2 | `-36158.38` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `20D` | Fold 3 | `-37865.54` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `20D` | Fold 4 | `-39497.08` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `20D` | Fold 5 | `-41018.33` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `60D` | Fold 1 | `-34169.65` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `60D` | Fold 2 | `-35702.42` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `60D` | Fold 3 | `-37387.35` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `60D` | Fold 4 | `-39024.53` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `60D` | Fold 5 | `-40570.39` | `{'order': (1, 0, 0), 'seasonal_order': (1, 0, 0, 5)}` |
| `120D` | Fold 1 | `-33691.93` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `120D` | Fold 2 | `-35204.95` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `120D` | Fold 3 | `-36871.88` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `120D` | Fold 4 | `-38526.01` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |
| `120D` | Fold 5 | `-40062.44` | `{'order': (1, 0, 0), 'seasonal_order': (0, 0, 0, 0)}` |

### 🔹 Mô Hình: `Transformer` (Optuna TPE — Thước đo: `Internal_Val_RMSE`)
| Chân Trời | Fold | Điểm Val Tốt Nhất | Bộ Tham Số Tối Ưu Được Chọn Sau Tuning |
| :---: | :---: | :---: | :--- |
| `1D` | Fold 1 | `0.009070` | `{'lookback': 40, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.26355899145378414, 'learning_rate': 0.0004967684390320199, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 2 | `0.010403` | `{'lookback': 40, 'd_model': 64, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.2958225797050406, 'learning_rate': 0.0003530122358365842, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 3 | `0.007890` | `{'lookback': 40, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.2581691086820058, 'learning_rate': 0.00027147127389564896, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 4 | `0.008601` | `{'lookback': 40, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.06336328512633407, 'learning_rate': 0.00019584260682666558, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `1D` | Fold 5 | `0.009152` | `{'lookback': 60, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 1, 'dim_feedforward': 128, 'dropout': 0.2833143475684623, 'learning_rate': 0.00020235315003022192, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 1 | `0.020136` | `{'lookback': 20, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 64, 'dropout': 0.21635464189616493, 'learning_rate': 0.00021179934425591344, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 2 | `0.019012` | `{'lookback': 40, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.21497465359228285, 'learning_rate': 4.9663730716995466e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 3 | `0.017697` | `{'lookback': 20, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.18823407432647885, 'learning_rate': 0.00017606483178281553, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 4 | `0.017322` | `{'lookback': 40, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.23243380590911633, 'learning_rate': 0.0001177098847725492, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `5D` | Fold 5 | `0.019018` | `{'lookback': 40, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.2962756788032009, 'learning_rate': 0.0001735773782470367, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 1 | `0.030989` | `{'lookback': 40, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.2769543866244206, 'learning_rate': 7.670631322417507e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 2 | `0.036711` | `{'lookback': 40, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 64, 'dropout': 0.28461841108395575, 'learning_rate': 0.00038456554482506214, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 3 | `0.028803` | `{'lookback': 60, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.09991844553958994, 'learning_rate': 7.475992999956501e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 4 | `0.029523` | `{'lookback': 40, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.23527909887412135, 'learning_rate': 0.00045809863175048893, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `20D` | Fold 5 | `0.032243` | `{'lookback': 20, 'd_model': 64, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.22754562430001113, 'learning_rate': 0.0004800378231362783, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 1 | `0.051286` | `{'lookback': 60, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 1, 'dim_feedforward': 64, 'dropout': 0.21814734713393932, 'learning_rate': 7.15175050783296e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 2 | `0.059093` | `{'lookback': 60, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.25073797473337345, 'learning_rate': 3.97695280547793e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 3 | `0.040776` | `{'lookback': 60, 'd_model': 64, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.09326911223627343, 'learning_rate': 0.00025310929005294304, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 4 | `0.047557` | `{'lookback': 20, 'd_model': 64, 'nhead': 2, 'num_encoder_layers': 1, 'dim_feedforward': 64, 'dropout': 0.2841882486253795, 'learning_rate': 9.255463913268682e-05, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `60D` | Fold 5 | `0.048923` | `{'lookback': 20, 'd_model': 64, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.2767459929366228, 'learning_rate': 0.00036822810258819354, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 1 | `0.077569` | `{'lookback': 20, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 64, 'dropout': 0.2672931630576574, 'learning_rate': 2.476921780644625e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 2 | `0.092950` | `{'lookback': 60, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 1, 'dim_feedforward': 128, 'dropout': 0.1819583023012845, 'learning_rate': 1.148052270061757e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 3 | `0.056342` | `{'lookback': 60, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 2, 'dim_feedforward': 128, 'dropout': 0.20485393971724664, 'learning_rate': 0.0002614391772436833, 'weight_decay': 0.0001, 'batch_size': 32, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 4 | `0.065369` | `{'lookback': 60, 'd_model': 32, 'nhead': 2, 'num_encoder_layers': 1, 'dim_feedforward': 128, 'dropout': 0.14033021096018644, 'learning_rate': 2.490103182139718e-05, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |
| `120D` | Fold 5 | `0.057190` | `{'lookback': 60, 'd_model': 64, 'nhead': 2, 'num_encoder_layers': 1, 'dim_feedforward': 128, 'dropout': 0.1000397117480112, 'learning_rate': 0.0004859061734665338, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` |

---

## 3. TRẠNG THÁI TUNING LAN03 & KẾT QUẢ LAN05

Lan03 chưa có kết quả tuning trong kho hiện tại nên tiếp tục để trống. Lan05 đã hoàn tất và được cập nhật từ CSV kết quả bên dưới.

### 3.1 Nhóm Deep Learning Chuỗi Thời Gian Tiên Tiến (Lần 03 — Lan03: 7 Mô Hình)
- **Mô hình:** `TCN`, `N-BEATS`, `N-HiTS`, `PatchTST`, `TSMixer`, `iTransformer`, `CNN`.
- **Phương pháp tuning:** Optuna TPE Sampler (30 trials/fold/horizon).
- **Không gian tìm kiếm đã cấu hình:**
  - `lookback`: $[20, 40, 60]$ ngày giao dịch quá khứ.
  - `hidden_dim` / `d_model`: $[32, 64, 128]$.
  - `num_layers`: $[1, 2, 3]$.
  - `dropout`: $[0.1, 0.5]$.
  - `learning_rate`: $[10^{-4}, 10^{-2}]$.
  - `batch_size`: $[16, 32, 64]$.
  - `patience`: $7$ epochs (Early Stopping).

### 3.2 Nhóm Ensemble & Chiến Lược Thích Ứng (Lan05: 9 trường hợp đã hoàn tất)

Optuna TPE cấu hình 50 trials cho mỗi fold/chân trời có lịch sử; fold 1 dùng khởi tạo. A1/A2/A3 tối ưu trọng số simplex; B1 hòa trộn top-3; B2 tối ưu power; B3 tối ưu trực tiếp trọng số simplex theo mã/kết quả hiện tại; C1 chọn mô hình theo regime và ghi nhận trọng số Optuna; C2 tối ưu N0 và power; C3 tối ưu mức phòng thủ và trọng số core.

Toàn bộ 225 bản ghi dưới đây giữ nguyên tham số và nhãn thước đo từ CSV. Mỗi bộ chỉ áp dụng cho đúng chân trời/fold tương ứng; dấu — biểu thị điểm khởi tạo không dùng để đánh giá tuning.

<details>
<summary>Ensemble_A1 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Historical_Mean': 0.07692307692307693, 'Ridge': 0.07692307692307693, 'Elastic_Net': 0.07692307692307693, 'ARIMA': 0.07692307692307693, 'ARIMAX': 0.07692307692307693, 'SARIMA': 0.07692307692307693, 'LightGBM': 0.07692307692307693, 'SVR': 0.07692307692307693, 'Random_Forest': 0.07692307692307693, 'XGBoost': 0.07692307692307693, 'RNN': 0.07692307692307693, 'LSTM': 0.07692307692307693, 'Transformer': 0.07692307692307693}}` |
| `1D` | 2 | `Hist_OOS_RMSE` | 0.010618 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Historical_Mean': 0.150497, 'Ridge': 0.013764, 'Elastic_Net': 0.135527, 'ARIMA': 0.110615, 'ARIMAX': 0.007232, 'SARIMA': 0.027731, 'LightGBM': 0.13693, 'SVR': 0.010824, 'Random_Forest': 0.04879, 'XGBoost': 0.163066, 'RNN': 0.032855, 'LSTM': 0.151922, 'Transformer': 0.010246}}` |
| `1D` | 3 | `Hist_OOS_RMSE` | 0.008273 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Historical_Mean': 0.070402, 'Ridge': 0.084126, 'Elastic_Net': 0.156819, 'ARIMA': 0.147058, 'ARIMAX': 0.073181, 'SARIMA': 0.038182, 'LightGBM': 0.137068, 'SVR': 0.008466, 'Random_Forest': 0.003765, 'XGBoost': 0.118777, 'RNN': 0.004664, 'LSTM': 0.151074, 'Transformer': 0.006419}}` |
| `1D` | 4 | `Hist_OOS_RMSE` | 0.008099 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Historical_Mean': 0.060647, 'Ridge': 0.024941, 'Elastic_Net': 0.14204, 'ARIMA': 0.077698, 'ARIMAX': 0.096887, 'SARIMA': 0.124869, 'LightGBM': 0.121281, 'SVR': 0.008941, 'Random_Forest': 0.051344, 'XGBoost': 0.144783, 'RNN': 0.012775, 'LSTM': 0.122912, 'Transformer': 0.010881}}` |
| `1D` | 5 | `Hist_OOS_RMSE` | 0.008688 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Historical_Mean': 0.043862, 'Ridge': 0.018383, 'Elastic_Net': 0.112595, 'ARIMA': 0.077588, 'ARIMAX': 0.065237, 'SARIMA': 0.122116, 'LightGBM': 0.108744, 'SVR': 0.004925, 'Random_Forest': 0.086254, 'XGBoost': 0.144366, 'RNN': 0.003685, 'LSTM': 0.12604, 'Transformer': 0.086206}}` |
| `5D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Historical_Mean': 0.07692307692307693, 'Ridge': 0.07692307692307693, 'Elastic_Net': 0.07692307692307693, 'ARIMA': 0.07692307692307693, 'ARIMAX': 0.07692307692307693, 'SARIMA': 0.07692307692307693, 'LightGBM': 0.07692307692307693, 'SVR': 0.07692307692307693, 'Random_Forest': 0.07692307692307693, 'XGBoost': 0.07692307692307693, 'RNN': 0.07692307692307693, 'LSTM': 0.07692307692307693, 'Transformer': 0.07692307692307693}}` |
| `5D` | 2 | `Hist_OOS_RMSE` | 0.022631 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Historical_Mean': 0.039923, 'Ridge': 0.038869, 'Elastic_Net': 0.01729, 'ARIMA': 0.02056, 'ARIMAX': 0.176424, 'SARIMA': 0.046694, 'LightGBM': 0.081558, 'SVR': 0.154556, 'Random_Forest': 0.151078, 'XGBoost': 0.137665, 'RNN': 0.027282, 'LSTM': 8e-06, 'Transformer': 0.108092}}` |
| `5D` | 3 | `Hist_OOS_RMSE` | 0.017512 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Historical_Mean': 0.075542, 'Ridge': 0.102322, 'Elastic_Net': 0.053786, 'ARIMA': 0.125984, 'ARIMAX': 0.134778, 'SARIMA': 0.005786, 'LightGBM': 0.067061, 'SVR': 0.033816, 'Random_Forest': 0.125169, 'XGBoost': 0.135644, 'RNN': 0.127183, 'LSTM': 0.011464, 'Transformer': 0.001465}}` |
| `5D` | 4 | `Hist_OOS_RMSE` | 0.017147 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Historical_Mean': 0.134287, 'Ridge': 0.132156, 'Elastic_Net': 0.124093, 'ARIMA': 0.086028, 'ARIMAX': 0.137698, 'SARIMA': 0.009796, 'LightGBM': 0.04849, 'SVR': 0.059748, 'Random_Forest': 0.076263, 'XGBoost': 0.090152, 'RNN': 0.053312, 'LSTM': 0.008721, 'Transformer': 0.039257}}` |
| `5D` | 5 | `Hist_OOS_RMSE` | 0.018565 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Historical_Mean': 0.037466, 'Ridge': 0.147654, 'Elastic_Net': 0.159157, 'ARIMA': 0.147811, 'ARIMAX': 0.140913, 'SARIMA': 0.016021, 'LightGBM': 0.006331, 'SVR': 0.023214, 'Random_Forest': 0.105968, 'XGBoost': 0.068498, 'RNN': 0.048862, 'LSTM': 0.005564, 'Transformer': 0.092541}}` |
| `20D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Historical_Mean': 0.07692307692307693, 'Ridge': 0.07692307692307693, 'Elastic_Net': 0.07692307692307693, 'ARIMA': 0.07692307692307693, 'ARIMAX': 0.07692307692307693, 'SARIMA': 0.07692307692307693, 'LightGBM': 0.07692307692307693, 'SVR': 0.07692307692307693, 'Random_Forest': 0.07692307692307693, 'XGBoost': 0.07692307692307693, 'RNN': 0.07692307692307693, 'LSTM': 0.07692307692307693, 'Transformer': 0.07692307692307693}}` |
| `20D` | 2 | `Hist_OOS_RMSE` | 0.040453 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Historical_Mean': 0.001184, 'Ridge': 0.083489, 'Elastic_Net': 0.127208, 'ARIMA': 0.143444, 'ARIMAX': 0.138462, 'SARIMA': 0.048481, 'LightGBM': 0.004919, 'SVR': 0.057655, 'Random_Forest': 0.078272, 'XGBoost': 0.061149, 'RNN': 0.125971, 'LSTM': 0.062325, 'Transformer': 0.067441}}` |
| `20D` | 3 | `Hist_OOS_RMSE` | 0.031118 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Historical_Mean': 0.12706, 'Ridge': 0.158271, 'Elastic_Net': 0.049212, 'ARIMA': 0.113965, 'ARIMAX': 0.053018, 'SARIMA': 0.099939, 'LightGBM': 0.006615, 'SVR': 0.046494, 'Random_Forest': 0.004969, 'XGBoost': 0.015475, 'RNN': 0.16379, 'LSTM': 0.125828, 'Transformer': 0.035364}}` |
| `20D` | 4 | `Hist_OOS_RMSE` | 0.029671 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Historical_Mean': 0.164173, 'Ridge': 0.143233, 'Elastic_Net': 0.126864, 'ARIMA': 0.175684, 'ARIMAX': 0.077276, 'SARIMA': 0.022614, 'LightGBM': 0.001297, 'SVR': 0.033604, 'Random_Forest': 0.034304, 'XGBoost': 0.04004, 'RNN': 0.12141, 'LSTM': 0.054646, 'Transformer': 0.004856}}` |
| `20D` | 5 | `Hist_OOS_RMSE` | 0.033797 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Historical_Mean': 0.121279, 'Ridge': 0.062107, 'Elastic_Net': 0.112047, 'ARIMA': 0.004061, 'ARIMAX': 0.142016, 'SARIMA': 0.099464, 'LightGBM': 0.0053, 'SVR': 0.004699, 'Random_Forest': 0.112393, 'XGBoost': 0.095799, 'RNN': 0.100833, 'LSTM': 0.134475, 'Transformer': 0.005528}}` |
| `60D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Historical_Mean': 0.07692307692307693, 'Ridge': 0.07692307692307693, 'Elastic_Net': 0.07692307692307693, 'ARIMA': 0.07692307692307693, 'ARIMAX': 0.07692307692307693, 'SARIMA': 0.07692307692307693, 'LightGBM': 0.07692307692307693, 'SVR': 0.07692307692307693, 'Random_Forest': 0.07692307692307693, 'XGBoost': 0.07692307692307693, 'RNN': 0.07692307692307693, 'LSTM': 0.07692307692307693, 'Transformer': 0.07692307692307693}}` |
| `60D` | 2 | `Hist_OOS_RMSE` | 0.058077 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Historical_Mean': 0.120104, 'Ridge': 0.142606, 'Elastic_Net': 0.023283, 'ARIMA': 0.071181, 'ARIMAX': 0.128835, 'SARIMA': 0.131079, 'LightGBM': 0.007316, 'SVR': 0.013229, 'Random_Forest': 0.007258, 'XGBoost': 0.005965, 'RNN': 0.160759, 'LSTM': 0.163879, 'Transformer': 0.024505}}` |
| `60D` | 3 | `Hist_OOS_RMSE` | 0.047254 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Historical_Mean': 0.16764, 'Ridge': 0.179011, 'Elastic_Net': 0.033681, 'ARIMA': 0.030599, 'ARIMAX': 0.185323, 'SARIMA': 0.070005, 'LightGBM': 0.044201, 'SVR': 0.060912, 'Random_Forest': 0.014257, 'XGBoost': 0.016412, 'RNN': 0.020415, 'LSTM': 0.163361, 'Transformer': 0.014183}}` |
| `60D` | 4 | `Hist_OOS_RMSE` | 0.048592 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Historical_Mean': 0.121449, 'Ridge': 0.08427, 'Elastic_Net': 0.142572, 'ARIMA': 0.069019, 'ARIMAX': 0.132366, 'SARIMA': 0.024287, 'LightGBM': 0.055219, 'SVR': 0.067386, 'Random_Forest': 0.065119, 'XGBoost': 0.000321, 'RNN': 0.089476, 'LSTM': 0.146804, 'Transformer': 0.001713}}` |
| `60D` | 5 | `Hist_OOS_RMSE` | 0.054263 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Historical_Mean': 0.190236, 'Ridge': 0.034797, 'Elastic_Net': 0.117261, 'ARIMA': 0.096104, 'ARIMAX': 0.074224, 'SARIMA': 0.080246, 'LightGBM': 0.005506, 'SVR': 0.102076, 'Random_Forest': 0.019948, 'XGBoost': 0.064192, 'RNN': 0.179016, 'LSTM': 0.021468, 'Transformer': 0.014927}}` |
| `120D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Historical_Mean': 0.07692307692307693, 'Ridge': 0.07692307692307693, 'Elastic_Net': 0.07692307692307693, 'ARIMA': 0.07692307692307693, 'ARIMAX': 0.07692307692307693, 'SARIMA': 0.07692307692307693, 'LightGBM': 0.07692307692307693, 'SVR': 0.07692307692307693, 'Random_Forest': 0.07692307692307693, 'XGBoost': 0.07692307692307693, 'RNN': 0.07692307692307693, 'LSTM': 0.07692307692307693, 'Transformer': 0.07692307692307693}}` |
| `120D` | 2 | `Hist_OOS_RMSE` | 0.065802 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Historical_Mean': 0.143348, 'Ridge': 0.148724, 'Elastic_Net': 0.057928, 'ARIMA': 0.100727, 'ARIMAX': 0.085004, 'SARIMA': 0.084305, 'LightGBM': 0.006201, 'SVR': 0.03121, 'Random_Forest': 0.003682, 'XGBoost': 0.010159, 'RNN': 0.153559, 'LSTM': 0.13203, 'Transformer': 0.043122}}` |
| `120D` | 3 | `Hist_OOS_RMSE` | 0.058833 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Historical_Mean': 0.123783, 'Ridge': 0.165562, 'Elastic_Net': 0.085161, 'ARIMA': 0.141268, 'ARIMAX': 0.017947, 'SARIMA': 0.130235, 'LightGBM': 0.006955, 'SVR': 0.009202, 'Random_Forest': 0.003173, 'XGBoost': 0.02967, 'RNN': 0.172055, 'LSTM': 0.086482, 'Transformer': 0.028505}}` |
| `120D` | 4 | `Hist_OOS_RMSE` | 0.059172 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Historical_Mean': 0.088647, 'Ridge': 0.151014, 'Elastic_Net': 0.08202, 'ARIMA': 0.151784, 'ARIMAX': 0.006881, 'SARIMA': 0.190205, 'LightGBM': 0.016502, 'SVR': 0.047194, 'Random_Forest': 0.003752, 'XGBoost': 0.029007, 'RNN': 0.085347, 'LSTM': 0.085676, 'Transformer': 0.061972}}` |
| `120D` | 5 | `Hist_OOS_RMSE` | 0.064809 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Historical_Mean': 0.166085, 'Ridge': 0.122503, 'Elastic_Net': 0.126915, 'ARIMA': 0.174887, 'ARIMAX': 0.026023, 'SARIMA': 0.112383, 'LightGBM': 0.000962, 'SVR': 0.028549, 'Random_Forest': 0.039784, 'XGBoost': 0.02522, 'RNN': 0.146224, 'LSTM': 0.021832, 'Transformer': 0.008632}}` |

</details>


<details>
<summary>Ensemble_A2 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `1D` | 2 | `Hist_OOS_RMSE` | 0.010634 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.001784, 'SVR': 0.004514, 'LightGBM': 0.643419, 'SARIMA': 0.350283}}` |
| `1D` | 3 | `Hist_OOS_RMSE` | 0.008286 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.005288, 'SVR': 0.003514, 'LightGBM': 0.491342, 'SARIMA': 0.499856}}` |
| `1D` | 4 | `Hist_OOS_RMSE` | 0.008104 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.094746, 'SVR': 6.1e-05, 'LightGBM': 0.42447, 'SARIMA': 0.480722}}` |
| `1D` | 5 | `Hist_OOS_RMSE` | 0.008698 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.010263, 'SVR': 0.003301, 'LightGBM': 0.210192, 'SARIMA': 0.776244}}` |
| `5D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `5D` | 2 | `Hist_OOS_RMSE` | 0.022745 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.569362, 'SVR': 0.36279, 'LightGBM': 0.026188, 'SARIMA': 0.04166}}` |
| `5D` | 3 | `Hist_OOS_RMSE` | 0.017528 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.528264, 'SVR': 0.00598, 'LightGBM': 0.461018, 'SARIMA': 0.004738}}` |
| `5D` | 4 | `Hist_OOS_RMSE` | 0.017136 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.737743, 'SVR': 0.010782, 'LightGBM': 0.249224, 'SARIMA': 0.002251}}` |
| `5D` | 5 | `Hist_OOS_RMSE` | 0.018625 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.816217, 'SVR': 0.03719, 'LightGBM': 0.095517, 'SARIMA': 0.051076}}` |
| `20D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `20D` | 2 | `Hist_OOS_RMSE` | 0.040051 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.180253, 'SVR': 0.486701, 'LightGBM': 0.001614, 'SARIMA': 0.331433}}` |
| `20D` | 3 | `Hist_OOS_RMSE` | 0.031055 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.471673, 'SVR': 0.166496, 'LightGBM': 0.013723, 'SARIMA': 0.348107}}` |
| `20D` | 4 | `Hist_OOS_RMSE` | 0.029518 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.747391, 'SVR': 0.016727, 'LightGBM': 0.007665, 'SARIMA': 0.228217}}` |
| `20D` | 5 | `Hist_OOS_RMSE` | 0.034087 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.516539, 'SVR': 0.196336, 'LightGBM': 0.000935, 'SARIMA': 0.28619}}` |
| `60D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `60D` | 2 | `Hist_OOS_RMSE` | 0.058276 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.778207, 'SVR': 0.014531, 'LightGBM': 0.006995, 'SARIMA': 0.200267}}` |
| `60D` | 3 | `Hist_OOS_RMSE` | 0.047064 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.624517, 'SVR': 0.021367, 'LightGBM': 0.023823, 'SARIMA': 0.330293}}` |
| `60D` | 4 | `Hist_OOS_RMSE` | 0.048355 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.498705, 'SVR': 0.007268, 'LightGBM': 0.004091, 'SARIMA': 0.489936}}` |
| `60D` | 5 | `Hist_OOS_RMSE` | 0.055333 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.16039, 'SVR': 0.279959, 'LightGBM': 0.007027, 'SARIMA': 0.552624}}` |
| `120D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `120D` | 2 | `Hist_OOS_RMSE` | 0.064351 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.778207, 'SVR': 0.014531, 'LightGBM': 0.006995, 'SARIMA': 0.200267}}` |
| `120D` | 3 | `Hist_OOS_RMSE` | 0.05751 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.523145, 'SVR': 0.007663, 'LightGBM': 0.00421, 'SARIMA': 0.464982}}` |
| `120D` | 4 | `Hist_OOS_RMSE` | 0.057874 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.424774, 'SVR': 0.009459, 'LightGBM': 0.004102, 'SARIMA': 0.561665}}` |
| `120D` | 5 | `Hist_OOS_RMSE` | 0.063872 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.296691, 'SVR': 0.012958, 'LightGBM': 0.004275, 'SARIMA': 0.686077}}` |

</details>


<details>
<summary>Ensemble_A3 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.3333333333333333, 'SARIMA': 0.3333333333333333, 'Historical_Mean': 0.3333333333333333}}` |
| `1D` | 2 | `Hist_OOS_RMSE` | 0.010623 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.041416, 'SARIMA': 0.000158, 'Historical_Mean': 0.958427}}` |
| `1D` | 3 | `Hist_OOS_RMSE` | 0.008284 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.011971, 'SARIMA': 0.044651, 'Historical_Mean': 0.943378}}` |
| `1D` | 4 | `Hist_OOS_RMSE` | 0.008104 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.088721, 'SARIMA': 0.339407, 'Historical_Mean': 0.571872}}` |
| `1D` | 5 | `Hist_OOS_RMSE` | 0.008694 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.022671, 'SARIMA': 0.447036, 'Historical_Mean': 0.530294}}` |
| `5D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.3333333333333333, 'SARIMA': 0.3333333333333333, 'Historical_Mean': 0.3333333333333333}}` |
| `5D` | 2 | `Hist_OOS_RMSE` | 0.022712 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.490484, 'SARIMA': 0.00099, 'Historical_Mean': 0.508526}}` |
| `5D` | 3 | `Hist_OOS_RMSE` | 0.017486 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.635552, 'SARIMA': 0.001334, 'Historical_Mean': 0.363114}}` |
| `5D` | 4 | `Hist_OOS_RMSE` | 0.017094 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.57344, 'SARIMA': 0.00479, 'Historical_Mean': 0.42177}}` |
| `5D` | 5 | `Hist_OOS_RMSE` | 0.018569 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.440858, 'SARIMA': 0.001324, 'Historical_Mean': 0.557818}}` |
| `20D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.3333333333333333, 'SARIMA': 0.3333333333333333, 'Historical_Mean': 0.3333333333333333}}` |
| `20D` | 2 | `Hist_OOS_RMSE` | 0.040198 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.663848, 'SARIMA': 0.334527, 'Historical_Mean': 0.001625}}` |
| `20D` | 3 | `Hist_OOS_RMSE` | 0.030878 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.710323, 'SARIMA': 0.274373, 'Historical_Mean': 0.015304}}` |
| `20D` | 4 | `Hist_OOS_RMSE` | 0.029411 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.538567, 'SARIMA': 0.045531, 'Historical_Mean': 0.415902}}` |
| `20D` | 5 | `Hist_OOS_RMSE` | 0.033495 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.074308, 'SARIMA': 0.009879, 'Historical_Mean': 0.915813}}` |
| `60D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.3333333333333333, 'SARIMA': 0.3333333333333333, 'Historical_Mean': 0.3333333333333333}}` |
| `60D` | 2 | `Hist_OOS_RMSE` | 0.056888 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.639333, 'SARIMA': 0.01196, 'Historical_Mean': 0.348707}}` |
| `60D` | 3 | `Hist_OOS_RMSE` | 0.045539 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.458295, 'SARIMA': 0.001065, 'Historical_Mean': 0.540641}}` |
| `60D` | 4 | `Hist_OOS_RMSE` | 0.046217 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.076251, 'SARIMA': 0.012912, 'Historical_Mean': 0.910837}}` |
| `60D` | 5 | `Hist_OOS_RMSE` | 0.051203 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.021779, 'SARIMA': 0.002295, 'Historical_Mean': 0.975926}}` |
| `120D` | 1 | `Cold_Start_Prior` | — (khởi tạo) | `{'mode': 'cold_start_prior', 'weights': {'Ridge': 0.3333333333333333, 'SARIMA': 0.3333333333333333, 'Historical_Mean': 0.3333333333333333}}` |
| `120D` | 2 | `Hist_OOS_RMSE` | 0.060581 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..1', 'weights': {'Ridge': 0.440858, 'SARIMA': 0.001324, 'Historical_Mean': 0.557818}}` |
| `120D` | 3 | `Hist_OOS_RMSE` | 0.055931 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..2', 'weights': {'Ridge': 0.003421, 'SARIMA': 0.216667, 'Historical_Mean': 0.779911}}` |
| `120D` | 4 | `Hist_OOS_RMSE` | 0.056815 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..3', 'weights': {'Ridge': 0.00865, 'SARIMA': 0.335312, 'Historical_Mean': 0.656038}}` |
| `120D` | 5 | `Hist_OOS_RMSE` | 0.062166 | `{'mode': 'optuna_tpe', 'tuning_folds': '1..4', 'weights': {'Ridge': 0.002023, 'SARIMA': 0.403747, 'Historical_Mean': 0.59423}}` |

</details>


<details>
<summary>Baseline_B1 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b1_prior_pre_design', 'selected_model': 'SARIMA', 'reason': 'Fold 1 cold-start fallback to preliminary design report'}` |
| `1D` | 2 | `Val_RMSE` | 0.010579 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['LSTM', 'Elastic_Net', 'ARIMAX'], 'blended_weights': {'LSTM': 0.573778, 'Elastic_Net': 0.411888, 'ARIMAX': 0.014335}, 'best_single_model': 'LSTM', 'top_mses': {'LSTM': 0.00011235, 'Elastic_Net': 0.0001127, 'ARIMAX': 0.00011279, 'Historical_Mean': 0.00011283, 'XGBoost': 0.00011285}}` |
| `1D` | 3 | `Val_RMSE` | 0.008278 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'XGBoost', 'Historical_Mean'], 'blended_weights': {'Elastic_Net': 0.927049, 'XGBoost': 0.064607, 'Historical_Mean': 0.008344}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 6.852e-05, 'XGBoost': 6.862e-05, 'Historical_Mean': 6.862e-05, 'ARIMA': 6.865e-05, 'LightGBM': 6.873e-05}}` |
| `1D` | 4 | `Val_RMSE` | 0.008099 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'ARIMA', 'XGBoost'], 'blended_weights': {'Elastic_Net': 0.517197, 'ARIMA': 0.471813, 'XGBoost': 0.010989}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 6.564e-05, 'ARIMA': 6.564e-05, 'XGBoost': 6.573e-05, 'Historical_Mean': 6.573e-05, 'SARIMA': 6.58e-05}}` |
| `1D` | 5 | `Val_RMSE` | 0.00869 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'ARIMA', 'Historical_Mean'], 'blended_weights': {'Elastic_Net': 0.661789, 'ARIMA': 0.336706, 'Historical_Mean': 0.001505}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 7.554e-05, 'ARIMA': 7.563e-05, 'Historical_Mean': 7.566e-05, 'XGBoost': 7.566e-05, 'SARIMA': 7.569e-05}}` |
| `5D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b1_prior_pre_design', 'selected_model': 'XGBoost', 'reason': 'Fold 1 cold-start fallback to preliminary design report'}` |
| `5D` | 2 | `Val_RMSE` | 0.022463 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'ARIMAX', 'Ridge'], 'blended_weights': {'Elastic_Net': 0.448292, 'ARIMAX': 0.544709, 'Ridge': 0.006999}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 0.00051547, 'ARIMAX': 0.00051646, 'Ridge': 0.00051904, 'Historical_Mean': 0.00051949, 'XGBoost': 0.00051953}}` |
| `5D` | 3 | `Val_RMSE` | 0.017465 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'Ridge', 'XGBoost'], 'blended_weights': {'Elastic_Net': 0.606139, 'Ridge': 0.392975, 'XGBoost': 0.000885}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 0.00030591, 'Ridge': 0.00030719, 'XGBoost': 0.00030839, 'Historical_Mean': 0.00030853, 'Random_Forest': 0.00030878}}` |
| `5D` | 4 | `Val_RMSE` | 0.017084 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'Ridge', 'XGBoost'], 'blended_weights': {'Elastic_Net': 0.569736, 'Ridge': 0.391506, 'XGBoost': 0.038759}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 0.00029258, 'Ridge': 0.00029372, 'XGBoost': 0.00029434, 'Historical_Mean': 0.00029447, 'Random_Forest': 0.00029507}}` |
| `5D` | 5 | `Val_RMSE` | 0.018557 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'Random_Forest', 'XGBoost'], 'blended_weights': {'Elastic_Net': 0.726217, 'Random_Forest': 0.257954, 'XGBoost': 0.015828}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 0.00034446, 'Random_Forest': 0.00034545, 'XGBoost': 0.00034614, 'Historical_Mean': 0.00034619, 'Ridge': 0.00034782}}` |
| `20D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b1_prior_pre_design', 'selected_model': 'Elastic_Net', 'reason': 'Fold 1 cold-start fallback to preliminary design report'}` |
| `20D` | 2 | `Val_RMSE` | 0.039789 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['ARIMAX', 'Ridge', 'SVR'], 'blended_weights': {'ARIMAX': 0.797945, 'Ridge': 0.186365, 'SVR': 0.01569}, 'best_single_model': 'ARIMAX', 'top_mses': {'ARIMAX': 0.00158803, 'Ridge': 0.00164222, 'SVR': 0.0016513, 'Historical_Mean': 0.00165386, 'Elastic_Net': 0.00165386}}` |
| `20D` | 3 | `Val_RMSE` | 0.030935 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Ridge', 'Elastic_Net', 'RNN'], 'blended_weights': {'Ridge': 0.542699, 'Elastic_Net': 0.441329, 'RNN': 0.015972}, 'best_single_model': 'Ridge', 'top_mses': {'Ridge': 0.00096534, 'Elastic_Net': 0.00097054, 'RNN': 0.00097749, 'Historical_Mean': 0.00098111, 'LSTM': 0.00098327}}` |
| `20D` | 4 | `Val_RMSE` | 0.029411 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Elastic_Net', 'Ridge', 'Historical_Mean'], 'blended_weights': {'Elastic_Net': 0.172664, 'Ridge': 0.476653, 'Historical_Mean': 0.350684}, 'best_single_model': 'Elastic_Net', 'top_mses': {'Elastic_Net': 0.00087223, 'Ridge': 0.00087706, 'Historical_Mean': 0.00087956, 'LSTM': 0.00089689, 'RNN': 0.00089725}}` |
| `20D` | 5 | `Val_RMSE` | 0.033493 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Historical_Mean', 'Elastic_Net', 'RNN'], 'blended_weights': {'Historical_Mean': 0.978552, 'Elastic_Net': 0.015474, 'RNN': 0.005974}, 'best_single_model': 'Historical_Mean', 'top_mses': {'Historical_Mean': 0.00112161, 'Elastic_Net': 0.00114689, 'RNN': 0.00116286, 'Random_Forest': 0.00116783, 'Ridge': 0.00117008}}` |
| `60D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b1_prior_pre_design', 'selected_model': 'Historical_Mean', 'reason': 'Fold 1 cold-start fallback to preliminary design report'}` |
| `60D` | 2 | `Val_RMSE` | 0.054206 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['ARIMAX', 'Ridge', 'RNN'], 'blended_weights': {'ARIMAX': 0.953479, 'Ridge': 0.018049, 'RNN': 0.028472}, 'best_single_model': 'ARIMAX', 'top_mses': {'ARIMAX': 0.00293858, 'Ridge': 0.00329482, 'RNN': 0.00332545, 'LSTM': 0.00337325, 'Historical_Mean': 0.00345451}}` |
| `60D` | 3 | `Val_RMSE` | 0.045546 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Historical_Mean', 'Ridge', 'Elastic_Net'], 'blended_weights': {'Historical_Mean': 0.512517, 'Ridge': 0.471881, 'Elastic_Net': 0.015602}, 'best_single_model': 'Historical_Mean', 'top_mses': {'Historical_Mean': 0.00215554, 'Ridge': 0.00217108, 'Elastic_Net': 0.00220511, 'LSTM': 0.00220845, 'ARIMAX': 0.00228228}}` |
| `60D` | 4 | `Val_RMSE` | 0.046228 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Historical_Mean', 'LSTM', 'Elastic_Net'], 'blended_weights': {'Historical_Mean': 0.979256, 'LSTM': 0.005908, 'Elastic_Net': 0.014836}, 'best_single_model': 'Historical_Mean', 'top_mses': {'Historical_Mean': 0.00213493, 'LSTM': 0.00219384, 'Elastic_Net': 0.00233301, 'SARIMA': 0.00240437, 'Ridge': 0.00244843}}` |
| `60D` | 5 | `Val_RMSE` | 0.051193 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Historical_Mean', 'Elastic_Net', 'LSTM'], 'blended_weights': {'Historical_Mean': 0.978552, 'Elastic_Net': 0.015474, 'LSTM': 0.005974}, 'best_single_model': 'Historical_Mean', 'top_mses': {'Historical_Mean': 0.00261451, 'Elastic_Net': 0.00299741, 'LSTM': 0.00304378, 'SARIMA': 0.00311437, 'ARIMAX': 0.00319895}}` |
| `120D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b1_prior_pre_design', 'selected_model': 'ARIMAX', 'reason': 'Fold 1 cold-start fallback to preliminary design report'}` |
| `120D` | 2 | `Val_RMSE` | 0.061978 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Historical_Mean', 'Elastic_Net', 'LSTM'], 'blended_weights': {'Historical_Mean': 0.558482, 'Elastic_Net': 0.439149, 'LSTM': 0.002368}, 'best_single_model': 'Historical_Mean', 'top_mses': {'Historical_Mean': 0.00384131, 'Elastic_Net': 0.00384131, 'LSTM': 0.0038554, 'Ridge': 0.00387075, 'ARIMAX': 0.00409576}}` |
| `120D` | 3 | `Val_RMSE` | 0.056331 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Historical_Mean', 'Elastic_Net', 'LSTM'], 'blended_weights': {'Historical_Mean': 0.558482, 'Elastic_Net': 0.439149, 'LSTM': 0.002368}, 'best_single_model': 'Historical_Mean', 'top_mses': {'Historical_Mean': 0.00317267, 'Elastic_Net': 0.00317267, 'LSTM': 0.00343699, 'SARIMA': 0.00365894, 'ARIMA': 0.00370449}}` |
| `120D` | 4 | `Val_RMSE` | 0.056572 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['Historical_Mean', 'Elastic_Net', 'ARIMAX'], 'blended_weights': {'Historical_Mean': 0.246555, 'Elastic_Net': 0.380491, 'ARIMAX': 0.372954}, 'best_single_model': 'Historical_Mean', 'top_mses': {'Historical_Mean': 0.00329985, 'Elastic_Net': 0.00329985, 'ARIMAX': 0.00348423, 'SARIMA': 0.00351102, 'ARIMA': 0.00354161}}` |
| `120D` | 5 | `Val_RMSE` | 0.06123 | `{'mode': 'b1_optuna_top3_blend', 'top_candidates': ['ARIMAX', 'Historical_Mean', 'Elastic_Net'], 'blended_weights': {'ARIMAX': 0.52746, 'Historical_Mean': 0.312108, 'Elastic_Net': 0.160432}, 'best_single_model': 'ARIMAX', 'top_mses': {'ARIMAX': 0.00390602, 'Historical_Mean': 0.00394554, 'Elastic_Net': 0.00394554, 'SARIMA': 0.00405807, 'ARIMA': 0.00407832}}` |

</details>


<details>
<summary>Baseline_B2 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b2_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'power': 1.0}` |
| `1D` | 2 | `Val_RMSE` | 0.010685 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.253187, 'SVR': 0.218367, 'LightGBM': 0.268506, 'SARIMA': 0.25994}, 'mses': {'Ridge': 0.00011445, 'SVR': 0.00011789, 'LightGBM': 0.00011311, 'SARIMA': 0.00011385}}` |
| `1D` | 3 | `Val_RMSE` | 0.008326 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.249091, 'SVR': 0.21826, 'LightGBM': 0.268217, 'SARIMA': 0.264432}, 'mses': {'Ridge': 6.976e-05, 'SVR': 7.163e-05, 'LightGBM': 6.873e-05, 'SARIMA': 6.893e-05}}` |
| `1D` | 4 | `Val_RMSE` | 0.008131 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.251695, 'SVR': 0.225062, 'LightGBM': 0.260564, 'SARIMA': 0.262679}, 'mses': {'Ridge': 6.637e-05, 'SVR': 6.787e-05, 'LightGBM': 6.591e-05, 'SARIMA': 6.58e-05}}` |
| `1D` | 5 | `Val_RMSE` | 0.00872 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.249016, 'SVR': 0.23499, 'LightGBM': 0.254671, 'SARIMA': 0.261322}, 'mses': {'Ridge': 7.643e-05, 'SVR': 7.732e-05, 'LightGBM': 7.608e-05, 'SARIMA': 7.569e-05}}` |
| `5D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b2_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'power': 1.0}` |
| `5D` | 2 | `Val_RMSE` | 0.02278 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.312688, 'SVR': 0.309722, 'LightGBM': 0.291421, 'SARIMA': 0.086169}, 'mses': {'Ridge': 0.00051904, 'SVR': 0.00052003, 'LightGBM': 0.0005264, 'SARIMA': 0.00067174}}` |
| `5D` | 3 | `Val_RMSE` | 0.017605 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.422951, 'SVR': 0.064877, 'LightGBM': 0.402214, 'SARIMA': 0.109958}, 'mses': {'Ridge': 0.00030719, 'SVR': 0.00044702, 'LightGBM': 0.0003103, 'SARIMA': 0.00040224}}` |
| `5D` | 4 | `Val_RMSE` | 0.017218 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.404357, 'SVR': 0.102742, 'LightGBM': 0.380611, 'SARIMA': 0.112289}, 'mses': {'Ridge': 0.00029372, 'SVR': 0.00038636, 'LightGBM': 0.0002973, 'SARIMA': 0.00037955}}` |
| `5D` | 5 | `Val_RMSE` | 0.018668 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.39619, 'SVR': 0.105373, 'LightGBM': 0.377417, 'SARIMA': 0.121019}, 'mses': {'Ridge': 0.00034782, 'SVR': 0.00045336, 'LightGBM': 0.00035121, 'SARIMA': 0.00044097}}` |
| `20D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b2_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'power': 1.0}` |
| `20D` | 2 | `Val_RMSE` | 0.040496 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.342766, 'SVR': 0.333447, 'LightGBM': 0.060065, 'SARIMA': 0.263723}, 'mses': {'Ridge': 0.00164222, 'SVR': 0.0016513, 'LightGBM': 0.00232688, 'SARIMA': 0.00173066}}` |
| `20D` | 3 | `Val_RMSE` | 0.031428 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.40463, 'SVR': 0.22103, 'LightGBM': 0.085473, 'SARIMA': 0.288867}, 'mses': {'Ridge': 0.00096534, 'SVR': 0.0010895, 'LightGBM': 0.00131761, 'SARIMA': 0.00103268}}` |
| `20D` | 4 | `Val_RMSE` | 0.0301 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.417005, 'SVR': 0.161783, 'LightGBM': 0.132087, 'SARIMA': 0.289125}, 'mses': {'Ridge': 0.00087706, 'SVR': 0.00106, 'LightGBM': 0.00110389, 'SARIMA': 0.00094374}}` |
| `20D` | 5 | `Val_RMSE` | 0.034339 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.330933, 'SVR': 0.226832, 'LightGBM': 0.203319, 'SARIMA': 0.238915}, 'mses': {'Ridge': 0.00117008, 'SVR': 0.00126193, 'LightGBM': 0.00128987, 'SARIMA': 0.00124889}}` |
| `60D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b2_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'power': 1.0}` |
| `60D` | 2 | `Val_RMSE` | 0.059037 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.675214, 'SVR': 0.053589, 'LightGBM': 0.008947, 'SARIMA': 0.26225}, 'mses': {'Ridge': 0.00329482, 'SVR': 0.00547017, 'LightGBM': 0.00782614, 'SARIMA': 0.00398118}}` |
| `60D` | 3 | `Val_RMSE` | 0.047453 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.566395, 'SVR': 0.096952, 'LightGBM': 0.018266, 'SARIMA': 0.318387}, 'mses': {'Ridge': 0.00217108, 'SVR': 0.0030907, 'LightGBM': 0.00431619, 'SARIMA': 0.0024363}}` |
| `60D` | 4 | `Val_RMSE` | 0.049024 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.372392, 'SVR': 0.182852, 'LightGBM': 0.036989, 'SARIMA': 0.407767}, 'mses': {'Ridge': 0.00244843, 'SVR': 0.0028229, 'LightGBM': 0.00388651, 'SARIMA': 0.00240437}}` |
| `60D` | 5 | `Val_RMSE` | 0.055845 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.280285, 'SVR': 0.294373, 'LightGBM': 0.068623, 'SARIMA': 0.356719}, 'mses': {'Ridge': 0.00326832, 'SVR': 0.00323641, 'LightGBM': 0.00433116, 'SARIMA': 0.00311437}}` |
| `120D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b2_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'power': 1.0}` |
| `120D` | 2 | `Val_RMSE` | 0.063302 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.872198, 'SVR': 0.011956, 'LightGBM': 0.00076, 'SARIMA': 0.115086}, 'mses': {'Ridge': 0.00387075, 'SVR': 0.00913195, 'LightGBM': 0.01585145, 'SARIMA': 0.00580483}}` |
| `120D` | 3 | `Val_RMSE` | 0.057779 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.380816, 'SVR': 0.070826, 'LightGBM': 0.006453, 'SARIMA': 0.541905}, 'mses': {'Ridge': 0.00392654, 'SVR': 0.00549766, 'LightGBM': 0.00887892, 'SARIMA': 0.00365894}}` |
| `120D` | 4 | `Val_RMSE` | 0.058162 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.303068, 'SVR': 0.069055, 'LightGBM': 0.010019, 'SARIMA': 0.617858}, 'mses': {'Ridge': 0.00404883, 'SVR': 0.0054432, 'LightGBM': 0.0080095, 'SARIMA': 0.00351102}}` |
| `120D` | 5 | `Val_RMSE` | 0.06414 | `{'mode': 'b2_optuna_power', 'power': 4.99783323305094, 'weights': {'Ridge': 0.228112, 'SVR': 0.057403, 'LightGBM': 0.011332, 'SARIMA': 0.703153}, 'mses': {'Ridge': 0.00508326, 'SVR': 0.00669941, 'LightGBM': 0.00926884, 'SARIMA': 0.00405807}}` |

</details>


<details>
<summary>Baseline_B3 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b3_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `1D` | 2 | `Val_RMSE` | 0.010634 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.001784, 'SVR': 0.004514, 'LightGBM': 0.643419, 'SARIMA': 0.350283}, 'optuna_trials': 50}` |
| `1D` | 3 | `Val_RMSE` | 0.008286 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.005288, 'SVR': 0.003514, 'LightGBM': 0.491342, 'SARIMA': 0.499856}, 'optuna_trials': 50}` |
| `1D` | 4 | `Val_RMSE` | 0.008104 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.094746, 'SVR': 6.1e-05, 'LightGBM': 0.42447, 'SARIMA': 0.480722}, 'optuna_trials': 50}` |
| `1D` | 5 | `Val_RMSE` | 0.008698 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.010263, 'SVR': 0.003301, 'LightGBM': 0.210192, 'SARIMA': 0.776244}, 'optuna_trials': 50}` |
| `5D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b3_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `5D` | 2 | `Val_RMSE` | 0.022745 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.569362, 'SVR': 0.36279, 'LightGBM': 0.026188, 'SARIMA': 0.04166}, 'optuna_trials': 50}` |
| `5D` | 3 | `Val_RMSE` | 0.017528 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.528264, 'SVR': 0.00598, 'LightGBM': 0.461018, 'SARIMA': 0.004738}, 'optuna_trials': 50}` |
| `5D` | 4 | `Val_RMSE` | 0.017136 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.737743, 'SVR': 0.010782, 'LightGBM': 0.249224, 'SARIMA': 0.002251}, 'optuna_trials': 50}` |
| `5D` | 5 | `Val_RMSE` | 0.018625 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.816217, 'SVR': 0.03719, 'LightGBM': 0.095517, 'SARIMA': 0.051076}, 'optuna_trials': 50}` |
| `20D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b3_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `20D` | 2 | `Val_RMSE` | 0.040051 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.180253, 'SVR': 0.486701, 'LightGBM': 0.001614, 'SARIMA': 0.331433}, 'optuna_trials': 50}` |
| `20D` | 3 | `Val_RMSE` | 0.031055 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.471673, 'SVR': 0.166496, 'LightGBM': 0.013723, 'SARIMA': 0.348107}, 'optuna_trials': 50}` |
| `20D` | 4 | `Val_RMSE` | 0.029518 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.747391, 'SVR': 0.016727, 'LightGBM': 0.007665, 'SARIMA': 0.228217}, 'optuna_trials': 50}` |
| `20D` | 5 | `Val_RMSE` | 0.034087 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.516539, 'SVR': 0.196336, 'LightGBM': 0.000935, 'SARIMA': 0.28619}, 'optuna_trials': 50}` |
| `60D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b3_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `60D` | 2 | `Val_RMSE` | 0.058276 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.778207, 'SVR': 0.014531, 'LightGBM': 0.006995, 'SARIMA': 0.200267}, 'optuna_trials': 50}` |
| `60D` | 3 | `Val_RMSE` | 0.047064 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.624517, 'SVR': 0.021367, 'LightGBM': 0.023823, 'SARIMA': 0.330293}, 'optuna_trials': 50}` |
| `60D` | 4 | `Val_RMSE` | 0.048355 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.498705, 'SVR': 0.007268, 'LightGBM': 0.004091, 'SARIMA': 0.489936}, 'optuna_trials': 50}` |
| `60D` | 5 | `Val_RMSE` | 0.055333 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.16039, 'SVR': 0.279959, 'LightGBM': 0.007027, 'SARIMA': 0.552624}, 'optuna_trials': 50}` |
| `120D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'b3_cold_start_prior', 'weights': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}` |
| `120D` | 2 | `Val_RMSE` | 0.064351 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.778207, 'SVR': 0.014531, 'LightGBM': 0.006995, 'SARIMA': 0.200267}, 'optuna_trials': 50}` |
| `120D` | 3 | `Val_RMSE` | 0.05751 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.523145, 'SVR': 0.007663, 'LightGBM': 0.00421, 'SARIMA': 0.464982}, 'optuna_trials': 50}` |
| `120D` | 4 | `Val_RMSE` | 0.057874 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.424774, 'SVR': 0.009459, 'LightGBM': 0.004102, 'SARIMA': 0.561665}, 'optuna_trials': 50}` |
| `120D` | 5 | `Val_RMSE` | 0.063872 | `{'mode': 'b3_optuna_simplex_weights', 'weights': {'Ridge': 0.296691, 'SVR': 0.012958, 'LightGBM': 0.004275, 'SARIMA': 0.686077}, 'optuna_trials': 50}` |

</details>


<details>
<summary>Baseline_C1 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c1_deterministic_argmin_mse', 'selected_models_by_regime': {'Low': 'Ridge', 'Normal': 'Ridge', 'High': 'Ridge'}, 'fallback_model': 'Ridge'}` |
| `1D` | 2 | `Val_RMSE` | 0.010609 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'RNN', 'Normal': 'ARIMAX', 'High': 'Elastic_Net'}, 'optuna_weights': {'ARIMAX': 0.489608, 'Elastic_Net': 0.507635, 'RNN': 0.002758}, 'fallback_model': 'LSTM'}` |
| `1D` | 3 | `Val_RMSE` | 0.008274 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'SARIMA', 'Normal': 'LightGBM', 'High': 'Elastic_Net'}, 'optuna_weights': {'LightGBM': 0.319897, 'SARIMA': 0.001026, 'Elastic_Net': 0.679078}, 'fallback_model': 'Elastic_Net'}` |
| `1D` | 4 | `Val_RMSE` | 0.008099 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'SARIMA', 'Normal': 'ARIMAX', 'High': 'ARIMA'}, 'optuna_weights': {'ARIMAX': 0.264729, 'SARIMA': 0.001081, 'ARIMA': 0.73419}, 'fallback_model': 'Elastic_Net'}` |
| `1D` | 5 | `Val_RMSE` | 0.008694 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'SARIMA', 'Normal': 'ARIMAX', 'High': 'ARIMA'}, 'optuna_weights': {'ARIMAX': 0.21272, 'SARIMA': 0.305154, 'ARIMA': 0.482127}, 'fallback_model': 'Elastic_Net'}` |
| `5D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c1_deterministic_argmin_mse', 'selected_models_by_regime': {'Low': 'Ridge', 'Normal': 'Ridge', 'High': 'Ridge'}, 'fallback_model': 'Ridge'}` |
| `5D` | 2 | `Val_RMSE` | 0.022458 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'ARIMAX', 'Normal': 'SVR', 'High': 'Elastic_Net'}, 'optuna_weights': {'ARIMAX': 0.472525, 'SVR': 0.002552, 'Elastic_Net': 0.524923}, 'fallback_model': 'Elastic_Net'}` |
| `5D` | 3 | `Val_RMSE` | 0.017492 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'LightGBM', 'Normal': 'Random_Forest', 'High': 'Elastic_Net'}, 'optuna_weights': {'LightGBM': 0.011823, 'Elastic_Net': 0.932762, 'Random_Forest': 0.055415}, 'fallback_model': 'Elastic_Net'}` |
| `5D` | 4 | `Val_RMSE` | 0.017084 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'Random_Forest', 'Normal': 'Ridge', 'High': 'Elastic_Net'}, 'optuna_weights': {'Elastic_Net': 0.639052, 'Ridge': 0.358766, 'Random_Forest': 0.002182}, 'fallback_model': 'Elastic_Net'}` |
| `5D` | 5 | `Val_RMSE` | 0.018557 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'Random_Forest', 'Normal': 'LightGBM', 'High': 'Elastic_Net'}, 'optuna_weights': {'LightGBM': 0.000322, 'Elastic_Net': 0.664388, 'Random_Forest': 0.33529}, 'fallback_model': 'Elastic_Net'}` |
| `20D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c1_deterministic_argmin_mse', 'selected_models_by_regime': {'Low': 'Ridge', 'Normal': 'Ridge', 'High': 'Ridge'}, 'fallback_model': 'Ridge'}` |
| `20D` | 2 | `Val_RMSE` | 0.039574 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'ARIMAX', 'Normal': 'ARIMAX', 'High': 'Historical_Mean'}, 'optuna_weights': {'ARIMAX': 0.665291, 'Historical_Mean': 0.334709}, 'fallback_model': 'ARIMAX'}` |
| `20D` | 3 | `Val_RMSE` | 0.030954 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'Ridge', 'Normal': 'ARIMAX', 'High': 'Historical_Mean'}, 'optuna_weights': {'ARIMAX': 0.038078, 'Ridge': 0.567845, 'Historical_Mean': 0.394077}, 'fallback_model': 'Ridge'}` |
| `20D` | 4 | `Val_RMSE` | 0.029586 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'LSTM', 'Normal': 'ARIMAX', 'High': 'Historical_Mean'}, 'optuna_weights': {'ARIMAX': 0.072964, 'LSTM': 0.264115, 'Historical_Mean': 0.662921}, 'fallback_model': 'Elastic_Net'}` |
| `20D` | 5 | `Val_RMSE` | 0.033531 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'LightGBM', 'Normal': 'Transformer', 'High': 'Historical_Mean'}, 'optuna_weights': {'LightGBM': 0.01916, 'Transformer': 0.016207, 'Historical_Mean': 0.964632}, 'fallback_model': 'Historical_Mean'}` |
| `60D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c1_deterministic_argmin_mse', 'selected_models_by_regime': {'Low': 'Ridge', 'Normal': 'Ridge', 'High': 'Ridge'}, 'fallback_model': 'Ridge'}` |
| `60D` | 2 | `Val_RMSE` | 0.058146 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'ARIMA', 'Normal': 'Transformer', 'High': 'LSTM'}, 'optuna_weights': {'LSTM': 0.979151, 'Transformer': 0.003967, 'ARIMA': 0.016883}, 'fallback_model': 'ARIMAX'}` |
| `60D` | 3 | `Val_RMSE` | 0.046963 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'SARIMA', 'Normal': 'Transformer', 'High': 'LSTM'}, 'optuna_weights': {'Transformer': 0.076887, 'SARIMA': 0.012645, 'LSTM': 0.910469}, 'fallback_model': 'Historical_Mean'}` |
| `60D` | 4 | `Val_RMSE` | 0.046871 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'Random_Forest', 'Normal': 'SVR', 'High': 'LSTM'}, 'optuna_weights': {'LSTM': 0.981763, 'SVR': 0.005396, 'Random_Forest': 0.012841}, 'fallback_model': 'Historical_Mean'}` |
| `60D` | 5 | `Val_RMSE` | 0.051203 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'Historical_Mean', 'Normal': 'SVR', 'High': 'RNN'}, 'optuna_weights': {'SVR': 0.003026, 'RNN': 0.020646, 'Historical_Mean': 0.976327}, 'fallback_model': 'Historical_Mean'}` |
| `120D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c1_deterministic_argmin_mse', 'selected_models_by_regime': {'Low': 'Ridge', 'Normal': 'Ridge', 'High': 'Ridge'}, 'fallback_model': 'Ridge'}` |
| `120D` | 2 | `Val_RMSE` | 0.060555 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'Ridge', 'Normal': 'Ridge', 'High': 'Historical_Mean'}, 'optuna_weights': {'Ridge': 0.490708, 'Historical_Mean': 0.509292}, 'fallback_model': 'Historical_Mean'}` |
| `120D` | 3 | `Val_RMSE` | 0.055921 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'ARIMA', 'Normal': 'ARIMAX', 'High': 'Historical_Mean'}, 'optuna_weights': {'ARIMAX': 0.00139, 'ARIMA': 0.230449, 'Historical_Mean': 0.768161}, 'fallback_model': 'Historical_Mean'}` |
| `120D` | 4 | `Val_RMSE` | 0.05567 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'ARIMA', 'Normal': 'ARIMAX', 'High': 'Historical_Mean'}, 'optuna_weights': {'ARIMAX': 0.427098, 'ARIMA': 0.394134, 'Historical_Mean': 0.178768}, 'fallback_model': 'Historical_Mean'}` |
| `120D` | 5 | `Val_RMSE` | 0.060318 | `{'mode': 'c1_optuna_tuned', 'selected_models_by_regime': {'Low': 'ARIMA', 'Normal': 'ARIMAX', 'High': 'ARIMAX'}, 'optuna_weights': {'ARIMAX': 0.556946, 'ARIMA': 0.443054}, 'fallback_model': 'ARIMAX'}` |

</details>


<details>
<summary>Baseline_C2 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c2_cold_start_prior', 'weights': {'Low': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'Normal': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'High': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}, 'N0': 50.0}` |
| `1D` | 2 | `Val_RMSE` | 0.010686 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.252069, 'SVR': 0.230888, 'LightGBM': 0.261008, 'SARIMA': 0.256036}, 'lambda_shrinkage': {'Low': 0.9226, 'Normal': 0.9293, 'High': 0.9429}, 'final_regime_weights': {'Low': {'Ridge': 0.244981, 'SVR': 0.237423, 'LightGBM': 0.253323, 'SARIMA': 0.264273}, 'Normal': {'Ridge': 0.253115, 'SVR': 0.244475, 'LightGBM': 0.259137, 'SARIMA': 0.243273}, 'High': {'Ridge': 0.253542, 'SVR': 0.224292, 'LightGBM': 0.26373, 'SARIMA': 0.258435}}}` |
| `1D` | 3 | `Val_RMSE` | 0.008327 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.249653, 'SVR': 0.230831, 'LightGBM': 0.260853, 'SARIMA': 0.258663}, 'lambda_shrinkage': {'Low': 0.9795, 'Normal': 0.9452, 'High': 0.9479}, 'final_regime_weights': {'Low': {'Ridge': 0.244154, 'SVR': 0.235262, 'LightGBM': 0.254459, 'SARIMA': 0.266125}, 'Normal': {'Ridge': 0.251073, 'SVR': 0.243561, 'LightGBM': 0.25924, 'SARIMA': 0.246126}, 'High': {'Ridge': 0.251869, 'SVR': 0.223383, 'LightGBM': 0.26484, 'SARIMA': 0.259907}}}` |
| `1D` | 4 | `Val_RMSE` | 0.008131 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.251116, 'SVR': 0.234996, 'LightGBM': 0.256328, 'SARIMA': 0.257561}, 'lambda_shrinkage': {'Low': 0.9872, 'Normal': 0.9565, 'High': 0.9625}, 'final_regime_weights': {'Low': {'Ridge': 0.244786, 'SVR': 0.233163, 'LightGBM': 0.25734, 'SARIMA': 0.264712}, 'Normal': {'Ridge': 0.257197, 'SVR': 0.247649, 'LightGBM': 0.251625, 'SARIMA': 0.243529}, 'High': {'Ridge': 0.252017, 'SVR': 0.230715, 'LightGBM': 0.257613, 'SARIMA': 0.259656}}}` |
| `1D` | 5 | `Val_RMSE` | 0.00872 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.249462, 'SVR': 0.241028, 'LightGBM': 0.252807, 'SARIMA': 0.256703}, 'lambda_shrinkage': {'Low': 0.9892, 'Normal': 0.9711, 'High': 0.9762}, 'final_regime_weights': {'Low': {'Ridge': 0.241976, 'SVR': 0.239936, 'LightGBM': 0.257267, 'SARIMA': 0.260822}, 'Normal': {'Ridge': 0.255126, 'SVR': 0.246888, 'LightGBM': 0.253119, 'SARIMA': 0.244867}, 'High': {'Ridge': 0.25109, 'SVR': 0.23948, 'LightGBM': 0.250525, 'SARIMA': 0.258904}}}` |
| `5D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c2_cold_start_prior', 'weights': {'Low': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'Normal': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'High': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}, 'N0': 50.0}` |
| `5D` | 2 | `Val_RMSE` | 0.022808 | `{'mode': 'c2_optuna_shrunk', 'N0': 27.827001341378086, 'power': 2.998153048811181, 'base_b2_weights': {'Ridge': 0.292872, 'SVR': 0.291202, 'LightGBM': 0.280755, 'SARIMA': 0.13517}, 'lambda_shrinkage': {'Low': 0.7371, 'Normal': 0.7555, 'High': 0.7951}, 'final_regime_weights': {'Low': {'Ridge': 0.272869, 'SVR': 0.26672, 'LightGBM': 0.281162, 'SARIMA': 0.179249}, 'Normal': {'Ridge': 0.29031, 'SVR': 0.317854, 'LightGBM': 0.287238, 'SARIMA': 0.104598}, 'High': {'Ridge': 0.304992, 'SVR': 0.298082, 'LightGBM': 0.274567, 'SARIMA': 0.122358}}}` |
| `5D` | 3 | `Val_RMSE` | 0.017622 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.363748, 'SVR': 0.119615, 'LightGBM': 0.35306, 'SARIMA': 0.163576}, 'lambda_shrinkage': {'Low': 0.9795, 'Normal': 0.9452, 'High': 0.9479}, 'final_regime_weights': {'Low': {'Ridge': 0.362434, 'SVR': 0.04822, 'LightGBM': 0.373496, 'SARIMA': 0.21585}, 'Normal': {'Ridge': 0.326938, 'SVR': 0.233877, 'LightGBM': 0.331127, 'SARIMA': 0.108058}, 'High': {'Ridge': 0.315614, 'SVR': 0.275377, 'LightGBM': 0.285698, 'SARIMA': 0.123311}}}` |
| `5D` | 4 | `Val_RMSE` | 0.017204 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.347706, 'SVR': 0.15425, 'LightGBM': 0.335444, 'SARIMA': 0.1626}, 'lambda_shrinkage': {'Low': 0.9872, 'Normal': 0.9565, 'High': 0.9625}, 'final_regime_weights': {'Low': {'Ridge': 0.343028, 'SVR': 0.080148, 'LightGBM': 0.340539, 'SARIMA': 0.236284}, 'Normal': {'Ridge': 0.304076, 'SVR': 0.238223, 'LightGBM': 0.297724, 'SARIMA': 0.159977}, 'High': {'Ridge': 0.315807, 'SVR': 0.299999, 'LightGBM': 0.291812, 'SARIMA': 0.092382}}}` |
| `5D` | 5 | `Val_RMSE` | 0.018693 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.342202, 'SVR': 0.15598, 'LightGBM': 0.332488, 'SARIMA': 0.169331}, 'lambda_shrinkage': {'Low': 0.9892, 'Normal': 0.9711, 'High': 0.9762}, 'final_regime_weights': {'Low': {'Ridge': 0.327862, 'SVR': 0.097061, 'LightGBM': 0.331036, 'SARIMA': 0.244041}, 'Normal': {'Ridge': 0.358434, 'SVR': 0.127947, 'LightGBM': 0.368785, 'SARIMA': 0.144834}, 'High': {'Ridge': 0.323783, 'SVR': 0.250521, 'LightGBM': 0.298791, 'SARIMA': 0.126905}}}` |
| `20D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c2_cold_start_prior', 'weights': {'Low': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'Normal': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'High': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}, 'N0': 50.0}` |
| `20D` | 2 | `Val_RMSE` | 0.040643 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.312928, 'SVR': 0.307853, 'LightGBM': 0.111361, 'SARIMA': 0.267858}, 'lambda_shrinkage': {'Low': 0.9226, 'Normal': 0.9293, 'High': 0.9429}, 'final_regime_weights': {'Low': {'Ridge': 0.30752, 'SVR': 0.332957, 'LightGBM': 0.188059, 'SARIMA': 0.171464}, 'Normal': {'Ridge': 0.289929, 'SVR': 0.306517, 'LightGBM': 0.194034, 'SARIMA': 0.20952}, 'High': {'Ridge': 0.309325, 'SVR': 0.288368, 'LightGBM': 0.078421, 'SARIMA': 0.323885}}}` |
| `20D` | 3 | `Val_RMSE` | 0.031375 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.343059, 'SVR': 0.239652, 'LightGBM': 0.136395, 'SARIMA': 0.280894}, 'lambda_shrinkage': {'Low': 0.9795, 'Normal': 0.9452, 'High': 0.9479}, 'final_regime_weights': {'Low': {'Ridge': 0.393591, 'SVR': 0.126796, 'LightGBM': 0.269766, 'SARIMA': 0.209847}, 'Normal': {'Ridge': 0.292613, 'SVR': 0.300309, 'LightGBM': 0.193468, 'SARIMA': 0.213611}, 'High': {'Ridge': 0.307789, 'SVR': 0.276457, 'LightGBM': 0.081815, 'SARIMA': 0.333938}}}` |
| `20D` | 4 | `Val_RMSE` | 0.029942 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.347155, 'SVR': 0.197961, 'LightGBM': 0.175522, 'SARIMA': 0.279362}, 'lambda_shrinkage': {'Low': 0.9872, 'Normal': 0.9565, 'High': 0.9625}, 'final_regime_weights': {'Low': {'Ridge': 0.357504, 'SVR': 0.098811, 'LightGBM': 0.303502, 'SARIMA': 0.240183}, 'Normal': {'Ridge': 0.288027, 'SVR': 0.278578, 'LightGBM': 0.221152, 'SARIMA': 0.212243}, 'High': {'Ridge': 0.314467, 'SVR': 0.281105, 'LightGBM': 0.0955, 'SARIMA': 0.308929}}}` |
| `20D` | 5 | `Val_RMSE` | 0.034241 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.296513, 'SVR': 0.236991, 'LightGBM': 0.222095, 'SARIMA': 0.244401}, 'lambda_shrinkage': {'Low': 0.9892, 'Normal': 0.9711, 'High': 0.9762}, 'final_regime_weights': {'Low': {'Ridge': 0.290556, 'SVR': 0.173265, 'LightGBM': 0.303882, 'SARIMA': 0.232297}, 'Normal': {'Ridge': 0.292427, 'SVR': 0.225706, 'LightGBM': 0.270531, 'SARIMA': 0.211336}, 'High': {'Ridge': 0.287864, 'SVR': 0.295623, 'LightGBM': 0.161747, 'SARIMA': 0.254767}}}` |
| `60D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c2_cold_start_prior', 'weights': {'Low': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'Normal': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'High': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}, 'N0': 50.0}` |
| `60D` | 2 | `Val_RMSE` | 0.058892 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.534766, 'SVR': 0.118956, 'LightGBM': 0.041134, 'SARIMA': 0.305145}, 'lambda_shrinkage': {'Low': 0.9226, 'Normal': 0.9293, 'High': 0.9429}, 'final_regime_weights': {'Low': {'Ridge': 0.349314, 'SVR': 0.08859, 'LightGBM': 0.161301, 'SARIMA': 0.400794}, 'Normal': {'Ridge': 0.355954, 'SVR': 0.281737, 'LightGBM': 0.244562, 'SARIMA': 0.117747}, 'High': {'Ridge': 0.519219, 'SVR': 0.058936, 'LightGBM': 0.012992, 'SARIMA': 0.408853}}}` |
| `60D` | 3 | `Val_RMSE` | 0.046643 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.45623, 'SVR': 0.160114, 'LightGBM': 0.059482, 'SARIMA': 0.324173}, 'lambda_shrinkage': {'Low': 0.9795, 'Normal': 0.9452, 'High': 0.9479}, 'final_regime_weights': {'Low': {'Ridge': 0.244185, 'SVR': 0.203263, 'LightGBM': 0.246426, 'SARIMA': 0.306127}, 'Normal': {'Ridge': 0.292315, 'SVR': 0.312611, 'LightGBM': 0.264904, 'SARIMA': 0.13017}, 'High': {'Ridge': 0.52048, 'SVR': 0.06246, 'LightGBM': 0.014111, 'SARIMA': 0.402949}}}` |
| `60D` | 4 | `Val_RMSE` | 0.048015 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.337247, 'SVR': 0.221154, 'LightGBM': 0.085699, 'SARIMA': 0.3559}, 'lambda_shrinkage': {'Low': 0.9872, 'Normal': 0.9565, 'High': 0.9625}, 'final_regime_weights': {'Low': {'Ridge': 0.212313, 'SVR': 0.243172, 'LightGBM': 0.268555, 'SARIMA': 0.27596}, 'Normal': {'Ridge': 0.253669, 'SVR': 0.38457, 'LightGBM': 0.193552, 'SARIMA': 0.168208}, 'High': {'Ridge': 0.407771, 'SVR': 0.087331, 'LightGBM': 0.0172, 'SARIMA': 0.487698}}}` |
| `60D` | 5 | `Val_RMSE` | 0.055362 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.276451, 'SVR': 0.284612, 'LightGBM': 0.11997, 'SARIMA': 0.318967}, 'lambda_shrinkage': {'Low': 0.9892, 'Normal': 0.9711, 'High': 0.9762}, 'final_regime_weights': {'Low': {'Ridge': 0.22989, 'SVR': 0.250806, 'LightGBM': 0.273, 'SARIMA': 0.246304}, 'Normal': {'Ridge': 0.226077, 'SVR': 0.391163, 'LightGBM': 0.188284, 'SARIMA': 0.194476}, 'High': {'Ridge': 0.293118, 'SVR': 0.228463, 'LightGBM': 0.044551, 'SARIMA': 0.433867}}}` |
| `120D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c2_cold_start_prior', 'weights': {'Low': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'Normal': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}, 'High': {'Ridge': 0.25, 'SVR': 0.25, 'LightGBM': 0.25, 'SARIMA': 0.25}}, 'N0': 50.0}` |
| `120D` | 2 | `Val_RMSE` | 0.064461 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.717091, 'SVR': 0.05628, 'LightGBM': 0.010971, 'SARIMA': 0.215659}, 'lambda_shrinkage': {'Low': 0.9226, 'Normal': 0.9293, 'High': 0.9429}, 'final_regime_weights': {'Low': {'Ridge': 0.805881, 'SVR': 0.04293, 'LightGBM': 0.062668, 'SARIMA': 0.088522}, 'Normal': {'Ridge': 0.816957, 'SVR': 0.0622, 'LightGBM': 0.015791, 'SARIMA': 0.105052}, 'High': {'Ridge': 0.534606, 'SVR': 0.047873, 'LightGBM': 0.005587, 'SARIMA': 0.411934}}}` |
| `120D` | 3 | `Val_RMSE` | 0.057427 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.371685, 'SVR': 0.137026, 'LightGBM': 0.03308, 'SARIMA': 0.458209}, 'lambda_shrinkage': {'Low': 0.9795, 'Normal': 0.9452, 'High': 0.9479}, 'final_regime_weights': {'Low': {'Ridge': 0.135224, 'SVR': 0.227183, 'LightGBM': 0.253839, 'SARIMA': 0.383754}, 'Normal': {'Ridge': 0.529812, 'SVR': 0.131452, 'LightGBM': 0.039685, 'SARIMA': 0.299052}, 'High': {'Ridge': 0.527337, 'SVR': 0.05905, 'LightGBM': 0.007728, 'SARIMA': 0.405885}}}` |
| `120D` | 4 | `Val_RMSE` | 0.058351 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.325307, 'SVR': 0.135279, 'LightGBM': 0.043039, 'SARIMA': 0.496375}, 'lambda_shrinkage': {'Low': 0.9872, 'Normal': 0.9565, 'High': 0.9625}, 'final_regime_weights': {'Low': {'Ridge': 0.179216, 'SVR': 0.269697, 'LightGBM': 0.185247, 'SARIMA': 0.36584}, 'Normal': {'Ridge': 0.541934, 'SVR': 0.104092, 'LightGBM': 0.040156, 'SARIMA': 0.313818}, 'High': {'Ridge': 0.363851, 'SVR': 0.045608, 'LightGBM': 0.008837, 'SARIMA': 0.581703}}}` |
| `120D` | 5 | `Val_RMSE` | 0.064484 | `{'mode': 'c2_optuna_shrunk', 'N0': 6.54584837750234, 'power': 2.964901710106421, 'base_b2_weights': {'Ridge': 0.280934, 'SVR': 0.123917, 'LightGBM': 0.047327, 'SARIMA': 0.547822}, 'lambda_shrinkage': {'Low': 0.9892, 'Normal': 0.9711, 'High': 0.9762}, 'final_regime_weights': {'Low': {'Ridge': 0.20368, 'SVR': 0.257026, 'LightGBM': 0.191875, 'SARIMA': 0.347419}, 'Normal': {'Ridge': 0.447698, 'SVR': 0.112688, 'LightGBM': 0.062383, 'SARIMA': 0.377231}, 'High': {'Ridge': 0.208598, 'SVR': 0.045363, 'LightGBM': 0.011109, 'SARIMA': 0.73493}}}` |

</details>


<details>
<summary>Baseline_C3 — 25 bản ghi chân trời/fold</summary>

| Chân trời | Fold | Thước đo | Điểm | Tham số được ghi nhận |
| :---: | :---: | :--- | ---: | :--- |
| `1D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c3_default_defensive_switch', 'rule': 'Low/Normal -> A2 (Core 4 Equal), High -> Historical Mean', 'high_defensive_weight': 1.0}` |
| `1D` | 2 | `Val_RMSE` | 0.01061 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9120918418652022, 'low_normal_core_weights': {'Ridge': 0.01625, 'SVR': 0.01682, 'LightGBM': 0.824354, 'SARIMA': 0.142577}}` |
| `1D` | 3 | `Val_RMSE` | 0.008271 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9463765598445045, 'low_normal_core_weights': {'Ridge': 0.007465, 'SVR': 0.028741, 'LightGBM': 0.504223, 'SARIMA': 0.459572}}` |
| `1D` | 4 | `Val_RMSE` | 0.008101 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.4862327641500984, 'low_normal_core_weights': {'Ridge': 0.025288, 'SVR': 0.004135, 'LightGBM': 0.487029, 'SARIMA': 0.483549}}` |
| `1D` | 5 | `Val_RMSE` | 0.008693 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.609598013624639, 'low_normal_core_weights': {'Ridge': 0.038403, 'SVR': 0.005898, 'LightGBM': 0.40204, 'SARIMA': 0.55366}}` |
| `5D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c3_default_defensive_switch', 'rule': 'Low/Normal -> A2 (Core 4 Equal), High -> Historical Mean', 'high_defensive_weight': 1.0}` |
| `5D` | 2 | `Val_RMSE` | 0.022688 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9870508608135671, 'low_normal_core_weights': {'Ridge': 0.230963, 'SVR': 0.180492, 'LightGBM': 0.578406, 'SARIMA': 0.010139}}` |
| `5D` | 3 | `Val_RMSE` | 0.017452 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.8815738415222171, 'low_normal_core_weights': {'Ridge': 0.430244, 'SVR': 0.014597, 'LightGBM': 0.542736, 'SARIMA': 0.012424}}` |
| `5D` | 4 | `Val_RMSE` | 0.017116 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.7593677509368647, 'low_normal_core_weights': {'Ridge': 0.667302, 'SVR': 0.016339, 'LightGBM': 0.314304, 'SARIMA': 0.002055}}` |
| `5D` | 5 | `Val_RMSE` | 0.018587 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.7743721161809601, 'low_normal_core_weights': {'Ridge': 0.524715, 'SVR': 0.032726, 'LightGBM': 0.434239, 'SARIMA': 0.00832}}` |
| `20D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c3_default_defensive_switch', 'rule': 'Low/Normal -> A2 (Core 4 Equal), High -> Historical Mean', 'high_defensive_weight': 1.0}` |
| `20D` | 2 | `Val_RMSE` | 0.039253 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9969132241136197, 'low_normal_core_weights': {'Ridge': 0.255128, 'SVR': 0.594157, 'LightGBM': 0.030851, 'SARIMA': 0.119864}}` |
| `20D` | 3 | `Val_RMSE` | 0.030568 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9926078348299993, 'low_normal_core_weights': {'Ridge': 0.443479, 'SVR': 0.146062, 'LightGBM': 0.048632, 'SARIMA': 0.361828}}` |
| `20D` | 4 | `Val_RMSE` | 0.029273 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9907684297438037, 'low_normal_core_weights': {'Ridge': 0.481894, 'SVR': 0.006076, 'LightGBM': 0.420068, 'SARIMA': 0.091962}}` |
| `20D` | 5 | `Val_RMSE` | 0.03345 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9873195973015957, 'low_normal_core_weights': {'Ridge': 0.496462, 'SVR': 0.078483, 'LightGBM': 0.350674, 'SARIMA': 0.074382}}` |
| `60D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c3_default_defensive_switch', 'rule': 'Low/Normal -> A2 (Core 4 Equal), High -> Historical Mean', 'high_defensive_weight': 1.0}` |
| `60D` | 2 | `Val_RMSE` | 0.054449 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9897875253587012, 'low_normal_core_weights': {'Ridge': 0.347223, 'SVR': 0.096692, 'LightGBM': 0.423949, 'SARIMA': 0.132136}}` |
| `60D` | 3 | `Val_RMSE` | 0.043935 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.997792903644071, 'low_normal_core_weights': {'Ridge': 0.086337, 'SVR': 0.428396, 'LightGBM': 0.233562, 'SARIMA': 0.251706}}` |
| `60D` | 4 | `Val_RMSE` | 0.045227 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9977427718114545, 'low_normal_core_weights': {'Ridge': 0.041537, 'SVR': 0.419237, 'LightGBM': 0.260532, 'SARIMA': 0.278694}}` |
| `60D` | 5 | `Val_RMSE` | 0.051289 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9899116320817142, 'low_normal_core_weights': {'Ridge': 0.168776, 'SVR': 0.540466, 'LightGBM': 0.15663, 'SARIMA': 0.134128}}` |
| `120D` | 1 | `Val_RMSE` | — (khởi tạo) | `{'mode': 'c3_default_defensive_switch', 'rule': 'Low/Normal -> A2 (Core 4 Equal), High -> Historical Mean', 'high_defensive_weight': 1.0}` |
| `120D` | 2 | `Val_RMSE` | 0.06227 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9490657522107996, 'low_normal_core_weights': {'Ridge': 0.451588, 'SVR': 0.03883, 'LightGBM': 0.060064, 'SARIMA': 0.449518}}` |
| `120D` | 3 | `Val_RMSE` | 0.054585 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.970031104812115, 'low_normal_core_weights': {'Ridge': 0.246504, 'SVR': 0.354128, 'LightGBM': 0.012445, 'SARIMA': 0.386922}}` |
| `120D` | 4 | `Val_RMSE` | 0.055458 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9722489743736693, 'low_normal_core_weights': {'Ridge': 0.229085, 'SVR': 0.358427, 'LightGBM': 0.018651, 'SARIMA': 0.393837}}` |
| `120D` | 5 | `Val_RMSE` | 0.061855 | `{'mode': 'c3_optuna_defensive_switch', 'best_alpha_high_defensive': 0.9934961721163775, 'low_normal_core_weights': {'Ridge': 0.17647, 'SVR': 0.360538, 'LightGBM': 0.047083, 'SARIMA': 0.41591}}` |

</details>


---

## 4. GIẢI ĐÁP: CÁC FILE KẾT QUẢ SAU HUẤN LUYỆN CÓ ĐƯỢC PUSH LÊN GITHUB KHÔNG?

> [!IMPORTANT]
> **CÓ, toàn bộ tệp kết quả nghiên cứu khoa học (CSVs, metrics, predictions, summary, logs, markdown reports, dashboard) ĐỀU ĐƯỢC PUSH lên GitHub đầy đủ!**  
> **CHỈ CÓ các file trọng số nhị phân siêu nặng (`.pt`, `.pth`, `.pkl`, `.h5`, `.joblib`) là KHÔNG PUSH lên GitHub.**

### Bảng kiểm kê chính sách Git theo `.gitignore` của Repository:

| Loại Tệp / Thư Mục | Trạng Thái Trên GitHub | Minh Chứng Cụ Thể |
| :--- | :---: | :--- |
| **Dữ liệu dự báo ngoại mẫu** (`results/*/predictions/*.csv`) |  **ĐƯỢC PUSH** | Lan01 đã push 95,200 dòng dự báo OOS đầy đủ. |
| **Chỉ số đánh giá từng Fold** (`results/*/fold_metrics/*.csv`) |  **ĐƯỢC PUSH** | Lan01 đã push 350 folds kiểm định ngoại mẫu ($R^2$, DA, RMSE, DM test). |
| **Tóm tắt hiệu năng mô hình** (`results/*/summary/*.csv`) |  **ĐƯỢC PUSH** | Bảng tổng hợp Mean & Std theo từng chân trời. |
| **Nhật ký siêu tham số** (`results/*/tuning/*.csv`) |  **ĐƯỢC PUSH** | Đã push 325 dòng lịch sử tham số tốt nhất của 13 mô hình. |
| **Báo cáo nghiên cứu khoa học** (`results/*/agent_reports/*.md`) |  **ĐƯỢC PUSH** | Toàn bộ 13 báo cáo tổng hợp giải đáp RQ1, RQ2, RQ3. |
| **Bảng điều khiển trực quan** (`dashboard.html`, `DASHBOARD.md`) |  **ĐƯỢC PUSH** | Dashboard giám sát tiến trình thời gian thực. |
| **File trọng số nhị phân mô hình** (`*.pt`, `*.pth`, `*.pkl`, `*.h5`) | ❌ **BỊ CHẶN (.gitignore)** | Mạng nơ-ron nhị phân dung lượng từ vài trăm MB đến hàng GB, vượt ngưỡng 100MB của GitHub. Chặn binary weights là best practice chuẩn quốc tế. |
| **Tệp rác / cache trung gian** (`__pycache__`, `latest_metrics.csv`, `.env`) | ❌ **BỊ CHẶN (.gitignore)** | Tránh xung đột môi trường và giữ kho mã nguồn gọn gàng. |