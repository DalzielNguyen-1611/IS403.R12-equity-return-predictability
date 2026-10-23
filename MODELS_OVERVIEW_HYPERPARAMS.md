# 📋 BẢNG TỔNG HỢP TOÀN BỘ MÔ HÌNH VÀ SIÊU THAM SỐ / TRỌNG SỐ TỐI ƯU HÓA

> **Dự án nghiên cứu:** *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*  
> **Mục tiêu tệp:** Danh mục tổng hợp toàn bộ 40 đối tượng thực nghiệm qua 5 đợt huấn luyện (Lan01 → Lan05), hiển thị chi tiết siêu tham số/trọng số tốt nhất đã tìm qua Optuna. Các mô hình **chưa Optuna** được để trống theo đúng yêu cầu.

---

## 1. BẢNG OVERVIEW TỔNG THỂ (MASTER MODEL & HYPERPARAMETER MATRIX)

| STT | Đợt Chạy | Tên Mô Hình | Họ Kiến Trúc | Phương Pháp Tuning | Số Trials | Trạng Thái Optuna | Siêu Tham Số / Trọng Số Tốt Nhất (Optuna) | Điểm Thẩm Định Tốt Nhất (Val Loss) |
| :-: | :---: | :--- | :--- | :--- | :-: | :--- | :--- | :-: |
| 1 | **Lan01** | `Historical_Mean` | Benchmark | None (Parameter-free) | - | Không dùng Optuna |  |  |
| 2 | **Lan01** | `Ridge` | Linear Model | Grid Search (alpha) | - | Không dùng Optuna (Grid) |  |  |
| 3 | **Lan01** | `Logistic_Regression` | Classification | Grid Search (C, class_weight) | - | Không dùng Optuna (Grid) |  |  |
| 4 | **Lan01** | `Elastic_Net` | Linear Model | Grid Search (alpha, l1_ratio) | - | Không dùng Optuna (Grid) |  |  |
| 5 | **Lan01** | `ARIMA` | Statistical TS | Grid Search (p, d, q AIC) | - | Không dùng Optuna (Grid) |  |  |
| 6 | **Lan01** | `ARIMAX` | Statistical TS | Fixed Spec (1,0,1 + Exog) | - | Không dùng Optuna (Fixed) |  |  |
| 7 | **Lan01** | `SARIMA` | Statistical TS | Fixed Spec (1,0,0)x(1,0,0,5) | - | Không dùng Optuna (Fixed) |  |  |
| 8 | **Lan01** | `SVR` | Kernel Method | Grid Search (C, gamma, eps) | - | Không dùng Optuna (Grid) |  |  |
| 9 | **Lan01** | `Random_Forest` | Tree Ensemble | Optuna TPE (CV_RMSE) | 30 | Đã hoàn tất Optuna | `{'n_estimators': 200, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt'}` | `0.008629` |
| 10 | **Lan01** | `XGBoost` | Tree Ensemble | Optuna TPE (CV_RMSE) | 50 | Đã hoàn tất Optuna | `{'n_estimators': 200, 'learning_rate': 0.09346777377493276, 'max_depth': 9, 'min_child_weight': 1, 'subsample': 0.7219192643841313, 'colsample_bytree': 0.6558289950203557, 'gamma': 0.0015032555243577606, 'reg_alpha': 0.14113536657860465, 'reg_lambda': 2.879747417492425}` | `0.008624` |
| 11 | **Lan01** | `LightGBM` | Tree Ensemble | Optuna TPE (CV_RMSE) | 50 | Đã hoàn tất Optuna | `{'n_estimators': 500, 'learning_rate': 0.017709597800863956, 'num_leaves': 56, 'max_depth': 2, 'min_child_samples': 22, 'subsample': 0.8439440638588244, 'colsample_bytree': 0.962726780479557, 'reg_alpha': 0.03669978237679096, 'reg_lambda': 3.5016679319694397}` | `0.008612` |
| 12 | **Lan01** | `RNN` | Deep Learning | Optuna TPE (Val_Loss) | 30 | Đã hoàn tất Optuna | `{'lookback': 40, 'hidden_size': 64, 'num_layers': 1, 'dropout': 0.3017249991009523, 'learning_rate': 0.000305485208547517, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` | `0.008670` |
| 13 | **Lan01** | `LSTM` | Deep Learning | Optuna TPE (Val_Loss) | 30 | Đã hoàn tất Optuna | `{'lookback': 20, 'hidden_size': 64, 'num_layers': 3, 'dropout': 0.3538584597651602, 'learning_rate': 0.0007662150068129557, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` | `0.008680` |
| 14 | **Lan01** | `Transformer` | Deep Learning | Optuna TPE (Val_Loss) | 30 | Đã hoàn tất Optuna | `{'lookback': 60, 'd_model': 32, 'nhead': 4, 'num_encoder_layers': 1, 'dim_feedforward': 128, 'dropout': 0.2833143475684623, 'learning_rate': 0.00020235315003022192, 'weight_decay': 0.0001, 'batch_size': 16, 'epochs': 40, 'patience': 7}` | `0.009152` |
| 15 | **Lan02** | `TCN` | Advanced DL (Temporal CNN) | Default / Pre-set | - | Không dùng Optuna (Mặc định) |  |  |
| 16 | **Lan02** | `N-BEATS` | Advanced DL (Basis Expansion) | Default / Pre-set | - | Không dùng Optuna (Mặc định) |  |  |
| 17 | **Lan02** | `N-HiTS` | Advanced DL (Hierarchical MLP) | Default / Pre-set | - | Không dùng Optuna (Mặc định) |  |  |
| 18 | **Lan02** | `PatchTST` | Advanced DL (Patch Transformer) | Default / Pre-set | - | Không dùng Optuna (Mặc định) |  |  |
| 19 | **Lan02** | `TSMixer` | Advanced DL (All-MLP) | Default / Pre-set | - | Không dùng Optuna (Mặc định) |  |  |
| 20 | **Lan02** | `iTransformer` | Advanced DL (Inverted Attn) | Default / Pre-set | - | Không dùng Optuna (Mặc định) |  |  |
| 21 | **Lan02** | `CNN` | Advanced DL (1D Conv Baseline) | Default / Pre-set | - | Không dùng Optuna (Mặc định) |  |  |
| 22 | **Lan03** | `TCN` | Advanced DL (Temporal CNN) | Optuna TPE (Val_Loss) | 30 | Chưa Optuna |  |  |
| 23 | **Lan03** | `N-BEATS` | Advanced DL (Basis Expansion) | Optuna TPE (Val_Loss) | 30 | Chưa Optuna |  |  |
| 24 | **Lan03** | `N-HiTS` | Advanced DL (Hierarchical MLP) | Optuna TPE (Val_Loss) | 30 | Chưa Optuna |  |  |
| 25 | **Lan03** | `PatchTST` | Advanced DL (Patch Transformer) | Optuna TPE (Val_Loss) | 30 | Chưa Optuna |  |  |
| 26 | **Lan03** | `TSMixer` | Advanced DL (All-MLP) | Optuna TPE (Val_Loss) | 30 | Chưa Optuna |  |  |
| 27 | **Lan03** | `iTransformer` | Advanced DL (Inverted Attn) | Optuna TPE (Val_Loss) | 30 | Chưa Optuna |  |  |
| 28 | **Lan03** | `CNN` | Advanced DL (1D Conv Baseline) | Optuna TPE (Val_Loss) | 30 | Chưa Optuna |  |  |
| 29 | **Lan04** | `Ensemble_A1` | Ensemble (All 13 Models) | Equal-Weight (1/13) | - | Không dùng Optuna (Mặc định) |  |  |
| 30 | **Lan04** | `Ensemble_A2` | Ensemble (Core 4 Models) | Equal-Weight (1/4) | - | Không dùng Optuna (Mặc định) |  |  |
| 31 | **Lan04** | `Ensemble_A3` | Ensemble (Ridge, SARIMA, HM) | Equal-Weight (1/3) | - | Không dùng Optuna (Mặc định) |  |  |
| 32 | **Lan04** | `Baseline_C1` | Regime Strategy (Best Model) | Argmin MSE per Regime | - | Không dùng Optuna (Mặc định) |  |  |
| 33 | **Lan04** | `Baseline_C2` | Regime Strategy (Shrunk Inv-MSE) | Shrinkage N0=50.0 fixed | - | Không dùng Optuna (Mặc định) |  |  |
| 34 | **Lan04** | `Baseline_C3` | Regime Strategy (Defensive Switch) | A2 / HM hard switch | - | Không dùng Optuna (Mặc định) |  |  |
| 35 | **Lan05** | `Ensemble_A1` | Ensemble (All 13 Models) | Optuna Simplex Weights | 50 | Chưa Optuna |  |  |
| 36 | **Lan05** | `Ensemble_A2` | Ensemble (Core 4 Models) | Optuna Simplex Weights | 50 | Chưa Optuna |  |  |
| 37 | **Lan05** | `Ensemble_A3` | Ensemble (Ridge, SARIMA, HM) | Optuna Simplex Weights | 50 | Chưa Optuna |  |  |
| 38 | **Lan05** | `Baseline_C1` | Regime Strategy (Best Model) | Optuna Regime Selection/Blend | 50 | Chưa Optuna |  |  |
| 39 | **Lan05** | `Baseline_C2` | Regime Strategy (Shrunk Inv-MSE) | Optuna N0 & Power gamma | 50 | Chưa Optuna |  |  |
| 40 | **Lan05** | `Baseline_C3` | Regime Strategy (Defensive Switch) | Optuna Alpha & Core Weights | 50 | Chưa Optuna |  |  |

---

## 2. CHI TIẾT SIÊU THAM SỐ CÁC MÔ HÌNH ĐÃ OPTUNA (LẦN 01 — LAN01)

Tại Lần 01, 6 mô hình phi tuyến và học sâu đã hoàn tất 100% quá trình tinh chỉnh Bayes thông qua Optuna TPE Sampler trên 5 chân trời và 5 Folds kiểm định ngoại mẫu:

### 🔹 Mô Hình: `Random_Forest` (Optuna 30 Trials/Fold)
| Chân trời | Fold | Điểm Val Loss (RMSE) | Bộ Siêu Tham Số Tối Ưu Nhất Được Chọn |
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

### 🔹 Mô Hình: `XGBoost` (Optuna 50 Trials/Fold)
| Chân trời | Fold | Điểm Val Loss (RMSE) | Bộ Siêu Tham Số Tối Ưu Nhất Được Chọn |
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

### 🔹 Mô Hình: `LightGBM` (Optuna 50 Trials/Fold)
| Chân trời | Fold | Điểm Val Loss (RMSE) | Bộ Siêu Tham Số Tối Ưu Nhất Được Chọn |
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

### 🔹 Mô Hình: `RNN` (Optuna 30 Trials/Fold)
| Chân trời | Fold | Điểm Val Loss (RMSE) | Bộ Siêu Tham Số Tối Ưu Nhất Được Chọn |
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

### 🔹 Mô Hình: `LSTM` (Optuna 30 Trials/Fold)
| Chân trời | Fold | Điểm Val Loss (RMSE) | Bộ Siêu Tham Số Tối Ưu Nhất Được Chọn |
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

### 🔹 Mô Hình: `Transformer` (Optuna 30 Trials/Fold)
| Chân trời | Fold | Điểm Val Loss (RMSE) | Bộ Siêu Tham Số Tối Ưu Nhất Được Chọn |
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

## 3. KHÔNG GIAN TÌM KIẾM CỦA CÁC MÔ HÌNH CHƯA OPTUNA (SẼ CHẠY Ở LẦN 03 VÀ LẦN 05)

Các mô hình dưới đây hiện tại **để trống** giá trị tham số tối ưu vì chưa bước vào phiên huấn luyện Optuna:

### 3.1 Nhóm Deep Learning Chuỗi Thời Gian Tiên Tiến (Lần 03 — Lan03)
- **Mô hình:** `TCN`, `N-BEATS`, `N-HiTS`, `PatchTST`, `TSMixer`, `iTransformer`, `CNN`.
- **Số lượng trials:** 30 trials / fold / horizon.
- **Không gian siêu tham số dự kiến:**
  - `lookback`: $[20, 40, 60]$
  - `hidden_dim` / `d_model`: $[32, 64, 128]$
  - `num_layers`: $[1, 2, 3]$
  - `dropout`: $[0.1, 0.5]$
  - `learning_rate`: $[10^{-4}, 10^{-2}]$ (log scale)
  - `batch_size`: $[16, 32, 64]$

### 3.2 Nhóm Tổ Hợp Ensemble & Chiến Lược Thích Ứng Regime (Lần 05 — Lan05)
- **Mô hình:** `Ensemble_A1`, `Ensemble_A2`, `Ensemble_A3`, `Baseline_C1`, `Baseline_C2`, `Baseline_C3`.
- **Số lượng trials:** 50 trials / fold / horizon.
- **Không gian tối ưu hóa tỷ trọng dự kiến:**
  - `A1, A2, A3`: Véc-tơ trọng số lồi trên không gian Simplex ($w_m \ge 0, \sum w_m = 1$).
  - `C1`: Tỷ trọng hòa trộn giữa các mô hình dẫn đầu theo từng chế độ biến động.
  - `C2`: Hệ số co rút James-Stein $N_0 \in [5.0, 200.0]$ và độ nhạy nghịch đảo phương sai $\gamma \in [0.5, 3.0]$.
  - `C3`: Tỷ trọng phòng thủ $\alpha_{\text{High}} \in [0, 1]$ khi thị trường High Volatility và phân bổ Core $A2$ trong Low/Normal.

---

## 4. GIẢI ĐÁP: CÁC FILE KẾT QUẢ SAU HUẤN LUYỆN CÓ ĐƯỢC PUSH LÊN GITHUB KHÔNG?

> [!IMPORTANT]
> **CÓ, toàn bộ tệp kết quả nghiên cứu khoa học (CSVs, metrics, predictions, summary, logs, markdown reports, dashboard) ĐỀU ĐƯỢC PUSH lên GitHub đầy đủ!**  
> **CHỈ CÓ các file trọng số nhị phân siêu nặng (`.pt`, `.pth`, `.pkl`, `.h5`, `.joblib`) là KHÔNG PUSH lên GitHub.**

### Chi tiết chính sách kiểm soát tệp theo `.gitignore` của Repository:

| Loại Tệp / Thư Mục | Trạng Thái Trên GitHub | Lý Do & Vai Trò Học Thuật |
| :--- | :---: | :--- |
| **Kết quả số liệu OOS Predictions** (`results/*/predictions/*.csv`) |  **ĐƯỢC PUSH** | Minh chứng dữ liệu dự báo ngoại mẫu từng ngày (Lan01 có 95,200 dòng đã push). |
| **Chỉ số đánh giá từng Fold** (`results/*/fold_metrics/*.csv`) |  **ĐƯỢC PUSH** | Bằng chứng kiểm định $R^2$, DA, RMSE, MAE, Diebold-Mariano test cho 350 folds. |
| **Tóm tắt hiệu năng mô hình** (`results/*/summary/*.csv`) |  **ĐƯỢC PUSH** | Tổng hợp trung bình và độ lệch chuẩn theo từng chân trời. |
| **Nhật ký siêu tham số** (`results/*/tuning/*.csv`) |  **ĐƯỢC PUSH** | Lưu lại 100% lịch sử Optuna và Grid Search để tái lập kết quả (Replication). |
| **Báo cáo nghiên cứu & Phân tích RQ** (`results/*/agent_reports/*.md`) |  **ĐƯỢC PUSH** | Tài liệu học thuật tổng hợp giải đáp RQ1, RQ2, RQ3. |
| **Bảng điều khiển trực quan** (`dashboard.html`, `DASHBOARD.md`) |  **ĐƯỢC PUSH** | Giao diện đồ họa giám sát thời gian thực. |
| **File trọng số nhị phân mô hình** (`*.pt`, `*.pth`, `*.pkl`, `*.h5`) | ❌ **BỊ CHẶN (.gitignore)** | Các file mạng nơ-ron nhị phân dung lượng từ vài trăm MB đến hàng GB. GitHub có giới hạn 100MB/file; việc chặn binary weights là quy chuẩn quốc tế để tránh hỏng repository. |
| **Tệp rác / cache trung gian** (`__pycache__`, `latest_metrics.csv`, `.env`) | ❌ **BỊ CHẶN (.gitignore)** | Tránh xung đột môi trường và giữ kho lưu trữ gọn gàng, bảo mật. |