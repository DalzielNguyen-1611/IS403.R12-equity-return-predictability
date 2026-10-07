# ARIMAX Model with Exogenous Signals

## 1. Phân loại & Vai trò
- **Nhóm:** `Classical / Statistical`
- **Mô hình:** ARIMA tích hợp biến ngoại sinh (Autoregressive Integrated Moving Average with Exogenous Regressors).
- **Mục tiêu:** Kết hợp quán tính chuỗi thời gian cùng các tín hiệu vĩ mô và biến động (`VIX`, `Momentum`, `Volatility`, `Volume`).

## 2. Danh sách biến ngoại sinh (Exogenous Variables)
`VIXCLS`, `VIX_lag_1`, `VIX_change_1D`, `VIX_pct_change_1D`, `Momentum_5D`, `Momentum_20D`, `Volatility_20D`, `HL_range`, `OC_return`, `Volume_change`.
- Chuẩn hóa: Fit scaler duy nhất trên tập Train của từng Fold.
- Cấm tuyệt đối: Không dùng thông tin tương lai hay `Volatility_Regime`.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/classical_statistical/arimax/arimax.py
```
