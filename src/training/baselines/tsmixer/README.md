# TSMixer (Time-Series Mixer)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning (All-MLP Architecture)`
- **Mô hình:** TSMixer: An All-MLP Architecture for Time Series Forecasting (Chen et al., Google Research, TMLR 2023).
- **Mục tiêu:** Thay thế hoàn toàn cơ chế Attention tốn kém bằng kiến trúc thuần MLP luân phiên giữa Time-Mixing (hòa trộn tương quan theo thời gian) và Feature-Mixing (hòa trộn tương quan giữa 16 đặc trưng tài chính).

## 2. Kiến trúc & Siêu tham số
- **Time-Mixing MLP:** Linear layer theo chiều thời gian + GELU + Dropout + LayerNorm + Residual.
- **Feature-Mixing MLP:** Linear layer theo chiều kênh biến + GELU + Dropout + LayerNorm + Residual.
- **Siêu tham số cơ sở:**
  - `num_blocks`: 3
  - `ff_dim`: 64
  - `lookback`: 20 ngày
  - `dropout`: 0.1
  - `learning_rate`: 1e-4

## 3. Cách chạy độc lập
```powershell
$env:EXP_RUN_NAME="Lan02"; python src/training/baselines/tsmixer/tsmixer_model.py
```
