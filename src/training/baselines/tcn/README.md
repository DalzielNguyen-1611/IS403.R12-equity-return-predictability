# Temporal Convolutional Network (TCN)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning (Temporal CNN)`
- **Mô hình:** Temporal Convolutional Network (Bai, Kolter, Koltun 2018).
- **Mục tiêu:** Mô hình hóa các quan hệ thời gian dài hạn bằng các lớp Causal Dilated Convolutions mở rộng theo cấp số nhân ($d = 1, 2, 4, 8$), thay thế RNN/LSTM tuần tự với khả năng tính toán song song hoàn toàn trên GPU.

## 2. Kiến trúc & Siêu tham số
- **Causal Convolutions:** Đảm bảo không rò rỉ thông tin tương lai (strictly causal chomp).
- **Dilated Residual Blocks:** Kết hợp Dilated Conv1d + BatchNorm1d + GELU + Dropout + 1x1 Residual Skip Connection.
- **Siêu tham số cơ sở:**
  - `num_channels`: [32, 64, 64]
  - `kernel_size`: 3
  - `lookback`: 20 ngày
  - `dropout`: 0.1
  - `learning_rate`: 1e-4

## 3. Cách chạy độc lập
```powershell
$env:EXP_RUN_NAME="Lan02"; python src/training/baselines/tcn/tcn_model.py
```
