# 1D Temporal CNN (Convolutional Neural Network)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning (Standard Temporal CNN)`
- **Mô hình:** Multi-layer 1D Temporal Convolutional Neural Network.
- **Mục tiêu:** Cung cấp mô hình nền tảng đối chứng (baseline) chuẩn cho nhóm tích chập thời gian 1 chiều, làm cơ sở so sánh với mô hình tích chập nhân quả giãn nở (Causal Dilated TCN).

## 2. Kiến trúc & Siêu tham số
- **1D Temporal Convolutions:** Lớp tích chập 1D theo trục thời gian + BatchNorm1d + GELU + Dropout.
- **Adaptive Average Pooling:** Thu gọn không gian đặc trưng trước khi đưa vào MLP head.
- **Siêu tham số cơ sở:**
  - `num_filters`: 64
  - `kernel_size`: 3
  - `num_layers`: 2
  - `lookback`: 20 ngày
  - `dropout`: 0.1
  - `learning_rate`: 1e-4

## 3. Cách chạy độc lập
```powershell
$env:EXP_RUN_NAME="Lan02"; python src/training/baselines/cnn/cnn_model.py
```
