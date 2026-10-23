# N-HiTS (Neural Hierarchical Interpolation for Time Series)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning (Hierarchical Interpolation / MLP)`
- **Mô hình:** Neural Hierarchical Interpolation for Time Series Forecasting (Challu et al., AAAI 2023).
- **Mục tiêu:** Mở rộng N-BEATS bằng cơ chế gom nhóm phân cấp đa tần suất (Multi-rate Hierarchical Pooling). Tầng tần số thấp bắt giữ xu hướng vĩ mô/dài hạn, trong khi tầng tần số cao bắt giữ biến động nhiễu ngắn hạn, kết hợp nội suy tuyến tính (hierarchical interpolation synthesis).

## 2. Kiến trúc & Siêu tham số
- **Multi-rate Hierarchical Pooling:** Kernel gom nhóm AvgPool1d [8, 4, 1] hoặc [4, 2, 1].
- **Cơ chế Doubly Residual:** Tương tự N-BEATS nhưng khử nhiễu đa tầng.
- **Siêu tham số cơ sở:**
  - `pooling_kernels`: [4, 2, 1]
  - `num_blocks_per_stack`: 1
  - `hidden_dim`: 128
  - `lookback`: 20 ngày
  - `dropout`: 0.1
  - `learning_rate`: 1e-4

## 3. Cách chạy độc lập
```powershell
$env:EXP_RUN_NAME="Lan02"; python src/training/baselines/nhits/nhits_model.py
```
