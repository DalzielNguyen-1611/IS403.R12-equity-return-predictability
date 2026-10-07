# Support Vector Regression (SVR)

## 1. Phân loại & Vai trò
- **Nhóm:** `Kernel-based`
- **Mô hình:** Hồi quy vector hỗ trợ nhân RBF phi tuyến (Support Vector Regression with Radial Basis Function kernel).
- **Mục tiêu:** Nắm bắt mối quan hệ phi tuyến giữa biến động tài chính và lợi suất tương lai.

## 2. Siêu tham số & Tuning
- Hàm nhân: RBF ($\exp(-\gamma \|x - x'\|^2)$)
- Siêu tham số: $C \in [0.1, 1.0, 10.0]$, $\gamma \in [\text{'scale'}, 0.01, 0.1]$, $\epsilon \in [0.01, 0.1]$.
- Tune hoàn toàn bằng `TimeSeriesSplit` trên dữ liệu Train của từng Fold.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/kernel_based/svr/svr.py
```
