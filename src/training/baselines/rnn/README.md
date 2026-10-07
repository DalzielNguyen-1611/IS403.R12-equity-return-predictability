# Vanilla Recurrent Neural Network (RNN)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning`
- **Mô hình:** Mạng nơ-ron hồi quy đơn giản (Vanilla SimpleRNN in PyTorch).
- **Mục tiêu:** Mô hình hóa sự phụ thuộc thời gian qua cửa sổ nhìn lùi tuần tự (`lookback = 20` ngày).

## 2. Kiến trúc & Huấn luyện
- Cấu trúc: `Input(20, 16) -> SimpleRNN(hidden=64, tanh) -> Dropout(0.1) -> Linear(1)`.
- Thiết bị: Hỗ trợ tự động GPU NVIDIA CUDA và CPU.
- Huấn luyện: Early stopping theo dõi validation loss (patience = 7) độc lập cho từng Fold.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/deep_learning/rnn/rnn_model.py
```
