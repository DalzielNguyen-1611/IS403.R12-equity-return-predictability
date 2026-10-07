# Time-Series Transformer Encoder

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning`
- **Mô hình:** Kiến trúc chú ý đa đầu (Multi-Head Self-Attention Transformer Encoder).
- **Mục tiêu:** Nắm bắt tương quan chéo giữa các mốc thời gian không phụ thuộc vào khoảng cách tuần tự.

## 2. Kiến trúc & Siêu tham số
- Chiều không gian ẩn (`d_model`): 64, Số đầu chú ý (`nhead`): 4.
- Lớp Transformer Encoder: 2 layers, Feedforward dimension: 128, Dropout: 0.1.
- Positional Encoding: Học được (learnable parameter) trên cửa sổ 20 ngày.
- Thiết bị: Hỗ trợ tự động GPU NVIDIA CUDA và CPU.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/deep_learning/transformer/transformer_model.py
```
