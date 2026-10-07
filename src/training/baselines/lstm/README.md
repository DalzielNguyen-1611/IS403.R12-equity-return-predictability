# Long Short-Term Memory (LSTM)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning`
- **Mô hình:** Mạng nơ-ron hồi quy có cổng nhớ dài hạn (Long Short-Term Memory Network).
- **Mục tiêu:** Giải quyết triệt để vấn đề biến mất đạo hàm (vanishing gradient) khi học các chuỗi tài chính dài.

## 2. Kiến trúc & Huấn luyện
- Cấu trúc: `Input(20, 16) -> LSTM(hidden=64) -> Dropout(0.1) -> Linear(1)`.
- Thiết bị: Hỗ trợ tự động GPU NVIDIA CUDA và CPU.
- Huấn luyện: Khởi tạo lại trọng số hoàn toàn mới qua từng Fold (train from scratch).

## 3. Cách chạy độc lập
```bash
python src/training/baselines/deep_learning/lstm/lstm_model.py
```
