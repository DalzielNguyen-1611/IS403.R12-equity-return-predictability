# LightGBM Regression

## 1. Phân loại & Vai trò
- **Nhóm:** `Tree-based / Ensemble`
- **Mô hình:** Cây tăng cường độ dốc tốc độ cao với thuật toán phát triển lá (Leaf-wise Tree Growth).
- **Mục tiêu:** Tối ưu hóa tốc độ huấn luyện và khả năng tổng quát hóa trên tập dữ liệu bảng.

## 2. Cấu hình mô hình
- `n_estimators`: 300, `max_depth`: 5, `learning_rate`: 0.03.
- `subsample`: 0.8, `colsample_bytree`: 0.8.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/tree_based/lightgbm/lightgbm_model.py
```
