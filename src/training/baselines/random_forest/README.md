# Random Forest Regression

## 1. Phân loại & Vai trò
- **Nhóm:** `Tree-based / Ensemble`
- **Mô hình:** Rừng ngẫu nhiên gồm các cây quyết định độc lập (Bagging Ensemble).
- **Mục tiêu:** Giảm phương sai (variance reduction) và xử lý tương tác phi tuyến tính bậc cao.

## 2. Cấu hình mô hình
- `n_estimators`: 300 cây.
- `max_depth`: 10, `min_samples_leaf`: 10.
- `max_features`: 'sqrt' (lấy căn bậc 2 số lượng features để chống tương quan giữa các cây).
- Không yêu cầu chuẩn hóa dữ liệu.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/tree_based/random_forest/random_forest.py
```
