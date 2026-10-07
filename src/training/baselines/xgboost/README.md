# XGBoost Regression

## 1. Phân loại & Vai trò
- **Nhóm:** `Tree-based / Ensemble`
- **Mô hình:** Tăng cường độ dốc cực đại (Extreme Gradient Boosting).
- **Mục tiêu:** Mô hình học máy dựa trên cây phổ biến nhất cho dữ liệu bảng tài chính (tabular financial data).

## 2. Cấu hình mô hình
- `n_estimators`: 300, `max_depth`: 4, `learning_rate`: 0.03.
- `subsample`: 0.8, `colsample_bytree`: 0.8, `tree_method`: 'hist'.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/tree_based/xgboost/xgboost_model.py
```
