# Logistic Regression Direction Classifier

## 1. Phân loại & Vai trò
- **Nhóm:** `Classification Side-task`
- **Mô hình:** Hồi quy logistic phân loại nhị phân chiều biến động lợi suất:
  $$Direction_{h,t} = \mathbb{I}(Target_{h,t} > 0)$$
- **Mục tiêu:** Cung cấp góc nhìn phân loại (Accuracy, Precision, Recall, F1, ROC-AUC) bổ trợ cho bài toán hồi quy liên tục.

## 2. Đầu ra
- Dự báo cả nhãn cứng (`Predicted_Class` 0/1) và xác suất mềm (`Predicted_Probability`) phục vụ tính ROC-AUC.

## 3. Cách chạy độc lập
```bash
python src/training/baselines/classification/logistic_regression/logistic_regression.py
```
