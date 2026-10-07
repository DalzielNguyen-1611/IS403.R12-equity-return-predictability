# Historical Mean Benchmark

## 1. Phân loại & Vai trò
- **Nhóm:** `Benchmark`
- **Vai trò:** Chuẩn đối sánh cơ sở không thiên kiến (Unbiased Baseline). Đại diện cho giả thuyết thị trường hiệu quả yếu (No-predictability hypothesis).

## 2. Công thức toán học
Tại mỗi phiên giao dịch $t$ trong tập Validation:
$$\hat{y}_t = \frac{1}{N_{t-1}} \sum_{\tau < t} y_\tau$$
Trong đó chỉ sử dụng các giá trị lợi suất mục tiêu trong quá khứ $\tau < t$.

## 3. Cấu trúc thư mục
- `historical_mean.py`: Mã nguồn thực thi mô hình.
- `README.md`: Tài liệu mô tả và hướng dẫn.
- `latest_metrics.csv`: Bảng số liệu MAE, RMSE, R2, Directional Accuracy qua 5 folds.
- `latest_predictions.csv`: Giá trị dự báo Out-of-Sample theo từng ngày kèm nhãn `Regime`.

## 4. Cách chạy độc lập
```bash
python src/training/baselines/benchmark/historical_mean/historical_mean.py
```
