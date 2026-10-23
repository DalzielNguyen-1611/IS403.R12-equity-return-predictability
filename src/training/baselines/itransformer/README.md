# iTransformer (Inverted Transformer)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning (Inverted Attention Transformer)`
- **Mô hình:** iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (Liu et al., ICLR 2024 Spotlight).
- **Mục tiêu:** Đảo ngược hoàn toàn cơ chế token hóa: thay vì coi mỗi bước thời gian là một token, iTransformer coi toàn bộ chuỗi lịch sử của từng biến tài chính (trong số 16 biến) là một token riêng biệt. Cơ chế Multi-Head Self-Attention được áp dụng trực tiếp giữa các biến để học tương quan chéo đa biến (multivariate cross-feature attention).

## 2. Kiến trúc & Siêu tham số
- **Inverted Variable Tokenization:** Tuyến tính hóa chuỗi thời gian của mỗi biến thành vector chiều `d_model`.
- **Cross-Variable Attention:** Tương tác trực tiếp giữa 16 đặc trưng tài chính (VIX, Momentum, Volatility, Volume, v.v.).
- **Siêu tham số cơ sở:**
  - `d_model`: 64, `nhead`: 4
  - `num_layers`: 2, `dim_feedforward`: 128
  - `lookback`: 20 ngày
  - `dropout`: 0.1
  - `learning_rate`: 1e-4

## 3. Cách chạy độc lập
```powershell
$env:EXP_RUN_NAME="Lan02"; python src/training/baselines/itransformer/itransformer_model.py
```
