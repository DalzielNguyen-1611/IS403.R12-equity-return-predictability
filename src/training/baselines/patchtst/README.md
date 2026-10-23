# PatchTST (Patch Time Series Transformer)

## 1. Phân loại & Vai trò
- **Nhóm:** `Deep Learning (Patch Transformer)`
- **Mô hình:** A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (Nie et al., ICLR 2023).
- **Mục tiêu:** Chia chuỗi thời gian liên tục thành các mẩu cục bộ (subseries patches), giúp bảo toàn cấu trúc chuỗi cục bộ và giảm độ phức tạp tính toán của cơ chế Self-Attention từ $O(L^2)$ xuống $O(N_p^2)$. Kết hợp Channel Independence (CI) ngăn ngừa hiện tượng quá khớp giữa các kênh đặc trưng.

## 2. Kiến trúc & Siêu tham số
- **Subseries Patching:** Chiều dài patch $P = 8$, bước trượt $S = 4$.
- **Channel Independence (CI):** 16 biến đặc trưng được xử lý độc lập qua chung một Transformer Encoder backbone.
- **Siêu tham số cơ sở:**
  - `patch_len`: 8, `stride`: 4
  - `d_model`: 32, `nhead`: 2
  - `num_layers`: 1, `dim_feedforward`: 64
  - `lookback`: 20 ngày
  - `dropout`: 0.1
  - `learning_rate`: 1e-4

## 3. Cách chạy độc lập
```powershell
$env:EXP_RUN_NAME="Lan02"; python src/training/baselines/patchtst/patchtst_model.py
```
