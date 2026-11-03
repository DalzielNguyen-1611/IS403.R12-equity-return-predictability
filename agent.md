# QUY TẮC DÀNH CHO AI AGENT (AGENT RULES)

## 🚨 1. QUY TẮC BẮT BUỘC KHI LÀM VIỆC VỚI GIT (GIT RULES)
1. **Không tự động push code:** 
   - Agent **TUYỆT ĐỐI KHÔNG** được tự ý chạy lệnh `git push` lên GitHub trừ khi người dùng có yêu cầu rõ ràng.
2. **Quy trình khi push:**
   - Khi người dùng yêu cầu push, Agent phải **gợi ý commit message** phù hợp và rõ nghĩa, hoặc hỏi ý kiến để **người dùng tự nhập commit message**.
3. **Quy tắc commit message mặc định cho `README.md`:**
   - Nếu commit/push có thay đổi file `README.md` ở thư mục gốc ngoài cùng, message mặc định sẽ là:  
     👉 **`Cập nhật README`** (trừ khi người dùng chỉ định nội dung message khác).
4. **Bảo mật file Agent & Gitignore:**
   - File `agent.md` và `.gitignore` chỉ lưu trữ ở local, tuyệt đối không được track hay push lên GitHub.
5. **Quản lý và push bài tập thực hành (btth):**
   - Cho phép commit và push các thư mục, file thuộc **Bài tập thực hành (btth / Lab assignments)** lên repository khi người dùng yêu cầu.
   - **Quy tắc commit message mặc định cho btth:** Message mặc định khi commit/push bài tập thực hành sẽ là:  
     👉 **`Thêm bài tập thực hành x`** (trong đó `x` là số thứ tự của bài tập, ví dụ: `Thêm bài tập thực hành 2`, `Thêm bài tập thực hành 3`... hoặc liệt kê các số bài nếu đẩy đồng thời nhiều bài), trừ khi người dùng chỉ định nội dung message khác.
6. **Đồng bộ cấu trúc thư mục trong `README.md`:**
   - Phần cấu trúc thư mục (Mục 5) trên file `README.md` phải luôn được cập nhật chính xác và đồng bộ theo những gì thực tế đã được commit/push lên GitHub.
   - **Quy cách:** Chỉ ghi cấu trúc đến cấp thư mục (folder level), không liệt kê chi tiết file con. Riêng thư mục `btth/` chỉ ghi đến cấp `btth/`, không cần liệt kê các thư mục con bên trong.
7. **Tách biệt dữ liệu (dataset) và mã nguồn (Code):**
   - Thư mục `dataset/` chỉ chứa các file dữ liệu (`.csv`, `.parquet`). Các file dữ liệu nằm trực tiếp trong thư mục `dataset/` (không chia thư mục con `raw/` và `processed/`). Tuyệt đối không lưu file mã nguồn hay notebook vào trong `dataset/`.
   - Thư mục `src/` là nơi lưu trữ toàn bộ file mã nguồn, scripts và notebook xử lý dữ liệu, mô hình hóa (`.py`, `.ipynb`).
   - Luôn duy trì sự tách biệt rõ ràng giữa dữ liệu (`dataset/`) và code (`src/`).
8. **Quy ước đặt tên thư mục:**
   - Toàn bộ tên các thư mục trong repository phải được viết bằng chữ thường (lowercase, ví dụ: `btth/`, `dataset/`, `src/`).
9. **Quy ước đánh số phiên bản và đồng bộ tên file (Versioning & Naming Convention):**
   - **Đồng bộ tên file:** File dataset `.csv` được xuất ra (lưu trong `dataset/`) **bắt buộc phải có cùng tên** với file notebook `.ipynb` tương ứng trong `src/` (chỉ khác đuôi mở rộng, ví dụ: notebook `src/x.y.0_data_cleaning.ipynb` xuất ra dataset `dataset/x.y.0_data_cleaning.csv`).
   - **Định dạng số hiệu phiên bản `x.y.z`:**
     - Tiền tố số hiệu có định dạng chuẩn: **`x.y.z_`** (trong đó `z` mặc định là `0`, và tăng khi người dùng yêu cầu các bước xử lý phụ/patch như `3.0.1_drop_missing`).
     - Phiên bản **`1.0.0`** (ví dụ: `1.0.0_sp500.csv`, `1.0.0_vixcls.csv`): Đại diện cho **bản dữ liệu raw (thô) đầu tiên** thu thập từ nguồn gốc.
   - **Quy tắc tăng số hiệu:**
     - **Chỉ tăng `x`, `y` hoặc `z` khi người dùng yêu cầu.**
     - **Khi tăng `x` thì `y` và `z` bắt buộc quay trở lại bằng 0** (ví dụ: từ `1.3.0` khi tăng `x` sẽ trở thành `2.0.0`).
     - Khi tăng `y` thì giữ nguyên `x` và đặt `z = 0` (ví dụ: từ `1.1.0` tăng `y` sẽ trở thành `1.2.0`).

10. **Quy tắc cô lập đường dẫn và không gian làm việc (Workspace Isolation Rule):**
    - Thư mục làm việc chính thức của dự án là: E:\Repo Project\IS403.R12-equity-return-predictability.
    - **TUYỆT ĐỐI KHÔNG** tạo bất kỳ file, folder, shortcut, symlink hay directory junction nào ra ngoài phạm vi thư mục gốc của repository (đặc biệt là ngoài root ổ đĩa E:\ hoặc sang các dự án khác trong E:\Repo Project\).
    - Mọi tác vụ đọc/ghi file (write_to_file, script xuất báo cáo, cache, scratch) chỉ được phép diễn ra bên trong thư mục repository. Mọi lệnh chạy terminal phải luôn đặt Cwd là thư mục gốc của repo này.

---

## 🛠️ 2. NGUYÊN TẮC CODE & QUẢN LÝ DỰ ÁN
- **Cấu trúc lưu trữ:** Tách biệt độc lập giữa `dataset/` (dữ liệu trực tiếp, không phân thư mục con) và `src/` (mã nguồn/notebooks). Toàn bộ tên thư mục viết thường (`lowercase`).
- **Đồng bộ tên file & phiên bản:** File `.csv` xuất ra trong `dataset/` trùng tên với notebook `.ipynb` tương ứng trong `src/`. Số hiệu `x.y.0` chỉ tăng khi có yêu cầu (tăng `x` thì `y` về `0`).
- **Ngôn ngữ:** Python (>= 3.9).
- **Thư viện chính:**
  - Dữ liệu & Xử lý: `pandas`, `numpy`, `scipy`
  - Trực quan hóa: `matplotlib`, `seaborn`, `plotly`
  - Mô hình hóa: `scikit-learn`, `statsmodels`, `xgboost`, `lightgbm`, `torch`
  - Dữ liệu tài chính: `yfinance`, `ta`, `pandas-ta`
- **Dữ liệu lớn:** Tránh lưu các file dataset nặng (> 100MB) vào Git tracking.

---

## 🛡️ 3. QUY TẮC KIỂM SOÁT DATA LEAKAGE (LEAKAGE CONTROL CHECKLIST)
Mọi bước xử lý dữ liệu và huấn luyện mô hình phải tuân thủ nghiêm ngặt checklist sau:

| Thành phần | Có thể leak? | Cách xử lý chuẩn (Best Practice) | Trạng thái dự án |
| :--- | :---: | :--- | :---: |
| **Target & Biến tương lai** | Có | **Cấm mọi biến từ tương lai trong $X$:** `Target_*`, future return, future price, hay bất kỳ feature nào dùng `.shift(-k)` ($k > 0$) đều chỉ được dùng để tạo $y$. | ✅ Đã tách biệt |
| **Lagged Return** | Có | Chỉ dùng dữ liệu quá khứ thông qua `.shift(k)` với $k \ge 1$. | ✅ Đã kiểm tra |
| **Rolling Feature** | Có | Hàm `.rolling()` chỉ nhìn lùi về quá khứ (không dùng `center=True`). | ✅ Đã kiểm tra |
| **VIX Features** | Có | Chỉ dùng VIX cùng ngày hoặc trễ (`VIXCLS`, `shift(1)`), không dùng thông tin tương lai. | ✅ Đã kiểm tra |
| **Quantile & Regime** | Có | Dùng **Walk-Forward Expanding Window**. Regime tại thời điểm $t$ phải được xác định bằng thông tin có sẵn trước khi dự báo Target tại $t$ (chỉ dùng thông tin $\le t$, cấm thông tin sau $t$). | ✅ Đã áp dụng ở 3.0.0 |
| **Scaling (Scaler)** | Có | Bắt buộc `fit()` chỉ trên tập Train (`scaler.fit(X_train)`), sau đó `transform()` cho cả Train & Test. | ⏳ Bắt buộc khi model |
| **Feature Selection** | Có | Lọc và chọn lọc feature hoàn toàn dựa trên tập Train, không tính trên full dataset. | ⏳ Bắt buộc khi model |
| **Hyperparameter Tuning** | Có | Tune bằng TimeSeriesSplit / Purged CV trên Train/Val, không dùng K-Fold ngẫu nhiên. | ⏳ Bắt buộc khi model |
| **Test Set** | Có | Khóa chặt tập Test (Out-of-Sample), chỉ dùng duy nhất một lần để đánh giá cuối cùng. | ⏳ Bắt buộc khi model |



## 🚫 3. QUY TẮC HUẤN LUYỆN MÔ HÌNH (NO AUTO-TRAIN RULE)
1. **Agent TUYỆT ĐỐI KHÔNG tự ý thực thi lệnh train:**
   - Khi cần huấn luyện, chạy lại mô hình (train, retrain, fine-tune, run_all_baselines, test_engine,...), Agent **NGHIÊM CẤM** tự ý gọi công cụ chạy lệnh huấn luyện trong terminal hay chạy ngầm.
   - Agent chỉ được phép: Kiểm tra mã nguồn, cấu hình, sửa lỗi kỹ thuật, và **cung cấp câu lệnh hoàn chỉnh** để **NGƯỜI DÙNG TỰ COPY VÀ CHẠY** trên terminal của mình.
