# 🛡️ HƯỚNG DẪN KIỂM SOÁT DATA LEAKAGE (LEAKAGE CONTROL GUIDE)

> **Tài liệu kiểm soát chất lượng & chống thiên kiến dữ liệu cho đề tài:**  
> *Dự báo khả năng sinh lời của cổ phiếu (Equity Return Predictability)*

---

## 📌 1. TỔNG QUAN VỀ DATA LEAKAGE TRONG CHUỖI THỜI GIAN TÀI CHÍNH

Trong bài toán dự báo chuỗi thời gian tài chính (Time-Series Financial Forecasting), **Data Leakage (Rò rỉ dữ liệu)** và **Look-Ahead Bias (Thiên kiến nhìn trước tương lai)** là những lỗi sai nghiêm trọng nhất. Nếu dữ liệu tương lai vô tình lọt vào quá trình huấn luyện:
* Kết quả kiểm thử (Backtest / Test Metric) sẽ cao bất thường và phi thực tế.
* Mô hình sẽ thất bại hoàn toàn khi triển khai thực tế vì không có sẵn thông tin tương lai.
* Bài nghiên cứu sẽ bị hội đồng chuyên môn đánh giá là không hợp lệ về mặt phương pháp luận.

---

## 📋 2. BẢNG CHECKLIST KIỂM SOÁT DATA LEAKAGE (LEAKAGE CHECKLIST)

| STT | Thành phần | Có thể leak? | Quy tắc xử lý chuẩn (Best Practice) | Trạng thái trong dự án |
| :---: | :--- | :---: | :--- | :---: |
| **1** | **Target & Biến tương lai** | **CÓ** | **Mọi biến tạo từ tương lai đều cấm xuất hiện trong $X$:** Bao gồm `Target_*`, future return, future price, hoặc bất kỳ feature nào dùng `shift(-k)` ($k > 0$). Tất cả chỉ được dùng để tạo biến mục tiêu $y$. | ✅ **ĐẠT** (Đã phân tách độc lập) |
| **2** | **Lagged Return** | **CÓ** | Chỉ dùng thông tin quá khứ thông qua hàm trễ `.shift(k)` với $k \ge 1$. | ✅ **ĐẠT** (`Return_lag_1, 5, 20`) |
| **3** | **Rolling Features** | **CÓ** | Hàm `.rolling()` chỉ nhìn lùi về quá khứ (backward-looking), tuyệt đối không bật `center=True`. | ✅ **ĐẠT** (`Momentum_*D`, `Volatility_*D`) |
| **4** | **VIX Features** | **CÓ** | Chỉ dùng VIX cùng ngày hoặc các phiên trước (`VIXCLS`, `shift(1)`, `diff(1)`). Không dùng dữ liệu VIX sau thời điểm $t$. | ✅ **ĐẠT** (`VIX_lag_1`, `VIX_change_1D`...) |
| **5** | **Quantile & Regime** | **CÓ** | **Không tính quantile cố định trên toàn bộ dataset 1990–2026**. Regime tại $t$ phải được xác định hoàn toàn bằng thông tin có sẵn $\le t$. Dùng cơ chế **Walk-Forward Expanding Window**. | ✅ **ĐẠT** (Đã áp dụng ở `3.0.0_regime`) |
| **6** | **Scaling (Chuẩn hóa)** | **CÓ** | Bắt buộc **chỉ `fit()` trên tập Train** (`scaler.fit(X_train)`), sau đó mới `transform()` cho cả Train và Test. | ⏳ **Bắt buộc khi Model** |
| **7** | **Feature Selection** | **CÓ** | Việc lựa chọn đặc trưng (Lasso, RFE, Correlation filtering, v.v.) chỉ được tính toán trên **tập Train**. | ⏳ **Bắt buộc khi Model** |
| **8** | **Hyperparameter Tuning** | **CÓ** | Tinh chỉnh siêu tham số bằng **TimeSeriesSplit** / Walk-Forward CV trên Train/Validation; tuyệt đối không dùng K-Fold xáo trộn ngẫu nhiên. | ⏳ **Bắt buộc khi Model** |
| **9** | **Test Set (Kiểm thử)** | **CÓ** | Khóa chặt tập Test (Out-of-sample), chỉ mở ra đánh giá **đúng 1 lần duy nhất** ở bước nghiệm thu cuối cùng. | ⏳ **Bắt buộc khi Model** |

---

## 🔍 3. CHI TIẾT CÁC NGUYÊN TẮC CỐT LÕI

### 3.1. Nguyên tắc 1: Phân tách triệt để giữa Target ($y$) và Features ($X$)
* **Định nghĩa Target:**
  $$\text{Target}_{h, t} = \ln\left(\frac{\text{Close}_{t+h}}{\text{Close}_t}\right), \quad h \in \{1, 5, 20, 60, 120\}$$
* **Quy tắc bất biến:**
  * Target không chỉ là các cột có tiền tố `Target_*` mà **bất kỳ biến nào được sinh ra từ tương lai đều không được phép xuất hiện trong ma trận đặc trưng $X$**.
  * Các biến như: `Target_1D`, `Target_5D`, lợi suất tương lai (future return), giá tương lai (future price), hay bất kỳ thao tác nào sử dụng `.shift(-k)` (với $k > 0$) ➔ **tất cả chỉ được dùng làm $y$ để huấn luyện và đánh giá mô hình**.
  * Khi nạp dữ liệu vào mô hình:
    ```python
    # Chọn một chân trời dự báo cụ thể, ví dụ h = 1 ngày:
    y = df['Target_1']

    # Loại bỏ TOÀN BỘ các cột target tương lai ra khỏi X:
    target_columns = ['Target_1', 'Target_5', 'Target_20', 'Target_60', 'Target_120']
    X = df.drop(columns=['Date'] + target_columns)
    ```

---

### 3.2. Nguyên tắc 2: Cơ chế Walk-Forward Expanding Quantile cho Regime
* **Bài toán:** Phân loại chế độ biến động (`Volatility_Regime`: `Low`, `Normal`, `High`) dựa trên phân vị của `Volatility_20D`.
* **Lỗi rò rỉ nếu làm sai:** Tính $Q_{33}$ và $Q_{67}$ trên toàn bộ dataset 1990–2026 sẽ làm rò rỉ thông tin khủng hoảng tương lai (2008, COVID-2020) vào giai đoạn quá khứ (1995).
* **Điều kiện tiên quyết để không bị Leak:**
  > **Regime tại thời điểm $t$ phải được xác định duy nhất bằng các thông tin có sẵn trước khi dự báo Target tại $t$.**
* **Công thức toán học tại thời điểm $t$:**
  $$Q_{33, t} = \text{Quantile}(\text{Volatility}_{20D, \le t}, 0.33)$$
  $$Q_{67, t} = \text{Quantile}(\text{Volatility}_{20D, \le t}, 0.67)$$
  $$\text{Volatility\_Regime}_t = \begin{cases} \text{Low}, & \text{Volatility}_{20D, t} \le Q_{33, t} \\ \text{Normal}, & Q_{33, t} < \text{Volatility}_{20D, t} \le Q_{67, t} \\ \text{High}, & \text{Volatility}_{20D, t} > Q_{67, t} \end{cases}$$
* **Ý nghĩa:**
  * Tại thời điểm $t$, ta được phép dùng `Volatility_20D(t)` và toàn bộ dữ liệu lịch sử tính đến $t$ ($\le t$).
  * Tuyệt đối không dùng bất kỳ thông tin nào phát sinh sau thời điểm $t$.

---

### 3.3. Nguyên tắc 3: Quy chuẩn cho giai đoạn Mô hình hóa (Modeling)
Khi bước vào giai đoạn huấn luyện mô hình, cần tuân thủ 3 nguyên tắc:

1. **Phân chia dữ liệu theo thời gian (Time-based Split):**
   * Không dùng `train_test_split(shuffle=True)`.
   * Chia nối tiếp theo trục thời gian, ví dụ:
     * **Train:** `1990-01-02` đến `2018-12-31` (~80%)
     * **Test:** `2019-01-02` đến `2026-09-28` (~20%)

2. **Quy chuẩn chuẩn hóa (Data Scaling):**
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   # CHỈ FIT TRÊN TẬP TRAIN:
   X_train_scaled = scaler.fit_transform(X_train)

   # TRANSFORM TRÊN TẬP TEST BẰNG THAM SỐ CỦA TRAIN:
   X_test_scaled = scaler.transform(X_test)
   ```

3. **Kiểm định chéo (Cross-Validation):**
   * Sử dụng `TimeSeriesSplit` hoặc cơ chế Walk-Forward / Rolling-Window Validation.
   * Đảm bảo tập Validation luôn nằm sau tập Train trong từng fold.

---

## 📂 4. DANH MỤC CÁC PHIÊN BẢN DỮ LIỆU ĐÃ CHUẨN HÓA

| Phiên bản | Tên Notebook (`src/preprocessing/`) | Tên Dataset (`dataset/`) | Nội dung thực hiện | Số dòng | Số cột |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **`1.0.0`** | *(Dữ liệu thô)* | `1.0.0_sp500.csv`<br>`1.0.0_vixcls.csv` | Dữ liệu thô ban đầu từ Yahoo Finance & FRED. | 9,253<br>9,585 | 7<br>2 |
| **`1.1.0`** | `1.1.0_data_cleaning.ipynb` | `1.1.0_data_cleaning.csv` | Chuẩn hóa Date, left merge, ffill VIX, bỏ dòng thừa. | 9,252 | 8 |
| **`2.0.0`** | `2.0.0_return.ipynb` | `2.0.0_return.csv` | Tính `Return_1D`, bảo toàn phiên đầu tiên bằng $\ln(\text{Close}_0/\text{Open}_0)$. | 9,252 | 9 |
| **`2.1.0`** | `2.1.0_target.ipynb` | `2.1.0_target.csv` | Tạo 5 chân trời dự báo: `Target_1, 5, 20, 60, 120` bằng `.shift(-h)`. | 9,252 | 14 |
| **`2.2.0`** | `2.2.0_feature.ipynb` | `2.2.0_feature.csv` | Tạo 15 đặc trưng kỹ thuật (Lag, Momentum, Volatility, VIX, Price, Volume). | 9,252 | 29 |
| **`3.0.0`** | `3.0.0_regime.ipynb` | `3.0.0_regime.csv` | Phân loại `Volatility_Regime` bằng Walk-Forward Expanding Quantile (chống leak). | 9,252 | 30 |
| **`3.0.1`** | `3.0.1_drop_missing.ipynb` | `3.0.1_drop_missing.csv` | Loại bỏ 179 dòng có ô trống (59 dòng đầu do lag 60D, 120 dòng cuối do target 120D), làm sạch 100%. | 9,073 | 30 |
| **`3.0.2`** | `3.0.2_split.ipynb` | `3.0.2_split.csv` | Thêm cột `Split` chia tuần tự theo thời gian: `1` (Train 70%), `2` (Validation 15%), `3` (Test 15%). | 9,073 | 31 |
| **`3.0.3`** | `3.0.3_val_folds.ipynb` | `3.0.3_val_folds.csv` | Chia tập Validation (1,360 dòng) thành 5 Folds theo Walk-Forward (Expanding Train, Fixed Val 272 dòng). Thêm cột `Val_Fold` (`0..5`). | 9,073 | 32 |
