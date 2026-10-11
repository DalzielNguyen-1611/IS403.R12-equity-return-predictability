# QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU & CẤU TRÚC ĐẶC TRƯNG CHUỖI THỜI GIAN (DATA PREPROCESSING PIPELINE)

> **Đề tài nghiên cứu:** *"When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis"*  
> **Thư mục mã nguồn:** `src/preprocessing/`  
> **Thư mục lưu trữ dữ liệu:** `dataset/`  
> **Nguyên tắc phương pháp luận tối thượng:** **Bảo tồn tính nhân quả thời gian (Temporal Causality)** và **Triệt tiêu hoàn toàn rò rỉ thông tin tương lai (Zero Look-Ahead Bias / Data Leakage)**.

---

## 📑 MỤC LỤC
1. [Tổng Quan Kiến Trúc Đường Ống Dữ Liệu](#1-tổng-quan-kiến-trúc-đường-ống-dữ-liệu)
2. [Bảng Ánh Xạ Tập Tin & Tiến Trình Xử Lý](#2-bảng-ánh-xạ-tập-tin--tiến-trình-xử-lý)
3. [Chi Tiết 8 Giai Đoạn Tiền Xử Lý (Phase-by-Phase Breakdown)](#3-chi-tiết-8-giai-đoạn-tiền-xử-lý)
   - [Giai đoạn 1: Làm Sạch & Đồng Bộ Dữ Liệu Gốc (1.1.0)](#giai-đoạn-1-làm-sạch--đồng-bộ-dữ-liệu-gốc-110)
   - [Giai đoạn 2: Xây Dựng Biến Lợi Suất Chuẩn (2.0.0)](#giai-đoạn-2-xây-dựng-biến-lợi-suất-chuẩn-200)
   - [Giai đoạn 3: Thiết Lập Biến Mục Tiêu Đa Chân Trời (2.1.0)](#giai-đoạn-3-thiết-lập-biến-mục-tiêu-đa-chân-trời-210)
   - [Giai đoạn 4: Kỹ Thuật Tạo Đặc Trưng Phi Rò Rỉ (2.2.0)](#giai-đoạn-4-kỹ-thuật-tạo-đặc-trưng-phi-rò-rỉ-220)
   - [Giai đoạn 5: Phân Loại Chế Độ Biến Động Walk-Forward (3.0.0)](#giai-đoạn-5-phân-loại-chế-độ-biến-động-walk-forward-300)
   - [Giai đoạn 6: Loại Bỏ Dữ Liệu Khuyết Thiếu (3.0.1)](#giai-đoạn-6-loại-bỏ-dữ-liệu-khuyết-thiếu-301)
   - [Giai đoạn 7: Phân Chia Tập Dữ Liệu Theo Thời Gian (3.0.2)](#giai-đoạn-7-phân-chia-tập-dữ-liệu-theo-thời-gian-302)
   - [Giai đoạn 8: Phân Chia Walk-Forward 5 Folds Ngoại Mẫu (3.0.3)](#giai-đoạn-8-phân-chia-walk-forward-5-folds-ngoại-mẫu-303)
4. [Nguyên Tắc Kiểm Soát Rò Rỉ Dữ Liệu (Leakage Auditing Framework)](#4-nguyên-tắc-kiểm-soát-rò-rỉ-dữ-liệu)

---

## 1. Tổng Quan Kiến Trúc Đường Ống Dữ Liệu

Đường ống tiền xử lý được thiết kế theo cấu trúc mô-đun hóa 8 tầng tuần tự. Mỗi tầng được đóng gói trong một Jupyter Notebook độc lập, tương ứng với một tập tin CSV kết quả được lưu trữ tại `dataset/`:

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│         1.0.0_sp500.csv         │       │        1.0.0_vixcls.csv         │
│   (Giá OHLCV S&P 500 từ Yahoo)  │       │     (Chỉ số VIX từ FRED CBOE)   │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └───────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │      1.1.0_data_cleaning.ipynb      │  ──► 1.1.0_data_cleaning.csv
                  │  (Ghép Left-Join & Ffill VIXCLS)    │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          2.0.0_return.ipynb         │  ──► 2.0.0_return.csv
                  │     (Tính toán Daily Log Return)    │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          2.1.0_target.ipynb         │  ──► 2.1.0_target.csv
                  │ (Xây dựng Target: 1D,5D,20D,60D,120D)│
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │         2.2.0_feature.ipynb         │  ──► 2.2.0_feature.csv
                  │  (15 Đặc trưng: Lợi suất, VIX, Vol) │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          3.0.0_regime.ipynb         │  ──► 3.0.0_regime.csv
                  │ (Phân loại chế độ VIX Walk-Forward) │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │       3.0.1_drop_missing.ipynb      │  ──► 3.0.1_drop_missing.csv
                  │ (Loại trừ 179 dòng khuyết thiếu NaN)│
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          3.0.2_split.ipynb          │  ──► 3.0.2_split.csv
                  │   (Chia tập: Train / Val / Test)    │
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │        3.0.3_val_folds.ipynb        │  ──► 3.0.3_val_folds.csv
                  │   (Thiết lập 5 Folds Walk-Forward)  │      (Tập dữ liệu huấn luyện)
                  └─────────────────────────────────────┘
```

---

## 2. Bảng Ánh Xạ Tập Tin & Tiến Trình Xử Lý

| Bước | Jupyter Notebook nguồn | Tập tin dữ liệu đầu ra | Kích thước | Khoảng thời gian | Nội dung bổ sung cốt lõi |
| :-: | :--- | :--- | :-: | :-: | :--- |
| **0** | *Dữ liệu thô thu thập* | `1.0.0_sp500.csv`<br>`1.0.0_vixcls.csv` | 9,253 × 7<br>9,585 × 2 | 1990-01-02 $\to$ 2026-09-29 | Giá OHLCV S&P 500 và chỉ số biến động VIX thô. |
| **1** | `1.1.0_data_cleaning.ipynb` | `1.1.0_data_cleaning.csv` | 9,252 × 8 | 1990-01-02 $\to$ 2026-09-28 | Khớp nối Left-join theo ngày S&P 500, ffill VIXCLS. |
| **2** | `2.0.0_return.ipynb` | `2.0.0_return.csv` | 9,252 × 9 | 1990-01-02 $\to$ 2026-09-28 | Lợi suất log hàng ngày: `Return_1D`. |
| **3** | `2.1.0_target.ipynb` | `2.1.0_target.csv` | 9,252 × 14 | 1990-01-02 $\to$ 2026-09-28 | 5 biến mục tiêu tương lai `Target_1, 5, 20, 60, 120`. |
| **4** | `2.2.0_feature.ipynb` | `2.2.0_feature.csv` | 9,252 × 29 | 1990-01-02 $\to$ 2026-09-28 | 15 đặc trưng kỹ thuật trễ, động lượng, biến động và khối lượng. |
| **5** | `3.0.0_regime.ipynb` | `3.0.0_regime.csv` | 9,252 × 30 | 1990-01-02 $\to$ 2026-09-28 | Nhãn trạng thái `Volatility_Regime` (Low, Normal, High). |
| **6** | `3.0.1_drop_missing.ipynb` | `3.0.1_drop_missing.csv` | 9,073 × 30 | 1990-03-27 $\to$ 2026-04-07 | Loại bỏ 179 hàng NaN biên đầu và đuôi. |
| **7** | `3.0.2_split.ipynb` | `3.0.2_split.csv` | 9,073 × 31 | 1990-03-27 $\to$ 2026-04-07 | Gán nhãn `Split` (1: Train, 2: Validation, 3: Test). |
| **8** | `3.0.3_val_folds.ipynb` | `3.0.3_val_folds.csv` | **9,073 × 32** | 1990-03-27 $\to$ 2026-04-07 | Gán nhãn `Val_Fold` (1 đến 5 cho Split 2). |

---

## 3. Chi Tiết 8 Giai Đoạn Tiền Xử Lý

### Giai đoạn 1: Làm Sạch & Đồng Bộ Dữ Liệu Gốc (1.1.0)
* **Mục tiêu:** Đồng bộ lịch giao dịch giữa thị trường cơ sở (S&P 500) và thị trường phái sinh/quyền chọn (VIX).
* **Quy trình kỹ thuật:**
  1. Định dạng cột `Date` theo chuẩn ISO-8601 (`YYYY-MM-DD`).
  2. Thực hiện ghép nối `pd.merge(df_sp500, df_vix, on='Date', how='left')` theo lịch giao dịch của S&P 500.
  3. Xử lý ngày nghỉ lễ của VIX bằng **Forward Fill (`ffill()`)**: Lấy giá trị đóng cửa VIX của ngày giao dịch liền trước. Đây là thao tác chuẩn hóa kinh tế lượng, bảo đảm nhà đầu tư tại thời điểm $t$ sử dụng thông tin VIX mới nhất đã được công bố mà không bị rò rỉ tương lai.

### Giai đoạn 2: Xây Dựng Biến Lợi Suất Chuẩn (2.0.0)
* **Mục tiêu:** Chuyển đổi chuỗi giá đóng cửa không dừng (non-stationary price levels) sang chuỗi lợi suất log dừng (stationary log returns).
* **Công thức toán học:**

$$\text{Return\_1D}_t = \ln\left(\frac{Close_t}{Close_{t-1}}\right) = \ln(Close_t) - \ln(Close_{t-1})$$

* **Đặc tính:** Có tính cộng gộp theo thời gian (time-additive), hỗ trợ việc tính toán lợi suất đa kỳ hạn ở các bước sau.

### Giai đoạn 3: Thiết Lập Biến Mục Tiêu Đa Chân Trời (2.1.0)
* **Mục tiêu:** Xây dựng biến phụ thuộc tương lai cho nghiên cứu câu hỏi **RQ1 (Predictability Decay)** qua 5 chân trời: 1 ngày ($1D$), 1 tuần ($5D$), 1 tháng ($20D$), 1 quý ($60D$), nửa năm ($120D$).
* **Công thức toán học:**

$$Target_{h, t} = \ln\left(\frac{Close_{t+h}}{Close_t}\right) = \sum_{i=1}^h \text{Return\_1D}_{t+i}, \quad h \in \{1, 5, 20, 60, 120\}$$

* **Biến mục tiêu tương ứng trong dataset:** `Target_1`, `Target_5`, `Target_20`, `Target_60`, `Target_120`.

### Giai đoạn 4: Kỹ Thuật Tạo Đặc Trưng Phi Rò Rỉ (2.2.0)
* **Mục tiêu:** Xây dựng 15 biến giải thích ($X_t$) phản ánh toàn diện các khía cạnh kỹ thuật, động lượng, độ biến động và tâm lý thị trường tính đến đúng ngày $t$.
* **Danh mục đặc trưng (15 features):**
  1. **Nhóm Trễ Lợi Suất (Lagged Returns):**
     - `Return_lag_1`: $\text{Return\_1D}_{t-1}$
     - `Return_lag_5`: $\text{Return\_1D}_{t-5}$
     - `Return_lag_20`: $\text{Return\_1D}_{t-20}$
  2. **Nhóm Động Lượng / Tích Lũy (Momentum):**
     - `Momentum_5D`: $\ln(Close_t / Close_{t-5})$
     - `Momentum_20D`: $\ln(Close_t / Close_{t-20})$
     - `Momentum_60D`: $\ln(Close_t / Close_{t-60})$
  3. **Nhóm Biến Động Thực Tế (Rolling Realized Volatility):**
     - `Volatility_5D`, `Volatility_20D`, `Volatility_60D`: Độ lệch chuẩn có hiệu chỉnh của lợi suất log trong cửa sổ trượt $5, 20, 60$ phiên quá khứ:

$$\sigma_{k, t} = \sqrt{\frac{1}{k-1} \sum_{j=0}^{k-1} (\text{Return\_1D}_{t-j} - \bar{r})^2}$$

  4. **Nhóm Tâm Lý Thị Trường (VIX-based Features):**
     - `VIX_lag_1`: Giá trị $VIX_{t-1}$
     - `VIX_change_1D`: $VIX_t - VIX_{t-1}$
     - `VIX_pct_change_1D`: $(VIX_t - VIX_{t-1}) / VIX_{t-1}$
  5. **Nhóm Cấu Trúc Giá Phiên (Intraday Price Action):**
     - `HL_range`: Biên độ dao động phiên $\ln(High_t / Low_t)$
     - `OC_return`: Lợi suất trong phiên $\ln(Close_t / Open_t)$
  6. **Nhóm Khối Lượng Giao Dịch (Volume Dynamic):**
     - `Volume_change`: Tỷ lệ thay đổi khối lượng $\ln(Volume_t / Volume_{t-1})$

### Giai đoạn 5: Phân Loại Chế Độ Biến Động Walk-Forward (3.0.0)
* **Mục tiêu:** Xác định chế độ thị trường (`Volatility_Regime`: Low, Normal, High) phục vụ kiểm định câu hỏi **RQ2 (Regime Moderation)** mà không gây rò rỉ phân vị tương lai.
* **Nguyên tắc Expanding Window Quantile:**
  - Tại mỗi ngày $t$, chỉ sử dụng chuỗi lịch sử độ biến động 20 phiên từ đầu mẫu đến $t$ ($\le t$), với độ dài tối thiểu 1 năm giao dịch (`min_periods = 252`).
  - Tính ngưỡng động 33% và 67%:

$$Q_{33}(t) = \text{Quantile}_{0.33}(\{\text{Volatility\_20D}_\tau\}_{\tau=1}^t)$$

$$Q_{67}(t) = \text{Quantile}_{0.67}(\{\text{Volatility\_20D}_\tau\}_{\tau=1}^t)$$

* **Phân loại nhãn:**

$$\text{Volatility\_Regime}_t = \begin{cases} \text{Low}, & \text{nếu } \text{Volatility\_20D}_t \le Q_{33}(t) \\ \text{Normal}, & \text{nếu } Q_{33}(t) < \text{Volatility\_20D}_t \le Q_{67}(t) \\ \text{High}, & \text{nếu } \text{Volatility\_20D}_t > Q_{67}(t) \end{cases}$$

* **Phân phối nhãn:** Low: $49.2\%$, Normal: $20.2\%$, High: $30.6\%$.

### Giai đoạn 6: Loại Bỏ Dữ Liệu Khuyết Thiếu (3.0.1)
* **Mục tiêu:** Loại bỏ các hàng $NaN$ do đặc tính tính toán chuỗi thời gian, đảm bảo bộ dữ liệu sạch hoàn hảo.
* **Nguyên nhân và phân bổ loại trừ (179 dòng):**
  - **Biên đầu (60 dòng đầu tiên: 1990-01-02 đến 1990-03-26):** Thiếu do các cửa sổ trượt dài nhất cần 60 phiên lịch sử (`Momentum_60D`, `Volatility_60D`).
  - **Biên đuôi (119 dòng cuối cùng: 2026-04-08 đến 2026-09-28):** Thiếu do tại thời điểm cuối tập dữ liệu, chưa có đủ 120 phiên tương lai để tính toán `Target_120`.
* **Kết quả:** Dataset giảm từ 9,252 xuống còn chính xác **9,073 dòng dữ liệu hoàn chỉnh** (từ ngày 1990-03-27 đến 2026-04-07).

### Giai đoạn 7: Phân Chia Tập Dữ Liệu Theo Thời Gian (3.0.2)
* **Mục tiêu:** Phân chia ranh giới 3 tập dữ liệu theo thứ tự thời gian tuyến tính, tuyệt đối không xáo trộn ngẫu nhiên.
* **Cấu trúc phân chia cột `Split`:**
  - **Split 1 (Training Set):** 6,348 phiên giao dịch (1990-03-27 $\to$ 2015-06-08) — Chiếm $69.97\%$. Phục vụ huấn luyện nền tảng ban đầu.
  - **Split 2 (Validation Set):** 1,360 phiên giao dịch (2015-06-09 $\to$ 2020-10-30) — Chiếm $14.99\%$. Phục vụ đánh giá Walk-Forward 5 Folds và nghiên cứu RQ1, RQ2, RQ3.
  - **Split 3 (Hold-out Test Set):** 1,365 phiên giao dịch (2020-11-02 $\to$ 2026-04-07) — Chiếm $15.04\%$. Phục vụ thẩm định độc lập cuối cùng.

### Giai đoạn 8: Phân Chia Walk-Forward 5 Folds Ngoại Mẫu (3.0.3)
* **Mục tiêu:** Thiết lập cơ chế thẩm định ngoại mẫu tịnh tiến mở rộng (Expanding Window Walk-Forward) trên tập Validation (Split 2).
* **Cấu trúc 5 Folds (mỗi Fold kiểm định 272 phiên):**

| Val_Fold | Cửa sổ Huấn luyện (Train Window) | Số dòng Train | Cửa sổ Kiểm định Ngoại mẫu (Val Window) | Số dòng Val |
| :-: | :---: | :-: | :---: | :-: |
| **Fold 1** | 1990-03-27 $\to$ 2015-06-09 | 6,351 | 2015-06-10 $\to$ 2016-07-07 | 272 |
| **Fold 2** | 1990-03-27 $\to$ 2016-07-07 | 6,623 | 2016-07-08 $\to$ 2017-08-04 | 272 |
| **Fold 3** | 1990-03-27 $\to$ 2017-08-04 | 6,895 | 2017-08-07 $\to$ 2018-09-04 | 272 |
| **Fold 4** | 1990-03-27 $\to$ 2018-09-04 | 7,167 | 2018-09-05 $\to$ 2019-10-03 | 272 |
| **Fold 5** | 1990-03-27 $\to$ 2019-10-03 | 7,439 | 2019-10-04 $\to$ 2020-10-30 | 272 |

* **Đầu ra cuối cùng:** Tập tin **`dataset/3.0.3_val_folds.csv`** (9,073 dòng, 32 cột) — Là nguồn dữ liệu duy nhất và chính thức nạp vào hệ thống huấn luyện 14 mô hình tại `src/training/baselines/`.

---

## 4. Nguyên Tắc Kiểm Soát Rò Rỉ Dữ Liệu

1. **Nguyên tắc Chỉ Nhìn Quá Khứ (Strict Trailing):** Toàn bộ đặc trưng $X_t$ tại dòng $t$ chỉ dùng thông tin đến đúng ngày $t$. Không sử dụng centered window hay giá trị của $t+1$.
2. **Không đưa `Volatility_Regime` vào ma trận $X$:** Đảm bảo 14 mô hình là Global Models (mô hình tổng quát). Nhãn Regime chỉ được giữ lại để phân tầng đánh giá hiệu năng ngoại mẫu (RQ2).
3. **Phân vị động (Walk-Forward Quantiles):** Ngưỡng phân loại VIX/Volatility được tính lũy tiến, loại trừ việc dùng phân vị toàn mẫu.
4. **Cô lập chuẩn hóa (Scaler Isolation):** Mọi thao tác chuẩn hóa (MinMaxScaler, StandardScaler) trong mã nguồn huấn luyện chỉ được `fit` trên tập Train của từng Fold rồi mới `transform` sang tập Val.
