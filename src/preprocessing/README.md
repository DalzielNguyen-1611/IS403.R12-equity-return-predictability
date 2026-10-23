# TIỀN XỬ LÝ DỮ LIỆU & QUY TRÌNH THỰC THI KINH TẾ LƯỢNG (DATA PREPROCESSING & ECONOMETRIC PIPELINE)

> **Dự án:** Nghiên cứu Thực nghiệm về Khả năng Dự báo Lợi suất Cổ phiếu: Tiếp cận Đa Kỳ hạn, Đa Chế độ Biến động và Đa Mô hình Học máy (S&P 500 & VIX, 1990 – 2026).  
> **Bộ môn:** Hệ thống Thông tin Quản lý / Phân tích Dữ liệu — Khoa Hệ thống Thông tin, Trường ĐH Công nghệ Thông tin, ĐHQG-HCM.  
> **Thực hiện:** Nhóm 6 (Nguyễn Thị Phương Hoa, Nguyễn Đoàn Đức Hiếu, Nguyễn Văn Phát, Trần Ngọc Thảo, Trần Anh Thư).

---

## 📑 MỤC LỤC

1. [Tổng Quan Kiến Trúc Đường Ống Dữ Liệu (Pipeline Architecture)](#1-tổng-quan-kiến-trúc-đường-ống-dữ-liệu)
2. [Bảng Ánh Xạ Tập Tin, Kích Thước & Dòng Thời Gian](#2-bảng-ánh-xạ-tập-tin-kích-thước--dòng-thời-gian)
3. [Chi Tiết 8 Giai Đoạn Tiền Xử Lý Dữ Liệu](#3-chi-tiết-8-giai-đoạn-tiền-xử-lý-dữ-liệu)
   - [Giai đoạn 1: Làm Sạch & Đồng Bộ Lịch Giao Dịch (1.1.0)](#giai-đoạn-1-làm-sạch--đồng-bộ-lịch-giao-dịch-110)
   - [Giai đoạn 2: Xây Dựng Biến Lợi Suất Chuẩn (2.0.0)](#giai-đoạn-2-xây-dựng-biến-lợi-suất-chuẩn-200)
   - [Giai đoạn 3: Thiết Lập Biến Mục Tiêu Đa Kỳ Hạn (2.1.0)](#giai-đoạn-3-thiết-lập-biến-mục-tiêu-đa-kỳ-hạn-210)
   - [Giai đoạn 4: Kỹ Thuật Tạo 16 Đặc Trưng Dự Báo (2.2.0)](#giai-đoạn-4-kỹ-thuật-tạo-16-đặc-trưng-dự-báo-220)
   - [Giai đoạn 5: Phân Loại Chế Độ Thị Trường Walk-Forward (3.0.0)](#giai-đoạn-5-phân-loại-chế-độ-thị-trường-walk-forward-300)
   - [Giai đoạn 6: Loại Bỏ Dữ Liệu Khuyết Thiếu Có Kiểm Soát (3.0.1)](#giai-đoạn-6-loại-bỏ-dữ-liệu-khuyết-thiếu-có-kiểm-soát-301)
   - [Giai đoạn 7: Phân Chia Tập Dữ Liệu Theo Tuyến Thời Gian (3.0.2)](#giai-đoạn-7-phân-chia-tập-dữ-liệu-theo-tuyến-thời-gian-302)
   - [Giai đoạn 8: Phân Chia Walk-Forward 5 Folds Ngoại Mẫu (3.0.3)](#giai-đoạn-8-phân-chia-walk-forward-5-folds-ngoại-mẫu-303)
4. [Nhật Ký Điều Chỉnh Kỹ Thuật & Khắc Phục Lỗi Xử Lý Dữ Liệu](#4-nhật-ký-điều-chỉnh-kỹ-thuật--khắc-phục-lỗi-xử-lý-dữ-liệu)
5. [Nguyên Tắc Kiểm Soát Rò Rỉ Tuyệt Đối & Giả Định Thực Thi](#5-nguyên-tắc-kiểm-soát-rò-rỉ-tuyệt-đối--giả-định-thực-thi)
   - [5.1 Bộ Ngũ Nguyên Tắc Chống Rò Rỉ (Anti-Leakage Core Rules)](#51-bộ-ngũ-nguyên-tắc-chống-rò-rỉ-anti-leakage-core-rules)
   - [5.2 Khoảng Đệm Purging Marcos López de Prado](#52-khoảng-đệm-purging-marcos-lópez-de-prado)
   - [5.3 Giả Định Khớp Lệnh MOC & Ma Sát Giao Dịch Sống](#53-giả-định-khớp-lệnh-moc--ma-sát-giao-dịch-sống)

---

## 1. Tổng Quan Kiến Trúc Đường Ống Dữ Liệu

Đường ống tiền xử lý dữ liệu được thiết kế theo cấu trúc mô-đun tuần tự (sequential modular pipeline) gồm 8 bước tương ứng với 8 Jupyter Notebook độc lập trong thư mục `src/preprocessing/`. Mỗi bước tiếp nhận trực tiếp tập tin CSV đầu ra của bước liền trước và xuất bản một tập tin CSV mới tại thư mục `dataset/`, bảo đảm tính tái lập (reproducibility), dễ dàng kiểm toán (auditability) và không xảy ra rò rỉ dữ liệu (zero data leakage).

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│         1.0.0_sp500.csv         │       │         1.0.0_vixcls.csv        │
│   (Giá OHLCV S&P 500 từ Yahoo)  │       │     (Chỉ số VIX từ FRED CBOE)   │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 └───────────────────┬─────────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │      1.1.0_data_cleaning.ipynb      │  ──► 1.1.0_data_cleaning.csv
                  │  (Ghép Left-Join & Ffill VIXCLS)    │      (9,252 × 8)
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          2.0.0_return.ipynb         │  ──► 2.0.0_return.csv
                  │     (Tính toán Daily Log Return)    │      (9,252 × 9)
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          2.1.0_target.ipynb         │  ──► 2.1.0_target.csv
                  │ (Xây dựng Target: 1D,5D,20D,60D,120D)│      (9,252 × 14)
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │         2.2.0_feature.ipynb         │  ──► 2.2.0_feature.csv
                  │  (16 Đặc trưng: Lợi suất, VIX, Vol) │      (9,252 × 29)
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          3.0.0_regime.ipynb         │  ──► 3.0.0_regime.csv
                  │ (Phân loại chế độ VIX Walk-Forward) │      (9,252 × 30)
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │       3.0.1_drop_missing.ipynb      │  ──► 3.0.1_drop_missing.csv
                  │ (Loại trừ 179 dòng khuyết thiếu NaN)│      (9,073 × 30)
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          3.0.2_split.ipynb          │  ──► 3.0.2_split.csv
                  │   (Chia tập: Train / Val / Test)    │      (9,073 × 31)
                  └──────────────────┬──────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │        3.0.3_val_folds.ipynb        │  ──► 3.0.3_val_folds.csv
                  │   (Thiết lập 5 Folds Walk-Forward)  │      (9,073 × 32 - File chính thức)
                  └─────────────────────────────────────┘
```

---

## 2. Bảng Ánh Xạ Tập Tin, Kích Thước & Dòng Thời Gian

Bảng đối chiếu chuẩn xác giữa mã nguồn thực thi, tập tin dữ liệu xuất bản và các chỉ số kích thước thực tế trong thư mục `dataset/`:

| Bước | Jupyter Notebook nguồn | Tập tin dữ liệu đầu ra | Kích thước | Số ô NaN | Khoảng thời gian | Nội dung kỹ thuật cốt lõi |
| :-: | :--- | :--- | :-: | :-: | :-: | :--- |
| **0** | *Dữ liệu thô ban đầu* | `1.0.0_sp500.csv`<br>`1.0.0_vixcls.csv` | 9,253 × 7<br>9,585 × 2 | 0<br>0 | 1990-01-02 $\to$ 2026-09-29<br>1990-01-02 $\to$ 2026-09-28 | Giá OHLCV S&P 500 từ Yahoo Finance và chỉ số VIX đóng cửa từ FRED/CBOE. |
| **1** | `1.1.0_data_cleaning.ipynb` | `1.1.0_data_cleaning.csv` | 9,252 × 8 | 0 | 1990-01-02 $\to$ 2026-09-28 | Khớp nối Left-join theo ngày giao dịch S&P 500, forward-fill VIXCLS khi lệch lịch nghỉ lễ. |
| **2** | `2.0.0_return.ipynb` | `2.0.0_return.csv` | 9,252 × 9 | 0 | 1990-01-02 $\to$ 2026-09-28 | Tính lợi suất log ngày: `Return_1D`, bảo toàn dòng $t=0$ qua $\ln(Close_0 / Open_0)$. |
| **3** | `2.1.0_target.ipynb` | `2.1.0_target.csv` | 9,252 × 14 | 206 | 1990-01-02 $\to$ 2026-09-28 | 5 biến mục tiêu tương lai `Target_1, 5, 20, 60, 120` ($1+5+20+60+120=206$ ô NaN ở đuôi). |
| **4** | `2.2.0_feature.ipynb` | `2.2.0_feature.csv` | 9,252 × 29 | 400 | 1990-01-02 $\to$ 2026-09-28 | Tạo 15 đặc trưng kỹ thuật mới, khử dị biệt $Volume=0$ ngày 2023-05-24 bằng forward-fill. |
| **5** | `3.0.0_regime.ipynb` | `3.0.0_regime.csv` | 9,252 × 30 | 419 | 1990-01-02 $\to$ 2026-09-28 | Phân loại trạng thái `Volatility_Regime` (Low, Normal, High) bằng phân vị động tịnh tiến. |
| **6** | `3.0.1_drop_missing.ipynb` | `3.0.1_drop_missing.csv` | 9,073 × 30 | 0 | 1990-03-27 $\to$ 2026-04-07 | Cắt bỏ 179 dòng NaN biên (60 dòng đầu do trễ 60D, 119 dòng cuối do Target 120D). |
| **7** | `3.0.2_split.ipynb` | `3.0.2_split.csv` | 9,073 × 31 | 0 | 1990-03-27 $\to$ 2026-04-07 | Phân chia tuyến tính: Train 70.00% (6,351 dòng), Val 14.99% (1,360 dòng), Test 15.01% (1,362 dòng). |
| **8** | `3.0.3_val_folds.ipynb` | `3.0.3_val_folds.csv` | **9,073 × 32** | 0 | 1990-03-27 $\to$ 2026-04-07 | Thiết lập 5 Folds Expanding Walk-Forward trên tập Validation. **Tập tin chính thức cho huấn luyện**. |

---

## 3. Chi Tiết 8 Giai Đoạn Tiền Xử Lý Dữ Liệu

### Giai đoạn 1: Làm Sạch & Đồng Bộ Lịch Giao Dịch (1.1.0)
* **Mục tiêu:** Khắc phục sự sai lệch về lịch làm việc giữa thị trường cơ sở (S&P 500 niêm yết trên NYSE/NASDAQ) và thị trường phái sinh quyền chọn (VIX được tính toán bởi CBOE).
* **Quy trình thực hiện:**
  1. Định dạng cột ngày `Date` theo chuẩn quốc tế ISO-8601 (`YYYY-MM-DD`).
  2. Thực hiện ghép nối `pd.merge(df_sp500, df_vix, on='Date', how='left')` căn cứ theo lịch giao dịch của S&P 500.
  3. **Xử lý ngày nghỉ lễ:** Các phiên NYSE mở cửa nhưng CBOE đóng cửa (hoặc dữ liệu VIX công bố muộn) được xử lý bằng **Forward Fill (`ffill()`)**: sử dụng mức đóng cửa VIX của ngày làm việc liền trước. Phương pháp này bảo đảm tính nhân quả (causality) vì tại ngày $t$, giá trị VIX của ngày $t-1$ đã được công khai trên thị trường.

### Giai đoạn 2: Xây Dựng Biến Lợi Suất Chuẩn (2.0.0)
* **Mục tiêu:** Chuyển đổi chuỗi giá đóng cửa không dừng (non-stationary price levels) sang chuỗi lợi suất log dừng (stationary log returns), đảm bảo tính cộng gộp theo thời gian (time-additivity).
* **Công thức toán học:**

$$\text{Return\_1D}_t = \ln\left(\frac{Close_t}{Close_{t-1}}\right)$$

* **Bảo toàn dữ liệu dòng đầu tiên ($t=0$):**
  Tại ngày đầu tiên của tập dữ liệu (1990-01-02), do không có phiên $t-1$, công thức trễ thông thường sẽ sinh ra $NaN$. Thay vì loại bỏ làm mất 1 phiên giao dịch, mô hình tính toán lợi suất trong ngày của phiên mở màn:
  $$\text{Return\_1D}_0 = \ln\left(\frac{Close_0}{Open_0}\right)$$
  Thao tác này giúp bảo toàn trọn vẹn 100% số quan sát (9,252 dòng) mà không làm suy hao dữ liệu.

### Giai đoạn 3: Thiết Lập Biến Mục Tiêu Đa Kỳ Hạn (2.1.0)
* **Mục tiêu:** Xây dựng biến phụ thuộc tương lai phục vụ trả lời câu hỏi nghiên cứu **RQ1 (Predictability Decay across Multi-Horizons)** trên 5 chân trời: 1 ngày ($1D$), 1 tuần ($5D$), 1 tháng ($20D$), 1 quý ($60D$), nửa năm ($120D$).
* **Công thức toán học:**

$$Target_{h, t} = \ln\left(\frac{Close_{t+h}}{Close_t}\right) = \sum_{i=1}^h \text{Return\_1D}_{t+i}, \quad h \in \{1, 5, 20, 60, 120\}$$

* **Đặc điểm khuyết thiếu tự nhiên:** Do nhìn về tương lai $h$ phiên, tại $h$ dòng cuối cùng của tập dữ liệu sẽ chưa có mức giá tương lai, dẫn đến chính xác $1 + 5 + 20 + 60 + 120 = 206$ ô giá trị $NaN$ (sẽ được xử lý dứt điểm ở bước 3.0.1).

### Giai đoạn 4: Kỹ Thuật Tạo 16 Đặc Trưng Dự Báo (2.2.0)
* **Mục tiêu:** Xây dựng tập đặc trưng dự báo toàn diện ($X_t$) bao gồm 15 đặc trưng kỹ thuật tạo mới và 1 đặc trưng cơ sở (`VIXCLS`), phản ánh 6 khía cạnh kinh tế lượng:

| STT | Nhóm đặc trưng | Tên cột đặc trưng | Công thức toán học / Kỹ thuật tính | Ý nghĩa kinh tế tài chính |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Trễ Lợi Suất** (3) | `Return_lag_k`<br>($k \in \{1, 5, 20\}$) | $\text{Return\_lag}_{k, t} = \text{Return\_1D}_{t-k}$ | Phản ánh cấu trúc tự tương quan ngắn, trung và dài hạn. |
| 2 | **Động Lượng Giá** (3) | `Momentum_kD`<br>($k \in \{5, 20, 60\}$) | $\text{Momentum}_{kD, t} = \sum_{j=0}^{k-1} \text{Return\_1D}_{t-j} = \ln(Close_t / Close_{t-k})$ | Động lượng giá tích lũy theo 1 tuần, 1 tháng và 1 quý. |
| 3 | **Biến Động Lịch Sử** (3) | `Volatility_kD`<br>($k \in \{5, 20, 60\}$) | $\text{Volatility}_{kD, t} = \text{std}(\{\text{Return\_1D}_{t-j}\}_{j=0}^{k-1})$ | Rủi ro biến động thực tế trong cửa sổ trượt $k$ ngày. |
| 4 | **Tín Hiệu VIX** (4) | `VIXCLS`, `VIX_lag_1`,<br>`VIX_change_1D`, `VIX_pct_change_1D` | Mức đóng cửa $VIX_t$, trễ $VIX_{t-1}$, thay đổi tuyệt đối $VIX_t - VIX_{t-1}$, và thay đổi tương đối $(VIX_t - VIX_{t-1}) / VIX_{t-1}$ | Thước đo tâm lý sợ hãi và kỳ vọng biến động thị trường quyền chọn. |
| 5 | **Cấu Trúc Phiên** (2) | `HL_range`, `OC_return` | $\text{HL\_range}_t = (High_t - Low_t) / Close_t$<br>$\text{OC\_return}_t = (Close_t - Open_t) / Open_t$ | Biên độ dao động và chiều hướng biến động nội phiên. |
| 6 | **Thanh Khoản** (1) | `Volume_change` | $\text{Volume\_change}_t = (Volume_t - Volume_{t-1}) / Volume_{t-1}$ | Tốc độ thay đổi thanh khoản giao dịch (khử dị biệt $Volume=0$). |

* **Xử lý dứt điểm điểm dị biệt $Volume = 0$:**
  Vào ngày **2023-05-24**, dữ liệu Yahoo Finance của S&P 500 xuất hiện điểm dị biệt $Volume = 0$. Nếu áp dụng trực tiếp `pct_change()`, ngày tiếp theo (2023-05-25) sẽ nhận giá trị $(Volume_t - 0)/0 = +\infty$ (`+inf`). Lỗi này được triệt tiêu hoàn toàn bằng thao tác:
  ```python
  df['Volume'] = df['Volume'].replace(0, np.nan).ffill()
  df['Volume_change'] = df['Volume'].pct_change(1)
  ```

### Giai đoạn 5: Phân Loại Chế Độ Thị Trường Walk-Forward (3.0.0)
* **Mục tiêu:** Xác định trạng thái thị trường (`Volatility_Regime`: Low, Normal, High) phục vụ kiểm định câu hỏi **RQ2 (Regime Moderation)** mà tuyệt đối không gây rò rỉ phân vị tương lai.
* **Nguyên tắc Expanding Window Quantile:**
  - Tại mỗi ngày $t$, chỉ sử dụng chuỗi lịch sử độ biến động 20 phiên từ đầu mẫu đến $t$ ($\le t$), với độ dài tối thiểu 1 năm giao dịch (`min_periods = 252`).
  - Tính ngưỡng động 33% và 67%:
    $$Q_{33}(t) = \text{Quantile}_{0.33}(\{\text{Volatility\_20D}_\tau\}_{\tau=1}^t)$$
    $$Q_{67}(t) = \text{Quantile}_{0.67}(\{\text{Volatility\_20D}_\tau\}_{\tau=1}^t)$$
* **Phân loại nhãn:**
  $$\text{Volatility\_Regime}_t = \begin{cases} \text{Low}, & \text{khi } \text{Volatility\_20D}_t \le Q_{33}(t) \\ \text{Normal}, & \text{khi } Q_{33}(t) < \text{Volatility\_20D}_t \le Q_{67}(t) \\ \text{High}, & \text{khi } \text{Volatility\_20D}_t > Q_{67}(t) \end{cases}$$
* **Phân phối thực tế trên toàn bộ dữ liệu (9,073 dòng):**
  - **Low Volatility:** 3,062 phiên ($33.75\%$)
  - **Normal Volatility:** 2,753 phiên ($30.34\%$)
  - **High Volatility:** 3,258 phiên ($35.91\%$)
* **Phân phối trên tập Validation (Split 2, 1,360 dòng):**
  - **Low Volatility:** 669 phiên ($49.19\%$)
  - **Normal Volatility:** 274 phiên ($20.15\%$)
  - **High Volatility:** 417 phiên ($30.66\%$)

### Giai đoạn 6: Loại Bỏ Dữ Liệu Khuyết Thiếu Có Kiểm Soát (3.0.1)
* **Mục tiêu:** Loại bỏ các hàng $NaN$ phát sinh do đặc tính trễ và nhìn về tương lai của chuỗi thời gian tài chính, bảo đảm bộ dữ liệu sạch 100%.
* **Nguyên nhân và phân bổ loại trừ (chính xác 179 dòng):**
  - **Biên đầu (60 dòng đầu: 1990-01-02 $\to$ 1990-03-26):** Thiếu do các đặc trưng cửa sổ trượt dài nhất cần 60 phiên lịch sử (`Momentum_60D`, `Volatility_60D`).
  - **Biên đuôi (119 dòng cuối: 2026-04-08 $\to$ 2026-09-28):** Thiếu do biến mục tiêu `Target_120` cần 120 phiên tương lai. Tại 119 ngày cuối cùng, không có đủ 120 phiên tương lai để tính toán mức giá $Close_{t+120}$.
* **Kết quả:** Dataset giảm từ 9,252 xuống còn đúng **9,073 dòng dữ liệu hoàn chỉnh** (từ ngày 1990-03-27 đến ngày 2026-04-07), hoàn toàn không còn bất kỳ giá trị khuyết thiếu nào ($NaN = 0$).

### Giai đoạn 7: Phân Chia Tập Dữ Liệu Theo Tuyến Thời Gian (3.0.2)
* **Mục tiêu:** Phân chia ranh giới 3 tập dữ liệu theo trình tự thời gian nghiêm ngặt, tuyệt đối không xáo trộn ngẫu nhiên (Zero Random Shuffling).
* **Cấu trúc phân chia cột `Split` (tỷ lệ 70% - 15% - 15%):**

| Mã Split | Tên tập dữ liệu | Số lượng quan sát | Tỷ lệ phần trăm | Khoảng thời gian giao dịch | Mục đích sử dụng |
| :-: | :--- | :-: | :-: | :---: | :--- |
| **1** | **Training Set** | **6,351 dòng** | **70.00%** | 1990-03-27 $\to$ 2015-06-09 | Huấn luyện khởi tạo ban đầu cho các mô hình học máy và thống kê. |
| **2** | **Validation Set** | **1,360 dòng** | **14.99%** | 2015-06-10 $\to$ 2020-10-30 | Đánh giá Walk-Forward 5 Folds, tối ưu hóa siêu tham số và nghiên cứu RQ1, RQ2, RQ3. |
| **3** | **Hold-out Test Set**| **1,362 dòng** | **15.01%** | 2020-11-02 $\to$ 2026-04-07 | Tập dữ liệu độc lập lưu giữ, kiểm định ngoại mẫu sau cùng (Final Out-of-Sample Evaluation). |
| **Tổng** | **Toàn bộ Dataset** | **9,073 dòng** | **100.00%** | **1990-03-27 $\to$ 2026-04-07** | Chuỗi thời gian 36 năm lịch sử tài chính S&P 500. |

### Giai đoạn 8: Phân Chia Walk-Forward 5 Folds Ngoại Mẫu (3.0.3)
* **Mục tiêu:** Thiết lập cơ chế kiểm định ngoại mẫu tịnh tiến mở rộng (Expanding Window Walk-Forward Validation) trên tập Validation (Split 2).
* **Cấu trúc 5 Folds (mỗi Fold kiểm định đúng 272 phiên):**

| Val_Fold | Cửa sổ Huấn luyện Tịnh tiến (Expanding Train) | Số dòng Train | Cửa sổ Kiểm định Ngoại mẫu (Validation Window) | Số dòng Val | Phân phối Chế độ Biến động (Low / Normal / High) |
| :-: | :---: | :-: | :---: | :-: | :---: |
| **Fold 1** | 1990-03-27 $\to$ 2015-06-09 | 6,351 | 2015-06-10 $\to$ 2016-07-07 | 272 | Low: 78 \| Normal: 86 \| High: 108 |
| **Fold 2** | 1990-03-27 $\to$ 2016-07-07 | 6,623 | 2016-07-08 $\to$ 2017-08-04 | 272 | Low: 234 \| Normal: 27 \| High: 11 |
| **Fold 3** | 1990-03-27 $\to$ 2017-08-04 | 6,895 | 2017-08-07 $\to$ 2018-09-04 | 272 | Low: 192 \| Normal: 31 \| High: 49 |
| **Fold 4** | 1990-03-27 $\to$ 2018-09-04 | 7,167 | 2018-09-05 $\to$ 2019-10-03 | 272 | Low: 95 \| Normal: 76 \| High: 101 |
| **Fold 5** | 1990-03-27 $\to$ 2019-10-03 | 7,439 | 2019-10-04 $\to$ 2020-10-30 | 272 | Low: 70 \| Normal: 54 \| High: 148 |
| **Tổng Val** | — | — | **2015-06-10 $\to$ 2020-10-30** | **1,360** | **Low: 669 \| Normal: 274 \| High: 417** |

* **Tập tin xuất bản cuối cùng:** `dataset/3.0.3_val_folds.csv` (9,073 dòng, 32 cột).

---

## 4. Nhật Ký Điều Chỉnh Kỹ Thuật & Khắc Phục Lỗi Xử Lý Dữ Liệu

Trong quá trình thực nghiệm và kiểm toán kinh tế lượng, hệ thống đã thực hiện **6 cải tiến kỹ thuật quan trọng** nhằm khắc phục lỗi tính toán, loại trừ hiện tượng rò rỉ dữ liệu và bảo đảm độ ổn định thuật toán:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│              CÁC ĐIỀU CHỈNH KỸ THUẬT TIỀN XỬ LÝ NHẰM KHẮC PHỤC LỖI & CHỐNG RÒ RỈ                 │
├────┬─────────────────────────────┬─────────────────────────────────┬─────────────────────────────┤
│ STT│ Vấn đề kỹ thuật / Lỗi       │ Cơ chế xử lý điều chỉnh         │ Tác động đến độ ổn định     │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 1  │ Điểm dị biệt Volume = 0     │ Forward-fill Volume tại         │ Triệt tiêu hoàn toàn +inf,  │
│    │ ngày 2023-05-24 sinh ra     │ 2.2.0_feature.ipynb & phòng thủ │ bảo vệ Ridge, SVR, MLP khỏi │
│    │ Volume_change = +inf.       │ 2 lớp trong utils.load_dataset. │ lỗi nổ ma trận / gradient.  │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 2  │ Mất dòng đầu tiên t=0       │ Tính ln(Close_0 / Open_0) thay  │ Giữ nguyên 100% số quan sát │
│    │ khi tính Return_1D trễ.     │ vì để NaN rồi drop.             │ 9,252 dòng từ điểm bắt đầu. │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 3  │ Look-ahead Bias khi phân    │ Chuyển sang Expanding Window    │ Ngưỡng động không thấy trước│
│    │ loại Volatility_Regime.     │ Quantiles (min_periods = 252).  │ biến động các cuộc khủng hoảng│
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 4  │ Rò rỉ chế độ thị trường     │ Tuyệt đối KHÔNG đưa nhãn        │ Các mô hình giữ vững tính   │
│    │ vào ma trận đặc trưng X.    │ Regime vào X huấn luyện.        │ Global Models, tránh bias.  │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 5  │ Rò rỉ chồng lấn nhãn        │ Áp dụng Marcos López de Prado   │ Triệt tiêu rò rỉ ranh giới  │
│    │ đa kỳ hạn qua ranh giới.    │ Purging: cắt bỏ h - 1 dòng cuối.│ nhãn vươn sang Validation.  │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 6  │ Lệch lịch nghỉ lễ CBOE      │ Áp dụng Forward-Fill cho        │ Bảo đảm tính nhân quả và    │
│    │ sinh ra NaN ở VIXCLS.       │ VIXCLS theo lịch S&P 500.       │ thông tin có sẵn tại t.     │
└────┴─────────────────────────────┴─────────────────────────────────┴─────────────────────────────┘
```

### Chi tiết từng trường hợp xử lý:

1. **Khử điểm dị biệt $Volume = 0$ (2023-05-24) và lỗi $Volume\_change = +\infty$:**
   - *Nguyên nhân:* Ngày 2023-05-24, dữ liệu Yahoo Finance của S&P 500 ghi nhận khối lượng bằng 0. Khi tính phần trăm thay đổi ở phiên kế tiếp (2023-05-25): `(Volume_t - 0) / 0 = +inf`.
   - *Hậu quả:* Giá trị vô cực làm sập các thuật toán giải phương trình chuẩn tắc (Normal Equations), gây `LinAlgError` ở Ridge/ElasticNet và nổ gradient ở mạng nơ-ron (RNN/LSTM/Transformer).
   - *Biện pháp:* 
     + Tại `2.2.0_feature.ipynb`: Thay thế $Volume = 0$ bằng giá trị của phiên trước thông qua `.replace(0, np.nan).ffill()`.
     + Tại `src/training/baselines/utils.py`: Bổ sung lớp kiểm toán phòng thủ: `df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan).ffill()`.
     + Xuất lại toàn bộ các tập tin dữ liệu từ `2.2.0` đến `3.0.3`.

2. **Bảo toàn dữ liệu dòng mở màn $t=0$ khi tính `Return_1D`:**
   - *Nguyên nhân:* Phép tính trễ $t-1$ thông thường sẽ làm dòng đầu tiên (1990-01-02) thành $NaN$. Nếu drop dòng này, tập dữ liệu bị mất một ngày giao dịch quý giá.
   - *Biện pháp:* Tại dòng $t=0$, tính xấp xỉ lợi suất trong ngày bằng $\ln(Close_0 / Open_0)$, bảo toàn trọn vẹn 9,252 phiên ban đầu.

3. **Chuyển đổi phân loại `Volatility_Regime` từ Full-sample sang Walk-Forward Expanding Quantiles:**
   - *Nguyên nhân:* Việc tính phân vị toàn mẫu (toàn bộ 36 năm từ 1990 đến 2026) khiến nhãn thị trường ở những năm 1990 bị "biết trước" phân phối biến động của các cuộc khủng hoảng tương lai (Dotcom 2000, Khủng hoảng tài chính toàn cầu 2008, COVID-19 2020).
   - *Biện pháp:* Chuyển sang cơ chế phân vị động lũy tiến (`expanding(min_periods=252).quantile()`). Tại mỗi thời điểm $t$, ngưỡng chỉ sử dụng thông tin quá khứ từ $1$ đến $t$.

4. **Cô lập `Volatility_Regime` khỏi ma trận đặc trưng $X$:**
   - *Nguyên nhân:* Đưa nhãn trạng thái thị trường vào biến độc lập sẽ làm mô hình bị thiên lệch và vi phạm nguyên tắc kiểm định nhân quả.
   - *Biện pháp:* Nhãn `Volatility_Regime` chỉ được giữ lại như một trường siêu dữ liệu (metadata) phục vụ phân tầng đánh giá hiệu năng ngoại mẫu (Stratified Post-hoc Analysis cho RQ2). Tất cả 14 mô hình đều được huấn luyện dưới dạng **Global Models** chỉ sử dụng 16 biến giải thích trong `FEATURE_COLS`.

5. **Cơ chế Purging Marcos López de Prado cách ly nhãn đa kỳ hạn:**
   - *Nguyên nhân:* Biến mục tiêu $Target_{h, t} = \ln(Close_{t+h} / Close_t)$ với $h > 1$ vươn chân sang tương lai $h$ ngày. Tại $h-1$ dòng cuối của tập Train, nhãn mục tiêu sử dụng giá đóng cửa nằm trong tập Validation.
   - *Biện pháp:* Tích hợp cơ chế tự động cắt bỏ $h-1$ dòng cuối của tập Train trong `src/training/baselines/utils.py` thông qua hàm `get_purged_train_df()`.

6. **Đồng bộ ngày nghỉ lễ VIX bằng Forward-Fill:**
   - *Nguyên nhân:* Do sàn CBOE và sàn NYSE có lịch nghỉ lễ đôi khi không trùng khớp, phép Left-join sinh ra giá trị $NaN$ tại cột `VIXCLS`.
   - *Biện pháp:* Dùng `ffill()` để gán mức VIX mới nhất đã được thị trường công bố, tuân thủ đúng nguyên tắc thông tin có sẵn tại thời điểm $t$.

---

## 5. Nguyên Tắc Kiểm Soát Rò Rỉ Tuyệt Đối & Giả Định Thực Thi

### 5.1 Bộ Ngũ Nguyên Tắc Chống Rò Rỉ (Anti-Leakage Core Rules)
1. **Nguyên tắc Chỉ Nhìn Quá Khứ (Strict Trailing):** Toàn bộ đặc trưng $X_t$ tại dòng $t$ chỉ dùng thông tin đến đúng ngày $t$. Không sử dụng centered window hay giá trị của $t+1$.
2. **Không đưa `Volatility_Regime` vào ma trận $X$:** Giữ các mô hình là Global Models thuần túy, nhãn Regime chỉ dùng để phân nhóm báo cáo kết quả RQ2.
3. **Phân vị động lũy tiến (Expanding Quantiles):** Loại trừ hoàn toàn phân vị toàn mẫu khi phân chia chế độ biến động.
4. **Cô lập chuẩn hóa (Scaler Isolation):** Mọi bộ chuẩn hóa (`StandardScaler`, `MinMaxScaler`) chỉ được `fit` trên tập Training của từng Fold rồi mới `transform` sang tập Validation.
5. **Khử Triệt Để Điểm Dị Biệt & Giá Trị Vô Cực ($Inf$ Sanitization):** Xử lý dứt điểm điểm dị biệt $Volume = 0$ và trang bị kiểm toán phòng thủ trong nạp dữ liệu.

### 5.2 Khoảng Đệm Purging Marcos López de Prado
Theo chuẩn mực kinh tế lượng tài chính hiện đại (*Marcos López de Prado, 2018 - Advances in Financial Machine Learning, Chapter 7*):
Khi dự báo lợi suất tích lũy đa kỳ hạn $Target_{h, t} = \ln(Close_{t+h} / Close_t)$, nhãn tại ngày cuối cùng của tập Train ($T_{\text{train}}$) đòi hỏi mức giá tại $T_{\text{train}} + h$. Nếu không áp dụng khoảng đệm cách ly, có chính xác **$h - 1$ quan sát cuối cùng** của tập Train mang nhãn vươn chân sang tập Validation (Label Overlap across Boundary).

Hệ thống thiết lập cơ chế **Purging tự động** trong `utils.prepare_fold_data()` và `utils.get_purged_train_df()`:
$$\text{Train}_{\text{purged}} = \text{Train}[0 \;:\; N_{\text{train}} - (h - 1)]$$

| Chân Trời ($h$) | Bước Dự Báo | Khoảng Đệm Purging Cắt Bỏ ($h - 1$) | Số Dòng Train Sau Purge (Fold 1) | Mức Độ Rò Rỉ Ranh Giới |
| :---: | :---: | :---: | :---: | :---: |
| **1D** | 1 ngày | **0 dòng** | 6,351 dòng | Tuyệt đối bằng 0 (Không chồng lấn) |
| **5D** | 5 ngày | **4 dòng** | 6,347 dòng | Đã triệt tiêu hoàn toàn |
| **20D** | 20 ngày | **19 dòng** | 6,332 dòng | Đã triệt tiêu hoàn toàn |
| **60D** | 60 ngày | **59 dòng** | 6,292 dòng | Đã triệt tiêu hoàn toàn |
| **120D** | 120 ngày | **119 dòng** | 6,232 dòng | Đã triệt tiêu hoàn toàn |

Nhờ cơ chế Purging, giá trị mục tiêu cuối cùng trong tập Train đạt trạng thái đóng kín: $t^* + h = (T_{\text{train}} - h + 1) + h = T_{\text{train}} + 1$, hoàn toàn không xâm lấn vào bất kỳ ngày giao dịch nào thuộc tập Validation.

### 5.3 Giả Định Khớp Lệnh MOC & Ma Sát Giao Dịch Sống
1. **Chuẩn Mực Học Thuật Quốc Tế (Academic Benchmark):**
   Đề tài tuân thủ giả định **Market-On-Close (MOC)** – chuẩn mực được thừa nhận rộng rãi trong các nghiên cứu kinh tế lượng tài chính kinh điển (*Campbell & Shiller 1988, Campbell & Thompson 2008, Welch & Goyal 2008, Gu, Kelly & Xiu 2020*).
   - Véc-tơ đặc trưng $X_t$ (bao gồm giá đóng cửa $Close_t$, biên độ $HL\_range_t$, biến động $OC\_return_t$, chỉ số $VIXCLS_t$) được tổng hợp tại thời điểm đóng cửa ngày $t$.
   - Vị thế được giả định khớp tại đúng mức giá $Close_t$ (thông qua lệnh ATC / MOC trên sàn chứng khoán).
   - Do đó, mục tiêu lợi suất $Target_{h, t} = \ln(Close_{t+h} / Close_t)$ phản ánh chính xác năng lực dự báo của tập đặc trưng $X_t$ mà không vi phạm tính nhân quả hay làm sai lệch phân phối lợi suất.
2. **Góc Nhìn Thực Tế Giao Dịch Sống (Live Algorithmic Execution):**
   Trong thực tế vận hành quỹ định lượng, nếu mô hình đòi hỏi thời gian tính toán qua đêm và chỉ có thể đẩy lệnh vào đầu phiên hôm sau, mốc giá vào vị thế thực tế sẽ là giá mở cửa $Open_{t+1}$. Lợi suất khả thi giao dịch khi đó chịu ma sát chênh lệch giá qua đêm:
   $$\text{Tradeable\_Return}_{h, t} = \ln\left(\frac{Close_{t+h}}{Open_{t+1}}\right) = Target_{h, t} - \ln\left(\frac{Open_{t+1}}{Close_t}\right)$$
   Việc duy trì thiết kế chuẩn tắc Close-to-Close (MOC) đảm bảo tính nhất quán lý thuyết phục vụ mục tiêu trả lời 3 câu hỏi nghiên cứu (RQ1, RQ2, RQ3), đồng thời lưu ý rõ ràng ma sát thực thi cho bất kỳ ứng dụng mở rộng nào trong thực tế.
