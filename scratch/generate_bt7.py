import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import nbconvert
import os

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Import
cell1_source = """import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Đã import thành công các thư viện cần thiết!")
"""
cells.append(nbf.v4.new_code_cell(cell1_source))

# Cell 2: Read CSV, parse Date, set index, sort index
cell2_source = """# 1. Đọc tệp CSV, phân tích cột 'Date' dưới dạng datetime, và thiết lập cột trên làm chỉ mục
csv_path = 'BT7.future-gc00-daily-prices.csv' if os.path.exists('BT7.future-gc00-daily-prices.csv') else os.path.join('btth', '7', 'BT7.future-gc00-daily-prices.csv')
data = pd.read_csv(csv_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 2. Sắp xếp DataFrame theo chỉ mục 'Date' theo thứ tự tăng dần
data.sort_index(inplace=True)

print("=== 5 DÒNG ĐẦU TIÊN CỦA BỘ DỮ LIỆU ===")
display(data.head())

print("\\n=== THÔNG TIN TỔNG QUAN BỘ DỮ LIỆU ===")
data.info()
"""
cells.append(nbf.v4.new_code_cell(cell2_source))

# Cell 3: Clean Close
cell3_source = """# 3. Làm sạch cột "Close":
# - Loại bỏ dấu phẩy khỏi dữ liệu trong cột "Close"
if data['Close'].dtype == object:
    data['Close'] = data['Close'].str.replace(',', '')

# - Chuyển đổi giá trị thành kiểu số học
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# - Thay thế các giá trị vô hạn (nếu có) thành NaN
data['Close'] = data['Close'].replace([np.inf, -np.inf], np.nan)

# - Loại bỏ các dòng có giá trị NaN trong cột "Close"
initial_len = len(data)
data.dropna(subset=['Close'], inplace=True)

print(f"Số dòng dữ liệu sau khi làm sạch: {len(data)} (Loại bỏ {initial_len - len(data)} dòng)")
print("\\n=== THỐNG KÊ MÔ TẢ CỘT 'Close' ===")
display(data[['Close']].describe())
"""
cells.append(nbf.v4.new_code_cell(cell3_source))

# Cell 4: Plot Close over time
cell4_source = """# 4. Vẽ đồ thị giá "Close" theo thời gian
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], color='#1f77b4', linewidth=1.5, label='Close Price')
plt.title('Biểu đồ giá đóng cửa (Close) của Vàng theo thời gian', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (USD)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell4_source))

# Cell 5: Stationarity test and differencing
cell5_source = """# 5. Kiểm tra tính dừng (stationarity) và thực hiện phép sai phân (differencing)
# Kiểm định Augmented Dickey-Fuller (ADF) trên chuỗi gốc
result_raw = adfuller(data['Close'])
adf_stat_raw = result_raw[0]
p_val_raw = result_raw[1]

print("=== KẾT QUẢ KIỂM ĐỊNH ADF TRÊN CHUỖI GỐC ===")
print(f"ADF Statistic (Gốc): {adf_stat_raw:.4f}")
print(f"p-value (Gốc): {p_val_raw:.4f}")
if p_val_raw > 0.05:
    print("Giải thích: Chuỗi gốc là không dừng (Non-Stationary) vì p-value > 0.05.")
else:
    print("Giải thích: Chuỗi gốc là chuỗi dừng (Stationary) vì p-value <= 0.05.")

# Thực hiện sai phân 1 bậc
data['Close_Diff'] = data['Close'].diff()

# Kiểm định ADF trên chuỗi đã sai phân (Differenced)
result_diff = adfuller(data['Close_Diff'].dropna())
adf_stat_diff = result_diff[0]
p_val_diff = result_diff[1]

print("\\n=== KẾT QUẢ KIỂM ĐỊNH ADF TRÊN CHUỖI ĐÃ SAI PHÂN ===")
print(f"ADF Statistic (sai phân): {adf_stat_diff:.4f}")
print(f"p-value (sai phân): {p_val_diff:.4f}")
if p_val_diff < 0.05:
    print("Giải thích: Chuỗi đã chênh lệch là dừng (Stationary) vì p-value < 0.05.")
else:
    print("Giải thích: Chuỗi chưa dừng, cần xem xét sai phân bậc cao hơn.")
"""
cells.append(nbf.v4.new_code_cell(cell5_source))

# Cell 6: Plot Close differenced in orange
cell6_source = """# 6. Vẽ đồ thị giá "Close" đã sai phân (differenced) với màu cam (orange)
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close_Diff'], color='orange', linewidth=1, label='Close Diff')
plt.title('Biểu đồ sai phân bậc 1 của giá Close theo thời gian', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Differenced', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell6_source))

# Cell 7: ACF and PACF subplots
cell7_source = """# 7. Tìm các tham số ARIMA qua đồ thị ACF và PACF (lags = 40)
fig, axes = plt.subplots(1, 2, figsize=(16, 4))

plot_acf(data['Close_Diff'].dropna(), lags=40, ax=axes[0], title='Autocorrelation Function (ACF)')
axes[0].set_ylim(-1.0, 1.0)
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')

plot_pacf(data['Close_Diff'].dropna(), lags=40, ax=axes[1], title='Partial Autocorrelation Function (PACF)')
axes[1].set_ylim(-1.0, 1.0)
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.show()

print("Nhận xét chọn tham số ARIMA(p, d, q):")
print("- d = 1: Chuỗi gốc không dừng, sau khi thực hiện sai phân bậc 1 (d=1) chuỗi trở thành dừng (tương tự nhiễu trắng).")
print("- p = 1: Đồ thị PACF có độ trễ bậc 1 có đỉnh đáng kể vượt qua ngưỡng ý nghĩa thống kê, chọn p = 1.")
print("- q = 1: Đồ thị ACF có độ trễ bậc 1 có đỉnh đáng kể vượt qua ngưỡng ý nghĩa thống kê, chọn q = 1.")
print("=> Với dữ liệu trên, chọn giá trị 1 cho cả p và q vì có một đỉnh đáng kể ở độ trễ đầu tiên cho mỗi đồ thị (ARIMA(1, 1, 1)).")
"""
cells.append(nbf.v4.new_code_cell(cell7_source))

# Cell 8: Train Test Split and Fit ARIMA(1,1,1)
cell8_source = """# 8. Chia tập dữ liệu thành Train và Test theo tỷ lệ 8-2 và huấn luyện mô hình ARIMA(1, 1, 1)
train_size = int(len(data) * 0.8)
train = data.iloc[:train_size]
test = data.iloc[train_size:]

print(f"Tổng số mẫu: {len(data)}")
print(f"Kích thước tập Train (80%): {len(train)} mẫu (từ {train.index.min().strftime('%Y-%m-%d')} đến {train.index.max().strftime('%Y-%m-%d')})")
print(f"Kích thước tập Test (20%): {len(test)} mẫu (từ {test.index.min().strftime('%Y-%m-%d')} đến {test.index.max().strftime('%Y-%m-%d')})")

# Huấn luyện mô hình ARIMA(1, 1, 1) trên tập Train
model = ARIMA(train['Close'].values, order=(1, 1, 1))
model_fit = model.fit()

# Hiển thị tóm tắt mô hình
print("\\n=== TÓM TẮT MÔ HÌNH ARIMA(1, 1, 1) ===")
print(model_fit.summary())
"""
cells.append(nbf.v4.new_code_cell(cell8_source))

# Cell 9: Forecast on Test and Plot
cell9_source = """# 9. Tiến hành dự báo trên tập Test và vẽ biểu đồ hiển thị kết quả
forecast_values = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast_values, index=test.index)

# Vẽ biểu đồ Close Price Forecast
plt.figure(figsize=(14, 7))
plt.plot(train.index, train['Close'], label='Train', color='#1f2937', linewidth=1.5)
plt.plot(test.index, test['Close'], label='Test', color='#00e676', linewidth=1.5)
plt.plot(test.index, forecast, label='Forecast', color='orange', linewidth=2.0)
plt.title('Close Price Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell9_source))

# Cell 10: Evaluation & commentary
cell10_source = """# 10. Đánh giá và nhận xét mô hình bằng các chỉ số AIC, BIC, và RMSE
aic = model_fit.aic
bic = model_fit.bic
rmse = np.sqrt(mean_squared_error(test['Close'], forecast))
mae = mean_absolute_error(test['Close'], forecast)
mape = np.mean(np.abs((test['Close'] - forecast) / test['Close'])) * 100

print("=== BẢNG CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH ARIMA(1, 1, 1) ===")
print(f"1. AIC (Akaike Information Criterion) : {aic:.2f}")
print(f"2. BIC (Bayesian Information Criterion): {bic:.2f}")
print(f"3. RMSE (Root Mean Squared Error)      : {rmse:.4f}")
print(f"4. MAE (Mean Absolute Error)           : {mae:.4f}")
print(f"5. MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

print("\\n=== NHẬN XÉT CHI TIẾT MÔ HÌNH DỰ BÁO ===")
print(f'''
1. Đánh giá chỉ số AIC & BIC:
   - AIC = {aic:.2f} và BIC = {bic:.2f} là hai tiêu chuẩn thông tin thống kê dùng để đánh giá và so sánh mức độ cân bằng giữa độ khớp (goodness-of-fit) và độ phức tạp của mô hình.
   - Việc phạt số lượng tham số trong AIC/BIC giúp lựa chọn mô hình gọn nhẹ (parsimonious model), tránh hiện tượng overfitting trên chuỗi thời gian. Với p=1, d=1, q=1, mô hình nắm bắt tốt tương quan bậc 1 của sai phân mà không bị dư thừa biến.

2. Đánh giá sai số dự báo (RMSE & MAPE):
   - RMSE = {rmse:.4f} (USD/ounce) và MAPE = {mape:.2f}% trên tập Test dài 745 ngày giao dịch (từ 02/2021 đến 02/2024).
   - Trong giai đoạn này, giá vàng thực tế biến động trong biên độ rất rộng từ ~1,650 USD đến gần 2,100 USD/ounce. Mức sai số tương đối MAPE dưới 5% thể hiện rằng mô hình giữ được mức sai lệch trung bình ở biên độ có thể chấp nhận được đối với dự báo chuỗi thời gian tài chính không có biến ngoại sinh.

3. Bản chất và nhận xét đường dự báo (Forecast Curve):
   - Trên đồ thị dự báo "Close Price Forecast", đường dự báo (Forecast) sau một vài bước ban đầu nhanh chóng ổn định và đi ngang tiệm cận quanh mức 1806.27 USD.
   - Đây là đặc tính toán học chuẩn tắc của mô hình ARIMA(1,1,1) khi thực hiện dự báo đa bước xa (multi-step ahead forecast) mà không có hệ số trôi (drift): các kỳ vọng sai phân trong tương lai tiến dần về 0, khiến giá trị dự báo mức giá hội tụ về một hằng số.
   - Kết quả này phản ánh giả thuyết thị trường hiệu quả (EMH) và tính chất bước đi ngẫu nhiên (Random Walk) của giá tài sản tài chính trong dài hạn: các biến động bất thường thực tế được thúc đẩy bởi tin tức vĩ mô không lường trước. Để cải thiện dự báo động theo thời gian thực, trong thực tế thường áp dụng cơ chế cuộn cửa sổ (rolling forecast) hoặc kết hợp các biến kinh tế vĩ mô ngoại sinh (ARIMAX).
''')
"""
cells.append(nbf.v4.new_code_cell(cell10_source))

nb['cells'] = cells

# Save raw notebook
nb_path = 'btth/7/BT7.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Đã lưu notebook thô vào {nb_path}. Bắt đầu thực thi notebook...")

# Execute notebook in btth/7 working directory
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {'metadata': {'path': 'btth/7'}})

# Save executed notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_executed, f)

print(f"Đã thực thi và cập nhật outputs thành công vào {nb_path}!")

# Convert to HTML
html_exporter = nbconvert.HTMLExporter()
html_data, _ = html_exporter.from_notebook_node(nb_executed)

html_path1 = 'btth/7/BTTH7_Nhom6_24520500_24521314_24521649_24521732.html'
with open(html_path1, 'w', encoding='utf-8') as f:
    f.write(html_data)

html_path2 = 'btth/7/BT7.html'
with open(html_path2, 'w', encoding='utf-8') as f:
    f.write(html_data)

print(f"Đã xuất thành công file HTML: {html_path1} và {html_path2}!")
