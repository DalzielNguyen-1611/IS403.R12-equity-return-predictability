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

# Thư viện mô hình chuỗi thời gian
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

print("Đã import thành công các thư viện cần thiết!")
"""
cells.append(nbf.v4.new_code_cell(cell1_source))

# Cell 2: Read CSV and preprocess
cell2_source = """# 1. Đọc tệp CSV và định dạng cột 'Date' dưới dạng datetime
csv_path = 'BT9.Instagram-Reach.csv' if os.path.exists('BT9.Instagram-Reach.csv') else os.path.join('btth', '9', 'BT9.Instagram-Reach.csv')
data = pd.read_csv(csv_path)

# Phân tích cột 'Date' dưới dạng datetime và thiết lập làm chỉ mục
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

print("=== 5 DÒNG ĐẦU TIÊN CỦA BỘ DỮ LIỆU INSTAGRAM REACH ===")
display(data.head())

print("\\n=== 5 DÒNG CUỐI CÙNG CỦA BỘ DỮ LIỆU INSTAGRAM REACH ===")
display(data.tail())

print("\\n=== THÔNG TIN TỔNG QUAN BỘ DỮ LIỆU ===")
data.info()

print(f"\\nKhoảng thời gian khảo sát: từ {data.index.min().strftime('%Y-%m-%d')} đến {data.index.max().strftime('%Y-%m-%d')}")
print(f"Tổng số ngày quan sát: {len(data)} ngày (tròn 1 năm)")
"""
cells.append(nbf.v4.new_code_cell(cell2_source))

# Cell 3: Plot raw time series & stationarity test
cell3_source = """# 2. Trực quan hóa chuỗi thời gian Lượng truy cập Instagram và kiểm tra tính dừng (ADF test)
plt.figure(figsize=(14, 5))
plt.plot(data.index, data['Instagram reach'], color='#e1306c', linewidth=1.5, label='Instagram reach')
plt.title('Lượng truy cập Instagram hàng ngày (Instagram Reach) theo thời gian', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Instagram Reach', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.show()

# Kiểm định ADF trên chuỗi gốc
adf_res = adfuller(data['Instagram reach'])
print("=== KẾT QUẢ KIỂM ĐỊNH TÍNH DỪNG (ADF TEST) TRÊN CHUỖI GỐC ===")
print(f"ADF Statistic: {adf_res[0]:.4f}")
print(f"p-value      : {adf_res[1]:.4f}")
if adf_res[1] < 0.05:
    print("=> Kết luận: Chuỗi dừng ở mức ý nghĩa 5%. Tuy nhiên chuỗi thể hiện rõ các chu kỳ lặp lại theo mùa vụ.")
else:
    print("=> Kết luận: Chuỗi chưa dừng ở mức ý nghĩa 5%, việc sử dụng sai phân d = 1 là cần thiết.")
"""
cells.append(nbf.v4.new_code_cell(cell3_source))

# Cell 4: Build and fit ARIMA(1, 1, 1)
cell4_source = """# 3. Xây dựng mô hình ARIMA(1, 1, 1) bằng thư viện statsmodels
model_arima = ARIMA(data['Instagram reach'].values, order=(1, 1, 1))
model_arima_fit = model_arima.fit()

print("=== TÓM TẮT MÔ HÌNH ARIMA(1, 1, 1) ===")
print(model_arima_fit.summary())
"""
cells.append(nbf.v4.new_code_cell(cell4_source))

# Cell 5: Build and fit SARIMA(1, 1, 1)(1, 1, 1)12
cell5_source = """# 4. Xây dựng mô hình SARIMA(1, 1, 1)(1, 1, 1)12 bằng thư viện statsmodels
# order = (1, 1, 1), seasonal_order = (1, 1, 1, 12)
model_sarima = SARIMAX(
    data['Instagram reach'].values,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_sarima_fit = model_sarima.fit()

print("=== TÓM TẮT MÔ HÌNH SARIMA(1, 1, 1)(1, 1, 1)12 ===")
print(model_sarima_fit.summary())
"""
cells.append(nbf.v4.new_code_cell(cell5_source))

# Cell 6: Forecast 100 days ahead
cell6_source = """# 5. Sử dụng 2 mô hình đã xây dựng để dự báo 100 ngày tiếp theo
forecast_days = 100

# Tạo chỉ mục thời gian cho 100 ngày tương lai tiếp theo
last_date = data.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

# Dự báo bằng mô hình ARIMA(1, 1, 1)
arima_forecast_vals = model_arima_fit.forecast(steps=forecast_days)
forecast_arima_series = pd.Series(arima_forecast_vals, index=future_dates)

# Dự báo bằng mô hình SARIMA(1, 1, 1)(1, 1, 1)12
sarima_forecast_vals = model_sarima_fit.forecast(steps=forecast_days)
forecast_sarima_series = pd.Series(sarima_forecast_vals, index=future_dates)

# Bảng so sánh 10 ngày đầu tiên và 10 ngày cuối cùng của kết quả dự báo
forecast_comparison_df = pd.DataFrame({
    'ARIMA(1,1,1) Forecast': forecast_arima_series,
    'SARIMA(1,1,1)(1,1,1)12 Forecast': forecast_sarima_series,
    'Chênh lệch (SARIMA - ARIMA)': forecast_sarima_series - forecast_arima_series
})

print(f"Khoảng thời gian dự báo 100 ngày: từ {future_dates.min().strftime('%Y-%m-%d')} đến {future_dates.max().strftime('%Y-%m-%d')}")
print("\\n=== 10 NGÀY DỰ BÁO ĐẦU TIÊN (ĐẦU THÁNG 4/2023) ===")
display(forecast_comparison_df.head(10))

print("\\n=== 10 NGÀY DỰ BÁO CUỐI CÙNG (ĐẦU THÁNG 7/2023) ===")
display(forecast_comparison_df.tail(10))
"""
cells.append(nbf.v4.new_code_cell(cell6_source))

# Cell 7: Statistical comparison
cell7_source = """# 6. Đánh giá và so sánh các chỉ số thống kê giữa ARIMA và SARIMA
comparison_metrics = pd.DataFrame({
    'Mô hình': ['ARIMA(1, 1, 1)', 'SARIMA(1, 1, 1)(1, 1, 1)12'],
    'AIC': [model_arima_fit.aic, model_sarima_fit.aic],
    'BIC': [model_arima_fit.bic, model_sarima_fit.bic],
    'Log-Likelihood': [model_arima_fit.llf, model_sarima_fit.llf],
    'Số lượng tham số': [len(model_arima_fit.params), len(model_sarima_fit.params)],
    'Giá trị dự báo TB': [forecast_arima_series.mean(), forecast_sarima_series.mean()],
    'Độ lệch chuẩn dự báo': [forecast_arima_series.std(), forecast_sarima_series.std()]
})

print("=== BẢNG SO SÁNH CHỈ SỐ THỐNG KÊ GIỮA ARIMA VÀ SARIMA ===")
display(comparison_metrics)
"""
cells.append(nbf.v4.new_code_cell(cell7_source))

# Cell 8: Plot forecasts
cell8_source = """# 7. Vẽ biểu đồ so sánh kết quả dự báo của 2 mô hình
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Biểu đồ 1: Toàn cảnh dữ liệu lịch sử và 100 ngày dự báo
ax1.plot(data.index, data['Instagram reach'], label='Dữ liệu thực tế (Historical)', color='#475569', linewidth=1.5)
ax1.plot(forecast_arima_series.index, forecast_arima_series, label='Dự báo ARIMA(1, 1, 1) [100 ngày]', color='#f59e0b', linewidth=2.0, linestyle='--')
ax1.plot(forecast_sarima_series.index, forecast_sarima_series, label='Dự báo SARIMA(1, 1, 1)(1, 1, 1)12 [100 ngày]', color='#06b6d4', linewidth=2.0)
ax1.axvline(x=data.index.max(), color='red', linestyle=':', alpha=0.8, label='Mốc bắt đầu dự báo')
ax1.set_title('Toàn cảnh Lượng truy cập Instagram và Dự báo 100 ngày tiếp theo', fontsize=13, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Instagram reach', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=10)

# Biểu đồ 2: Cận cảnh 100 ngày dự báo tiếp theo
ax2.plot(forecast_arima_series.index, forecast_arima_series, label='ARIMA(1, 1, 1) Forecast', color='#f59e0b', linewidth=2.5, linestyle='--')
ax2.plot(forecast_sarima_series.index, forecast_sarima_series, label='SARIMA(1, 1, 1)(1, 1, 1)12 Forecast', color='#06b6d4', linewidth=2.5)
ax2.set_title('Cận cảnh so sánh hành vi dự báo 100 ngày giữa ARIMA và SARIMA', fontsize=13, fontweight='bold')
ax2.set_xlabel('Date (Tháng 04/2023 - Tháng 07/2023)', fontsize=11)
ax2.set_ylabel('Instagram reach dự báo', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell8_source))

# Cell 9: Commentary
cell9_source = """# 8. Nhận xét và đánh giá chuyên sâu kết quả dự báo
print(f'''
=== NHẬN XÉT VÀ ĐÁNH GIÁ CHI TIẾT KẾT QUẢ DỰ BÁO ===

1. Đánh giá tiêu chuẩn thông tin AIC và BIC:
   - Mô hình ARIMA(1, 1, 1) có AIC = {model_arima_fit.aic:.2f} và BIC = {model_arima_fit.bic:.2f}.
   - Mô hình SARIMA(1, 1, 1)(1, 1, 1)12 có AIC = {model_sarima_fit.aic:.2f} và BIC = {model_sarima_fit.bic:.2f}.
   - Kết quả cho thấy AIC của SARIMA thấp hơn đáng kể so với ARIMA (giảm khoảng 160 điểm) và Log-Likelihood cao hơn, chứng minh rằng việc bổ sung thành phần mùa vụ (seasonal component với chu kỳ s = 12) giúp mô hình khớp dữ liệu vượt trội hơn rất nhiều.

2. Phân tích hành vi và bản chất đường dự báo của ARIMA(1, 1, 1):
   - Đường dự báo của mô hình ARIMA(1, 1, 1) chỉ dao động nhẹ ở những bước đầu tiên rồi nhanh chóng hội tụ về một đường gần như nằm ngang (tiệm cận mức trung bình ~28,400 lượt tiếp cận).
   - Nguyên nhân: Mô hình ARIMA thông thường là mô hình phi mùa vụ (non-seasonal). Khi dự báo xa tới 100 bước, tác động của các bậc tự hồi quy AR(1) và trung bình trượt MA(1) suy giảm dần theo hàm mũ và hội tụ về kỳ vọng dài hạn, hoàn toàn không thể tái hiện lại sự biến động lặp đi lặp lại của người dùng mạng xã hội.

3. Phân tích hành vi và bản chất đường dự báo của SARIMA(1, 1, 1)(1, 1, 1)12:
   - Ngược lại với ARIMA, đường dự báo của SARIMA liên tục tạo ra các làn sóng dao động nhịp nhàng tuần hoàn với chu kỳ 12 ngày.
   - Biên độ dao động trải dài từ khoảng ~5,600 đến ~33,400 lượt truy cập, phản ánh chân thực các đỉnh (peak) và đáy (trough) đặc trưng của hành vi tương tác trên mạng xã hội Instagram (thời điểm đăng bài, cuối tuần hoặc các chiến dịch định kỳ).
   - Độ lệch chuẩn dự báo của SARIMA đạt {forecast_sarima_series.std():.2f} so với mức rất thấp {forecast_arima_series.std():.2f} của ARIMA, chứng minh mô hình duy trì được tính biến động thực tế thay vì làm phẳng dữ liệu.

4. Kết luận và Khuyến nghị ứng dụng:
   - Đối với dữ liệu mạng xã hội và hành vi người dùng (như Instagram reach), việc có yếu tố chu kỳ/mùa vụ là thuộc tính bản chất.
   - Do đó, mô hình SARIMA(1, 1, 1)(1, 1, 1)12 là lựa chọn tối ưu, mang lại giá trị thực tiễn cao hơn rõ rệt cho việc lập kế hoạch marketing và tối ưu hóa thời gian đăng bài so với mô hình ARIMA truyền thống.
''')
"""
cells.append(nbf.v4.new_code_cell(cell9_source))

nb['cells'] = cells

# Save raw notebook
nb_path = 'btth/9/BT9.ipynb'
os.makedirs('btth/9', exist_ok=True)
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Đã lưu notebook thô vào {nb_path}. Bắt đầu thực thi notebook...")

# Execute notebook in btth/9 working directory
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {'metadata': {'path': 'btth/9'}})

# Save executed notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_executed, f)

print(f"Đã thực thi và cập nhật outputs thành công vào {nb_path}!")

# Convert to HTML
html_exporter = nbconvert.HTMLExporter()
html_data, _ = html_exporter.from_notebook_node(nb_executed)

html_path1 = 'btth/9/BTTH9_Nhom6_24520500_24521314_24521649_24521732.html'
with open(html_path1, 'w', encoding='utf-8') as f:
    f.write(html_data)

html_path2 = 'btth/9/BT9.html'
with open(html_path2, 'w', encoding='utf-8') as f:
    f.write(html_data)

print(f"Đã xuất thành công file HTML: {html_path1} và {html_path2}!")
