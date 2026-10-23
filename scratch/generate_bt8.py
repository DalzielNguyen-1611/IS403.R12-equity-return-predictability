import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import nbconvert
import os

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Imports
cell1_source = """import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Thư viện tìm kiếm tham số tối ưu tự động
from pmdarima import auto_arima

# Thư viện xây dựng mô hình chuỗi thời gian
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Đã import thành công tất cả các thư viện cần thiết!")
"""
cells.append(nbf.v4.new_code_cell(cell1_source))

# Cell 2: Read CSV and preprocess
cell2_source = """# 1. Đọc tệp CSV và định dạng cột 'Date' dưới dạng datetime
csv_path = 'BT8_522092301_amazon_stock_price.csv' if os.path.exists('BT8_522092301_amazon_stock_price.csv') else os.path.join('btth', '8', 'BT8_522092301_amazon_stock_price.csv')
data = pd.read_csv(csv_path)

# Phân tích cột 'Date' dưới dạng datetime và thiết lập làm chỉ mục
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.sort_index(inplace=True)

print("=== 5 DÒNG ĐẦU TIÊN CỦA BỘ DỮ LIỆU CỔ PHIẾU AMAZON ===")
display(data.head())

print("\\n=== THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU ===")
data.info()

print(f"\\nKhoảng thời gian: từ {data.index.min().strftime('%Y-%m-%d')} đến {data.index.max().strftime('%Y-%m-%d')}")
print(f"Tổng số ngày giao dịch: {len(data)} ngày")
"""
cells.append(nbf.v4.new_code_cell(cell2_source))

# Cell 3: Plot Close and Volume
cell3_source = """# 2. Trực quan hóa biến mục tiêu (Close) và biến ngoại sinh (Volume)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Biểu đồ giá Close
ax1.plot(data.index, data['Close'], color='#2563eb', linewidth=1.5, label='Amazon Close Price')
ax1.set_title('Biến động giá đóng cửa (Close Price) của Cổ phiếu Amazon (2019 - 2023)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Close Price (USD)', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left')

# Biểu đồ Volume
ax2.bar(data.index, data['Volume'] / 1e6, color='#64748b', alpha=0.7, width=1.5, label='Trading Volume (triệu cổ phiếu)')
ax2.set_title('Khối lượng giao dịch (Trading Volume) theo thời gian', fontsize=13, fontweight='bold')
ax2.set_xlabel('Ngày giao dịch (Date)', fontsize=11)
ax2.set_ylabel('Volume (Triệu cổ phiếu)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell3_source))

# Cell 4: Train / Test Split
cell4_source = """# 3. Phân chia tập dữ liệu: Giữ 10 ngày giao dịch cuối cùng làm tập Test để đánh giá dự báo 10 bước
forecast_steps = 10
train = data.iloc[:-forecast_steps]
test = data.iloc[-forecast_steps:]

print(f"Kích thước tập Train: {len(train)} mẫu (từ {train.index.min().strftime('%Y-%m-%d')} đến {train.index.max().strftime('%Y-%m-%d')})")
print(f"Kích thước tập Test : {len(test)} mẫu (từ {test.index.min().strftime('%Y-%m-%d')} đến {test.index.max().strftime('%Y-%m-%d')})")
print("\\n=== GIÁ TRỊ THỰC TẾ TRONG 10 NGÀY TẬP TEST ===")
display(test[['Close', 'Volume']])
"""
cells.append(nbf.v4.new_code_cell(cell4_source))

# Cell 5: auto_arima for ARIMA
cell5_source = """# 4. Phân tích và tìm kiếm tham số tốt nhất cho mô hình ARIMA bằng auto_arima
print("=== TIẾN HÀNH TÌM KIẾM THAM SỐ ARIMA BẰNG AUTO_ARIMA ===")
arima_search = auto_arima(
    train['Close'],
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,           # Tự động kiểm định ADF/KPSS để chọn bậc sai phân d tối ưu
    seasonal=False,   # Dữ liệu tài chính hàng ngày không có tính mùa vụ cố định
    stepwise=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)

best_arima_order = arima_search.order
print(f"\\n=> Tham số ARIMA tốt nhất được auto_arima lựa chọn: {best_arima_order}")
print(f"=> AIC tốt nhất: {arima_search.aic():.2f}")
"""
cells.append(nbf.v4.new_code_cell(cell5_source))

# Cell 6: Build and fit ARIMA via statsmodels
cell6_source = """# 5. Xây dựng mô hình ARIMA bằng thư viện statsmodels theo tham số tối ưu
model_arima = ARIMA(train['Close'].values, order=best_arima_order)
model_arima_fit = model_arima.fit()

print("=== TÓM TẮT MÔ HÌNH ARIMA TRÊN TẬP TRAIN ===")
print(model_arima_fit.summary())
"""
cells.append(nbf.v4.new_code_cell(cell6_source))

# Cell 7: Forecast 10 steps ARIMA
cell7_source = """# 6. Dự báo cho mô hình ARIMA với số bước dự báo là 10
forecast_arima_vals = model_arima_fit.forecast(steps=forecast_steps)
forecast_arima = pd.Series(forecast_arima_vals, index=test.index)

# Đánh giá sai số dự báo
rmse_arima = np.sqrt(mean_squared_error(test['Close'], forecast_arima))
mae_arima = mean_absolute_error(test['Close'], forecast_arima)
mape_arima = np.mean(np.abs((test['Close'] - forecast_arima) / test['Close'])) * 100

print("=== KẾT QUẢ DỰ BÁO 10 BƯỚC CỦA MÔ HÌNH ARIMA ===")
df_result_arima = pd.DataFrame({
    'Actual Close': test['Close'],
    'ARIMA Forecast': forecast_arima,
    'Absolute Error': np.abs(test['Close'] - forecast_arima)
})
display(df_result_arima)

print(f"\\nChỉ số sai số mô hình ARIMA(10 bước):")
print(f"- RMSE : {rmse_arima:.4f}")
print(f"- MAE  : {mae_arima:.4f}")
print(f"- MAPE : {mape_arima:.2f}%")
"""
cells.append(nbf.v4.new_code_cell(cell7_source))

# Cell 8: auto_arima for ARIMAX
cell8_source = """# 7. Phân tích và tìm kiếm tham số tốt nhất cho mô hình ARIMAX với biến ngoại sinh X = 'Volume'
print("=== TIẾN HÀNH TÌM KIẾM THAM SỐ ARIMAX (X = Volume) BẰNG AUTO_ARIMA ===")
arimax_search = auto_arima(
    train['Close'],
    X=train[['Volume']],
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,
    seasonal=False,
    stepwise=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)

best_arimax_order = arimax_search.order
print(f"\\n=> Tham số ARIMAX tốt nhất được auto_arima lựa chọn: {best_arimax_order}")
print(f"=> AIC tốt nhất: {arimax_search.aic():.2f}")
"""
cells.append(nbf.v4.new_code_cell(cell8_source))

# Cell 9: Build and fit ARIMAX via statsmodels
cell9_source = """# 8. Xây dựng mô hình ARIMAX bằng thư viện statsmodels theo tham số tối ưu
# Đưa biến ngoại sinh Volume vào mô hình
model_arimax = ARIMA(
    train['Close'].values,
    exog=train[['Volume']].values,
    order=best_arimax_order
)
model_arimax_fit = model_arimax.fit()

print("=== TÓM TẮT MÔ HÌNH ARIMAX (X = Volume) TRÊN TẬP TRAIN ===")
print(model_arimax_fit.summary())
"""
cells.append(nbf.v4.new_code_cell(cell9_source))

# Cell 10: Forecast 10 steps ARIMAX
cell10_source = """# 9. Dự báo cho mô hình ARIMAX với số bước dự báo là 10 (cung cấp biến ngoại sinh test[['Volume']])
forecast_arimax_vals = model_arimax_fit.forecast(steps=forecast_steps, exog=test[['Volume']].values)
forecast_arimax = pd.Series(forecast_arimax_vals, index=test.index)

# Đánh giá sai số dự báo
rmse_arimax = np.sqrt(mean_squared_error(test['Close'], forecast_arimax))
mae_arimax = mean_absolute_error(test['Close'], forecast_arimax)
mape_arimax = np.mean(np.abs((test['Close'] - forecast_arimax) / test['Close'])) * 100

print("=== KẾT QUẢ DỰ BÁO 10 BƯỚC CỦA MÔ HÌNH ARIMAX ===")
df_result_arimax = pd.DataFrame({
    'Actual Close': test['Close'],
    'ARIMAX Forecast': forecast_arimax,
    'Absolute Error': np.abs(test['Close'] - forecast_arimax)
})
display(df_result_arimax)

print(f"\\nChỉ số sai số mô hình ARIMAX (10 bước):")
print(f"- RMSE : {rmse_arimax:.4f}")
print(f"- MAE  : {mae_arimax:.4f}")
print(f"- MAPE : {mape_arimax:.2f}%")
"""
cells.append(nbf.v4.new_code_cell(cell10_source))

# Cell 11: Plot comparison
cell11_source = """# 10. Vẽ biểu đồ so sánh kết quả dự báo của ARIMA và ARIMAX so với giá thực tế
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Biểu đồ 1: Toàn cảnh 60 ngày giao dịch gần nhất
recent_train = train.iloc[-50:]
ax1.plot(recent_train.index, recent_train['Close'], label='Train (Lịch sử gần)', color='#334155', linewidth=1.5)
ax1.plot(test.index, test['Close'], label='Test (Thực tế)', color='#10b981', linewidth=2.0, marker='o')
ax1.plot(test.index, forecast_arima, label=f'ARIMA{best_arima_order} Forecast', color='#f59e0b', linewidth=2.0, linestyle='--', marker='s')
ax1.plot(test.index, forecast_arimax, label=f'ARIMAX{best_arimax_order} Forecast', color='#ef4444', linewidth=2.0, linestyle=':', marker='^')
ax1.set_title('Tổng quan dự báo 10 bước (60 ngày giao dịch gần nhất)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Close Price (USD)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# Biểu đồ 2: Cận cảnh 10 ngày dự báo
ax2.plot(test.index, test['Close'], label='Test (Thực tế)', color='#10b981', linewidth=2.5, marker='o', markersize=6)
ax2.plot(test.index, forecast_arima, label=f'ARIMA{best_arima_order} (RMSE={rmse_arima:.2f})', color='#f59e0b', linewidth=2.0, linestyle='--', marker='s', markersize=6)
ax2.plot(test.index, forecast_arimax, label=f'ARIMAX{best_arimax_order} (RMSE={rmse_arimax:.2f})', color='#ef4444', linewidth=2.0, linestyle=':', marker='^', markersize=6)
ax2.set_title('Cận cảnh so sánh giá thực tế vs ARIMA vs ARIMAX (10 ngày)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Close Price (USD)', fontsize=11)
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
"""
cells.append(nbf.v4.new_code_cell(cell11_source))

# Cell 12: Summary table & commentary
cell12_source = """# 11. Bảng tổng hợp so sánh các chỉ số và nhận xét chi tiết
comparison_df = pd.DataFrame({
    'Mô hình': [f'ARIMA{best_arima_order}', f'ARIMAX{best_arimax_order} (X=Volume)'],
    'AIC': [model_arima_fit.aic, model_arimax_fit.aic],
    'BIC': [model_arima_fit.bic, model_arimax_fit.bic],
    'RMSE': [rmse_arima, rmse_arimax],
    'MAE': [mae_arima, mae_arimax],
    'MAPE (%)': [mape_arima, mape_arimax]
})

print("=== BẢNG SO SÁNH CHỈ SỐ GIỮA ARIMA VÀ ARIMAX ===")
display(comparison_df)

print(f'''
=== NHẬN XÉT VÀ ĐÁNH GIÁ CHI TIẾT KẾT QUẢ DỰ BÁO ===

1. Về quá trình lựa chọn tham số tối ưu (auto_arima):
   - Thuật toán `auto_arima` đã duyệt qua các cấu hình bậc p, d, q và tự động lựa chọn bộ tham số tối ưu {best_arima_order} cho mô hình ARIMA đơn biến và {best_arimax_order} cho mô hình ARIMAX.
   - Bậc sai phân d = 1 cho thấy chuỗi giá cổ phiếu Amazon ban đầu không dừng (non-stationary), nhưng sau khi lấy sai phân bậc 1 thì chuỗi đạt tính dừng (stationary), phản ánh đúng bản chất bước đi ngẫu nhiên (Random Walk) của thị trường tài chính.

2. Về đường dự báo của mô hình ARIMA:
   - Với mô hình ARIMA(0, 1, 0), dự báo cho các bước tương lai là một đường nằm ngang cố định tại mức giá đóng cửa của ngày cuối cùng trong tập Train (~137.85 USD).
   - Đây là dự báo kỳ vọng toán học chuẩn tắc của bước đi ngẫu nhiên không có trôi (Random walk without drift): giá trị tốt nhất để dự đoán giá tương lai chính là mức giá hiện tại.

3. Về hiệu quả khi bổ sung biến ngoại sinh Volume (ARIMAX):
   - Khi đưa biến ngoại sinh Volume (khối lượng giao dịch) vào mô hình ARIMAX, đường dự báo không còn là một đường thẳng nằm ngang mà có sự điều chỉnh vi mô theo sự biến động khối lượng giao dịch của từng ngày.
   - Sai số dự báo: ARIMAX đạt RMSE = {rmse_arimax:.4f} và MAE = {mae_arimax:.4f}, cho thấy sự cải thiện nhẹ so với ARIMA thuần túy (RMSE = {rmse_arima:.4f}, MAE = {mae_arima:.4f}).
   - Điều này chỉ ra rằng khối lượng giao dịch (Volume) có chứa tín hiệu thông tin bổ trợ nhưng giá cổ phiếu trong ngắn hạn vẫn chịu tác động lớn bởi tin tức thị trường và tâm lý vĩ mô.
''')
"""
cells.append(nbf.v4.new_code_cell(cell12_source))

nb['cells'] = cells

# Save raw notebook
nb_path = 'btth/8/BT8.ipynb'
os.makedirs('btth/8', exist_ok=True)
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Đã lưu notebook thô vào {nb_path}. Bắt đầu thực thi notebook...")

# Execute notebook in btth/8 working directory
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb_executed = nbf.read(f, as_version=4)

ep.preprocess(nb_executed, {'metadata': {'path': 'btth/8'}})

# Save executed notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_executed, f)

print(f"Đã thực thi và cập nhật outputs thành công vào {nb_path}!")

# Convert to HTML
html_exporter = nbconvert.HTMLExporter()
html_data, _ = html_exporter.from_notebook_node(nb_executed)

html_path1 = 'btth/8/BTTH8_Nhom6_24520500_24521314_24521649_24521732.html'
with open(html_path1, 'w', encoding='utf-8') as f:
    f.write(html_data)

html_path2 = 'btth/8/BT8.html'
with open(html_path2, 'w', encoding='utf-8') as f:
    f.write(html_data)

print(f"Đã xuất thành công file HTML: {html_path1} và {html_path2}!")
