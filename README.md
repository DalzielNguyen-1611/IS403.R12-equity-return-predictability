# ĐỒ ÁN MÔN HỌC: PHÂN TÍCH DỮ LIỆU KINH DOANH

## 📊 Đề tài: When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis

> Kho lưu trữ toàn bộ mã nguồn, dữ liệu, tài liệu và báo cáo phục vụ cho đồ án môn học Phân tích dữ liệu kinh doanh.

---

## 📌 1. THÔNG TIN HỌC PHẦN & ĐỒ ÁN

| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Tên môn học** | Phân tích dữ liệu kinh doanh (Business Data Analysis) |
| **Mã môn học** | IS402 |
| **Lớp học phần** | IS402.R12 |
| **Đề tài đồ án** | **When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis** |
| **Học kỳ / Năm học** | Học kì 1 — Năm học 2026 - 2027 |
| **Khoa / Trường** | Khoa Hệ thống Thông tin — Trường ĐH Công nghệ Thông tin (UIT - ĐHQG-HCM) |

---

## 👨‍🏫 2. GIẢNG VIÊN HƯỚNG DẪN

- **Giảng viên lý thuyết:** ThS. Dương Phi Long

---

## 👨‍🎓 3. THÀNH VIÊN NHÓM

| STT | Họ và Tên | MSSV | Lớp | Vai trò | GitHub | Email |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | **Nguyễn Đoàn Đức Hiếu** | 24520500 | IS402.R12 | **Leader** | [@DalzielNguyen-1611](https://github.com/DalzielNguyen-1611) | [24520500@gm.uit.edu.vn](mailto:24520500@gm.uit.edu.vn) |
| 2 | **Nguyễn Thị Phương Hoa** | 23520506 | IS402.R12 | Member | — | [23520506@gm.uit.edu.vn](mailto:23520506@gm.uit.edu.vn) |
| 3 | **Nguyễn Văn Phát** | 24521314 | IS402.R12 | Member | — | [24521314@gm.uit.edu.vn](mailto:24521314@gm.uit.edu.vn) |
| 4 | **Trần Anh Thư** | 24521732 | IS402.R12 | Member | — | [24521732@gm.uit.edu.vn](mailto:24521732@gm.uit.edu.vn) |
| 5 | **Trần Ngọc Thảo** | 24521649 | IS402.R12 | Member | — | [24521649@gm.uit.edu.vn](mailto:24521649@gm.uit.edu.vn) |

---

## 🎯 4. MỤC TIÊU CỦA ĐỒ ÁN

Repository này được tạo ra nhằm phục vụ riêng cho đồ án môn học với các mục tiêu chính:
- **Lưu trữ & Quản lý đồ án:** Quản lý tập trung toàn bộ mã nguồn, tập dữ liệu (dataset), tài liệu nghiên cứu và báo cáo của đồ án.
- **Khám phá & Xử lý dữ liệu tài chính:** Thu thập, làm sạch và thực hiện phân tích khám phá (EDA) dữ liệu chuỗi thời gian, các chỉ số tài chính và kinh tế vĩ mô liên quan đến khả năng sinh lời của cổ phiếu.
- **Xây dựng & Đánh giá mô hình dự báo:** Áp dụng và so sánh các mô hình thống kê, kinh tế lượng và học máy (Machine Learning / Deep Learning) trong bài toán dự đoán tỷ suất sinh lợi (Equity Return Predictability).
- **Tổng kết & Đưa ra hàm ý kinh doanh:** Trực quan hóa kết quả phân tích (Data Visualization), xây dựng báo cáo chi tiết và đề xuất các chiến lược đầu tư / quản trị rủi ro khả thi.

---

## 📂 5. CẤU TRÚC THƯ MỤC

```text
IS403.R12-equity-return-predictability/
├── data/                    # Thư mục lưu trữ dữ liệu đồ án
│   ├── raw/                 # Dữ liệu thô thu thập từ các nguồn (Yahoo Finance, v.v.)
│   └── processed/           # Dữ liệu sau khi làm sạch và tạo đặc trưng (features)
├── notebooks/               # Jupyter Notebooks phục vụ EDA, thử nghiệm và huấn luyện mô hình
├── src/                     # Mã nguồn xử lý dữ liệu, định nghĩa mô hình và hàm tiện ích
├── models/                  # Lưu trữ trọng số / artifacts của các mô hình đã huấn luyện
├── reports/                 # Báo cáo đồ án, slide thuyết trình và các biểu đồ kết quả
└── README.md                # Tài liệu giới thiệu thông tin đồ án môn học
```
