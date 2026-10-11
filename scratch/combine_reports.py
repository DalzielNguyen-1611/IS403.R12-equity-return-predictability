import os

report_dir = r"e:\IS403.R12-equity-return-predictability\results\agent_reports"
output_path = os.path.join(report_dir, "ALL_REPORTS_COMBINED.md")

file_order = [
    ("FINAL_SYNTHESIS.md", "TỔNG HỢP KẾT QUẢ NGHIÊN CỨU TOÀN DIỆN (FINAL SYNTHESIS)"),
    ("01_all_fold_metrics_report.md", "Báo Cáo 01: Tính Toàn Vẹn Thực Nghiệm & Đo Lường Cấp Fold"),
    ("02_model_horizon_summary_report.md", "Báo Cáo 02: Tổng Hợp Hiệu Năng Đa Mô Hình & Đa Chân Trời"),
    ("03_all_oos_predictions_report.md", "Báo Cáo 03: Phân Tích Dữ Liệu Dự Báo Ngoại Mẫu (OOS)"),
    ("04_best_hyperparameters_report.md", "Báo Cáo 04: Đánh Giá Tinh Chỉnh Siêu Tham Số & Rò Rỉ Dữ Liệu"),
    ("05_rq1_horizon_report.md", "Báo Cáo 05 (RQ1): Sự Suy Thoái Năng Lực Dự Báo Theo Chân Trời Thời Gian"),
    ("06_rq2_regime_report.md", "Báo Cáo 06 (RQ2): Năng Lực Dự Báo Theo Chế Độ Biến Động Thị Trường"),
    ("07_rq3_time_report.md", "Báo Cáo 07 (RQ3): Tính Ổn Định Năng Lực Dự Báo Qua Các Chu Kỳ Thời Gian"),
    ("07_rq3_time_audit_log.md", "Báo Cáo 07 Audit Log: Kiểm Toán Phương Pháp Luận & Thống Kê RQ3"),
    ("08_statistical_tests_report.md", "Báo Cáo 08: Kiểm Định Thống Kê Năng Lực Dự Báo (DM, Clark-West, SPA)"),
    ("08_statistical_tests_audit_log.md", "Báo Cáo 08 Audit Log: Kiểm Toán Phương Pháp Luận & Thống Kê Báo Cáo 08")
]

lines = []
lines.append("# TOÀN BỘ BÁO CÁO NGHIÊN CỨU TỔNG HỢP (MASTER RESEARCH REPORT)")
lines.append("")
lines.append("> **Đề tài:** *When Does Equity Return Predictability Decay? A Multi-Horizon and Regime-Dependent Out-of-Sample Analysis*  ")
lines.append("> **Mục đích tài liệu:** Hợp nhất toàn diện 11 báo cáo chuyên sâu và biên bản kiểm toán phương pháp luận của dự án vào một tệp duy nhất để tiện tra cứu, đối chiếu và lưu trữ.  ")
lines.append("> **Đường dẫn thư mục gốc:** `results/agent_reports/`  ")
lines.append("")
lines.append("---")
lines.append("")
lines.append("<a id=\"muc-luc\"></a>")
lines.append("## 📑 MỤC LỤC TỔNG HỢP HỆ THỐNG BÁO CÁO")
lines.append("")

for idx, (fname, desc) in enumerate(file_order, start=1):
    tag = fname.replace(".", "-").lower()
    lines.append(f"{idx}. [**{desc}** (`{fname}`)](#{tag})")

lines.append("")
lines.append("---")
lines.append("")

for idx, (fname, desc) in enumerate(file_order, start=1):
    fpath = os.path.join(report_dir, fname)
    if not os.path.exists(fpath):
        print(f"Warning: {fpath} does not exist!")
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    tag = fname.replace(".", "-").lower()
    lines.append(f"<a id=\"{tag}\"></a>")
    lines.append(f"# [PHẦN {idx:02d}] {desc.upper()}")
    lines.append(f"*Tập tin nguồn:* [`{fname}`](./{fname}) | [⬆ Quay lại Mục Lục](#muc-luc)")
    lines.append("")
    lines.append(content)
    lines.append("")
    lines.append("---")
    lines.append("")

final_doc = "\n".join(lines)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_doc)

print(f"Hoan tat! Da ghi thanh cong {len(final_doc)} ky tu vao {output_path}")
