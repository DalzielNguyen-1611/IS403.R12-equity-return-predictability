# Springer LNCS Manuscript Package

Thư mục này chứa trọn bộ bài báo xuất bản độc lập theo chuẩn Springer LNCS, đã được tách biệt sạch sẽ khỏi tất cả các file nháp, file trung gian và file tiếng Việt.

## 1. Danh Mục Các File

| Tên File | Mô Tả |
| :--- | :--- |
| `samplepaper.tex` | Mã nguồn LaTeX chính của bài báo (bản tiếng Anh hoàn chỉnh, 57 trang, 27 bảng biểu). |
| `samplepaper.pdf` | Bản PDF biên dịch hoàn thiện, tích hợp sẵn hệ thống mục lục thanh bên (PDF Bookmarks/Outline). |
| `llncs.cls` | Document class định dạng chuẩn của Springer Lecture Notes in Computer Science (Version 2.25). |
| `splncs04.bst` | File định dạng trích dẫn tài liệu tham khảo BibTeX chuẩn Springer LNCS. |
| `references.bib` | File cơ sở dữ liệu tài liệu tham khảo BibTeX (37 trích dẫn đầy đủ). |
| `fig1.eps` | File hình minh họa gốc định dạng EPS. |
| `fig1-eps-converted-to.pdf` | Bản PDF vector của hình minh họa được chuyển đổi tự động khi biên dịch. |

## 2. Hướng Dẫn Biên Dịch Thủ Công

Bạn có thể mở PowerShell hoặc Command Prompt tại thư mục này và biên dịch lại bất kỳ lúc nào bằng lệnh:

```powershell
pdflatex -interaction=nonstopmode samplepaper.tex
```

*(Biên dịch 2 lần để cập nhật toàn bộ liên kết nội bộ, chỉ số bảng biểu và mục lục PDF).*

