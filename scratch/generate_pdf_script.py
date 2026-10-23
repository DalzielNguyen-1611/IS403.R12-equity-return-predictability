import os
import re
import markdown
import pymupdf

def clean_math_to_html(text: str) -> str:
    """Chuyển đổi các ký hiệu LaTeX toán học sang thẻ HTML đẹp mắt, dễ đọc khi in ấn."""
    # Target formula
    text = text.replace(r"\ln\left(\frac{Close_{t+h}}{Close_t}\right)", "ln( Close<sub>t+h</sub> / Close<sub>t</sub> )")
    text = text.replace(r"\ln(Close_{t+h} / Close_t)", "ln( Close<sub>t+h</sub> / Close<sub>t</sub> )")
    text = text.replace(r"\ln(Close_{t+h}/Close_t)", "ln( Close<sub>t+h</sub> / Close<sub>t</sub> )")
    text = text.replace(r"$Target_{h,t} = \ln\left(\frac{Close_{t+h}}{Close_t}\right)$", "<strong>Target<sub>h,t</sub> = ln( Close<sub>t+h</sub> / Close<sub>t</sub> )</strong>")
    text = text.replace(r"$Target_{h,t} = \ln(Close_{t+h} / Close_t)$", "<strong>Target<sub>h,t</sub> = ln( Close<sub>t+h</sub> / Close<sub>t</sub> )</strong>")
    text = text.replace(r"Target_{h,t}", "Target<sub>h,t</sub>")
    
    # Sum formula
    text = text.replace(r"$r_{t, t+h} = \sum_{j=1}^h r_{t+j}$", "<strong>r<sub>t,t+h</sub> = &Sigma;<sub>j=1</sub><sup>h</sup> r<sub>t+j</sub></strong>")
    text = text.replace(r"r_{t, t+h}", "r<sub>t,t+h</sub>")
    text = text.replace(r"r_{t+j}", "r<sub>t+j</sub>")
    
    # R2 OOS formulas
    text = text.replace(r"$R^2_{\text{OOS}} = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y}_{\text{bench},t})^2}$", "<strong>R²<sub>OOS</sub> = 1 - &Sigma;(y<sub>t</sub> - ŷ<sub>t</sub>)² / &Sigma;(y<sub>t</sub> - ȳ<sub>bench,t</sub>)²</strong>")
    text = text.replace(r"$R^2_{\text{OOS}} > 0$", "<strong>R²<sub>OOS</sub> &gt; 0</strong>")
    text = text.replace(r"$R^2_{\text{OOS}}$", "<strong>R²<sub>OOS</sub></strong>")
    text = text.replace(r"$R^2$", "<strong>R²</strong>")
    text = text.replace(r"R^2_{\text{OOS}}", "R²<sub>OOS</sub>")
    text = text.replace(r"\bar{y}_{\text{bench},t}", "ȳ<sub>bench,t</sub>")
    
    # P-value & symbols
    text = text.replace(r"$p < 0.05$", "p &lt; 0.05")
    text = text.replace(r"$p < 0.10$", "p &lt; 0.10")
    text = text.replace(r"$\rightarrow$", "&rarr;")
    text = text.replace(r"\rightarrow", "&rarr;")
    text = text.replace(r"$h$", "<em>h</em>")
    text = text.replace(r"$1D \to 120D$", "1D &rarr; 120D")
    text = text.replace(r"$1D$", "1D")
    text = text.replace(r"$5D$", "5D")
    text = text.replace(r"$20D$", "20D")
    text = text.replace(r"$60D$", "60D")
    text = text.replace(r"$120D$", "120D")
    
    # Clean remaining $ signs
    text = re.sub(r"\$([^\$]+)\$", r"<em>\1</em>", text)
    
    return text

def build_pdf():
    script_md_path = r"E:\IS403.R12-equity-return-predictability\docs\presentation_script.md"
    output_pdf_docs = r"E:\IS403.R12-equity-return-predictability\docs\presentation_script.pdf"
    output_pdf_root = r"E:\IS403.R12-equity-return-predictability\Giới thiệu đồ án IS403_script.pdf"

    with open(script_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-clean formulas
    cleaned_md = clean_math_to_html(md_text)

    # Convert markdown to html
    html_body = markdown.markdown(
        cleaned_md,
        extensions=['tables', 'fenced_code', 'nl2br']
    )

    # Wrap in modern styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            font-size: 9.5pt;
            line-height: 1.55;
        }}
        h1 {{
            color: #0f172a;
            font-size: 16pt;
            border-bottom: 2.5px solid #2563eb;
            padding-bottom: 6px;
            margin-top: 0;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        h2 {{
            color: #1e3a8a;
            font-size: 12.5pt;
            border-bottom: 1.5px solid #cbd5e1;
            padding-bottom: 4px;
            margin-top: 18px;
            margin-bottom: 10px;
        }}
        h3 {{
            color: #1d4ed8;
            font-size: 11pt;
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 4px 8px;
            margin-top: 14px;
            margin-bottom: 8px;
            border-radius: 0 4px 4px 0;
        }}
        p {{
            margin-top: 3px;
            margin-bottom: 6px;
        }}
        strong {{
            color: #0f172a;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 8px;
            margin-bottom: 12px;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 5px 8px;
            font-size: 9pt;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            color: #0f172a;
            font-weight: bold;
        }}
        blockquote {{
            border-left: 3.5px solid #2563eb;
            background-color: #f8fafc;
            padding: 8px 12px;
            margin: 6px 0 10px 0;
            border-radius: 0 5px 5px 0;
            font-size: 9.5pt;
            color: #0f172a;
        }}
        ul, ol {{
            margin-top: 3px;
            margin-bottom: 6px;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 3px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 12px 0;
        }}
    </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    mediabox = pymupdf.paper_rect('a4')
    margin_x = 45
    margin_y = 48
    where = mediabox + (margin_x, margin_y, -margin_x, -margin_y)

    temp_pdf = os.path.join(os.path.dirname(output_pdf_docs), "_temp_rendered.pdf")
    if os.path.exists(temp_pdf):
        os.remove(temp_pdf)

    writer = pymupdf.DocumentWriter(temp_pdf)
    story = pymupdf.Story(full_html)
    story.write(writer, rectfn=lambda rect_num, filled: (mediabox, where, None))
    writer.close()

    # Re-open with pymupdf to add running header & footer using insert_htmlbox for 100% Unicode
    doc = pymupdf.open(temp_pdf)
    total_pages = doc.page_count

    for page_idx, page in enumerate(doc):
        # Header Box
        header_rect = pymupdf.Rect(margin_x, 18, mediabox.width - margin_x, 38)
        header_html = """
        <div style="font-family:sans-serif;font-size:7.5pt;color:#64748b;display:flex;justify-content:space-between;border-bottom:0.75px solid #cbd5e1;padding-bottom:3px;">
            <span>IS403.R12 — KỊCH BẢN THUYẾT TRÌNH BÁO CÁO ĐỀ TÀI</span>
            <span style="text-align:right;float:right;">GVHD: ThS. Dương Phi Long</span>
        </div>
        """
        page.insert_htmlbox(header_rect, header_html)

        # Footer Box
        footer_rect = pymupdf.Rect(margin_x, mediabox.height - 32, mediabox.width - margin_x, mediabox.height - 12)
        footer_html = f"""
        <div style="font-family:sans-serif;font-size:7.5pt;color:#64748b;border-top:0.75px solid #cbd5e1;padding-top:4px;">
            <span>Đề tài: When Does Equity Return Predictability Decay? — Nhóm 6</span>
            <span style="float:right;text-align:right;font-weight:bold;">Trang {page_idx + 1} / {total_pages}</span>
        </div>
        """
        page.insert_htmlbox(footer_rect, footer_html)

    # Save to finalized paths
    doc.save(output_pdf_docs)
    doc.save(output_pdf_root)
    doc.close()

    if os.path.exists(temp_pdf):
        try:
            os.remove(temp_pdf)
        except Exception:
            pass

    print(f"SUCCESS: Generated {output_pdf_docs} ({total_pages} pages)")
    print(f"SUCCESS: Copied to {output_pdf_root}")

if __name__ == "__main__":
    build_pdf()
