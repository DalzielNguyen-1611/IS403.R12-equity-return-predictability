import os
import re

report_dir = r"e:\IS403.R12-equity-return-predictability\results\agent_reports"
files = [f for f in os.listdir(report_dir) if f.endswith(".md") and f != "ALL_REPORTS_COMBINED.md"]

for fname in files:
    fpath = os.path.join(report_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    # Normalize newlines
    lines = content.split("\n")
    new_lines = []
    
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        
        # Check if line is a standalone $$...$$ block or opening/closing $$
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            # Clean trailing spaces
            math_content = stripped
            # Check if previous line in new_lines is not empty
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
            new_lines.append(math_content)
            # Check if next line is not empty, add blank line
            if i + 1 < n and lines[i+1].strip() != "":
                new_lines.append("")
            i += 1
        elif stripped == "$$":
            # Multi-line block math
            if new_lines and new_lines[-1].strip() != "":
                new_lines.append("")
            new_lines.append("$$")
            i += 1
            while i < n and lines[i].strip() != "$$":
                new_lines.append(lines[i].rstrip())
                i += 1
            if i < n and lines[i].strip() == "$$":
                new_lines.append("$$")
                if i + 1 < n and lines[i+1].strip() != "":
                    new_lines.append("")
                i += 1
        else:
            new_lines.append(line.rstrip())
            i += 1

    updated_content = "\n".join(new_lines) + "\n"
    if updated_content != content:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Fixed KaTeX spacing in: {fname}")

print("Rebuilding ALL_REPORTS_COMBINED.md...")
os.system("python scratch/combine_reports.py")
print("Done!")
