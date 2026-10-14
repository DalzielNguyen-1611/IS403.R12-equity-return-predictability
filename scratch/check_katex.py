import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report_dir = os.path.join(REPO_ROOT, "results", "Lan01", "agent_reports")
files = [f for f in os.listdir(report_dir) if f.endswith(".md") and f != "ALL_REPORTS_COMBINED.md"]

issues = []
for fname in files:
    fpath = os.path.join(report_dir, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            prev_line = lines[idx-1].strip() if idx > 0 else ""
            next_line = lines[idx+1].strip() if idx < len(lines)-1 else ""
            if prev_line != "" or next_line != "":
                issues.append((fname, idx+1, prev_line[:30], next_line[:30]))

print(f"Tong so block math can cach dong: {len(issues)}")
for iss in issues:
    print(f"  {iss[0]}:{iss[1]} | prev: '{iss[2]}' | next: '{iss[3]}'")
