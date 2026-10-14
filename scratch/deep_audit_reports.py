import os
import re
import glob
import pandas as pd

REPO_ROOT = r"e:\Repo Project\IS403.R12-equity-return-predictability"
reports_dir = os.path.join(REPO_ROOT, "results", "Lan01", "agent_reports")

print("=== 1. CHECKING MARKDOWN LINKS IN ALL REPORTS ===")
md_files = glob.glob(os.path.join(reports_dir, "*.md"))

broken_links = []
old_path_links = []

for fpath in md_files:
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all markdown links: [text](link)
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    for text, link in links:
        link_clean = link.split("#")[0] # remove anchor
        if not link_clean:
            continue
        if link_clean.startswith("http://") or link_clean.startswith("https://"):
            continue
        if link_clean.startswith("file:///"):
            local_path = link_clean.replace("file:///", "").replace("%20", " ")
            # On windows, e.g. e:/Repo Project/...
            local_path = local_path.replace("/", "\\")
            if not os.path.exists(local_path):
                broken_links.append((fname, text, link, local_path))
        elif not link.startswith("#"):
            # relative path
            target = os.path.normpath(os.path.join(os.path.dirname(fpath), link_clean))
            if not os.path.exists(target):
                broken_links.append((fname, text, link, target))

print(f"Total broken links found: {len(broken_links)}")
for fname, text, link, target in broken_links:
    print(f"  [{fname}] Link '{text}' -> '{link}' (Resolved: {target}) DOES NOT EXIST")

print("\n=== 2. CHECKING FOR PLACEHOLDER TOKENS (TODO, FIXME, NaN, undefined) ===")
placeholders = ["TODO", "FIXME", "TBD", "undefined", "PLACEHOLDER", "???"]
for fpath in md_files:
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    for token in placeholders:
        matches = [m.start() for m in re.finditer(re.escape(token), content, re.IGNORECASE)]
        if matches:
            print(f"  [{fname}] Found placeholder '{token}' ({len(matches)} times)")

print("\n=== 3. CHECKING TABLE PIPE BALANCE & FORMATTING ===")
table_errors = []
for fpath in md_files:
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line_str = line.strip()
        if line_str.startswith("|") and line_str.endswith("|"):
            # Check if separator line
            if re.match(r'^\|[\s\-:|]+\|$', line_str):
                continue
            # Just check if pipe count is reasonable
            pipes = line_str.count("|")
            if pipes < 2:
                table_errors.append((fname, i+1, line_str))

print(f"Table line errors: {len(table_errors)}")

print("\n=== 4. CHECKING KATEX / MATH DELIMITERS ===")
math_errors = []
for fpath in md_files:
    fname = os.path.basename(fpath)
    with open(fpath, "r", encoding="utf-8") as f:
        text = f.read()
    # Check for unclosed $$
    double_dollars = text.count("$$")
    if double_dollars % 2 != 0:
        math_errors.append((fname, f"Unclosed $$ block (count: {double_dollars})"))
    # Check lines with single $
    lines = text.split("\n")
    for idx, l in enumerate(lines, 1):
        # exclude lines with code block
        if l.strip().startswith("```"):
            continue
        # remove $$ first
        l_no_double = l.replace("$$", "")
        # count single $ not escaped
        singles = re.findall(r'(?<!\\)\$', l_no_double)
        if len(singles) % 2 != 0:
            math_errors.append((fname, f"Line {idx}: Unclosed inline $ (count: {len(singles)}): {l[:80]}"))

print(f"Math delimiter issues found: {len(math_errors)}")
for fname, err in math_errors[:15]:
    print(f"  [{fname}] {err}")

print("\nScan completed.")
