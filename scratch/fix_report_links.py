import os
import glob

repo_dir = r"e:\Repo Project\IS403.R12-equity-return-predictability"
reports_dir = os.path.join(repo_dir, "results", "Lan01", "agent_reports")

old_str = "file:///e:/IS403.R12-equity-return-predictability"
new_str = "file:///e:/Repo%20Project/IS403.R12-equity-return-predictability"

md_files = glob.glob(os.path.join(reports_dir, "*.md"))
for fpath in md_files:
    fname = os.path.basename(fpath)
    if fname == "3.0.3_ALL_REPORTS_COMBINED.md":
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    if old_str in content:
        count = content.count(old_str)
        content = content.replace(old_str, new_str)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {count} occurrences in {fname}")

print("Path replacement complete.")
