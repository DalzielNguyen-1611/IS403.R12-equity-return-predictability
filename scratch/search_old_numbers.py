import os, glob, re

repo = r"e:\Repo Project\IS403.R12-equity-return-predictability"
reports_dir = os.path.join(repo, "results", "Lan01", "agent_reports")

old_numbers = [
    "0.0467", "0.0444", "0.3693", "0.6709", "0.0020", "0.0003", "0.0001", "0.0028"
]

for f in sorted(glob.glob(os.path.join(reports_dir, "*.md"))):
    fname = os.path.basename(f)
    if fname == "3.0.3_ALL_REPORTS_COMBINED.md": continue
    with open(f, "r", encoding="utf-8") as fp:
        c = fp.read()
    for num in old_numbers:
        if num in c:
            print(f"[{fname}] contains old number: {num}")
