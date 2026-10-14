import os
import glob
import filecmp
import pandas as pd

dir1 = "results/Lan01"
dir2 = "results/Lan01_backup_pre_audit"

print(f"Comparing {dir1} vs {dir2}...")

# Check all files in dir2
files2 = []
for root, _, fnames in os.walk(dir2):
    for f in fnames:
        rel = os.path.relpath(os.path.join(root, f), dir2)
        files2.append(rel)

print(f"Total files in backup: {len(files2)}")

diff_files = []
identical_files = []
missing_in_main = []

for rel in sorted(files2):
    p1 = os.path.join(dir1, rel)
    p2 = os.path.join(dir2, rel)
    if not os.path.exists(p1):
        missing_in_main.append(rel)
    else:
        if filecmp.cmp(p1, p2, shallow=False):
            identical_files.append(rel)
        else:
            diff_files.append(rel)

print(f"Identical files: {len(identical_files)}")
print(f"Different files: {len(diff_files)}")
print(f"Files in backup missing in Lan01: {len(missing_in_main)}")

print("\n--- List of Different Files ---")
for f in diff_files:
    print(" -", f)

print("\n--- Details of Differences ---")
for f in diff_files:
    if f.endswith(".csv"):
        df1 = pd.read_csv(os.path.join(dir1, f))
        df2 = pd.read_csv(os.path.join(dir2, f))
        print(f"\nFile: {f}")
        print(f"  Lan01 shape: {df1.shape}, Backup shape: {df2.shape}")
        if 'Model' in df1.columns and 'Model' in df2.columns:
            # Check which models differ
            for m in df1['Model'].unique():
                sub1 = df1[df1['Model'] == m]
                sub2 = df2[df2['Model'] == m] if m in df2['Model'].values else None
                if sub2 is None:
                    print(f"    Model {m}: missing in backup")
                elif not sub1.equals(sub2):
                    print(f"    Model {m}: DIFFERENT (Audited/Retrained)")
                else:
                    # identical
                    pass
