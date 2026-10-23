# Script to synchronize updated metrics across all markdown reports in results/Lan01/agent_reports/
import os, re

BASE_DIR = 'results/Lan01/agent_reports'

def replace_in_file(fname, replacements):
    fpath = os.path.join(BASE_DIR, fname)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    orig = content
    for old_str, new_str in replacements:
        if old_str not in content:
            print(f'WARNING in {fname}: target not found: {old_str[:60]}...')
        content = content.replace(old_str, new_str)
    if content != orig:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Successfully updated {fname}')
    else:
        print(f'No changes made to {fname}')

print('Script template ready.')
