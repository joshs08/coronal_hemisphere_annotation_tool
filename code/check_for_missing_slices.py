from pathlib import Path
import re
from datetime import datetime
import csv
from collections import Counter

def find_duplicate_tif_indices(path):
    path = Path(path)
    pat = re.compile(r'^Image_(\d+).*\.(tif|tiff)$', re.IGNORECASE)

    for folder in sorted(p for p in path.iterdir() if p.is_dir()):
        nums = [int(m.group(1)) for m in (pat.match(f.name) for f in folder.iterdir() if f.is_file()) if m]
        if not nums:
            continue
        dupes = sorted(n for n, c in Counter(nums).items() if c > 1)
        if dupes:
            print(f"{folder.name}: duplicates {dupes}")
        else: print(f"{folder.name}: duplicates {dupes}")

def check_missing_indices(path, csv_path):
    path = Path(path)
    #pat = re.compile(r'^Image_(\d+).*\.(tif|tiff|tig)$', re.IGNORECASE)
    pat = re.compile(r'^Image_(\d+).*\.(czi)$', re.IGNORECASE)

    # collect dates to include from CSV
    with open(csv_path, newline='', encoding='utf-8') as f:
        dates = {row.get('date', '').strip() for row in csv.DictReader(f) if row.get('date')}

    allowed = set()
    for s in dates:
        d = datetime.strptime(s, '%Y-%m-%d')
        dd = f"{d.day:02d}"
        yy = f"{d.year % 100:02d}"
        allowed.add(f"{dd}_{d.month:02d}_{yy}")  # dd_mm_yy
        allowed.add(f"{dd}_{d.month}_{yy}")      # dd_m_yy

    for folder in sorted(p for p in path.iterdir()
                         if p.is_dir() and p.name[:1].isdigit() and p.name in allowed):
        matches = [pat.match(f.name) for f in folder.iterdir() if f.is_file()]
        matches = [m for m in matches if m]

        if not matches:
            print(f"{folder.name}: no matching tifs")
            continue

        nums = [int(m.group(1)) for m in matches]
        expected_n = len(matches)
        present = set(nums)
        missing = any(i not in present for i in range(1, expected_n + 1))

        print(f"{folder.name}: {'missing' if missing else 'ok'}")


path = r"L:\Barcode_RNA seq\Whole brain confocal images\Barcode_images"
csv_path = r"J:\CHAT\data\plates_6_11_12_13_14_16_17_18.csv"
#check_missing_indices(path, csv_path)
print("W")

find_duplicate_tif_indices(r"J:\CHAT\data")