from pathlib import Path
import re

def check_missing_indices(path):
    path = Path(path)
    #pat = re.compile(r'^Image_(\d+).*\.(tif|tiff|tig)$', re.IGNORECASE)
    pat = re.compile(r'^Image_(\d+).*\.(czi)$', re.IGNORECASE)

    for folder in sorted(p for p in path.iterdir() if p.is_dir() and p.name[:1].isdigit()):
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
check_missing_indices(path)
print("W")