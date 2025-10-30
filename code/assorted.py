from pathlib import Path
import shutil

def copy_midlines_and_exclude(input_path, output_path):

    in_root = Path(input_path)
    out_root = Path(output_path)

    targets = ("midlines.npy", "exclude.npy")
    copied = 0
    missing = []

    for sub in in_root.iterdir():
        seg = sub / "segmentation"
        if not seg.is_dir():
            continue

        files_to_copy = [seg / name for name in targets if (seg / name).exists()]
        if not files_to_copy:
            print(f"none for {sub}")

        dest_dir = out_root / sub.name / "segmentation"
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in files_to_copy:
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
        print(f"for {sub} copied {copied}")

input = r"J:\CHAT\output_251030"
output = r"J:\CHAT\output_251030_new_midlines"

copy_midlines_and_exclude(input, output)
