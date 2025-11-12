from pathlib import Path
import shutil
import pandas as pd

def copy_midlines_and_exclude(input_path, output_path):
    input_subfolder = "segmentation"
    output_subfolder = "registration"
    in_root = Path(input_path)
    out_root = Path(output_path)

    targets = ("deepslice_registration_results.json",)
    copied = 0
    missing = []

    for sub in in_root.iterdir():
        #seg = sub / input_subfolder
        seg = sub
        if not seg.is_dir():
            continue
        
        #files_to_copy = [seg / name for name in targets if (seg / name).exists()]
        files_to_copy = list(seg.glob("*.csv"))
        if not files_to_copy:
            print(f"none for {sub}")

        #dest_dir = out_root / sub.name / output_subfolder
        dest_dir = out_root / sub.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in files_to_copy:
            dst = dest_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
        print(f"for {sub} copied {copied}")

def combine_csvs(path):
    base = Path(path) 
    out_csv = base / "combined_sample_data.csv"

    dfs = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        fn = sub / f"sample_data_{sub.name}.csv"
        if fn.exists():
            dfs.append(pd.read_csv(fn))
        else:
            print(f"no csv {sub.name}")

    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        big.to_csv(out_csv, index=False)

input = r"J:\CHAT\output_251030_new_midlines"
output = r"J:\CHAT\QuickNII_processing"

#copy_midlines_and_exclude(input, output)
combine_csvs(r"J:\CHAT\QuickNII_processing")
