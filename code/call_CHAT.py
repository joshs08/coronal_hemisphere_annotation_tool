from pathlib import Path
import subprocess, sys, time
from datetime import datetime
import os
import pandas as pd
import json

def to_yyyy_mm_dd(name: str) -> str:
    return datetime.strptime(name, "%d_%m_%y").strftime("%Y-%m-%d")

def run(*args):
    print(">", " ".join(map(str, args)))
    subprocess.run(args, check=True)

def join_dir(*parts) -> str:
    """Join parts and guarantee a trailing slash."""
    return os.path.join(*parts, "")

def json_to_csv(json_file, csv_file):
    names = ["ox","oy","oz","ux","uy","uz","vx","vy","vz"]
    with open(json_file) as f:
        data = json.load(f)

    # If your JSON has a list of items you want as rows:
    df = pd.json_normalize(data["slices"])  # adjust the key to your data
    df[names] = pd.DataFrame(df["anchoring"].apply(lambda x: (x or [None]*9)[:9]).tolist(), index=df.index)
    df.drop(columns=["anchoring"], inplace=True, errors="ignore")   
    df.to_csv(csv_file, index=False)

input_csv = "J:/CHAT/data/plates_6_11_12_13_14_16_17_18.csv"
input_image_root = "J:/CHAT/data/"
output_dir = "J:/CHAT/output_251030_new_midlines/"
clean_dir = "J:/CHAT/VisuAlign_processing"
code_dir = Path("C:/Users/Josh Selfe/Documents/GitHub/coronal_hemisphere_annotation_tool/code/")
allen = "J:/CHAT/ABA"

exclude_completed = False # Do we bypass folders already present in output folder?
exclude_slices = False # Do we run the script to exclude damaged slices?
run_pipeline = True # Do we run the rest of the pipeline?
run_figures = False

excluded_folders = []

image_dates = sorted(
    p.name for p in Path(input_image_root).iterdir()
    if p.is_dir() and "copy" not in p.name.lower()
)

converted = {d: to_yyyy_mm_dd(d) for d in image_dates}
if exclude_completed:
    existing_dates = {p.name for p in Path(output_dir).iterdir() if p.is_dir()} if Path(output_dir).exists() else set()
    image_dates = [d for d, conv in converted.items() if conv not in existing_dates]

#image_dates = ["04_09_18"]

slice_direction = "rostro-caudal"
slice_thickness = 250

for image_date in image_dates:
    date = datetime.strptime(image_date, "%d_%m_%y").strftime("%Y-%m-%d")
    if date in excluded_folders:
        continue
    sample_data_csv = os.path.join(clean_dir, date, f"sample_data_{date}.csv")
    input_image_dir = join_dir(input_image_root, image_date)
    # Get old midlines
    #midlines_date_dir = os.path.join(input_image_dir, "segmentation", "midlines.npy")

    output_dir_date = join_dir(clean_dir, date) # clean_dir
    Path(output_dir_date).mkdir(parents=True, exist_ok=True)

    seg = join_dir(output_dir, date, "segmentation")#join_dir(output_dir_date,
    VA_seg = join_dir(clean_dir, date, "segmentation")
    ann = join_dir(clean_dir, date, "annotation")
    reg, fig = (join_dir(output_dir_date, "registration"), 
                          join_dir(output_dir_date, "figures"))
    #if Path(reg).exists():
    #    continue
    
    midlines_path = os.path.join(seg, "midlines.npy")

    t0 = time.time()

    pdf_file = join_dir(fig, f"individual_slices_{date.replace('-', '_')}.pdf")
    if run_pipeline:
        #run(sys.executable, code_dir/"extract_sample_data.py", str(input_csv), "date", str(date), str(sample_data_csv), "--show")
        #run(sys.executable, code_dir/"convert_sample_coordinates.py", str(sample_data_csv), str(input_image_dir))
        #run(sys.executable, code_dir/"hemisphere_segmentation_and_alignment.py", str(input_image_dir), str(sample_data_csv), str(seg), "--midlines", str(midlines_path))
        print("Done")
    if exclude_slices:
        try:
            #run(sys.executable, code_dir/"hemisphere_segmentation_and_alignment.py", str(input_image_dir), str(sample_data_csv), str(seg), "--midlines", str(midlines_date_dir))
            run(sys.executable, code_dir/"hemisphere_segmentation_and_alignment.py", str(input_image_dir), str(sample_data_csv), str(seg), "--segment_bool")
            #run(sys.executable, code_dir/"exclude_damaged_slices.py", str(seg), str(pdf_file), str(sample_data_csv))
        except:
            print(f"Exclusion not possible for {date}.")
    if run_pipeline:
        # Register using deep slice
        #run(sys.executable, code_dir/"image_registration.py", str(seg), str(reg), str(seg),
        #    "--slice_direction", str(slice_direction), "--slice_thickness", str(slice_thickness))
        # Convert json to csv (for QuickNII output)
        fn = "deepslice_registration_results_cleaned"
        if not os.path.exists(os.path.join(reg, fn + ".csv")):
            json_to_csv(str(os.path.join(reg, fn + ".json")), str(os.path.join(reg, fn + ".csv")))
        # Annotate using ABA
        #run(sys.executable, code_dir/"image_annotation.py", str(os.path.join(reg, "deepslice_registration_results_cleaned.csv")), str(allen), str(ann))
        #run(sys.executable, code_dir/"image_annotation_VisuAlign.py", str(allen), str(ann))
        # Annotate samples with known image locations
        #run(sys.executable, code_dir/"sample_annotation.py", str(ann), str(sample_data_csv))
        run(sys.executable, code_dir/"sample_annotation_VisuAlign.py", str(VA_seg), str(ann), str(sample_data_csv), str(seg))
    if run_figures:
        run(sys.executable, code_dir/"make_figures.py", str(seg), str(ann), str(sample_data_csv), str(fig), str(input_image_dir))
    print(f"✓ {date} finished in {time.time()-t0:.1f}s")