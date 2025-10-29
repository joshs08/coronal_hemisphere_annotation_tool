from pathlib import Path
import subprocess, sys, time
from datetime import datetime
import os

def to_yyyy_mm_dd(name: str) -> str:
    return datetime.strptime(name, "%d_%m_%y").strftime("%Y-%m-%d")

def run(*args):
    print(">", " ".join(map(str, args)))
    subprocess.run(args, check=True)

def join_dir(*parts) -> str:
    """Join parts and guarantee a trailing slash."""
    return os.path.join(*parts, "")

input_csv = "J:/CHAT/data/plates_6_11_12_13_14_16_17_18.csv"
input_image_root = "J:/CHAT/data/"
output_dir = "J:/CHAT/output_251029/"
code_dir = Path("C:/Users/Josh Selfe/Documents/GitHub/coronal_hemisphere_annotation_tool/code/")
allen = "J:/CHAT/ABA"

exclude_completed = True # Do we bypass folders already present in output folder?
exclude_slices = False # Do we run the script to exclude damaged slices?
run_pipeline = True # Do we run the rest of the pipeline?
run_figures = True

#excluded_folders = ["2018-10-02", "2018-10-03"]
#dates = ["2019-04-11"] # YYYY-MM-DD
image_dates = sorted(
    p.name for p in Path(input_image_root).iterdir()
    if p.is_dir() and "copy" not in p.name.lower()
)

converted = {d: to_yyyy_mm_dd(d) for d in image_dates}
if exclude_completed:
    existing_dates = {p.name for p in Path(output_dir).iterdir() if p.is_dir()} if Path(output_dir).exists() else set()
    image_dates = [d for d, conv in converted.items() if conv not in existing_dates]

#image_dates = ["04_09_18"]

slice_direction = "caudal-rostro"
slice_thickness = 150

for image_date in image_dates:
    date = datetime.strptime(image_date, "%d_%m_%y").strftime("%Y-%m-%d")
    sample_data_csv = os.path.join(output_dir, date, f"sample_data_{date}.csv")
    input_image_dir = join_dir(input_image_root, image_date)
    # Get old midlines
    midlines_date_dir = os.path.join(input_image_dir, "segmentation", "midlines.npy")

    output_dir_date = join_dir(output_dir, date)
    Path(output_dir_date).mkdir(parents=True, exist_ok=True)

    seg, reg, ann, fig = (join_dir(output_dir_date, "segmentation"), 
                          join_dir(output_dir_date, "registration"), 
                          join_dir(output_dir_date, "annotation"), 
                          join_dir(output_dir_date, "figures"))
    # Get new midlines
    #midlines_date_dir = os.path.join(seg, "midlines.npy")

    t0 = time.time()

    pdf_file = join_dir(fig, f"individual_slices_{date.replace('-', '_')}.pdf")
    if run_pipeline:
        run(sys.executable, code_dir/"extract_sample_data.py", str(input_csv), "date", str(date), str(sample_data_csv), "--show")
        run(sys.executable, code_dir/"convert_sample_coordinates.py", str(sample_data_csv), str(input_image_dir))
        run(sys.executable, code_dir/"hemisphere_segmentation_and_alignment.py", str(input_image_dir), str(sample_data_csv), str(seg), "--midlines", str(midlines_date_dir))
    if exclude_slices:
        try:
            #run(sys.executable, code_dir/"hemisphere_segmentation_and_alignment.py", str(input_image_dir), str(sample_data_csv), str(seg), "--midlines", str(midlines_date_dir))
            run(sys.executable, code_dir/"hemisphere_segmentation_and_alignment.py", str(input_image_dir), str(sample_data_csv), str(seg), "--show")
            #run(sys.executable, code_dir/"exclude_damaged_slices.py", str(seg), str(pdf_file), str(sample_data_csv))
        except:
            print(f"Exclusion not possible for {date}.")
    if run_pipeline:
        # Register using deep slice
        run(sys.executable, code_dir/"image_registration.py", str(seg), str(reg),
            "--slice_direction", str(slice_direction), "--slice_thickness", str(slice_thickness))
        # Annotate using ABA
        run(sys.executable, code_dir/"image_annotation.py", str(os.path.join(reg, "deepslice_registration_results.csv")), str(allen), str(ann))
        # Annotate samples with known image locations
        run(sys.executable, code_dir/"sample_annotation.py", str(ann), str(sample_data_csv))
    if run_figures:
        run(sys.executable, code_dir/"make_figures.py", str(seg), str(ann), str(sample_data_csv), str(fig))
    print(f"✓ {date} finished in {time.time()-t0:.1f}s")