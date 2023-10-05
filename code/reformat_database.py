import pathlib
import pandas as pd

filepath = pathlib.Path("/home/paul/src/coronal_hemisphere_annotation_tool/data/plates_6_11_12_13_14_16_17_18.tsv")
data = pd.read_csv(filepath, sep='\t')

# normalize / rename variables
data.rename(columns={
    "X   (µm)"     : "image_x",
    "Y  (µm)"      : "image_y",
    "Z (µm)"       : "image_z",
    "Slice number" : "slice_number",
    "Date"         : "date",
    "Mouse"        : "mouse",
}, inplace=True)

# export
output_columns = [
    "plate", "well", "barcode", "date", "mouse",
    "class_label", "class_color", "subclass_label", "subclass_color",
    "image_x", "image_y", "image_z", "slice_number",
]
data.to_csv(filepath.with_suffix(".csv"), columns=output_columns, index=False)

print(data[output_columns])
