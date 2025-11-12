"""
Use the image annotation results to annotate samples with known image locations.

Example
-------
python code/sample_annotation.py test/annotation/ test/sample_data.csv
"""

import numpy as np
import pandas as pd

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path


def find_nearest_nonzero_entry(arr, row, column):
    rows, columns = np.where(arr)
    distance_squared = (rows - row)**2 + (columns - column)**2
    idx = np.argmin(distance_squared)
    return arr[rows[idx], columns[idx]]

def convert_coordinates(sample_row, sample_col, orig_width, orig_height, VA_width, VA_height):
    converted_row = int((sample_row/orig_height) * VA_height)
    converted_col = int((sample_col/orig_width) * VA_width)
    return converted_row, converted_col

if __name__ == "__main__":
    # load flat
    # get dimensions of original image
    # flat file dimensions
    # convert segmentation_row and segmentation_col 
    # pull label 
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("visualign_seg_directory", help="/path/to/VisuAlign_seg/directory/", type=str)
    parser.add_argument("annotation_directory", help="/path/to/annotation/directory/", type=str)
    parser.add_argument("sample_data",          help="/path/to/sample_data.csv",       type=str)
    parser.add_argument("segmentation_results_directory",     help="/path/to/segmentation/directory/", type=str)
    args = parser.parse_args()

    # load segmentation
    segmentation_directory = Path(args.segmentation_results_directory)
    data = np.load(segmentation_directory / "segmentation_results.npz")
    slice_images  = data["slice_images"]
    slice_masks   = data["slice_masks"]
    sample_masks  = data["sample_masks"]
    slice_numbers = data["slice_numbers"]
    orig_height = slice_images.shape[1]
    orig_width = slice_images.shape[2]
    num_slices = slice_images.shape[0]

    # Load annotation name map
    annotation_data = np.load(Path(args.annotation_directory) / "annotation_results.npz", allow_pickle=True)
    name_map = dict(annotation_data["name_map"])

    # load VisuAlign output
    VA_output_directory = args.visualign_seg_directory
    labels_all = []
    for i in range(num_slices):
        fn = rf"{VA_output_directory}\brain_slice_s{i+1:03d}_nl.flat"
        with open(fn,'rb') as fp:
            buffer = fp.read()
        nDims = int(buffer[0])
        VA_shape = np.frombuffer(buffer, dtype=np.dtype('>i4'), offset=1, count=2) 
        labels = np.frombuffer(buffer, dtype=np.dtype('>u2'), offset=9)
        labels = labels.reshape(VA_shape[::-1])
        labels_all.append(labels)
        print(nDims,VA_shape)

    # load sample data
    sample_data = pd.read_csv(args.sample_data, index_col="sample_id")
    sample_data["annotation_id"]   = np.full((len(sample_data)), np.nan)
    sample_data["annotation_name"] = np.full((len(sample_data)), np.nan, dtype=str)
    for idx, row in sample_data.iterrows():
        ii = int(row["image_index"])
        jj = int(row["segmentation_row"])
        kk = int(row["segmentation_col"])
        H, W = labels_all[ii].shape
        VA_row, VA_column = convert_coordinates(sample_row=jj, sample_col=kk, orig_width=orig_width, orig_height=orig_height, VA_width=W, VA_height=H)
        annotation_id = int(labels_all[ii][VA_row, VA_column])
        if annotation_id:
            sample_data.at[idx, "annotation_id"] = annotation_id
            sample_data.at[idx, "annotation_name"] = name_map[annotation_id]
        else:
            # sample_data.at[idx, "annotation_id"] = annotation_id
            # sample_data.at[idx, "annotation_name"] = "n/a"
            annotation_id = find_nearest_nonzero_entry(labels_all[ii], jj, kk)
            sample_data.at[idx, "annotation_id"] = annotation_id
            sample_data.at[idx, "annotation_name"] = name_map[annotation_id]
            msg = f"Sample with index {idx} was annotated as background!"
            msg += f"\nSelecting the nearest annotated region instead: {name_map[annotation_id]}"
            import warnings
            warnings.warn(msg)

    sample_data.to_csv(args.sample_data)
    print(sample_data)
