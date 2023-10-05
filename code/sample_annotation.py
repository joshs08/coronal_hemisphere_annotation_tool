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


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("annotation_directory", help="/path/to/annotation/directory/", type=str)
    parser.add_argument("sample_data",          help="/path/to/sample_data.csv",       type=str)
    args = parser.parse_args()

    annotation_data = np.load(Path(args.annotation_directory) / "annotation_results.npz", allow_pickle=True)
    annotations = annotation_data["annotations"]
    name_map = dict(annotation_data["name_map"])

    sample_data = pd.read_csv(args.sample_data, index_col="sample_id")
    sample_data["annotation_id"]   = np.full((len(sample_data)), np.nan)
    sample_data["annotation_name"] = np.full((len(sample_data)), np.nan, dtype=str)
    for idx, row in sample_data.iterrows():
        ii = int(row["image_index"])
        jj = int(row["segmentation_row"])
        kk = int(row["segmentation_col"])
        annotation_id = annotations[ii, jj, kk]
        if annotation_id:
            sample_data.at[idx, "annotation_id"] = annotation_id
            sample_data.at[idx, "annotation_name"] = name_map[annotation_id]
        else:
            # sample_data.at[idx, "annotation_id"] = annotation_id
            # sample_data.at[idx, "annotation_name"] = "n/a"
            annotation_id = find_nearest_nonzero_entry(annotations[ii], jj, kk)
            sample_data.at[idx, "annotation_id"] = annotation_id
            sample_data.at[idx, "annotation_name"] = name_map[annotation_id]
            msg = f"Sample with index {idx} was annotated as background!"
            msg += f"\nSelecting the nearest annotated region instead: {name_map[annotation_id]}"
            import warnings
            warnings.warn(msg)

    sample_data.to_csv(args.sample_data)
    print(sample_data)
