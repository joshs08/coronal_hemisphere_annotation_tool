#!/usr/bin/env python
"""
Given sample coordinates in micrometers, this script determines the
corresponding image pixels.

The sample data CSV has to contain the following columns:

- slice_number,
- image_x,
- image_y.

The images in the image directory have to be TIFF files containing
XResolution and YResolution meta data tags, and have to match the
following file name pattern: Image_[0-9]*.tif, where the number
indicates the slice number.

Example
-------
python code/convert_sample_coordinates.py test/sample_data.csv test/ --show

"""

import glob
import numpy as np
import pandas as pd
import os

from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from PIL import Image
from PIL.TiffTags import TAGS


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("sample_data",                 help="/path/to/sample_data.csv",  type=str)
    parser.add_argument("image_directory",             help="/path/to/image/directory/", type=str)
    parser.add_argument("--show", action="store_true", help="Print header of output data frame.")
    args = parser.parse_args()

    data = pd.read_csv(args.sample_data)

    # glob image file path and order by slice number
    dir_with_sep = os.path.join(args.image_directory, '')
    filepaths = [Path(path) for path in glob.glob(dir_with_sep + "*.tif")]
    slice_numbers = [next(int(substring) for substring in path.stem.split("_") if substring.isdigit()) for path in filepaths]
    order = np.argsort(slice_numbers)
    slice_numbers = [slice_numbers[ii] for ii in order]
    filepaths = [filepaths[ii] for ii in order]

    print("Processing coordinates by image...")
    data["image_index"]        = np.full(len(data), np.nan, dtype=int)
    data["image_col"]          = np.full(len(data), np.nan, dtype=int)
    data["image_row"]          = np.full(len(data), np.nan, dtype=int)
    data["image_x_resolution"] = np.full(len(data), np.nan)
    data["image_y_resolution"] = np.full(len(data), np.nan)
    for idx, (slice_number, filepath) in enumerate(zip(slice_numbers, filepaths)):
        print(filepath)
        is_in_slice = data["slice_number"] == slice_number

        if np.any(is_in_slice):
            img = Image.open(filepath)
            meta_data = {TAGS[key] : img.tag[key] for key in img.tag_v2}
            x_resolution = meta_data["XResolution"][0][0] / meta_data["XResolution"][0][1]
            y_resolution = meta_data["YResolution"][0][0] / meta_data["YResolution"][0][1]

            sample_coordinates = data[is_in_slice][["image_x", "image_y"]].values
            rows = np.round(sample_coordinates[:, 1] * y_resolution).astype(int)
            cols = np.round(sample_coordinates[:, 0] * x_resolution).astype(int)

            data.loc[is_in_slice, "image_index"]        = idx
            data.loc[is_in_slice, "image_col"]          = cols
            data.loc[is_in_slice, "image_row"]          = rows
            data.loc[is_in_slice, "image_x_resolution"] = x_resolution
            data.loc[is_in_slice, "image_y_resolution"] = y_resolution

    # export
    data.to_csv(args.sample_data, index=False)

    if args.show:
        print()
        print(data)
