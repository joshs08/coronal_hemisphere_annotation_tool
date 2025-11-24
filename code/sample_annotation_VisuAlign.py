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
from math import ceil, sqrt
import matplotlib
matplotlib.use("Agg") 
from matplotlib import pyplot as plt


def find_nearest_nonzero_entry(arr, row, column):
    rows, columns = np.where(arr)
    distance_squared = (rows - row)**2 + (columns - column)**2
    idx = np.argmin(distance_squared)
    return arr[rows[idx], columns[idx]]

def convert_coordinates(sample_row, sample_col, orig_width, orig_height, VA_width, VA_height):
    converted_row = int((sample_row/orig_height) * VA_height)
    converted_col = int((sample_col/orig_width) * VA_width)
    return converted_row, converted_col

def convert_coordinates_2(
    sample_row,
    sample_col,
    orig_height,
    orig_width,
    VA_height,
    VA_width,
    flip_y=False,
):
    """
    Convert coordinates from raw / segmentation space -> VA label space.

    sample_row, sample_col: indices in [0 .. orig_height-1], [0 .. orig_width-1]
    """

    # Work in float
    r = float(sample_row)
    c = float(sample_col)

    # Normalised to [0, 1]
    # Use (dim - 1) so that max index maps to max index
    r_norm = r / (orig_height - 1)
    c_norm = c / (orig_width  - 1)

    # Scale to VA space
    va_r = r_norm * (VA_height - 1)
    va_c = c_norm * (VA_width  - 1)

    # Optional vertical flip if the VA space uses opposite y direction
    if flip_y:
        va_r = (VA_height - 1) - va_r

    # Round to nearest pixel, then clip
    va_r = int(np.rint(va_r))
    va_c = int(np.rint(va_c))

    va_r = np.clip(va_r, 0, VA_height - 1)
    va_c = np.clip(va_c, 0, VA_width  - 1)

    return va_r, va_c

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
        labels = np.frombuffer(buffer, dtype=np.dtype('>i2'), offset=9)
        labels = labels.reshape(VA_shape[::-1])

        plt.imshow(labels)
        labels_all.append(labels)
        print(nDims,VA_shape)

    # load sample data
    sample_data = pd.read_csv(args.sample_data, index_col="sample_id")
    sample_data["annotation_id"]   = np.full((len(sample_data)), np.nan)
    sample_data["annotation_name"] = np.full((len(sample_data)), np.nan, dtype=str)

    # extra CODE
    
    by_index = {}
    for sid, r in sample_data.iterrows():
        idx_val = int(r["image_index"])
        by_index.setdefault(idx_val, []).append((sid, r))

    # Where to save previews with points
    preview_dir = Path(args.annotation_directory) / "_va_previews_with_samples"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_slices):
        labels = labels_all[i]
        H, W = labels.shape

        csv_idx = i

        # Build the figure
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(labels, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"s{i+1:03d}")

        # If we have any samples for this slice, convert, annotate, and plot them
        if csv_idx in by_index:
            rs, cs = [], []
            for sid, r in by_index[csv_idx]:
                seg_row = float(r["segmentation_row"])
                seg_col = float(r["segmentation_col"])

                VA_row, VA_col = convert_coordinates_2(
                    sample_row=seg_row,
                    sample_col=seg_col,
                    orig_height=orig_height,
                    orig_width=orig_width,
                    VA_height=H,
                    VA_width=W,
                    flip_y=False,   # flip to True if you observe vertical inversion
                )

                # Get label at mapped coord; 0 means background
                ann_id = int(labels[VA_row, VA_col])
                if ann_id == 0:
                    # fallback: choose the nearest nonzero in VA space
                    # ensure we look for nonzero pixels
                    mask_nonzero = (labels != 0)
                    if mask_nonzero.any():
                        rows, cols = np.where(mask_nonzero)
                        d2 = (rows - VA_row)**2 + (cols - VA_col)**2
                        near_idx = int(np.argmin(d2))
                        ann_id = int(labels[rows[near_idx], cols[near_idx]])
                    else:
                        ann_id = 0  # truly empty; leave as background

                # write back to the correct row (sample_id), not by image index
                if ann_id != 0:
                    sample_data.at[sid, "annotation_id"] = ann_id
                    try:
                        sample_data.at[sid, "annotation_name"] = name_map[int(ann_id)]
                    except:
                        sample_data.at[sid, "annotation_name"] = f"Annotation for {ann_id} does not exist."
                else:
                    # If you want to record background explicitly, uncomment:
                    # sample_data.at[sid, "annotation_id"] = 0
                    # sample_data.at[sid, "annotation_name"] = "background"
                    pass

                # collect for plotting
                rs.append(VA_row)
                cs.append(VA_col)

            # Overlay points on the label image
            ax.scatter(cs, rs, s=12, c="r", marker="o", linewidths=0)

        # Save & close
        fig.savefig(preview_dir / f"s{i+1:03d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    ## end
    
    """
    for idx, row in sample_data.iterrows():
        ii = int(row["image_index"])
        jj = int(row["segmentation_row"])
        kk = int(row["segmentation_col"])
        H, W = labels_all[ii].shape
        
        VA_row, VA_column = convert_coordinates_2(
            sample_row=jj,
            sample_col=kk,
            orig_width=orig_width,
            orig_height=orig_height,
            VA_width=W,
            VA_height=H,
            flip_y=False,  # try False first; if clearly inverted, change to True
        )
        
        #VA_row, VA_column = convert_coordinates(sample_row=jj, sample_col=kk, orig_width=orig_width, orig_height=orig_height, VA_width=W, VA_height=H)
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
        """
    sample_data.to_csv(args.sample_data)
    print(sample_data)
