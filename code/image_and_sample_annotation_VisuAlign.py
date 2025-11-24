#!/usr/bin/env python3
"""
Use VisuAlign .flat label images to annotate samples and save per-slice previews.

Example:
python sample_annotation_visualign.py \
    /path/to/VisuAlign_seg \
    /path/to/annotation_dir_containing_annotation_results.npz \
    /path/to/sample_data.csv \
    /path/to/segmentation_results_dir \
    /path/to/ccf2017_labels_table.tsv
"""

from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import numpy as np
import pandas as pd

# ---- plotting (save-to-file; avoids GUI hangs) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
CSV_IMAGE_INDEX_IS_ONE_BASED = False   # <-- you said it's NOT one-based


# ==============================
# Utilities
# ==============================
def convert_coordinates_2(
    sample_row: float,
    sample_col: float,
    orig_height: int,
    orig_width: int,
    VA_height: int,
    VA_width: int,
    flip_y: bool = False,
):
    """
    Convert (row, col) from original segmentation image size -> VisuAlign label image size.

    - Uses (dim-1) scaling so max index maps to max index.
    - Optional vertical flip if needed (usually False if images share the same origin).
    """
    r = float(sample_row)
    c = float(sample_col)

    r_norm = r / max(1.0, (orig_height - 1))
    c_norm = c / max(1.0, (orig_width  - 1))

    va_r = r_norm * max(1.0, (VA_height - 1))
    va_c = c_norm * max(1.0, (VA_width  - 1))

    if flip_y:
        va_r = (VA_height - 1) - va_r

    va_r = int(np.rint(va_r))
    va_c = int(np.rint(va_c))

    va_r = int(np.clip(va_r, 0, VA_height - 1))
    va_c = int(np.clip(va_c, 0, VA_width  - 1))
    return va_r, va_c


def csv_index_for_slice(i: int) -> int:
    """Map 0-based slice idx i -> CSV image_index value."""
    return (i + 1) if CSV_IMAGE_INDEX_IS_ONE_BASED else i


def load_name_map(annotation_npz_path: Path) -> dict:
    """Load Allen structure_id -> structure_name dict from your annotation_results.npz."""
    data = np.load(annotation_npz_path, allow_pickle=True)
    # Expect array of (id, name)
    name_map_array = data["name_map"]
    return dict(name_map_array)


def load_visualign_flat(fn: Path) -> np.ndarray:
    """
    Load a VisuAlign .flat file as big-endian unsigned 16-bit label indices.

    File structure (as used here):
      byte 0: nDims (uint8)
      bytes 1..8: two >i4 ints with dims
      bytes 9.. : >u2 label codes, row-major after reshape with VA_shape[::-1]
    """
    with open(fn, "rb") as fp:
        buffer = fp.read()
    nDims = int(buffer[0])  # not used, but kept for parity
    _VA_shape = np.frombuffer(buffer, dtype=np.dtype(">i4"), offset=1, count=2)  # (W, H) or (H, W) depending on exporter
    labels = np.frombuffer(buffer, dtype=np.dtype(">u2"), offset=9)

    # Empirically, the provided shape vector works with a reverse to get (H, W)
    labels = labels.reshape(_VA_shape[::-1])
    return labels


def load_index_to_allen_id(labels_table_tsv: Path) -> dict:
    """
    Load VisuAlign/QuickNII label table (TSV) mapping label 'index' -> Allen 'id'.

    Expected headers: at least 'index' and 'id'.
    """
    df = pd.read_csv(labels_table_tsv, sep="\t")
    # Keep only required cols; cast to int
    idx = df["index"].astype(int).to_numpy()
    aid = df["id"].astype(int).to_numpy()
    return {int(i): int(a) for i, a in zip(idx, aid)}


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("visualign_seg_directory", help="Folder containing brain_slice_sXXX_nl.flat files.", type=str)
    parser.add_argument("annotation_directory",    help="Folder containing annotation_results.npz (with name_map).", type=str)
    parser.add_argument("sample_data",             help="Path to sample_data.csv (will be updated).", type=str)
    parser.add_argument("segmentation_results_directory", help="Folder with segmentation_results.npz for original image sizes.", type=str)
    parser.add_argument("labels_table_tsv",        help="Path to ccf2017 labels table TSV (with 'index' and 'id').", type=str)
    args = parser.parse_args()

    visu_dir   = Path(args.visualign_seg_directory)
    annot_dir  = Path(args.annotation_directory)
    samples_csv = Path(args.sample_data)
    seg_dir    = Path(args.segmentation_results_directory)
    labels_tsv = Path(args.labels_table_tsv)

    # --- sizes from segmentation (for coordinate scaling) ---
    seg = np.load(seg_dir / "segmentation_results.npz")
    slice_images  = seg["slice_images"]
    # slice_masks   = seg["slice_masks"]   # not needed here
    # sample_masks  = seg["sample_masks"]
    # slice_numbers = seg["slice_numbers"]
    orig_height = int(slice_images.shape[1])
    orig_width  = int(slice_images.shape[2])
    num_slices  = int(slice_images.shape[0])

    # --- name map (Allen structure_id -> name) ---
    name_map = load_name_map(annot_dir / "annotation_results.npz")

    # --- VisuAlign label index -> Allen ID mapping ---
    index_to_allen_id = load_index_to_allen_id(labels_tsv)

    # --- load all .flat labels ---
    labels_all = []
    for i in range(num_slices):
        fn = visu_dir / f"brain_slice_s{i+1:03d}_nl.flat"
        labels = load_visualign_flat(fn)      # dtype: >u2, shape: (H, W)
        labels_all.append(labels)

    # --- load samples; prepare output columns ---
    sample_data = pd.read_csv(samples_csv, index_col="sample_id")
    if "annotation_id" not in sample_data.columns:
        sample_data["annotation_id"] = np.nan
    if "annotation_name" not in sample_data.columns:
        sample_data["annotation_name"] = ""

    # Group by CSV image_index, but keep sample_id so we can write back correctly
    by_index = {}
    for sid, row in sample_data.iterrows():
        idx_val = int(row["image_index"])
        by_index.setdefault(idx_val, []).append((sid, row))

    # --- previews output dir ---
    preview_dir = annot_dir / "_va_previews_with_samples"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # --- iterate slices ---
    for i in range(num_slices):
        labels = labels_all[i]
        H, W = labels.shape
        csv_idx = csv_index_for_slice(i)

        # build figure
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(labels, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"s{i+1:03d}")

        if csv_idx in by_index:
            rs_plot, cs_plot = [], []

            for sid, row in by_index[csv_idx]:
                seg_row = float(row["segmentation_row"])
                seg_col = float(row["segmentation_col"])

                va_r, va_c = convert_coordinates_2(
                    sample_row=seg_row,
                    sample_col=seg_col,
                    orig_height=orig_height,
                    orig_width=orig_width,
                    VA_height=H,
                    VA_width=W,
                    flip_y=False,  # set True only if you observe vertical inversion
                )

                # value from .flat is a *label index*, not an Allen structure id
                code = int(labels[va_r, va_c])

                # background (0) fallback: nearest non-zero in VA space
                if code == 0:
                    nz = np.where(labels != 0)
                    if nz[0].size > 0:
                        d2 = (nz[0] - va_r) ** 2 + (nz[1] - va_c) ** 2
                        j = int(np.argmin(d2))
                        code = int(labels[nz[0][j], nz[1][j]])

                # translate index -> allen_id -> name
                allen_id = int(index_to_allen_id.get(code, 0))
                if allen_id != 0:
                    sample_data.at[sid, "annotation_id"] = allen_id
                    # name_map keys are ints or strings depending on np.savez; cast to int
                    if allen_id in name_map:
                        sample_data.at[sid, "annotation_name"] = name_map[allen_id]
                    else:
                        # Sometimes name_map keys load as str; try str fallback
                        nm = name_map.get(str(allen_id), "")
                        sample_data.at[sid, "annotation_name"] = nm

                # collect for plotting
                rs_plot.append(va_r)
                cs_plot.append(va_c)

            # overlay points
            if rs_plot:
                ax.scatter(cs_plot, rs_plot, s=12, c="r", marker="o", linewidths=0)

        # save & close
        fig.savefig(preview_dir / f"s{i+1:03d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # --- write CSV back ---
    sample_data.to_csv(samples_csv)
    print("Updated:", samples_csv)
