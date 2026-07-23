"""
Annotate samples directly from VisuAlign nonlinear ``.flat`` files.

This is useful for quick checks. For the full CHAT plotting pipeline, prefer:

1. image_annotation_VisuAlign.py
2. sample_annotation.py
3. make_figures.py
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import re

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

from visualign_flat import (
    convert_point_between_shapes,
    flat_file_for_slice,
    infer_rainbow_json,
    load_rainbow_index_to_allen_id,
    nearest_nonzero,
    read_flat_labels,
    translate_indices_to_allen_ids,
)


def parse_slice_number(path):
    match = re.search(r"_s(\d+)", Path(path).stem)
    if not match:
        raise ValueError(f"Could not parse slice number from {path}.")
    return int(match.group(1))


def get_segmentation_stack_info(segmentation_directory):
    segmentation_directory = Path(segmentation_directory)
    pngs = sorted(segmentation_directory.glob("brain_slice_s*.png"))
    if pngs:
        slice_numbers = [parse_slice_number(path) for path in pngs]
        width, height = Image.open(pngs[0]).size
        return slice_numbers, (height, width)

    data = np.load(segmentation_directory / "segmentation_results.npz")
    slice_numbers = [int(x) for x in data["slice_numbers"]]
    image_shape = tuple(int(x) for x in data["slice_images"].shape[1:3])
    return slice_numbers, image_shape


def load_name_map(annotation_directory):
    annotation_data = np.load(Path(annotation_directory) / "annotation_results.npz", allow_pickle=True)
    return {int(k): str(v) for k, v in annotation_data["name_map"]}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("visualign_seg_directory", help="Folder containing brain_slice_sXXX_nl.flat files.", type=str)
    parser.add_argument("annotation_directory", help="Folder containing annotation_results.npz.", type=str)
    parser.add_argument("sample_data", help="/path/to/sample_data.csv; overwritten with annotations.", type=str)
    parser.add_argument("segmentation_results_directory", help="CHAT segmentation folder.", type=str)
    parser.add_argument("--flip-y", action="store_true", help="Flip sample rows while mapping into VisuAlign space.")
    parser.add_argument(
        "--one-based-image-index",
        action="store_true",
        help="Use this if sample_data.image_index stores 1 for the first slice.",
    )
    parser.add_argument(
        "--aba-directory",
        help="Allen Brain Atlas data folder containing structures.json.",
        type=str,
        default="J:/CHAT/ABA",
    )
    parser.add_argument(
        "--rainbow-json",
        help="Path to the Rainbow 2017.json table used by QuickNII/VisuAlign.",
        type=str,
        default=None,
    )
    parser.add_argument("--no-previews", action="store_true", help="Do not write per-slice sample overlay PNGs.")
    args = parser.parse_args()

    slice_numbers, segmentation_shape = get_segmentation_stack_info(args.segmentation_results_directory)
    name_map = load_name_map(args.annotation_directory)
    rainbow_json = Path(args.rainbow_json) if args.rainbow_json else infer_rainbow_json(args.visualign_seg_directory)
    index_to_allen_id = load_rainbow_index_to_allen_id(rainbow_json, args.aba_directory)

    labels_all = []
    for slice_number in slice_numbers:
        labels = read_flat_labels(flat_file_for_slice(args.visualign_seg_directory, slice_number))
        labels_all.append(translate_indices_to_allen_ids(labels, index_to_allen_id))

    sample_data = pd.read_csv(args.sample_data, index_col="sample_id")
    sample_data["annotation_id"] = np.full(len(sample_data), np.nan)
    sample_data["annotation_name"] = np.full(len(sample_data), np.nan, dtype=object)

    by_index = {}
    for sample_id, row in sample_data.iterrows():
        image_index = int(row["image_index"])
        if args.one_based_image_index:
            image_index -= 1
        by_index.setdefault(image_index, []).append((sample_id, row))

    preview_dir = Path(args.annotation_directory) / "_va_previews_with_samples"
    if not args.no_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for image_index, labels in enumerate(labels_all):
        rows_to_plot = []
        cols_to_plot = []

        for sample_id, row in by_index.get(image_index, []):
            va_row, va_col = convert_point_between_shapes(
                row=float(row["segmentation_row"]),
                col=float(row["segmentation_col"]),
                source_shape=segmentation_shape,
                target_shape=labels.shape,
                flip_y=args.flip_y,
            )

            annotation_id = int(labels[va_row, va_col])
            if annotation_id == 0:
                annotation_id = nearest_nonzero(labels, va_row, va_col)

            if annotation_id != 0:
                sample_data.at[sample_id, "annotation_id"] = annotation_id
                sample_data.at[sample_id, "annotation_name"] = name_map.get(
                    annotation_id,
                    f"Unknown VisuAlign/Allen label {annotation_id}",
                )

            rows_to_plot.append(va_row)
            cols_to_plot.append(va_col)

        if not args.no_previews:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(labels, interpolation="nearest")
            ax.axis("off")
            ax.set_title(f"s{slice_numbers[image_index]:03d}")
            if rows_to_plot:
                ax.scatter(cols_to_plot, rows_to_plot, s=12, c="r", marker="o", linewidths=0)
            fig.savefig(preview_dir / f"s{slice_numbers[image_index]:03d}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

    sample_data.to_csv(args.sample_data)
    print(sample_data)
