"""
Convert VisuAlign nonlinear ``.flat`` files into CHAT annotation results.

The output ``annotation_results.npz`` uses the same fields as
``image_annotation.py``:

- annotations: Allen structure IDs in CHAT segmentation image space
- images: RGB renderings of the annotations
- color_map: rows of ``annotation_id, red, green, blue``
- name_map: rows of ``annotation_id, structure_name``

Example
-------
python code/image_annotation_VisuAlign.py \
    J:/CHAT/VisuAlign_processing/2019-03-22/segmentation \
    J:/CHAT/output_251030_new_midlines/2019-03-22/segmentation \
    J:/CHAT/ABA \
    J:/CHAT/VisuAlign_processing/2019-03-22/annotation \
    --reference-annotation J:/CHAT/QuickNII_processing/2019-03-22/annotation
"""

import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
from PIL import Image

from visualign_flat import (
    deterministic_color,
    flat_file_for_slice,
    infer_rainbow_json,
    image_from_labels,
    load_rainbow_index_to_allen_id,
    read_flat_labels,
    resize_labels_nearest,
    translate_indices_to_allen_ids,
)


def parse_slice_number(path):
    match = re.search(r"_s(\d+)", Path(path).stem)
    if not match:
        raise ValueError(f"Could not parse slice number from {path}.")
    return int(match.group(1))


def get_segmentation_stack_info(segmentation_directory):
    """Return ``(slice_numbers, image_shape)`` without loading the large NPZ."""
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


def load_metadata_from_annotation_npz(annotation_directory):
    path = Path(annotation_directory) / "annotation_results.npz"
    data = np.load(path, allow_pickle=True)

    name_map = {}
    if "name_map" in data.files:
        name_map = {int(k): str(v) for k, v in data["name_map"]}

    color_map = {}
    if "color_map" in data.files:
        color_map_array = data["color_map"]
        for row in color_map_array:
            color_map[int(row[0])] = np.asarray(row[1:4], dtype=np.uint8)

    return name_map, color_map


def load_metadata_from_allen_sdk(aba_directory):
    try:
        from allensdk.core.reference_space_cache import ReferenceSpaceCache
    except ImportError as exc:
        raise RuntimeError(
            "AllenSDK is not installed in this Python environment. Pass "
            "--reference-annotation pointing at an existing CHAT annotation "
            "folder, or run this from the CHAT conda environment."
        ) from exc

    aba_directory = Path(aba_directory)
    aba_directory.mkdir(exist_ok=True)

    resolution = 25
    reference_space_key = "annotation/ccf_2017"
    rspc = ReferenceSpaceCache(
        resolution,
        reference_space_key,
        manifest=aba_directory / "manifest.json",
    )
    tree = rspc.get_structure_tree(structure_graph_id=1)
    name_map = {int(k): str(v) for k, v in tree.get_name_map().items()}
    color_map = {
        int(k): np.asarray(v, dtype=np.uint8)
        for k, v in tree.get_colormap().items()
    }
    return name_map, color_map


def ensure_metadata_for_labels(name_map, color_map, labels):
    """Fill missing names/colours for all observed non-zero label IDs."""
    for label_id in sorted(int(x) for x in np.unique(labels) if int(x) != 0):
        if label_id not in name_map:
            name_map[label_id] = f"Unknown VisuAlign/Allen label {label_id}"
        if label_id not in color_map:
            color_map[label_id] = deterministic_color(label_id)


def save_color_map_array(color_map):
    rows = [
        [int(label_id), *np.asarray(color, dtype=np.uint8).tolist()]
        for label_id, color in sorted(color_map.items())
    ]
    return np.asarray(rows, dtype=np.int64)


def save_name_map_array(name_map):
    return np.asarray(sorted((int(k), str(v)) for k, v in name_map.items()), dtype=object)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "visualign_seg_directory",
        help="Folder containing brain_slice_sXXX_nl.flat files.",
        type=str,
    )
    parser.add_argument(
        "segmentation_directory",
        help="CHAT segmentation folder containing brain_slice_sXXX.png files.",
        type=str,
    )
    parser.add_argument(
        "aba_directory",
        help="Allen Brain Atlas data folder, used only if metadata is not supplied.",
        type=str,
    )
    parser.add_argument(
        "output_directory",
        help="Folder where annotation_results.npz and preview PNGs will be written.",
        type=str,
    )
    parser.add_argument(
        "--reference-annotation",
        help="Existing CHAT annotation folder to reuse name_map/color_map from.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--rainbow-json",
        help="Path to the Rainbow 2017.json table used by QuickNII/VisuAlign.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no-rgb-stack",
        help="Do not store the RGB image stack inside annotation_results.npz.",
        action="store_true",
    )
    args = parser.parse_args()

    visualign_seg_directory = Path(args.visualign_seg_directory)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    slice_numbers, output_shape = get_segmentation_stack_info(args.segmentation_directory)
    rainbow_json = Path(args.rainbow_json) if args.rainbow_json else infer_rainbow_json(visualign_seg_directory)
    index_to_allen_id = load_rainbow_index_to_allen_id(rainbow_json, args.aba_directory)

    if args.reference_annotation is not None:
        name_map, color_map = load_metadata_from_annotation_npz(args.reference_annotation)
    elif (output_directory / "annotation_results.npz").exists():
        name_map, color_map = load_metadata_from_annotation_npz(output_directory)
    else:
        name_map, color_map = load_metadata_from_allen_sdk(args.aba_directory)

    annotations = np.empty((len(slice_numbers), *output_shape), dtype=np.int32)
    images = None
    if not args.no_rgb_stack:
        images = np.empty((len(slice_numbers), *output_shape, 3), dtype=np.uint8)

    print("Reading VisuAlign flat files...")
    for stack_index, slice_number in enumerate(slice_numbers):
        flat_file = flat_file_for_slice(visualign_seg_directory, slice_number)
        print(f"{stack_index + 1} / {len(slice_numbers)}: {flat_file.name}")
        labels = read_flat_labels(flat_file)
        labels = resize_labels_nearest(labels, output_shape)
        labels = translate_indices_to_allen_ids(labels, index_to_allen_id)
        ensure_metadata_for_labels(name_map, color_map, labels)
        annotations[stack_index] = labels

        rgb = image_from_labels(labels, color_map)
        Image.fromarray(rgb).save(output_directory / f"brain_slice_s{slice_number:03d}_aba_annotation.png")
        if not args.no_rgb_stack:
            images[stack_index] = rgb

    color_map_array = save_color_map_array(color_map)
    name_map_array = save_name_map_array(name_map)

    print("Saving annotation_results.npz...")
    if args.no_rgb_stack:
        np.savez(
            output_directory / "annotation_results.npz",
            annotations=annotations,
            color_map=color_map_array,
            name_map=name_map_array,
        )
    else:
        np.savez(
            output_directory / "annotation_results.npz",
            annotations=annotations,
            images=images,
            color_map=color_map_array,
            name_map=name_map_array,
        )

    print(f"Saved: {output_directory / 'annotation_results.npz'}")
