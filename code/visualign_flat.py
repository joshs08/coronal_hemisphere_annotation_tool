"""
Utilities for reading VisuAlign flat label images.

VisuAlign ``.flat`` files store a small header followed by big-endian
16-bit label values. The exported shape is stored as width, height, so the
label array is reshaped as height, width for normal image indexing.
"""

import json
from pathlib import Path

import numpy as np


def read_flat_labels(flat_file):
    """Read a 2D VisuAlign ``.flat`` file into a ``(rows, columns)`` array."""
    flat_file = Path(flat_file)
    buffer = flat_file.read_bytes()
    if len(buffer) < 9:
        raise ValueError(f"{flat_file} is too small to be a VisuAlign .flat file.")

    n_dims = int(buffer[0])
    if n_dims != 2:
        raise ValueError(f"{flat_file} has {n_dims} dimensions; only 2D .flat files are supported.")

    dims = np.frombuffer(buffer, dtype=np.dtype(">i4"), offset=1, count=n_dims)
    data_offset = 1 + 4 * n_dims
    labels = np.frombuffer(buffer, dtype=np.dtype(">u2"), offset=data_offset)
    expected = int(np.prod(dims))
    if labels.size != expected:
        raise ValueError(
            f"{flat_file} contains {labels.size} labels, but its header declares {expected}."
        )

    return labels.reshape(tuple(dims[::-1])).astype(np.uint16, copy=False)


def flat_file_for_slice(visualign_directory, slice_number):
    """Return the expected VisuAlign nonlinear flat file for a CHAT slice number."""
    return Path(visualign_directory) / f"brain_slice_s{int(slice_number):03d}_nl.flat"


def resize_labels_nearest(labels, output_shape):
    """
    Resize a label image with nearest-neighbour sampling.

    This uses the same endpoint-preserving coordinate convention as the sample
    coordinate conversion, so the first and last input pixels map to the first
    and last output pixels.
    """
    output_rows, output_cols = map(int, output_shape)
    input_rows, input_cols = labels.shape
    if (input_rows, input_cols) == (output_rows, output_cols):
        return labels.copy()

    row_idx = np.rint(np.linspace(0, input_rows - 1, output_rows)).astype(int)
    col_idx = np.rint(np.linspace(0, input_cols - 1, output_cols)).astype(int)
    return labels[row_idx[:, np.newaxis], col_idx[np.newaxis, :]]


def infer_rainbow_json(visualign_directory):
    """Infer the ``Rainbow 2017.json`` table path for a VisuAlign date folder."""
    visualign_directory = Path(visualign_directory)
    candidates = [
        visualign_directory / "Rainbow 2017.json",
        visualign_directory.parent / "QN" / "Rainbow 2017.json",
        visualign_directory.parent / "Rainbow 2017.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Rainbow 2017.json. Pass --rainbow-json explicitly."
    )


def find_structures_json(aba_directory):
    """Find an Allen ``structures.json`` file below the atlas directory."""
    aba_directory = Path(aba_directory)
    candidates = [
        aba_directory / "structures.json",
        aba_directory / "ccf_2017" / "structures.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(aba_directory.rglob("structures.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find structures.json below {aba_directory}.")


def load_rainbow_index_to_allen_id(rainbow_json, aba_directory):
    """
    Map VisuAlign/Rainbow table indices to real Allen structure IDs.

    The numeric values in VisuAlign ``.flat`` files are the ``index`` values
    from ``Rainbow 2017.json``. They are not Allen IDs. The table names map
    uniquely to Allen structures in ``structures.json``.
    """
    with open(rainbow_json) as fp:
        rainbow_table = json.load(fp)
    with open(find_structures_json(aba_directory)) as fp:
        structures = json.load(fp)

    name_to_id = {}
    duplicate_names = set()
    for structure in structures:
        name = structure["name"]
        if name in name_to_id:
            duplicate_names.add(name)
        name_to_id[name] = int(structure["id"])
    if duplicate_names:
        duplicates = ", ".join(sorted(duplicate_names)[:5])
        raise ValueError(f"Allen structure names are not unique: {duplicates}")

    mapping = {0: 0}
    missing = []
    for item in rainbow_table:
        index = int(item["index"])
        name = str(item["name"])
        if name == "empty":
            mapping[index] = 0
            continue
        allen_id = name_to_id.get(name)
        if allen_id is None:
            missing.append((index, name))
        else:
            mapping[index] = allen_id

    if missing:
        examples = ", ".join(f"{index}: {name}" for index, name in missing[:5])
        raise ValueError(f"Rainbow labels missing from Allen structures.json: {examples}")
    return mapping


def translate_indices_to_allen_ids(labels, index_to_allen_id):
    """Translate VisuAlign/Rainbow index labels to Allen structure IDs."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(np.int32)

    max_label = int(labels.max())
    missing = sorted(
        int(label)
        for label in np.unique(labels)
        if int(label) not in index_to_allen_id
    )
    if missing:
        raise ValueError(f"Missing Rainbow-to-Allen mapping for labels: {missing[:20]}")

    lookup = np.zeros(max_label + 1, dtype=np.int32)
    for index, allen_id in index_to_allen_id.items():
        if index <= max_label:
            lookup[int(index)] = int(allen_id)
    return lookup[labels]


def convert_point_between_shapes(row, col, source_shape, target_shape, flip_y=False):
    """Convert an image-space row/column coordinate between two image shapes."""
    source_rows, source_cols = map(float, source_shape)
    target_rows, target_cols = map(int, target_shape)

    row_norm = float(row) / max(1.0, source_rows - 1.0)
    col_norm = float(col) / max(1.0, source_cols - 1.0)

    target_row = row_norm * max(1, target_rows - 1)
    target_col = col_norm * max(1, target_cols - 1)
    if flip_y:
        target_row = (target_rows - 1) - target_row

    target_row = int(np.clip(np.rint(target_row), 0, target_rows - 1))
    target_col = int(np.clip(np.rint(target_col), 0, target_cols - 1))
    return target_row, target_col


def deterministic_color(label_id):
    """Generate a stable RGB colour for labels missing from the Allen colour map."""
    label_id = int(label_id)
    value = (label_id * 2654435761) & 0xFFFFFFFF
    red = 60 + (value & 0x7F)
    green = 60 + ((value >> 8) & 0x7F)
    blue = 60 + ((value >> 16) & 0x7F)
    return np.array([red, green, blue], dtype=np.uint8)


def image_from_labels(labels, color_map):
    """Convert a label image to RGB using an Allen-style ``id -> RGB`` colour map."""
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label_id in np.unique(labels):
        label_id = int(label_id)
        if label_id == 0:
            continue
        color = color_map.get(label_id)
        if color is None:
            color = deterministic_color(label_id)
        image[labels == label_id] = np.asarray(color, dtype=np.uint8)
    return image


def nearest_nonzero(labels, row, col):
    """Return the nearest non-zero label to ``(row, col)``, or 0 if none exist."""
    rows, cols = np.where(labels != 0)
    if rows.size == 0:
        return 0
    distances = (rows - row) ** 2 + (cols - col) ** 2
    idx = int(np.argmin(distances))
    return int(labels[rows[idx], cols[idx]])
