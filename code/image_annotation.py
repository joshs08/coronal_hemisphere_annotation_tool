"""
Use the image registration results to query the Allen Brain Atlas for a corresponding annotation.

Example
-------
python code/image_annotation.py test/registration/ data/ test/annotation/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from PIL import Image

from allensdk.core.reference_space_cache import ReferenceSpaceCache


def get_aba_annotation(annotation_volume, image_height, image_width, o, u, v):
    total_pixels = image_height * image_width
    annotation = np.zeros((total_pixels))
    xv, yv, zv = get_aba_coordinates(image_height, image_width, o, u, v)
    max_xv, max_yv, max_zv = annotation_volume.shape
    is_valid = (xv < max_xv) & (yv < max_yv) & (zv < max_zv)
    annotation[is_valid] = annotation_volume[xv[is_valid], yv[is_valid], zv[is_valid]]
    return annotation.reshape((image_height, image_width))


def get_aba_coordinates(image_height, image_width, o, u, v):
    # transform x, y pixel coordinates into ABA voxel coordinates
    # https://www.nitrc.org/plugins/mwiki/index.php?title=quicknii:Image_coordinates

    x = np.arange(0, image_width) / image_width
    y = np.arange(0, image_height) / image_height
    xx, yy = np.meshgrid(x, y)
    zz = np.ones_like(xx)
    xxyyzz = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    transformation_matrix = np.c_[u, v, o]
    xxyyzz_transformed = xxyyzz @ transformation_matrix.T
    xv, yv, zv = np.round(xxyyzz_transformed).astype(int).transpose()

    # # alternatively
    # ux = u[:, np.newaxis] * xx.ravel()[np.newaxis, :]
    # vy = v[:, np.newaxis] * yy.ravel()[np.newaxis, :]
    # out = o[:, np.newaxis] + ux + vy
    # xv, yv, zv = np.round(out).astype(int)

    # switch and flip coordinates
    # https://www.nitrc.org/plugins/mwiki/index.php?title=quicknii:Coordinate_systems
    xv, yv, zv = 528 - yv, 320 - zv, xv

    return xv, yv, zv


def get_aba_image(annotation, cmap):
    image = np.zeros((*annotation.shape, 3), dtype=np.uint8)
    for value in np.unique(annotation):
        if value > 0:
            mask = annotation == value
            image[mask] = cmap[value]
    return image


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("registration_file", help="/path/to/registration/results.csv",          type=str)
    parser.add_argument("aba_directory",     help="/path/to/Allen/Brain/Atlas/data/directory/", type=str)
    parser.add_argument("output_directory",  help="/path/to/output/directory/",                 type=str)
    args = parser.parse_args()

    aba_directory = Path(args.aba_directory)
    aba_directory.mkdir(exist_ok=True)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    print("Constructing the ABA annotation volume...")
    resolution = 25
    reference_space_key = "annotation/ccf_2017"
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=aba_directory / 'manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1) # ID 1 is the adult mouse structure graph
    annotation, meta = rspc.get_annotation_volume()
    rsp = rspc.get_reference_space()
    color_map = tree.get_colormap()
    name_map = tree.get_name_map()

    print("Computing annotations for each image...")
    registration = pd.read_csv(args.registration_file)
    aba_annotations = []
    for ii, row in registration.iterrows():
        print(f"{ii + 1} / {len(registration)}")
        u = np.array([row.ux, row.uy, row.uz])
        v = np.array([row.vx, row.vy, row.vz])
        o = np.array([row.ox, row.oy, row.oz])
        aba_annotation = get_aba_annotation(rsp.annotation, row.height, row.width, o, u, v)
        aba_annotations.append(aba_annotation)

    print("Visualising annotations...")
    aba_images = []
    for ii, row in registration.iterrows():
        print(f"{ii + 1} / {len(registration)}")
        aba_image = get_aba_image(aba_annotations[ii], cmap=color_map)
        aba_images.append(aba_image)
        try:
            Image.fromarray(aba_image).save(output_directory / row.Filenames.replace(".png", "_aba_annotation.png"))
        except:
            Image.fromarray(aba_image).save(output_directory / row.filename.replace(".png", "_aba_annotation.png"))


    print("Saving results...")
    color_map_array = np.c_[list(color_map.keys()), list(color_map.values())]
    name_map_array = np.array(list(name_map.items()), dtype=object)
    np.savez(
        output_directory / "annotation_results.npz",
        annotations = np.array(aba_annotations),
        images      = np.array(aba_images),
        color_map   = color_map_array,
        name_map    = name_map_array,
    )
