"""
Use the image registration results to query the Allen Brain Atlas for a corresponding annotation.

Example
-------
python code/image_annotation.py test/registration/ data/ test/annotation/
"""

import numpy as np


from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from allensdk.core.reference_space_cache import ReferenceSpaceCache


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("aba_directory",     help="/path/to/Allen/Brain/Atlas/data/directory/", type=str)
    parser.add_argument("output_directory",  help="/path/to/output/directory/",                 type=str)
    args = parser.parse_args()

    aba_directory = Path(args.aba_directory)
    aba_directory.mkdir(exist_ok=True)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    print("Constructing the name map array...")
    resolution = 25
    reference_space_key = "annotation/ccf_2017"
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=aba_directory / 'manifest.json')
    tree = rspc.get_structure_tree(structure_graph_id=1) # ID 1 is the adult mouse structure graph
    name_map = tree.get_name_map()

    name_map_array = np.array(list(name_map.items()), dtype=object)
    np.savez(
        output_directory / "annotation_results.npz",
        name_map    = name_map_array,
    )
