#!/usr/bin/env python
"""
Register a series of mouse coronal sections to the Allen Brain Atlas using DeepSlice.

Images in the provided directory have to be PNG files and contain the
slice number in the filename separated by underscores. For example: Image_01.png.

Example
-------
python image_registration.py \
    test/segmentation \
    test/registration \
    --slice_direction caudal-rostro \
    --slice_thickness 150 \

See also
--------
https://github.com/PolarBean/DeepSlice

"""

from pathlib import Path
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from DeepSlice import DSModel

if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("image_directory", help="/path/to/image/directory/", type=str)
    parser.add_argument("output_directory", help="/path/to/output/directory/", type=str)
    parser.add_argument("--slice_thickness", help="Slice thickness in micrometers.", type=float, default=150)
    parser.add_argument("--slice_direction", help="Either rostro-caudal or caudal-rostro.", type=str, default="caudal-rostro")
    args = parser.parse_args()

    Model = DSModel("mouse")
    Model.predict(args.image_directory, ensemble=True, section_numbers=True)
    if args.slice_direction == "caudal-rostro":
        Model.enforce_index_spacing(section_thickness=-args.slice_thickness)
    elif args.slice_direction == "rostro-caudal":
        Model.enforce_index_spacing(section_thickness=args.slice_thickness)
    else:
        raise ValueError("Parameter section_thickness one of rostro-caudal or caudal-rostro, not {args.section_direction}.")
    Model.propagate_angles()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    Model.save_predictions(str(output_directory) + '/deepslice_registration_results')
