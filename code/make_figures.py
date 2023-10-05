"""
Create an output PDF that displays each segmented slice,
the corresponding Allen Brain Atlas annnotation,
and the isolated samples including their barcode.

Example
-------
python code/05_make_figures.py test/segmentation/ test/annotation/ test/sample_data.csv test/figures/

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from matplotlib.backends import backend_pdf
from matplotlib.colors import to_hex
from skimage.measure import label, find_contours
from shapely.geometry import Polygon
from shapely.ops import polylabel


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)

    parser = ArgumentParser()
    parser.add_argument("segmentation",     help="/path/to/segmentation/directory/", type=str)
    parser.add_argument("annotation",       help="/path/to/annotation/directory/",   type=str)
    parser.add_argument("sample_data",      help="/path/to/sample_data.csv",         type=str)
    parser.add_argument("output_directory", help="/path/to/ouput/directory/",        type=str)
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    # load segmentation
    segmentation_directory = Path(args.segmentation)
    data = np.load(segmentation_directory / "segmentation_results.npz")
    slice_images  = data["slice_images"]
    slice_masks   = data["slice_masks"]
    sample_masks  = data["sample_masks"]
    slice_numbers = data["slice_numbers"]

    df = pd.read_csv(args.sample_data)

    # load annotation
    data = np.load(Path(args.annotation) / "annotation_results.npz", allow_pickle=True)
    annotations     = data["annotations"]
    color_map_array = data["color_map"]
    name_map_array  = data["name_map"]

    color_map = dict(zip(color_map_array[:, 0], color_map_array[:, 1:]))
    name_map = dict(name_map_array)

    # determine image center
    total_slices, total_rows, total_cols = slice_images.shape
    xc = total_cols / 2
    yc = total_rows / 2

    # --------------------------------------------------------------------------------

    print("Plotting slices & annotations...")

    with backend_pdf.PdfPages(output_directory / "individual_slices.pdf") as pdf:

        # for ii in range(total_slices):
        for ii in range(total_slices):
            print(f"{ii+1} / {total_slices}")
            slice_number = slice_numbers[ii]
            slice_image  = slice_images[ii]
            slice_mask   = slice_masks[ii]
            annotation   = annotations[ii]
            sample_mask  = sample_masks[ii]

            fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 10))
            for ax in axes:
                ax.axis("off")
                ax.set_aspect("equal")
            axes[0].imshow(slice_image, cmap="gray")

            # plot slice contour
            contours = find_contours(slice_mask, 0.5)
            for contour in contours:
                x = contour[:, 1]
                y = contour[:, 0]
                axes[-1].plot(x, y, "#677e8c", linewidth=1.0)

            # compute slice radius
            slice_radius = 0
            center = np.array([yc, xc])
            for contour in contours:
                delta = contour - center[np.newaxis, :]
                distance = np.linalg.norm(delta, axis=1)
                slice_radius = max(slice_radius, np.max(distance))

            # plot region contours
            for annotation_id in np.unique(annotation):
                if annotation_id > 0: # 0 is background
                    region_mask = annotation == annotation_id
                    for contour in find_contours(region_mask, 0.5):
                        x = contour[:, 1]
                        y = contour[:, 0]
                        axes[-1].plot(x, y, color=to_hex(color_map[annotation_id]/255), linewidth=0.25)

                    # label regions that have a sample in them
                    if np.any(np.logical_and(region_mask, ~np.isnan(sample_mask))):
                        region_mask[:, :int(xc) + 1] = False
                        # If the current slice/view contains several disjunct regions with that label,
                        # only plot and label the subregion(s) containing the sample.
                        labelled_region_mask, num = label(region_mask, return_num=True)
                        if num > 1:
                            subregions = [subregion for subregion in np.unique(labelled_region_mask[~np.isnan(sample_mask)]) if subregion != 0]
                            tmp = np.zeros_like(region_mask)
                            for subregion in subregions:
                                is_subregion = labelled_region_mask == subregion
                                tmp[is_subregion] = True
                            region_mask = tmp
                        for contour in find_contours(region_mask, 0.5):
                            patch = plt.Polygon(np.c_[contour[:, 1], contour[:, 0]], color=to_hex(color_map[annotation_id]/255))
                            axes[-1].add_patch(patch)
                            polygon = Polygon(np.c_[contour[:, 1], contour[:, 0]])
                            poi = polylabel(polygon, tolerance=0.1)
                            x, y = poi.x, poi.y
                            # axes[-1].plot(x, y, 'o', color=to_hex(color_map[annotation_id]/255))
                            delta_xy = np.array([x - xc, y - yc])
                            distance_xy = np.linalg.norm(delta_xy)
                            axes[-1].annotate(
                                name_map[int(annotation_id)],
                                (x, y),
                                np.array([x, y]) + (1.1 * slice_radius - distance_xy) * delta_xy / distance_xy,
                                ha="left", va="bottom",
                                fontsize=5,
                                color=to_hex(color_map[annotation_id]/255),
                                arrowprops=dict(arrowstyle="-", color="#677e8c", linewidth=0.25),
                                wrap=True,
                            )

            # plot and label samples
            for _, row in df[df["slice_number"] == slice_number].iterrows():
                x = row["segmentation_col"]
                y = row["segmentation_row"]
                axes[-1].plot(x, y, linestyle="", marker='o', color=row["subclass_color"])
                delta_xy = np.array([x - xc, y - yc])
                distance_xy = np.linalg.norm(delta_xy)
                axes[-1].annotate(
                    row["barcode"],
                    (x, y),
                    np.array([x, y]) + (1.1 * slice_radius - distance_xy) * delta_xy / distance_xy,
                    ha="right", va="bottom",
                    fontsize=5,
                    arrowprops=dict(arrowstyle="-", color="#677e8c", linewidth=0.25),
                    wrap=True,
                )
            fig.savefig(output_directory / f"slice_{slice_number:03d}.svg")
            pdf.savefig(fig)
            plt.close(fig)

    # --------------------------------------------------------------------------------

    print("Plotting slices aligned in 3D...")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis("off")

    for ii in range(total_slices):
        print(f"{ii+1} / {total_slices}")
        slice_number = slice_numbers[ii]
        slice_image  = slice_images[ii]
        slice_mask   = slice_masks[ii]
        annotation   = annotations[ii]
        sample_mask  = sample_masks[ii]

        # plot slice contour
        contours = find_contours(slice_mask, 0.5)
        for contour in contours:
            x = contour[:, 1]
            y = contour[:, 0]
            z = -slice_number * np.ones_like(x)
            ax.plot(z, x, -y, "#677e8c", linewidth=1.0)

        # compute slice radius
        slice_radius = 0
        center = np.array([yc, xc])
        for contour in contours:
            delta = contour - center[np.newaxis, :]
            distance = np.linalg.norm(delta, axis=1)
            slice_radius = max(slice_radius, np.max(distance))

        # plot and label samples
        for _, row in df[df["slice_number"] == slice_number].iterrows():
            x = row["segmentation_col"]
            y = row["segmentation_row"]
            xyz = np.array([-slice_number, x, -y])
            ax.plot(*xyz, linestyle="", marker='o', color=row["subclass_color"])

    # label clones
    for barcode in np.unique(df["barcode"]):
        clone = df[df["barcode"] == barcode]
        if len(clone) > 1:
            Z = -clone["slice_number"]
            X = clone["segmentation_col"]
            Y = clone["segmentation_row"]
            zt = Z.mean()
            xt = xc + 2 * (X.mean() - xc)
            yt = yc + 2 * (Y.mean() - yc)
            ax.text(zt, xt, -yt, barcode + " ", ha="right")
            for zz, xx, yy in zip(Z, X, Y):
                ax.plot([zt, zz], [xt, xx], [-yt, -yy], linewidth=0.5, color="#677e8c")

    print("Select the desired view. The figure will be saved on closing.")
    plt.show()
    fig.savefig(output_directory / "reconstruction_in_3d.pdf")
    fig.savefig(output_directory / "reconstruction_in_3d.svg")
