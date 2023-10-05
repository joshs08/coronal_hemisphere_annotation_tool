#!/usr/bin/env python
"""
Prepare a series of mouse coronal hemisphere sections for registration.
1) Isolate a single hemisphere (remove background, other hemisphere) in each image.
2) Reflect hemispheres to create complete coronal sections.
3) Align sections with each other in one image stack.

Images in the provided folder have to be in TIF format and contain the slice number in the filename separated by underscores.
For example: Image_01.tif.

Sample coordinates are provided as a CSV spreadsheet.
These coordinates are then transformed such that locations in the input image
are mapped to the right locations in the output image.
The spreadsheet has to contain the following four columns:

  - sample_id    : an integer sample ID,
  - image_x      : the X coordinate (or column) in the input image,
  - image_y      : the Y coordinate (or row) in the input image, and
  - slice_number : the Z coordinate (or slice number).

This script creates the following files in the provided output directory:

  - an NPZ file containing the aligned image stacks and masks (open with numpy.load),
  - the segmented and aligned slices as individual PNG files (for registration), and
  - a spreadsheet with the transformed coordinates.

Example
-------
python code/hemisphere_segmentation_and_alignment.py test/ test/sample_data.csv test/segmentation/ --show

"""

import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from time import sleep
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from PIL import Image
from matplotlib.backend_bases import MouseButton
from skimage.transform import rotate
from skimage.filters import threshold_minimum as threshold
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_dilation,
    binary_opening,
    disk,
)
from skimage.measure import label, find_contours


def equalize(img):
    minimum = np.percentile(img[img>0], 2)
    maximum = np.percentile(img[img>0], 98)
    equalized = np.clip(img, minimum, maximum)
    equalized -= minimum
    equalized /= maximum - minimum
    equalized *= 255
    return equalized


def select_midline(img, with_refinement=True):
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle("Select the ventral and dorsal endpoints of the midline.")
    ax.imshow(img, cmap='gray')
    midline = np.array(ginput(fig, 2, blit=True))

    if with_refinement:
        pad = 500
        fig.suptitle("Adjust the ventral endpoint of the midline.")
        ax.plot(*midline[0], 'y+', markersize=10)
        ax.axis((midline[0, 0]-pad, midline[0, 0]+pad, midline[0, 1]+pad, midline[0, 1]-pad))
        out = np.squeeze(ginput(fig, 1, blit=True))
        if np.any(out):
            midline[0] = out

        fig.suptitle("Adjust the dorsal endpoint of the midline.")
        ax.plot(*midline[1], 'y+', markersize=10)
        ax.axis((midline[1, 0]-pad, midline[1, 0]+pad, midline[1, 1]+pad, midline[1, 1]-pad))
        out = np.squeeze(ginput(fig, 1, blit=True))
        if np.any(out):
            midline[1] = out

    plt.close(fig)

    # print(midline)

    return midline


def blocking_input_loop(figure, event_names, timeout, handler):
    """
    Run *figure*'s event loop while listening to interactive events.

    The events listed in *event_names* are passed to *handler*.

    This function is used to implement `.Figure.waitforbuttonpress`,
    `.Figure.ginput`, and `.Axes.clabel`.

    Parameters
    ----------
    figure : `~matplotlib.figure.Figure`
    event_names : list of str
        The names of the events passed to *handler*.
    timeout : float
        If positive, the event loop is stopped after *timeout* seconds.
    handler : Callable[[Event], Any]
        Function called for each event; it can force an early exit of the event
        loop by calling ``canvas.stop_event_loop()``.
    """
    if figure.canvas.manager:
        figure.show()  # Ensure that the figure is shown if we are managing it.
    # Connect the events to the on_event function call.
    cids = [figure.canvas.mpl_connect(name, handler) for name in event_names]
    try:
        figure.canvas.start_event_loop(timeout)  # Start event loop.
    finally:  # Run even on exception like ctrl-c.
        # Disconnect the callbacks.
        for cid in cids:
            figure.canvas.mpl_disconnect(cid)


def ginput(figure, n=1, timeout=-1, show_clicks=True,
           mouse_add=MouseButton.LEFT,
           mouse_pop=MouseButton.RIGHT,
           mouse_stop=MouseButton.MIDDLE,
           blit=False):
    """
    Blocking call to interact with a figure.

    Wait until the user clicks *n* times on the figure, and return the
    coordinates of each click in a list.

    There are three possible interactions:

    - Add a point.
    - Remove the most recently added point.
    - Stop the interaction and return the points added so far.

    The actions are assigned to mouse buttons via the arguments
    *mouse_add*, *mouse_pop* and *mouse_stop*.

    Parameters
    ----------
    n : int, default: 1
        Number of mouse clicks to accumulate. If negative, accumulate
        clicks until the input is terminated manually.
    timeout : float, default: 30 seconds
        Number of seconds to wait before timing out. If zero or negative
        will never time out.
    show_clicks : bool, default: True
        If True, show a red cross at the location of each click.
    mouse_add : `.MouseButton` or None, default: `.MouseButton.LEFT`
        Mouse button used to add points.
    mouse_pop : `.MouseButton` or None, default: `.MouseButton.RIGHT`
        Mouse button used to remove the most recently added point.
    mouse_stop : `.MouseButton` or None, default: `.MouseButton.MIDDLE`
        Mouse button used to stop input.
    blit : bool
        If blit is True, use blitting to accelerate re-drawing the figure after
        each input.

    Returns
    -------
    list of tuples
        A list of the clicked (x, y) coordinates.

    Notes
    -----
    The keyboard can also be used to select points in case your mouse
    does not have one or more of the buttons.  The delete and backspace
    keys act like right-clicking (i.e., remove last point), the enter key
    terminates input and any other key (not already used by the window
    manager) selects a point.
    """
    clicks = []
    marks = []

    if blit:
        plt.show(block=False)
        plt.pause(0.1)
        background = figure.canvas.copy_from_bbox(figure.bbox)
        figure.canvas.blit(figure.bbox)

    def handler(event):
        is_button = event.name == "button_press_event"
        is_key = event.name == "key_press_event"
        # Quit (even if not in infinite mode; this is consistent with
        # MATLAB and sometimes quite useful, but will require the user to
        # test how many points were actually returned before using data).
        if (is_button and event.button == mouse_stop
                or is_key and event.key in ["escape", "enter"]):
            figure.canvas.stop_event_loop()
        # Pop last click.
        elif (is_button and event.button == mouse_pop
              or is_key and event.key in ["backspace", "delete"]):
            if clicks:
                clicks.pop()
                if show_clicks:
                    marks.pop().remove()
                    if blit:
                        figure.canvas.restore_region(background)
                        for line in marks:
                            event.inaxes.draw_artist(line)
                        figure.canvas.blit(figure.bbox)
                        figure.canvas.flush_events()
                    else:
                        figure.canvas.draw()
        # Add new click.
        elif (is_button and event.button == mouse_add
              # On macOS/gtk, some keys return None.
              or is_key and event.key is not None):
            if event.inaxes:
                clicks.append((event.xdata, event.ydata))
                # _log.info("input %i: %f, %f",
                #           len(clicks), event.xdata, event.ydata)
                if show_clicks:
                    line = mpl.lines.Line2D([event.xdata], [event.ydata],
                                            marker="+", color="r", animated=blit)
                    event.inaxes.add_line(line)
                    marks.append(line)

                    if blit:
                        figure.canvas.restore_region(background)
                        for line in marks:
                            event.inaxes.draw_artist(line)
                        figure.canvas.blit(figure.bbox)
                        figure.canvas.flush_events()
                    else:
                        figure.canvas.draw()

        if len(clicks) == n and n > 0:
            sleep(0.5)
            figure.canvas.stop_event_loop()

    blocking_input_loop(
        figure, ["button_press_event", "key_press_event"], timeout, handler)

    # Cleanup.
    for mark in marks:
        mark.remove()

    if blit:
        figure.canvas.restore_region(background)
        figure.canvas.blit(figure.bbox)
        figure.canvas.flush_events()
    else:
        figure.canvas.draw()

    return clicks


def get_angle(dx, dy, radians=False):
    """Angle of a vector in 2D."""
    angle = np.arctan2(dy, dx)
    if radians:
        return angle
    else:
        return angle * 360 / (2.0 * np.pi)


def segment(image, show=False):
    mask = get_mask(image, show=show)
    segmented = image.copy()
    segmented[~mask] *= 0
    return segmented, mask


def get_mask(rotated, show=False):
    # binarize
    smoothed = anisodiff(rotated, niter=30, gamma=0.25, option=1)
    binary = smoothed > threshold(smoothed)

    # clean up small imperfections
    cleaned = remove_small_objects(binary, 1000)
    cleaned = remove_small_holes(cleaned, 1000)
    try:
        cleaned = binary_opening(cleaned, disk(25, decomposition="crosses"))
    except TypeError: # old scikit-image version
        cleaned = binary_opening(cleaned, disk(25))

    # compute outline
    labelled = label(cleaned)
    largest = labelled == np.argmax(np.bincount(labelled.flat, weights=cleaned.flat))

    if show:
        # display pipeline results
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
        axes = axes.ravel()
        axes[0].imshow(rotated, cmap="gray")
        axes[1].imshow(smoothed, cmap="gray")
        axes[2].imshow(binary,  cmap="gray")
        axes[3].imshow(cleaned, cmap="gray")
        axes[4].imshow(largest, cmap="gray")

        weighted = rotated.copy()
        weighted[~largest] *= 0.33
        axes[5].imshow(weighted, cmap="gray")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()

    return largest


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1.,1.), option=1, show=False):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            show - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")
        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

        if show:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            plt.pause(0.1)

    return imgout


def trim_excess(img, mask, pad=100):
    rows, columns = np.where(mask)
    row_min = max(np.min(rows) - pad, 0)
    row_max = min(np.max(rows) + pad, img.shape[0])
    column_min = max(np.min(columns) - pad, 0)
    column_max = min(np.max(columns) + pad, img.shape[1])
    mask = mask[row_min:row_max, column_min:column_max]
    img = img[row_min:row_max, column_min:column_max]
    return img, mask


def reflect(img):
    height, width = img.shape
    mirrored_image = np.zeros((height, 2 * width -1))
    mirrored_image[:, :width] = img
    mirrored_image[:, width:] = img[:, -2::-1]
    return mirrored_image


def rotate_points(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def segment_coronal_slice_hemisphere(img, midline, sample_ids=None, sample_coordinates=None, show=False):
    # improve contrast
    equalized = equalize(np.array(img))

    # adjust rotation
    delta = midline[1] - midline[0]
    angle = get_angle(delta[0], -delta[1]) # y-axis inverted for images
    midpoint = midline[0] + 0.5 * delta
    rotated = rotate(equalized, 90-angle, resize=False, center=midpoint, preserve_range=True)

    # remove everything right of the midline
    single_hemisphere = rotated[:, :int(np.ceil(midpoint[0]))].copy()

    # isolate slice from background
    segmented, mask = segment(single_hemisphere, show=show)

    # crop
    trimmed_segmented, trimmed_mask = trim_excess(segmented, mask, pad=100)

    # re-create full slice
    mirrored_image, mirrored_mask = reflect(trimmed_segmented), reflect(trimmed_mask)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes[0].imshow(mirrored_image, cmap="gray")
        axes[0].axis("off")
        for contour in find_contours(mirrored_mask, 0.5):
            axes[1].plot(contour[:, 1], contour[:, 0], "#677e8c")
        xmax, xmin, ymin, ymax = axes[0].axis() # due to imshow
        axes[1].axis((xmin, xmax, ymin, ymax))
        axes[1].set_aspect("equal")
        axes[1].axis("off")

    if sample_ids is None:
        return mirrored_image, mirrored_mask
    else:
        cols, rows = sample_coordinates.transpose()
        sample_mask = np.full_like(equalized, np.nan)
        sample_mask[np.round(rows).astype(int), np.round(cols).astype(int)] = sample_ids

        # Unfortunately, image rotation doesn't work for our purpose as the rotation "smears" pixel intensities and thus destroys the sample identity.
        # rotated_sample_mask = rotate(sample_mask, 90-angle, resize=False, center=midpoint, preserve_range=True)
        rotated_rows, rotated_cols = rotate_points(np.c_[rows, cols], origin=midpoint, degrees=90-angle).T
        rotated_sample_mask = np.full_like(rotated, np.nan)
        rotated_sample_mask[np.round(rotated_rows).astype(int), np.round(rotated_cols).astype(int)] = sample_ids

        cropped_sample_mask = rotated_sample_mask[:, :int(np.ceil(midpoint[0]))].copy()
        trimmed_sample_mask, _ = trim_excess(cropped_sample_mask, mask, pad=100)
        mirrored_sample_mask = reflect(trimmed_sample_mask)
        mirrored_sample_mask[:, trimmed_sample_mask.shape[1]:] = np.nan

        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            try:
                ax1.imshow(equalized + 200 * binary_dilation(~np.isnan(sample_mask), disk(10, decomposition="crosses")), cmap="gray")
                ax2.imshow(trimmed_segmented + 200 * binary_dilation(~np.isnan(trimmed_sample_mask), disk(10, decomposition="crosses")), cmap="gray")
            except TypeError:
                ax1.imshow(equalized + 200 * binary_dilation(~np.isnan(sample_mask), disk(10)), cmap="gray")
                ax2.imshow(trimmed_segmented + 200 * binary_dilation(~np.isnan(trimmed_sample_mask), disk(10)), cmap="gray")

        return mirrored_image, mirrored_mask, mirrored_sample_mask


def convert_to_image_array(images, fill_value=0):
    """Create a 3D image array out of a list of differently sized images."""
    shapes = [img.shape for img in images]
    max_rows, max_columns = np.max(shapes, axis=0)
    # arr = np.zeros((len(images), max_rows, max_columns))
    arr = np.full((len(images), max_rows, max_columns), fill_value, dtype=images[0].dtype)

    for ii, (image, (rows, columns)) in enumerate(zip(images, shapes)):
        start_row = int(np.ceil((max_rows - rows) / 2))
        start_column = int(np.ceil((max_columns - columns) / 2))
        arr[ii, start_row:start_row+rows, start_column:start_column+columns] = image

    return arr


if __name__ == "__main__":

    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("image_directory",             help="/path/to/image/directory/",  type=str)
    parser.add_argument("sample_data",                 help="/path/to/sample_data.csv",   type=str)
    parser.add_argument("output_directory",            help="/path/to/output/directory/", type=str)
    parser.add_argument("--midlines",                  help="/path/to/midlines.npy",      type=str, default=None)
    parser.add_argument("--show", action="store_true", help="Display figures.")
    args = parser.parse_args()

    # create output directory
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    # glob image file path and order by slice number
    filepaths = [Path(path) for path in glob.glob(args.image_directory + "*.tif")]
    slice_numbers = [next(int(substring) for substring in path.stem.split("_") if substring.isdigit()) for path in filepaths]
    order = np.argsort(slice_numbers)
    slice_numbers = [slice_numbers[ii] for ii in order]
    filepaths = [filepaths[ii] for ii in order]

    # load coordinates
    sample_data = pd.read_csv(args.sample_data, index_col="sample_id")

    if args.midlines is None:
        print("Selecting midlines...")
        midlines = []
        for (slice_number, filepath) in zip(slice_numbers, filepaths):
            print(slice_number, filepath.stem)
            image = Image.open(filepath)
            midline = select_midline(equalize(np.array(image)))
            midlines.append(midline)
        np.save(output_directory / "midlines.npy", midlines)
    else:
        midlines = np.load(args.midlines)

    print("Segmenting images...")
    slice_images, slice_masks, sample_masks = [], [], []
    for (slice_number, filepath, midline) in zip(slice_numbers, filepaths, midlines):
        print(slice_number, filepath.stem)
        image = Image.open(filepath)
        is_in_slice = sample_data["slice_number"] == slice_number
        if np.any(is_in_slice):
            slice_image, slice_mask, sample_mask = segment_coronal_slice_hemisphere(
                image, midline,
                sample_ids=sample_data[is_in_slice].index.values,
                sample_coordinates=sample_data.loc[is_in_slice, ["image_col", "image_row"]].values,
                show=args.show
            )
            slice_images.append(slice_image)
            slice_masks.append(slice_mask)
            sample_masks.append(sample_mask)
        else:
            slice_image, slice_mask = segment_coronal_slice_hemisphere(
                image, midline,
                show=args.show
            )
            slice_images.append(slice_image)
            slice_masks.append(slice_mask)
            sample_masks.append(np.full_like(slice_mask, np.nan))

    print("Aligning images...")
    slice_image_stack = convert_to_image_array(slice_images)
    slice_mask_stack  = convert_to_image_array(slice_masks)
    sample_mask_stack = convert_to_image_array(sample_masks, np.nan)

    print("Exporting images...")
    # save image stacks
    np.savez(output_directory / "segmentation_results.npz",
             slice_images  = slice_image_stack,
             slice_masks   = slice_mask_stack,
             sample_masks  = sample_mask_stack,
             slice_numbers = slice_numbers,
    )

    # export slice images as PNGs for DeepSlice / QuickNII
    for slice_number, img in zip(slice_numbers, slice_image_stack):
        Image.fromarray(img).convert("L").save(output_directory / f"brain_slice_s{slice_number:03d}.png")

    print("Exporting sample coordinates...")
    # save out sample masks as pixel coordinates
    slice_numbers, rows, columns = np.where(~np.isnan(sample_mask_stack))
    sample_ids = sample_mask_stack[slice_numbers, rows, columns].astype(int)

    sample_data["segmentation_row"] = np.full(len(sample_data), np.nan)
    sample_data["segmentation_col"] = np.full(len(sample_data), np.nan)
    for sample_id, row, col in zip(sample_ids, rows, columns):
        sample_data.at[sample_id, "segmentation_row"] = row
        sample_data.at[sample_id, "segmentation_col"] = col
    sample_data.reset_index().to_csv(args.sample_data, index=False)
