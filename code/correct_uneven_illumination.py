import warnings
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy.optimize import curve_fit
from scipy.interpolate import griddata, RegularGridInterpolator


def correct_uneven_illumination(image, border_width=50, interpolate_between="frame", show=False, fill_nans=True):

    # Extract border pixel intensities
    border_pixel_intensities = dict(
        left   = image[:, :border_width].mean(axis=1),
        right  = image[:, -border_width:].mean(axis=1),
        top    = image[:border_width, :].mean(axis=0),
        bottom = image[-border_width:, :].mean(axis=0)
    )

    if show:
        image_with_border = image.copy()
        image_max = image.max()
        image_with_border[:,  border_width] = 2 * image_max
        image_with_border[:, -border_width] = 2 * image_max
        image_with_border[ border_width, :] = 2 * image_max
        image_with_border[-border_width, :] = 2 * image_max
        fig, ax = plt.subplots()
        ax.imshow(np.log1p(image_with_border), cmap="gray")

    # Get a smooth estimate of the variation in illumination along the borders of the image.
    def linear(x, a, b):
        return a * x + b

    def exponential(x, a, b, c):
        return a * np.exp(-b * x) + c

    fitted_border_pixel_intensities = dict()
    for side, intensities in border_pixel_intensities.items():
        total_pixels = len(intensities)
        x = np.arange(total_pixels)
        try: # exponential fit; curve fitting fails sometimes for unknown reasons
            if np.sum(intensities[:total_pixels//2]) > np.sum(intensities[total_pixels//2:]):
                (a, b, c), _ = curve_fit(exponential, x, intensities)
                fitted_border_pixel_intensities[side] = exponential(x, a, b, c)
            else: # intensities are not decaying but increasing; invert order for fitting
                (a, b, c), _ = curve_fit(exponential, x, intensities[::-1])
                fitted_border_pixel_intensities[side] = exponential(x, a, b, c)[::-1]
        except: # fall back to linear fit
            warnings.warn(f"Failed to fit an exponential to {side} border. Falling back to a linear fit.")
            (a, b), _ = curve_fit(linear, x, intensities)
            fitted_border_pixel_intensities[side] = linear(x, a, b)

    if show:
        fig, axes = plt.subplots(2, 2, sharey=True)
        axes = axes.ravel()
        for side, ax in zip(border_pixel_intensities.keys(), axes):
            ax.plot(border_pixel_intensities[side])
            ax.plot(fitted_border_pixel_intensities[side], ls="--")
            ax.set_title(side.capitalize())
            ax.set_xlabel("Row/Column Index")
            ax.set_ylabel("Pixel intensity")
        fig.tight_layout()

    # Interpolate across the image to compute a correction at every pixel.
    image_to_interpolate = np.zeros_like(image)
    if interpolate_between == "frame":
        image_to_interpolate[:,  0] = fitted_border_pixel_intensities["left"]
        image_to_interpolate[:, -1] = fitted_border_pixel_intensities["right"]
        image_to_interpolate[ 0, :] = fitted_border_pixel_intensities["top"]
        image_to_interpolate[-1, :] = fitted_border_pixel_intensities["bottom"]
    elif interpolate_between == "top-bottom":
        image_to_interpolate[ 0, :] = fitted_border_pixel_intensities["top"]
        image_to_interpolate[-1, :] = fitted_border_pixel_intensities["bottom"]
    elif interpolate_between == "left-right":
        image_to_interpolate[ :, 0] = fitted_border_pixel_intensities["left"]
        image_to_interpolate[ :,-1] = fitted_border_pixel_intensities["right"]
    elif interpolate_between == "corners":
        image_to_interpolate[ 0,  0] = (fitted_border_pixel_intensities["left"][0]   + fitted_border_pixel_intensities["top"][0])     / 2
        image_to_interpolate[ 0, -1] = (fitted_border_pixel_intensities["right"][0]  + fitted_border_pixel_intensities["top"][-1])    / 2
        image_to_interpolate[-1,  0] = (fitted_border_pixel_intensities["left"][-1]  + fitted_border_pixel_intensities["bottom"][0])  / 2
        image_to_interpolate[-1, -1] = (fitted_border_pixel_intensities["right"][-1] + fitted_border_pixel_intensities["bottom"][-1]) / 2
    else:
        raise ValueError(f"Variable `interpolate_between` one of: 'frame', 'top-bottom', 'left-right', or 'corners', not {interpolate_between}!")

    points = np.transpose(np.where(image_to_interpolate))
    values = image_to_interpolate[points[:, 0], points[:, 1]]
    desired = np.transpose(np.where(np.ones_like(image)))
    image_correction = griddata(
        points = points,
        values = values,
        xi = desired,
        method="linear",
    )
    image_correction = image_correction.reshape(image.shape)

    # Fill in NaNs if there are any in the interpolated image.
    if np.any(np.isnan(image_correction)) & fill_nans:
        warnings.warn("Interpolation yielded NaNs. Re-running interpolation to fill these in.")
        rows, columns = np.where(~np.isnan(image_correction))
        image_correction = griddata(
            points = np.c_[rows, columns],
            values = image_correction[rows, columns],
            xi = np.transpose(np.where(np.ones_like(image))),
            method="nearest",
        )
        image_correction = image_correction.reshape(image.shape)

    # Apply correction
    corrected_image = image - image_correction

    if show:
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15,5))
        for img, label, ax in zip(
                [np.log1p(image), image_correction, np.log1p(corrected_image)],
                ["Original", "Correction", "Corrected"],
                axes):
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(label)

    return corrected_image


def linearly_correct_uneven_illumination(image, border_width=50, interpolate_between="frame", show=False, fill_nans=True):
    # Extract border pixel intensities
    border_pixel_intensities = dict(
        left   = image[:, :border_width].mean(axis=1),
        right  = image[:, -border_width:].mean(axis=1),
        top    = image[:border_width, :].mean(axis=0),
        bottom = image[-border_width:, :].mean(axis=0)
    )

    if show:
        image_with_border = image.copy()
        image_max = image.max()
        image_with_border[:,  border_width] = 2 * image_max
        image_with_border[:, -border_width] = 2 * image_max
        image_with_border[ border_width, :] = 2 * image_max
        image_with_border[-border_width, :] = 2 * image_max
        fig, ax = plt.subplots()
        ax.imshow(np.log1p(image_with_border), cmap="gray")

    # Get a smooth estimate of the variation in illumination along the borders of the image.
    def linear(x, a, b):
        return a * x + b

    def exponential(x, a, b, c):
        return a * np.exp(-b * x) + c

    fitted_border_pixel_intensities = dict()
    for side, intensities in border_pixel_intensities.items():
        total_pixels = len(intensities)
        x = np.arange(total_pixels)
        try: # exponential fit; curve fitting fails sometimes for unknown reasons
            if np.sum(intensities[:total_pixels//2]) > np.sum(intensities[total_pixels//2:]):
                (a, b, c), _ = curve_fit(exponential, x, intensities)
                fitted_border_pixel_intensities[side] = exponential(x, a, b, c)
            else: # intensities are not decaying but increasing; invert order for fitting
                (a, b, c), _ = curve_fit(exponential, x, intensities[::-1])
                fitted_border_pixel_intensities[side] = exponential(x, a, b, c)[::-1]
        except: # fall back to linear fit
            warnings.warn(f"Failed to fit an exponential to {side} border. Falling back to a linear fit.")
            (a, b), _ = curve_fit(linear, x, intensities)
            fitted_border_pixel_intensities[side] = linear(x, a, b)

    if show:
        fig, axes = plt.subplots(2, 2, sharey=True)
        axes = axes.ravel()
        for side, ax in zip(border_pixel_intensities.keys(), axes):
            ax.plot(border_pixel_intensities[side])
            ax.plot(fitted_border_pixel_intensities[side], ls="--")
            ax.set_title(side.capitalize())
            ax.set_xlabel("Row/Column Index")
            ax.set_ylabel("Pixel intensity")
        fig.tight_layout()

    total_rows, total_columns = img.shape
    rows, columns = np.mgrid[0:total_rows, 0:total_columns]
    horizontal_correction = fitted_border_pixel_intensities["left"][rows]      *      columns / total_columns \
                          + fitted_border_pixel_intensities["right"][rows]     * (1 - columns / total_columns)
    vertical_correction   = fitted_border_pixel_intensities["top"][columns]    *         rows / total_rows \
                          + fitted_border_pixel_intensities["bottom"][columns] * (1 -    rows / total_rows)
    image_correction = (horizontal_correction + vertical_correction) / 2

    # # or
    # distance_from_left_or_right_border = np.minimum(rows,    total_rows    - 1 - rows)
    # distance_from_top_or_bottom_border = np.minimum(columns, total_columns - 1 - columns)
    # # image_correction = np.where(distance_from_left_or_right_border < distance_from_top_or_bottom_border, vert_est, horizontal_correction)
    # horizontal_weight = distance_from_left_or_right_border / np.maximum(distance_from_left_or_right_border + distance_from_top_or_bottom_border, 1)
    # image_correction = horizontal_correction * horizontal_weight + vertical_correction * (1 - horizontal_weight)

    # Apply correction
    corrected_image = image - image_correction

    if show:
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15,5))
        for img, label, ax in zip(
                [np.log1p(image), image_correction, np.log1p(corrected_image)],
                ["Original", "Correction", "Corrected"],
                axes):
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(label)

    return corrected_image


if __name__ == "__main__":

    image = imread("/home/paul/src/coronal_hemisphere_annotation_tool/test_images/uneven_illumination.tif")
    interpolation_bases = [
        "frame",
        "top-bottom",
        "left-right",
        "corners",
    ]
    corrected_images = dict()
    for interpolate_between in interpolation_bases:
        corrected_images[interpolate_between] = correct_uneven_illumination(
            image, 50, show=False, interpolate_between=interpolate_between, fill_nans=True)

    corrections = {key : image - corrected for key, corrected in corrected_images.items()}
    stack = np.array(list(corrections.values()))
    vmin_corrections = stack.min()
    vmax_corrections = stack.max()

    differences = stack[1:] - stack[0, np.newaxis, np.newaxis]
    vmin_differences = differences.min()
    vmax_differences = differences.max()

    fig, axes = plt.subplots(len(interpolation_bases[1:]), 2, sharex=True, sharey=True, figsize=(10, len(interpolation_bases[1:]) * 5))
    for interpolate_between, (ax1, ax2) in zip(interpolation_bases[1:], axes):
        ax1.imshow(corrections[interpolate_between],                        cmap="gray", vmin=vmin_corrections, vmax=vmax_corrections)
        ax2.imshow(corrections[interpolate_between] - corrections["frame"], cmap="gray", vmin=vmin_differences, vmax=vmax_differences)
        ax1.set_ylabel(interpolate_between)
    axes[0, 0].set_title("Correction")
    axes[0, 1].set_title("Difference to interpolation based on full frame")
    fig.tight_layout()

    plt.show()
