import time
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy.optimize import curve_fit
from scipy.interpolate import griddata


def correct_uneven_illumination(image, border_width=50, show=False, fill_nans=True):

    # Extract border pixel intensities
    border_pixel_intensities = dict(
        Left   = image[:, :border_width].mean(axis=1),
        Right  = image[:, -border_width:].mean(axis=1),
        Top    = image[:border_width, :].mean(axis=0),
        Bottom = image[-border_width:, :].mean(axis=0)
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
        x = np.arange(len(intensities))
        try: # exponential fit; curve fitting fails sometimes for unknown reasons
            (a2, b2, c2), _ = curve_fit(exponential, x, intensities)
            fitted_border_pixel_intensities[side] = exponential(x, a2, b2, c2)
        except: # fall back to linear fit
            (a1, b1), _ = curve_fit(linear, x, intensities)
            fitted_border_pixel_intensities[side] = linear(x, a1, b1)

    if show:
        fig, axes = plt.subplots(2, 2, sharey=True)
        axes = axes.ravel()
        for side, ax in zip(border_pixel_intensities.keys(), axes):
            ax.plot(border_pixel_intensities[side])
            ax.plot(fitted_border_pixel_intensities[side], ls="--")
            ax.set_title(side)
            ax.set_xlabel("Row/Column Index")
            ax.set_ylabel("Pixel intensity")
        fig.tight_layout()

    # Interpolate across the image to compute a correction at every pixel.
    fitted_frame = np.zeros_like(image)
    fitted_frame[:,  0] = fitted_border_pixel_intensities["Left"]
    fitted_frame[:, -1] = fitted_border_pixel_intensities["Right"]
    fitted_frame[ 0, :] = fitted_border_pixel_intensities["Top"]
    fitted_frame[-1, :] = fitted_border_pixel_intensities["Bottom"]
    rows, columns = np.where(fitted_frame)
    values = fitted_frame[rows, columns]
    image_correction = griddata(
        points = np.c_[rows, columns],
        values = values,
        xi = np.transpose(np.where(np.ones_like(image))),
        method="linear",
    )
    image_correction = image_correction.reshape(image.shape)

    # Fill in NaNs if there are any in the interpolated image.
    if np.any(np.isnan(image_correction)) & fill_nans:
        rows, columns = np.where(~np.isnan(image_correction))
        values = image_correction[rows, columns]
        image_correction = griddata(
            points = np.c_[rows, columns],
            values = values,
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


if __name__ == "__main__":

    image = imread("/home/paul/src/coronal_hemisphere_annotation_tool/test_images/uneven_illumination.tif")
    tic  = time.time()
    corrected = correct_uneven_illumination(image, 50, show=True, fill_nans=True)
    toc = time.time()
    print(f"Time elapsed: {toc - tic:.2f} seconds")
    plt.show()
