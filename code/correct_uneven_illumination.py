import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy.optimize import curve_fit
from scipy.interpolate import griddata

def correct_uneven_illumination(image, border_width=50, show=False):

    # Get a smooth estimate of the variation in illumination along the borders of the image.
    left   = image[:, :border_width].mean(axis=1)
    right  = image[:, -border_width:].mean(axis=1)
    top    = image[:border_width, :].mean(axis=0)
    bottom = image[-border_width:, :].mean(axis=0)

    def linear(x, a, b):
        return a * x + b

    def exponential(x, a, b, c):
        return a * np.exp(-b * x) + c

    borders = [left, right, top, bottom]
    interpolated_borders = []
    for border in borders:
        x = np.arange(len(border))
        try: # exponential fit
            (a2, b2, c2), _ = curve_fit(exponential, x, border)
            interpolated_border = exponential(x, a2, b2, c2)
        except: # linear fit
            (a1, b1), _ = curve_fit(linear, x, border)
            interpolated_border = linear(x, a1, b1)
        interpolated_borders.append(interpolated_border)

    if show:
        fig, axes = plt.subplots(2, 2, sharey=True)
        axes = axes.ravel()
        labels = ["left", "right", "top", "bottom"]
        for border, interpolated_border, label, ax in zip(borders, interpolated_borders, labels, axes):
            ax.plot(border)
            ax.plot(interpolated_border, ls="--")
            ax.set_title(label)

    # Interpolate across the image to compute a correction at every pixel.
    left, right, top, bottom = interpolated_borders
    interpolated_frame = np.zeros_like(image)
    interpolated_frame[:, 0] = left
    interpolated_frame[:, -1] = right
    interpolated_frame[0, :] = top
    interpolated_frame[-1, :] = bottom
    rows, columns = np.where(interpolated_frame)
    values = interpolated_frame[rows, columns]
    image_correction = griddata(
        points = np.c_[rows, columns],
        values = values,
        xi = np.transpose(np.where(np.ones_like(image))),
    )
    image_correction = image_correction.reshape(image.shape)

    # Fill in NaNs if there are any in the interpolated image.
    if np.any(np.isnan(image_correction)):
        rows, columns = np.where(~np.isnan(image_correction))
        values = image_correction[rows, columns]
        image_correction = griddata(
            points = np.c_[rows, columns],
            values = values,
            xi = np.transpose(np.where(np.ones_like(image)))
        )
        image_correction = image_correction.reshape(image.shape)

    # Apply correction
    corrected_image = image - image_correction

    if show:
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        for img, label, ax in zip(
                [image, image_correction, corrected_image],
                ["Original", "Correction", "Corrected"],
                axes):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(label)

    return corrected_image


if __name__ == "__main__":

    image = imread("/home/paul/src/coronal_hemisphere_annotation_tool/test_images/uneven_illumination.tif")
    corrected = correct_uneven_illumination(image, 50, show=True)
    plt.show()
