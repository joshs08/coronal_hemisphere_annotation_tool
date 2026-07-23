"""
Plot 3D stacks before and after VisuAlign nonlinear alignment.

By default this compares QuickNII rainbow label images with VisuAlign
``*_nl_rbw.png`` images for matching date folders.

Example
-------
python code/plot_3d_pre_and_post.py \
    --dates 2019-03-22 \
    --pre-root J:/CHAT/QuickNII_processing \
    --post-root J:/CHAT/VisuAlign_processing \
    --output-root J:/CHAT/VisuAlign_processing/2019-03-22/figures
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.measure import find_contours


def load_and_pad_images(folder, pattern):
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    images = []
    shapes = []
    for file in files:
        image = imread(file)
        if image.ndim == 3:
            image = rgb2gray(image)
        images.append(image)
        shapes.append(image.shape)

    if not images:
        return []

    max_rows = max(rows for rows, _ in shapes)
    max_cols = max(cols for _, cols in shapes)
    padded = []
    for image in images:
        rows, cols = image.shape
        row_delta = max_rows - rows
        col_delta = max_cols - cols
        top = row_delta // 2
        bottom = row_delta - top
        left = col_delta // 2
        right = col_delta - left
        padded.append(np.pad(image, ((top, bottom), (left, right)), mode="constant"))
    return padded


def plot_stack(ax, images, color):
    for index, image in enumerate(images):
        mask = image > 0
        for contour in find_contours(mask.astype(float), level=0.5):
            x = contour[:, 1]
            y = contour[:, 0]
            z = np.full_like(x, -index, dtype=float)
            ax.plot(x, -y, z, color=color, linewidth=1.0)


def link_3d_views(fig, ax_a, ax_b, sync_limits=True):
    state = {"busy": False}

    def sync(event):
        if event.inaxes not in (ax_a, ax_b) or state["busy"]:
            return
        source = event.inaxes
        target = ax_b if source is ax_a else ax_a
        state["busy"] = True
        target.view_init(elev=source.elev, azim=source.azim)
        if sync_limits:
            target.set_xlim3d(source.get_xlim3d())
            target.set_ylim3d(source.get_ylim3d())
            target.set_zlim3d(source.get_zlim3d())
        fig.canvas.draw_idle()
        state["busy"] = False

    fig.canvas.mpl_connect("motion_notify_event", sync)
    fig.canvas.mpl_connect("button_release_event", sync)


def discover_dates(pre_root, post_root):
    pre_dates = {path.name for path in Path(pre_root).iterdir() if path.is_dir()}
    post_dates = {path.name for path in Path(post_root).iterdir() if path.is_dir()}
    return sorted(pre_dates & post_dates)


def plot_date(date, args):
    pre_dir = Path(args.pre_root) / date / args.pre_subdir
    post_dir = Path(args.post_root) / date / args.post_subdir

    pre_images = load_and_pad_images(pre_dir, args.pre_pattern)
    post_images = load_and_pad_images(post_dir, args.post_pattern)
    if not pre_images:
        print(f"Skipping {date}: no pre images found in {pre_dir}")
        return
    if not post_images:
        print(f"Skipping {date}: no post images found in {post_dir}")
        return

    fig = plt.figure(figsize=(14, 10))
    ax_pre = fig.add_subplot(1, 2, 1, projection="3d")
    ax_post = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (ax_pre, ax_post):
        ax.axis("off")

    plot_stack(ax_pre, pre_images, color=args.pre_color)
    plot_stack(ax_post, post_images, color=args.post_color)
    ax_pre.set_title(f"{date} - {args.pre_label}", pad=10)
    ax_post.set_title(f"{date} - {args.post_label}", pad=10)
    ax_post.view_init(ax_pre.elev, ax_pre.azim)
    link_3d_views(fig, ax_pre, ax_post, sync_limits=False)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / f"{date}_pre_post_visualign_alignment"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")

    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved: {base.with_suffix('.svg')}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--pre-root", type=str, default="J:/CHAT/QuickNII_processing")
    parser.add_argument("--post-root", type=str, default="J:/CHAT/VisuAlign_processing")
    parser.add_argument("--output-root", type=str, default="J:/CHAT/figures/visualign_alignment")
    parser.add_argument("--pre-subdir", type=str, default="QN")
    parser.add_argument("--post-subdir", type=str, default="segmentation")
    parser.add_argument("--pre-pattern", type=str, default="*Rainbow_2017.png")
    parser.add_argument("--post-pattern", type=str, default="*_nl_rbw.png")
    parser.add_argument("--pre-label", type=str, default="QuickNII")
    parser.add_argument("--post-label", type=str, default="VisuAlign")
    parser.add_argument("--pre-color", type=str, default="#677e8c")
    parser.add_argument("--post-color", type=str, default="#d9730d")
    parser.add_argument("--dates", type=str, nargs="*", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    dates = args.dates if args.dates else discover_dates(args.pre_root, args.post_root)
    for date in dates:
        plot_date(date, args)
