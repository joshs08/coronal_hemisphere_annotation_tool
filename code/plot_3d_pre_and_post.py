from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import find_contours

pre_root  = Path(r"J:\CHAT\output_251030_new_midlines")
post_root = Path(r"J:\CHAT\QuickNII_processing")
out_root  = Path(r"J:\CHAT\figures")
out_root.mkdir(parents=True, exist_ok=True)

def load_and_pad_images(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))
    imgs = []
    shapes = []
    for f in files:
        img = imread(f)
        if img.ndim == 3:
            img = rgb2gray(img)
        imgs.append(img)
        shapes.append(img.shape)
    if not imgs:
        return []
    max_h = max(h for h, w in shapes)
    max_w = max(w for h, w in shapes)
    padded = []
    for img in imgs:
        h, w = img.shape
        dh, dw = max_h - h, max_w - w
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded.append(np.pad(img, ((top, bottom), (left, right)), mode="constant", constant_values=0))
    return padded

def plot_stack(ax, images, color):
    for ii, img in enumerate(images):
        mask = img > 0
        for c in find_contours(mask.astype(float), level=0.5):
            x = c[:, 1]
            y = c[:, 0]
            z = np.full_like(x, -ii, dtype=float)
            ax.plot(x, -y, z, color=color, linewidth=1.0)

def link_3d_views(fig, ax_a, ax_b, sync_limits=True):
    state = {"busy": False}

    def sync(event):
        if event.inaxes not in (ax_a, ax_b) or state["busy"]:
            return
        src = event.inaxes
        dst = ax_b if src is ax_a else ax_a
        state["busy"] = True
        # copy camera angles
        dst.view_init(elev=src.elev, azim=src.azim)
        # optionally copy zoom/pan too
        if sync_limits:
            dst.set_xlim3d(src.get_xlim3d())
            dst.set_ylim3d(src.get_ylim3d())
            dst.set_zlim3d(src.get_zlim3d())
        fig.canvas.draw_idle()
        state["busy"] = False

    # update during drag and once on release
    fig.canvas.mpl_connect("motion_notify_event", sync)
    fig.canvas.mpl_connect("button_release_event", sync)

# union of date folders from both roots
dates = sorted(
    {p.name for p in pre_root.iterdir() if p.is_dir()} |
    {p.name for p in post_root.iterdir() if p.is_dir()}
)

for d in dates:
    pre_dir  = pre_root  / d / "annotation"
    post_dir = post_root / d / "QN"

    pre_imgs  = load_and_pad_images(pre_dir,  "*.png")
    post_imgs = load_and_pad_images(post_dir, "*Rainbow_2017.png")

    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for ax in (ax1, ax2):
        ax.axis("off")
    link_3d_views(fig, ax1, ax2, sync_limits=False)
    ax2.view_init(ax1.elev, ax1.azim)
    plot_stack(ax1, pre_imgs,  color="#677e8c")
    plot_stack(ax2, post_imgs, color="#d9730d")

    ax1.set_title(f"{d} — PRE",  pad=10)
    ax2.set_title(f"{d} — POST", pad=10)
    plt.show()
    fig.savefig(out_root / f"{d}.svg", bbox_inches="tight")
    plt.close(fig)