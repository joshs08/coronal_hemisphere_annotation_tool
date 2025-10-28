"""
Review .tif images and mark keep/discard.

Controls:
  - SPACE: keep=True
  - X:     keep=False
  - B:     go back one image (undo last decision)
  - Q/ESC: quit and save

Usage:
  python review_slices.py /path/to/images /path/to/save.json
"""

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def wait_for_key(fig):
    """Block until a key is pressed; return the key string (e.g., 'x', 'space', ' '), or None."""
    pressed = {"key": None}

    def on_key(event):
        pressed["key"] = event.key
        plt.close(event.canvas.figure)

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()  # blocks until window closed by on_key
    fig.canvas.mpl_disconnect(cid)
    return pressed["key"]


def main(folder: str):
    img_dir = Path(folder).expanduser().resolve()
    if not img_dir.is_dir():
        raise SystemExit(f"Not a directory: {img_dir}")

    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    files.sort(key=lambda p: natural_key(p.name))
    if not files:
        raise SystemExit(f"No .png files found in: {img_dir}")

    excluded: list[str] = []

    for p in files:
        # Load image
        img = mpimg.imread(str(p))

        # Display
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.set_title(p.name)
        ax.axis("off")

        # Wait for valid key
        while True:
            key = wait_for_key(fig)
            if key in (" ", "space"):  # SPACE -> keep
                break
            if key and key.lower() == "x":  # X -> exclude
                excluded.append(Path(p.name).stem)
                break
            # If closed/other key, re-show the same image
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.set_title(p.name)
            ax.axis("off")

    out_path = img_dir / "exclude.npy"
    np.save(str(out_path), np.array(excluded, dtype=object), allow_pickle=True)
    print(f"Saved {len(excluded)} excluded sections to: {out_path}")
    print("Load for DeepSlice:")
    print(f"  bad_sections = np.load(r\"{out_path}\", allow_pickle=True).tolist()")
    print("  Model.set_bad_sections(bad_sections=bad_sections)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])