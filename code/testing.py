import numpy as np
"""
path = r"J:\CHAT\output_251030_new_midlines\2018-09-04\segmentation\exclude.npy"
ex = np.load(path, allow_pickle=True)
ex[4] = "brain_slice_s013"
np.save(str(path), np.array(ex, dtype=object), allow_pickle=True)
print(ex)
print("W")
"""
from pathlib import Path
import io, pandas as pd, numpy as np

def load_flat(path, dtype=None, shape=None):
    b = Path(path).read_bytes()
    # If told it's binary (or you pass a dtype), use NumPy
    if dtype is not None:
        a = np.frombuffer(b, dtype=dtype)
        return a.reshape(shape) if shape else a
    # Try tabular text
    try:
        return pd.read_csv(io.BytesIO(b), sep=None, engine="python")
    except Exception:
        # Fallback to raw text lines
        return b.decode("utf-8", errors="replace").splitlines()
path = r"J:\CHAT\VisuAlign_processing\2018-09-06\segmentation\brain_slice_s001_nl.flat"
data = load_flat(path)
print("W")