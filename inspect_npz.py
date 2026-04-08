"""Quick inspection of .npz files — prints array names, shapes, and dtypes."""

import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "pileup_waveforms.npz"
data = np.load(path)

print(f"File: {path}")
print(f"Arrays: {len(data.files)}\n")

for name in data.files:
    arr = data[name]
    print(f"  {name:20s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")

print()

# Show a few values from label/delay arrays
for name in ["primary_label", "secondary_label", "delays_samples", "clean_indices"]:
    if name in data.files:
        arr = data[name]
        print(f"  {name}: {arr[:10]} ...")
