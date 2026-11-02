"""
Open testImages/nucleus_cropped_1.tif, print 10 rows x 10 columns of the
pixel matrix while skipping 2 rows and 2 columns each time (every 3rd one),
report the number type, then show the image in napari.

Run from the image_analysis_workshop folder:
    uv run python testImages/view_nucleus_1.py
"""

from pathlib import Path

import numpy as np
import tifffile
import napari

IMAGE_NAME = "nucleus_cropped_1.tif"
N_SHOW = 10   # how many rows and columns to print
SKIP = 2      # how many rows/columns to skip between the ones we keep
STEP = SKIP + 1  # keep one, skip two -> every 3rd row/column

# Find the image relative to where this script lives, so it runs from anywhere.
script_dir = Path(__file__).resolve().parent
candidates = [
    script_dir / "testImages" / IMAGE_NAME,
    script_dir / IMAGE_NAME,
    script_dir.parent / "testImages" / IMAGE_NAME,
]
IMAGE_PATH = next((p for p in candidates if p.exists()), None)
if IMAGE_PATH is None:
    raise FileNotFoundError(
        f"Could not find {IMAGE_NAME}. Looked in:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )

# 1. Open the file -> a NumPy array of pixel values
image = tifffile.imread(IMAGE_PATH)

# 2. Take every STEP-th row and column, and show the first N_SHOW of each.
#    image[start:stop:step] -> rows 0, 3, 6, ... 27 (and the same for columns)
stop = N_SHOW * STEP
subset = image[:stop:STEP, :stop:STEP]

kept = list(range(0, min(stop, image.shape[0]), STEP))
print(f"Rows and columns kept (skipping {SKIP} each time): {kept}")
with np.printoptions(linewidth=200):
    print(subset)
print(f"Subset shape: {subset.shape}")

# 3. Number type
print(f"Number type: {image.dtype}")

# 4. Display in napari
viewer = napari.Viewer()
viewer.add_image(image, name=IMAGE_PATH.stem)
napari.run()
