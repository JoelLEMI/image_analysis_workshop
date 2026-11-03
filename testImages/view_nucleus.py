"""
Open an image from testImages/, print a 10x10 corner of the pixel matrix,
report its size and number type, then show the image in napari.

Usage (from the image_analysis_workshop folder):
    uv run python testImages/view_nucleus.py                     # nucleus_cropped_1.tif
    uv run python testImages/view_nucleus.py nucleus_cropped_2   # any image name
    uv run python testImages/view_nucleus.py nucleus_cropped_3.tif
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
import napari

# The image name comes from the command line; default to nucleus_cropped_1.
IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else "nucleus_cropped_1.tif"
if not IMAGE_NAME.lower().endswith((".tif", ".tiff")):
    IMAGE_NAME += ".tif"   # allow typing the name without ".tif"

# Look for the image relative to where THIS script lives (not where the
# terminal happens to be), so it works whether the script sits in the
# project folder or inside testImages/.
script_dir = Path(__file__).resolve().parent
candidates = [
    script_dir / "testImages" / IMAGE_NAME,  # script in project folder
    script_dir / IMAGE_NAME,                 # script inside testImages/
    script_dir.parent / "testImages" / IMAGE_NAME,  # script in a subfolder
]
IMAGE_PATH = next((p for p in candidates if p.exists()), None)
if IMAGE_PATH is None:
    raise FileNotFoundError(
        f"Could not find {IMAGE_NAME}. Looked in:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )

# 1. Open the file -> a NumPy array of pixel values
image = tifffile.imread(IMAGE_PATH)

# 2. Show the first 10 rows and 10 columns as a matrix
print(f"First 10 rows x 10 columns of {IMAGE_PATH}:")
corner = image[:10, :10] if image.ndim == 2 else image[0, :10, :10]
if image.ndim > 2:
    print(f"(image has {image.ndim} dimensions; showing the first plane)")
with np.printoptions(linewidth=200):
    print(corner)

# 3. Size and number type
print()
print(f"Number of rows:    {image.shape[-2]}")
print(f"Number of columns: {image.shape[-1]}")
print(f"Full shape:        {image.shape}")
print(f"Number type:       {image.dtype}  "
      f"(min {image.min()}, max {image.max()})")

# 4. Display in napari
viewer = napari.Viewer()
viewer.add_image(image, name=IMAGE_PATH.stem)
napari.run()
