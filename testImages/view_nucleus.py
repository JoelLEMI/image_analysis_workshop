"""
Open testImages/nucleus_cropped_1.tif, print a 10x10 corner of the pixel
matrix, report its size and number type, then show the image in napari.

Requirements:  pip install tifffile numpy "napari[all]"
Run from anywhere:  python path/to/view_nucleus.py
"""

from pathlib import Path

import numpy as np
import tifffile
import napari

IMAGE_NAME = "nucleus_cropped_1.tif"

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
