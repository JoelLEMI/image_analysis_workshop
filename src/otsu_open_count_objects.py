"""Clean up an Otsu segmentation with erosion and dilation ("opening").

Run it from the top folder of the workshop repository with:

    uv run src/otsu_open_count_objects.py

The DAPI channel is thresholded with Otsu, then eroded twice and dilated twice.
The script counts the objects before and after, and shows a histogram of the
sizes of the remaining objects.  The histogram is saved in
testImages/segmentation/.  Close the histogram window to end the script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / "testImages" / "3Channel_FF_Corrected_1"
INPUT = FOLDER / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
OUTPUT = REPO / "testImages" / "segmentation" / "otsu_opened_object_sizes.png"

# The channels in this file are Cy3 (0), FITC (1) and DAPI (2).
DAPI_CHANNEL = 2

# How many rounds of erosion, followed by the same number of dilation rounds.
ROUNDS = 2

# The structuring element: a 3 x 3 plus sign.
ELEMENT = ndimage.generate_binary_structure(2, 1)

with tifffile.TiffFile(INPUT) as tif:
    dapi = tif.series[0].asarray()[DAPI_CHANNEL]

threshold = threshold_otsu(dapi)
mask = dapi > threshold


def object_sizes(binary_image):
    """Return the size in pixels of each connected object (8 neighbors)."""
    return np.array([region.area for region in regionprops(label(binary_image))])


before = object_sizes(mask)
print(f"Otsu threshold: {threshold}")
print(f"Before: {len(before)} objects, {mask.sum()} foreground pixels")

# Erosion removes everything too thin to hold the element; dilation grows what
# is left back.  What vanished during the erosion can not come back.
eroded = ndimage.binary_erosion(mask, structure=ELEMENT, iterations=ROUNDS)
opened = ndimage.binary_dilation(eroded, structure=ELEMENT, iterations=ROUNDS)

after = object_sizes(opened)
print(f"After {ROUNDS} erosions and {ROUNDS} dilations: {len(after)} objects, "
      f"{opened.sum()} foreground pixels")
print(f"Size in pixels: smallest {after.min():.0f}, median {np.median(after):.0f}, "
      f"largest {after.max():.0f}")

figure, axes = plt.subplots(figsize=(8, 4.5))
axes.hist(after, bins=25, color="#3b6fb6", edgecolor="white", linewidth=1)
axes.set_xlabel("Object size (pixels)")
axes.set_ylabel("Number of objects")
axes.set_title(f"{len(after)} objects remain after {ROUNDS} erosions and "
               f"{ROUNDS} dilations", loc="left")
axes.grid(axis="y", color="#e5e5e5", linewidth=0.8)
axes.set_axisbelow(True)
axes.spines[["top", "right"]].set_visible(False)
figure.tight_layout()

OUTPUT.parent.mkdir(exist_ok=True)
figure.savefig(OUTPUT, dpi=150)
print(f"Saved {OUTPUT.relative_to(REPO)}")
plt.show()
