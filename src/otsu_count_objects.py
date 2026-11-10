"""Threshold the DAPI channel with the Otsu method and count the objects.

Run it from the top folder of the workshop repository with:

    uv run src/otsu_count_objects.py

The script counts all objects, whatever their size, and shows a histogram of
their sizes.  The histogram is saved in testImages/segmentation/.
Close the histogram window to end the script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / "testImages" / "3Channel_FF_Corrected_1"
INPUT = FOLDER / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
OUTPUT = REPO / "testImages" / "segmentation" / "otsu_object_sizes.png"

# The channels in this file are Cy3 (0), FITC (1) and DAPI (2).
DAPI_CHANNEL = 2

with tifffile.TiffFile(INPUT) as tif:
    dapi = tif.series[0].asarray()[DAPI_CHANNEL]

# Otsu tries every threshold, and picks the one that makes the two groups
# (below and above the threshold) each as uniform in brightness as possible.
threshold = threshold_otsu(dapi)
mask = dapi > threshold
print(f"Otsu threshold: {threshold}")
print(f"Foreground: {100 * mask.mean():.2f}% of the pixels")

# Connected foreground pixels form one object (8 neighbors, the default).
objects = label(mask)
sizes = np.array([region.area for region in regionprops(objects)])
print(f"Total number of objects: {len(sizes)}")
print(f"Size in pixels: smallest {sizes.min():.0f}, median {np.median(sizes):.0f}, "
      f"largest {sizes.max():.0f}")

# Most objects are either tiny or nucleus-sized, so the bins get wider with
# size: on a logarithmic axis, both groups are visible.  The edges fall between
# whole numbers, because a size is always a whole number of pixels.
edges = np.unique(np.round(np.geomspace(1, sizes.max() + 1, 31))) - 0.5

figure, axes = plt.subplots(figsize=(8, 4.5))
axes.hist(sizes, bins=edges, color="#3b6fb6", edgecolor="white", linewidth=1)
axes.set_xscale("log")
axes.xaxis.set_major_formatter(ScalarFormatter())  # 1000 instead of 10^3
axes.set_xlabel("Object size (pixels, logarithmic axis)")
axes.set_ylabel("Number of objects")
axes.set_title(f"{len(sizes)} objects after Otsu thresholding (threshold {threshold})",
               loc="left")
axes.grid(axis="y", color="#e5e5e5", linewidth=0.8)
axes.set_axisbelow(True)
axes.spines[["top", "right"]].set_visible(False)
figure.tight_layout()

OUTPUT.parent.mkdir(exist_ok=True)
figure.savefig(OUTPUT, dpi=150)
print(f"Saved {OUTPUT.relative_to(REPO)}")
plt.show()
