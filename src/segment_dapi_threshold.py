"""Segment the nuclei in the DAPI channel with one fixed threshold.

Run it from the top folder of the workshop repository with:

    uv run src/segment_dapi_threshold.py

Every pixel above the threshold becomes 1 (nucleus), every other pixel 0
(background).  The result is saved in testImages/segmentation/ and shown in
napari on top of the DAPI image.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from skimage.measure import label, regionprops

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / "testImages" / "3Channel_FF_Corrected_1"
INPUT = FOLDER / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
OUTPUT = REPO / "testImages" / "segmentation" / "DAPI_threshold_mask.tif"

# Nuclei are above this value, background below it.  Try other values!
THRESHOLD = 1300

# The channels in this file are Cy3 (0), FITC (1) and DAPI (2).
DAPI_CHANNEL = 2

with tifffile.TiffFile(INPUT) as tif:
    dapi = tif.series[0].asarray()[DAPI_CHANNEL]
    # The file stores pixels per centimeter: convert to microns per pixel.
    page = tif.series[0].pages[0]
    numerator, denominator = page.tags["XResolution"].value
    pixel_size = 10_000 * denominator / numerator
print(f"DAPI channel: {dapi.shape[0]} x {dapi.shape[1]} pixels, "
      f"pixel size {pixel_size:.4f} microns")

# The segmentation itself: one comparison for every pixel.
mask = dapi > THRESHOLD
print(f"Threshold {THRESHOLD}: {100 * mask.mean():.2f}% of the pixels are foreground")

# Connected foreground pixels form one object (8 neighbors, the default).
objects = label(mask)
areas = np.array([region.area for region in regionprops(objects)])
print(f"Objects: {len(areas)}, median size {np.median(areas):.0f} pixels, "
      f"smallest {areas.min():.0f}, largest {areas.max():.0f}")

# Save as 8-bit (0 and 1), with the pixel size of the original image.
OUTPUT.parent.mkdir(exist_ok=True)
tifffile.imwrite(OUTPUT, mask.astype(np.uint8), imagej=True,
                 resolution=(1 / pixel_size, 1 / pixel_size),
                 metadata={"unit": "um"})
print(f"Saved {OUTPUT.relative_to(REPO)}")

viewer = napari.Viewer(title="DAPI threshold segmentation")
scale = (pixel_size, pixel_size)
viewer.add_image(dapi, name="DAPI", colormap="gray", scale=scale, units="um",
                 contrast_limits=(0, 6000))
viewer.add_labels(mask.astype(np.uint8), name=f"Threshold {THRESHOLD}",
                  scale=scale, units="um", opacity=0.4)
viewer.scale_bar.visible = True
napari.run()
