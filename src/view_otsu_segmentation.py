"""Show the cleaned-up Otsu segmentation on top of the DAPI channel in napari.

Run it from the top folder of the workshop repository with:

    uv run src/view_otsu_segmentation.py

The DAPI channel is thresholded with Otsu, then eroded twice and dilated twice.
Each object gets its own color, so nuclei that were merged share one color.
"""

from pathlib import Path

import napari
import tifffile
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label

REPO = Path(__file__).resolve().parent.parent
FOLDER = REPO / "testImages" / "3Channel_FF_Corrected_1"
INPUT = FOLDER / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"

# The channels in this file are Cy3 (0), FITC (1) and DAPI (2).
DAPI_CHANNEL = 2

# How many rounds of erosion, followed by the same number of dilation rounds.
ROUNDS = 2

# The structuring element: a 3 x 3 plus sign.
ELEMENT = ndimage.generate_binary_structure(2, 1)

with tifffile.TiffFile(INPUT) as tif:
    dapi = tif.series[0].asarray()[DAPI_CHANNEL]
    # The file stores pixels per centimeter: convert to microns per pixel.
    numerator, denominator = tif.series[0].pages[0].tags["XResolution"].value
    pixel_size = 10_000 * denominator / numerator

mask = dapi > threshold_otsu(dapi)
eroded = ndimage.binary_erosion(mask, structure=ELEMENT, iterations=ROUNDS)
opened = ndimage.binary_dilation(eroded, structure=ELEMENT, iterations=ROUNDS)

# Give every connected object its own number (8 neighbors, the default).
objects = label(opened)
print(f"Showing {objects.max()} objects")

viewer = napari.Viewer(title="Otsu segmentation on DAPI")
scale = (pixel_size, pixel_size)
viewer.add_image(dapi, name="DAPI", colormap="gray", scale=scale, units="um",
                 contrast_limits=(0, 6000))
outlines = viewer.add_labels(objects, name="Segmentation", scale=scale, units="um")
# Draw only the outline of each object, so the DAPI signal stays visible.
# Set "contour" to 0 in the layer controls to fill the objects instead.
outlines.contour = 3
viewer.scale_bar.visible = True
napari.run()
