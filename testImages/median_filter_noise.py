"""
Remove noise from testImages/Nucleus_cropped_noise.tif with a median filter,
using three different neighborhoods (kernels):

    plus3   3 x 3 kernel without the corners (a plus sign, 5 pixels)
    square3 3 x 3 kernel (9 pixels)
    square5 5 x 5 kernel (25 pixels)

For every pixel, the median filter takes the pixel values covered by the kernel,
sorts them, and keeps the middle one. Isolated noisy pixels are outvoted by
their neighbors, while edges stay fairly sharp.

The filtered images are saved as .tif files in testImages/median_filtered/
and shown next to the original in napari.

Run from the image_analysis_workshop folder:
    uv run python testImages/median_filter_noise.py
Or pass another image name:
    uv run python testImages/median_filter_noise.py nucleus_cropped_1
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
import napari
from scipy import ndimage

# ---- Which image -----------------------------------------------------------
IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else "Nucleus_cropped_noise.tif"
if not IMAGE_NAME.lower().endswith((".tif", ".tiff")):
    IMAGE_NAME += ".tif"

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

# ---- The three kernels (1 = pixel is used, 0 = pixel is ignored) -----------
KERNELS = {
    "plus3": np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool),
    "square3": np.ones((3, 3), dtype=bool),
    "square5": np.ones((5, 5), dtype=bool),
}

# ---- Load ------------------------------------------------------------------
image = tifffile.imread(IMAGE_PATH)
print(f"Image: {IMAGE_PATH.name}  shape {image.shape}  type {image.dtype}  "
      f"(min {image.min()}, max {image.max()})")

out_dir = IMAGE_PATH.parent / "median_filtered"
out_dir.mkdir(exist_ok=True)

# ---- Filter, save and report -----------------------------------------------
results = {}
for name, kernel in KERNELS.items():
    # footprint = the shape of the neighborhood; mode="reflect" handles edges
    filtered = ndimage.median_filter(image, footprint=kernel, mode="reflect")
    results[name] = filtered

    out_path = out_dir / f"{IMAGE_PATH.stem}_median_{name}.tif"
    tifffile.imwrite(out_path, filtered)

    print(f"\n{name}: {kernel.sum()} pixels in the kernel")
    print(kernel.astype(int))
    print(f"  saved to {out_path.relative_to(IMAGE_PATH.parent.parent)}")
    print(f"  min {filtered.min()}, max {filtered.max()}, "
          f"std {filtered.std():.1f} (original std {image.std():.1f})")

# ---- Show in napari ----------------------------------------------------------
# Use the same contrast for every layer so they can be compared fairly.
limits = (float(image.min()), float(image.max()))

viewer = napari.Viewer()
viewer.add_image(image, name="original", contrast_limits=limits)
for name, filtered in results.items():
    viewer.add_image(filtered, name=f"median {name}", contrast_limits=limits)
viewer.grid.enabled = True   # show all four side by side
napari.run()
