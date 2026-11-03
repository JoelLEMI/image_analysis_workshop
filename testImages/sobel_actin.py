"""
Sobel edge filters on testImages/Actin_cropped.tif.

    Sobel x (left -> right)       Sobel y (top -> bottom)
      -1   0   1                    -1  -2  -1
      -2   0   2                     0   0   0
      -1   0   1                     1   2   1

Sobel x = (right side) - (left side): large where brightness changes from
          left to right, i.e. at the edges of VERTICAL structures.
Sobel y = (bottom) - (top): large where brightness changes from top to
          bottom, i.e. at the edges of HORIZONTAL structures.

Gradient magnitude = sqrt(Sobel_x^2 + Sobel_y^2): how steep the brightness
changes, in any direction.

The kernels are applied exactly as written (ndimage.correlate). Note:
ndimage.convolve would flip the kernel and swap the signs (+/-).

The three images are saved in testImages/sobel/.

Run from the image_analysis_workshop folder:
    uv run python testImages/sobel_actin.py
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
import napari
from scipy import ndimage

# ---- Which image -----------------------------------------------------------
IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else "Actin_cropped.tif"
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

# ---- Kernels ------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)

# ---- Load and filter (in float: Sobel results are often negative) -----------
image = tifffile.imread(IMAGE_PATH)
img = image.astype(np.float32)

sobel_x = ndimage.correlate(img, SOBEL_X, mode="reflect")
sobel_y = ndimage.correlate(img, SOBEL_Y, mode="reflect")
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

out_dir = IMAGE_PATH.parent / "sobel"
out_dir.mkdir(exist_ok=True)
for name, a in [("sobel_x", sobel_x), ("sobel_y", sobel_y),
                ("gradient_magnitude", magnitude)]:
    tifffile.imwrite(out_dir / f"{IMAGE_PATH.stem}_{name}.tif", a)
print(f"Input: {IMAGE_PATH.name}  shape {image.shape}  type {image.dtype}")
print(f"Saved 3 images (32-bit float) in "
      f"{out_dir.relative_to(IMAGE_PATH.parent.parent)}/")

for name, a in [("Sobel x", sobel_x), ("Sobel y", sobel_y),
                ("gradient magnitude", magnitude)]:
    print(f"  {name:<20} min {a.min():>9.1f}  max {a.max():>9.1f}  "
          f"mean {a.mean():>8.1f}")

# ---- Which direction do the edges mostly run? ---------------------------------
# Compare how much total signal each filter picks up.
ex, ey = np.abs(sobel_x).sum(), np.abs(sobel_y).sum()
print(f"\nTotal |Sobel x| / total |Sobel y| = {ex / ey:.2f}")
print("  > 1: more left-right changes (vertical fibers dominate)")
print("  < 1: more top-bottom changes (horizontal fibers dominate)")

# ---- One line or two? A profile across a fiber --------------------------------
# Take the brightest point of a smoothed image (center of a fiber), then a
# 15-pixel line through it, across the fiber.
smooth = ndimage.gaussian_filter(img, 1)
r, c = np.unravel_index(np.argmax(smooth), smooth.shape)
# Direction across the fiber = direction of the strongest change around it.
gx = ndimage.correlate(smooth, SOBEL_X)[r - 3:r + 4, c - 3:c + 4]
gy = ndimage.correlate(smooth, SOBEL_Y)[r - 3:r + 4, c - 3:c + 4]
across = "horizontal" if np.abs(gx).sum() >= np.abs(gy).sum() else "vertical"
if across == "horizontal":
    cols = np.clip(np.arange(c - 7, c + 8), 0, img.shape[1] - 1)
    prof_in, prof_mag = img[r, cols], magnitude[r, cols]
else:
    rows = np.clip(np.arange(r - 7, r + 8), 0, img.shape[0] - 1)
    prof_in, prof_mag = img[rows, c], magnitude[rows, c]

print(f"\nProfile across a bright fiber at row {r}, column {c} "
      f"({across} line, 15 pixels):")
print("  position :", " ".join(f"{i:>5}" for i in range(-7, 8)))
print("  input    :", " ".join(f"{v:>5.0f}" for v in prof_in))
print("  magnitude:", " ".join(f"{v:>5.0f}" for v in prof_mag))
print("  -> input has ONE peak (the fiber); magnitude is high on its two")
print("     sides (where brightness rises and falls) and drops at the center.")

# ---- Show in napari ---------------------------------------------------------------
viewer = napari.Viewer()
viewer.add_image(img, name="input")
m = float(np.percentile(np.abs(np.concatenate([sobel_x, sobel_y])), 99.5))
viewer.add_image(sobel_x, name="Sobel x (left to right)",
                 contrast_limits=(-m, m))
viewer.add_image(sobel_y, name="Sobel y (top to bottom)",
                 contrast_limits=(-m, m))
viewer.add_image(magnitude, name="gradient magnitude",
                 contrast_limits=(0, float(np.percentile(magnitude, 99.5))))
viewer.grid.enabled = True
viewer.grid.shape = (2, 2)
napari.run()
