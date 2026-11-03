"""
Apply a Sharpen filter to testImages/Actin_cropped.tif and compare the result
with the input.

The kernel (the same one ImageJ uses for Process > Sharpen):

    -1  -1  -1
    -1  12  -1
    -1  -1  -1

Its weights add up to 12 - 8 = 4, so we divide by 4 to keep the overall
brightness the same. Each output pixel is then:

    (12 x center - sum of the 8 neighbors) / 4
  = center + 2 x (center - average of the 8 neighbors)

In words: every pixel is pushed further away from its surroundings.
Differences between neighbors (edges, fine details ... and noise) get bigger.

The result is saved as testImages/sharpened/Actin_cropped_sharpen.tif.

Run from the image_analysis_workshop folder:
    uv run python testImages/sharpen_actin.py
Or on another image:
    uv run python testImages/sharpen_actin.py Nucleus_cropped_noise
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

# ---- The Sharpen kernel, normalized by its sum ------------------------------
KERNEL = np.array([[-1, -1, -1],
                   [-1, 12, -1],
                   [-1, -1, -1]], dtype=np.float32)
print(f"Kernel sum = {KERNEL.sum():.0f}, so every weight is divided by it.")
KERNEL = KERNEL / KERNEL.sum()

# ---- Load and filter --------------------------------------------------------
image = tifffile.imread(IMAGE_PATH)
img = image.astype(np.float32)          # floats: results can be negative
sharp = ndimage.convolve(img, KERNEL, mode="reflect")

out_dir = IMAGE_PATH.parent / "sharpened"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"{IMAGE_PATH.stem}_sharpen.tif"
tifffile.imwrite(out_path, sharp)
print(f"Image: {IMAGE_PATH.name}  shape {image.shape}  type {image.dtype}")
print(f"Saved: {out_path.relative_to(IMAGE_PATH.parent.parent)}")


# ---- Compare -------------------------------------------------------------------
def noise_level(a, background):
    """Fine-grain noise in the background: how much each pixel differs from the
    median of its 5 x 5 neighborhood. Slow brightness changes are ignored, so
    what is left is the pixel-to-pixel 'grain' (noise)."""
    grain = a - ndimage.median_filter(a, size=5)
    return float(np.std(grain[background]))


def neighbor_variation(a):
    """Spread of the differences between neighboring pixels (whole image)."""
    return float(np.std(np.diff(a, axis=1)))


def edge_strength(a, structure):
    """Average gradient on the actin structures: how steep the edges are."""
    grad = np.hypot(*np.gradient(a))
    return float(np.mean(grad[structure]))


# Split the image into background and structure using a smoothed version,
# so the split itself is not affected by noise.
smooth = ndimage.gaussian_filter(img, 2)
background = smooth < np.percentile(smooth, 30)     # dimmest 30% = background
structure = smooth > np.percentile(smooth, 85)      # brightest 15% = actin

rows = [("input", img), ("sharpened", sharp)]
print(f"\n{'':<11}{'min':>9}{'max':>9}{'mean':>9}{'bg noise':>10}"
      f"{'nbr diff':>10}{'edges':>9}")
for name, a in rows:
    print(f"{name:<11}{a.min():>9.1f}{a.max():>9.1f}{a.mean():>9.1f}"
          f"{noise_level(a, background):>10.1f}{neighbor_variation(a):>10.1f}"
          f"{edge_strength(a, structure):>9.1f}")

print(f"\nbackground noise: x{noise_level(sharp, background) / noise_level(img, background):.1f}")
print(f"edge steepness:   x{edge_strength(sharp, structure) / edge_strength(img, structure):.1f}")
n_neg = int((sharp < 0).sum())
print(f"negative pixels after sharpening: {n_neg} "
      f"({100 * n_neg / sharp.size:.1f}% of the image)")

print("\nbg noise = fine-grain noise in the dimmest 30% of the image (background)")
print("nbr diff = spread of differences between neighboring pixels")
print("edges    = average gradient on the brightest 15% (actin structures)")

# ---- Show in napari ---------------------------------------------------------------
limits = (float(img.min()), float(img.max()))
viewer = napari.Viewer()
viewer.add_image(img, name="input", contrast_limits=limits)
viewer.add_image(sharp, name="sharpened (same contrast)", contrast_limits=limits)
viewer.add_image(sharp - img, name="what sharpening added", colormap="PiYG",
                 contrast_limits=(-float(np.percentile(np.abs(sharp - img), 99)),
                                  float(np.percentile(np.abs(sharp - img), 99))))
viewer.grid.enabled = True
viewer.grid.shape = (1, 3)
napari.run()
