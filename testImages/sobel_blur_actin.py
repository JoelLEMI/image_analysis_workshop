"""
Gradient magnitude of testImages/Actin_cropped.tif with and without blurring
first.

Edge filters (like Sobel) respond to differences between neighboring pixels.
Noise also creates such differences, so it shows up as "edges" everywhere.
Blurring first with a small Gaussian removes most of the pixel-to-pixel noise
while keeping the larger edges of the actin fibers.

Steps:
  1. gradient magnitude of the input                      (no blur)
  2. 5 x 5 Gaussian blur (sigma 0.67, same as before), then gradient magnitude
  3. compare noise (background) and edges (on the fibers) in both

The images are saved in testImages/sobel/.

Run from the image_analysis_workshop folder:
    uv run python testImages/sobel_blur_actin.py
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

# ---- Kernels -------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T.copy()          # [[-1,-2,-1],[0,0,0],[1,2,1]]


def gaussian_kernel(size):
    """size x size Gaussian kernel (sigma = (size-1)/6), weights sum to 1."""
    sigma = (size - 1) / 6
    x = np.arange(size) - size // 2
    g = np.exp(-x**2 / (2 * sigma**2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


def gradient_magnitude(a):
    gx = ndimage.correlate(a, SOBEL_X, mode="reflect")
    gy = ndimage.correlate(a, SOBEL_Y, mode="reflect")
    return np.sqrt(gx**2 + gy**2)


# ---- Compute ------------------------------------------------------------------
image = tifffile.imread(IMAGE_PATH)
img = image.astype(np.float32)

blurred = ndimage.convolve(img, gaussian_kernel(5), mode="reflect")
mag_raw = gradient_magnitude(img)
mag_blur = gradient_magnitude(blurred)

out_dir = IMAGE_PATH.parent / "sobel"
out_dir.mkdir(exist_ok=True)
tifffile.imwrite(out_dir / f"{IMAGE_PATH.stem}_gradient_magnitude.tif", mag_raw)
tifffile.imwrite(out_dir / f"{IMAGE_PATH.stem}_gauss5_gradient_magnitude.tif",
                 mag_blur)
print(f"Input: {IMAGE_PATH.name}  shape {image.shape}  type {image.dtype}")
print(f"Saved 2 gradient magnitude images in "
      f"{out_dir.relative_to(IMAGE_PATH.parent.parent)}/")

# ---- Where is background, where are real edges? ---------------------------------
# Decided on a strongly smoothed copy, so the choice is the same for both
# results and is not influenced by the noise itself.
ref = ndimage.gaussian_filter(img, 3)
background = ref < np.percentile(ref, 15)            # dark area outside the cell
ref_edges = gradient_magnitude(ref)
edges = ref_edges > np.percentile(ref_edges, 90)     # strongest 10% = real edges

print(f"\nBackground: {background.sum()} pixels (darkest 15%, outside the cell)")
print(f"Edges:      {edges.sum()} pixels (strongest 10% of real edges)\n")

print(f"{'':<22}{'noise':>10}{'edges':>10}{'edges/noise':>13}{'max':>10}")
for name, m in [("no blur", mag_raw), ("Gaussian 5x5 first", mag_blur)]:
    noise = m[background].mean()
    edge = m[edges].mean()
    print(f"{name:<22}{noise:>10.1f}{edge:>10.1f}{edge / noise:>13.1f}"
          f"{m.max():>10.1f}")

n_raw, n_blur = mag_raw[background].mean(), mag_blur[background].mean()
e_raw, e_blur = mag_raw[edges].mean(), mag_blur[edges].mean()
print(f"\nnoise: x{n_blur / n_raw:.2f}  (down {100 * (1 - n_blur / n_raw):.0f}%)")
print(f"edges: x{e_blur / e_raw:.2f}  (down {100 * (1 - e_blur / e_raw):.0f}%)")
print("\nnoise       = mean gradient magnitude in the background (should be ~0)")
print("edges       = mean gradient magnitude on real edges")
print("edges/noise = how well edges stand out from the noise (higher = better)")

# ---- Show in napari ---------------------------------------------------------------
top = float(np.percentile(mag_raw, 99.5))     # same contrast for both results
viewer = napari.Viewer()
viewer.add_image(img, name="input")
viewer.add_image(blurred, name="Gaussian 5x5")
viewer.add_image(mag_raw, name="gradient magnitude, no blur",
                 contrast_limits=(0, top))
viewer.add_image(mag_blur, name="gradient magnitude, Gaussian 5x5 first",
                 contrast_limits=(0, top))
viewer.grid.enabled = True
viewer.grid.shape = (2, 2)
napari.run()
