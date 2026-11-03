"""
Smooth testImages/Nucleus_cropped_noise.tif with three linear (convolution)
kernels, each normalized by its sum, and compare them with each other and with
median filters.

Linear kernels (each output pixel = weighted average of its neighborhood):
    mean3     3 x 3, all ones                         -> divided by 9
    center3   3 x 3, all ones but 4 in the center     -> divided by 12
    mean7     7 x 7, all ones                         -> divided by 49

Median kernels (each output pixel = middle value of its sorted neighborhood):
    median_plus3, median_square3, median_square5

Normalizing by the sum makes the weights add up to 1, so the overall
brightness of the image stays the same.

The linear results are saved as .tif files in testImages/linear_filtered/.

Run from the image_analysis_workshop folder:
    uv run python testImages/linear_vs_median_filter.py
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

# ---- Linear kernels, normalized by their sum ---------------------------------
center3 = np.ones((3, 3))
center3[1, 1] = 4

LINEAR_KERNELS = {
    "mean3": np.ones((3, 3)),
    "center3": center3,
    "mean7": np.ones((7, 7)),
}
LINEAR_KERNELS = {name: k / k.sum() for name, k in LINEAR_KERNELS.items()}

# ---- Median kernels (True = pixel is used) ----------------------------------
MEDIAN_KERNELS = {
    "median_plus3": np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=bool),
    "median_square3": np.ones((3, 3), dtype=bool),
    "median_square5": np.ones((5, 5), dtype=bool),
}

# ---- Load ---------------------------------------------------------------------
image = tifffile.imread(IMAGE_PATH)
print(f"Image: {IMAGE_PATH.name}  shape {image.shape}  type {image.dtype}")

# Work in floating point so averages are not rounded off.
img = image.astype(np.float32)

# ---- Apply all filters ----------------------------------------------------------
results = {}
for name, kernel in LINEAR_KERNELS.items():
    results[name] = ndimage.convolve(img, kernel, mode="reflect")
for name, footprint in MEDIAN_KERNELS.items():
    results[name] = ndimage.median_filter(img, footprint=footprint, mode="reflect")

# Save the linear results (float32 keeps the decimals of the averages).
out_dir = IMAGE_PATH.parent / "linear_filtered"
out_dir.mkdir(exist_ok=True)
for name in LINEAR_KERNELS:
    tifffile.imwrite(out_dir / f"{IMAGE_PATH.stem}_{name}.tif", results[name])

# ---- Print the kernels ----------------------------------------------------------
print("\nLinear kernels after normalization (weights sum to 1):")
for name, kernel in LINEAR_KERNELS.items():
    with np.printoptions(precision=3, suppress=True, linewidth=200):
        if kernel.shape[0] <= 3:
            print(f"\n{name}  (sum {kernel.sum():.3f})\n{kernel}")
        else:
            print(f"\n{name}  (sum {kernel.sum():.3f}): every weight = "
                  f"{kernel[0, 0]:.4f} (= 1/{kernel.size})")


# ---- Compare ----------------------------------------------------------------------
def noise_level(a):
    """Rough noise estimate: spread of the difference between neighboring pixels.
    Smooth image -> neighbors are similar -> small number."""
    return float(np.std(np.diff(a, axis=1)))


def edge_strength(a):
    """Average gradient magnitude. It grows with sharp edges AND with leftover
    noise, so compare it together with the noise column."""
    return float(np.mean(np.hypot(*np.gradient(a))))


names = ["original"] + list(results)
all_images = {"original": img, **results}

print("\nSummary (original vs each filter):")
print(f"{'filter':<16}{'min':>9}{'max':>9}{'mean':>9}{'noise':>9}{'gradient':>10}"
      f"{'|diff| vs orig':>16}")
for name in names:
    a = all_images[name]
    diff = np.mean(np.abs(a - img))
    print(f"{name:<16}{a.min():>9.1f}{a.max():>9.1f}{a.mean():>9.1f}"
          f"{noise_level(a):>9.1f}{edge_strength(a):>10.1f}{diff:>16.1f}")

print("\nMean absolute difference between filtered images "
      "(small = the two filters give similar results):")
filt = list(results)
header = " " * 16 + "".join(f"{n[:12]:>14}" for n in filt)
print(header)
for a_name in filt:
    row = f"{a_name:<16}"
    for b_name in filt:
        d = np.mean(np.abs(results[a_name] - results[b_name]))
        row += f"{d:>14.1f}"
    print(row)

print("\nnoise = std of neighbor-to-neighbor differences (lower = smoother)")
print("gradient = mean gradient magnitude: edges PLUS leftover noise (lower = smoother/blurrier)")

# ---- Show in napari ---------------------------------------------------------------
limits = (float(img.min()), float(img.max()))
viewer = napari.Viewer()
for name in names:
    viewer.add_image(all_images[name], name=name, contrast_limits=limits)
# 8th panel: what the 3x3 median removed (original minus filtered).
# Bright dots = the outlier ("salt") pixels the median threw away.
removed = img - results["median_square3"]
viewer.add_image(removed, name="removed by median_square3",
                 contrast_limits=(0, float(np.percentile(removed, 99.5))),
                 colormap="magma")

viewer.grid.enabled = True
viewer.grid.shape = (2, 4)   # 2 rows x 4 columns: original + 6 filters + removed
napari.run()
