"""
Cross-correlate testImages/Mito_Cropped.tif with testImages/Mito_template.tif,
WITHOUT any normalization.

Cross-correlation slides the template over the image. At every position it
multiplies the template with the image pixels underneath, pixel by pixel, and
adds everything up:

    CC(r, c) = sum over the template of  image(r + i, c + j) * template(i, j)

The output has the same size as the image; the value at (r, c) is for the
template CENTERED on that pixel.

The script:
  1. computes the cross-correlation and saves it
  2. prints where the highest value is
  3. finds where the template was really cut from (exact pixel match)
  4. compares the two, and shows everything in napari

Run from the image_analysis_workshop folder:
    uv run python testImages/cross_correlate_mito.py
"""

from pathlib import Path

import numpy as np
import tifffile
import napari
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

# ---- Find the files (relative to this script, so it runs from anywhere) ----
script_dir = Path(__file__).resolve().parent


def find(name):
    for p in (script_dir / "testImages" / name, script_dir / name,
              script_dir.parent / "testImages" / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {name}")


IMAGE_PATH = find("Mito_Cropped.tif")
TEMPLATE_PATH = find("Mito_template.tif")

image = tifffile.imread(IMAGE_PATH)
template = tifffile.imread(TEMPLATE_PATH)
print(f"Image:    {IMAGE_PATH.name}  shape {image.shape}  type {image.dtype}")
print(f"Template: {TEMPLATE_PATH.name}  shape {template.shape}  "
      f"type {template.dtype}")

# Float64: the sums are huge (millions x hundreds) and would overflow uint16.
img = image.astype(np.float64)
tpl = template.astype(np.float64)
th, tw = tpl.shape

# ---- 1. Plain cross-correlation (no normalization) -------------------------
# ndimage.correlate uses the template exactly as it is (no flipping).
# Outside the image is treated as 0.
cc = ndimage.correlate(img, tpl, mode="constant", cval=0.0)

out_dir = IMAGE_PATH.parent / "cross_correlation"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "Mito_cross_correlation.tif"
tifffile.imwrite(out_path, cc.astype(np.float32))
print(f"\nSaved: {out_path.relative_to(IMAGE_PATH.parent.parent)}")
print(f"Cross-correlation: min {cc.min():.3e}, max {cc.max():.3e}")

# ---- 2. Where is the highest value? -----------------------------------------
max_r, max_c = np.unravel_index(np.argmax(cc), cc.shape)
print(f"\nHighest cross-correlation at row {max_r}, column {max_c} "
      f"(value {cc[max_r, max_c]:.3e})")

# ---- 3. Where was the template really cut from? ------------------------------
# Compare the template with every 11 x 11 patch of the image; the patch with
# zero difference is the one it was cut from.
patches = sliding_window_view(img, tpl.shape)
diff = np.abs(patches - tpl).sum(axis=(2, 3))
top, left = np.unravel_index(np.argmin(diff), diff.shape)
true_r, true_c = top + th // 2, left + tw // 2
print(f"Template was cut from rows {top}-{top + th - 1}, "
      f"columns {left}-{left + tw - 1}  -> center row {true_r}, column {true_c}"
      f"  (difference {diff[top, left]:.0f}: exact match)")

# ---- 4. Compare ------------------------------------------------------------------
rank = int((cc > cc[true_r, true_c]).sum()) + 1
distance = np.hypot(max_r - true_r, max_c - true_c)
print(f"\nCross-correlation at the true location: {cc[true_r, true_c]:.3e}")
print(f"  -> {cc[max_r, max_c] / cc[true_r, true_c]:.1f} times LOWER than the maximum")
print(f"  -> only number {rank} out of {cc.size} pixels")
print(f"Distance between the maximum and the true location: {distance:.0f} pixels")

# Why? Compare with the local average brightness (11 x 11 mean).
local_mean = ndimage.uniform_filter(img, size=th, mode="constant")
r = np.corrcoef(cc.ravel(), local_mean.ravel())[0, 1]
print(f"\nLocal mean brightness at the maximum:       {local_mean[max_r, max_c]:.0f}")
print(f"Local mean brightness at the true location: {local_mean[true_r, true_c]:.0f}")
print(f"Correlation between the cross-correlation map and local brightness: {r:.3f}")
print("  -> the plain cross-correlation mostly measures how BRIGHT an area is,")
print("     not how similar it looks to the template.")

# ---- Show in napari ---------------------------------------------------------------
viewer = napari.Viewer()
viewer.add_image(img, name="Mito_Cropped")
viewer.add_image(tpl, name="Mito_template (11x11)",
                 translate=(top, left))  # shown where it was cut from
viewer.add_image(cc, name="cross-correlation", colormap="inferno")
viewer.add_points([[max_r, max_c]], name="highest cross-correlation",
                  face_color="red", size=8)
viewer.add_points([[true_r, true_c]], name="where the template was cut",
                  face_color="lime", size=8)
napari.run()
