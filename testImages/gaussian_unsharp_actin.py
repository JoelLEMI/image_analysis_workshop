"""
Gaussian blur and "unsharp masked" images of testImages/Actin_cropped.tif.

1. Blur the image with a 5 x 5 and a 21 x 21 Gaussian kernel.
2. Subtract each blurred image from the input:
       unsharp = input - blurred
   What is left are the fine details (edges, fibers ... and noise) that the
   blur removed. Everything slow and smooth (uneven background) cancels out.
3. Save all four images and show them in napari.
4. Check whether the pixel values are still whole numbers.

Kernel size and sigma: a Gaussian is ~0 beyond 3 sigma on each side, so a
kernel of size N covers +/- 3 sigma when sigma = (N - 1) / 6.
    5 x 5   -> sigma = 0.67 pixels
    21 x 21 -> sigma = 3.33 pixels

The images are saved in testImages/gaussian/.

Run from the image_analysis_workshop folder:
    uv run python testImages/gaussian_unsharp_actin.py
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

KERNEL_SIZES = [5, 21]


def gaussian_kernel(size):
    """size x size Gaussian kernel, normalized so its weights sum to 1."""
    sigma = (size - 1) / 6
    x = np.arange(size) - size // 2                  # e.g. -2 -1 0 1 2
    g1d = np.exp(-x**2 / (2 * sigma**2))
    kernel = np.outer(g1d, g1d)                      # 2-D = 1-D x 1-D
    return kernel / kernel.sum(), sigma


# ---- Load ----------------------------------------------------------------------
image = tifffile.imread(IMAGE_PATH)
print(f"Input: {IMAGE_PATH.name}  shape {image.shape}  type {image.dtype}")
img = image.astype(np.float32)   # floats, so averages and negatives are kept

out_dir = IMAGE_PATH.parent / "gaussian"
out_dir.mkdir(exist_ok=True)

blurred, unsharp = {}, {}
for size in KERNEL_SIZES:
    kernel, sigma = gaussian_kernel(size)
    if size == 5:
        with np.printoptions(precision=3, suppress=True):
            print(f"\n5 x 5 Gaussian kernel (sigma {sigma:.2f}), weights sum to 1:\n{kernel}")
    else:
        print(f"\n{size} x {size} Gaussian kernel (sigma {sigma:.2f}): "
              f"center weight {kernel.max():.4f}, corner weight {kernel.min():.1e}")

    blurred[size] = ndimage.convolve(img, kernel, mode="reflect")
    unsharp[size] = img - blurred[size]

    tifffile.imwrite(out_dir / f"{IMAGE_PATH.stem}_gauss{size}.tif", blurred[size])
    tifffile.imwrite(out_dir / f"{IMAGE_PATH.stem}_unsharp{size}.tif", unsharp[size])

print(f"\nSaved 4 images (32-bit float) in "
      f"{out_dir.relative_to(IMAGE_PATH.parent.parent)}/")

# ---- Look at the pixel values ---------------------------------------------------
def show_corner(name, a):
    with np.printoptions(precision=2, suppress=True, linewidth=200):
        print(f"\n{name}  type {a.dtype}  min {a.min():.2f}  max {a.max():.2f}  "
              f"mean {a.mean():.2f}")
        print(a[100:104, 100:105])

print("\n===== Pixel values (rows 100-103, columns 100-104) =====")
show_corner("input", image)
for size in KERNEL_SIZES:
    show_corner(f"Gaussian {size}x{size}", blurred[size])
for size in KERNEL_SIZES:
    show_corner(f"unsharp {size}x{size} (input - blurred)", unsharp[size])

for size in KERNEL_SIZES:
    whole = np.mean(blurred[size] == np.round(blurred[size])) * 100
    neg = np.mean(unsharp[size] < 0) * 100
    print(f"\n{size}x{size}: {whole:.1f}% of blurred pixels are whole numbers; "
          f"{neg:.1f}% of unsharp pixels are negative")

# What would happen if we had stayed in uint16 (whole numbers 0..65535)?
wrong = image - ndimage.convolve(image, gaussian_kernel(21)[0], mode="reflect")
print("\nSame subtraction done in uint16 instead of float (WRONG):")
print(f"  type {wrong.dtype}, min {wrong.min()}, max {wrong.max()}  <- negatives "
      f"wrapped around to huge positive numbers")

# ---- Show in napari ---------------------------------------------------------------
limits = (float(img.min()), float(img.max()))
viewer = napari.Viewer()
viewer.add_image(img, name="input", contrast_limits=limits)
for size in KERNEL_SIZES:
    viewer.add_image(blurred[size], name=f"Gaussian {size}x{size}",
                     contrast_limits=limits)
for size in KERNEL_SIZES:
    m = float(np.percentile(np.abs(unsharp[size]), 99.5))
    viewer.add_image(unsharp[size], name=f"unsharp {size}x{size}",
                     contrast_limits=(-m, m))
viewer.add_image(wrong, name="unsharp 21x21 in uint16 (WRONG)")
viewer.grid.enabled = True
viewer.grid.shape = (2, 3)   # 6 panels
napari.run()
