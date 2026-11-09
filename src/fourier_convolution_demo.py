"""Show that convolution in real space is multiplication in Fourier space.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_convolution_demo.py

The noisy nucleus image is smoothed with an N x N kernel in two ways: by
convolution, and by multiplying the Fourier images and transforming back.
The script prints how much the two results differ, and saves a picture in
Pictures/fourier2d/.

Choose the kernel with KIND and SIZE below:
    KIND = "smooth"    all ones, divided by SIZE x SIZE (3 = Session 3 Smooth)
    KIND = "gaussian"  Gaussian with sigma = (SIZE - 1) / 6, weights sum to 1
                       (5 = the 5 x 5 Gaussian from Session 3, sigma 0.67)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"
KIND = "gaussian"  # "smooth" or "gaussian"
SIZE = 10           # kernel size (3 x 3, 5 x 5, 7 x 7, ...)


def gaussian_kernel(size):
    """size x size Gaussian kernel, sigma = (size - 1) / 6 (the kernel covers
    +/- 3 sigma), normalized so the weights sum to 1."""
    sigma = (size - 1) / 6
    x = np.arange(size) - size // 2
    g = np.exp(-x**2 / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum(), sigma


if KIND == "smooth":
    KERNEL = np.ones((SIZE, SIZE)) / SIZE**2  # all ones, divided by SIZE x SIZE
    LABEL = f"{SIZE} x {SIZE} Smooth kernel"
elif KIND == "gaussian":
    KERNEL, SIGMA = gaussian_kernel(SIZE)
    LABEL = f"{SIZE} x {SIZE} Gaussian kernel (sigma {SIGMA:.2f})"
else:
    raise ValueError('KIND must be "smooth" or "gaussian"')

# The file name always says which kernel was used, for example
#   convolution_theorem_smooth_7x7.png  or  convolution_theorem_gaussian_5x5.png
# so changing KIND or SIZE never overwrites another picture.
OUTPUT = (REPO / "Pictures" / "fourier2d"
          / f"convolution_theorem_{KIND}_{SIZE}x{SIZE}.png")


def fourier(image):
    """2D Fourier transform, with the zero frequency moved to the center."""
    return np.fft.fftshift(np.fft.fft2(image))


def kernel_as_image(kernel, shape):
    """Put the small kernel in an image of the given size (zeros elsewhere),
    with the kernel's center pixel at position (0, 0).  The Fourier transform
    needs the kernel to be the same size as the image to multiply them."""
    padded = np.zeros(shape)
    rows, cols = kernel.shape
    padded[:rows, :cols] = kernel
    # Move the center of the kernel to the top left corner.  Pixels that move
    # "off" the top or left edge come back in at the bottom or right edge.
    return np.roll(padded, (-(rows // 2), -(cols // 2)), axis=(0, 1))


image = tifffile.imread(INPUT).astype(float)
kernel_image = kernel_as_image(KERNEL, image.shape)

# Way 1: convolution in real space.  mode="wrap" treats the image as repeating
# like tiles, which is what the Fourier transform does too (see the README).
smoothed = ndimage.convolve(image, KERNEL, mode="wrap")

# Way 2: multiply the Fourier images, pixel by pixel, and transform back.
image_f = np.fft.fft2(image)
kernel_f = np.fft.fft2(kernel_image)
smoothed_via_fourier = np.real(np.fft.ifft2(image_f * kernel_f))

difference = np.abs(smoothed - smoothed_via_fourier).max()
print(f"Pixel values of the smoothed image: {smoothed.min():.0f} - {smoothed.max():.0f}")
print(f"Largest difference between the two ways: {difference:.2g}")

rows, cols = image.shape
kernel_fourier = np.real(np.fft.fft2(kernel_as_image(KERNEL, image.shape)))
print(f"\n{LABEL}: Fourier image of the kernel goes from "
      f"{kernel_fourier.min():.4f} to {kernel_fourier.max():.4f}")
if KIND == "smooth":
    # Where the kernel removes patterns completely: frequencies m / SIZE.
    print("Dark lines (patterns removed completely):")
    for m in range(1, (SIZE - 1) // 2 + 1 if SIZE % 2 else SIZE // 2 + 1):
        f = m / SIZE
        print(f"  f = {m}/{SIZE} = {f:.3f} cycles/pixel  (stripes repeating every "
              f"{1 / f:.2f} px): {cols * f:.1f} px left/right and {rows * f:.1f} px "
              f"above/below the center of the Fourier image")
else:
    # A Gaussian has no dark lines: its Fourier image is a Gaussian too.
    f_sigma = 1 / (2 * np.pi * SIGMA)
    print(f"No dark lines: the Fourier image of a Gaussian is a Gaussian "
          f"(width {f_sigma:.2f} cycles/pixel = 1 / (2 pi sigma)).")
    print(f"Negative values (contrast inversion): "
          f"{'none' if kernel_fourier.min() >= 0 else f'{kernel_fourier.min():.4f}'}")
    # Computed from the actual 5 x 5 kernel (one row of it, as a 1-D profile),
    # for stripes running straight up and down.
    profile = KERNEL.sum(axis=0)
    x = np.arange(SIZE) - SIZE // 2
    print("How much of a stripe pattern the kernel keeps "
          "(vertical stripes, from the actual kernel):")
    for period in (20, 10, 7, 5, 3, 2):
        f = 1 / period
        kept = np.sum(profile * np.cos(2 * np.pi * f * x))
        print(f"  stripes every {period:>2} px (f = {f:.3f}): {100 * kept:5.1f}% kept")

# --- The picture --------------------------------------------------------------

image_amp = np.log1p(np.abs(fourier(image)))
smoothed_amp = np.log1p(np.abs(fourier(smoothed)))
kernel_amp = np.abs(fourier(kernel_image))  # between 0 and 1: no log needed
# Same brightness scale for both images, and for both Fourier images, so that
# you can compare them directly.
img_range = dict(vmin=image.min(), vmax=image.max())
amp_range = dict(vmin=smoothed_amp.min(), vmax=image_amp.max())

# Show the kernel enlarged: a small area around the center of the kernel image.
zoom = 5
kernel_zoom = np.fft.fftshift(kernel_image)
center_row, center_col = kernel_zoom.shape[0] // 2, kernel_zoom.shape[1] // 2
kernel_zoom = kernel_zoom[center_row - zoom:center_row + zoom + 1,
                          center_col - zoom:center_col + zoom + 1]

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
panels = [
    (axes[0, 0], image, "noisy image", img_range),
    (axes[0, 1], kernel_zoom, f"{LABEL}\n(enlarged)", {}),
    (axes[0, 2], smoothed, "smoothed image", img_range),
    (axes[1, 0], image_amp, "Fourier image (amplitudes, log)", amp_range),
    (axes[1, 1], kernel_amp, "Fourier image of the kernel\n(amplitudes, not log)",
     dict(vmin=0, vmax=1)),
    (axes[1, 2], smoothed_amp, "Fourier image (amplitudes, log)", amp_range),
]
for ax, data, title, brightness in panels:
    ax.imshow(data, cmap="gray", interpolation="nearest", **brightness)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

# The "equation" signs between the panels, placed in figure coordinates.
for x, y, sign in [(0.355, 0.73, "⊛"), (0.672, 0.73, "="),
                   (0.355, 0.27, "×"), (0.672, 0.27, "=")]:
    fig.text(x, y, sign, fontsize=30, ha="center", va="center")
fig.text(0.015, 0.73, "real space", rotation=90, fontsize=13, va="center")
fig.text(0.015, 0.27, "Fourier space", rotation=90, fontsize=13, va="center")
fig.suptitle("Convolution in real space = multiplication in Fourier space",
             fontsize=14)
fig.subplots_adjust(left=0.04, right=0.99, bottom=0.03, top=0.9,
                    wspace=0.25, hspace=0.2)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=100)
print(f"Saved {OUTPUT}")
