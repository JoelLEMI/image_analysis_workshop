"""Remove noise by keeping mostly the slow waves: a Gaussian low-pass filter.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_gaussian_lowpass_nucleus.py

The script calculates the Fourier image of Nucleus_cropped_noise.tif, multiplies
it with a Gaussian-shaped mask (1 in the center, smoothly fading out to 0), and
transforms back.  Two images are saved next to the original in testImages/:

    Nucleus_cropped_noise_gaussian_mask_fourier.tif  the Fourier image (amplitudes,
                                                     log) after applying the mask
    Nucleus_cropped_noise_gaussian_mask_lowpass.tif  the image transformed back
"""

from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# The width (sigma) of the Gaussian mask, as a fraction of the image size.
# Try other values: a narrower mask removes more noise, but blurs more.
SIGMA_FRACTION = 0.048

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray().astype(float)
    # Keep the pixel size, so the filtered image has the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels")

# The Fourier image, with the zero frequency (the average brightness) moved to
# the center, so that the slow waves are in the middle and the fast waves
# (fine details and noise) further out.
fourier = np.fft.fftshift(np.fft.fft2(image))

# Make the mask: for every pixel of the Fourier image, calculate how far it is
# from the center, as a fraction of the image size in that direction.  The image
# is not exactly square, and this keeps the blur the same in both directions.
row_distance, col_distance = np.ogrid[:rows, :cols]
row_fraction = (row_distance - rows // 2) / rows
col_fraction = (col_distance - cols // 2) / cols
mask = np.exp(-(row_fraction**2 + col_fraction**2) / (2 * SIGMA_FRACTION**2))

# A Gaussian mask in Fourier space is the same as convolving with a Gaussian
# kernel in real space.  The wider the mask, the narrower the kernel.
kernel_sigma = 1 / (2 * np.pi * SIGMA_FRACTION)
print(f"Gaussian mask with sigma {SIGMA_FRACTION:.3f} of the image size: "
      f"the same as a Gaussian kernel with sigma {kernel_sigma:.2f} pixels")

# Multiply: each wave is weakened by the value of the mask at its position.
filtered_fourier = fourier * mask

# Transform back.  ifftshift undoes the fftshift from above.  The result has
# tiny imaginary parts left over from rounding; "real" drops them.
filtered = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fourier)))
print(f"Pixel values: original {image.min():.0f} - {image.max():.0f}, "
      f"filtered {filtered.min():.0f} - {filtered.max():.0f}")

# The Fourier image to look at: the amplitudes, on a log scale (otherwise only
# the very bright center is visible).  Pixels in Fourier space are frequencies,
# not distances, so the pixel size of the image does not apply to this one.
fourier_view = np.log1p(np.abs(filtered_fourier))
output = INPUT.with_name(f"{INPUT.stem}_gaussian_mask_fourier.tif")
tifffile.imwrite(output, fourier_view.astype(np.float32), imagej=True)
print(f"Saved {output.name}")

# Saved as a 32-bit float image: the filtered values have fractions.
output = INPUT.with_name(f"{INPUT.stem}_gaussian_mask_lowpass.tif")
tifffile.imwrite(output, filtered.astype(np.float32), imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"Saved {output.name}")
