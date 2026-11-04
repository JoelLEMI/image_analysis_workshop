"""Stripe images next to their Fourier images, with labelled axes.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_stripes_pair.py

The pictures are saved in Pictures/fourier2d/:

    stripes_pair.png          3 stripes from left to right
    stripes_tilted_pair.png   tilted stripes
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "Pictures" / "fourier2d"
ORANGE = "#eb6834"
SIZE = 64


def stripes_pair(kx, ky, title, filename):
    """Draw a stripe image that repeats kx times from left to right and
    ky times from top to bottom, next to its Fourier image."""
    # --- The stripe image ---
    y, x = np.mgrid[0:SIZE, 0:SIZE] / SIZE  # 0 ... 1 across the image
    image = 0.5 + 0.5 * np.cos(2 * np.pi * (kx * x + ky * y))

    # --- Its Fourier image (zero frequency moved to the center) ---
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    log_amplitude = np.log1p(np.abs(spectrum))  # log, so weak spots are visible too

    # Frequencies (in cycles per image) belonging to each column/row of the Fourier image.
    freqs = np.fft.fftshift(np.fft.fftfreq(SIZE, d=1 / SIZE))  # -32 ... 31

    fig, (ax_img, ax_fft) = plt.subplots(1, 2, figsize=(10, 5))

    # Left: the real image, axes in pixels.
    ax_img.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_img.set_title(f"image: {title}")
    ax_img.set_xlabel("x (pixels)")
    ax_img.set_ylabel("y (pixels)")

    # Right: the Fourier image, axes in frequencies (cycles per image).
    # Each pixel covers one frequency step; centre the pixels on whole numbers.
    extent = [freqs[0] - 0.5, freqs[-1] + 0.5, freqs[-1] + 0.5, freqs[0] - 0.5]
    ax_fft.imshow(log_amplitude, cmap="gray", interpolation="nearest", extent=extent)
    ax_fft.set_title("Fourier image (log of amplitudes)")
    ax_fft.set_xlabel("frequency in x (cycles per image)")
    ax_fft.set_ylabel("frequency in y (cycles per image)")

    # How to read a spot: distance from the center = frequency,
    # angle around the center = direction of the wave.
    ax_fft.plot(0, 0, "+", color=ORANGE, ms=12, mew=1.5)
    ax_fft.annotate("", xy=(kx, ky), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
    ax_fft.text(-2, 4, "frequency\n(distance from center)",
                color=ORANGE, ha="right", va="top", fontsize=9)

    # Direction: the angle measured from the "frequency in x" axis.
    angle = np.degrees(np.arctan2(ky, kx))
    radius = 2 * np.hypot(kx, ky) + 4  # draw the arc outside the spot
    if abs(angle) < 1:  # horizontal frequency: draw a small arc up to angle 0
        ax_fft.add_patch(Arc((0, 0), 2 * radius, 2 * radius,
                             theta1=-60, theta2=0, color=ORANGE, lw=1.5))
        start = np.radians(-10)
        ax_fft.annotate("", xy=(radius, 0),
                        xytext=(radius * np.cos(start), radius * np.sin(start)),
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
        text_xy = (radius + 1, -radius)
    else:  # draw an arc from the x axis to the line through the spot
        ax_fft.plot([0, radius], [0, 0], ":", color=ORANGE, lw=1)
        ax_fft.plot([0, radius * np.cos(np.radians(angle))],
                    [0, radius * np.sin(np.radians(angle))], ":", color=ORANGE, lw=1)
        ax_fft.add_patch(Arc((0, 0), 2 * radius, 2 * radius,
                             theta1=min(0, angle), theta2=max(0, angle),
                             color=ORANGE, lw=1.5))
        text_xy = (radius + 1, radius * np.sin(np.radians(angle)) / 2)
        ax_fft.text(radius * 0.55, radius * np.sin(np.radians(angle / 2)) * 0.45,
                    f"{abs(angle):.0f}°", color=ORANGE, ha="center", va="center",
                    fontsize=9)
    ax_fft.text(*text_xy, "direction\n(angle around center)",
                color=ORANGE, ha="left", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=100)
    plt.close(fig)
    print(f"Saved {OUT_DIR / filename}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stripes_pair(3, 0, "3 stripes, left to right", "stripes_pair.png")
    stripes_pair(6, 6, "tilted stripes", "stripes_tilted_pair.png")


if __name__ == "__main__":
    main()
