"""Show the Fourier image, the low-pass result and the noisy original side by side.

Run src/fourier_lowpass_nucleus.py first to make the images, then run this from
the top folder of the workshop repository with:

    uv run testImages/napari_side_by_side.py
"""

from pathlib import Path

import napari
import tifffile

FOLDER = Path(__file__).resolve().parent

# In this order, the images appear from left to right in the napari grid.
LAYERS = [
    ("Fourier (circle)", "Nucleus_cropped_noise_fourier_circle.tif"),
    ("Low-pass", "Nucleus_cropped_noise_lowpass.tif"),
    ("Original (noisy)", "Nucleus_cropped_noise.tif"),
]

viewer = napari.Viewer(title="Nucleus low-pass: side by side")
for name, filename in LAYERS:
    viewer.add_image(tifffile.imread(FOLDER / filename), name=name, colormap="gray")

# One row of three columns, with one layer in each grid position.
viewer.grid.enabled = True
viewer.grid.shape = (1, 3)
viewer.grid.stride = 1

viewer.window._qt_window.resize(1500, 650)
viewer.reset_view()
napari.run()
