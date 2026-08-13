import matplotlib

# sets matplotlib backend to "agg" = Anti-Grain Geometry
# suited for raster images
# renders without opening a graphical window
matplotlib.use("Agg")

# the submodule pyplot provides the plotting API
# assigned to the alias "plt" - it's not an "object", just an abbreviation
import matplotlib.pyplot as plt

# Python Imaging Library. The module "Image" represents a loaded image as an object
# (open, display, resize, convert format, edit pixels, etc.)
from PIL import Image

img = Image.open("coffee.jpg")

# creates an Artist object (more precisely: AxesImage)
# and adds it to the current axes area (Axes)
# incl. axis scaling, numbers with pixel coordinates, color normalization
plt.imshow(img)

# "saveFigure" saves the prepared graphic
plt.savefig("../images/coffee_matplotlib.png")
