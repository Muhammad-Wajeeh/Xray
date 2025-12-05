import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

def create_shepp_logan(nx=256, ny=256):
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (nx, ny), anti_aliasing=True)

    # BOOST μ values dramatically
    phantom = 0.2 + phantom * 1.5   # strong contrast

    return phantom
