from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import numpy as np

def create_shepp_logan(nx=256, ny=256):
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (nx, ny), anti_aliasing=True)

    # Boost μ values for visible attenuation in this simple model
    phantom = 0.1 + phantom * 1.5

    return phantom
