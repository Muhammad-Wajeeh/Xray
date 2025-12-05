import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

def create_shepp_logan(nx=256, ny=256):
    """
    Returns a classic Shepp–Logan phantom (μ-map).
    Output shape = (nx, ny).
    """
    phantom = shepp_logan_phantom()        # default size ≈ 400x400
    phantom = resize(phantom, (nx, ny), anti_aliasing=True)

    # Scale intensities to realistic μ values
    # (Soft tissue ~0.02–0.03, bone ~0.06–0.08)
    phantom = 0.02 + phantom * 0.06

    return phantom
