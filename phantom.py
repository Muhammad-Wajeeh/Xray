import numpy as np

def create_simple_phantom(nx=256, ny=256):
    """
    Create a simple 2D phantom.
    - background = soft tissue (mu = 0.02)
    - circle = bone (mu = 0.06)
    Returns: μ-map (attenuation coefficient map)
    """
    phantom = np.ones((nx, ny)) * 0.02  # soft tissue

    cx, cy = nx // 2, ny // 2
    radius = nx // 5

    for i in range(nx):
        for j in range(ny):
            if (i - cx)**2 + (j - cy)**2 <= radius**2:
                phantom[i, j] = 0.06  # bone

    return phantom
