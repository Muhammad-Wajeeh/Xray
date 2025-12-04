import numpy as np

def simulate_projection(phantom, I0=1.0):
    """
    Simulate an X-ray projection by summing attenuation along columns.
    Very simple baseline model: straight, vertical rays.
    """
    # Sum attenuation along the ray direction
    path_integral = np.sum(phantom, axis=0)

    # Beer–Lambert
    I = I0 * np.exp(-path_integral)

    return I
