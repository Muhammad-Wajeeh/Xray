import numpy as np
from skimage.transform import rotate

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


def simulate_projection_angle(phantom, angle_deg, I0=1.0):
    """
    Simulate X-ray projection at an angle by rotating the phantom.
    - angle_deg: angle to rotate the phantom (degrees)
    """
    # Rotate phantom (preserve size, bilinear interpolation)
    rotated = rotate(phantom, angle=angle_deg, resize=False, mode='edge')

    # Vertical integration of attenuation (path integral)
    path_integral = np.sum(rotated, axis=0)

    # Beer–Lambert law
    I = I0 * np.exp(-path_integral)

    return I, rotated


def simulate_2d_projection(phantom, angle_deg, I0=1.0):
    """
    Generates a 2D film-like projection by summing along ray direction.
    Equivalent to Radon transform for one angle (simplified).
    """
    rotated = rotate(phantom, angle=angle_deg, resize=False, mode='edge')

    # Sum along vertical direction -> 1D projection
    projection = np.sum(rotated, axis=0)

    # Expand the 1D projection into a 2D film (simple expansion)
    film = np.tile(projection, (phantom.shape[0], 1))

    # Apply Beer–Lambert law
    film = I0 * np.exp(-film)

    return film
