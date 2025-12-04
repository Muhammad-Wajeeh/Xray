import numpy as np

def rotate_image_nn(image: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    Rotate a 2D image around its center by theta_deg using nearest-neighbor
    interpolation. No SciPy required.

    Positive theta = counter-clockwise.
    Output has the same shape as input.
    """
    if theta_deg % 360 == 0:
        return image.copy()

    theta = np.deg2rad(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    nx, ny = image.shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    # output grid
    x_out, y_out = np.indices((nx, ny))
    x_out_c = x_out - cx
    y_out_c = y_out - cy

    # inverse rotation to sample from input
    x_in_c = cos_t * x_out_c + sin_t * y_out_c
    y_in_c = -sin_t * x_out_c + cos_t * y_out_c

    x_in = np.round(x_in_c + cx).astype(int)
    y_in = np.round(y_in_c + cy).astype(int)

    rotated = np.zeros_like(image)

    inside = (
        (x_in >= 0) & (x_in < nx) &
        (y_in >= 0) & (y_in < ny)
    )

    rotated[inside] = image[x_in[inside], y_in[inside]]
    return rotated


def apply_magnification(phantom: np.ndarray,
                        sid: float,
                        sdd: float) -> np.ndarray:
    """
    Very simple magnification model:
    magnification M = SDD / SID.
    We rescale the phantom with nearest neighbor to emulate
    how structures appear larger on the detector.

    This keeps the array size fixed; it just stretches/compresses
    content inside it.
    """
    M = sdd / sid
    if np.isclose(M, 1.0):
        return phantom.copy()

    nx, ny = phantom.shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    x_out, y_out = np.indices((nx, ny))
    x_out_c = x_out - cx
    y_out_c = y_out - cy

    # Map detector coords back to object plane
    x_in_c = x_out_c / M
    y_in_c = y_out_c / M

    x_in = np.round(x_in_c + cx).astype(int)
    y_in = np.round(y_in_c + cy).astype(int)

    mag = np.zeros_like(phantom)
    inside = (
        (x_in >= 0) & (x_in < nx) &
        (y_in >= 0) & (y_in < ny)
    )
    mag[inside] = phantom[x_in[inside], y_in[inside]]
    return mag


def apply_energy_scaling(path_integral: np.ndarray,
                         energy_keV: float,
                         ref_energy_keV: float = 30.0) -> np.ndarray:
    """
    Toy model for energy dependence:
    mu(E) ~ mu_ref * (ref_E / E)

    So higher kVp -> lower effective attenuation.
    """
    scale = ref_energy_keV / energy_keV
    return path_integral * scale


def apply_filtration(path_integral: np.ndarray,
                     filtration_mmAl: float,
                     energy_keV: float) -> np.ndarray:
    """
    Toy beam hardening / filtration model.

    We pretend there is an equivalent extra Al thickness in front
    of everything. mu_Al is a made-up value; tweak if you want.
    """
    # crude: make filter slightly less effective at higher kVp
    mu_al_ref = 0.15  # arbitrary
    mu_al = mu_al_ref * (30.0 / energy_keV)

    extra = filtration_mmAl * mu_al
    return path_integral + extra


def apply_exposure(I: np.ndarray,
                   exposure_time: float,
                   ref_time: float = 1.0) -> np.ndarray:
    """
    Simple exposure model:
    intensity ∝ exposure time.
    """
    return I * (exposure_time / ref_time)
