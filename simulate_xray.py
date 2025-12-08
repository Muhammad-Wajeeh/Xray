import numpy as np
from skimage.transform import rotate, rescale

def simulate_xray_2d(
        phantom,
        angle_deg,
        I0=1.0,
        sid=500.0,
        sdd=1000.0,
        kVp=30.0,
        exposure_time=1.0,
        filtration_mmAl=0.0,
        grid_ratio=1.0,
    ):
    """
    Compute 2D radiograph: rotate phantom, magnify (sdd/sid), apply Beer–Lambert with energy,
    filtration, exposure, and grid scaling. Params: phantom (2D μ), angle_deg, I0, sid, sdd,
    kVp, exposure_time, filtration_mmAl, grid_ratio.
    """
    rotated = rotate(phantom, angle=angle_deg, resize=False, mode="edge")
    nx, ny = rotated.shape

    M = sdd / sid
    mag = rescale(rotated, M, mode='edge', preserve_range=True, anti_aliasing=False)

    sx, sy = mag.shape
    out = np.zeros_like(rotated)
    x_start = max((sx - nx) // 2, 0)
    y_start = max((sy - ny) // 2, 0)
    x_end = x_start + min(nx, sx)
    y_end = y_start + min(ny, sy)
    ox = max((nx - sx) // 2, 0)
    oy = max((ny - sy) // 2, 0)
    out[ox:ox + (x_end - x_start), oy:oy + (y_end - y_start)] = mag[x_start:x_end, y_start:y_end]
    mag = out

    raw_path = np.cumsum(mag, axis=1)
    path_integral = raw_path / (ny * 0.05)

    energy_factor = (60.0 / kVp)
    path_integral *= energy_factor

    mu_al_ref = 0.12
    mu_al = mu_al_ref * (30.0 / kVp)
    path_integral += filtration_mmAl * mu_al

    I = I0 * np.exp(-path_integral)

    I *= (exposure_time * 1.2)
    I = _apply_grid(I, grid_ratio)

    I = np.clip(I, 0.0, 1.0)

    return I




def _apply_magnification(image, sid, sdd):
    """Magnify by M=sdd/sid using rescale + center crop/pad to original size."""
    M = sdd / sid
    if np.isclose(M, 1.0):
        return image

    nx, ny = image.shape

    scaled = rescale(
        image,
        scale=M,
        mode='edge',
        anti_aliasing=False,
        preserve_range=True,
    )

    sx, sy = scaled.shape
    out = np.zeros_like(image)

    x_start = max((sx - nx) // 2, 0)
    y_start = max((sy - ny) // 2, 0)
    x_end = x_start + min(nx, sx)
    y_end = y_start + min(ny, sy)

    ox_start = max((nx - sx) // 2, 0)
    oy_start = max((ny - sy) // 2, 0)
    ox_end = ox_start + (x_end - x_start)
    oy_end = oy_start + (y_end - y_start)

    out[ox_start:ox_end, oy_start:oy_end] = scaled[x_start:x_end, y_start:y_end]

    return out

def _apply_energy_scaling(path_integral, kVp, ref_kVp=30.0):
    """Scale attenuation by ref_kVp/kVp (higher kVp => lower effective μ)."""
    return path_integral * (ref_kVp / kVp)


def _apply_filtration(path_integral, filtration_mmAl, kVp):
    """Add filtration term proportional to mm Al and inverse kVp."""
    mu_al_ref = 0.15
    mu_al = mu_al_ref * (30.0 / kVp)
    extra = filtration_mmAl * mu_al
    return path_integral + extra


def _apply_exposure(I, exposure_time, ref_time=1.0):
    """Scale intensity by exposure_time/ref_time."""
    return I * (exposure_time / ref_time)

def _apply_grid(I, grid_ratio=1.0):
    """Apply grid attenuation multiplier (<=1)."""
    return I * grid_ratio


def simulate_projection(phantom, I0=1.0,
                        sid=500.0, sdd=1000.0,
                        kVp=30.0,
                        exposure_time=1.0,
                        filtration_mmAl=0.0,
                        grid_ratio=1.0):
    """
    1D vertical projection with magnification and Beer–Lambert physics.
    Params: phantom, I0, sid, sdd, kVp, exposure_time, filtration_mmAl, grid_ratio.
    """
    mag_phantom = _apply_magnification(phantom, sid, sdd)

    path_integral = np.sum(mag_phantom, axis=0)

    path_integral = _apply_energy_scaling(path_integral, kVp)
    path_integral = _apply_filtration(path_integral, filtration_mmAl, kVp)

    I = I0 * np.exp(-path_integral)

    I = _apply_exposure(I, exposure_time)
    I = _apply_grid(I, grid_ratio)

    return I


def simulate_projection_angle(phantom, angle_deg, I0=1.0,
                              sid=500.0, sdd=1000.0,
                              kVp=30.0,
                              exposure_time=1.0,
                              filtration_mmAl=0.0,
                              grid_ratio=1.0):
    """
    Angled projection with rotation, magnification, energy/filtration, exposure, grid.
    Params: phantom, angle_deg, I0, sid, sdd, kVp, exposure_time, filtration_mmAl, grid_ratio.
    """
    rotated = rotate(phantom, angle=angle_deg, resize=False, mode='edge')
    rotated_mag = _apply_magnification(rotated, sid, sdd)

    path_integral = np.sum(rotated_mag, axis=0)

    path_integral = _apply_energy_scaling(path_integral, kVp)
    path_integral = _apply_filtration(path_integral, filtration_mmAl, kVp)

    I = I0 * np.exp(-path_integral)
    I = _apply_exposure(I, exposure_time)
    I = _apply_grid(I, grid_ratio)

    return I, rotated_mag


def simulate_2d_projection(phantom, angles_deg, I0=1.0):
    """
    Compute a 2D Radon sinogram:
    Each row = projection at one angle
    Each column = detector pixel
    Params: phantom, angles_deg (iterable), I0 incident intensity.
    """
    sinogram = []

    for angle in angles_deg:
        rotated = rotate(phantom, angle=angle, resize=False, mode='edge')

        projection = np.sum(rotated, axis=0)

        I = I0 * np.exp(-projection)

        sinogram.append(I)

    return np.array(sinogram)

import numpy as np
from skimage.transform import rotate

def simulate_sinogram(
        phantom,
        max_angle_deg=180.0,
        angle_step_deg=1.0,
        I0=1.0,
        sid=500.0,
        sdd=1000.0,
        kVp=30.0,
        exposure_time=1.0,
        filtration_mmAl=0.0,
        grid_ratio=1.0,
    ):
    """
    Compute sinogram with magnification, energy/filtration, exposure, grid.
    Params: phantom, max_angle_deg, angle_step_deg, I0, sid, sdd, kVp, exposure_time, filtration_mmAl, grid_ratio.
    """

    mag_phantom = _apply_magnification(phantom, sid, sdd)

    if max_angle_deg <= 0:
        max_angle_deg = 1.0
    angles_deg = np.arange(0.0, max_angle_deg + 1e-6, angle_step_deg)

    sinogram_rows = []

    thickness_scale = 40.0

    for ang in angles_deg:
        rotated = rotate(mag_phantom, angle=ang, resize=False, mode="edge")

        path_integral = np.sum(rotated, axis=0) / thickness_scale

        path_integral = _apply_energy_scaling(path_integral, kVp)
        path_integral = _apply_filtration(path_integral, filtration_mmAl, kVp)

        I = I0 * np.exp(-path_integral)

        I = _apply_exposure(I, exposure_time)
        I = _apply_grid(I, grid_ratio)

        sinogram_rows.append(I)

    sinogram = np.array(sinogram_rows)

    sinogram = np.clip(sinogram, 0.0, 1.0)

    return sinogram, angles_deg


from skimage.transform import radon

def simulate_projection_single(phantom, angle_deg, sid, sdd, kVp, exposure, filtration, grid_ratio=1.0):
    """Single-angle Radon projection expanded to 2D for display. Params: phantom, angle_deg, sid, sdd, kVp, exposure, filtration, grid_ratio."""
    mag = _apply_magnification(phantom, sid, sdd)

    theta = [angle_deg]
    sinogram = radon(mag, theta=theta, circle=False)

    proj = sinogram[:, 0]

    proj = _apply_energy_scaling(proj, kVp)
    proj = _apply_filtration(proj, filtration, kVp)
    I = np.exp(-proj)
    I = _apply_exposure(I, exposure)
    I = _apply_grid(I, grid_ratio)

    img = np.tile(I, (phantom.shape[0], 1))
    return np.clip(img, 0, 1)

def simulate_sinogram(phantom, max_angle, sid, sdd, kVp, exposure, filtration, grid_ratio=1.0):
    """Legacy sinogram builder using Radon on magnified phantom. Params: phantom, max_angle, sid, sdd, kVp, exposure, filtration, grid_ratio."""
    mag = _apply_magnification(phantom, sid, sdd)
    angles = np.arange(0, max_angle + 1, 1)
    sino = radon(mag, theta=angles, circle=False)

    sino = _apply_energy_scaling(sino, kVp)
    sino = _apply_filtration(sino, filtration, kVp)
    sino = np.exp(-sino)
    sino = _apply_exposure(sino, exposure)
    sino = _apply_grid(sino, grid_ratio=grid_ratio)

    return np.clip(sino, 0, 1), angles
