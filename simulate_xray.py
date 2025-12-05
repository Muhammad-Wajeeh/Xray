import numpy as np
from skimage.transform import rotate, rescale

import numpy as np
from skimage.transform import rotate, rescale

# keep your existing _apply_magnification, _apply_energy_scaling,
# _apply_filtration, _apply_exposure at the top of this file

import numpy as np
from skimage.transform import rotate, rescale

# keep your existing helper functions (_apply_magnification, etc.)

def simulate_xray_2d(
        phantom,
        angle_deg,
        I0=1.0,
        sid=500.0,
        sdd=1000.0,
        kVp=30.0,
        exposure_time=1.0,
        filtration_mmAl=0.0,
    ):
    """
    Balanced 2D X-ray simulation:
    - rotation
    - magnification (SID/SDD)
    - cumulative attenuation (reduced strength)
    - energy scaling (moderate)
    - filtration (moderate)
    - exposure (strong)
    """

    # 1. Rotate phantom
    rotated = rotate(phantom, angle=angle_deg, resize=False, mode="edge")
    nx, ny = rotated.shape

    # 2. Magnify image
    M = sdd / sid
    mag = rescale(rotated, M, mode='edge', preserve_range=True, anti_aliasing=False)

    # Crop/pad back to original size
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

    # 3. Path integral (reduced strength)
    raw_path = np.cumsum(mag, axis=1)
    path_integral = raw_path / (ny * 0.5)    # divide by 0.5 ny → weaker attenuation

    # 4. Energy scaling (mild)
    energy_factor = (40.0 / kVp)   # moderate change
    path_integral *= energy_factor

    # 5. Filtration (mild)
    mu_al_ref = 0.12               # reduced from 0.25
    mu_al = mu_al_ref * (30.0 / kVp)
    path_integral += filtration_mmAl * mu_al

    # 6. Beer–Lambert
    I = I0 * np.exp(-path_integral)

    # 7. Exposure (strong)
    I *= (exposure_time * 3.0)     # brighten radiograph significantly

    # 8. Clip
    I = np.clip(I, 0.0, 1.0)

    return I




def _apply_magnification(image, sid, sdd):
    """
    Very simple magnification model.
    M = SDD / SID. We rescale the image and then center-crop/pad
    back to the original size so the output shape is unchanged.
    """
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

    # crop or pad in x
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
    """
    Crude energy dependence: mu(E) ∝ (ref_kVp / kVp).
    Higher kVp => lower effective attenuation.
    """
    return path_integral * (ref_kVp / kVp)


def _apply_filtration(path_integral, filtration_mmAl, kVp):
    """
    Very simple beam hardening / filtration model.
    We pretend there's an extra Al thickness in front.
    """
    mu_al_ref = 0.15  # arbitrary
    mu_al = mu_al_ref * (30.0 / kVp)
    extra = filtration_mmAl * mu_al
    return path_integral + extra


def _apply_exposure(I, exposure_time, ref_time=1.0):
    """
    Intensity proportional to exposure time.
    """
    return I * (exposure_time / ref_time)


def simulate_projection(phantom, I0=1.0,
                        sid=500.0, sdd=1000.0,
                        kVp=30.0,
                        exposure_time=1.0,
                        filtration_mmAl=0.0):
    """
    1D vertical projection with magnification + basic physics.
    """
    mag_phantom = _apply_magnification(phantom, sid, sdd)

    path_integral = np.sum(mag_phantom, axis=0)

    # Energy + filtration
    path_integral = _apply_energy_scaling(path_integral, kVp)
    path_integral = _apply_filtration(path_integral, filtration_mmAl, kVp)

    # Beer–Lambert
    I = I0 * np.exp(-path_integral)

    # Exposure
    I = _apply_exposure(I, exposure_time)

    return I


def simulate_projection_angle(phantom, angle_deg, I0=1.0,
                              sid=500.0, sdd=1000.0,
                              kVp=30.0,
                              exposure_time=1.0,
                              filtration_mmAl=0.0):
    """
    Angled projection with magnification + basic physics.
    """
    rotated = rotate(phantom, angle=angle_deg, resize=False, mode='edge')
    rotated_mag = _apply_magnification(rotated, sid, sdd)

    path_integral = np.sum(rotated_mag, axis=0)

    path_integral = _apply_energy_scaling(path_integral, kVp)
    path_integral = _apply_filtration(path_integral, filtration_mmAl, kVp)

    I = I0 * np.exp(-path_integral)
    I = _apply_exposure(I, exposure_time)

    return I, rotated_mag


def simulate_2d_projection(phantom, angles_deg, I0=1.0):
    """
    Compute a 2D Radon sinogram:
    Each row = projection at one angle
    Each column = detector pixel
    """
    sinogram = []

    for angle in angles_deg:
        rotated = rotate(phantom, angle=angle, resize=False, mode='edge')

        # 1D projection = line integral
        projection = np.sum(rotated, axis=0)

        # Beer–Lambert
        I = I0 * np.exp(-projection)

        sinogram.append(I)

    return np.array(sinogram)

import numpy as np
from skimage.transform import rotate

# ... keep your existing helper functions here ...

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
    ):
    """
    Compute a 2D projection (sinogram) with simple X-ray physics.

    Each row = projection at one angle.
    Each column = detector position.

    Parameters
    ----------
    phantom : 2D np.ndarray
        μ-map (attenuation coefficients).
    max_angle_deg : float
        Maximum angle in degrees (0 → max_angle_deg).
    angle_step_deg : float
        Step between angles in degrees.
    I0 : float
        Incident intensity.
    sid, sdd : float
        Source–isocenter distance, source–detector distance (for magnification).
    kVp : float
        Tube voltage (affects effective attenuation).
    exposure_time : float
        Exposure time in seconds (brightness).
    filtration_mmAl : float
        Extra filtration thickness (mm Al).

    Returns
    -------
    sinogram : 2D np.ndarray (num_angles, num_detector_pixels)
    angles_deg : 1D np.ndarray of angle values used.
    """

    # 1) Magnify phantom according to geometry
    mag_phantom = _apply_magnification(phantom, sid, sdd)

    # 2) Angle list
    if max_angle_deg <= 0:
        max_angle_deg = 1.0  # avoid empty angle set
    angles_deg = np.arange(0.0, max_angle_deg + 1e-6, angle_step_deg)

    sinogram_rows = []

    # Thickness scale to keep Beer–Lambert in a nice range
    thickness_scale = 40.0  # tweakable

    for ang in angles_deg:
        # Rotate for current angle
        rotated = rotate(mag_phantom, angle=ang, resize=False, mode="edge")

        # Line integral along "beam" direction (vertical sum)
        path_integral = np.sum(rotated, axis=0) / thickness_scale

        # Energy + filtration
        path_integral = _apply_energy_scaling(path_integral, kVp)
        path_integral = _apply_filtration(path_integral, filtration_mmAl, kVp)

        # Beer–Lambert
        I = I0 * np.exp(-path_integral)

        # Exposure
        I = _apply_exposure(I, exposure_time)

        sinogram_rows.append(I)

    sinogram = np.array(sinogram_rows)

    # Clamp to [0, 1] so GUI can use fixed display range
    sinogram = np.clip(sinogram, 0.0, 1.0)

    return sinogram, angles_deg


from skimage.transform import radon

def simulate_projection_single(phantom, angle_deg, sid, sdd, kVp, exposure, filtration):
    # magnify phantom
    mag = _apply_magnification(phantom, sid, sdd)

    # radon expects angles in degrees
    theta = [angle_deg]
    sinogram = radon(mag, theta=theta, circle=False)

    # sinogram shape is (rows, 1)
    proj = sinogram[:, 0]

    # apply physics
    proj = _apply_energy_scaling(proj, kVp)
    proj = _apply_filtration(proj, filtration, kVp)
    I = np.exp(-proj)
    I = _apply_exposure(I, exposure)

    # expand to 2D (so GUI can display it)
    img = np.tile(I, (phantom.shape[0], 1))
    return np.clip(img, 0, 1)

from skimage.transform import radon
import numpy as np

def simulate_sinogram(phantom, max_angle, sid, sdd, kVp, exposure, filtration):
    mag = _apply_magnification(phantom, sid, sdd)
    angles = np.arange(0, max_angle + 1, 1)
    sino = radon(mag, theta=angles, circle=False)

    # apply physics per projection (column-wise)
    sino = _apply_energy_scaling(sino, kVp)
    sino = _apply_filtration(sino, filtration, kVp)
    sino = np.exp(-sino)
    sino = _apply_exposure(sino, exposure)

    return np.clip(sino, 0, 1), angles
