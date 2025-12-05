import numpy as np
from skimage.transform import rotate, rescale

import numpy as np
from skimage.transform import rotate, rescale

# keep your existing _apply_magnification, _apply_energy_scaling,
# _apply_filtration, _apply_exposure at the top of this file

def simulate_xray_2d(phantom,
                     angle_deg,
                     I0=1.0,
                     sid=500.0,
                     sdd=1000.0,
                     kVp=30.0,
                     exposure_time=1.0,
                     filtration_mmAl=0.0):
    """
    Simple 2D X-ray 'radiograph':

    - rotate phantom by angle_deg
    - apply magnification using SID/SDD
    - integrate attenuation across the image (cumulative sum)
    - apply energy scaling, filtration and exposure
    """

    # 1) Rotate phantom
    rotated = rotate(phantom, angle=angle_deg, resize=False, mode="edge")

    # 2) Magnify according to geometry (SID / SDD)
    mag = _apply_magnification(rotated, sid, sdd)   # uses SDD/SID internally

    # 3) Build a 2D path integral using cumulative sum along x
    #    (each pixel sees different path length → visible structure)
    path_integral = np.cumsum(mag, axis=1)

    # Scale by magnification so SID / SDD visibly change contrast
    M = sdd / sid
    path_integral = path_integral * M / mag.shape[1]

    # 4) Apply energy scaling & filtration
    path_integral = _apply_energy_scaling(path_integral, kVp)
    path_integral = _apply_filtration(path_integral, filtration_mmAl, kVp)

    # 5) Beer–Lambert (per pixel)
    I = I0 * np.exp(-path_integral)

    # 6) Exposure
    I = _apply_exposure(I, exposure_time)

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



