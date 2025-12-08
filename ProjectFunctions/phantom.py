from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import numpy as np


def create_shepp_logan(nx=256, ny=256):
    """Return resized Shepp-Logan phantom with boosted μ."""
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (nx, ny), anti_aliasing=True)
    phantom = 0.1 + phantom * 1.5
    return phantom


def create_breast_phantom(
    nx=256,
    ny=256,
    lesion_radius=25,
    compression=False,
    compression_factor=0.65,
):
    """
    Build 2D breast phantom with skin, pectoral wedge, glandular crescent, lesion, calc spots, benign ellipse.
    Returns (phantom, info) where info contains ROI masks (lesion/background) and μ values.
    """
    adipose_mu = 0.22
    gland_mu = 0.40
    lesion_mu = 0.75
    micro_mu = 0.55
    muscle_mu = 0.50
    skin_mu = 0.80

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    breast_mask = (xx**2) / (0.9**2) + (yy**2) / (1.0**2) <= 1.0
    phantom = np.full((nx, ny), adipose_mu)
    phantom[~breast_mask] = 0.0

    thickness = np.exp(-3.0 * (xx**2 + yy**2))
    phantom *= 0.8 + 0.2 * thickness

    skin_rim = (np.abs((xx**2) / (0.92**2) + (yy**2) / (1.02**2) - 1.0) < 0.03)
    phantom[skin_rim] = skin_mu

    pec_mask = (xx < -0.55) & (yy > -0.2) & (yy < 0.9) & ((yy + 0.9) > (xx + 0.2))
    phantom[pec_mask] = muscle_mu

    gland_mask = ((xx + 0.15) ** 2) / (0.55**2) + (yy**2) / (0.6**2) <= 1.0
    gland_mask |= ((xx + 0.05) ** 2) / (0.45**2) + ((yy + 0.15) ** 2) / (0.5**2) <= 1.0
    phantom[gland_mask & breast_mask] = gland_mu

    cx = nx // 2
    cy = ny // 2
    rr = lesion_radius
    lesion_mask = (
        (np.arange(nx)[:, None] - cx) ** 2 + (np.arange(ny)[None, :] - cy - 25) ** 2
    ) <= rr**2
    phantom[lesion_mask] = lesion_mu

    rng = np.random.default_rng(42)
    num_spots = 7
    spot_centers = rng.normal(loc=[-0.1, 0.1], scale=0.18, size=(num_spots, 2))
    for sx, sy in spot_centers:
        rad = rng.uniform(0.015, 0.04)
        spot_mask = (xx - sx) ** 2 + (yy - sy) ** 2 <= rad**2
        phantom[spot_mask & breast_mask] = micro_mu

    benign_mask = ((xx + 0.35) ** 2) / (0.12**2) + ((yy - 0.25) ** 2) / (0.08**2) <= 1.0
    phantom[benign_mask & breast_mask] = (gland_mu + micro_mu) * 0.5

    info = {
        "adipose_mu": adipose_mu,
        "gland_mu": gland_mu,
        "lesion_mu": lesion_mu,
        "lesion_mask": lesion_mask,
        "background_mask": breast_mask & (~lesion_mask),
    }

    if compression:
        phantom, info = _compress_phantom(phantom, info, compression_factor)

    return phantom, info


def _compress_phantom(phantom, info, factor):
    """Compress along superior-inferior axis by factor, pad to original size, and adjust masks."""
    nx, ny = phantom.shape
    comp_nx = max(1, int(nx * factor))
    compressed = resize(
        phantom,
        (comp_nx, ny),
        anti_aliasing=True,
        preserve_range=True,
    )

    pad_top = (nx - comp_nx) // 2
    pad_bottom = nx - comp_nx - pad_top
    compressed = np.pad(compressed, ((pad_top, pad_bottom), (0, 0)), mode="edge")

    def compress_mask(mask):
        cm = resize(
            mask.astype(float),
            (comp_nx, ny),
            anti_aliasing=False,
            preserve_range=True,
        )
        cm = cm > 0.5
        cm = np.pad(cm, ((pad_top, pad_bottom), (0, 0)), mode="edge")
        return cm

    comp_info = info.copy()
    comp_info["lesion_mask"] = compress_mask(info["lesion_mask"])
    comp_info["background_mask"] = compress_mask(info["background_mask"])
    comp_info["compressed"] = True
    comp_info["compression_factor"] = factor
    comp_info["adipose_mu"] = info["adipose_mu"]
    comp_info["gland_mu"] = info["gland_mu"]
    comp_info["lesion_mu"] = info["lesion_mu"]

    return compressed, comp_info
