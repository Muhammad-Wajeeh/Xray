"""
Generate required figures and ROI stats for the project deliverables.
Outputs saved in figs/.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from phantom import create_breast_phantom, create_shepp_logan
from simulate_xray import (
    simulate_xray_2d,
    simulate_projection,
    simulate_projection_angle,
    simulate_sinogram,
)
from utils import roi_mean_std, roi_contrast

FIG_DIR = Path("figs")
FIG_DIR.mkdir(exist_ok=True)


def save_fig(name):
    """Save current Matplotlib figure to figs/name (png)."""
    path = FIG_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=200)
    print(f"saved {path}")


def baseline_and_variations():
    """Generate baseline radiograph and parameter variation figures; print ROI stats."""
    phantom, info = create_breast_phantom()
    params = dict(sid=500, sdd=1000, kVp=35, exposure_time=1.0, filtration_mmAl=2.0, grid_ratio=0.9)

    plt.figure(figsize=(6, 6))
    plt.imshow(phantom, cmap="magma")
    plt.title("Phantom μ map (ground truth)")
    plt.axis("off")
    save_fig("phantom_ground_truth.png")
    plt.close()

    img0 = simulate_xray_2d(phantom, angle_deg=0, **params)
    plt.figure(figsize=(6, 6))
    plt.imshow(img0, cmap="gray", vmin=0, vmax=0.7)
    plt.title("Baseline Radiograph (0°)")
    plt.axis("off")
    save_fig("baseline_radiograph.png")
    plt.close()

    img_sid_near = simulate_xray_2d(phantom, angle_deg=0, sid=350, sdd=1000, **{k: v for k, v in params.items() if k not in ["sid", "sdd"]})
    img_sid_far = simulate_xray_2d(phantom, angle_deg=0, sid=700, sdd=1000, **{k: v for k, v in params.items() if k not in ["sid", "sdd"]})
    plt.figure(figsize=(12, 4))
    for i, (im, title) in enumerate([
        (img0, "Baseline (SID 500)"),
        (img_sid_near, "Closer SID 350"),
        (img_sid_far, "Farther SID 700"),
    ]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(im, cmap="gray", vmin=0, vmax=0.7)
        plt.title(title)
        plt.axis("off")
    save_fig("distance_variation.png")
    plt.close()

    dense_phantom = phantom * 1.2
    img_dense = simulate_xray_2d(dense_phantom, angle_deg=0, **params)
    plt.figure(figsize=(8, 4))
    for i, (im, title) in enumerate([
        (img0, "Baseline μ"),
        (img_dense, "Higher μ (denser)"),
    ]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(im, cmap="gray", vmin=0, vmax=0.7)
        plt.title(title)
        plt.axis("off")
    save_fig("mu_variation.png")
    plt.close()

    angles = [0, 15, 30]
    plt.figure(figsize=(12, 4))
    for i, a in enumerate(angles):
        img = simulate_xray_2d(phantom, angle_deg=a, **params)
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=0.7)
        plt.title(f"Angle {a}°")
        plt.axis("off")
    save_fig("angle_variation.png")
    plt.close()

    baseline_profile = simulate_projection(phantom, I0=1.0, **params)
    dist_near = simulate_projection(phantom, I0=1.0, sid=350, sdd=1000, **{k: v for k, v in params.items() if k not in ["sid", "sdd"]})
    dense_profile = simulate_projection(dense_phantom, I0=1.0, **params)
    angle_profile, _ = simulate_projection_angle(phantom, angle_deg=20, I0=1.0, **params)

    x = np.arange(baseline_profile.size)
    plt.figure(figsize=(8, 4))
    plt.plot(x, baseline_profile, label="Baseline", linewidth=2)
    plt.plot(x, dist_near, label="Closer SID 350", linestyle="--")
    plt.plot(x, dense_profile, label="Higher μ", linestyle="-.")
    plt.plot(x, angle_profile, label="Angle 20°", linestyle=":")
    plt.xlabel("Detector position (px)")
    plt.ylabel("Intensity")
    plt.title("Profile overlays (distance, μ, angle)")
    plt.legend()
    plt.grid(alpha=0.2)
    save_fig("profile_overlays.png")
    plt.close()

    phantom_comp, info_comp = create_breast_phantom(compression=True)
    profile_comp = simulate_projection(phantom_comp, I0=1.0, **params)
    plt.figure(figsize=(8, 4))
    plt.plot(x, baseline_profile, label="Baseline", linewidth=2)
    plt.plot(x, profile_comp, label="Compressed", linestyle="--")
    plt.xlabel("Detector position (px)")
    plt.ylabel("Intensity")
    plt.title("Profile: baseline vs compressed")
    plt.legend()
    plt.grid(alpha=0.2)
    save_fig("profile_compressed.png")
    plt.close()

    def roi_stats(p, info_local, label):
        lesion_mean, lesion_std = roi_mean_std(p, info_local["lesion_mask"])
        bg_mean, bg_std = roi_mean_std(p, info_local["background_mask"])
        contrast = roi_contrast(lesion_mean, bg_mean)
        print(f"{label}: lesion {lesion_mean:.3f}±{lesion_std:.3f}, bg {bg_mean:.3f}±{bg_std:.3f}, contrast {contrast:.2f}")
        return lesion_mean, lesion_std, bg_mean, bg_std, contrast

    roi_stats(phantom, info, "Baseline phantom μ")
    roi_stats(phantom_comp, info_comp, "Compressed phantom μ")


def sinogram_and_schematic():
    """Generate sinogram and phantom schematic figures."""
    phantom, _ = create_breast_phantom()
    angles = np.linspace(0, 180, 181)
    sino, _ = simulate_sinogram(phantom, max_angle=180, sid=500, sdd=1000, kVp=35, exposure=1.0, filtration=2.0, grid_ratio=0.9)
    plt.figure(figsize=(8, 4))
    plt.imshow(sino, cmap="gray", aspect="auto")
    plt.xlabel("Detector position")
    plt.ylabel("Angle (deg)")
    plt.title("Sinogram (0-180°)")
    save_fig("sinogram.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(phantom, cmap="magma")
    plt.title("Phantom μ map (ground truth)")
    plt.axis("off")
    save_fig("phantom_ground_truth.png")
    plt.close()


def main():
    """Entry point: build all figures."""
    baseline_and_variations()
    sinogram_and_schematic()


if __name__ == "__main__":
    main()
