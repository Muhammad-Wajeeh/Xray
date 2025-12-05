import numpy as np
import matplotlib.pyplot as plt

from phantom import create_shepp_logan
from simulate_xray import (
    simulate_projection,
    simulate_projection_angle,
    simulate_2d_projection
)


def test_basic(phantom):
    # Show phantom
    plt.figure()
    plt.imshow(phantom, cmap='gray')
    plt.title("Phantom (Ground Truth)")
    plt.colorbar(label="μ")
    plt.show()

    # 1D vertical projection at 0°
    projection = simulate_projection(
        phantom,
        I0=1.0,
        sid=500,
        sdd=1000,
        kVp=30,
        exposure_time=1.0,
        filtration_mmAl=2.0,
    )

    plt.figure()
    plt.plot(projection)
    plt.title("1D X-ray Projection at 0°")
    plt.xlabel("Detector Position")
    plt.ylabel("Intensity")
    plt.show()


def plot_profile_overlays(phantom):
    """
    Overlay baseline profile with distance, attenuation, and angle variations.
    Shows how edges/magnification shift when parameters change.
    """
    # Baseline settings
    base_sid = 500
    base_sdd = 1000
    base_kvp = 30
    base_exposure = 1.0
    base_filt = 2.0

    x = np.arange(phantom.shape[1])
    baseline = simulate_projection(
        phantom,
        I0=1.0,
        sid=base_sid,
        sdd=base_sdd,
        kVp=base_kvp,
        exposure_time=base_exposure,
        filtration_mmAl=base_filt,
    )

    # Distance variation: move source closer (more magnification)
    closer_sid = 350
    closer_sdd = 1000
    dist_var = simulate_projection(
        phantom,
        I0=1.0,
        sid=closer_sid,
        sdd=closer_sdd,
        kVp=base_kvp,
        exposure_time=base_exposure,
        filtration_mmAl=base_filt,
    )

    # Attenuation coefficient variation: boost μ to mimic denser material
    dense_phantom = phantom * 1.25
    att_var = simulate_projection(
        dense_phantom,
        I0=1.0,
        sid=base_sid,
        sdd=base_sdd,
        kVp=base_kvp,
        exposure_time=base_exposure,
        filtration_mmAl=base_filt,
    )

    # Angle variation: tilt object
    angle_deg = 20
    angle_var, _ = simulate_projection_angle(
        phantom,
        angle_deg,
        I0=1.0,
        sid=base_sid,
        sdd=base_sdd,
        kVp=base_kvp,
        exposure_time=base_exposure,
        filtration_mmAl=base_filt,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(x, baseline, label="Baseline (0°, SID 500/SDD 1000)", linewidth=2)
    plt.plot(x, dist_var, label="Closer source (SID 350 → more magnification)", linestyle="--")
    plt.plot(x, att_var, label="Higher μ (denser material)", linestyle="-.")
    plt.plot(x, angle_var, label=f"Tilted object ({angle_deg}°)", linestyle=":")

    plt.title("Intensity Profile Overlays (Signal vs Position)")
    plt.xlabel("Detector Position (pixels)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.2)

    note = (
        "Notes:\n"
        "• Smaller SID => larger magnification, edges spread outward.\n"
        "• Higher μ deepens valleys/peaks (stronger attenuation).\n"
        "• Tilting shifts edge positions due to foreshortening."
    )
    plt.gcf().text(0.62, 0.4, note, fontsize=9,
                   bbox=dict(facecolor="white", alpha=0.7, edgecolor="0.8"))
    plt.tight_layout()
    plt.show()


def test_angles(phantom):
    angles = [0, 15, 30, 45, 60]
    plt.figure(figsize=(10, 5))

    for a in angles:
        I, _ = simulate_projection_angle(
            phantom,
            a,
            I0=1.0,
            sid=500,
            sdd=1000,
            kVp=30,
            exposure_time=1.0,
            filtration_mmAl=2.0
        )
        plt.plot(I, label=f"{a}°")

    plt.title("Projection Profiles at Different Angles")
    plt.xlabel("Detector Position")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()


def test_sinogram(phantom):
    angles = np.linspace(0, 180, 181)   # 0° → 180° in 1° steps
    sinogram = simulate_2d_projection(
        phantom,
        angles,
    )

    plt.figure()
    plt.imshow(sinogram, cmap='gray', aspect='auto')
    plt.title("2D Projection (Sinogram)")
    plt.xlabel("Detector Position")
    plt.ylabel("Angle (degrees)")
    plt.colorbar()
    plt.show()


def main():
    phantom = create_shepp_logan()

    test_basic(phantom)
    plot_profile_overlays(phantom)
    test_angles(phantom)
    test_sinogram(phantom)


if __name__ == "__main__":
    main()
