import numpy as np
import matplotlib.pyplot as plt

from phantom import create_simple_phantom
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
    phantom = create_simple_phantom()

    test_basic(phantom)
    test_angles(phantom)
    test_sinogram(phantom)


if __name__ == "__main__":
    main()
