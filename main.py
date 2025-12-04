import matplotlib.pyplot as plt
from phantom import create_simple_phantom
from simulate_xray import (
    simulate_projection,
    simulate_projection_angle,
    simulate_2d_projection,
)

def test_basic(phantom):
    # Show phantom (ground truth)
    plt.imshow(phantom, cmap='gray')
    plt.title("Phantom (Ground Truth)")
    plt.colorbar(label="μ")
    plt.show()

    # 1D vertical projection
    projection = simulate_projection(phantom)

    plt.plot(projection)
    plt.title("Simulated X-ray Projection (0°)")
    plt.xlabel("Detector position")
    plt.ylabel("Intensity")
    plt.show()


def test_angles(phantom):
    angles = [0, 15, 30, 45, 60]
    plt.figure(figsize=(10, 5))

    for a in angles:
        I, _ = simulate_projection_angle(phantom, a)
        plt.plot(I, label=f"{a}°")

    plt.title("Projection Profiles at Different Angles")
    plt.xlabel("Detector Position")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()


def test_2d_film(phantom):
    film = simulate_2d_projection(phantom, angle_deg=30)

    plt.imshow(film, cmap='gray')
    plt.title("Simulated 2D Film Projection at 30°")
    plt.colorbar()
    plt.show()


def main():
    phantom = create_simple_phantom()

    test_basic(phantom)
    test_angles(phantom)
    test_2d_film(phantom)


if __name__ == "__main__":
    main()
