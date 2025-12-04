import matplotlib.pyplot as plt
from phantom import create_simple_phantom
from simulate_xray import simulate_projection

def main():
    phantom = create_simple_phantom()
    
    plt.imshow(phantom, cmap='gray')
    plt.title("Phantom (Ground Truth)")
    plt.colorbar(label="μ")
    plt.show()

    projection = simulate_projection(phantom)

    plt.plot(projection)
    plt.title("Simulated X-ray Projection (1D Film)")
    plt.xlabel("Detector position")
    plt.ylabel("Intensity")
    plt.show()

if __name__ == "__main__":
    main()
