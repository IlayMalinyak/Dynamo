import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def kroupa_imf(m):
    """
    Kroupa Initial Mass Function

    Parameters:
    m : array-like
        Stellar masses in solar mass units

    Returns:
    xi : array-like
        Number of stars per mass interval
    """
    # Define the exponents for different mass ranges
    alpha_1 = 1.3  # For m in [0.08, 0.5) solar masses
    alpha_2 = 2.3  # For m in [0.5, infinity) solar masses

    # Initialize the output array
    xi = np.zeros_like(m, dtype=float)

    # Calculate IMF for different mass ranges
    # Note: We implement both ranges even though our focus is 0.3-2 solar masses
    mask_low = (m >= 0.08) & (m < 0.5)
    mask_high = m >= 0.5

    # Set the normalization at 0.5 solar masses
    m_0 = 0.5

    # Calculate IMF values
    xi[mask_low] = m[mask_low] ** (-alpha_1)
    xi[mask_high] = m_0 ** (alpha_2 - alpha_1) * m[mask_high] ** (-alpha_2)

    return xi


# Create a function to sample from this distribution
def sample_kroupa_imf(n_samples, m_min=0.3, m_max=2.0):
    """
    Generate random samples from the Kroupa IMF

    Parameters:
    n_samples : int
        Number of samples to generate
    m_min : float
        Minimum mass in solar masses
    m_max : float
        Maximum mass in solar masses

    Returns:
    masses : ndarray
        Array of sampled masses
    """
    # Create a fine grid for numerical integration and interpolation
    m_grid = np.logspace(np.log10(m_min), np.log10(m_max), 1000)
    imf_values = kroupa_imf(m_grid)

    # Create cumulative distribution function
    cdf = np.cumsum(imf_values)
    cdf = cdf / cdf[-1]  # Normalize

    # Draw random samples from uniform distribution
    u = np.random.random(n_samples)

    # Interpolate to find the corresponding masses
    masses = np.interp(u, cdf, m_grid)

    return masses

if __name__ == '__main__':
    # Generate a sample of stellar masses
    np.random.seed(42)  # For reproducibility
    n_stars = 50000
    stellar_masses = sample_kroupa_imf(n_stars)

    # Plot the distribution
    plt.figure(figsize=(10, 6))

    # Histogram of the sampled masses
    plt.hist(stellar_masses, bins=50, density=True, alpha=0.7, label='Sampled masses')

    # Plot the theoretical distribution for comparison
    m_plot = np.linspace(0.3, 2.0, 1000)
    imf_plot = kroupa_imf(m_plot)

    # Use trapezoid rule for integration instead of simpson
    area = np.trapz(imf_plot, m_plot)
    imf_norm = imf_plot / area

    plt.plot(m_plot, imf_norm, 'r-', lw=2, label='Kroupa IMF')

    plt.xlabel('Mass (Solar Masses)')
    plt.ylabel('Probability Density')
    plt.title('Kroupa Initial Mass Function: 0.3-2.0 Solar Masses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Number of stars: {n_stars}")
    print(f"Mean mass: {np.mean(stellar_masses):.2f} solar masses")
    print(f"Median mass: {np.median(stellar_masses):.2f} solar masses")
    print(f"Mass range: {np.min(stellar_masses):.2f} - {np.max(stellar_masses):.2f} solar masses")

    # Let's check the distribution of stars in different mass bins
    bins = [0.3, 0.5, 1.0, 1.5, 2.0]
    hist, _ = np.histogram(stellar_masses, bins=bins)
    print("\nDistribution of stars by mass range:")
    for i in range(len(bins) - 1):
        print(f"{bins[i]}-{bins[i + 1]} solar masses: {hist[i]} stars ({hist[i] / n_stars * 100:.1f}%)")