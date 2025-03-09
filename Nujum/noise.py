import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def add_kepler_noise(time, flux, cdpp_ppm=30, transit_duration=0.5,
                     flicker_timescale=6.5, outlier_rate=0.001,
                     systematic_timescale=10, systematic_amplitude=0.0001,
                     quaternion_jumps=True, safe_modes=True):
    """
    Add realistic Kepler-like noise to simulated light curves.

    Parameters:
    -----------
    time : array-like
        Time array in days
    flux : array-like
        Normalized flux array
    cdpp_ppm : float
        Combined Differential Photometric Precision in parts per million
        Default is 30 ppm (typical for a 12th magnitude star)
    transit_duration : float
        Expected transit duration in hours, used for CDPP scaling
    flicker_timescale : float
        Stellar variability timescale in days
    outlier_rate : float
        Rate of outlier points (0-1)
    systematic_timescale : float
        Timescale for systematic trends in days
    systematic_amplitude : float
        Amplitude of systematic trends
    quaternion_jumps : bool
        Whether to add quaternion adjustment discontinuities
    safe_modes : bool
        Whether to add gaps and discontinuities similar to safe mode events

    Returns:
    --------
    noisy_flux : array-like
        Flux with realistic Kepler-like noise added
    noise_components : dict
        Dictionary of the individual noise components
    """
    # Make a copy of the input flux
    noisy_flux = np.copy(flux)
    noise_components = {}

    # 1. White noise (CDPP-like)
    cdpp_sigma = cdpp_ppm / 1e6  # Convert ppm to relative flux
    white_noise = np.random.normal(0, cdpp_sigma, len(time))
    noisy_flux += white_noise
    noise_components['white_noise'] = white_noise

    # mean_photon_count = flux * photons_per_unit  # photons_per_unit would depend on stellar magnitude
    # photon_counts = np.random.poisson(mean_photon_count)
    # noisy_flux = photon_counts / photons_per_unit

    # 2. Stellar variability (red noise) using a random walk
    if flicker_timescale > 0:
        dt = np.median(np.diff(time))
        steps = len(time)
        # Scale the amplitude by the timescale
        flicker_amp = cdpp_sigma * np.sqrt(dt / flicker_timescale) * 3

        # Generate random walk
        random_walk = np.cumsum(np.random.normal(0, flicker_amp, steps))

        # Use a Gaussian process or spline to make it smoother
        x_smooth = np.linspace(0, steps - 1, 100)
        spline = interpolate.splrep(np.arange(steps), random_walk, s=steps / 10)
        random_walk_smooth = interpolate.splev(np.linspace(0, steps - 1, steps), spline)

        # Normalize to have reasonable amplitude
        random_walk_smooth = random_walk_smooth / np.std(random_walk_smooth) * flicker_amp * 5

        noisy_flux += random_walk_smooth
        noise_components['stellar_variability'] = random_walk_smooth

    # 3. Systematic trends (long-term drifts)
    if systematic_amplitude > 0:
        # Create a combination of sines with different periods
        systematic_trend = np.zeros_like(time)
        for i in range(3):  # Use 3 components
            period = systematic_timescale * (1 + i * 0.7)  # Slightly different periods
            phase = np.random.uniform(0, 2 * np.pi)
            systematic_trend += systematic_amplitude * (i + 1) / 3 * np.sin(2 * np.pi * time / period + phase)

        noisy_flux += systematic_trend
        noise_components['systematic_trend'] = systematic_trend

    # 4. Outliers (cosmic rays, etc.)
    if outlier_rate > 0:
        n_outliers = int(outlier_rate * len(time))
        outlier_indices = np.random.choice(np.arange(len(time)), size=n_outliers, replace=False)
        outlier_values = np.random.normal(1, cdpp_sigma * 5, size=n_outliers)

        outliers = np.zeros_like(time)
        outliers[outlier_indices] = outlier_values - 1  # Subtract 1 to get the deviation

        noisy_flux[outlier_indices] = outlier_values
        noise_components['outliers'] = outliers

    # 5. Quaternion adjustment discontinuities
    if quaternion_jumps:
        # Add sudden jumps at random positions
        n_jumps = max(1, int(len(time) / 500))  # About 1 jump per 500 points
        jump_indices = np.sort(np.random.choice(np.arange(1, len(time) - 1), size=n_jumps, replace=False))

        jumps = np.zeros_like(time)
        for idx in jump_indices:
            jump_size = np.random.normal(0, cdpp_sigma * 10)
            jumps[idx:] += jump_size

        noisy_flux += jumps
        noise_components['quaternion_jumps'] = jumps

    # 6. Safe mode events (gaps and discontinuities)
    if safe_modes and len(time) > 1000:
        # Add a safe mode event (gap and discontinuity)
        safe_mode_pos = np.random.randint(len(time) // 4, 3 * len(time) // 4)
        safe_mode_width = np.random.randint(10, 50)

        safe_mode = np.zeros_like(time)
        if safe_mode_pos + safe_mode_width < len(time):
            # Add discontinuity after the gap
            jump_size = np.random.normal(0, cdpp_sigma * 20)
            safe_mode[(safe_mode_pos + safe_mode_width):] += jump_size

            noise_components['safe_mode'] = safe_mode
            noisy_flux += safe_mode

    return noisy_flux, noise_components


def plot_noise_components(time, flux, noisy_flux, noise_components):
    """
    Plot the original flux, noisy flux, and individual noise components
    """
    n_components = len(noise_components)
    fig, axes = plt.subplots(n_components + 2, 1, figsize=(18, 4 * (n_components + 2)), sharex=True)

    # Plot original flux
    axes[0].plot(time, flux, 'k-', lw=1)
    axes[0].set_title('Original Simulated Flux')
    axes[0].set_ylabel('Flux')

    # Plot noisy flux
    axes[1].plot(time, noisy_flux, 'b-', lw=1)
    axes[1].set_title('Flux with Kepler-like Noise')
    axes[1].set_ylabel('Flux')

    # Plot individual noise components
    for i, (name, component) in enumerate(noise_components.items()):
        axes[i + 2].plot(time, component, 'r-', lw=1)
        axes[i + 2].set_title(f'Noise Component: {name}')
        axes[i + 2].set_ylabel('Amplitude')

    axes[-1].set_xlabel('Time (days)')
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Generate a simple simulated light curve with a transit
    np.random.seed(42)
    days = 30
    cadence_min = 30  # Kepler long cadence
    time = np.arange(0, days, cadence_min / (60 * 24))
    flux = np.ones_like(time)

    # Add a transit
    transit_period = 3.5
    transit_duration = 0.1
    transit_depth = 0.01

    for t0 in np.arange(transit_period / 2, days, transit_period):
        in_transit = np.abs(np.mod(time - t0, transit_period)) < transit_duration / 2
        flux[in_transit] -= transit_depth

    # Add Kepler-like noise
    noisy_flux, noise_components = add_kepler_noise(
        time, flux,
        cdpp_ppm=50,  # 50 ppm for a ~12th mag star
        flicker_timescale=1.0,  # 5-day timescale for stellar variability
        outlier_rate=0.1,  # 0.1% outliers
        systematic_timescale=15,  # 15-day systematic trend
        systematic_amplitude=0,  # Low amplitude systematic trend
        quaternion_jumps=False,  # Include quaternion adjustment discontinuities
        safe_modes=True  # Include safe mode events
    )

    # Plot the results
    fig = plot_noise_components(time, flux, noisy_flux, noise_components)
    plt.show()