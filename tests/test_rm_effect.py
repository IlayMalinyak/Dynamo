import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from Dynamo.star import Star
import Dynamo.spectra as spectra

def test_rm_effect():
    print("Simulating Rossiter-McLaughlin effect...")
    
    # 1. Setup Star
    s = Star()
    # High resolution grid for better RM
    s.n_grid_rings = 20 
    s.radius = 1.0
    s.rotation_period = 2.0 # Fast rotation for large vsini
    s.inclination = 90.0   # Equator-on
    s.distance = 10.0
    
    # vsini should be ~25 km/s if 2.0d period
    print(f"vsini: {s.vsini:.2f} km/s")
    
    # 2. Setup Wavelength and intensities
    wv = np.linspace(5000, 5010, 2000) # High res lambda
    
    # Create a spectral line (Gaussian)
    def spectral_line(w, center, depth, sigma):
        return 1.0 - depth * np.exp(-0.5 * ((w - center) / sigma)**2)
    
    sini = np.linspace(0.1, 1.0, 10)
    photo_flux = np.zeros((len(sini), len(wv)))
    for i, mu in enumerate(sini):
        # Add limb darkening (I proportional to mu)
        photo_flux[i] = mu * spectral_line(wv, 5005.0, 0.5, 0.1)
    
    spot_flux = photo_flux * 0.5 # Dimmer spots
    
    # 3. Setup Planet Transit
    # We want a mid-transit snapshot (impact param b=0)
    # Planet at center blocks the zero-velocity part of the star.
    # Planet at side blocks red/blue shifted part.
    
    Ngrid_in_ring, cos_centers, proj_area, phi, theta, vec_grid = s.get_theta_phi()
    N_pixels = sum(Ngrid_in_ring)
    
    # We'll simulate 3 positions:
    # 1. Out of transit
    # 2. Approaching limb (blueshifted blocked) -> Star looks redshifted
    # 3. Center (v=0 blocked) -> Star looks broadened/distorted
    # 4. Receding limb (redshifted blocked) -> Star looks blueshifted
    
    positions = [
        None,             # Out
        (-0.5, 0.0, 0.1), # Blue limb (x, y, r)
        (0.0, 0.0, 0.1),  # Center
        (0.5, 0.0, 0.1)   # Red limb
    ]
    labels = ["Out", "Blue Blocked", "Center Blocked", "Red Blocked"]
    
    plt.figure(figsize=(10, 8))
    
    base_spectrum = None
    
    # Geometry for manual planet blocking (since compute_planet_pos is time-based)
    # Actually, we can use calculate_circular_coverage or similar.
    # But for a simple test, let's just do it manually.
    
    # generate_grid_coordinates_nb returns ys, zs (on-sky)
    _, _, _, _, _, _, xs_grid, ys_grid, zs_grid, _, _ = spectra.nbspectra.generate_grid_coordinates_nb(s.n_grid_rings)
    
    # Geometry for manual planet blocking
    _, _, _, _, _, _, xs_grid, ys_grid, zs_grid, _, _ = spectra.nbspectra.generate_grid_coordinates_nb(s.n_grid_rings)
    
    velocities = []
    c_light = 299792.458 # km/s
    
    # Define a range of positions across the star
    transit_x = np.linspace(-1.2, 1.2, 20)
    
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    
    reference_flux = None
    
    for px in transit_x:
        coverage = np.zeros((N_pixels, 4))
        coverage[:, 0] = 1.0 # 100% photosphere
        
        # Planet parameters
        pr = 0.1
        py = 0.0
        
        # Check if planet is transiting
        dist = np.sqrt((ys_grid - px)**2 + (zs_grid - py)**2)
        blocked = dist < pr
        
        is_transiting = np.any(blocked)
        if is_transiting:
            coverage[blocked, 3] = 1.0 # Planet
            coverage[blocked, 0] = 0.0 # No photosphere
             
        flux, _ = spectra.create_observed_spectra_kernel(
            s, wv, photo_flux, spot_flux, sini, coverage,
            dist=10.0, rad=1.0, flux_scale=1.0, resample=False
        )
        
        if reference_flux is None:
             reference_flux = flux
             # Plot reference
             plt.plot(wv, flux, 'k-', label="Out of Transit", lw=2, alpha=0.5)
             
        # Normalize flux to continuum 1.0 to avoid scaling effects
        norm_flux = flux / np.max(flux)
        
        # Use Line Centroid method
        # V = c * (lambda_cen - lambda_0) / lambda_0
        # lambda_cen = Integ(lambda * (Ic - I)) / Integ(Ic - I)
        
        # Estimate continuum from the edge of the window
        continuum = np.mean(norm_flux[:10]) 
        
        # Line profile (depth)
        depth = continuum - norm_flux
        
        # Only use the part of the line where depth is significant to reduce noise
        mask = depth > 0.05 * np.max(depth)
        
        if np.sum(mask) > 0:
            centroid_lambda = np.sum(wv[mask] * depth[mask]) / np.sum(depth[mask])
            rv = c_light * (centroid_lambda - 5005.0) / 5005.0
            velocities.append(rv)
            print(f"Px: {px:.2f}, Centroid: {centroid_lambda:.5f} A, RV: {rv:.4f} km/s")
        else:
            velocities.append(0.0)
            print(f"Px: {px:.2f}, RV: 0.0000 km/s (Mask empty)")
        
        if reference_flux is None:
             reference_flux = flux
             # Plot reference
             plt.plot(wv, flux, 'k-', label="Out of Transit", lw=2, alpha=0.5)

        # ... (normalization and centroid calculation code) ...

        # Plot profiles at specific positions: Max Blueshift, Center, Max Redshift
        # We know transit is roughly -0.1 to 0.1 for center, and limbs are +/- 1.0
        # Let's plot indices 5, 10, 15 from the 20 steps
        # indices 5 (~ -0.57), 10 (0.06), 15 (0.69)
        
        # Or just plot based on checking the list index
        idx_px = np.where(transit_x == px)[0][0]
        if idx_px in [5, 10, 15] and is_transiting:
             label = f"Pos {px:.2f} (RV={rv:.1f})"
             plt.plot(wv, flux, label=label)

    plt.title("Spectral Lines during Transit")
    plt.ylabel("Flux")
    plt.legend()
    
    # Calculate theoretical RM curve shape (qualitative)
    # v_anomaly = - v_subplanet * (flux_blocked / total_flux)
    
    plt.subplot(2, 1, 2)
    plt.plot(transit_x, velocities, 'o-', color='red')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(-1.0, color='gray', linestyle=':')
    plt.axvline(1.0, color='gray', linestyle=':')
    plt.title("Apparent Radial Velocity (Rossiter-McLaughlin Effect)")
    plt.ylabel("RV [km/s]")
    plt.xlabel("Planet Position (Stellar Radii)")
    
        
   
    plt.tight_layout()
    plt.savefig("tests/images/rm_effect_detailed.png")
    print("Plot saved to tests/images/rm_effect_detailed.png")

if __name__ == "__main__":
    test_rm_effect()
