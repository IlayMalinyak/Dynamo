import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from Dynamo.star import Star
import Dynamo.spectra as spectra

def check_flux_consistency():
    # Setup star
    s = Star()
    s.n_grid_rings = 10
    s.radius = 1.0
    s.rotation_period = 25.0
    s.inclination = 0.0 # vsini = 0
    s.distance = 10.0
    
    wv = np.linspace(5000, 5100, 1000)
    
    # Mock Phoenix data: constant intensity 1.0 for all mu
    sini = np.linspace(0.1, 1.0, 10)
    photo_flux = np.ones((len(sini), len(wv)))
    spot_flux = np.zeros_like(photo_flux)
    
    # Get mu for integration
    Ngrid_in_ring, cos_centers, proj_area, phi, theta, vec_grid = s.get_theta_phi()
    
    # 1. Old Method (create_observed_spectra)
    f_old, _ = spectra.create_observed_spectra(
        s, wv, photo_flux, spot_flux, sini, ff_sp=0.0,
        dist=10.0, rad=1.0, flux_scale=1.0,
        instrument_resolution=1000, wavelength_range=(5000, 5100)
    )
    print(f"Old method flux level (const I=1): {np.mean(f_old):.6e}")
    
    # 2. New Method (create_observed_spectra_kernel)
    N_pixels = sum(Ngrid_in_ring)
    coverage = np.zeros((N_pixels, 4))
    coverage[:, 0] = 1.0 # 100% photosphere
    
    f_new, _ = spectra.create_observed_spectra_kernel(
        s, wv, photo_flux, spot_flux, sini, coverage,
        dist=10.0, rad=1.0, flux_scale=1.0,
        instrument_resolution=1000, wavelength_range=(5000, 5100)
    )
    print(f"New method flux level (const I=1): {np.mean(f_new):.6e}")
    
    # We want f_new to match f_old
    ratio = np.mean(f_new) / np.mean(f_old) if np.mean(f_old) != 0 else 0
    print(f"Ratio New/Old: {ratio:.6f}")

if __name__ == "__main__":
    check_flux_consistency()
