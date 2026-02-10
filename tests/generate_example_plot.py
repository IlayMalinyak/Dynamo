
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import sys

# Add parent directory to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Dynamo.star import Star
from Dynamo.create_data import generate_simdata

def generate_example_plot():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Directory setup
    dataset_dir = os.path.join(current_dir, "dataset_example")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Generate 1 simulation parameter set. 
    logger.info("Generating simulation parameters...")
    # Nlc=1 for a single example
    sims = generate_simdata(dataset_dir, Nlc=1, logger=logger, add_noise=True, sim_name="example_sim")
    
    # Get the first (and only) row
    sim_row = sims.iloc[0]
    
    # Initialize Star
    logger.info("Initializing Star object...")
    # Star will look for star.conf in its own directory by default
    # or we can pass the path to Dynamo/star.conf relative to CWD if running from root
    # or absolute path. 
    # Let's rely on Star finding it in Dynamo/Dynamo/star.conf if we don't pass anything,
    # OR we pass the absolute path to be sure.
    conf_path = os.path.join(parent_dir, 'Dynamo', 'star.conf')
    if not os.path.exists(conf_path):
         # Try default import location
         conf_path = 'star.conf'
        
    sm = Star(conf_file_path=conf_path)
    
    # Set parameters
    sm.set_stellar_parameters(sim_row)
    sm.set_planet_parameters(sim_row)
    
    # Generate spots
    logger.info("Generating spots...")
    ndays = 100 # Short simulation for example
    sm.generate_spot_map(ndays=ndays)
    
    # Setup time arrays for photometry instruments (Kepler, TESS)
    # create_data.py logic:
    t_dict = {}
    if not sm.photometry_names:
        logger.warning("No photometry instruments configured in star.conf")
        
    for name, cad in zip(sm.photometry_names, sm.photometry_cadences):
        n_points = int(ndays / cad)
        # Time array from 0 to ndays
        t_dict[name] = np.linspace(0, ndays, n_points)
        
    # Run simulation
    logger.info("Running simulation (compute_forward)...")
    sm.compute_forward(t=t_dict)
    
    # Extract results
    lcs = sm.results.get('lcs', {})
    spectra = sm.results.get('spectra', {})
    
    # Plotting
    logger.info("Plotting results...")
    
    # Create figure with 2 subplots (vertical stack)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Subplot 1: Lightcurves
    colors = ['black', 'red']  # Cycle through these if needed
    for i, (name, (t, flux)) in enumerate(lcs.items()):
        # Normalize to median for visual comparison (as flux levels differ)
        norm_flux = flux / np.median(flux)
        
        c = colors[i % len(colors)]
        
        ax1.plot(t, norm_flux, label=name, alpha=0.4, color=c)
        
    ax1.set_ylabel("Normalized Flux")
    ax1.set_xlabel("Time (days)")
    ax1.legend(fontsize='small')
    # No title
    
    # Subplot 2: Spectra (APOGEE, LAMOST)
    # We expect 'APOGEE' and 'LAMOST' in spectra keys if configured correctly.
    
    for name, data in spectra.items():
        # data structure from create_data.py / star.py: (wavelength, flux) or (wavelength, flux_array)
        wv = data[0]
        flx = data[1]
        
        # If flux is 2D (epochs), take the first epoch
        if flx.ndim == 2:
            flx = flx[0]
            
        
        ax2.plot(wv, flx, label=name, alpha=0.8)
        
    ax2.set_ylabel("Flux")
    ax2.set_xlabel("Wavelength ($\\AA$)")
    ax2.legend(fontsize='small')
    # No title
    
    plt.tight_layout()
    
    assets_dir = os.path.join(parent_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    out_file = os.path.join(assets_dir, "example_plot.png")
    plt.savefig(out_file, dpi=150)
    logger.info(f"Saved plot to {out_file}")

if __name__ == "__main__":
    generate_example_plot()
