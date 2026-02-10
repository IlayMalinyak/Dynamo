
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import configparser
import logging

# Add project root to sys.path
sys.path.append(os.getcwd())

from Dynamo.star import Star
from Dynamo.create_data import simulate_one

def test_visualize_observables_multi():
    print("Setting up simulation...")
    os.makedirs("images", exist_ok=True)
    
    # 1. Define 3 sets of random-ish parameters for variety
    samples = [
        {'Teff': 5800, 'logg': 4.44, 'Period': 25.0, 'Activity': 0.1, 'Inclination': 60.0, 'name': 'Sun-like'},
        {'Teff': 4800, 'logg': 4.5, 'Period': 7.0, 'Activity': 0.8, 'Inclination': 80.0, 'name': 'Active K-dwarf'},
        {'Teff': 3800, 'logg': 4.8, 'Period': 2.5, 'Activity': 1.5, 'Inclination': 45.0, 'name': 'Ultra-active M-dwarf'}
    ]
    
    fig, axes = plt.subplots(len(samples), 3, figsize=(24, 6 * len(samples)))
    
    for i, params in enumerate(samples):
        print(f"Processing Sample {i+1}: {params['name']}...")
        try:
            # Initialize Star
            sm = Star(conf_file_path='star.conf')
            sm.feh = 0.0 # Use default 0.0
            sm.distance = 10.0
            sm.activity = params['Activity']
            sm.rotation_period = params['Period']
            sm.temperature_photosphere = params['Teff']
            sm.logg = params['logg']
            
            # Use smaller contrast for cool stars to stay within model grid (minimal 3000K usually)
            sm.spot_T_contrast = 400.0 if params['Teff'] < 4000 else 1000.0
            
            sm.inclination = params['Inclination']
            
            # Add missing parameters for generate_spot_map
            sm.cycle_len = 11.0 # years
            sm.cycle_overlap = 0.5
            sm.spot_max_lat = 40.0
            sm.spot_min_lat = 0.0
            sm.butterfly = True
            
            # Generate MANY spots
            print(f"  Generating spots (Activity={params['Activity']})...")
            sm.generate_spot_map(ndays=1000) # 1000 days to get many spots
            
            # Compute forward
            t = np.linspace(0, 50, 500)
            sm.compute_forward(t=t)
            
            # --- Column 1: Latitude Distribution ---
            ax_dist = axes[i, 0]
            # Spot latitudes (90 - colat)
            lats = 90 - sm.spot_map[:, 2]
            areas = sm.spot_map[:, 4]**2 # Radius squared
            
            # 1D Histogram of spot latitudes
            ax_dist.hist(lats, bins=np.linspace(-90, 90, 31), weights=areas, color='orange', alpha=0.7, edgecolor='black')
            ax_dist.set_xlabel('Latitude [deg]')
            ax_dist.set_ylabel('Total Spot Area')
            ax_dist.set_title(f'Lat Distribution (Activity={params["Activity"]})')
            ax_dist.set_xlim(-90, 90)
            ax_dist.grid(True, alpha=0.3)
            ax_dist.axvline(0, color='black', linewidth=1, linestyle='--')
            
            # --- Column 2: Light Curve ---
            ax_lc = axes[i, 1]
            if 'lcs' in sm.results:
                for instr, (t_ins, f_ins) in sm.results['lcs'].items():
                    ax_lc.plot(t_ins, f_ins, label=instr)
            ax_lc.set_xlabel('Time [days]')
            ax_lc.set_ylabel('Flux')
            ax_lc.set_title(f'Light Curves (Prot={params["Period"]}d)')
            ax_lc.legend()
            ax_lc.grid(True, alpha=0.3)
            
            # --- Column 3: Spectra ---
            ax_spec = axes[i, 2]
            if 'spectra' in sm.results:
                for instr, (wv, flx) in sm.results['spectra'].items():
                    if flx.ndim == 2:
                        ax_spec.plot(wv, flx[0], label=f'{instr} early', alpha=0.7)
                        # ax_spec.plot(wv, flx[-1], label=f'{instr} late', alpha=0.5, linestyle='--')
                    else:
                        ax_spec.plot(wv, flx, label=instr)
            ax_spec.set_xlabel('Wavelength [A]')
            ax_spec.set_ylabel('Flux')
            ax_spec.set_title(f'Spectra (Teff={params["Teff"]}K)')
            ax_spec.legend(prop={'size': 8})
            ax_spec.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            axes[i, 1].text(0.5, 0.5, f"Error: {e}", transform=axes[i, 1].transAxes, ha='center')

    plt.tight_layout()
    plt.suptitle("Multi-Sample Observables vs Spot Distribution", fontsize=24, y=1.02)
    plot_path = "tests/images/multi_sample_viz.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"DONE! Multi-sample visualization saved to {plot_path}")

if __name__ == "__main__":
    test_visualize_observables_multi()
