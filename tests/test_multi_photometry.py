
import os
import sys
import numpy as np
from pathlib import Path
import configparser
import pandas as pd
import shutil

# Add project root to sys.path
sys.path.append(os.getcwd())

from Dynamo.star import Star
from Dynamo.create_data import simulate_one, save_batch

def test_multi_photometry():
    project_root = Path(os.getcwd())
    dynamo_dir = project_root / "Dynamo"
    real_config_path = dynamo_dir / "star.conf"
    test_config_path = dynamo_dir / "test_multi_phot.conf"
    
    # Read real config
    conf = configparser.ConfigParser(inline_comment_prefixes='#')
    conf.read(real_config_path)
    
    # Modify for test
    conf.set('general', 'photometry_names', 'Kepler, TESS')
    conf.set('general', 'photometry_filters', 'Kepler.dat, TESS.dat')
    
    # Ensure spot map exists
    spotmap_path = dynamo_dir / "spotmap.dat"
    spotmap_backup = None
    if spotmap_path.exists():
        spotmap_backup = spotmap_path.read_text()
    
    with open(spotmap_path, 'w') as f:
        f.write("0.0 100.0 90.0 0.0 10.0")

    with open(test_config_path, 'w') as f:
        conf.write(f)

    test_dataset_dir = "test_dataset_multi_phot"
    if os.path.exists(test_dataset_dir):
        shutil.rmtree(test_dataset_dir)
    os.makedirs(f"{test_dataset_dir}/lc", exist_ok=True)
    os.makedirs(f"{test_dataset_dir}/spots", exist_ok=True)
    os.makedirs(f"{test_dataset_dir}/configs", exist_ok=True)

    try:
        # Initialize Star
        sm = Star(conf_file_path='test_multi_phot.conf')
        
        print(f"Photometry names: {sm.photometry_names}")
        print(f"Photometry filters: {sm.photometry_filters}")
        
        assert len(sm.photometry_names) == 2
        assert sm.photometry_names[0] == 'Kepler'
        assert sm.photometry_names[1] == 'TESS'

        # Set some dummy params
        sm.feh = 0.0
        sm.distance = 10.0
        
        # Test 1: single time array
        t_sampling = np.linspace(0, 10, 101)
        sm.compute_forward(t=t_sampling)
        
        assert 'Kepler' in sm.results['lcs']
        assert 'TESS' in sm.results['lcs']
        assert len(sm.results['lcs']['Kepler'][1]) == 101
        
        print("Single time array compute_forward successful")

        # Test 2: Dict of time arrays (different cadences)
        t_kepler = np.linspace(0, 10, 21) # 0.5 day cadence
        t_tess = np.linspace(0, 10, 101)   # 0.1 day cadence
        
        sm.compute_forward(t={'Kepler': t_kepler, 'TESS': t_tess})
        
        assert len(sm.results['lcs']['Kepler'][1]) == 21
        assert len(sm.results['lcs']['TESS'][1]) == 101
        
        # Check that Kepler times are correct
        assert np.allclose(sm.results['lcs']['Kepler'][0], t_kepler)
        
        print("Dict time array compute_forward successful")

        # Test 3: create_data.py save_batch
        res = {
            'id': 0,
            'lcs': sm.results['lcs'],
            'spectra': sm.results['spectra'],
            'spots': sm.spot_map,
            'config': {'dummy': 'data'}
        }
        
        for name, (t_instr, flux_instr) in res['lcs'].items():
            print(f"Instr: {name}, t shape: {t_instr.shape}, flux shape: {flux_instr.shape}")
            combined = np.c_[t_instr, flux_instr]
            print(f"Combined shape: {combined.shape}")
            
        save_batch([res], 0, test_dataset_dir)
        
        kepler_pqt = Path(test_dataset_dir) / "lc" / "Kepler" / "chunk_0.pqt"
        tess_pqt = Path(test_dataset_dir) / "lc" / "TESS" / "chunk_0.pqt"
        
        assert kepler_pqt.exists()
        assert tess_pqt.exists()
        
        df_k = pd.read_parquet(kepler_pqt)
        df_t = pd.read_parquet(tess_pqt)
        
        assert len(df_k) == 21
        assert len(df_t) == 101
        
        print("save_batch verification successful")

        # Test 4: create_data.py simulate_one
        import logging
        logger = logging.getLogger('test')
        sim_row = {
            'mass': 1.0, 'age': 5.0, 'FeH': 0.0, 'alpha/H': 0.0,
            'Teff': 5800, 'logg': 4.4, 'L': 1.0, 'R': 1.0,
            'Inclination': 90.0, 'Period': 25.0, 'Shear': 0.0,
            'Distance': 100.0, 'convective_shift': 1.0,
            'Activity Rate': 0.1, 'Cycle Length': 10.0, 'Cycle Overlap': 0.5,
            'Spot Min': 20.0, 'Spot Max': 70.0, 'Decay Time': 5.0,
            'Butterfly': True, 'CDPP': 100.0, 'Outlier Rate': 0.0,
            'Flicker Time Scale': 1.0, 'simulate_planet': False
        }
        # Add planet keys
        for k in ['planet_period', 'planet_radius', 'planet_esinw', 'planet_ecosw', 
                  'planet_impact', 'planet_spin_orbit_angle', 'planet_transit_t0', 'planet_semi_amplitude']:
            sim_row[k] = 0.0
            
        res_sim = simulate_one(models_root=sm.models_root, sim_row=sim_row, sim_dir=".", 
                               idx=1, logger=logger, ndays=10, plot_every=np.inf)
        
        lcs_sim = res_sim['lcs']
        assert 'Kepler' in lcs_sim
        assert 'TESS' in lcs_sim
        
        # Kepler cadence in config was ~0.02, TESS ~0.001
        # In test_multi_phot.conf (copied from starsim.conf) it should be:
        # photometry_cadences : 0.020833, 0.001389
        # ndays=10
        # n_k = 10 / 0.02 = ~500
        # n_t = 10 / 0.0013 = ~7600
        
        print(f"Kepler points: {len(lcs_sim['Kepler'][0])}")
        print(f"TESS points: {len(lcs_sim['TESS'][0])}")
        
        assert len(lcs_sim['Kepler'][0]) > 400
        assert len(lcs_sim['TESS'][0]) > 7000
        
        print("simulate_one multi-cadence verification successful")

        # Test 5: Plot comparison
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 8))
        
        tk, fk = lcs_sim['Kepler']
        tt, ft = lcs_sim['TESS']
        
        # Plot TESS as line (high res)
        plt.plot(tt, ft, label='TESS (2 min)', alpha=0.3, color='blue', linewidth=1)
        # Plot Kepler as dots/line (lower res)
        plt.plot(tk, fk, 'o-', label='Kepler (30 min)', color='red', markersize=4, alpha=0.7)
        
        plt.xlim(0, 2) # Zoom into first 2 days
        plt.xlabel('Time [days]')
        plt.ylabel('Normalized Flux')
        plt.title('Cadence Comparison: Kepler (30m) vs TESS (2m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = "tests/images/cadence_comparison.png"
        plt.savefig(plot_path)
        print(f"Comparison plot saved to {plot_path}")

    finally:
        # Cleanup
        if test_config_path.exists():
            os.remove(test_config_path)
        if spotmap_backup is not None:
            with open(spotmap_path, 'w') as f:
                f.write(spotmap_backup)
        if os.path.exists(test_dataset_dir):
            shutil.rmtree(test_dataset_dir)

if __name__ == "__main__":
    try:
        test_multi_photometry()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
