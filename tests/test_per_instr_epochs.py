
import os
import sys
import numpy as np
from pathlib import Path
import configparser

# Add project root to sys.path
sys.path.append(os.getcwd())

from Dynamo.star import Star

def test_per_instrument_epochs():
    project_root = Path(os.getcwd())
    dynamo_dir = project_root / "Dynamo"
    real_config_path = dynamo_dir / "star.conf"
    test_config_path = dynamo_dir / "test_spectra.conf"
    
    # Read real config
    conf = configparser.ConfigParser(inline_comment_prefixes='#')
    conf.read(real_config_path)
    
    # Modify for test
    conf.set('general', 'spectra_names', 'INST1, INST2')
    conf.set('general', 'spectra_resolutions', '1000, 5000')
    conf.set('general', 'spectra_filters', 'Raw, Raw')
    conf.set('general', 'spectra_ranges', '4000-5000, 15000-16000')
    conf.set('general', 'n_spectra_epochs', '2, 4')
    conf.set('general', 'spectra_cadence', '1.0, 0.5')
    conf.set('general', 'spectra_start_time', '0.0, 1.0')
    
    # Ensure spot map exists with 5 columns
    # tini, dur, colat, longi, Rcoef_0 (radius)
    spotmap_path = dynamo_dir / "spotmap.dat"
    spotmap_backup = None
    if spotmap_path.exists():
        spotmap_backup = spotmap_path.read_text()
    
    with open(spotmap_path, 'w') as f:
        # One spot from t=0 to 100, at equator (colat=90), long=0, radius=10 deg
        f.write("0.0 100.0 90.0 0.0 10.0")

    with open(test_config_path, 'w') as f:
        conf.write(f)

    try:
        # Initialize Star
        sm = Star(conf_file_path='test_spectra.conf')
        
        print(f"Parsed n_epochs: {sm.n_spectra_epochs_list}")
        print(f"Parsed cadence: {sm.spectra_cadence_list}")
        print(f"Parsed start_time: {sm.spectra_start_time_list}")

        # Set some dummy params needed for compute_forward
        sm.feh = 0.0
        sm.distance = 10.0
        
        # Run compute_forward
        t_sampling = np.linspace(0, 10, 101) # 0 to 10 days, 0.1 day step
        sm.compute_forward(t=t_sampling)

        # Check indices/times
        times1 = sm.results['spectra_times']['INST1']
        times2 = sm.results['spectra_times']['INST2']

        print(f"INST1 times: {times1}")
        print(f"INST2 times: {times2}")

        assert len(times1) == 2
        assert len(times2) == 4
        
        # Check cadence for INST2 (0.5 days)
        cadence2 = np.diff(times2)
        assert np.allclose(cadence2, 0.5)
        
        # Check start times
        assert np.isclose(times1[0], 0.0)
        assert np.isclose(times2[0], 1.0)

        print("Test passed!")

    finally:
        # Cleanup
        if test_config_path.exists():
            os.remove(test_config_path)
        if spotmap_backup is not None:
            with open(spotmap_path, 'w') as f:
                f.write(spotmap_backup)
        elif spotmap_path.exists():
             os.remove(spotmap_path)

if __name__ == "__main__":
    try:
        test_per_instrument_epochs()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
