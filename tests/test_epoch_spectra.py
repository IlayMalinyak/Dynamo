import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from Dynamo.star import Star
import configparser

def test_epoch_spectra():
    # Setup - write a temporary config
    config = configparser.ConfigParser()
    config_path = os.path.join('Dynamo', 'star.conf')
    if not os.path.exists(config_path):
        # Fallback if run from different dir
        config_path = 'star.conf'
    config.read(config_path)
    
    # Enable multiple epochs
    config.set('general', 'n_spectra_epochs', '3')
    config.set('general', 'spectra_cadence', '2.0')
    config.set('general', 'spectra_start_time', '10.0')
    
    temp_conf_path = os.path.abspath(os.path.join('tests', 'temp_star.conf'))
    with open(temp_conf_path, 'w') as f:
        config.write(f)
        
    try:
        # Initialize star
        s = Star(conf_file_path=temp_conf_path)
        
        # Minimal parameters for Star.set_stellar_parameters
        params = {
            'Activity Rate': 1.0,
            'Shear': 0.0,
            'Period': 10.0,
            'convective_shift': 0.0,
            'R': 1.0,
            'mass': 1.0,
            'L': 1.0,
            'Teff': 3500.0,
            'logg': 5.0,
            'FeH': 0.0,
            'age': 1.0,
            'CDPP': 100.0,
            'Outlier Rate': 0.0,
            'Flicker Time Scale': 0.0,
            'Inclination': np.deg2rad(90.0),
            'Cycle Length': 10.0,
            'Cycle Overlap': 0.0,
            'Spot Max': 40.0,
            'Spot Min': 0.0,
            'Decay Time': 5.0,
            'Butterfly': True,
            'Distance': 10.0,
            'simulate_planet': 1,
            'planet_period': 1.0,
            'planet_radius': 0.1,
            'planet_esinw': 0.0,
            'planet_ecosw': 0.0,
            'planet_impact': 0.0,
            'planet_spin_orbit_angle': 0.0,
            'planet_transit_t0': 2.0,
            'planet_semi_amplitude': 1.0
        }
        s.set_stellar_parameters(params)
        s.set_planet_parameters(params)
        s.generate_spot_map(ndays=30)
        
        # Mocking or setting up minimal data if needed, but let's try calling compute_forward
        # We need a time array
        t = np.linspace(0, 30, 300) # 30 days, 0.1 day cadence
        
        print("Running compute_forward with 3 epochs at 2.0d cadence starting at 10.0d...")
        s.compute_forward(t=t)
        
        # Verify results
        spectra_times = s.results.get('spectra_times')
        print(f"Sampled spectra times: {spectra_times}")
        
        expected_times = np.array([10.0, 12.0, 14.0])
        # spectra_times may be a dict keyed by instrument name
        if isinstance(spectra_times, dict):
            for instr, times in spectra_times.items():
                np.testing.assert_allclose(times, expected_times, atol=0.2)
        else:
            np.testing.assert_allclose(spectra_times, expected_times, atol=0.2)
        print("Verification of timestamps PASSED.")
        
        for name, (wv, flux) in s.results['spectra'].items():
             print(f"Instrument: {name}, Flux shape: {flux.shape}")
             assert flux.ndim == 2, f"Flux for {name} should be 2D"
             assert flux.shape[0] == 3, f"Flux for {name} should have 3 epochs"
             
        print("Verification of flux dimensions PASSED.")

        # Plot the results
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(s.results['spectra']), 1, figsize=(12, 4 * len(s.results['spectra'])))
        if len(s.results['spectra']) == 1: axes = [axes]
        
        for ax, (name, (wv, flux)) in zip(axes, s.results['spectra'].items()):
            # Get times for this instrument (or use first available)
            if isinstance(spectra_times, dict):
                instr_times = spectra_times.get(name, list(spectra_times.values())[0])
            else:
                instr_times = spectra_times
            for i in range(flux.shape[0]):
                ax.plot(wv, flux[i], label=f"Epoch {i} (t={instr_times[i]:.1f})", alpha=0.7)
            ax.set_title(f"Instrument: {name}")
            ax.set_xlabel("Wavelength [A]")
            ax.set_ylabel("Flux")
            ax.legend()
        
        plt.tight_layout()
        plot_path = 'tests/images/epoch_spectra_test.png'
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Test single epoch backward compatibility
        config.set('general', 'n_spectra_epochs', '1')
        with open(temp_conf_path, 'w') as f:
            config.write(f)
            
        s2 = Star(conf_file_path=temp_conf_path)
        s2.set_stellar_parameters(params)
        s2.set_planet_parameters(params)
        s2.generate_spot_map(ndays=30)
        s2.compute_forward(t=t)
        
        for name, (wv, flux) in s2.results['spectra'].items():
             print(f"Single epoch - Instrument: {name}, Flux shape: {flux.shape}")
             assert flux.ndim == 1, f"Flux for {name} should be 1D for single epoch"
             
        print("Backward compatibility (single epoch) PASSED.")

    finally:
        if os.path.exists('tests/temp_star.conf'):
            os.remove('tests/temp_star.conf')

if __name__ == "__main__":
    test_epoch_spectra()
