"""
Smoke test for multi-planet support in Dynamo.
Tests that the simulation correctly handles 0, 1, and 2+ planets per sample.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Dynamo.star import Star
from Dynamo.planet_params import generate_planet_parameters


def test_planet_params_generation():
    """Test that planet_params generates varying numbers of planets."""
    print("\n=== Test 1: Planet Parameters Generation ===")
    
    np.random.seed(42)
    n_samples = 100
    mass = np.random.uniform(0.5, 1.5, n_samples)
    radius = np.random.uniform(0.5, 1.5, n_samples)
    
    result = generate_planet_parameters(mass, radius, planet_prob=0.3, max_planets=3)
    
    print(f"n_planets distribution: {np.bincount(result['n_planets'])}")
    print(f"Samples with 0 planets: {np.sum(result['n_planets'] == 0)}")
    print(f"Samples with 1 planet: {np.sum(result['n_planets'] == 1)}")
    print(f"Samples with 2+ planets: {np.sum(result['n_planets'] >= 2)}")
    
    # Check that planets list has correct structure
    assert len(result['planets']) == n_samples
    for i, planets in enumerate(result['planets']):
        assert len(planets) == result['n_planets'][i]
        for p in planets:
            assert 'period' in p
            assert 'radius' in p
            assert 'semi_amplitude' in p
    
    print("✓ Planet parameters generation works correctly")
    return result


def test_set_planet_parameters():
    """Test that Star.set_planet_parameters correctly handles multi-planet."""
    print("\n=== Test 2: Star.set_planet_parameters ===")
    
    star = Star(conf_file_path='star.conf')
    
    # Test with new format (planets list)
    planets_list = [
        {'period': 1.0, 'transit_t0': 0.5, 'radius': 0.1, 'impact': 0.0,
         'esinw': 0.0, 'ecosw': 0.0, 'spin_orbit_angle': 0.0, 'semi_amplitude': 10.0},
        {'period': 3.0, 'transit_t0': 1.5, 'radius': 0.05, 'impact': 0.3,
         'esinw': 0.0, 'ecosw': 0.0, 'spin_orbit_angle': 0.0, 'semi_amplitude': 5.0},
    ]
    
    star.set_planet_parameters({'planets': planets_list})
    
    assert star.n_planets == 2
    assert len(star.planets) == 2
    assert star.planets[0]['period'] == 1.0
    assert star.planets[1]['period'] == 3.0
    assert star.simulate_planet == 1
    
    # Backward compat - first planet values set
    assert star.planet_period == 1.0
    
    print("✓ Multi-planet set_planet_parameters works")
    
    # Test with old format
    star2 = Star(conf_file_path='star.conf')
    star2.set_planet_parameters({
        'simulate_planet': 1,
        'planet_period': 2.0, 'planet_transit_t0': 1.0,
        'planet_radius': 0.08, 'planet_impact': 0.2,
        'planet_esinw': 0.0, 'planet_ecosw': 0.0,
        'planet_spin_orbit_angle': 0.0, 'planet_semi_amplitude': 8.0
    })
    
    assert star2.n_planets == 1
    assert len(star2.planets) == 1
    
    print("✓ Old format backward compatibility works")


def test_two_planet_simulation():
    """Test full simulation with two planets."""
    print("\n=== Test 3: Two-Planet Simulation ===")
    
    star = Star(conf_file_path='star.conf')
    star.models_root = Path("C:/Users/Ilay/.dynamo")
    
    # Setup star parameters
    star.activity = 0.1
    star.butterfly = False
    star.cycle_len = 10
    star.cycle_overlap = 0
    star.spot_max_lat = 30
    star.spot_min_lat = 5
    star.feh = 0.0
    star.cdpp = 0.0
    star.outliers_rate = 0.0
    star.flicker = 0.0
    
    # Two planets with different periods
    planets_list = [
        {'period': 1.5, 'transit_t0': 1.0, 'radius': 0.08, 'impact': 0.0,
         'esinw': 0.0, 'ecosw': 0.0, 'spin_orbit_angle': 0.0, 'semi_amplitude': 15.0},
        {'period': 4.0, 'transit_t0': 2.5, 'radius': 0.05, 'impact': 0.2,
         'esinw': 0.0, 'ecosw': 0.0, 'spin_orbit_angle': 0.0, 'semi_amplitude': 7.0},
    ]
    star.set_planet_parameters({'planets': planets_list})
    
    # Generate minimal spot map
    star.generate_spot_map(ndays=10)
    
    # Run simulation
    t = np.linspace(0, 10, 500)
    star.compute_forward(t=t)
    
    lc = star.results['lc']
    rv = star.results['rv']
    
    print(f"Light curve range: {np.min(lc):.4f} to {np.max(lc):.4f}")
    print(f"RV range: {np.min(rv):.2f} to {np.max(rv):.2f} m/s")
    
    # RV should show superposition of two signals
    assert np.abs(rv).max() > 0, "RV should be non-zero with planets"
    
    # Check that we have variability in light curve (from planets + spots)
    assert np.std(lc) > 0, "Light curve should have variability"
    
    print("✓ Two-planet simulation completed successfully")
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(t, lc, 'b-', lw=0.5)
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Normalized Flux')
    axes[0].set_title('Light Curve with Two Planets')
    
    axes[1].plot(t, rv, 'r-')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('RV (m/s)')
    axes[1].set_title('RV Signal (Sum of Two Planets)')
    
    plt.tight_layout()
    plt.savefig('tests/images/multi_planet_test.png', dpi=150)
    print("✓ Plot saved to tests/multi_planet_test.png")
    plt.close()


def test_zero_planets():
    """Test that zero planets works correctly."""
    print("\n=== Test 4: Zero Planets ===")
    
    star = Star(conf_file_path='star.conf')
    star.set_planet_parameters({'planets': []})
    
    assert star.n_planets == 0
    assert star.simulate_planet == 0
    
    print("✓ Zero planets configuration works")


if __name__ == "__main__":
    print("=" * 50)
    print("Multi-Planet Support Smoke Test")
    print("=" * 50)
    
    test_planet_params_generation()
    test_set_planet_parameters()
    test_zero_planets()
    test_two_planet_simulation()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
