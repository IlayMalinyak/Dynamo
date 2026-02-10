"""
Comprehensive Period Recovery Test Suite

Tests rotation period recovery using ACF across 20 different scenarios:
- 5 single-spot stars with different periods
- 5 single-spot + planet stars with different periods
- 5 multi-spot stars with different periods
- 5 multi-spot + planet stars with different periods

Uses the ACF logic from Dynamo/classical_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from Dynamo.star import Star
from Dynamo.classical_analysis import analyze_lc_kepler

# Test periods to use
TEST_PERIODS = [10.0, 15.0, 20.0, 25.0, 30.0]


def create_star_with_params(period, activity, shear=0.0, with_planet=False):
    """Create a star with specified parameters."""
    sm = Star(conf_file_path='star.conf')
    
    params = {
        'Activity Rate': activity,
        'Shear': shear,
        'Period': period,
        'convective_shift': 0.0,
        'R': 1.0,
        'mass': 1.0,
        'L': 1.0,
        'Teff': 5777,
        'logg': 4.44,
        'FeH': 0.0,
        'age': 4.5,
        'CDPP': 0.0,  # No noise for clean test
        'Outlier Rate': 0.0,
        'Flicker Time Scale': 8.0,
        'Inclination': np.pi/2,  # Edge-on
        'Cycle Length': 3650,
        'Cycle Overlap': 0.5,
        'Spot Max': 45,
        'Spot Min': 0,  # Northern hemisphere only
        'Decay Time': 5.0,
        'Butterfly': False,
        'Distance': 100.0,
    }
    
    # Planet parameters
    if with_planet:
        params.update({
            'simulate_planet': 1,
            'planet_period': period / 3.0,  # Planet period is 1/3 of stellar rotation
            'planet_transit_t0': 5.0,
            'planet_radius': 0.1,
            'planet_impact': 0.5,
            'planet_esinw': 0.0,
            'planet_ecosw': 0.0,
            'planet_spin_orbit_angle': 0.0,
            'planet_semi_amplitude': 0.0
        })
    else:
        params.update({
            'simulate_planet': 0,
            'planet_period': 10.0,
            'planet_transit_t0': 5.0,
            'planet_radius': 0.0,
            'planet_impact': 0.5,
            'planet_esinw': 0.0,
            'planet_ecosw': 0.0,
            'planet_spin_orbit_angle': 0.0,
            'planet_semi_amplitude': 0.0
        })
    
    sm.set_stellar_parameters(params)
    sm.set_planet_parameters(params)
    
    return sm


def test_single_spot(period, test_id, with_planet=False):
    """Test with a single equatorial spot."""
    sm = create_star_with_params(period, activity=1.0, with_planet=with_planet)
    
    # Create single spot
    sm.spot_map = np.array([[
        0.0,      # init_time
        500.0,    # duration
        90.0,     # colatitude (equator)
        0.0,      # longitude
        10.0,     # radius
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]])
    
    # Generate light curve
    ndays = 200
    t_sampling = np.linspace(0, ndays, int(ndays * 48))
    sm.compute_forward(t=t_sampling)
    lc = sm.results['lc']
    
    # Analyze with ACF
    pred_period, lags, acf, peaks = analyze_lc_kepler(lc, day_cadence=1/48, prom=0.05)
    
    return {
        'test_id': test_id,
        'scenario': 'Single Spot + Planet' if with_planet else 'Single Spot',
        'true_period': period,
        'predicted_period': pred_period,
        'error': pred_period - period,
        'relative_error': abs(pred_period - period) / period,
        'num_spots': 1,
        'has_planet': with_planet,
        'time': t_sampling,
        'flux': lc,
        'acf': acf,
        'acf_lags': lags,
        'acf_peaks': peaks
    }


def test_multi_spot(period, test_id, with_planet=False):
    """Test with multiple spots."""
    sm = create_star_with_params(period, activity=5.0, shear=0.1, with_planet=with_planet)
    
    # Generate spot map
    sm.generate_spot_map(ndays=200)
    
    # Generate light curve
    ndays = 200
    t_sampling = np.linspace(0, ndays, int(ndays * 48))
    sm.compute_forward(t=t_sampling)
    lc = sm.results['lc']
    
    # Analyze with ACF
    pred_period, lags, acf, peaks = analyze_lc_kepler(lc, day_cadence=1/48, prom=0.05)
    
    return {
        'test_id': test_id,
        'scenario': 'Multi Spot + Planet' if with_planet else 'Multi Spot',
        'true_period': period,
        'predicted_period': pred_period,
        'error': pred_period - period,
        'relative_error': abs(pred_period - period) / period,
        'num_spots': len(sm.spot_map),
        'has_planet': with_planet,
        'time': t_sampling,
        'flux': lc,
        'acf': acf,
        'acf_lags': lags,
        'acf_peaks': peaks
    }


def plot_test_result(result, output_dir):
    """Plot individual test result with light curve and ACF."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    test_id = result['test_id']
    scenario = result['scenario']
    true_p = result['true_period']
    pred_p = result['predicted_period']
    error_pct = result['relative_error'] * 100
    
    fig.suptitle(
        f"Test {test_id}: {scenario} | True P={true_p:.1f}d | Pred P={pred_p:.1f}d | Error={error_pct:.1f}%",
        fontsize=14, fontweight='bold'
    )
    
    # Full light curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(result['time'], result['flux'], linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Flux')
    ax1.set_title(f'Full Light Curve ({len(result["flux"])} points, {result["num_spots"]} spots)')
    ax1.grid(True, alpha=0.3)
    
    # Zoomed light curve (first 100 days)
    ax2 = fig.add_subplot(gs[1, 0])
    mask = result['time'] < 100
    ax2.plot(result['time'][mask], result['flux'][mask], linewidth=1)
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Flux')
    ax2.set_title('Zoomed: First 100 days')
    ax2.grid(True, alpha=0.3)
    
    # Mark expected rotation periods
    for i in range(1, 5):
        ax2.axvline(i * true_p, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Zoomed light curve (first 3 periods)
    ax3 = fig.add_subplot(gs[1, 1])
    mask = result['time'] < (3 * true_p)
    ax3.plot(result['time'][mask], result['flux'][mask], linewidth=1)
    ax3.set_xlabel('Time [days]')
    ax3.set_ylabel('Flux')
    ax3.set_title(f'Zoomed: First 3 Periods ({3*true_p:.1f} days)')
    ax3.grid(True, alpha=0.3)
    
    # Mark periods
    for i in range(1, 4):
        ax3.axvline(i * true_p, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # ACF
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(result['acf_lags'], result['acf'], linewidth=1, color='blue')
    
    # Mark peaks
    if len(result['acf_peaks']) > 0:
        peak_lags = result['acf_lags'][result['acf_peaks']]
        peak_vals = result['acf'][result['acf_peaks']]
        ax4.plot(peak_lags, peak_vals, 'ro', markersize=4, label=f'{len(result["acf_peaks"])} peaks detected')
    
    # Mark true and predicted periods
    ax4.axvline(true_p, color='g', linestyle='--', linewidth=2, label=f'True Period ({true_p:.1f}d)')
    ax4.axvline(pred_p, color='r', linestyle='--', linewidth=2, label=f'Predicted Period ({pred_p:.1f}d)')
    
    ax4.set_xlabel('Lag [days]')
    ax4.set_ylabel('ACF')
    ax4.set_title('Autocorrelation Function')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, min(100, max(result['acf_lags'])))
    
    # Save
    filename = f"test_{test_id:02d}_{scenario.replace(' ', '_').lower()}_P{true_p:.0f}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def plot_summary(results_df, output_dir):
    """Create summary plots for all tests."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Predicted vs True Period (by scenario)
    ax = axes[0, 0]
    scenarios = results_df['scenario'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        data = results_df[results_df['scenario'] == scenario]
        ax.scatter(data['true_period'], data['predicted_period'], 
                  label=scenario, alpha=0.7, s=100, color=colors[i])
    
    ax.plot([0, 35], [0, 35], 'k--', label='Perfect prediction', linewidth=2)
    ax.set_xlabel('True Period [days]', fontsize=12)
    ax.set_ylabel('Predicted Period [days]', fontsize=12)
    ax.set_title('Period Predictions by Scenario', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 35)
    ax.set_ylim(5, 35)
    
    # 2. Relative Error by Scenario
    ax = axes[0, 1]
    scenario_errors = []
    scenario_labels = []
    for scenario in scenarios:
        data = results_df[results_df['scenario'] == scenario]
        scenario_errors.append(data['relative_error'].values * 100)
        scenario_labels.append(scenario)
    
    bp = ax.boxplot(scenario_errors, labels=scenario_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Relative Error [%]', fontsize=12)
    ax.set_title('Period Error Distribution by Scenario', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 3. Error vs True Period
    ax = axes[1, 0]
    for i, scenario in enumerate(scenarios):
        data = results_df[results_df['scenario'] == scenario]
        ax.scatter(data['true_period'], data['relative_error'] * 100,
                  label=scenario, alpha=0.7, s=100, color=colors[i])
    
    ax.axhline(10, color='orange', linestyle='--', label='10% error', linewidth=2)
    ax.axhline(20, color='red', linestyle='--', label='20% error', linewidth=2)
    ax.set_xlabel('True Period [days]', fontsize=12)
    ax.set_ylabel('Relative Error [%]', fontsize=12)
    ax.set_title('Error vs Period', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Success Rate Table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate success rates
    table_data = []
    table_data.append(['Scenario', 'Count', '<10% Error', '<20% Error', 'Mean Error'])
    
    for scenario in scenarios:
        data = results_df[results_df['scenario'] == scenario]
        count = len(data)
        within_10 = (data['relative_error'] < 0.1).sum()
        within_20 = (data['relative_error'] < 0.2).sum()
        mean_error = data['relative_error'].mean() * 100
        
        table_data.append([
            scenario,
            f'{count}',
            f'{within_10}/{count} ({100*within_10/count:.0f}%)',
            f'{within_20}/{count} ({100*within_20/count:.0f}%)',
            f'{mean_error:.1f}%'
        ])
    
    # Overall
    count = len(results_df)
    within_10 = (results_df['relative_error'] < 0.1).sum()
    within_20 = (results_df['relative_error'] < 0.2).sum()
    mean_error = results_df['relative_error'].mean() * 100
    
    table_data.append([
        'OVERALL',
        f'{count}',
        f'{within_10}/{count} ({100*within_10/count:.0f}%)',
        f'{within_20}/{count} ({100*within_20/count:.0f}%)',
        f'{mean_error:.1f}%'
    ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.3, 0.1, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style last row
    for i in range(5):
        table[(len(table_data)-1, i)].set_facecolor('#d0d0d0')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')
    
    ax.set_title('Success Rate Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'summary_results.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath


def run_all_tests():
    """Run all 20 test scenarios."""
    print("="*80)
    print("COMPREHENSIVE PERIOD RECOVERY TEST SUITE")
    print("="*80)
    print("\nRunning 20 test scenarios:")
    print("  - 5 single-spot stars")
    print("  - 5 single-spot + planet stars")
    print("  - 5 multi-spot stars")
    print("  - 5 multi-spot + planet stars")
    print("="*80)
    
    # Create output directory
    output_dir = 'tests/images/period_recovery_tests'
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    test_id = 1
    
    # Test 1-5: Single spot, no planet
    print("\n[Tests 1-5] Single Spot (No Planet)")
    print("-"*80)
    for period in TEST_PERIODS:
        print(f"  Test {test_id}: Period = {period:.1f} days... ", end='', flush=True)
        result = test_single_spot(period, test_id, with_planet=False)
        results.append(result)
        plot_test_result(result, output_dir)
        print(f"✓ Predicted: {result['predicted_period']:.2f}d (Error: {result['relative_error']*100:.1f}%)")
        test_id += 1
    
    # Test 6-10: Single spot + planet
    print("\n[Tests 6-10] Single Spot + Planet")
    print("-"*80)
    for period in TEST_PERIODS:
        print(f"  Test {test_id}: Period = {period:.1f} days... ", end='', flush=True)
        result = test_single_spot(period, test_id, with_planet=True)
        results.append(result)
        plot_test_result(result, output_dir)
        print(f"✓ Predicted: {result['predicted_period']:.2f}d (Error: {result['relative_error']*100:.1f}%)")
        test_id += 1
    
    # Test 11-15: Multi spot, no planet
    print("\n[Tests 11-15] Multi Spot (No Planet)")
    print("-"*80)
    for period in TEST_PERIODS:
        print(f"  Test {test_id}: Period = {period:.1f} days... ", end='', flush=True)
        result = test_multi_spot(period, test_id, with_planet=False)
        results.append(result)
        plot_test_result(result, output_dir)
        print(f"✓ Predicted: {result['predicted_period']:.2f}d (Error: {result['relative_error']*100:.1f}%) [{result['num_spots']} spots]")
        test_id += 1
    
    # Test 16-20: Multi spot + planet
    print("\n[Tests 16-20] Multi Spot + Planet")
    print("-"*80)
    for period in TEST_PERIODS:
        print(f"  Test {test_id}: Period = {period:.1f} days... ", end='', flush=True)
        result = test_multi_spot(period, test_id, with_planet=True)
        results.append(result)
        plot_test_result(result, output_dir)
        print(f"✓ Predicted: {result['predicted_period']:.2f}d (Error: {result['relative_error']*100:.1f}%) [{result['num_spots']} spots]")
        test_id += 1
    
    # Create DataFrame
    results_df = pd.DataFrame([{
        'test_id': r['test_id'],
        'scenario': r['scenario'],
        'true_period': r['true_period'],
        'predicted_period': r['predicted_period'],
        'error': r['error'],
        'relative_error': r['relative_error'],
        'num_spots': r['num_spots'],
        'has_planet': r['has_planet']
    } for r in results])
    
    # Save results
    results_path = os.path.join(output_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Create summary plots
    print("\nCreating summary plots...")
    summary_path = plot_summary(results_df, output_dir)
    print(f"✓ Summary plot saved to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for scenario in results_df['scenario'].unique():
        data = results_df[results_df['scenario'] == scenario]
        print(f"\n{scenario}:")
        print(f"  Mean error: {data['relative_error'].mean()*100:.1f}%")
        print(f"  Median error: {data['relative_error'].median()*100:.1f}%")
        print(f"  Within 10%: {(data['relative_error'] < 0.1).sum()}/{len(data)}")
        print(f"  Within 20%: {(data['relative_error'] < 0.2).sum()}/{len(data)}")
    
    print(f"\nOVERALL:")
    print(f"  Mean error: {results_df['relative_error'].mean()*100:.1f}%")
    print(f"  Median error: {results_df['relative_error'].median()*100:.1f}%")
    print(f"  Within 10%: {(results_df['relative_error'] < 0.1).sum()}/{len(results_df)}")
    print(f"  Within 20%: {(results_df['relative_error'] < 0.2).sum()}/{len(results_df)}")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}/")
    print("="*80)
    
    return results_df


if __name__ == '__main__':
    # Run only a small subset to verify fix quickly
    print("Running quick verification (Test 1 only)...")
    # We can just call test_single_spot directly
    output_dir = 'images/period_recovery_tests'
    os.makedirs(output_dir, exist_ok=True)
    
    result = test_single_spot(10.0, 1, with_planet=False)
    plot_test_result(result, output_dir)
    print(f"Test 1 Result: Predicted={result['predicted_period']:.2f}, Error={result['relative_error']*100:.1f}%")
    
    result_multi = test_multi_spot(15, 2, with_planet=True)
    plot_test_result(result_multi, output_dir)
    print(f"Test 2 Result: Predicted={result_multi['predicted_period']:.2f}, Error={result_multi['relative_error']*100:.1f}%")
    if result['relative_error'] < 0.1 and result_multi['relative_error'] < 0.1:
        print("VERIFICATION PASSED: Period recovered successfully.")
    else:
        print("VERIFICATION FAILED: Period recovery still failing.")
    print("\n✓ Test suite complete!")
