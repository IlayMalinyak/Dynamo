"""
Classical Period Analysis for Simulated Lightcurves

Uses ACF-based classical methods to predict rotation periods as a baseline
for comparison with deep learning approaches.

Adapted from period_analysis.ipynb classical methods.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.stattools import acf as A
from tqdm import tqdm
import argparse

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


MINIMAL_PERIOD = 6  # Minimum period in days


def find_local_minima(array):
    """Find local minima in an array."""
    local_mins = [i for i in range(len(array) - 1)
                  if (array[i - 1] > array[i]) & (array[i] < array[i + 1])]
    return np.array(local_mins)


def find_closest_minima(array, peaks):
    """Find closest minima on both sides of the highest peak."""
    if len(peaks) == 0:
        return None, None, None
    
    minima = find_local_minima(array)
    
    # Find index of the highest peak
    highest_peak_index = peaks[np.argmax(array[peaks])]
    
    # Find closest minima on both sides of the highest peak
    left_minima = minima[minima < highest_peak_index]
    right_minima = minima[minima > highest_peak_index]
    
    closest_left_min_index = np.argmax(left_minima) if len(left_minima) else None
    closest_right_min_index = np.argmin(right_minima) if len(right_minima) else None
    
    closest_left_min_index = left_minima[closest_left_min_index] if len(left_minima) > 0 else None
    closest_right_min_index = right_minima[closest_right_min_index] if len(right_minima) > 0 else None
    
    return highest_peak_index, closest_left_min_index, closest_right_min_index


def find_period(data, lags, prom=None, method='slope'):
    """
    Find period from ACF using peak detection.
    
    Parameters
    ----------
    data : np.ndarray
        ACF values
    lags : np.ndarray
        Lag values (in days)
    prom : float, optional
        Prominence threshold for peak detection
    method : str
        Method for selecting period: 'first', 'max', or 'slope'
        
    Returns
    -------
    period : float
        Detected period in days
    peaks : np.ndarray
        Indices of detected peaks
    lph : float
        Local peak height
    """
    peaks, _ = find_peaks(data, distance=5, prominence=prom)
    highest_peak, closest_left, closest_right = find_closest_minima(data, peaks)
    
    # Calculate local peak height
    if closest_left is not None and closest_right is not None:
        lph = data[highest_peak] - np.mean((data[closest_left], data[closest_right]))
    else:
        lph = 0
    
    if len(peaks) == 0:
        return 0, peaks, lph
    
    i = 0
    max_peak = -np.inf
    max_idx = i
    
    if method == 'first':
        # Find first peak above minimum period
        while lags[peaks[i]] < MINIMAL_PERIOD:
            i += 1
            if i == len(peaks):
                return np.inf, peaks, lph
        p = lags[peaks[i]]
        
    elif method == 'max':
        # Find maximum peak
        while i < len(peaks):
            if data[peaks[i]] > max_peak:
                max_peak = data[peaks[i]]
                max_idx = i
            i += 1
        p = lags[peaks[max_idx]]
        
    elif method == 'slope':
        # Use slope method (most robust)
        if len(peaks) > 2:
            if (data[peaks[0]] < data[peaks[1]]) & (data[peaks[2]] < data[peaks[1]]):
                return lags[peaks[1]], peaks, lph
        
        first_peaks = []
        while i < 4:
            first_peaks.append(lags[peaks[i]])
            i += 1
            if i == len(peaks):
                break
        
        if i >= 4:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(
                np.arange(len(first_peaks)), first_peaks
            )
            p = slope
        else:
            p = first_peaks[0] if len(first_peaks) > 0 else 0
    
    return p, peaks, lph


def analyze_lc_kepler(lc, day_cadence=1/48, prom=0.12, method='max'):
    """
    Analyze lightcurve using ACF method (Kepler-style).
    
    Parameters
    ----------
    lc : np.ndarray
        Lightcurve flux array
    day_cadence : float
        Cadence in days (default: 30 minutes = 1/48 day)
    prom : float
        Prominence threshold for peak detection
    method : str
        Method for period selection: 'first', 'max', or 'slope'
        Default is 'max' (highest peak) which works best for multi-spot stars
        
    Returns
    -------
    period : float
        Detected period in days
    lags : np.ndarray
        ACF lag values
    acf : np.ndarray
        ACF values
    peaks : np.ndarray
        Detected peak indices
    """
    # Compute ACF
    xcf = A(lc, nlags=len(lc) - 1)
    xcf = xcf - np.median(xcf)
    xcf = gaussian_filter1d(xcf, 7.5)
    
    # Create lag array
    xcf_lags = np.arange(0, len(xcf) * day_cadence, day_cadence)
    
    # Find period using specified method
    xcf_period, peaks, lph = find_period(xcf, xcf_lags, prom=prom, method=method)
    
    return np.abs(xcf_period), xcf_lags, xcf, peaks


def load_simulated_lightcurve(lc_path, sim_id, chunk_id):
    """
    Load a simulated lightcurve from parquet chunk.
    
    Parameters
    ----------
    lc_path : str
        Path to lightcurve directory
    sim_id : int
        Simulation ID
    chunk_id : int
        Chunk ID
        
    Returns
    -------
    flux : np.ndarray
        Flux array
    """
    chunk_path = os.path.join(lc_path, f'chunk_{chunk_id}.pqt')
    df = pd.read_parquet(chunk_path)
    
    # Filter for this simulation
    sim_data = df[df['sim_id'] == sim_id].sort_values('time')
    
    if len(sim_data) == 0:
        return None
    
    flux = sim_data['flux'].values
    
    # Normalize by max (matching dataset preprocessing)
    max_flux = np.abs(flux).max()
    if max_flux > 0:
        flux = flux / max_flux
    
    return flux


def analyze_simulated_dataset(
    metadata_path: str,
    lc_path: str,
    output_path: str,
    max_samples: int = None,
    day_cadence: float = 1/48,
    prom: float = 0.12
):
    """
    Analyze simulated dataset using classical ACF method.
    
    Parameters
    ----------
    metadata_path : str
        Path to simulation_properties.csv
    lc_path : str
        Path to lightcurve directory
    output_path : str
        Path to save results CSV
    max_samples : int, optional
        Maximum number of samples to analyze
    day_cadence : float
        Cadence in days
    prom : float
        Prominence threshold
    """
    print("="*80)
    print("CLASSICAL PERIOD ANALYSIS - ACF METHOD")
    print("="*80)
    
    # Load metadata
    print(f"\nLoading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    if max_samples is not None:
        df = df.iloc[:max_samples]
    
    print(f"Analyzing {len(df)} simulations...")
    
    # Add chunk_id column
    df['chunk_id'] = df['Simulation Number'] // 100
    
    # Results storage
    results = []
    
    # Analyze each simulation
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing lightcurves"):
        sim_id = int(row['Simulation Number'])
        chunk_id = int(row['chunk_id'])
        true_period = row['Period']
        
        # Load lightcurve
        flux = load_simulated_lightcurve(lc_path, sim_id, chunk_id)
        
        if flux is None:
            print(f"Warning: Could not load simulation {sim_id}")
            results.append({
                'sim_id': sim_id,
                'true_period': true_period,
                'predicted_period': np.nan,
                'error': np.nan,
                'relative_error': np.nan,
                'num_peaks': 0
            })
            continue
        
        # Analyze with ACF
        try:
            pred_period, lags, acf, peaks = analyze_lc_kepler(
                flux, day_cadence=day_cadence, prom=prom
            )
            
            # Calculate errors
            error = pred_period - true_period
            rel_error = np.abs(error) / true_period if true_period > 0 else np.nan
            
            results.append({
                'sim_id': sim_id,
                'true_period': true_period,
                'predicted_period': pred_period,
                'error': error,
                'relative_error': rel_error,
                'num_peaks': len(peaks)
            })
            
        except Exception as e:
            print(f"Error analyzing simulation {sim_id}: {e}")
            results.append({
                'sim_id': sim_id,
                'true_period': true_period,
                'predicted_period': np.nan,
                'error': np.nan,
                'relative_error': np.nan,
                'num_peaks': 0
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    valid_results = results_df[results_df['predicted_period'].notna()]
    
    print(f"\nTotal simulations: {len(results_df)}")
    print(f"Successfully analyzed: {len(valid_results)} ({100*len(valid_results)/len(results_df):.1f}%)")
    
    if len(valid_results) > 0:
        print(f"\nPeriod Statistics:")
        print(f"  Mean absolute error: {valid_results['error'].abs().mean():.3f} days")
        print(f"  Median absolute error: {valid_results['error'].abs().median():.3f} days")
        print(f"  Mean relative error: {100*valid_results['relative_error'].mean():.1f}%")
        print(f"  Median relative error: {100*valid_results['relative_error'].median():.1f}%")
        
        # Accuracy within thresholds
        within_10pct = (valid_results['relative_error'] < 0.1).sum()
        within_20pct = (valid_results['relative_error'] < 0.2).sum()
        
        print(f"\nAccuracy:")
        print(f"  Within 10%: {within_10pct} ({100*within_10pct/len(valid_results):.1f}%)")
        print(f"  Within 20%: {within_20pct} ({100*within_20pct/len(valid_results):.1f}%)")
    
    print("="*80)
    
    return results_df


def plot_results(results_df, output_dir):
    """Create diagnostic plots for classical analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    valid_results = results_df[results_df['predicted_period'].notna()]
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    # 1. Predicted vs True Period
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(valid_results['true_period'], valid_results['predicted_period'], 
                    alpha=0.5, s=10)
    axes[0].plot([0, 70], [0, 70], 'r--', label='Perfect prediction')
    axes[0].set_xlabel('True Period (days)')
    axes[0].set_ylabel('Predicted Period (days)')
    axes[0].set_title('ACF Period Predictions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Relative Error Distribution
    axes[1].hist(valid_results['relative_error'], bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(valid_results['relative_error'].median(), color='r', 
                    linestyle='--', label=f'Median: {valid_results["relative_error"].median():.2f}')
    axes[1].set_xlabel('Relative Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Relative Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'acf_results.png'), dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {os.path.join(output_dir, 'acf_results.png')}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classical period analysis for simulated lightcurves')
    parser.add_argument('--metadata_path', type=str, 
                        default='/home/ilay.kamai/work/Dynamo/dataset/simulation_properties.csv',
                        help='Path to simulation metadata CSV')
    parser.add_argument('--lc_path', type=str,
                        default='/home/ilay.kamai/work/Dynamo/dataset/lc',
                        help='Path to lightcurve directory')
    parser.add_argument('--output_path', type=str,
                        default='/home/ilay.kamai/work/SimLearning/results/acf_predictions.csv',
                        help='Path to save results CSV')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to analyze')
    parser.add_argument('--day_cadence', type=float, default=1/48,
                        help='Cadence in days (default: 30 min = 1/48 day)')
    parser.add_argument('--prominence', type=float, default=0.12,
                        help='Prominence threshold for peak detection')
    parser.add_argument('--plot', action='store_true',
                        help='Create diagnostic plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Run analysis
    results_df = analyze_simulated_dataset(
        metadata_path=args.metadata_path,
        lc_path=args.lc_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        day_cadence=args.day_cadence,
        prom=args.prominence
    )
    
    # Create plots if requested
    if args.plot:
        plot_dir = os.path.dirname(args.output_path)
        plot_results(results_df, plot_dir)
    
    print("\nDone!")
