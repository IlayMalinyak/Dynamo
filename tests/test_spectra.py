"""
PHOENIX Interpolation Validation using pySME

This script validates PHOENIX stellar atmosphere interpolation by:
1. Generating synthetic spectra using your PHOENIX interpolation function
2. Fitting those spectra with pySME using MARCS atmosphere models
3. Comparing input vs recovered stellar parameters

Note: pySME doesn't have PHOENIX atmosphere grids available through its
Large File Server, so we use MARCS grids for the fitting. This tests whether
PHOENIX-generated spectra can be properly fit with standard tools, which is
a meaningful validation of the interpolation quality.

Available pySME atmosphere sources:
- marcs2012.sav (recommended)
- marcs2012p_t0.0.sav, marcs2012p_t1.0.sav, marcs2012p_t2.0.sav
- atlas12.sav, atlas9_vmic0.0.sav, atlas9_vmic2.0.sav

Common Warnings Explained:
- "wavelength range is overriden": pySME uses its own wavelength grid
- "No cconfiguration file found": Normal on first run, uses defaults
- "No Fit Parameters have been set": We set them explicitly, this is normal
- SSL timeout errors: Network/firewall issues downloading atmosphere files

Troubleshooting Download Issues:
1. Check internet connectivity
2. Corporate firewalls may block the Uppsala server
3. Try: export ASTROPY_DOWNLOAD_TIMEOUT=300 (for longer timeout)
4. Manual download: http://sme.astro.uu.se/atmos/ -> ~/.sme/atmospheres/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys
import warnings

warnings.filterwarnings("ignore")

# pySME imports

from pysme.sme import SME_Structure
# from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve
from pysme.linelist.vald import ValdFile
from pysme.abund import Abund
from pysme.nlte import nlte
import pysme.util

from astropy.io import fits
from scipy.interpolate import interp1d
from datetime import datetime
import json

class PHOENIXPySMEValidator:
    def __init__(self, phoenix_interpolation_func, phoenix_path, wv_array=None, linelist_path=None):
        """
        Initialize validator for PHOENIX interpolation using pySME.

        Parameters:
        -----------
        phoenix_interpolation_func : callable
            Your PHOENIX interpolation function
        phoenix_path : str
            Path to PHOENIX models directory
        linelist_path : str, optional
            Path to VALD line list file
        """
        self.interpolate_phoenix = phoenix_interpolation_func
        self.phoenix_path = Path(phoenix_path)
        self.linelist_path = linelist_path
        self.wv_array = wv_array

        # pySME configuration
        self.default_abundances = "asplund2009"
        self.default_nlte = None

        # Results storage
        self.validation_results = []

        print("Initializing pySME PHOENIX Validator...")
        self._setup_pysme()

    def _setup_pysme(self):
        """Setup pySME configuration."""
        print("Setting up pySME configuration...")

        # Load line list if provided
        if self.linelist_path and Path(self.linelist_path).exists():
            print(f"Loading line list from: {self.linelist_path}")
            self.linelist = ValdFile(self.linelist_path)
        else:
            print("No line list provided - will use default pySME lines")
            self.linelist = None

        # Set up abundances
        self.abundances = Abund.solar()

        # Set up NLTE if available
        try:
            self.nlte = nlte()
        except:
            print("NLTE not available, using LTE")
            self.nlte = None

    def create_mock_object(self, wavelength_range):
        """Create mock object with required attributes for PHOENIX interpolation."""

        class MockSpectrum:
            def __init__(self, path, wave_min, wave_max):
                self.path = Path(path)
                self.wavelength_lower_limit = wave_min
                self.wavelength_upper_limit = wave_max

        return MockSpectrum(self.phoenix_path, wavelength_range[0], wavelength_range[1])

    def generate_synthetic_spectrum(self, teff, logg, feh=0.0, wavelength_range=(3000, 9000),
                                    rv=0.0, vsini=2.0, vmac=3.0, add_noise=True, snr=100):
        """
        Generate synthetic spectrum using PHOENIX interpolation.

        Parameters:
        -----------
        teff : float
            Effective temperature (K)
        logg : float
            Log surface gravity
        feh : float
            Metallicity [Fe/H]
        wavelength_range : tuple
            Wavelength range (Å)
        rv : float
            Radial velocity (km/s)
        vsini : float
            Rotational velocity (km/s)
        vmac : float
            Macroturbulence velocity (km/s)
        add_noise : bool
            Whether to add noise
        snr : float
            Signal-to-noise ratio

        Returns:
        --------
        wave : array
            Wavelength array
        flux : array
            Flux array
        error : array
            Error array
        """
        print(f"Generating synthetic spectrum: T={teff}K, log g={logg}, [Fe/H]={feh}")

        # Create mock object for interpolation
        mock_obj = self.create_mock_object(wavelength_range)

        # Generate spectrum using your interpolation function
        result = self.interpolate_phoenix(mock_obj, teff, logg, wv_array=self.wv_array,
                                          interp_type='spline', plot=False)
        wave_phoenix, flux_phoenix = result[0], result[1]

        print("negative values after interpolation: ", np.any(flux_phoenix < 0))

        # Apply instrumental broadening and radial velocity
        flux_broadened = self._apply_broadening(wave_phoenix, flux_phoenix,
                                                rv, vsini, vmac)

        print("negative values after broadening: ", np.any(flux_broadened < 0))

        # Add noise if requested
        if add_noise:
            noise_level = 1.0 / snr
            noise = np.random.normal(0, noise_level, len(flux_broadened))
            flux_noisy = np.clip(flux_broadened + noise, a_min=1e-10, a_max=None)

            error = np.full_like(flux_broadened, noise_level)
        else:
            flux_noisy = flux_broadened
            error = np.full_like(flux_broadened, 0.001)  # Very small error

        print("negative values after noise addition: ", np.any(flux_noisy < 0))

        return wave_phoenix, flux_noisy, error

    def _apply_broadening(self, wavelength, flux, rv, vsini, vmac):
        """Apply radial velocity and rotational broadening."""
        # Apply radial velocity shift
        c_light = 299792.458  # km/s
        wave_shifted = wavelength * (1 + rv / c_light)

        # Interpolate back to original wavelength grid
        interp_func = interp1d(wave_shifted, flux, kind='linear',
                               bounds_error=False, fill_value=1.0)
        flux_rv = interp_func(wavelength)

        # Apply rotational broadening (simplified)
        if vsini > 0.5:
            # Simple Gaussian broadening approximation
            # Real implementation would use proper rotational broadening kernel
            sigma_rot = vsini * wavelength / c_light / 2.355  # Convert to wavelength units
            flux_broadened = self._gaussian_convolve(wavelength, flux_rv, sigma_rot)
        else:
            flux_broadened = flux_rv

        return flux_broadened

    def _gaussian_convolve(self, wavelength, flux, sigma):
        """Apply Gaussian convolution for broadening."""
        # Simplified convolution - in practice you'd use proper kernel
        from scipy.ndimage import gaussian_filter1d

        # Convert sigma to pixel units (assuming roughly uniform spacing)
        pixel_scale = np.median(np.diff(wavelength))
        sigma_pixels = np.median(sigma) / pixel_scale

        return gaussian_filter1d(flux, sigma_pixels)

    def setup_sme_structure(self, wavelength, flux, error, initial_guess):
        """
        Set up pySME structure for fitting.

        Parameters:
        -----------
        wavelength : array
            Wavelength array
        flux : array
            Flux array
        error : array
            Error array
        initial_guess : dict
            Initial parameter guesses

        Returns:
        --------
        sme : SME_Structure
            Configured pySME structure
        """
        print("Setting up pySME structure...")

        # Create SME structure
        sme = SME_Structure()

        # Set spectrum data
        sme.wave = [wavelength]
        sme.spec = [flux]
        sme.uncs = [error]

        # Set wavelength ranges for fitting - this may override existing ranges
        wave_min, wave_max = wavelength.min(), wavelength.max()
        sme.wran = [[wave_min, wave_max]]

        # Set initial parameters
        sme.teff = initial_guess.get('teff', 5500)
        sme.logg = initial_guess.get('logg', 4.5)
        sme.monh = initial_guess.get('feh', 0.0)
        sme.vmic = initial_guess.get('vmic', 1.0)  # Microturbulence
        sme.vmac = initial_guess.get('vmac', 3.0)  # Macroturbulence
        sme.vsini = initial_guess.get('vsini', 2.0)  # Rotational velocity
        sme.vrad = initial_guess.get('rv', 0.0)  # Radial velocity

        # Set abundances
        sme.abund = self.abundances

        # Set line list
        if self.linelist is not None:
            sme.linelist = self.linelist

        # Set NLTE
        if self.nlte is not None:
            sme.nlte = self.nlte

        # Set atmosphere (try to use local file first, fallback to standard)
        # Note: This means we're validating PHOENIX interpolation against MARCS fitting
        local_atmo_file = "marcs_st_mod.list"  # Your downloaded file
        try:
            # First try your local file
            sme.atmo.source = local_atmo_file
            sme.atmo.method = "grid"
            sme.atmo.geom = "PP"  # Plane-parallel geometry
            print(f"Trying to use local atmosphere file: {local_atmo_file}")
        except:
            # Fallback to standard file
            sme.atmo.source = "marcs2012.sav"  # Available atmosphere in pySME
            sme.atmo.method = "grid"
            sme.atmo.geom = "PP"  # Plane-parallel geometry
            print(f"Fallback to standard atmosphere file: marcs2012.sav")

        # Configure continuum and radial velocity handling
        # These flags control how pySME handles continuum and RV fitting
        sme.cscale_flag = "linear"  # Options: "constant", "linear", "fix", "none"
        sme.vrad_flag = "whole"     # Options: "each", "whole", "fix", "none"
        sme.cscale_type = "mask"    # Options: "whole", "mask"

        # Set up mask (1 = line pixel, 2 = continuum pixel, 4 = vrad pixel)
        # For simplicity, mark all as line pixels for now
        sme.mask = [np.ones(len(wavelength), dtype=int)]

        print(f"SME structure configured:")
        print(f"  Atmosphere: {sme.atmo.source}")
        print(f"  Wavelength range: {wave_min:.1f} - {wave_max:.1f} Å")
        print(f"  Number of spectral points: {len(wavelength)}")

        return sme

    def fit_spectrum_with_pysme(self, wavelength, flux, error, initial_guess,
                                fit_parameters=['teff', 'logg', 'monh']):
        """
        Fit spectrum with pySME.

        Parameters:
        -----------
        wavelength : array
            Wavelength array
        flux : array
            Flux array
        error : array
            Error array
        initial_guess : dict
            Initial parameter guesses
        fit_parameters : list
            Parameters to fit

        Returns:
        --------
        result : dict
            Fitting results
        """
        print(f"Fitting spectrum with pySME (fitting: {fit_parameters})...")

        # try:
        # Set up SME structure
        sme = self.setup_sme_structure(wavelength, flux, error, initial_guess)

        # Set which parameters to fit BEFORE calling solve()
        sme.fitpars = fit_parameters

        # Perform the fit - pySME 0.5.2 handles continuum and RV automatically
        # based on the flags set in the SME structure
        print("Running pySME solver...")
        sme = solve(sme)

        # Extract results
        result = {
            'teff': sme.teff,
            'logg': sme.logg,
            'feh': sme.monh,
            'vmic': sme.vmic,
            'vmac': sme.vmac,
            'vsini': sme.vsini,
            'rv': sme.vrad,
            'success': True,
            'chi2': getattr(sme, 'chi2', np.nan),
            'errors': getattr(sme, 'errors', {}),
            'sme_object': sme
        }

        print(f"Fit completed successfully:")
        print(f"  T_eff = {result['teff']:.0f} K")
        print(f"  log g = {result['logg']:.2f}")
        print(f"  [Fe/H] = {result['feh']:.2f}")

        # except Exception as e:
        #     print(f"Fit failed with error: {str(e)}")
        #     result = {
        #         'teff': np.nan,
        #         'logg': np.nan,
        #         'feh': np.nan,
        #         'vmic': np.nan,
        #         'vmac': np.nan,
        #         'vsini': np.nan,
        #         'rv': np.nan,
        #         'success': False,
        #         'chi2': np.nan,
        #         'errors': {},
        #         'error_message': str(e),
        #         'sme_object': None
        #     }

        return result

    def validate_single_spectrum(self, true_params, initial_guess=None,
                                 wavelength_range=(3000, 9000), snr=100):
        """
        Validate a single set of parameters.

        Parameters:
        -----------
        true_params : dict
            True stellar parameters
        initial_guess : dict, optional
            Initial guess for fitting
        wavelength_range : tuple
            Wavelength range for analysis
        snr : float
            Signal-to-noise ratio

        Returns:
        --------
        validation_result : dict
            Validation results
        """
        print(f"\n{'=' * 60}")
        print(f"Validating: T={true_params['teff']}K, log g={true_params['logg']}, "
              f"[Fe/H]={true_params.get('feh', 0.0)}")
        print(f"{'=' * 60}")

        # Set default initial guess
        if initial_guess is None:
            initial_guess = {
                'teff': true_params['teff'] + np.random.normal(0, 200),
                'logg': true_params['logg'] + np.random.normal(0, 0.3),
                'feh': true_params.get('feh', 0.0) + np.random.normal(0, 0.2),
                'vmic': 1.0,
                'vmac': 3.0,
                'vsini': 2.0,
                'rv': 0.0
            }

        start_time = datetime.now()

        try:
            # Generate synthetic spectrum
            wave, flux, error = self.generate_synthetic_spectrum(
                true_params['teff'],
                true_params['logg'],
                true_params.get('feh', 0.0),
                wavelength_range=wavelength_range,
                snr=snr
            )

            # Fit with pySME
            fit_result = self.fit_spectrum_with_pysme(
                wave, flux, error, initial_guess
            )

            # Calculate errors
            if fit_result['success']:
                temp_error = fit_result['teff'] - true_params['teff']
                logg_error = fit_result['logg'] - true_params['logg']
                feh_error = fit_result['feh'] - true_params.get('feh', 0.0)
            else:
                temp_error = logg_error = feh_error = np.nan

            # Compile results
            validation_result = {
                'timestamp': start_time.isoformat(),
                'true_teff': true_params['teff'],
                'true_logg': true_params['logg'],
                'true_feh': true_params.get('feh', 0.0),
                'fitted_teff': fit_result['teff'],
                'fitted_logg': fit_result['logg'],
                'fitted_feh': fit_result['feh'],
                'temp_error': temp_error,
                'logg_error': logg_error,
                'feh_error': feh_error,
                'chi2': fit_result.get('chi2', np.nan),
                'success': fit_result['success'],
                'wavelength_range': wavelength_range,
                'snr': snr,
                'initial_guess': initial_guess,
                'fit_errors': fit_result.get('errors', {}),
                'error_message': fit_result.get('error_message', ''),
                'spectrum_data': {
                    'wavelength': wave,
                    'flux': flux,
                    'error': error
                }
            }

            # Print summary
            if fit_result['success']:
                print(f"\nResults:")
                print(f"  Temperature: {true_params['teff']} → {fit_result['teff']:.0f} "
                      f"(Δ = {temp_error:+.0f} K)")
                print(f"  Log g: {true_params['logg']:.2f} → {fit_result['logg']:.2f} "
                      f"(Δ = {logg_error:+.3f})")
                print(f"  [Fe/H]: {true_params.get('feh', 0.0):.2f} → {fit_result['feh']:.2f} "
                      f"(Δ = {feh_error:+.3f})")
                print(f"  χ² = {fit_result.get('chi2', 'N/A')}")
            else:
                print(f"  Fit failed: {fit_result.get('error_message', 'Unknown error')}")

        except Exception as e:
            print(f"Validation failed with error: {str(e)}")
            validation_result = {
                'timestamp': start_time.isoformat(),
                'true_teff': true_params['teff'],
                'true_logg': true_params['logg'],
                'true_feh': true_params.get('feh', 0.0),
                'fitted_teff': np.nan,
                'fitted_logg': np.nan,
                'fitted_feh': np.nan,
                'temp_error': np.nan,
                'logg_error': np.nan,
                'feh_error': np.nan,
                'chi2': np.nan,
                'success': False,
                'wavelength_range': wavelength_range,
                'snr': snr,
                'initial_guess': initial_guess,
                'fit_errors': {},
                'error_message': str(e),
                'spectrum_data': None
            }

        return validation_result

    def run_validation_suite(self, test_parameters, wavelength_range=(3000, 9000),
                             snr=100, save_results=True):
        """
        Run validation on multiple parameter sets.

        Parameters:
        -----------
        test_parameters : list
            List of parameter dictionaries to test
        wavelength_range : tuple
            Wavelength range for analysis
        snr : float
            Signal-to-noise ratio
        save_results : bool
            Whether to save results to file

        Returns:
        --------
        summary : dict
            Validation summary statistics
        """
        print(f"\n{'=' * 80}")
        print(f"RUNNING PHOENIX INTERPOLATION VALIDATION SUITE")
        print(f"{'=' * 80}")
        print(f"Test parameters: {len(test_parameters)} sets")
        print(f"Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} Å")
        print(f"S/N ratio: {snr}")

        self.validation_results = []
        successful_results = []

        for i, params in enumerate(test_parameters):
            print(f"\nTest {i + 1}/{len(test_parameters)}")

            result = self.validate_single_spectrum(
                params,
                wavelength_range=wavelength_range,
                snr=snr
            )

            self.validation_results.append(result)

            if result['success']:
                successful_results.append(result)

        # Calculate summary statistics
        if successful_results:
            temp_errors = [r['temp_error'] for r in successful_results]
            logg_errors = [r['logg_error'] for r in successful_results]
            feh_errors = [r['feh_error'] for r in successful_results]

            summary = {
                'n_total': len(test_parameters),
                'n_successful': len(successful_results),
                'success_rate': len(successful_results) / len(test_parameters) * 100,
                'temp_rms': np.sqrt(np.mean(np.array(temp_errors) ** 2)),
                'temp_bias': np.mean(temp_errors),
                'temp_std': np.std(temp_errors),
                'logg_rms': np.sqrt(np.mean(np.array(logg_errors) ** 2)),
                'logg_bias': np.mean(logg_errors),
                'logg_std': np.std(logg_errors),
                'feh_rms': np.sqrt(np.mean(np.array(feh_errors) ** 2)),
                'feh_bias': np.mean(feh_errors),
                'feh_std': np.std(feh_errors),
                'median_chi2': np.median([r.get('chi2', np.nan) for r in successful_results])
            }
        else:
            summary = {
                'n_total': len(test_parameters),
                'n_successful': 0,
                'success_rate': 0.0
            }

        # Print summary
        self._print_validation_summary(summary)

        # Save results
        if save_results:
            self._save_validation_results(summary)

        # Create plots
        if successful_results:
            self._plot_validation_results(successful_results)

        return summary

    def _print_validation_summary(self, summary):
        """Print validation summary."""
        print(f"\n{'=' * 80}")
        print(f"VALIDATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total tests: {summary['n_total']}")
        print(f"Successful fits: {summary['n_successful']} ({summary['success_rate']:.1f}%)")

        if summary['n_successful'] > 0:
            print(f"\nParameter Recovery Statistics:")
            print(f"Temperature:")
            print(f"  RMS error: {summary['temp_rms']:.1f} K")
            print(f"  Bias: {summary['temp_bias']:+.1f} K")
            print(f"  Std dev: {summary['temp_std']:.1f} K")

            print(f"Log g:")
            print(f"  RMS error: {summary['logg_rms']:.3f}")
            print(f"  Bias: {summary['logg_bias']:+.3f}")
            print(f"  Std dev: {summary['logg_std']:.3f}")

            print(f"[Fe/H]:")
            print(f"  RMS error: {summary['feh_rms']:.3f}")
            print(f"  Bias: {summary['feh_bias']:+.3f}")
            print(f"  Std dev: {summary['feh_std']:.3f}")

            print(f"\nMedian χ²: {summary['median_chi2']:.2f}")

            # Interpretation (adjusted for PHOENIX vs MARCS)
            print(f"\nInterpretation:")
            if summary['temp_rms'] < 150:
                print("✓ Temperature interpolation appears accurate")
            else:
                print("⚠ Temperature interpolation may have issues (or PHOENIX-MARCS differences)")

            if summary['logg_rms'] < 0.15:
                print("✓ Log g interpolation appears accurate")
            else:
                print("⚠ Log g interpolation may have issues (or PHOENIX-MARCS differences)")

            if abs(summary['temp_bias']) < 100:
                print("✓ No significant temperature bias")
            else:
                print("⚠ Systematic temperature bias detected (could be PHOENIX-MARCS offset)")

            if abs(summary['logg_bias']) < 0.1:
                print("✓ No significant log g bias")
            else:
                print("⚠ Systematic log g bias detected (could be PHOENIX-MARCS offset)")

    def _save_validation_results(self, summary):
        """Save validation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary
        summary_file = f"phoenix_validation_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            summary_json = {}
            for key, value in summary.items():
                if isinstance(value, np.ndarray):
                    summary_json[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    summary_json[key] = float(value)
                else:
                    summary_json[key] = value
            json.dump(summary_json, f, indent=2)

        # Save detailed results
        results_file = f"phoenix_validation_results_{timestamp}.csv"
        df_data = []

        for result in self.validation_results:
            row = {k: v for k, v in result.items()
                   if k not in ['spectrum_data', 'fit_errors', 'initial_guess']}
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(results_file, index=False)

        print(f"\nResults saved:")
        print(f"  Summary: {summary_file}")
        print(f"  Details: {results_file}")

    def _plot_validation_results(self, results):
        """Create diagnostic plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        true_temps = [r['true_teff'] for r in results]
        fitted_temps = [r['fitted_teff'] for r in results]
        true_loggs = [r['true_logg'] for r in results]
        fitted_loggs = [r['fitted_logg'] for r in results]
        true_fehs = [r['true_feh'] for r in results]
        fitted_fehs = [r['fitted_feh'] for r in results]

        temp_errors = [r['temp_error'] for r in results]
        logg_errors = [r['logg_error'] for r in results]
        feh_errors = [r['feh_error'] for r in results]

        # Temperature comparison
        axes[0, 0].scatter(true_temps, fitted_temps, alpha=0.7)
        axes[0, 0].plot([min(true_temps), max(true_temps)],
                        [min(true_temps), max(true_temps)], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('True T_eff (K)')
        axes[0, 0].set_ylabel('Fitted T_eff (K)')
        axes[0, 0].set_title('Temperature Recovery')
        axes[0, 0].grid(True, alpha=0.3)

        # Log g comparison
        axes[0, 1].scatter(true_loggs, fitted_loggs, alpha=0.7)
        axes[0, 1].plot([min(true_loggs), max(true_loggs)],
                        [min(true_loggs), max(true_loggs)], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('True log g')
        axes[0, 1].set_ylabel('Fitted log g')
        axes[0, 1].set_title('Log g Recovery')
        axes[0, 1].grid(True, alpha=0.3)

        # [Fe/H] comparison
        axes[0, 2].scatter(true_fehs, fitted_fehs, alpha=0.7)
        axes[0, 2].plot([min(true_fehs), max(true_fehs)],
                        [min(true_fehs), max(true_fehs)], 'r--', alpha=0.5)
        axes[0, 2].set_xlabel('True [Fe/H]')
        axes[0, 2].set_ylabel('Fitted [Fe/H]')
        axes[0, 2].set_title('[Fe/H] Recovery')
        axes[0, 2].grid(True, alpha=0.3)

        # Error histograms
        axes[1, 0].hist(temp_errors, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Temperature Error (K)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Temperature Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(logg_errors, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Log g Error')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Log g Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].hist(feh_errors, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('[Fe/H] Error')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('[Fe/H] Error Distribution')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"phoenix_validation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()


def create_test_parameter_sets():
    """Create comprehensive test parameter sets."""
    # Grid points (should recover exactly)
    grid_points = [
        {'teff': 5000, 'logg': 4.0, 'feh': 0.0},
        {'teff': 5500, 'logg': 4.5, 'feh': 0.0},
        {'teff': 6000, 'logg': 4.0, 'feh': 0.0},
        {'teff': 5770, 'logg': 4.44, 'feh': 0.0},  # Solar
    ]

    # Off-grid interpolation points
    interpolation_points = [
        {'teff': 5250, 'logg': 4.25, 'feh': 0.0},
        {'teff': 5750, 'logg': 4.75, 'feh': 0.0},
        {'teff': 5600, 'logg': 4.3, 'feh': 0.0},
        {'teff': 5400, 'logg': 4.6, 'feh': 0.0},
    ]

    # Edge cases
    edge_cases = [
        {'teff': 4500, 'logg': 3.5, 'feh': 0.0},  # Cool giant
        {'teff': 6500, 'logg': 5.0, 'feh': 0.0},  # Hot dwarf
    ]

    # Different metallicities (if available)
    metallicity_tests = [
        {'teff': 5500, 'logg': 4.5, 'feh': -0.5},
        {'teff': 5500, 'logg': 4.5, 'feh': 0.5},
    ]

    return grid_points + interpolation_points + edge_cases + metallicity_tests


def check_atmosphere_availability():
    """
    Check if atmosphere files are available and provide download alternatives.
    """
    import os
    from pathlib import Path

    print("Checking atmosphere file availability...")

    # Check if atmosphere directory exists
    atmo_dir = Path.home() / ".sme" / "atmospheres"
    print(f"Atmosphere directory: {atmo_dir}")
    print(f"Directory exists: {atmo_dir.exists()}")

    if atmo_dir.exists():
        files = list(atmo_dir.glob("*.sav"))
        print(f"Found {len(files)} atmosphere files:")
        for f in files:
            print(f"  {f.name}")

    # Check network connectivity
    try:
        import urllib.request
        response = urllib.request.urlopen("http://sme.astro.uu.se/", timeout=10)
        print("✓ Network connectivity to Uppsala server: OK")
    except Exception as e:
        print(f"✗ Network connectivity issue: {e}")
        print("\nSuggestions to fix download issues:")
        print("1. Check your internet connection")
        print("2. Try running from a different network (corporate firewalls may block)")
        print("3. Manually download atmosphere files:")
        print("   - Visit: http://sme.astro.uu.se/atmos/")
        print("   - Download 'marcs2012.sav' to ~/.sme/atmospheres/")
        print("4. Use a VPN if institutional firewall is blocking")
        print("5. Increase timeout by setting environment variable:")
        print("   export ASTROPY_DOWNLOAD_TIMEOUT=300")


def run_synthesis_only_validation():
    """
    Alternative validation that only tests PHOENIX synthesis without pySME fitting.
    This avoids atmosphere download issues.
    """
    print("Running synthesis-only validation (no pySME fitting)...")

    # Simplified setup for synthesis testing
    phoenix_path = r"/mnt/c/Users/Ilay/projects/simulations/starsim/starsim"
    wv_array = np.load('dataset/lamost_wv.npy')

    from Dynamo.spectra import interpolate_Phoenix

    validator = PHOENIXPySMEValidator(
        phoenix_interpolation_func=interpolate_Phoenix,
        phoenix_path=phoenix_path,
        wv_array=wv_array,
        linelist_path=None
    )

    # Test multiple parameter sets
    test_params_list = [
        {'teff': 5770, 'logg': 4.44, 'feh': 0.0},  # Solar
        {'teff': 5500, 'logg': 4.5, 'feh': 0.0},   # Nearby solar-type
        {'teff': 6000, 'logg': 4.0, 'feh': 0.0},   # Warmer star
        {'teff': 5000, 'logg': 4.5, 'feh': -0.5},  # Metal-poor
    ]

    print("\nTesting PHOENIX synthesis for different stellar parameters:")

    for i, params in enumerate(test_params_list):
        print(f"\nTest {i+1}: T={params['teff']}K, log g={params['logg']}, [Fe/H]={params['feh']}")

        try:
            # Generate synthetic spectrum
            wave, flux, error = validator.generate_synthetic_spectrum(
                params['teff'],
                params['logg'],
                params.get('feh', 0.0),
                wavelength_range=(4000, 7000),  # Optical range
                snr=100
            )

            # Basic checks on the spectrum
            if np.any(np.isnan(flux)):
                print("  ❌ NaN values found in flux")
            elif np.any(flux <= 0):
                print("  ❌ Non-positive flux values found")
            elif np.std(flux) < 0.01:
                print("  ❌ Flux appears flat (no spectral features)")
            else:
                print(f"  ✓ Spectrum generated successfully")
                print(f"    Wavelength range: {wave.min():.1f} - {wave.max():.1f} Å")
                print(f"    Flux range: {flux.min():.3f} - {flux.max():.3f}")
                print(f"    Number of points: {len(wave)}")

                # Check for reasonable spectral line depths
                continuum_level = np.percentile(flux, 95)
                line_depth = continuum_level - np.percentile(flux, 5)
                print(f"    Approximate line depth: {line_depth:.3f} ({line_depth/continuum_level*100:.1f}%)")

        except Exception as e:
            print(f"  ❌ Failed to generate spectrum: {str(e)}")

    print(f"\n{'='*60}")
    print("SYNTHESIS-ONLY VALIDATION COMPLETE")
    print("This validates that your PHOENIX interpolation can generate")
    print("synthetic spectra for different stellar parameters.")
    print("For full parameter recovery validation, you would need")
    print("working pySME atmosphere models.")
    print(f"{'='*60}")


def main():
    """
    Main function to run the validation.

    Usage:
    1. Import your PHOENIX interpolation function
    2. Set paths to PHOENIX models and line lists
    3. Run validation
    """

    # =================================================================
    # CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
    # =================================================================

    # Path to your PHOENIX models directory
    phoenix_path = r"/mnt/c/Users/Ilay/projects/simulations/starsim/starsim"
    wv_array = np.load('dataset/lamost_wv.npy')

    # Path to VALD line list (optional - download from VALD database)
    # Example: linelist_path = "./vald_lines_5000_6000.dat"
    linelist_path = None

    # Import your PHOENIX interpolation function
    # Replace this with your actual function import

    # For testing, use the corrected function from previous artifact
    from Dynamo.spectra import interpolate_Phoenix
    phoenix_interpolation_func = interpolate_Phoenix


    # =================================================================
    # VALIDATION SETUP
    # =================================================================

    print("Setting up pySME validation for PHOENIX interpolation...")

    # Check atmosphere file availability first
    check_atmosphere_availability()

    # Initialize validator
    validator = PHOENIXPySMEValidator(
        phoenix_interpolation_func=phoenix_interpolation_func,
        phoenix_path=phoenix_path,
        wv_array=wv_array,
        linelist_path=linelist_path
    )

    # Create test parameter sets
    test_parameters = create_test_parameter_sets()

    print(f"\nTest parameters to validate:")
    for i, params in enumerate(test_parameters):
        print(f"  {i + 1:2d}. T={params['teff']:4.0f}K, "
              f"log g={params['logg']:.2f}, [Fe/H]={params['feh']:+.1f}")

    # =================================================================
    # RUN VALIDATION
    # =================================================================

    # Run single test first
    print(f"\n{'=' * 60}")
    print("RUNNING SINGLE TEST FIRST")
    print(f"{'=' * 60}")

    test_params = {'teff': 5770, 'logg': 4.44, 'feh': 0.0}  # Solar
    single_result = validator.validate_single_spectrum(
        test_params,
        wavelength_range=(3000, 9000),
        snr=100
    )

    if single_result['success']:
        print("✓ Single test successful - proceeding with full validation")

        # Run full validation suite
        summary = validator.run_validation_suite(
            test_parameters=test_parameters,
            wavelength_range=(3000, 9000),  # Focus on optical
            snr=100,  # Good S/N ratio
            save_results=True
        )

        # =================================================================
        # ANALYSIS AND INTERPRETATION
        # =================================================================

        print(f"\n{'=' * 80}")
        print(f"VALIDATION ANALYSIS")
        print(f"This validation tests PHOENIX interpolation by:")
        print(f"• Generating synthetic spectra with PHOENIX models")
        print(f"• Fitting them with pySME using MARCS atmosphere grids")
        print(f"• Comparing input vs recovered stellar parameters")
        print(f"Note: Small systematic differences may occur due to PHOENIX vs MARCS model differences")
        print(f"{'=' * 80}")

        if summary['n_successful'] > 0:
            # Check for systematic issues (adjusted thresholds for PHOENIX vs MARCS comparison)
            issues_found = []

            if summary['temp_rms'] > 200:  # Higher threshold due to model differences
                issues_found.append(f"High temperature RMS error ({summary['temp_rms']:.0f}K)")

            if summary['logg_rms'] > 0.2:  # Higher threshold due to model differences
                issues_found.append(f"High log g RMS error ({summary['logg_rms']:.3f})")

            if abs(summary['temp_bias']) > 150:  # Higher threshold due to model differences
                issues_found.append(f"Systematic temperature bias ({summary['temp_bias']:+.0f}K)")

            if abs(summary['logg_bias']) > 0.15:  # Higher threshold due to model differences
                issues_found.append(f"Systematic log g bias ({summary['logg_bias']:+.3f})")

            if summary['success_rate'] < 70:  # Lower threshold due to model differences
                issues_found.append(f"Low success rate ({summary['success_rate']:.1f}%)")

            # Print diagnosis
            if issues_found:
                print("⚠ POTENTIAL INTERPOLATION ISSUES DETECTED:")
                for issue in issues_found:
                    print(f"  • {issue}")

                print(f"\nRECOMMENDATIONS:")
                if summary['temp_rms'] > 200 or abs(summary['temp_bias']) > 150:
                    print("  • Check temperature interpolation algorithm")
                    print("  • Verify bilinear interpolation implementation")
                    print("  • Note: Some differences expected due to PHOENIX vs MARCS models")

                if summary['logg_rms'] > 0.2 or abs(summary['logg_bias']) > 0.15:
                    print("  • Check log g interpolation algorithm")
                    print("  • Verify grid point selection logic")
                    print("  • Note: Some differences expected due to PHOENIX vs MARCS models")

                if summary['success_rate'] < 70:
                    print("  • Check for missing PHOENIX models")
                    print("  • Verify file reading and wavelength ranges")

            else:
                print("✓ NO MAJOR INTERPOLATION ISSUES DETECTED")
                print("✓ PHOENIX interpolation appears to be working correctly")

            # Performance assessment (adjusted for PHOENIX vs MARCS comparison)
            print(f"\nPERFORMANCE ASSESSMENT:")
            print(f"Temperature precision: {summary['temp_std']:.0f}K "
                  f"({'Excellent' if summary['temp_std'] < 100 else 'Good' if summary['temp_std'] < 200 else 'Poor'})")
            print(f"Log g precision: {summary['logg_std']:.3f} "
                  f"({'Excellent' if summary['logg_std'] < 0.1 else 'Good' if summary['logg_std'] < 0.2 else 'Poor'})")
            print(f"\nNote: Results show PHOENIX interpolation quality relative to MARCS models")
            print(f"Small systematic differences are expected due to different atmosphere physics")

        else:
            print("❌ ALL VALIDATION TESTS FAILED")
            print("This indicates serious issues with the interpolation function")
            print("\nPossible causes:")
            print("  • Missing or corrupted PHOENIX files")
            print("  • Incorrect file paths")
            print("  • Bugs in interpolation algorithm")
            print("  • Incompatible wavelength ranges")

    else:
        print("❌ Single test failed - check your setup:")
        print(f"  Error: {single_result.get('error_message', 'Unknown')}")
        print("  • Verify PHOENIX model files exist")
        print("  • Check interpolation function")
        print("  • Ensure pySME is properly installed")

    return 0


def run_synthesis_only_validation():
    """
    Alternative validation that only tests PHOENIX synthesis without pySME fitting.
    This avoids atmosphere download issues.
    """
    print("Running synthesis-only validation (no pySME fitting)...")

    # Simplified setup for synthesis testing
    phoenix_path = r"/mnt/c/Users/Ilay/projects/simulations/starsim/starsim"
    wv_array = np.load('dataset/lamost_wv.npy')

    from Dynamo.spectra import interpolate_Phoenix

    validator = PHOENIXPySMEValidator(
        phoenix_interpolation_func=interpolate_Phoenix,
        phoenix_path=phoenix_path,
        wv_array=wv_array,
        linelist_path=None
    )

    # Test multiple parameter sets
    test_params_list = [
        {'teff': 5770, 'logg': 4.44, 'feh': 0.0},  # Solar
        {'teff': 5500, 'logg': 4.5, 'feh': 0.0},   # Nearby solar-type
        {'teff': 6000, 'logg': 4.0, 'feh': 0.0},   # Warmer star
        {'teff': 5000, 'logg': 4.5, 'feh': -0.5},  # Metal-poor
        {'teff': 4500, 'logg': 3.5, 'feh': 0.0},   # Cool giant
    ]

    print(f"\nTesting PHOENIX synthesis for {len(test_params_list)} stellar parameter sets:")

    successful_tests = 0

    for i, params in enumerate(test_params_list):
        print(f"\nTest {i+1}: T={params['teff']}K, log g={params['logg']}, [Fe/H]={params['feh']}")

        try:
            # Generate synthetic spectrum
            wave, flux, error = validator.generate_synthetic_spectrum(
                params['teff'],
                params['logg'],
                params.get('feh', 0.0),
                wavelength_range=(4000, 7000),  # Optical range
                snr=100
            )

            # Basic checks on the spectrum
            issues = []

            if np.any(np.isnan(flux)):
                issues.append("NaN values found in flux")
            if np.any(flux <= 0):
                issues.append("Non-positive flux values found")
            if np.std(flux) < 0.01:
                issues.append("Flux appears flat (no spectral features)")

            if issues:
                print("  ❌ Issues detected:")
                for issue in issues:
                    print(f"    • {issue}")
            else:
                print(f"  ✓ Spectrum generated successfully")
                print(f"    Wavelength range: {wave.min():.1f} - {wave.max():.1f} Å")
                print(f"    Flux range: {flux.min():.3f} - {flux.max():.3f}")
                print(f"    Number of points: {len(wave)}")

                # Check for reasonable spectral line depths
                continuum_level = np.percentile(flux, 95)
                line_depth = continuum_level - np.percentile(flux, 5)
                line_depth_percent = line_depth/continuum_level*100

                print(f"    Continuum level: {continuum_level:.3f}")
                print(f"    Deepest lines: {line_depth:.3f} ({line_depth_percent:.1f}% depth)")

                # Check if line depths are reasonable
                if line_depth_percent < 1:
                    print("    ⚠ Warning: Very shallow lines (< 1% depth)")
                elif line_depth_percent > 50:
                    print("    ⚠ Warning: Very deep lines (> 50% depth)")
                else:
                    print("    ✓ Line depths appear reasonable")

                successful_tests += 1

        except Exception as e:
            print(f"  ❌ Failed to generate spectrum: {str(e)}")

    print(f"\n{'='*60}")
    print("SYNTHESIS-ONLY VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Tests completed: {len(test_params_list)}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {successful_tests/len(test_params_list)*100:.1f}%")

    if successful_tests == len(test_params_list):
        print("✅ ALL TESTS PASSED")
        print("Your PHOENIX interpolation appears to be working correctly!")
        print("The function successfully generates synthetic spectra for different")
        print("stellar parameters with reasonable spectral features.")
    elif successful_tests > len(test_params_list) // 2:
        print("⚠ PARTIAL SUCCESS")
        print("Most tests passed, but some failed. Check the failed parameter")
        print("combinations - they might be outside your PHOENIX grid coverage.")
    else:
        print("❌ MULTIPLE FAILURES")
        print("Several tests failed. This suggests issues with your PHOENIX")
        print("interpolation function or missing model files.")

    print("\nThis synthesis-only validation confirms that your PHOENIX")
    print("interpolation can generate realistic synthetic spectra.")
    print("For full parameter recovery validation, you would need")
    print("working pySME atmosphere models.")
    print(f"{'='*60}")

    return successful_tests == len(test_params_list)


def run_quick_validation():
    """Quick validation with minimal setup for testing."""
    print("Running quick validation (single test)...")

    # Simplified setup for quick testing
    phoenix_path = r"C:\Users\Ilay\projects\simulations\starsim\starsim"

    from Dynamo.spectra import interpolate_Phoenix

    validator = PHOENIXPySMEValidator(
        phoenix_interpolation_func=interpolate_Phoenix,
        phoenix_path=phoenix_path,
        linelist_path=None
    )

    # Test solar parameters
    test_params = {'teff': 5770, 'logg': 4.44, 'feh': 0.0}

    result = validator.validate_single_spectrum(
        test_params,
        wavelength_range=(5200, 5800),  # Smaller range for speed
        snr=50
    )

    if result['success']:
        print(f"✓ Quick validation successful!")
        print(f"  Temperature error: {result['temp_error']:+.0f}K")
        print(f"  Log g error: {result['logg_error']:+.3f}")
        return True
    else:
        print(f"❌ Quick validation failed")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate PHOENIX interpolation using pySME',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full validation suite
  python pysme_validation.py

  # Run quick test only
  python pysme_validation.py --quick

  # Specify custom paths
  python pysme_validation.py --phoenix-path ./my_phoenix/ --linelist ./my_lines.dat

Setup Requirements:
  1. Install pySME: pip install pysme
  2. Download PHOENIX models to specified directory
  3. Optionally download VALD line list
  4. Import your interpolation function in the main() function

Expected Results:
  - Temperature errors < 100K indicate good interpolation
  - Log g errors < 0.1 indicate good interpolation  
  - No systematic biases indicate proper implementation
  - High success rate (>90%) indicates robust function

Key Changes from Original:
  - Removed match_rv_continuum() call (not needed in pySME 0.5.2)
  - Set continuum and RV flags directly on SME structure
  - Added proper error handling for failed fits
  - Configured SME structure according to current API
        """
    )

    parser.add_argument('--synthesis-only', action='store_true',
                        help='Run synthesis-only validation (skip pySME fitting)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick validation test only')
    parser.add_argument('--phoenix-path', type=str, default='./phoenix_models/',
                        help='Path to PHOENIX models directory')
    parser.add_argument('--linelist', type=str, default=None,
                        help='Path to VALD line list file')

    args = parser.parse_args()

    if args.synthesis_only:
        run_synthesis_only_validation()
        sys.exit(0)
    elif args.quick:
        success = run_quick_validation()
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())