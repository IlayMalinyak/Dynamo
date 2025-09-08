import tqdm
import numpy as np
import emcee
import gc
from pathlib import Path
import configparser
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import sys
from . import nbspectra
from .cgs_consts import *
from . rotator import Rotator
from . import noise
from . import spots
from . import spectra

# initialize numba
nbspectra.dummy()


class Star:
    """
    Starsim class for stellar simulation with configuration-based initialization.

    This class reads a configuration file and sets up parameters for stellar
    simulations, including star properties, spots, planets, and observational
    characteristics.
    """

    def __init__(self, conf_file_path='starsim.conf'):
        # Set up paths and configuration
        self.path = Path(__file__).parent
        self.conf_file_path = self.path / conf_file_path
        self.conf_file = self._initialize_config()

        # Initialize main parameters
        self._initialize_file_params()
        self._initialize_general_params()
        self._initialize_star_params()
        self._initialize_limb_darkening_params()
        self._initialize_spots_params()
        self._initialize_planet_params()

        # Read and validate spot map
        self._load_spot_map()

        # Validate simulation mode and CCF template
        self._validate_modes()

    def _initialize_config(self):
        """Initialize configuration file parser."""
        conf_file_object = configparser.ConfigParser(inline_comment_prefixes='#')
        if not conf_file_object.read([self.conf_file_path]):
            print(f"Configuration file at {self.conf_file_path} could not be read.")
            sys.exit(1)
        return conf_file_object

    def _get_config_value(self, section, key, value_type=str):
        """Safely retrieve configuration values with type conversion."""
        try:
            return value_type(self.conf_file.get(section, key))
        except Exception as e:
            print(f"Error reading {section}.{key}: {e}")
            return None

    def _initialize_file_params(self):
        """Initialize file-related parameters."""
        self.models_root = self._get_config_value(section='files', key='models_root')
        self.filter_name = self._get_config_value('files', 'filter_name')
        self.spectra_filter_name = self._get_config_value('files', 'spectra_filter_name')
        self.orders_CRX_filename = self._get_config_value('files', 'orders_CRX_filename')

        self.models_root = Path(self.models_root)

    def _initialize_general_params(self):
        """Initialize general simulation parameters."""
        self.simulation_mode = self._get_config_value('general', 'simulation_mode')
        self.wavelength_lower_limit = self._get_config_value('general', 'wavelength_lower_limit', float)
        self.wavelength_upper_limit = self._get_config_value('general', 'wavelength_upper_limit', float)
        self.spectra_resolution = self._get_config_value('general', 'spectra_resolution', int)
        self.n_grid_rings = self._get_config_value('general', 'n_grid_rings', int)

    def _initialize_star_params(self):
        """Initialize star-related parameters."""
        star_params = {
            'radius': float, 'mass': float, 'rotation_period': float,
            'inclination': float, 'temperature_photosphere': float,
            'spot_T_contrast_max': float, 'spot_T_contrast_min': float, 'facula_T_contrast': float,
            'convective_shift': float, 'logg': float,
            'facular_area_ratio': float, 'differential_rotation': float
        }
        for param, type_conv in star_params.items():
            setattr(self, param, self._get_config_value('star', param, type_conv))

    def _initialize_limb_darkening_params(self):
        """Initialize limb darkening parameters."""
        self.use_phoenix_limb_darkening = self._get_config_value('LD', 'use_phoenix_limb_darkening', int)
        self.limb_darkening_law = self._get_config_value('LD', 'limb_darkening_law')
        self.limb_darkening_q1 = self._get_config_value('LD', 'limb_darkening_q1', float)
        self.limb_darkening_q2 = self._get_config_value('LD', 'limb_darkening_q2', float)

    def _initialize_spots_params(self):
        """Initialize spots-related parameters."""
        self.spots_evo_law = self._get_config_value('spots', 'spots_evo_law')
        self.plot_grid_map = self._get_config_value('spots', 'plot_grid_map', int)
        self.reference_time = self._get_config_value('spots', 'reference_time', float)
        self.spots_decay_time = self._get_config_value('spots', 'spots_decay_time', float)
        self.max_n_spots = self._get_config_value('spots', 'max_n_spots', float)


    def _initialize_noise_params(self):
        """
        Initialize noise-related parameters.
        """
        self.cdpp = self._get_config_value('noise', 'cdpp', float)
        self.outliers_rate = self._get_config_value('noise', 'outliers_rate', float)
        self.flicker = self._get_config_value('noise', 'flicker', float)

    def _initialize_planet_params(self):
        """Initialize planet-related parameters."""
        planet_params = {
            'planet_period': float, 'planet_transit_t0': float,
            'planet_radius': float, 'planet_impact_param': float,
            'planet_spin_orbit_angle': float, 'simulate_planet': int,
            'planet_semi_amplitude': float, 'planet_esinw': float,
            'planet_ecosw': float
        }
        for param, type_conv in planet_params.items():
            value = self._get_config_value('planet', param, type_conv)
            if param == 'planet_spin_orbit_angle':
                value *= np.pi / 180  # Convert to radians
            setattr(self, param, value)

    def _load_spot_map(self):
        """Load and validate spot map."""
        pathspots = self.path / 'spotmap.dat'
        self.spot_map = np.loadtxt(pathspots)

        if self.spot_map.ndim == 0:
            sys.exit('The spot map file spotmap.dat is empty')
        elif self.spot_map.ndim == 1:
            self.spot_map = np.array([self.spot_map])

    def _validate_modes(self):
        """Validate simulation mode and CCF template."""
        valid_simulation_modes = ['grid', 'fast']
        if self.simulation_mode not in valid_simulation_modes:
            sys.exit(f'Invalid simulation mode. Choose from {valid_simulation_modes}')

    @property
    def temperature_spot(self):
        return self.temperature_photosphere - self.spot_T_contrast

    @property
    def temperature_facula(self):
        return self.temperature_photosphere + self.facula_T_contrast

    @property
    def vsini(self):
        """Calculate projected rotational velocity."""
        # Calculate equatorial velocity (v)
        # v = 2πR/P
        radius_km = self.radius * R_sun / 1e5
        period_seconds = self.rotation_period * 24 * 60 * 60
        v_equatorial = 2 * np.pi * radius_km / period_seconds  # km/s

        # Calculate the projected velocity (vsini)
        vsini = v_equatorial * np.sin(np.pi/2 - self.inclination)
        return  vsini

    @property
    def planet_semi_major_axis(self):
        """Calculate planet's semi-major axis in stellar radius units."""
        return 4.2097 * self.planet_period ** (2 / 3) * self.mass ** (1 / 3) / self.radius

    def set_stellar_parameters(self, params_dict):
        """
        Update stellar parameters based on optimized values.

        Args:
            p (list): List of optimized parameter values
        """
        self.spot_T_contrast_min = min(params_dict['Activity Rate'] * 10, 500)
        self.spot_T_contrast_max = self.spot_T_contrast_min + 50
        self.differential_rotation = params_dict['Shear']
        self.rotation_period = params_dict['Period']
        self.convective_shift = params_dict['convective_shift']
        self.radius = params_dict['R']
        self.mass = params_dict['mass']
        self.luminosity = params_dict['L']
        self.temperature_photosphere = params_dict['Teff']
        self.logg = params_dict['logg']
        self.feh = params_dict['FeH']
        self.age = params_dict['age']
        self.cdpp = params_dict['CDPP']
        self.outliers_rate = params_dict['Outlier Rate']
        self.flicker = params_dict['Flicker Time Scale']
        self.inclination = 90 - np.rad2deg(params_dict['Inclination'])
        self.activity = params_dict['Activity Rate']
        self.cycle_len = params_dict['Cycle Length']
        self.cycle_overlap = params_dict['Cycle Overlap']
        self.spot_max_lat = params_dict['Spot Max']
        self.spot_min_lat = params_dict['Spot Min']
        self.spots_decay_time = params_dict['Decay Time']
        self.butterfly = params_dict['Butterfly']


    def set_planet_parameters(self, params_dict):
        self.simulate_planet = params_dict['simulate_planet']
        self.planet_period = params_dict['planet_period']
        self.planet_transit_t0 = params_dict['planet_transit_t0']
        self.planet_radius = params_dict['planet_radius']
        self.planet_impact_param = params_dict['planet_impact']
        self.planet_esinw = params_dict['planet_esinw']
        self.planet_ecosw = params_dict['planet_ecosw']
        self.planet_spin_orbit_angle = params_dict['planet_spin_orbit_angle']
        self.planet_semi_amplitude = params_dict['planet_semi_amplitude']


    def generate_spot_map(self, ndays):
        s = spots.SpotsGenerator()
        regions = s.emerge_regions(
            ndays=ndays,
            activity_level=self.activity,
            cycle_period=self.cycle_len,
            cycle_overlap=self.cycle_overlap,
            max_lat=self.spot_max_lat,
            min_lat=self.spot_min_lat,
            butterfly=self.butterfly,
            seed=1234
        )
        ref_time = regions[0]['nday'] if len(regions) > 0 else 0
        regions['nday'] -= ref_time
        self.reference_time = 0
        spots_config = np.zeros((len(regions), 14))
        spots_config[:, 0] = regions['nday']
        spots_duration = max(self.spots_decay_time * self.rotation_period, 50)
        spots_config[:, 1] = spots_duration * np.ones(len(spots_config))
        spots_config[:, 2] = np.rad2deg((regions['thpos'] + regions['thneg']) / 2)
        spots_config[:, 3] = np.rad2deg((regions['phpos'] + regions['phneg']) / 2)
        spots_config[:, 4] = np.sqrt(regions['bmax']) # bmax is proportional to area (radius ** 2)
        self.spot_map = spots_config
        self.n_grid_rings = int(max(10, 120/(spots_config[:, 4].min()))) if len(spots_config) > 0 else 10
        print("number of spots: ", len(regions), 'activity: ', self.activity)
        print("number of grid rings: ", self.n_grid_rings)

    def create_lamost_spectra(self, wv_array, ff_sp):
        """
        Create synthetic LAMOST-like spectra with proper disk integration and broadening.
        """
        mu, wvp_lc, photo_flux = spectra.interpolate_Phoenix_mu_lc(self,
                                                                   self.temperature_photosphere,
                                                                   self.logg,
                                                                   wv_array=wv_array)
        mu, wvp_lc, spot_flux = spectra.interpolate_Phoenix_mu_lc(self,
                                                                  self.temperature_spot,
                                                                  self.logg,
                                                                  wv_array=wv_array)

        disk_integrated_photo = np.zeros(len(wv_array))
        disk_integrated_spot = np.zeros(len(wv_array))

        # Sort mu in ascending order if needed
        if len(mu) > 1 and mu[0] > mu[-1]:
            mu = mu[::-1]
            photo_flux = photo_flux[::-1, :]
            spot_flux = spot_flux[::-1, :]

        # Proper disk integration: F = 2π ∫ I(μ) μ dμ
        for i in range(len(mu) - 1):
            delta_mu = mu[i + 1] - mu[i]
            # Trapezoidal rule
            weight_i = mu[i] * delta_mu / 2
            weight_i_plus_1 = mu[i + 1] * delta_mu / 2

            disk_integrated_photo += (photo_flux[i, :] * weight_i +
                                      photo_flux[i + 1, :] * weight_i_plus_1)
            disk_integrated_spot += (spot_flux[i, :] * weight_i +
                                     spot_flux[i + 1, :] * weight_i_plus_1)

        # Combine photosphere and spots using fill factor
        fill_factor = ff_sp.mean() / 100  # Assuming ff_sp is in percentage
        combined_spectrum = ((1 - fill_factor) * disk_integrated_photo +
                             fill_factor * disk_integrated_spot)

        # Apply stellar rotation broadening FIRST
        combined_spectrum = spectra.apply_rotational_broadening(wv_array, combined_spectrum, self.vsini)

        # Apply instrumental broadening (LAMOST resolution)
        R_lamost = 1800
        c = 299792.458  # km/s
        sigma_instrumental = c / R_lamost  # ~167 km/s
        combined_spectrum = spectra.apply_rotational_broadening(wv_array, combined_spectrum, sigma_instrumental)

        # Apply instrument sensitivity
        if self.spectra_filter_name != 'None':
            lamost_sensitivity = spectra.interpolate_filter(self, self.spectra_filter_name)
            # CORRECTED: Proper sensitivity application
            combined_spectrum = combined_spectrum * lamost_sensitivity(wv_array)
        else:
            lamost_sensitivity_values = self.create_synthetic_lamost_sensitivity(wv_array)
            combined_spectrum = combined_spectrum * lamost_sensitivity_values

        base_snr = self.cdpp * 2

        # Simple uniform SNR or wavelength-dependent
        if hasattr(self, 'spectra_filter_name') and self.spectra_filter_name != 'None':
            sensitivity_values = lamost_sensitivity(wv_array)
            wavelength_dependent_snr = base_snr * np.sqrt(sensitivity_values / np.max(sensitivity_values))
        else:
            wavelength_dependent_snr = np.full_like(wv_array, base_snr)

        # Add noise
        combined_spectrum_with_noise = np.zeros_like(combined_spectrum)
        for i in range(len(wv_array)):
            if combined_spectrum[i] > 0 and wavelength_dependent_snr[i] > 0:
                local_noise_level = combined_spectrum[i] / wavelength_dependent_snr[i]
                noise = np.random.normal(0, local_noise_level)
                combined_spectrum_with_noise[i] = max(combined_spectrum[i] + noise, 1e-10)
            else:
                combined_spectrum_with_noise[i] = max(combined_spectrum[i], 1e-10)

        return combined_spectrum_with_noise, wvp_lc

    def create_synthetic_lamost_sensitivity(self, wv_array):
        """
        Create a synthetic LAMOST sensitivity curve if no filter is provided.
        """
        # Simple model: peak around 5500Å, declining toward red/blue
        central_wl = 5500.0  # Angstroms
        width = 2000.0  # Angstroms

        sensitivity = np.exp(-0.5 * ((wv_array - central_wl) / width) ** 2)

        # Add some realistic features
        # Red cutoff around 9000Å
        red_cutoff = 1.0 / (1.0 + np.exp((wv_array - 8500) / 200))
        # Blue cutoff around 3800Å
        blue_cutoff = 1.0 / (1.0 + np.exp((3800 - wv_array) / 200))

        sensitivity = sensitivity * red_cutoff * blue_cutoff
        return sensitivity

    def compute_forward(self, t=None, wv_array=None):
        print(f"\ncomputing forward with: {len(self.spot_map)} spots and {int(self.simulate_planet)} planets")
        self.inclination = np.deg2rad(self.inclination)
        self.spot_T_contrast = 200
        self.tau_emerge = min(2, self.rotation_period * self.spots_decay_time / 10)
        self.tau_decay = max(self.rotation_period * self.spots_decay_time, 50)
        self.results = {}
        self.wavelength_lower_limit = float(self.conf_file.get('general', 'wavelength_lower_limit'))
        self.wavelength_upper_limit = float(self.conf_file.get('general', 'wavelength_upper_limit'))

        if t is None:
            sys.exit('Please provide a valid time in compute_forward(observables,t=time)')

        self.obs_times = t

        Ngrid_in_ring, cos_centers, proj_area, phi, theta, vec_grid = self.get_theta_phi()

        sini, wvp, photo_flux = spectra.interpolate_Phoenix_mu_lc_with_metallicity(
            self, self.temperature_photosphere, self.logg, self.feh, wv_array=wv_array, plot=True
        )
        sini, wvp, spot_flux = spectra.interpolate_Phoenix_mu_lc_with_metallicity(
            self, self.temperature_spot, self.logg, self.feh, wv_array=wv_array
        )
        # For LIGHT CURVES: Apply photometric filter and compute integrated flux
        f_filt_lc = spectra.interpolate_filter(self, self.filter_name)

        brigh_grid_ph, flx_ph = spectra.compute_immaculate_lc_with_vsini(
            self, Ngrid_in_ring, sini, cos_centers, proj_area,
            photo_flux, f_filt_lc, wvp, self.vsini
        )

        brigh_grid_sp, flx_sp = spectra.compute_immaculate_lc_with_vsini(
            self, Ngrid_in_ring, sini, cos_centers, proj_area,
            spot_flux, f_filt_lc, wvp, self.vsini
        )

        brigh_grid_fc, flx_fc = brigh_grid_sp, flx_sp  # if there are no faculae
        if self.facular_area_ratio > 0:
            raise NotImplementedError('facular calculation is not yet implemented')

        # Generate rotation light curve
        rotator = Rotator(self)
        spots_positions, FLUX, ff_ph, ff_sp, ff_fc, ff_pl = rotator.generate_rotating_photosphere_lc(
            Ngrid_in_ring, proj_area, cos_centers,
            brigh_grid_ph, brigh_grid_sp, brigh_grid_fc,
            flx_ph, vec_grid, plot_map=self.plot_grid_map
        )
        self.final_spots_positions = spots_positions

        # For SPECTRA: Use the SAME Phoenix models, just with spectroscopic filter
        # No need to recalculate Phoenix models!
        if wv_array is not None or hasattr(self, 'spectra_filter_name'):
            # Create spectra using the already-computed Phoenix models
            spectra_flux, wvp_spec = spectra.create_observed_spectra(
                self, wvp, photo_flux, spot_flux, sini, ff_sp,
                spectra_filter_name=self.spectra_filter_name
            )
        else:
            spectra_flux = None
            wvp_spec = wvp

        # Store results
        self.results['time'] = t
        self.results['lc'] = FLUX
        self.results['spectra'] = spectra_flux
        self.results['ff_ph'] = ff_ph
        self.results['ff_sp'] = ff_sp
        self.results['ff_pl'] = ff_pl
        self.results['ff_fc'] = ff_fc
        self.results['flp'] = flx_ph
        self.results['wvp'] = wvp_spec if spectra_flux is not None else wvp

        # Calculate RV if planet present
        if self.simulate_planet:
            rvkepler = spectra.keplerian_orbit(
                t, [self.planet_period, self.planet_semi_amplitude,
                    self.planet_esinw, self.planet_ecosw, self.planet_transit_t0]
            )
        else:
            rvkepler = 0.0

        return

    def get_theta_phi(self):
        Ngrids, Ngrid_in_ring, centres, cos_centers, rs, alphas, xs, ys, zs, area, proj_area = nbspectra.generate_grid_coordinates_nb(
            self.n_grid_rings)
        vec_grid = np.array([xs, ys, zs]).T  # coordinates in cartesian
        theta, phi = np.arccos(zs * np.cos(-self.inclination) - xs * np.sin(-self.inclination)), np.arctan2(ys,
                                                                                                            xs * np.cos(
                                                                                                                -self.inclination) + zs * np.sin(
                                                                                                                -self.inclination))  # coordinates in the star reference
        return Ngrid_in_ring, cos_centers, proj_area, phi, theta, vec_grid


    def load_data(self, filename=None, t=None, y=None, yerr=None, instrument=None, observable=None, wvmin=None,
                  wvmax=None, filter_name=None, offset=None, fix_offset=False, jitter=0.0, fix_jitter=False):

        if observable not in ['lc', 'rv', 'bis', 'fwhm', 'contrast', 'crx']:
            sys.exit('Observable not valid. Use one of the following: lc, rv, bis, fwhm, contrast or crx')

        if wvmin == None and wvmax == None:
            print('Wavelength range of the instrument not specified. Using the values in the file starsim.conf, ',
                  self.wavelength_lower_limit, 'and ', self.wavelength_upper_limit)

        if observable == 'lc' and filter_name == None:
            print('Filter file name not specified. Using the values in ', self.filter_name,
                  '. Filters can be retrieved from http://svo2.cab.inta-csic.es/svo/theory/fps3/')
            filter_name = self.filter_name

        self.data[instrument][observable] = {}
        self.data[instrument]['wvmin'] = wvmin
        self.data[instrument]['wvmax'] = wvmax
        self.data[instrument]['filter'] = filter_name
        self.data[instrument][observable]['offset'] = offset
        self.data[instrument][observable]['jitter'] = jitter
        self.data[instrument][observable]['fix_offset'] = fix_offset
        self.data[instrument][observable]['fix_jitter'] = fix_jitter

        if filename != None:
            filename = self.path / filename
            self.data[instrument][observable]['t'], self.data[instrument][observable]['y'], \
            self.data[instrument][observable]['yerr'] = np.loadtxt(filename, unpack=True)
        else:
            self.data[instrument][observable]['t'], self.data[instrument][observable]['y'], \
            self.data[instrument][observable]['yerr'] = t, y, yerr
            if t is None:
                sys.exit('Please provide a valid filename with the input data')

        if observable in ['lc', 'fwhm', 'bis']:
            if offset == 0.0:
                sys.exit("Error in the input offset of the observable:", observable,
                         ". It is a multiplicative offset, can't be 0")
            if offset is None:
                offset = 1.0
            self.data[instrument][observable]['yerr'] = np.sqrt(
                self.data[instrument][observable]['yerr'] ** 2 + jitter ** 2) / offset
            self.data[instrument][observable]['y'] = self.data[instrument][observable]['y'] / offset
            self.data[instrument][observable]['offset_type'] = 'multiplicative'
        else:
            if offset is None:
                offset = 0.0
            self.data[instrument][observable]['y'] = self.data[instrument][observable]['y'] - offset
            self.data[instrument][observable]['yerr'] = np.sqrt(
                self.data[instrument][observable]['yerr'] ** 2 + jitter ** 2)
            self.data[instrument][observable]['offset_type'] = 'linear'


