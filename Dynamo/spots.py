import warnings

import numpy as np
import matplotlib.pylab as plt
import astropy.units as u
from astropy.table import Table
from astropy.io import fits


D2S = 1 * u.day.to(u.s)

PROT_SUN = 24.5


def _symtruncnorm(v):
    """
    Symmetric truncated normal random variable. Returns a value drawn
    from a normal distribution truncated between +/- v.

    Note that scipy.stats.truncnorm exists for this, but as of
    my current version of scipy (1.9.3), this implementation is almost
    100 times faster on a single draw.
    """
    while True:
        x = np.random.normal()
        if np.abs(x) < v:
            return x


def tilt(lat):
    """
    Active region tilt following Joy's Law, as implemented in the original ipython notebook.

    The behavior is that 86% of the time, the tilt angle is proportional to the latitude as:

        ang = lat/2 + 2 (in degrees)

    with non-gaussian scatter of roughly 15 degrees. The other 14% of the time, the tilt
    angle is random but near zero, drawn from a truncated normal distribution
    between -0.5 and 0.5 degrees.

    NOTE: In the original implementation, the `else` block was converted to radians, but the
    output of this function also gets converted to radians... so I've removed the first
    conversion from this function.

    For tilt angles, see
        Wang and Sheeley, Sol. Phys. 124, 81 (1989)
        Wang and Sheeley, ApJ. 375, 761 (1991)
        Howard, Sol. Phys. 137, 205 (1992)
    """
    z = np.random.uniform()
    if z > 0.14:
        # 86% of the time, the angle is proportional to the latitude with some scatter.
        x = _symtruncnorm(1.6)
        y = _symtruncnorm(1.8)
        return (0.5 * lat + 2.0) + 27 * x * y
    else:
        # The other 14%, the angle is near zero, randomly between -0.5 and 0.5 deg.
        return _symtruncnorm(0.5)

def random_latitudes(minlat, maxlat):
    latavg = (maxlat + minlat) / 2.
    latrms = (maxlat - minlat)
    return latavg, latrms

def exponential_latitudes(minlat, maxlat, phase):
    # Based on Hathaway 2015, LRSP 12: 4
    latavg = maxlat * (minlat/maxlat)**phase
    # See Hathaway 2010, p. 37 for a discussion of the width
    latrms = maxlat/5 - phase*(maxlat-minlat)/7

    return latavg, latrms


class SpotsGenerator(object):
    """Create a blank surface to emerge active regions and evolve star spots.

    The `Surface` consists of a grid of `nlat` latitude bins by `nlon`
    longitude bins, over which emergence probabilities are to be computed.

    Active region areas are drawn from a log-uniform distribution consisting
    of `nbins` values, starting at area `max_area` and spacing `delta_lnA`.

    Active regions can be "uncorrelated" or "correlated" to other regions.
    "Correlated" regions can only emerge near regions with the largest area
    (`max_area`) between `tau1` and `tau2` days after the preexisting region's
    emergence.

    Attributes:
        nbins (int): the number of discrete active region areas.
        delta_lnA (float): logarithmic spacing of area values.
        max_area (float): maximum active region area in square degrees.
        tau1 (int): first allowable day of "correlated" emergence, after
            preexisting region's emergence.
        tau2 (int): last allowable day of "correlated" emergence, after
            preexisting region's emergence.
        nlon (int): number of longitude bins in the Surface grid.
        nlat (int): number of latitude bins in the Surface grid.

        duration (int): number of days to emerge regions.
        activity_level (float): Number of magnetic bipoles, normalized such
            that for the Sun, activity_level = 1.
        butterfly (bool): Have spots decrease from maxlat to minlat (True) or
            be randomly located in latitude (False).
        cycle_period (float): Interval (years) between cycle starts (Sun is 11).
        cycle_overlap (float): Overlap of cycles in years.
        max_lat (float): Maximum latitude of spot emergence in degrees.
        min_lat (float): Minimum latitutde of spot emergence in degrees.
        prob_corr (float): The probability of correlated active region
            emergence (relative to uncorrelated emergence).

        regions (astropy Table): list of active regions with timestamp,
            asterographic coordinates of positive and negative bipoles,
            magnetic field strength, and bipole tilt relative to equator.

        inclination (float): Inclination angle of the star in radians, where
            inclination is the angle between the pole and the line of sight.
        period (float): Equatorial rotation period of the star in days.
        omega (float): Equatorial angular velocity in rad/s, equal to 2*pi/period.
        shear (float): Differential rotation rate of the star in units of
            equatorial rotation velocity. I.e., `shear` is alpha = delta_omega / omega.
        diffrot_func (function): Differential rotation function.
            Default is sin^2 (latitude).
        spot_func (function): Spot evolution function.
            Default is double-sided gaussian with time.
        tau_emerge (float): Spot emergence timescale in days.
        tau_decay (float): Spot decay timescale in days.
        nspots (int): total number of star spots.
        tmax (numpy array): time of maximum area for each spot.
        lat (numpy array): latitude of each spot.
        lon (numpy array): longitude of each spot.
        amax (numpy array): maximum area of each spot in millionths of
            solar hemisphere.

        lightcurve (LightCurve): time and flux modulation for all spots.

        wavelet_power (numpy array): the wavelet transform of the lightcurve.

    """

    def __init__(
            self,
            nbins=5,
            delta_lnA=0.5,
            max_area=100,
            tau1=5,
            tau2=15,
            nlon=36,
            nlat=16,
    ):
        """
        Note:
            You usually don't need to change the defaults.

        Args:
            nbins (int): the number of discrete active region areas.
            delta_lnA (float): logarithmic spacing of area values.
            max_area (float): maximum active region area in square degrees.
            tau1 (int): first allowable day of "correlated" emergence, after
                preexisting region's emergence.
            tau2 (int): last allowable day of "correlated" emergence, after
                preexisting region's emergence.
            nlon (int): number of longitude bins in the Surface grid.
            nlat (int): number of latitude bins in the Surface grid.

        """
        # self.areas = max_area / np.exp(delta_lnA * np.arange(nbins))
        self.nbins = nbins  # number of area bins
        self.delta_lnA = delta_lnA  # delta ln(A)
        self.max_area = max_area  # orig. area of largest bipoles (deg^2)
        self.tau1 = tau1
        self.tau2 = tau2
        self.nlon = nlon
        self.nlat = nlat

        self.regions = None
        self.nspots = None
        self.lightcurve = None
        self.wavelet_power = None

    def __repr__(self):
        """Representation method for Surface.
        """
        repr = f"butterpy Surface from {type(self)} with:"

        repr += f"\n    {self.nlat} latitude bins by {self.nlon} longitude bins"

        if self.regions is not None:
            repr += f"\n    N regions = {len(self.regions)}"

        if self.lightcurve is not None:
            repr += f"\n    lightcurve length = {len(self.time)}, duration = {self.time.max() - self.time.min()}"

        if self.wavelet_power is not None:
            repr += f"\n    wavelet_power shape = {self.wavelet_power.shape}"

        return repr

    def assert_regions(self):
        """ If `regions` hasn't been run, raise an error
        """
        assert self.regions is not None, "Set `regions` first with `Surface.emerge_regions`."

    def emerge_regions(
            self,
            ndays=1000,
            nspots=None,  # New parameter to directly control number of spots
            activity_level=1,
            butterfly=True,
            cycle_period=11,
            cycle_overlap=2,
            max_lat=28,
            min_lat=7,
            prob_corr=0.2,
            seed=None
    ):
        """
        Simulates the emergence of starspots with uniform temporal distribution.

        Args:
            ndays (int): Number of days to emerge spots
            nspots (int, optional): Total number of spots to emerge. If None,
                                   calculated from activity_level
            activity_level (float): Used to calculate nspots if not provided
            ... (other parameters same as before)
        """
        # Set attributes
        self.duration = ndays
        self.activity_level = activity_level
        self.butterfly = butterfly
        self.cycle_period = cycle_period
        self.cycle_overlap = cycle_overlap
        self.max_lat = max_lat
        self.min_lat = min_lat
        self.prob_corr = prob_corr
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        # Calculate total number of spots if not provided
        if nspots is None:
            base_rate = 0.1 * np.sqrt(activity_level)  # spots per day. No theoretical intuition.
            nspots = min(max(int(base_rate * ndays), 20), 500)  # minimum 20 spots, maximum 500. arbitrary

        self.nspots = nspots

        # Pre-allocate emergence times uniformly
        emergence_times = np.sort(np.random.choice(ndays, size=nspots, replace=True))

        # Initialize regions table
        self.regions = Table(names=('nday', 'thpos', 'phpos', 'thneg', 'phneg', 'width', 'bmax', 'ang'),
                             dtype=(int, float, float, float, float, float, float, float))

        # Setup parameters (same as original)
        ncycle = 365 * cycle_period
        nclen = 365 * (cycle_period + cycle_overlap)
        amplitude = 10 * activity_level

        if butterfly:
            l1 = max(min_lat - 7, 0)
            l2 = min(max_lat + 7, 90)
        else:
            l1, l2 = min_lat, max_lat
        dlat = (l2 - l1) / self.nlat
        dlon = 360 / self.nlon

        # Area and size calculations
        fact = np.exp(self.delta_lnA * np.arange(self.nbins))
        bsiz = np.sqrt(self.max_area / fact)

        # Process each spot
        for spot_idx, nday in enumerate(emergence_times):
            nday = int(nday)

            # Determine cycle parameters
            ncur = nday // ncycle
            cycle_found = False

            for icycle in [0, 1]:
                nc = ncur - icycle
                nstart = ncycle * nc
                phase = (nday - nstart) / nclen

                if not (0 <= phase <= 1):
                    continue

                cycle_found = True
                break

            if not cycle_found:
                # Default to first cycle if no valid cycle found
                nc = 0
                phase = 0.5

            # Determine latitude distribution
            if butterfly:
                latavg, latrms = exponential_latitudes(min_lat, max_lat, phase)
            else:
                latavg, latrms = random_latitudes(min_lat, max_lat)

            # Sample latitude based on distribution
            jlat = np.arange(self.nlat, dtype=int)
            lat_centers = l1 + dlat * (0.5 + jlat)
            lat_probs = np.exp(-((lat_centers - latavg) / latrms) ** 2)
            lat_probs = lat_probs / lat_probs.sum()

            # Choose latitude bin
            j = np.random.choice(jlat, p=lat_probs)
            lat = l1 + dlat * (np.random.uniform() + j)

            # Choose hemisphere (North or South)
            k = np.random.choice([0, 1])

            # Choose longitude uniformly
            lon = np.random.uniform(0, 360)

            # Choose bipole size based on power law distribution
            # Sample from power law: P(A) ~ A^(-2)
            size_probs = fact / fact.sum()
            nb = np.random.choice(self.nbins, p=size_probs)
            bsize = bsiz[nb]

            # Add the region
            self._add_region_cycle(nday, nc, lat, lon, k, bsize)

        return self.regions

    def _add_region_cycle(self, nday, nc, lat, lon, k, bsize):
        """
        Add one active region of a particular size at a particular location,
        caring about the cycle (for tilt angles).

        Joy's law tilt angle is computed here as well.
        For tilt angles, see
            Wang and Sheeley, Sol. Phys. 124, 81 (1989)
            Wang and Sheeley, ApJ. 375, 761 (1991)
            Howard, Sol. Phys. 137, 205 (1992)

        Args:
            nday (int): day index
            nc (int): cycle index
            lat (float): latitude
            lon (float): longitude
            k (int): hemisphere index (0 for North, 1 for South)
            bsize (float): the size of the bipole

        Adds a row with the following values to `self.regions`:

            nday (int): day index
            thpos (float): theta of positive bipole
            phpos (float): longitude of positive bipole
            thneg (float): theta of negative bipole
            phneg (float): longitude of negative bipole
            width (float): bipole width threshold, always 4...?
            bmax (float): magnetic field strength of bipole
            ang (float): Joy's law bipole angle (from equator)

        Returns None.
        """
        self.assert_regions()

        ic = 1. - 2. * (nc % 2)  # +1 for even, -1 for odd cycle
        width = 4.0  # this is not actually used... remove?
        bmax = 2.5 * bsize ** 2  # original was bmax = 250*(0.4*bsize / width)**2, this is equivalent
        ang = tilt(lat)

        # Convert angles to radians
        ang *= np.pi / 180
        lat *= np.pi / 180
        phcen = lon * np.pi / 180.
        bsize *= np.pi / 180
        width *= np.pi / 180

        # Compute bipole positions
        dph = ic * 0.5 * bsize * np.cos(ang) / np.cos(lat)
        dth = ic * 0.5 * bsize * np.sin(ang)
        thcen = 0.5 * np.pi - lat + 2 * k * lat  # k determines hemisphere
        phpos = phcen + dph
        phneg = phcen - dph
        thpos = thcen + dth
        thneg = thcen - dth

        self.regions.add_row([nday, thpos, phpos, thneg, phneg, width, bmax, ang])

    def add_region(self, nday, lat, lon, bmax):
        """
        Add one active region of a particular size at a particular location,
        ignoring Joy's law tilt and cycle.

        This is meant to be a user-facing function.

        Args:
            nday (int): day index
            lat (float): latitude
            lon (float): longitude
            bmax (float): magnetic field strength of bipole

        Adds a row with the following values to `self.regions`:

            nday (int): day index
            thpos (float): theta of positive bipole
            phpos (float): longitude of positive bipole
            thneg (float): theta of negative bipole
            phneg (float): longitude of negative bipole
            bmax (float): magnetic field strength of bipole

        Returns None.
        """
        if self.regions is None:
            self.regions = Table(names=('nday', 'thpos', 'phpos', 'thneg', 'phneg', 'bmax'),
                                 dtype=(int, float, float, float, float, float))

        # Convert angles to radians
        lat *= np.pi / 180
        phcen = lon * np.pi / 180.
        bsize = np.sqrt(bmax / 2.5) * np.pi / 180

        # Compute bipole positions
        dph = 0
        dth = 0.5 * bsize
        thcen = 0.5 * np.pi - lat  # k determines hemisphere
        phpos = phcen + dph
        phneg = phcen - dph
        thpos = thcen + dth
        thneg = thcen - dth

        self.regions.add_row([nday, thpos, phpos, thneg, phneg, bmax])

    def plot_emergence_distribution(self, bins=50):
        """
        Plot the temporal distribution of spot emergence to verify uniformity.
        """
        if self.regions is None:
            print("No regions generated yet. Run emerge_regions first.")
            return

        plt.figure(figsize=(12, 6))

        # Plot histogram of emergence times
        plt.subplot(1, 2, 1)
        plt.hist(self.regions['nday'], bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Day')
        plt.ylabel('Number of Spots')
        plt.title('Temporal Distribution of Spot Emergence')
        plt.grid(True, alpha=0.3)

        # Plot cumulative distribution
        plt.subplot(1, 2, 2)
        sorted_days = np.sort(self.regions['nday'])
        cumulative = np.arange(1, len(sorted_days) + 1)
        plt.plot(sorted_days, cumulative, 'b-', linewidth=2)

        # Plot ideal uniform distribution for comparison
        if len(sorted_days) > 0:
            ideal_days = np.linspace(sorted_days[0], sorted_days[-1], len(sorted_days))
            ideal_cumulative = np.arange(1, len(sorted_days) + 1)
            plt.plot(ideal_days, ideal_cumulative, 'r--', linewidth=2, label='Ideal Uniform')

        plt.xlabel('Day')
        plt.ylabel('Cumulative Number of Spots')
        plt.title('Cumulative Spot Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        if len(self.regions) > 0:
            print(f"Total spots: {len(self.regions)}")
            print(f"Duration: {self.duration} days")
            print(f"Average rate: {len(self.regions) / self.duration:.3f} spots/day")
            print(f"Time range: {self.regions['nday'].min()} to {self.regions['nday'].max()}")