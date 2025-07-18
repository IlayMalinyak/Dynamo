import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import optimize
import numpy as np
from scipy import interpolate, signal
from pathlib import Path
from scipy import interpolate
import sys
import math as m
from . import nbspectra

########################################################################################
########################################################################################
#                                PHOTOMETRY FUNCTIONS                                  #
########################################################################################
########################################################################################


def interpolate_Phoenix_mu_lc(star, temp, grav, wv_array=None):
    """Cut and interpolate phoenix models at the desired wavelengths, temperatures, logg and metallicity(not yet). For spectroscopy.

    Inputs
    temp: temperature of the model
    grav: logg of the model
    wv_array: optional specific wavelengths to interpolate at

    Returns
    tuple: (mu angles, wavelength array, interpolated flux at each angle)
    """
    import warnings
    warnings.filterwarnings("ignore")
    if isinstance(star.models_root, str):
        star.models_root = Path(star.models_root)
    path = star.models_root / 'models' / 'Phoenix_mu'  # path relative to working directory

    # Cache this operation for better performance if called multiple times
    if not hasattr(star, '_phoenix_model_cache'):
        files = [x.name for x in path.glob('lte*fits') if x.is_file()]
        list_temp = np.unique([float(t[3:8]) for t in files])
        list_grav = np.unique([float(t[9:13]) for t in files])
        star._phoenix_model_cache = {
            'files': files,
            'temperatures': list_temp,
            'gravities': list_grav
        }
    else:
        files = star._phoenix_model_cache['files']
        list_temp = star._phoenix_model_cache['temperatures']
        list_grav = star._phoenix_model_cache['gravities']

    # Check if the parameters are inside the grid of models
    if grav < np.min(list_grav) or grav > np.max(list_grav):
        print(f'Warning: The desired logg ({grav}) is outside the grid of models, extrapolation is not supported.')
        # Use closest available value instead of exiting
        grav = np.min(list_grav) if grav < np.min(list_grav) else np.max(list_grav)
        print(f'Using closest available logg: {grav}')

    if temp < np.min(list_temp) or temp > np.max(list_temp):
        print(f'Warning: The desired T ({temp}) is outside the grid of models, extrapolation is not supported.')
        # Use closest available value instead of exiting
        temp = np.min(list_temp) if temp < np.min(list_temp) else np.max(list_temp)
        print(f'Using closest available T: {temp}')

    lowT = list_temp[list_temp <= temp].max() if any(list_temp <= temp) else list_temp.min()
    uppT = list_temp[list_temp >= temp].min() if any(list_temp >= temp) else list_temp.max()
    lowg = list_grav[list_grav <= grav].max() if any(list_grav <= grav) else list_grav.min()
    uppg = list_grav[list_grav >= grav].min() if any(list_grav >= grav) else list_grav.max()

    # Generate file names
    model_files = {
        'lowTlowg': f'lte{int(lowT):05d}-{lowg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits',
        'lowTuppg': f'lte{int(lowT):05d}-{uppg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits',
        'uppTlowg': f'lte{int(uppT):05d}-{lowg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits',
        'uppTuppg': f'lte{int(uppT):05d}-{uppg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
    }

    # Check which files exist and determine interpolation strategy
    available_files = {k: (v in files) for k, v in model_files.items()}

    if not any(available_files.values()):
        sys.exit(
            'Error: None of the required files for interpolation exist. Please download Phoenix intensity models from https://phoenix.astro.physik.uni-goettingen.de/?page_id=73')

    # Generate wavelength array more efficiently
    # Only generate the needed range if possible
    if wv_array is None:
        wmin = max(500, star.wavelength_lower_limit)
        wmax = min(26000, star.wavelength_upper_limit)
        wavelength = np.arange(wmin, wmax)
        idx_wv = np.ones(len(wavelength), dtype=bool)
    else:
        wavelength = np.arange(500, 26000)
        differences = np.abs(wavelength[:, np.newaxis] - wv_array)
        idx_wv = np.argmin(differences, axis=0)

    # Initialize flux containers
    flux_models = {}

    # Helper function to load a model if available
    def load_model(model_key):
        if available_files[model_key]:
            with fits.open(path / model_files[model_key]) as hdul:
                if model_key == 'lowTlowg':  # Only need to load mu once
                    amu = hdul[1].data
                    amu = np.append(amu[::-1], 0.0)
                return hdul[0].data[:, idx_wv], True
        return None, False

    # Load available models
    amu = None
    for key in model_files:
        flux, success = load_model(key)
        if key == 'lowTlowg' and success:
            with fits.open(path / model_files[key]) as hdul:
                amu = hdul[1].data
                amu = np.append(amu[::-1], 0.0)
        flux_models[key] = flux if success else None

    # If we couldn't load the amu from the first file, try others
    if amu is None:
        for key in model_files:
            if available_files[key]:
                with fits.open(path / model_files[key]) as hdul:
                    amu = hdul[1].data
                    amu = np.append(amu[::-1], 0.0)
                break

    if amu is None:
        sys.exit('Error: Could not load angle information from any model file')

    # Determine interpolation strategy based on available files
    if uppT == lowT or not (flux_models['lowTlowg'] is not None and flux_models['uppTlowg'] is not None):
        print("Can't interpolate in temperature for low gravity")
        # Can't interpolate in temperature for low gravity
        flux_lowg = flux_models['lowTlowg'] if flux_models['lowTlowg'] is not None else flux_models['uppTlowg']
    else:
        # Normal temperature interpolation
        flux_lowg = flux_models['lowTlowg'] + ((temp - lowT) / (uppT - lowT)) * (
                    flux_models['uppTlowg'] - flux_models['lowTlowg'])

    if uppT == lowT or not (flux_models['lowTuppg'] is not None and flux_models['uppTuppg'] is not None):
        print("Can't interpolate in temperature for upper gravity")
        # Can't interpolate in temperature for upper gravity
        flux_uppg = flux_models['lowTuppg'] if flux_models['lowTuppg'] is not None else flux_models['uppTuppg']
    else:
        # Normal temperature interpolation
        flux_uppg = flux_models['lowTuppg'] + ((temp - lowT) / (uppT - lowT)) * (
                    flux_models['uppTuppg'] - flux_models['lowTuppg'])

    # Gravity interpolation
    if uppg == lowg or flux_uppg is None or flux_lowg is None:
        print("Can't interpolate in gravity")
        # Can't interpolate in gravity, use whichever is available
        flux = flux_lowg if flux_lowg is not None else flux_uppg
    else:
        # Normal gravity interpolation
        flux = flux_lowg + ((grav - lowg) / (uppg - lowg)) * (flux_uppg - flux_lowg)

    # Handle the case where we have no usable flux
    if flux is None:
        print(f"Warning: Could not perform interpolation with available files. Try downloading more models.")
        # Try to use any available model instead of failing
        for f in flux_models.values():
            if f is not None:
                flux = f
                break

        if flux is None:
            sys.exit('Error: No usable model files found for interpolation')

    # Create the final flux array
    angle0 = flux[0] * 0.0  # LD of 90 deg
    flux_joint = np.vstack([flux[::-1], angle0])  # add LD coeffs at 0 and 1 proj angles

    return amu, wavelength[idx_wv], flux_joint

def interpolate_filter(self, filter_name):

    if isinstance(self.models_root, str):
        self.models_root = Path(self.models_root)
    path = self.models_root / 'models' / 'filters' / filter_name

    try:
        wv, filt = np.loadtxt(path,unpack=True)
    except: #if the filter do not exist, create a tophat filter from the wv range
        wv=np.array([self.wavelength_lower_limit,self.wavelength_upper_limit])
        filt=np.array([1,1])
        print('Filter ',self.filter_name,' do not exist inside the filters folder. Using wavelength range in starsim.conf. Filters are available at http://svo2.cab.inta-csic.es/svo/theory/fps3/')

    f = interpolate.interp1d(wv,filt,bounds_error=False,fill_value=0)

    return f

def limb_darkening_law(self,amu):

    if self.limb_darkening_law == 'linear':
        mu=1-self.limb_darkening_q1*(1-amu)

    elif self.limb_darkening_law == 'quadratic':
        a=2*np.sqrt(self.limb_darkening_q1)*self.limb_darkening_q2
        b=np.sqrt(self.limb_darkening_q1)*(1-2*self.limb_darkening_q2)
        mu=1-a*(1-amu)-b*(1-amu)**2

    elif self.limb_darkening_law == 'sqrt':
        a=np.sqrt(self.limb_darkening_q1)*(1-2*self.limb_darkening_q2) 
        b=2*np.sqrt(self.limb_darkening_q1)*self.limb_darkening_q2
        mu=1-a*(1-amu)-b*(1-np.sqrt(amu))

    elif self.limb_darkening_law == 'log':
        a=self.limb_darkening_q2*self.limb_darkening_q1**2+1
        b=self.limb_darkening_q1**2-1
        mu=1-a*(1-amu)-b*amu*(1-np.log(amu))

    else:
        sys.exit('Error in limb darkening law, please select one of the following: phoenix, linear, quadratic, sqrt, logarithmic')

    return mu


def apply_rotational_broadening(wavelength, flux, vsini, epsilon=0.6):
    """
    Apply rotational broadening to a spectrum.

    Parameters:
    wavelength : array-like
        Wavelength array in Angstroms
    flux : array-like
        Flux array
    vsini : float
        Projected rotational velocity in km/s
    epsilon : float, optional
        Limb darkening coefficient (default 0.6)

    Returns:
    array-like
        Broadened flux array
    """
    import numpy as np
    from scipy import interpolate, signal

    c = 299792.458  # Speed of light in km/s

    if vsini <= 0:
        return flux  # No broadening required

    # Average wavelength spacing
    delta_lambda = np.mean(np.diff(wavelength))

    # Calculate the width of the broadening kernel in wavelength units
    # The factor 2 comes from the full width of the line profile
    sigma = wavelength.mean() * vsini / c

    # Make sure the kernel width is sufficient
    if sigma < delta_lambda:
        return flux  # Broadening smaller than resolution

    # Calculate width of kernel in array elements
    kernel_width = int(np.ceil(5 * sigma / delta_lambda))
    if kernel_width % 2 == 0:
        kernel_width += 1  # Make sure it's odd

    # Create the kernel array
    kernel_x = np.linspace(-kernel_width // 2, kernel_width // 2, kernel_width) * delta_lambda
    kernel = np.zeros_like(kernel_x)

    # Populate the kernel using the rotational broadening profile
    x = kernel_x / sigma
    mask = np.abs(x) < 1.0
    kernel[mask] = (2 / np.pi) * (1.0 - epsilon * (1.0 - np.sqrt(1.0 - x[mask] ** 2))) * np.sqrt(1.0 - x[mask] ** 2)

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    # Apply the kernel via convolution
    broadened_flux = signal.convolve(flux, kernel, mode='same')

    return broadened_flux


def compute_immaculate_lc_with_vsini(self, Ngrid_in_ring, acd, amu, pare, flnp, f_filt, wv, vsini=0.0):
    """
    Compute immaculate light curve with rotational broadening.

    Parameters are the same as compute_immaculate_lc, with the addition of:
    vsini : float, optional
        Projected rotational velocity in km/s
    """
    N = self.n_grid_rings  # Number of concentric rings
    flxph = 0.0  # initialize flux of photosphere
    sflp = np.zeros(N)  # brightness of ring
    flp = np.zeros([N, len(wv)])  # spectra of each ring convolved by filter

    # Calculate differential rotation for each ring if enabled
    if hasattr(self, 'differential_rotation') and self.differential_rotation:
        vsini_rings = calculate_differential_vsini(self, N, vsini, amu)
    else:
        vsini_rings = [vsini] * N  # Same vsini for all rings


    # Computing flux of immaculate photosphere and of every pixel
    for i in range(0, N):  # Loop for each ring, to compute the flux of the star.
        # Interpolate Phoenix intensity models to correct projected angle:
        if self.use_phoenix_limb_darkening:
            acd_low = np.max(acd[acd < amu[i]])  # angles above and below the proj. angle of the grid
            acd_upp = np.min(acd[acd >= amu[i]])
            idx_low = np.where(acd == acd_low)[0][0]
            idx_upp = np.where(acd == acd_upp)[0][0]
            dlp = flnp[idx_low] + (flnp[idx_upp] - flnp[idx_low]) * (amu[i] - acd_low) / (
                        acd_upp - acd_low)  # limb darkening
        else:  # or use a specified limb darkening law
            dlp = flnp[0] * limb_darkening_law(self, amu[i])

        # Apply rotational broadening based on the vsini for this ring
        if vsini_rings[i] > 0:
            dlp = apply_rotational_broadening(wv, dlp, vsini_rings[i])

        # Apply filter and calculate flux contribution
        flp[i, :] = dlp * pare[i] / (4 * np.pi) * f_filt(wv)  # spectra of one grid in ring N multiplied by the filter
        sflp[i] = np.sum(flp[i, :])  # brightness of one grid in ring N
        flxph = flxph + sflp[i] * Ngrid_in_ring[i]  # total BRIGHTNESS of the immaculate photosphere

    return sflp, flxph


def calculate_differential_vsini(self, n_rings, vsini_equator, mu_values):
    """
    Calculate vsini for each ring based on a differential rotation law.

    Parameters:
    self : object
        The parent object containing rotation parameters
    n_rings : int
        Number of rings
    vsini_equator : float
        Projected rotational velocity at equator in km/s
    mu_values : array-like
        Cosine of the angle between the line of sight and the local normal

    Returns:
    list
        vsini values for each ring
    """
    # Example differential rotation law: Ω(θ) = Ω_equator * (1 - alpha * sin²(θ))
    # where θ is the latitude and alpha is the differential rotation parameter

    # Default to no differential rotation if parameter is not set
    alpha = getattr(self, 'differential_rotation', 0.0)

    vsini_rings = []
    for i in range(n_rings):
        # Convert mu to latitude (approximate)
        # mu = cos(i) * cos(latitude)
        # For a star seen edge-on (i=90°), mu directly gives sin(latitude)
        sin_lat = mu_values[i]
        # Apply differential rotation law
        scale_factor = 1.0 - alpha * sin_lat ** 2
        vsini_rings.append(vsini_equator * scale_factor)

    return vsini_rings


########################################################################################
########################################################################################
#                              SPECTROSCOPY FUNCTIONS                                  #
########################################################################################
########################################################################################


def interpolate_Phoenix(self,temp,grav,plot=False):
    """Cut and interpolate phoenix models at the desired wavelengths, temperatures, logg and metalicity(not yet). For spectroscopy.
    Inputs
    temp: temperature of the model; 
    grav: logg of the model
    Returns
    creates a temporal file with the interpolated spectra at the temp and grav desired, for each surface element.
    """
    #Demanar tambe la resolucio i ficarho aqui.

    import warnings
    warnings.filterwarnings("ignore")

    path = self.path / 'models' / 'Phoenix' #path relatve to working directory 
    files = [x.name for x in path.glob('lte*fits') if x.is_file()]
    list_temp=np.unique([float(t[3:8]) for t in files])
    list_grav=np.unique([float(t[9:13]) for t in files])

    #check if the parameters are inside the grid of models
    if grav<np.min(list_grav) or grav>np.max(list_grav):
        sys.exit('Error in the interpolation of Phoenix models. The desired logg ({}) is outside the grid of models, extrapolation is not supported. Please download the \
        Phoenix models covering the desired logg from http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'.format(grav))

    if temp<np.min(list_temp) or temp>np.max(list_temp):
        sys.exit('Error in the interpolation of Phoenix models. The desired T ({}) is outside the grid of models, extrapolation is not supported. Please download the \
        Phoenix models covering the desired T from http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/'.format(temp))
        


    lowT=list_temp[list_temp<=temp].max() #find the model with the temperature immediately below the desired temperature
    uppT=list_temp[list_temp>=temp].min() #find the model with the temperature immediately above the desired temperature
    lowg=list_grav[list_grav<=grav].max() #find the model with the logg immediately below the desired logg
    uppg=list_grav[list_grav>=grav].min() #find the model with the logg immediately above the desired logg

    #load the Phoenix wavelengths.
    if not (path / 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits').exists():
        sys.exit('Error in reading the file WAVE_PHOENIX-ACES-AGSS-COND-2011.fits. Please download it from http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/')
    with fits.open(path / 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits') as hdul:
        wavelength=hdul[0].data
    #cut the wavelength at the ranges set by the user. Adding an overhead of 0.1 nm to allow for high Doppler shifts without losing info
    overhead=1.0 #Angstrom
    idx_wv=np.array(wavelength>self.wavelength_lower_limit-overhead) & np.array(wavelength<self.wavelength_upper_limit+overhead)
    #load the flux of the four phoenix model
    name_lowTlowg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(int(lowT),lowg)
    name_lowTuppg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(int(lowT),uppg)
    name_uppTlowg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(int(uppT),lowg)
    name_uppTuppg='lte{:05d}-{:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(int(uppT),uppg)

    #Check if the files exist in the folder
    if name_lowTlowg not in files:
        sys.exit('The file '+name_lowTlowg+' required for the interpolation does not exist. Please download it from http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ and add it to your path')
    if name_lowTuppg not in files:
        sys.exit('The file '+name_lowTuppg+' required for the interpolation does not exist. Please download it from http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ and add it to your path')
    if name_uppTlowg not in files:
        sys.exit('The file '+name_uppTlowg+' required for the interpolation does not exist. Please download it from http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ and add it to your path')
    if name_uppTuppg not in files:
        sys.exit('The file '+name_uppTuppg+' required for the interpolation does not exist. Please download it from http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/ and add it to your path')

    #read flux files and cut at the desired wavelengths
    with fits.open(path / name_lowTlowg) as hdul:
        flux_lowTlowg=hdul[0].data[idx_wv]
    with fits.open(path / name_lowTuppg) as hdul:
        flux_lowTuppg=hdul[0].data[idx_wv]
    with fits.open(path / name_uppTlowg) as hdul:
        flux_uppTlowg=hdul[0].data[idx_wv]
    with fits.open(path / name_uppTuppg) as hdul:
        flux_uppTuppg=hdul[0].data[idx_wv]

    #interpolate in temperature for the two gravities
    if uppT==lowT: #to avoid nans
        flux_lowg = flux_lowTlowg 
        flux_uppg = flux_lowTuppg
    else:
        flux_lowg = flux_lowTlowg + ( (temp - lowT) / (uppT - lowT) ) * (flux_uppTlowg - flux_lowTlowg)
        flux_uppg = flux_lowTuppg + ( (temp - lowT) / (uppT - lowT) ) * (flux_uppTuppg - flux_lowTuppg)
    #interpolate in log g
    if uppg==lowg: #to avoid dividing by 0
        flux = flux_lowg
    else:
        flux = flux_lowg + ( (grav - lowg) / (uppg - lowg) ) * (flux_uppg - flux_lowg)


    #Normalize by fitting a 6th degree polynomial to the maximum of the bins of the binned spectra
    #nbins depend on the Temperature and wavelength range. 20 bins seems to work for all reasonable parameters. With more bins it starts to pick absorption lines. Less bins degrades the fit. 
    bins=np.linspace(self.wavelength_lower_limit-overhead,self.wavelength_upper_limit+overhead,20)
    wv= wavelength[idx_wv]
    x_bin,y_bin=nbspectra.normalize_spectra_nb(bins,np.asarray(wv,dtype=np.float64),np.asarray(flux,dtype=np.float64))


    # #divide by 6th deg polynomial
    coeff = np.polyfit(x_bin, y_bin, 6)
    flux_norm = flux / np.poly1d(coeff)(wv)
    #plots to check normalization. For debugging purposes.
    if plot:
        plt.plot(wv,flux)
        plt.plot(x_bin,y_bin,'ok')
        plt.plot(wv,np.poly1d(coeff)(wv))
        plt.show()
        plt.close()

    interpolated_spectra = np.array([wv,flux_norm])

    return interpolated_spectra


def keplerian_orbit(x,params):
    period=params[0]
    t_trans=params[4]
    krv=params[1]
    esinw=params[2]
    ecosw=params[3]
    
    if(esinw==0 and ecosw==0):
       ecc=0
       omega=0
    else:
       ecc=np.sqrt(esinw*esinw+ecosw*ecosw)
       omega=np.arctan2(esinw,ecosw)

    t_peri = Ttrans_2_Tperi(t_trans, period, ecc, omega)
    sinf,cosf=true_anomaly(x,period,ecc,t_peri)
    cosftrueomega=cosf*np.cos(omega)-sinf*np.sin(omega)
    y= krv*(ecc*np.cos(omega)+cosftrueomega)

    return y
#   
def true_anomaly(x,period,ecc,tperi):
    sinf=[]
    cosf=[]
    for i in range(len(x)):
        fmean=2.0*np.pi*(x[i]-tperi)/period
        #Solve by Newton's method x(n+1)=x(n)-f(x(n))/f'(x(n))
        fecc=fmean
        diff=1.0
        while(diff>1.0E-6):
            fecc_0=fecc
            fecc=fecc_0-(fecc_0-ecc*np.sin(fecc_0)-fmean)/(1.0-ecc*np.cos(fecc_0))
            diff=np.abs(fecc-fecc_0)
        sinf.append(np.sqrt(1.0-ecc*ecc)*np.sin(fecc)/(1.0-ecc*np.cos(fecc)))
        cosf.append((np.cos(fecc)-ecc)/(1.0-ecc*np.cos(fecc)))
    return np.array(sinf),np.array(cosf)


def Ttrans_2_Tperi(T0, P, e, w):

    f = np.pi/2 - w
    E = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-e)/(1+e)))  # eccentric anomaly
    Tp = T0 - P/(2*np.pi) * (E - e*np.sin(E))      # time of periastron

    return Tp


########################################################################################
########################################################################################
#                              SPOTMAP/GRID FUNCTIONS                                  #
########################################################################################
########################################################################################
def compute_spot_position(self,t):


    pos=np.zeros([len(self.spot_map),4])

    for i in range(len(self.spot_map)):
        tini = self.spot_map[i][0] #time of spot apparence
        dur = self.spot_map[i][1] #duration of the spot
        tfin = tini + dur #final time of spot
        colat = self.spot_map[i][2] #colatitude
        lat = 90 - colat #latitude
        longi = self.spot_map[i][3] #longitude
        Rcoef = self.spot_map[i][4::] #coefficients for the evolution od the radius. Depends on the desired law.

        #update longitude adding diff rotation
        pht = longi + (t-self.reference_time)/self.rotation_period%1*360 + (t-self.reference_time)*self.differential_rotation*(1.698*np.sin(np.deg2rad(lat))**2+2.346*np.sin(np.deg2rad(lat))**4)
        phsr = pht%360 #make the phase between 0 and 360. 

        if self.spots_evo_law == 'constant':
            if t>=tini and t<=tfin: 
                rad=Rcoef[0] 
            else:
                rad=0.0

        elif self.spots_evo_law == 'linear':
            if t>=tini and t<=tfin:
                rad=Rcoef[0]+(t-tini)*(Rcoef[1]-Rcoef[0])/dur
            else:
                rad=0.0
        elif self.spots_evo_law == 'quadratic':
            if t>=tini and t<=tfin:
                rad=-4*Rcoef[0]*(t-tini)*(t-tini-dur)/dur**2
            else:
                rad=0.0

        else:
            sys.exit('Spot evolution law not implemented yet')
        
        if self.facular_area_ratio!=0.0: #to speed up the code when no fac are present
            rad_fac=np.deg2rad(rad)*np.sqrt(1+self.facular_area_ratio) 
        else: rad_fac=0.0

        pos[i]=np.array([np.deg2rad(colat), np.deg2rad(phsr), np.deg2rad(rad), rad_fac])
        #return position and radii of spots at t in radians.

    return pos

def compute_planet_pos(self,t):
    
    if(self.planet_esinw==0 and self.planet_ecosw==0):
       ecc=0
       omega=0
    else:
       ecc=np.sqrt(self.planet_esinw**2+self.planet_ecosw**2)
       omega=np.arctan2(self.planet_esinw,self.planet_ecosw)

    t_peri = Ttrans_2_Tperi(self.planet_transit_t0,self.planet_period, ecc, omega)
    sinf,cosf=true_anomaly([t],self.planet_period,ecc,t_peri)


    cosftrueomega=cosf*np.cos(omega+np.pi/2)-sinf*np.sin(omega+np.pi/2) #cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
    sinftrueomega=cosf*np.sin(omega+np.pi/2)+sinf*np.cos(omega+np.pi/2) #sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

    if cosftrueomega>0.0: return np.array([1+self.planet_radius*2, 0.0, self.planet_radius]) #avoid secondary transits

    cosi = (self.planet_impact_param/self.planet_semi_major_axis)*(1+self.planet_esinw)/(1-ecc**2) #cosine of planet inclination (i=90 is transit)

    rpl=self.planet_semi_major_axis*(1-ecc**2)/(1+ecc*cosf)
    xpl=rpl*(-np.cos(self.planet_spin_orbit_angle)*sinftrueomega-np.sin(self.planet_spin_orbit_angle)*cosftrueomega*cosi)
    ypl=rpl*(np.sin(self.planet_spin_orbit_angle)*sinftrueomega-np.cos(self.planet_spin_orbit_angle)*cosftrueomega*cosi)

    rhopl=np.sqrt(ypl**2+xpl**2)
    thpl=np.arctan2(ypl,xpl)

    pos=np.array([float(rhopl), float(thpl), self.planet_radius]) #rho, theta, and radii (in Rstar) of the planet
    return pos

def plot_spot_map_grid(self,vec_grid,typ,inc,time):
    filename = self.path / 'plots' / 'map_t_{:.4f}.png'.format(time)

    x=np.linspace(-0.999,0.999,1000)
    h=np.sqrt((1-x**2)/(np.tan(inc)**2+1))
    color_dict = { 0:'red', 1:'black', 2:'yellow', 3:'blue'}
    plt.figure(figsize=(4,4))
    plt.title('t={:.3f}'.format(time))
    plt.scatter(vec_grid[:,1],vec_grid[:,2], color=[ color_dict[np.argmax(i)] for i in typ ],s=2 )
    plt.plot(x,h,'k')
    plt.savefig(filename,dpi=100)
    plt.close()




