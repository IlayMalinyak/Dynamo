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
#                                GENERAL FUNCTIONS                                     #
########################################################################################
########################################################################################

def black_body(wv,T):
    #Computes the BB flux with temperature T at wavelengths wv(in nanometers)
    c = 2.99792458e10 #speed of light in cm/s
    k = 1.380658e-16  #boltzmann constant
    h = 6.6260755e-27 #planck
    w=wv*1e-8 #Angstrom to cm
    bb=2*h*c**2*w**(-5)*(np.exp(h*c/k/T/w)-1)**(-1)
    return bb

def vacuum2air(wv): #wv in angstroms
	wv=wv*1e-4 #A to micrometer
	a=0
	b1=5.792105e-2
	b2=1.67917e-3
	c1=238.0185
	c2=57.362

	n=1+a+b1/(c1-(1/wv**2))+b2/(c2-(1/wv**2))

	w=(wv/n)*1e4 #to Angstroms
	return w

def air2vacuum(wv): #wv in angstroms
    wv=wv*1e-4 #A to micrometer
    a=0
    b1=5.792105e-2
    b2=1.67917e-3
    c1=238.0185
    c2=57.362

    n=1+a+b1/(c1-(1/wv**2))+b2/(c2-(1/wv**2))

    w=(wv*n)*1e4 #to Angstroms
    return w

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


def apply_line_broadening_to_all_filters(wavelength, flux, resolution=None, vsini=None, macroturb=None):
    """
    Apply various broadening mechanisms to a spectrum.

    Parameters:
    wavelength : array-like
        Wavelength array in Angstroms
    flux : array-like
        Flux array
    resolution : float, optional
        Spectral resolution (R = λ/Δλ)
    vsini : float, optional
        Projected rotational velocity in km/s
    macroturb : float, optional
        Macroturbulence velocity in km/s

    Returns:
    array-like
        Broadened flux array
    """
    # Make a copy to avoid modifying the original
    broadened_flux = np.copy(flux)

    # Apply rotational broadening
    if vsini is not None and vsini > 0:
        broadened_flux = apply_rotational_broadening(wavelength, broadened_flux, vsini)

    # Apply instrumental broadening based on resolution
    if resolution is not None and resolution > 0:
        # Calculate sigma in wavelength units
        sigma_lambda = wavelength / resolution

        # Apply Gaussian broadening
        for i in range(len(wavelength)):
            sigma = sigma_lambda[i]
            kernel_width = int(np.ceil(5 * sigma / np.mean(np.diff(wavelength))))
            if kernel_width % 2 == 0:
                kernel_width += 1

            if kernel_width > 3:  # Only apply if kernel is big enough
                x = np.linspace(-kernel_width // 2, kernel_width // 2, kernel_width)
                kernel = np.exp(-0.5 * (x / sigma) ** 2)
                kernel = kernel / np.sum(kernel)

                # Apply convolution to a segment around the current wavelength
                half_width = kernel_width // 2
                idx_start = max(0, i - half_width * 2)
                idx_end = min(len(wavelength), i + half_width * 2 + 1)

                if idx_end - idx_start > kernel_width:
                    segment = broadened_flux[idx_start:idx_end]
                    segment_convolved = signal.convolve(segment, kernel, mode='same')
                    broadened_flux[idx_start:idx_end] = segment_convolved

    # Apply macroturbulent broadening (similar to instrumental but with different kernel)
    if macroturb is not None and macroturb > 0:
        c = 299792.458  # Speed of light in km/s
        # Convert macroturbulence velocity to wavelength
        sigma = wavelength.mean() * macroturb / c

        # Calculate width of kernel in array elements
        delta_lambda = np.mean(np.diff(wavelength))
        kernel_width = int(np.ceil(5 * sigma / delta_lambda))
        if kernel_width % 2 == 0:
            kernel_width += 1

        # Create Gaussian kernel for macroturbulence
        x = np.linspace(-kernel_width // 2, kernel_width // 2, kernel_width) * delta_lambda / sigma
        kernel = np.exp(-0.5 * x ** 2)
        kernel = kernel / np.sum(kernel)

        # Apply the kernel via convolution
        broadened_flux = signal.convolve(broadened_flux, kernel, mode='same')

    return broadened_flux

def compute_immaculate_lc(self,Ngrid_in_ring,acd,amu,pare,flnp,f_filt,wv):


    N = self.n_grid_rings #Number of concentric rings
    flxph = 0.0 #initialze flux of photosphere
    sflp=np.zeros(N) #brightness of ring
    flp=np.zeros([N,len(wv)]) #spectra of each ring convolved by filter

    #Computing flux of immaculate photosphere and of every pixel
    for i in range(0,N): #Loop for each ring, to compute the flux of the star.   

        #Interpolate Phoenix intensity models to correct projected ange:
        if self.use_phoenix_limb_darkening:
            acd_low=np.max(acd[acd<amu[i]]) #angles above and below the proj. angle of the grid
            acd_upp=np.min(acd[acd>=amu[i]])
            idx_low=np.where(acd==acd_low)[0][0]
            idx_upp=np.where(acd==acd_upp)[0][0]
            dlp = flnp[idx_low]+(flnp[idx_upp]-flnp[idx_low])*(amu[i]-acd_low)/(acd_upp-acd_low) #limb darkening
        
        else: #or use a specified limb darkening law
            dlp = flnp[0]*limb_darkening_law(self,amu[i])


        flp[i,:]=dlp*pare[i]/(4*np.pi)*f_filt(wv) #spectra of one grid in ring N multiplied by the filter.
        sflp[i]=np.sum(flp[i,:]) #brightness of onegrid in ring N.  
        flxph=flxph+sflp[i]*Ngrid_in_ring[i] #total BRIGHTNESS of the immaculate photosphere
    
    
    return sflp, flxph



def compute_immaculate_facula_lc(self,Ngrid_in_ring,acd,amu,pare,flnp,f_filt,wv):
    '''Compute thespectra of each grid element adding LD.
    '''
    N = self.n_grid_rings #Number of concentric rings
    flxfc = 0.0 #initialze flux of photosphere
    sflf=np.zeros(N) #brightness of ring
    flf=np.zeros([N,len(wv)]) #spectra of each ring convolved by filter

    #Computing flux of immaculate photosphere and of every pixel
    for i in range(0,N): #Loop for each ring, to compute the flux of the star.   

        #Interpolate Phoenix intensity models to correct projected ange:
        if self.use_phoenix_limb_darkening:
            acd_low=np.max(acd[acd<amu[i]]) #angles above and below the proj. angle of the grid
            acd_upp=np.min(acd[acd>=amu[i]])
            idx_low=np.where(acd==acd_low)[0][0]
            idx_upp=np.where(acd==acd_upp)[0][0]
            dlp = flnp[idx_low]+(flnp[idx_upp]-flnp[idx_low])*(amu[i]-acd_low)/(acd_upp-acd_low) #limb darkening
        
        else: #or use a specified limb darkening law
            dlp = flnp[0]*limb_darkening_law(self,amu[i])

        flf[i,:]=dlp*pare[i]/(4*np.pi)*f_filt(wv) #spectra of one grid in ring N multiplied by the filter.
        #Limb brightening
        dtfmu=250.9-407.4*amu[i]+190.9*amu[i]**2 #(T_fac-T_ph) multiplied by a factor depending on the 
        sflf[i]=np.sum(flf[i,:])*((self.temperature_photosphere+dtfmu)/(self.temperature_facula))**4 #brightness of onegrid in ring N.  
        flxfc=flxfc+sflf[i]*Ngrid_in_ring[i]  #total BRIGHTNESS of the immaculate photosphere

    return sflf, flxfc

 

def generate_rotating_photosphere_lc(self,Ngrid_in_ring,pare,amu,bph,bsp,bfc,flxph,vec_grid,inversion,plot_map=True):
    '''Loop for all the pixels and assign the flux corresponding to the grid element.
    '''
    simulate_planet=self.simulate_planet
    N = self.n_grid_rings #Number of concentric rings
    
    iteration=0

    #Now loop for each Observed time and for each grid element. Compute if the grid is ph spot or fc and assign the corresponding CCF.
    # print('Diff rotation law is hard coded. Check ref time for inverse problem. Add more Spot evo laws')
    if not inversion:
        sys.stdout.write(" ")
    flux=np.zeros([len(self.obs_times)]) #initialize total flux at each timestamp
    filling_sp=np.zeros(len(self.obs_times))
    filling_ph=np.zeros(len(self.obs_times))
    filling_pl=np.zeros(len(self.obs_times))
    filling_fc=np.zeros(len(self.obs_times))

    for k,t in enumerate(self.obs_times):
        typ=[] #type of grid, ph sp or fc
        
        if simulate_planet:
            planet_pos=compute_planet_pos(self,t)#compute the planet position at current time. In polar coordinates!! 
        else:
            planet_pos = [2.0,0.0,0.0]


        if self.spot_map.size==0:
            spot_pos=np.array([np.array([m.pi/2,-m.pi,0.0,0.0])])
        else:
            spot_pos=compute_spot_position(self,t) #compute the position of all spots at the current time. Returns theta and phi of each spot.      

        vec_spot=np.zeros([len(self.spot_map),3])
        xspot = np.cos(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])+np.sin(self.inclination)*np.cos(spot_pos[:,0])
        yspot = np.sin(spot_pos[:,0])*np.sin(spot_pos[:,1])
        zspot = np.cos(spot_pos[:,0])*np.cos(self.inclination)-np.sin(self.inclination)*np.sin(spot_pos[:,0])*np.cos(spot_pos[:,1])
        vec_spot[:,:]=np.array([xspot,yspot,zspot]).T #spot center in cartesian

        #COMPUTE IF ANY SPOT IS VISIBLE
        vis=np.zeros(len(vec_spot)+1)
        for i in range(len(vec_spot)):
            dist=m.acos(np.dot(vec_spot[i],np.array([1,0,0])))
            
            if (dist-spot_pos[i,2]*np.sqrt(1+self.facular_area_ratio)) <= (np.pi/2):
                vis[i]=1.0
        
        if (planet_pos[0]-planet_pos[2]<1):
            vis[-1]=1.0
 


        #Loop for each ring.
        if (np.sum(vis)==0.0):
            flux[k],typ, filling_ph[k], filling_sp[k], filling_fc[k], filling_pl[k] = flxph, [[1.0,0.0,0.0,0.0]]*np.sum(Ngrid_in_ring), np.dot(Ngrid_in_ring,pare), 0.0, 0.0, 0.0
        else:
            flux[k],typ, filling_ph[k], filling_sp[k], filling_fc[k], filling_pl[k] = nbspectra.loop_generate_rotating_lc_nb(N,Ngrid_in_ring,pare,amu,spot_pos,vec_grid,vec_spot,simulate_planet,planet_pos,bph,bsp,bfc,flxph,vis)


        filling_ph[k]=100*filling_ph[k]/np.dot(Ngrid_in_ring,pare)
        filling_sp[k]=100*filling_sp[k]/np.dot(Ngrid_in_ring,pare)
        filling_fc[k]=100*filling_fc[k]/np.dot(Ngrid_in_ring,pare)
        filling_pl[k]=100*filling_pl[k]/np.dot(Ngrid_in_ring,pare)
        
        if not inversion:
            sys.stdout.write("\rDate {0}. ff_ph={1:.3f}%. ff_sp={2:.3f}%. ff_fc={3:.3f}%. ff_pl={4:.3f}%. [{5}/{6}]%".format(t,filling_ph[k],filling_sp[k],filling_fc[k],filling_pl[k],k+1,len(self.obs_times)))

        if plot_map:
            plot_spot_map_grid(self,vec_grid,typ,self.inclination,t)


    return self.obs_times, flux/flxph, filling_ph, filling_sp, filling_fc, filling_pl









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



def bisector_fit(self,rv,ccf,plot_test=False,kind_interp='linear',integrated_bis=False):
    ''' Fit the bisector of the CCF with a 5th deg polynomial
    '''
    xnew,ynew,xbis,ybis=nbspectra.speed_bisector_nb(rv,ccf,integrated_bis)

    f = interpolate.interp1d(ybis,xbis,kind=kind_interp,fill_value=(xbis[0],xbis[-1]),bounds_error=False) #return a function rv=f(ccf) interpolating the BIS for all values of ccf height.
    
    if plot_test: #for debuggin purposes
        ys=np.linspace(0,1,1000)
        # xs = f(ys)
        # plt.plot(xs,ys)
        plt.plot(xbis,ybis,'.')
        plt.plot(rv,ccf)
        plt.plot(f(ccf),ccf)
        plt.show()

    return f

def cifist_coeff_interpolate(amu):
    '''Interpolate the cifist bisectors as a function of the projected angle
    '''
    amv=np.arange(1,0.0,-0.1) #list of angles defined in cfist
    if amu<=0.1:
        amv_low=0
    else:
        amv_low=np.max(amv[amv<amu]) #angles above and below the proj. angle of the grid
        idx_low=np.where(amv==amv_low)[0][0] #find indexs of below and above angles

    amv_upp=np.min(amv[amv>=amu])
    idx_upp=np.where(amv==amv_upp)[0][0]

    cxm=np.zeros([len(amv),7]) #coeff of the bisectors. NxM, N is number of angles, M=7, the degree of the polynomial
    #PARAMS FROM A CCF COMPUTED WITH HARPS MASK.
    cxm[0,:]=np.array([-3.51974861,11.1702017,-13.22368296,6.67694456,-0.63201573,-0.44695616,-0.36838495]) #1.0
    cxm[1,:]=np.array([-4.05903967,13.21901003,-16.47215949,9.51023171,-2.13104764,-0.05153799,-0.36973749]) #0.9
    cxm[2,:]=np.array([-3.92153131,12.76694663,-15.96958217,9.39599116,-2.34394028,0.12546611,-0.42092905]) #0.8
    cxm[3,:]=np.array([-3.81892968,12.62209118,-16.06973368,9.71487198,-2.61439945,0.25356088,-0.43310756]) #0.7
    cxm[4,:]=np.array([-5.37213406,17.6604689,-22.52477323,13.91461247,-4.13186181,0.60271171,-0.46427559]) #0.6
    cxm[5,:]=np.array([-6.35351933,20.92046705,-26.83933359,16.86220487,-5.28285592,0.90643187,-0.47696283]) #0.5
    cxm[6,:]=np.array([-7.67270144,25.60866105,-33.4381214,21.58855269,-7.1527039,1.35990694,-0.48001707]) #0.4
    cxm[7,:]=np.array([-9.24152009,31.09337903,-41.07410957,27.04196984,-9.32910982,1.89291407,-0.455407]) #0.3
    cxm[8,:]=np.array([-11.62006536,39.30962189,-52.38161244,34.98243089,-12.40650704,2.57940618,-0.37337442]) #0.2
    cxm[9,:]=np.array([-14.14768805,47.9566719,-64.20294114,43.23156971,-15.57423374,3.13318175,-0.14451226]) #0.1

    #PARAMS FROM A CCF COMPUTED WITH PHOENIX TEMPLATE T=5770
    # cxm[0,:]=np.array([1.55948401e+01, -5.59100775e+01,  7.98788742e+01, -5.79129621e+01, 2.23124361e+01, -4.37451926e+00,  2.76815127e-02 ]) 
    # cxm[1,:]=np.array([1.48171843e+01, -5.31901561e+01,  7.60918868e+01, -5.51846846e+01, 2.12359712e+01, -4.15656905e+00,  3.09723630e-02 ])
    # cxm[2,:]=np.array([1.26415104e+01, -4.56361886e+01,  6.57500389e+01, -4.81159578e+01, 1.87476161e+01, -3.73215320e+00, -2.45358044e-02 ])
    # cxm[3,:]=np.array([1.10344258e+01, -3.99142119e+01,  5.76936246e+01, -4.24457366e+01, 1.66941114e+01, -3.37376671e+00, -4.49380604e-02 ])
    # cxm[4,:]=np.array([9.9741693 , -36.19064232,  52.47896315, -38.75624903, 15.32328162,  -3.09800143,  -0.07223029 ])
    # cxm[5,:]=np.array([9.76117497, -35.11883268,  50.48605512, -36.96972057, 14.50139362,  -2.88347426,  -0.08276774]) #0.5
    # cxm[6,:]=np.array([10.38959989, -36.94083878,  52.3841557 , -37.73932243,14.50154753,  -2.76975367,  -0.07371497 ]) #0.4
    # cxm[7,:]=np.array([1.18987101e+01, -4.18327688e+01,  5.84865087e+01, -4.13494763e+01,  1.54611520e+01, -2.78820894e+00, -2.90506536e-02 ]) #0.3
    # cxm[8,:]=np.array([13.77559813, -48.38724031,  67.48002787, -47.40940284, 17.46750576,  -3.01431973,   0.09248942 ]) #0.2
    # cxm[9,:]=np.array([16.73411412, -59.08156701,  82.84718709, -58.44626604, 21.52853771,  -3.72660173,   0.37589346 ]) #0.1

    #extrapolate for amu<0.1
    if amu<=0.1:
        cxu=cxm[9]+(cxm[8]-cxm[9])*(amu-amv[9])/(amv[8]-amv[9])
    else: #interpolate 
        cxu=cxm[idx_low]+(cxm[idx_upp]-cxm[idx_low])*(amu-amv[idx_low])/(amv[idx_upp]-amv[idx_low])

    p=np.poly1d(cxu) #numpy function to generate the RV for any given CCF value

    return p


def dumusque_coeffs(amu):
    coeffs=np.array([-1.51773453,  3.52774949, -3.18794328,  1.22541774,  -0.22479665]) #Polynomial fit to ccf in Fig 2 of Dumusque 2014, plus 400m/s to match Fig6 in Herrero 2016
    p=np.poly1d(coeffs)
    return p


# @profile
def compute_ccf_params(self,rv,ccf,plot_test):
    '''
    Compute the parameters of the CCF and its bisector span 
    (10-40% bottom minus 60-90% top)
    '''
    rvs=np.zeros(len(ccf)) #initialize
    fwhm=np.zeros(len(ccf))
    contrast=np.zeros(len(ccf))
    BIS=np.zeros(len(ccf))

    for i in range(len(ccf)): #loop for each ccf
        ccf[i] = ccf[i] - ccf[i].min() + 0.000001
        #Compute bisector and remove wings
        cutleft0,cutright0,xbis,ybis=nbspectra.speed_bisector_nb(rv,ccf[i]/ccf[i].max(),integrated_bis=True) #FAST
        BIS[i]=np.mean(xbis[np.array(ybis>=0.1) & np.array(ybis<=0.4)])-np.mean(xbis[np.array(ybis<=0.9) & np.array(ybis>=0.6)]) #FAST
        if i==0:
            cutleft,cutright=cutleft0,cutright0
        try:
            popt,_=optimize.curve_fit(nbspectra.gaussian2, rv[cutleft:cutright], ccf[i][cutleft:cutright],p0=[np.max(ccf[i][cutleft:cutright]),rv[cutleft:cutright][np.argmax(ccf[i][cutleft:cutright])]+100,1.5*self.vsini+1000,0.000001]) #fit a gaussian
        except:
            popt=[1.0,100000.0,1,100000.0]
        contrast[i]=popt[0] #amplitude
        rvs[i]=popt[1] #mean
        fwhm[i]=2*m.sqrt(2*np.log(2))*np.abs(popt[2]) #fwhm relation to std

        if plot_test: 
            # plt.plot(rv,1-ccf[i]/ccf[i].max())
            plt.plot(xbis,1-ybis,'b')
            plt.show(block=True)
    

    return rvs, contrast, fwhm, BIS


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




