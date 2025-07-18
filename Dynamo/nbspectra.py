#NUMBA ############################################
import numba as nb
import numpy as np
import math as m

@nb.njit
def dummy():
    return None

@nb.njit(cache=True,error_model='numpy')
def normalize_spectra_nb(bins,wavelength,flux):

    x_bin=np.zeros(len(bins)-1)
    y_bin=np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        idxup = wavelength>bins[i]
        idxdown= wavelength<bins[i+1]
        idx=idxup & idxdown
        y_bin[i]=flux[idx].max()
        x_bin[i]=wavelength[idx][np.argmax(flux[idx])]
    #divide by 6th deg polynomial

    return x_bin, y_bin

@nb.njit(cache=True,error_model='numpy')
def limb_darkening_law(LD_law,LD1,LD2,amu):

    if LD_law == 'linear':
        mu=1-LD1*(1-amu)

    elif LD_law == 'quadratic':
        a=2*np.sqrt(LD1)*LD2
        b=np.sqrt(LD1)*(1-2*LD2)
        mu=1-a*(1-amu)-b*(1-amu)**2

    elif LD_law == 'sqrt':
        a=np.sqrt(LD1)*(1-2*LD2) 
        b=2*np.sqrt(LD1)*LD2
        mu=1-a*(1-amu)-b*(1-np.sqrt(amu))

    elif LD_law == 'log':
        a=LD2*LD1**2+1
        b=LD1**2-1
        mu=1-a*(1-amu)-b*amu*(1-np.log(amu))

    else:
        print('LD law not valid.')

    return mu

@nb.njit(cache=True,error_model='numpy')
def compute_spot_position(t,spot_map,ref_time,Prot,diff_rot,Revo,Q):
    pos=np.zeros((len(spot_map),4))

    for i in range(len(spot_map)):
        tini = spot_map[i][0] #time of spot apparence
        dur = spot_map[i][1] #duration of the spot
        tfin = tini + dur #final time of spot
        colat = spot_map[i][2] #colatitude
        lat = 90 - colat #latitude
        longi = spot_map[i][3] #longitude
        Rcoef = spot_map[i][4:7] #coefficients for the evolution od the radius. Depends on the desired law.

        pht = longi + (t-ref_time)/Prot%1*360
        #update longitude adding diff rotation
        phsr= pht + (t-ref_time)*diff_rot*(1.698*m.sin(np.deg2rad(lat))**2+2.346*m.sin(np.deg2rad(lat))**4)


        if Revo == 'constant':
            if t>=tini and t<=tfin: 
                rad=Rcoef[0] 
            else:
                rad=0.0
        elif Revo == 'linear':
            if t>=tini and t<=tfin:
                rad=Rcoef[0]+(t-tini)*(Rcoef[1]-Rcoef[0])/dur
            else:
                rad=0.0
        elif Revo == 'quadratic':
            if t>=tini and t<=tfin:
                rad=-4*Rcoef[0]*(t-tini)*(t-tini-dur)/dur**2
            else:
                rad=0.0
        
        else:
            print('Spot evolution law not implemented yet. Only constant and linear are implemented.')
        
        if Q!=0.0: #to speed up the code when no fac are present
            rad_fac=np.deg2rad(rad)*m.sqrt(1+Q) 
        else: rad_fac=0.0

        pos[i]=np.array([np.deg2rad(colat), np.deg2rad(phsr), np.deg2rad(rad), rad_fac])
        #return position and radii of spots at t in radians.

    return pos


@nb.njit(cache=True,error_model='numpy')
def compute_planet_pos(t,esinw,ecosw,T0p,Pp,rad_pl,b,a,alp):
    
    if(esinw==0 and ecosw==0):
       ecc=0
       omega=0
    else:
       ecc=m.sqrt(esinw**2+ecosw**2)
       omega=m.atan2(esinw,ecosw)

    t_peri = Ttrans_2_Tperi(T0p,Pp, ecc, omega)
    sinf,cosf=true_anomaly(t,Pp,ecc,t_peri)


    cosftrueomega=cosf*m.cos(omega+m.pi/2)-sinf*m.sin(omega+np.pi/2) #cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
    sinftrueomega=cosf*m.sin(omega+m.pi/2)+sinf*m.cos(omega+np.pi/2) #sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

    if cosftrueomega>0.0: return np.array([1+rad_pl*2, 0.0, rad_pl]) #avoid secondary transits

    cosi = (b/a)*(1+esinw)/(1-ecc**2) #cosine of planet inclination (i=90 is transit)

    rpl=a*(1-ecc**2)/(1+ecc*cosf)
    xpl=rpl*(-m.cos(alp)*sinftrueomega-m.sin(alp)*cosftrueomega*cosi)
    ypl=rpl*(m.sin(alp)*sinftrueomega-m.cos(alp)*cosftrueomega*cosi)

    rhopl=m.sqrt(ypl**2+xpl**2)
    thpl=m.atan2(ypl,xpl)
    pos=np.array([rhopl, thpl, rad_pl],dtype=np.float64) #rho, theta, and radii (in Rstar) of the planet
    return pos


@nb.njit(cache=True,error_model='numpy')
def Ttrans_2_Tperi(T0, P, e, w):

    f = m.pi/2 - w
    E = 2 * m.atan(m.tan(f/2.) * m.sqrt((1.-e)/(1.+e)))  # eccentric anomaly
    Tp = T0 - P/(2*np.pi) * (E - e*m.sin(E))      # time of periastron

    return Tp

@nb.njit(cache=True,error_model='numpy')
def true_anomaly(x,period,ecc,tperi):
    fmean=2.0*m.pi*(x-tperi)/period
    #Solve by Newton's method x(n+1)=x(n)-f(x(n))/f'(x(n))
    fecc=fmean
    diff=1.0
    while(diff>1.0E-6):
        fecc_0=fecc
        fecc=fecc_0-(fecc_0-ecc*m.sin(fecc_0)-fmean)/(1.0-ecc*m.cos(fecc_0))
        diff=m.fabs(fecc-fecc_0)
    sinf=m.sqrt(1.0-ecc*ecc)*m.sin(fecc)/(1.0-ecc*m.cos(fecc))
    cosf=(m.cos(fecc)-ecc)/(1.0-ecc*m.cos(fecc))
    return sinf, cosf
########################################################################################
########################################################################################
#                              SPECTROSCOPY FUNCTIONS  FOR SPHERICAL GRID              #
########################################################################################
########################################################################################



#with this the x and y width of each grid is the same, thus the area of the grids is similar in all the sphere, avoiding an over/under sampling of the poles/center
@nb.njit(cache=True,error_model='numpy')
def generate_grid_coordinates_nb(N):

    Nt=2*N-1 #N is number of concentric rings. Nt is counting them two times minus the center one.
    width=180.0/(2*N-1) #width of one grid element.

    centres=np.append(0,np.linspace(width,90-width/2,N-1)) #colatitudes of the concentric grids. The pole of the grid faces the observer.
    anglesout=np.linspace(0,360-width,2*Nt) #longitudes of the grid edges of the most external grid. This grids fix the area of the grids in other rings.
    
    radi=np.sin(np.pi*centres/180) #projected polar radius of the ring.
    amu=np.cos(np.pi*centres/180) #amus

    ts=[0.0] #central grid radius
    alphas=[0.0] #central grid angle

    area=[2.0*np.pi*(1.0-np.cos(width*np.pi/360.0))] #area of spherical cap (only for the central element)
    parea=[np.pi*np.sin(width*np.pi/360.0)**2]

    Ngrid_in_ring=[1]
    
    for i in range(1,len(amu)): #for each ring except firs
        Nang=int(round(len(anglesout)*(radi[i]))) #Number of longitudes to have grids of same width
        w=360/Nang #width i angles
        Ngrid_in_ring.append(Nang)

        angles=np.linspace(0,360-w,Nang)
        area.append(radi[i]*width*w*np.pi*np.pi/(180*180)) #area of each grid
        parea.append(amu[i]*area[-1]) #PROJ. AREA OF THE GRID

        for j in range(Nang):
            ts.append(centres[i]) #latitude
            alphas.append(angles[j]) #longitude


    alphas=np.array(alphas) #longitude of grid (pole faces observer)
    ts=np.array(ts) #colatitude of grid
    Ngrids=len(ts)  #number of grids

    rs = np.sin(np.pi*ts/180) #projected polar radius of grid

    xs = np.cos(np.pi*ts/180) #grid elements in cartesian coordinates. Note that pole faces the observer.
    ys = rs*np.sin(np.pi*alphas/180)
    zs = -rs*np.cos(np.pi*alphas/180)

    return Ngrids,Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, area, parea

@nb.njit(cache=True, error_model='numpy')
def loop_generate_rotating_lc_nb(N, Ngrid_in_ring, proj_area, cos_centers, spot_pos, vec_grid, vec_spot,
                                 simulate_planet, planet_pos, brigh_ph, brigh_sp, brigh_fc, flx_ph, vis):
    """
    Calculate the flux emitted from a rotating star with spots, faculae, and possibly planets.

    Parameters:
    -----------
    N : int
        Number of concentric rings in the grid
    Ngrid_in_ring : array
        Number of grid elements in each ring
    proj_area : array
        Projected area of each ring
    cos_centers : array
        Cosine of the angles at the centers
    spot_pos : array
        Positions and sizes of spots [theta, phi, radius_spot, radius_facula]
    vec_grid : array
        Unit vectors for each grid point
    vec_spot : array
        Unit vectors for each spot
    simulate_planet : bool
        Whether to simulate a planet
    planet_pos : array
        Planet position [distance, angle, radius]
    brigh_ph : array
        Brightness of photosphere for each ring
    brigh_sp : array
        Brightness of spots for each ring
    brigh_fc : array
        Brightness of faculae for each ring
    flx_ph : float
        Total flux of photosphere
    vis : array
        Visibility of each spot and the planet

    Returns:
    --------
    flux : float
        Calculated flux
    typ : list
        Type of grid elements (photosphere, spot, facula, planet) fractions
    Aph, Asp, Afc, Apl : float
        Total area fractions of photosphere, spots, faculae, and planet
    """
    # Define constants
    width = np.pi / (2 * N - 1)  # Width of one grid element in radians
    flux = np.float64(flx_ph)  # Ensure flux is float64 to prevent overflow

    # Get indices of visible spots for efficient iteration
    vis_spots_idx = [i for i in range(len(vis) - 1) if vis[i] == 1.0]

    # Initialize area fractions
    Aph = 0.0  # Total photosphere area
    Asp = 0.0  # Total spot area
    Afc = 0.0  # Total facula area
    Apl = 0.0  # Total planet area

    # List to store grid type information
    typ = []

    # Process each grid element
    for iteration in range(sum(Ngrid_in_ring)):
        # Get the ring index for the current grid element
        ring_idx = 0
        if iteration > 0:
            sum_grids = 0
            for i in range(N):
                sum_grids += Ngrid_in_ring[i]
                if iteration < sum_grids:
                    ring_idx = i
                    break

        # Get the grid element characteristics based on ring location
        if ring_idx == 0:
            # Central grid (circle)
            grid_info = process_central_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                                             width, simulate_planet, planet_pos, vis)
        else:
            # Other grids (square-like)
            grid_info = process_outer_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                                           width, simulate_planet, planet_pos, vis, ring_idx, cos_centers)

        # Unpack grid coverage information
        aph, asp, afc, apl = grid_info

        # Add to the total flux
        flux = flux - (1 - aph) * brigh_ph[ring_idx] + asp * brigh_sp[ring_idx] + afc * brigh_fc[ring_idx]

        # Update area totals
        Aph += aph * proj_area[ring_idx]
        Asp += asp * proj_area[ring_idx]
        Afc += afc * proj_area[ring_idx]
        Apl += apl * proj_area[ring_idx]

        # Store grid type information
        typ.append([aph, asp, afc, apl])

    return flux, typ, Aph, Asp, Afc, Apl


@nb.njit(cache=True)
def process_central_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                         width, simulate_planet, planet_pos, vis):
    """
    Process the central grid element (circular).

    Returns:
    --------
    aph, asp, afc, apl : float
        Area fractions of photosphere, spots, faculae, and planet
    """
    # Initialize area fractions for this grid
    asp = 0.0  # Fraction covered by spots
    afc = 0.0  # Fraction covered by faculae
    apl = 0.0  # Fraction covered by planet

    # Process spots and faculae
    for spot_idx in vis_spots_idx:
        # Skip if spot has zero radius
        if spot_pos[spot_idx][2] == 0.0:
            continue

        # Calculate angular distance between grid and spot
        dist = m.acos(np.dot(vec_grid[iteration], vec_spot[spot_idx]))

        # Calculate spot coverage
        spot_radius = spot_pos[spot_idx][2]
        asp += calculate_circular_coverage(dist, width, spot_radius)

        # Calculate facula coverage (if present)
        facula_radius = spot_pos[spot_idx][3]
        if facula_radius > 0.0:
            facula_coverage = calculate_circular_coverage(dist, width, facula_radius)
            # Facula area excludes the spot area
            afc += max(0.0, facula_coverage - asp)

    # Process planet (if visible)
    if simulate_planet and vis[-1] == 1.0:
        # Calculate grid-planet distance
        dist = m.sqrt((planet_pos[0] * m.cos(planet_pos[1]) - vec_grid[iteration, 1]) ** 2 +
                      (planet_pos[0] * m.sin(planet_pos[1]) - vec_grid[iteration, 2]) ** 2)

        width2 = 2 * m.sin(width / 2)
        if dist > width2 / 2 + planet_pos[2]:
            apl = 0.0
        elif dist < planet_pos[2] - width2 / 2:
            apl = 1.0
        else:
            apl = -(dist - planet_pos[2] - width2 / 2) / width2

    # Apply constraints and adjust area fractions
    asp, afc, apl = adjust_area_fractions(asp, afc, apl)

    # Calculate photosphere fraction
    aph = 1 - asp - afc - apl

    return aph, asp, afc, apl


@nb.njit(cache=True)
def process_outer_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                       width, simulate_planet, planet_pos, vis, ring_idx, cos_centers):
    """
    Process grid elements in outer rings (square-like).

    Returns:
    --------
    aph, asp, afc, apl : float
        Area fractions of photosphere, spots, faculae, and planet
    """
    # Initialize area fractions for this grid
    asp = 0.0  # Fraction covered by spots
    afc = 0.0  # Fraction covered by faculae
    apl = 0.0  # Fraction covered by planet

    # Process spots and faculae
    for spot_idx in vis_spots_idx:
        # Skip if spot has zero radius
        if spot_pos[spot_idx][2] == 0.0:
            continue

        # Calculate angular distance between grid and spot
        dist = m.acos(np.dot(vec_grid[iteration], vec_spot[spot_idx]))

        # Calculate spot coverage for outer grid (square-like)
        spot_radius = spot_pos[spot_idx][2]
        asp += calculate_square_coverage(dist, width, spot_radius)

        # Calculate facula coverage (if present)
        facula_radius = spot_pos[spot_idx][3]
        if facula_radius > 0.0:
            facula_coverage = calculate_square_coverage(dist, width, facula_radius)
            # Facula area excludes the spot area
            afc += max(0.0, facula_coverage - asp)

    # Process planet (if visible)
    if simulate_planet and vis[-1] == 1.0:
        # Calculate grid-planet distance
        dist = m.sqrt((planet_pos[0] * m.cos(planet_pos[1]) - vec_grid[iteration, 1]) ** 2 +
                      (planet_pos[0] * m.sin(planet_pos[1]) - vec_grid[iteration, 2]) ** 2)

        width2 = cos_centers[ring_idx] * width
        if dist > width2 / 2 + planet_pos[2]:
            apl = 0.0
        elif dist < planet_pos[2] - width2 / 2:
            apl = 1.0
        else:
            apl = -(dist - planet_pos[2] - width2 / 2) / width2

    # Apply constraints and adjust area fractions
    asp, afc, apl = adjust_area_fractions(asp, afc, apl)

    # Calculate photosphere fraction
    aph = 1 - asp - afc - apl

    return aph, asp, afc, apl


@nb.njit(cache=True)
def calculate_circular_coverage(dist, width, radius):
    """
    Calculate the coverage fraction for a circular feature (central grid).

    Parameters:
    -----------
    dist : float
        Angular distance between grid center and feature center
    width : float
        Width of grid element in radians
    radius : float
        Radius of the feature in radians

    Returns:
    --------
    coverage : float
        Fraction of grid covered by the feature
    """
    # No coverage if feature is too far
    if dist > (width / 2 + radius):
        return 0.0

    # Feature potentially covers grid completely
    if (width / 2) < radius:
        if dist <= radius - (width / 2):  # Grid completely covered
            return 1.0
        else:  # Grid partially covered
            return -(dist - radius - width / 2) / width

    # Grid potentially covers feature completely
    else:
        if dist <= (width / 2 - radius):  # Feature completely inside grid
            return (2 * radius / width) ** 2
        else:  # Grid partially covered
            return -2 * radius * (dist - width / 2 - radius) / width ** 2


@nb.njit(cache=True)
def calculate_square_coverage(dist, width, radius):
    """
    Calculate the coverage fraction for a circular feature on a square-like grid.

    Parameters:
    -----------
    dist : float
        Angular distance between grid center and feature center
    width : float
        Width of grid element in radians
    radius : float
        Radius of the feature in radians

    Returns:
    --------
    coverage : float
        Fraction of grid covered by the feature
    """
    # No coverage if feature is too far
    if dist > (width / 2 + radius):
        return 0.0

    # Feature potentially covers grid completely
    if (width / m.sqrt(2)) < radius:
        if dist <= (m.sqrt(radius ** 2 - (width / 2) ** 2) - width / 2):  # Grid completely covered
            return 1.0
        else:  # Grid partially covered
            return -(dist - radius - width / 2) / (width + radius - m.sqrt(radius ** 2 - (width / 2) ** 2))

    # Grid potentially covers feature completely
    elif (width / 2) > radius:
        if dist <= (width / 2 - radius):  # Feature completely inside grid
            return (np.pi / 4) * (2 * radius / width) ** 2
        else:  # Grid partially covered
            return (np.pi / 4) * ((2 * radius / width) ** 2 - (2 * radius / width ** 2) * (dist - width / 2 + radius))

    # Feature is larger than grid but not enough to cover it completely
    else:
        A1 = (width / 2) * m.sqrt(radius ** 2 - (width / 2) ** 2)
        A2 = (radius ** 2 / 2) * (m.pi / 2 - 2 * m.asin(m.sqrt(radius ** 2 - (width / 2) ** 2) / radius))
        Ar = 4 * (A1 + A2) / width ** 2
        return -Ar * (dist - width / 2 - radius) / (width / 2 + radius)


@nb.njit(cache=True)
def adjust_area_fractions(asp, afc, apl):
    """
    Apply constraints to ensure area fractions are physically valid.

    Parameters:
    -----------
    asp, afc, apl : float
        Initial area fractions for spots, faculae, and planet

    Returns:
    --------
    asp, afc, apl : float
        Adjusted area fractions
    """
    # Cap individual fractions to 1.0
    asp = min(asp, 1.0)
    afc = min(afc, 1.0)
    apl = min(apl, 1.0)

    # Adjust for spot and facula overlap
    if afc + asp > 1.0:
        afc = 1.0 - asp

    # Adjust for planet overlap (planet takes precedence)
    if apl > 0.0:
        asp = asp * (1 - apl)
        afc = afc * (1 - apl)

    return asp, afc, apl

