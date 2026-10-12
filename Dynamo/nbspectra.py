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

# Gauss-Legendre nodes/weights on [0, pi] for the colatitude integral of the exact coverage
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_U = 0.5 * np.pi * (_GL_X + 1.0)
_GL_WU = 0.5 * np.pi * _GL_W


@nb.njit(cache=True, error_model='numpy')
def _cap_half_width(th, theta_c, cr, ctc, stc):
    """Half-width in longitude of the cap at colatitude th (0 .. pi)."""
    den = m.sin(th) * stc
    if den < 1e-14:
        return np.pi if m.cos(th - theta_c) >= cr else 0.0
    c = (cr - m.cos(th) * ctc) / den
    if c <= -1.0:
        return np.pi
    if c >= 1.0:
        return 0.0
    return m.acos(c)


@nb.njit(cache=True, error_model='numpy')
def _cap_segment(theta_c, alpha_c, r, a, b, nphi, out, offset):
    """Add the cap & cell areas for colatitudes in [a, b] (see _cap_ring_overlap)."""
    w = 2.0 * np.pi / nphi
    cr = m.cos(r)
    ctc = m.cos(theta_c)
    stc = m.sin(theta_c)
    for k in range(_GL_U.shape[0]):
        u = _GL_U[k]
        th = a + 0.5 * (b - a) * (1.0 - m.cos(u))
        weight = _GL_WU[k] * 0.5 * (b - a) * m.sin(u) * m.sin(th)
        half = _cap_half_width(th, theta_c, cr, ctc, stc)
        if half <= 0.0:
            continue
        if half >= np.pi:
            for j in range(nphi):
                out[offset + j] += weight * w
            continue
        # interval [s, e] in a frame where cell j spans [j w, (j+1) w]
        s = (alpha_c - half + 0.5 * w) % (2.0 * np.pi)
        e = s + 2.0 * half
        j0 = int(m.floor(s / w))
        j1 = int(m.floor(e / w))
        for jj in range(j0, j1 + 1):
            lo = max(s, jj * w)
            hi = min(e, (jj + 1) * w)
            if hi > lo:
                out[offset + (jj % nphi)] += weight * (hi - lo)


@nb.njit(cache=True, error_model='numpy')
def _cap_ring_overlap(theta_c, alpha_c, r, th1, th2, nphi, out, offset):
    """Add the exact area of (spherical cap) & (each cell of one grid ring) to out.

    Grid frame: colatitude th from the pole facing the observer, longitude alpha as
    in generate_grid_coordinates_nb (cell j of the ring is centred on j*w and spans
    [(j-1/2)w, (j+1/2)w], w = 2 pi / nphi). The cap has centre (theta_c, alpha_c) and
    angular radius r. At colatitude th the cap occupies the longitude interval
    alpha_c +- D(th), cos D = (cos r - cos th cos theta_c) / (sin th sin theta_c).
    The longitude overlap with every cell is exact; the colatitude integral uses
    8-point Gauss-Legendre after th = a + (b-a)(1-cos u)/2, which removes the
    square-root behaviour of D at the cap edges, and is split where the cap starts
    to cover whole circles (th = r - theta_c). Cells of a ring tile the circle, so a
    spot moving in longitude conserves its area exactly.
    """
    a = max(th1, theta_c - r)
    b = min(th2, theta_c + r)
    if a >= b:
        return
    split = r - theta_c          # below this colatitude the whole circle is inside
    if a < split < b:
        _cap_segment(theta_c, alpha_c, r, a, split, nphi, out, offset)
        _cap_segment(theta_c, alpha_c, r, split, b, nphi, out, offset)
    else:
        _cap_segment(theta_c, alpha_c, r, a, b, nphi, out, offset)


@nb.njit(cache=True, error_model='numpy')
def exact_spot_coverage(N, Ngrid_in_ring, vis_spots_idx, spot_pos, vec_spot):
    """Exact spot and facula area fractions of every grid cell.

    Returns (asp, afc) arrays over all cells: asp = sum over visible spots of
    area(spot cap & cell) / area(cell); afc = the same for the facular annulus
    (cap of the facula radius minus the spot cap of the same feature).
    [own assumption] overlaps between DIFFERENT features are not resolved here; sums
    are capped later by adjust_area_fractions. [numerical] exact cap areas use
    8-point Gauss-Legendre quadrature.
    """
    width = np.pi / (2 * N - 1)
    ncell = 0
    for i in range(N):
        ncell += Ngrid_in_ring[i]
    a_sp = np.zeros(ncell)
    a_fc_outer = np.zeros(ncell)
    a_fc = np.zeros(ncell)
    cell_area = np.zeros(ncell)
    # ring colatitude bounds
    offset = 0
    for i in range(N):
        th1 = 0.0 if i == 0 else (i - 0.5) * width
        th2 = 0.5 * width if i == 0 else (i + 0.5) * width
        nphi = Ngrid_in_ring[i]
        ca = (m.cos(th1) - m.cos(th2)) * 2.0 * np.pi / nphi
        for j in range(nphi):
            cell_area[offset + j] = ca
        offset += nphi
    for spot_idx in vis_spots_idx:
        r_sp = spot_pos[spot_idx][2]
        r_fc = spot_pos[spot_idx][3]
        if r_sp <= 0.0 and r_fc <= 0.0:
            continue
        x = min(1.0, max(-1.0, vec_spot[spot_idx][0]))
        theta_c = m.acos(x)
        alpha_c = m.atan2(vec_spot[spot_idx][1], -vec_spot[spot_idx][2])
        rmax = max(r_sp, r_fc)
        offset = 0
        for i in range(N):
            th1 = 0.0 if i == 0 else (i - 0.5) * width
            th2 = 0.5 * width if i == 0 else (i + 0.5) * width
            nphi = Ngrid_in_ring[i]
            if th1 <= theta_c + rmax and th2 >= theta_c - rmax:
                if r_sp > 0.0:
                    _cap_ring_overlap(theta_c, alpha_c, r_sp, th1, th2, nphi, a_sp, offset)
                if r_fc > r_sp:
                    _cap_ring_overlap(theta_c, alpha_c, r_fc, th1, th2, nphi, a_fc_outer, offset)
                    if r_sp > 0.0:
                        # annulus = outer cap - this spot's cap (accumulated per feature)
                        tmp = np.zeros(nphi)
                        _cap_ring_overlap(theta_c, alpha_c, r_sp, th1, th2, nphi, tmp, 0)
                        for j in range(nphi):
                            a_fc[offset + j] -= tmp[j]
            offset += nphi
    for c in range(ncell):
        a_fc[c] += a_fc_outer[c]
        a_sp[c] = a_sp[c] / cell_area[c]
        a_fc[c] = max(0.0, a_fc[c]) / cell_area[c]
    return a_sp, a_fc


@nb.njit(cache=True, error_model='numpy')
def loop_generate_rotating_lc_nb(N, Ngrid_in_ring, proj_area, cos_centers, spot_pos, vec_grid, vec_spot,
                                 simulate_planet, planet_pos, brigh_ph, brigh_sp, brigh_fc, flx_ph, vis,
                                 exact_coverage=True):
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
        Planet positions array of shape (n_planets, 3) with [rho, theta, radius] for each.
        If no planets, should be empty array (0, 3).
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
    exact_coverage : bool
        True: exact spot/facula area of every cell (exact_spot_coverage); False:
        the original linear-in-distance approximation, which does not conserve a
        spot's area as it crosses cells (ripple at P_rot / cells-per-ring).

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
    
    # Check if any planet is transiting
    n_planets = planet_pos.shape[0] if planet_pos.ndim == 2 else 0
    any_planet_visible = vis[-1] == 1.0 if n_planets > 0 else False

    # Initialize area fractions
    Aph = 0.0  # Total photosphere area
    Asp = 0.0  # Total spot area
    Afc = 0.0  # Total facula area
    Apl = 0.0  # Total planet area

    # List to store grid type information
    typ = []

    if exact_coverage:
        ex_sp, ex_fc = exact_spot_coverage(N, Ngrid_in_ring, vis_spots_idx, spot_pos, vec_spot)

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
        if exact_coverage:
            grid_info = exact_grid_cell(iteration, ex_sp[iteration], ex_fc[iteration], vec_grid, width,
                                        simulate_planet, planet_pos, cos_centers, ring_idx,
                                        any_planet_visible)
        elif ring_idx == 0:
            # Central grid (circle)
            grid_info = process_central_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                                             width, simulate_planet, planet_pos, any_planet_visible)
        else:
            # Other grids (square-like)
            grid_info = process_outer_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                                           width, simulate_planet, planet_pos, cos_centers, ring_idx,
                                           any_planet_visible)

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
def exact_grid_cell(iteration, asp, afc, vec_grid, width, simulate_planet, planets_pos,
                    cos_centers, ring_idx, any_planet_visible):
    """Cell fractions from exact spot/facula coverage plus the planet coverage."""
    apl = 0.0
    n_planets = planets_pos.shape[0] if planets_pos.ndim == 2 else 0
    if simulate_planet and any_planet_visible and n_planets > 0:
        if ring_idx == 0:
            width2 = 2 * m.sin(width / 2)
        else:
            width2 = cos_centers[ring_idx] * width
        for p_idx in range(n_planets):
            planet_pos = planets_pos[p_idx]
            dist = m.sqrt((planet_pos[0] * m.cos(planet_pos[1]) - vec_grid[iteration, 1]) ** 2 +
                          (planet_pos[0] * m.sin(planet_pos[1]) - vec_grid[iteration, 2]) ** 2)
            if dist > width2 / 2 + planet_pos[2]:
                apl_single = 0.0
            elif dist < planet_pos[2] - width2 / 2:
                apl_single = 1.0
            else:
                apl_single = -(dist - planet_pos[2] - width2 / 2) / width2
            apl += apl_single
    asp, afc, apl = adjust_area_fractions(asp, afc, apl)
    return 1 - asp - afc - apl, asp, afc, apl


@nb.njit(cache=True)
def process_central_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                         width, simulate_planet, planets_pos, any_planet_visible):
    """
    Process the central grid element (circular).

    Returns:
    --------
    aph, asp, afc, apl : float
        Area fractions of photosphere, spots, faculae, and planet(s)
    """
    # Initialize area fractions for this grid
    asp = 0.0  # Fraction covered by spots
    afc = 0.0  # Fraction covered by faculae
    apl = 0.0  # Fraction covered by planet(s)

    # Process spots and faculae
    for spot_idx in vis_spots_idx:
        spot_radius = spot_pos[spot_idx][2]
        facula_radius = spot_pos[spot_idx][3]
        # Skip if neither a spot nor a facula is present
        if spot_radius == 0.0 and facula_radius == 0.0:
            continue

        # Calculate angular distance between grid and spot
        dist = m.acos(min(1.0, max(-1.0, np.dot(vec_grid[iteration], vec_spot[spot_idx]))))

        # Spot coverage of this grid element
        cov_sp = 0.0
        if spot_radius > 0.0:
            cov_sp = calculate_circular_coverage(dist, width, spot_radius)
        asp += cov_sp

        # Facula coverage: the annulus around THIS spot (excludes this spot's area)
        if facula_radius > 0.0:
            facula_coverage = calculate_circular_coverage(dist, width, facula_radius)
            afc += max(0.0, facula_coverage - cov_sp)

    # Process ALL planets (sum coverage)
    n_planets = planets_pos.shape[0] if planets_pos.ndim == 2 else 0
    if simulate_planet and any_planet_visible and n_planets > 0:
        width2 = 2 * m.sin(width / 2)
        for p_idx in range(n_planets):
            planet_pos = planets_pos[p_idx]
            # Calculate grid-planet distance
            dist = m.sqrt((planet_pos[0] * m.cos(planet_pos[1]) - vec_grid[iteration, 1]) ** 2 +
                          (planet_pos[0] * m.sin(planet_pos[1]) - vec_grid[iteration, 2]) ** 2)

            if dist > width2 / 2 + planet_pos[2]:
                apl_single = 0.0
            elif dist < planet_pos[2] - width2 / 2:
                apl_single = 1.0
            else:
                apl_single = -(dist - planet_pos[2] - width2 / 2) / width2
            apl += apl_single

    # Apply constraints and adjust area fractions
    asp, afc, apl = adjust_area_fractions(asp, afc, apl)

    # Calculate photosphere fraction
    aph = 1 - asp - afc - apl

    return aph, asp, afc, apl


@nb.njit(cache=True)
def process_outer_grid(iteration, vis_spots_idx, spot_pos, vec_grid, vec_spot,
                       width, simulate_planet, planets_pos, cos_centers, ring_idx,
                       any_planet_visible):
    """
    Process grid elements in outer rings (square-like).

    Returns:
    --------
    aph, asp, afc, apl : float
        Area fractions of photosphere, spots, faculae, and planet(s)
    """
    # Initialize area fractions for this grid
    asp = 0.0  # Fraction covered by spots
    afc = 0.0  # Fraction covered by faculae
    apl = 0.0  # Fraction covered by planet(s)

    # Process spots and faculae
    for spot_idx in vis_spots_idx:
        spot_radius = spot_pos[spot_idx][2]
        facula_radius = spot_pos[spot_idx][3]
        # Skip if neither a spot nor a facula is present
        if spot_radius == 0.0 and facula_radius == 0.0:
            continue

        # Calculate angular distance between grid and spot
        dist = m.acos(min(1.0, max(-1.0, np.dot(vec_grid[iteration], vec_spot[spot_idx]))))

        # Spot coverage of this grid element
        cov_sp = 0.0
        if spot_radius > 0.0:
            cov_sp = calculate_square_coverage(dist, width, spot_radius)
        asp += cov_sp

        # Facula coverage: the annulus around THIS spot (excludes this spot's area)
        if facula_radius > 0.0:
            facula_coverage = calculate_square_coverage(dist, width, facula_radius)
            afc += max(0.0, facula_coverage - cov_sp)

    # Process ALL planets (sum coverage)
    n_planets = planets_pos.shape[0] if planets_pos.ndim == 2 else 0
    if simulate_planet and any_planet_visible and n_planets > 0:
        width2 = cos_centers[ring_idx] * width
        for p_idx in range(n_planets):
            planet_pos = planets_pos[p_idx]
            # Calculate grid-planet distance
            dist = m.sqrt((planet_pos[0] * m.cos(planet_pos[1]) - vec_grid[iteration, 1]) ** 2 +
                          (planet_pos[0] * m.sin(planet_pos[1]) - vec_grid[iteration, 2]) ** 2)

            if dist > width2 / 2 + planet_pos[2]:
                apl_single = 0.0
            elif dist < planet_pos[2] - width2 / 2:
                apl_single = 1.0
            else:
                apl_single = -(dist - planet_pos[2] - width2 / 2) / width2
            apl += apl_single

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

