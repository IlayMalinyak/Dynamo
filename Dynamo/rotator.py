import numpy as np
from . import nbspectra
from . import vizualisation as viz
import sys
import math as m
import matplotlib.pyplot as plt
import astropy.units as u

D2S = 1*u.day.to(u.s)

def gaussian_spots(t, tau_emerge, tau_decay):
    area_factor = np.ones_like(t)

    l = t < 0
    area_factor[l] *= np.exp(-t[l]**2 / tau_emerge**2) # emergence

    l = t > 0
    area_factor[l] *= np.exp(-t[l]**2 / tau_decay**2) # decay

    return area_factor

class Rotator():
    def __init__(self, star):
        self.star = star

    def generate_rotating_photosphere_lc(self, Ngrid_in_ring, proj_area, cos_centers,
                                         brigh_grid_ph, brigh_grid_sp, brigh_grid_fc,
                                         flx_ph, vec_grid, plot_map=True):
        '''Loop for all the pixels and assign the flux corresponding to the grid element.
        '''
        simulate_planet = self.star.simulate_planet
        N = self.star.n_grid_rings  # Number of concentric rings

        # Ensure flx_ph is not too small to prevent division overflow
        if np.abs(flx_ph) < 1e-10:
            flx_ph = 1e-10 * (1.0 if flx_ph >= 0 else -1.0)
            print("Warning: flx_ph is very small, set to", flx_ph)

        # Initialize flux and filling factor arrays
        flux = np.zeros([len(self.star.obs_times)])
        filling_sp = np.zeros(len(self.star.obs_times))
        filling_ph = np.zeros(len(self.star.obs_times))
        filling_pl = np.zeros(len(self.star.obs_times))
        filling_fc = np.zeros(len(self.star.obs_times))

        spots_positions_arr = np.zeros((len(self.star.obs_times), len(self.star.spot_map), 3))

        for k, t in enumerate(self.star.obs_times):
            typ = []  # type of grid, ph sp or fc

            if simulate_planet:
                planet_pos = self.compute_planet_pos(t)  # compute the planet position at current time
            else:
                planet_pos = [2.0, 0.0, 0.0]  # Position outside the star

            if self.star.spot_map.size == 0:
                spot_pos = np.array([np.array([m.pi / 2, -m.pi, 0.0, 0.0])])
            else:
                spot_pos = self.compute_spot_position_vec(t)  # compute the position of all spots

            spots_positions_arr[k, :, :] = spot_pos[:, :3]

            # Compute spot vectors
            vec_spot = np.zeros([len(self.star.spot_map), 3])
            xspot = np.cos(self.star.inclination) * np.sin(spot_pos[:, 0]) * np.cos(spot_pos[:, 1]) + np.sin(
                self.star.inclination) * np.cos(spot_pos[:, 0])
            yspot = np.sin(spot_pos[:, 0]) * np.sin(spot_pos[:, 1])
            zspot = np.cos(spot_pos[:, 0]) * np.cos(self.star.inclination) - np.sin(self.star.inclination) * np.sin(
                spot_pos[:, 0]) * np.cos(spot_pos[:, 1])
            vec_spot[:, :] = np.array([xspot, yspot, zspot]).T  # spot center in cartesian

            # Compute visibility of spots
            vis = np.zeros(len(vec_spot) + 1)
            for i in range(len(vec_spot)):
                dist = m.acos(np.dot(vec_spot[i], np.array([1, 0, 0])))
                if (dist - spot_pos[i, 2] * np.sqrt(1 + self.star.facular_area_ratio)) <= (np.pi / 2):
                    vis[i] = 1.0

            if (planet_pos[0] - planet_pos[2] < 1):
                vis[-1] = 1.0

            # Calculate flux
            try:
                # If no spots are visible, use a faster calculation
                if np.sum(vis) == 0.0:
                    flux[k] = flx_ph
                    typ = [[1.0, 0.0, 0.0, 0.0]] * np.sum(Ngrid_in_ring)
                    filling_ph[k] = np.dot(Ngrid_in_ring, proj_area)
                    filling_sp[k] = 0.0
                    filling_fc[k] = 0.0
                    filling_pl[k] = 0.0
                else:
                    # Call the numba function with bounds checking
                    flux[k], typ, filling_ph[k], filling_sp[k], filling_fc[k], filling_pl[k] = \
                        nbspectra.loop_generate_rotating_lc_nb(
                            N, Ngrid_in_ring, proj_area, cos_centers, spot_pos, vec_grid,
                            vec_spot, simulate_planet, planet_pos, brigh_grid_ph, brigh_grid_sp,
                            brigh_grid_fc, flx_ph, vis)

                    # Check for non-finite values
                    if not np.isfinite(flux[k]):
                        print(f"Warning: Non-finite flux detected at time {t}, resetting to flx_ph")
                        flux[k] = flx_ph

                    # Check for extreme values
                    if np.abs(flux[k] / flx_ph) > 10.0:
                        print(f"Warning: Extreme flux detected at time {t}: {flux[k]} (ratio: {flux[k] / flx_ph})")
            except Exception as e:
                # Handle any runtime errors
                print(f"Error at time {t}: {str(e)}")
                flux[k] = flx_ph
                filling_ph[k] = np.dot(Ngrid_in_ring, proj_area)
                filling_sp[k] = 0.0
                filling_fc[k] = 0.0
                filling_pl[k] = 0.0

            # Calculate percentages for filling factors
            total_area = np.dot(Ngrid_in_ring, proj_area)
            filling_ph[k] = 100 * filling_ph[k] / total_area
            filling_sp[k] = 100 * filling_sp[k] / total_area
            filling_fc[k] = 100 * filling_fc[k] / total_area
            filling_pl[k] = 100 * filling_pl[k] / total_area

            # Status output
            # sys.stdout.write(
            #     "\rDate {0}. ff_ph={1:.3f}%. ff_sp={2:.3f}%. ff_fc={3:.3f}%. ff_pl={4:.3f}%. flux={5:.3f}%. flx_ph={6:.3f}%."
            #     " [{7}/{8}]%".format(
            #         t, filling_ph[k], filling_sp[k], filling_fc[k], filling_pl[k], flux[k], flx_ph, k + 1,
            #         len(self.star.obs_times)))

        # Create plots if requested
        if plot_map:
            plt.scatter(spots_positions_arr[:, :, 1] % (2 * np.pi), spots_positions_arr[:, :, 0], )
            plt.xlabel('lon')
            plt.ylabel('lat')
            plt.show()
            filename = self.star.path / 'plots' / 'spot_map.gif'
            spots_per_day = spots_positions_arr[::48 * 4, :, :]
            viz.plot_objects_on_sphere(spots_per_day, show_animation=True, save_animation=filename.as_posix())

        # Return with normalized flux and clamp extreme values
        normalized_flux = flux / flx_ph
        # Find and report any extreme values
        extreme_indices = np.where(np.abs(normalized_flux) > 10.0)[0]
        if len(extreme_indices) > 0:
            print(f"Warning: {len(extreme_indices)} extreme flux values detected")
            print(f"Extreme indices: {extreme_indices}")
            print(f"Extreme values: {normalized_flux[extreme_indices]}")
            # Clamp extreme values to a reasonable range
            normalized_flux = np.clip(normalized_flux, -10.0, 10.0)

        return spots_positions_arr, normalized_flux, filling_ph, filling_sp, filling_fc, filling_pl

    def compute_spot_position(self, t):

        pos = np.zeros([len(self.star.spot_map), 4])

        for i in range(len(self.star.spot_map)):
            tini = self.star.spot_map[i][0]  # time of spot apparence
            dur = self.star.spot_map[i][1]  # duration of the spot
            tfin = tini + dur  # final time of spot
            colat = self.star.spot_map[i][2]  # colatitude
            lat = 90 - colat  # latitude
            longi = self.star.spot_map[i][3]  # longitude
            Rcoef = self.star.spot_map[i][4::]  # coefficients for the evolution od the radius. Depends on the desired law.

            time_diff = t = self.star.reference_time
            # update longitude adding diff rotation

            omega = 2 * np.pi / (self.star.rotation_period * D2S)

            phase = omega * (1 - self.star.differential_rotation * np.sin(lat) ** 2)

            phsr = longi + omega * time_diff * D2S * phase


            if self.star.spots_evo_law == 'constant':
                if t >= tini and t <= tfin:
                    rad = Rcoef[0]
                else:
                    rad = 0.0

            elif self.star.spots_evo_law == 'linear':
                if t >= tini and t <= tfin:
                    rad = Rcoef[0] + (t - tini) * (Rcoef[1] - Rcoef[0]) / dur
                else:
                    rad = 0.0
            elif self.star.spots_evo_law == 'quadratic':
                if t >= tini and t <= tfin:
                    rad = -4 * Rcoef[0] * (t - tini) * (t - tini - dur) / dur ** 2
                else:
                    rad = 0.0

            else:
                sys.exit('Spot evolution law not implemented yet')

            if self.star.facular_area_ratio != 0.0:  # to speed up the code when no fac are present
                rad_fac = np.deg2rad(rad) * np.sqrt(1 + self.star.facular_area_ratio)
            else:
                rad_fac = 0.0

            pos[i] = np.array([np.deg2rad(colat), np.deg2rad(phsr), np.deg2rad(rad), rad_fac])
            # return position and radii of spots at t in radians.

        return pos

    def compute_spot_position_vec(self, t):
        # Get the number of spots
        num_spots = len(self.star.spot_map)

        # Extract all parameters from spot_map into arrays
        spot_map = np.array(self.star.spot_map)
        tini = spot_map[:, 0]  # time of spot appearance
        dur = spot_map[:, 1]  # duration of the spot
        tfin = tini + dur  # final time of spot
        colat = spot_map[:, 2]  # colatitude
        lat = 90 - colat  # latitude
        longi = spot_map[:, 3]  # longitude

        time_diff = t - self.star.reference_time

        # Calculate updated longitude with differential rotation

        omega = 2 * np.pi / (self.star.rotation_period * D2S)

        phase = omega * (1 - self.star.differential_rotation * np.sin(np.deg2rad(lat)) ** 2)

        phsr = longi + np.rad2deg(time_diff * D2S * phase)


        # base_rotation = longi + (time_diff / self.star.rotation_period % 1) * 360
        # diff_rotation = time_diff * self.star.differential_rotation * (
        #         1.698 * np.sin(np.deg2rad(lat)) ** 2 + 2.346 * np.sin(np.deg2rad(lat)) ** 4
        # )
        # pht = base_rotation + diff_rotation
        # phsr = pht % 360  # make the phase between 0 and 360

        # Initialize radius array
        rad = np.zeros(num_spots)

        # Create time conditions
        active_spots = (t >= tini) & (t <= tfin)

        # Handle different evolution laws
        if self.star.spots_evo_law == 'constant':
            # Extract first coefficient for all spots
            Rcoef_0 = spot_map[:, 4]
            rad[active_spots] = Rcoef_0[active_spots]

        elif self.star.spots_evo_law == 'linear':
            # Extract first and second coefficients
            Rcoef_0 = spot_map[:, 4]
            Rcoef_1 = spot_map[:, 5]

            # Calculate linear evolution
            time_factor = (t - tini[active_spots]) / dur[active_spots]
            rad[active_spots] = Rcoef_0[active_spots] + time_factor * (Rcoef_1[active_spots] - Rcoef_0[active_spots])

        elif self.star.spots_evo_law == 'quadratic':
            # Extract coefficient
            Rcoef_0 = spot_map[:, 4]

            # Calculate quadratic evolution
            time_from_start = t - tini[active_spots]
            time_from_end = time_from_start - dur[active_spots]
            rad[active_spots] = -4 * Rcoef_0[active_spots] * time_from_start * time_from_end / dur[active_spots] ** 2

        elif self.star.spots_evo_law == 'gaussian':
            Rcoef_0 = spot_map[:, 4]
            factor = np.ones_like(Rcoef_0)
            tt = t - tini
            factor[tt < 0] = np.exp(-tt[tt < 0] ** 2 / self.star.tau_emerge ** 2)
            factor[tt >= 0] = np.exp(-tt[tt >= 0] ** 2 / self.star.tau_decay ** 2)
            rad = factor * Rcoef_0
        else:
            sys.exit('Spot evolution law not implemented yet')

        # Calculate facular radius
        if self.star.facular_area_ratio != 0.0:
            rad_fac = np.deg2rad(rad) * np.sqrt(1 + self.star.facular_area_ratio)
        else:
            rad_fac = np.zeros(num_spots)

        # Combine all parameters into result
        pos = np.zeros((num_spots, 4))
        pos[:, 0] = np.deg2rad(colat)
        pos[:, 1] = np.deg2rad(phsr)
        pos[:, 2] = np.deg2rad(rad)
        pos[:, 3] = rad_fac

        return pos

    def compute_planet_pos(self, t):

        if (self.star.planet_esinw == 0 and self.star.planet_ecosw == 0):
            ecc = 0
            omega = 0
        else:
            ecc = np.sqrt(self.star.planet_esinw ** 2 + self.star.planet_ecosw ** 2)
            omega = np.arctan2(self.star.planet_esinw, self.star.planet_ecosw)

        t_peri = nbspectra.Ttrans_2_Tperi(self.star.planet_transit_t0, self.star.planet_period, ecc, omega)
        sinf, cosf = nbspectra.true_anomaly(t, self.star.planet_period, ecc, t_peri)

        cosftrueomega = cosf * np.cos(omega + np.pi / 2) - sinf * np.sin(
            omega + np.pi / 2)  # cos(f+w)=cos(f)*cos(w)-sin(f)*sin(w)
        sinftrueomega = cosf * np.sin(omega + np.pi / 2) + sinf * np.cos(
            omega + np.pi / 2)  # sin(f+w)=cos(f)*sin(w)+sin(f)*cos(w)

        if cosftrueomega > 0.0: return np.array(
            [1 + self.star.planet_radius * 2, 0.0, self.star.planet_radius])  # avoid secondary transits

        cosi = (self.star.planet_impact_param / self.star.planet_semi_major_axis) * (1 + self.star.planet_esinw) / (
                    1 - ecc ** 2)  # cosine of planet inclination (i=90 is transit)

        rpl = self.star.planet_semi_major_axis * (1 - ecc ** 2) / (1 + ecc * cosf)
        xpl = rpl * (-np.cos(self.star.planet_spin_orbit_angle) * sinftrueomega - np.sin(
            self.star.planet_spin_orbit_angle) * cosftrueomega * cosi)
        ypl = rpl * (np.sin(self.star.planet_spin_orbit_angle) * sinftrueomega - np.cos(
            self.star.planet_spin_orbit_angle) * cosftrueomega * cosi)

        rhopl = np.sqrt(ypl ** 2 + xpl ** 2)
        thpl = np.arctan2(ypl, xpl)

        pos = np.array(
            [float(rhopl), float(thpl), self.star.planet_radius])  # rho, theta, and radii (in Rstar) of the planet
        return pos

    def plot_spot_map_grid(self, vec_grid, typ, inc, time):
        filename = self.star.path / 'plots' / 'map_t_{:.4f}.png'.format(time)

        x = np.linspace(-0.999, 0.999, 1000)
        h = np.sqrt((1 - x ** 2) / (np.tan(inc) ** 2 + 1))
        color_dict = {0: 'red', 1: 'black', 2: 'yellow', 3: 'blue'}
        plt.figure(figsize=(4, 4))
        plt.title('t={:.3f}'.format(time))
        plt.scatter(vec_grid[:, 1], vec_grid[:, 2], color=[color_dict[np.argmax(i)] for i in typ], s=2)
        plt.plot(x, h, 'k')
        plt.savefig(filename, dpi=100)
        plt.show()