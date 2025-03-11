import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def plot_spot_map(self, best_maps, tref=None):
    N_div = 100
    Ngrids, Ngrid_in_ring, centres, amu, rs, alphas, xs, ys, zs, are, pare = nbspectra.generate_grid_coordinates_nb(
        N_div)
    vec_grid = np.array([xs, ys, zs]).T  # coordinates in cartesian
    theta, phi = np.arccos(zs * np.cos(-self.inclination) - xs * np.sin(-self.inclination)), np.arctan2(ys,
                                                                                                        xs * np.cos(
                                                                                                            -self.inclination) + zs * np.sin(
                                                                                                            -self.inclination))  # coordinates in the star reference

    if tref is None:
        tref = best_maps[0][0][0]
    elif len(tref) == 1:
        tref = [tref]

    for t in tref:
        Surface = np.zeros(len(vec_grid[:, 0]))  # len Ngrids
        for k in range(len(best_maps)):

            self.spot_map[:, 0:7] = best_maps[k]
            spot_pos = spectra.compute_spot_position(self, t)  # return colat, longitude and raddii in radians

            vec_spot = np.zeros([len(self.spot_map), 3])
            xspot = np.cos(self.inclination) * np.sin(spot_pos[:, 0]) * np.cos(spot_pos[:, 1]) + np.sin(
                self.inclination) * np.cos(spot_pos[:, 0])
            yspot = np.sin(spot_pos[:, 0]) * np.sin(spot_pos[:, 1])
            zspot = np.cos(spot_pos[:, 0]) * np.cos(self.inclination) - np.sin(self.inclination) * np.sin(
                spot_pos[:, 0]) * np.cos(spot_pos[:, 1])
            vec_spot[:, :] = np.array([xspot, yspot, zspot]).T  # spot center in cartesian

            for s in range(len(best_maps[k])):
                if spot_pos[s, 2] == 0:
                    continue

                for i in range(len(vec_grid[:, 0])):
                    dist = m.acos(np.dot(vec_spot[s], vec_grid[i]))
                    if dist < spot_pos[s, 2]:
                        Surface[i] += 1

        cm = plt.cm.get_cmap('afmhot_r')
        # make figure
        fig = plt.figure(1, figsize=(6, 6))
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)),
                'k')  # circumference
        x = np.linspace(-0.999, 0.999, 1000)
        h = np.sqrt((1 - x ** 2) / (np.tan(self.inclination) ** 2 + 1))
        plt.plot(x, h, 'k--')
        spotmap = ax.scatter(vec_grid[:, 1], vec_grid[:, 2], marker='o', c=Surface / len(best_maps), s=5.0,
                             edgecolors='none', cmap=cm, vmax=(Surface.max() + 0.1) / len(best_maps),
                             vmin=-0.2 * (Surface.max() + 0.1) / len(best_maps))
        # cb = plt.colorbar(spotmap,ax=ax, fraction=0.035, pad=0.05, aspect=20)
        ofilename = self.path / 'plots' / 'inversion_spotmap_t_{:.4f}.png'.format(t)
        plt.savefig(ofilename, dpi=200)
        # plt.show()
        plt.close()


def plot_active_longitudes(self, best_maps, tini=None, tfin=None, N_obs=100):
    N_div = 500

    if tini is None:
        tini = best_maps[0][0][0]
    if tfin is None:
        tfin = best_maps[0][0][0] + 1.0

    tref = np.linspace(tini, tfin, N_obs)
    longs = np.linspace(0, 2 * np.pi, N_div)
    Surface = np.zeros([N_obs, N_div])
    for j in range(N_obs):
        for k in range(len(best_maps)):
            self.spot_map[:, 0:7] = best_maps[k]
            spot_pos = spectra.compute_spot_position(self, tref[j])  # return colat, longitude and raddii in radians

            # update longitude adding diff rotation
            for s in range(len(best_maps[k])):
                ph_s = (spot_pos[s, 1] - (
                        (tref[j] - self.reference_time) / self.rotation_period % 1 * 360) * np.pi / 180) % (
                               2 * np.pi)  # longitude
                r_s = spot_pos[s, 2]  # radius
                if r_s == 0.0:
                    continue

                for i in range(N_div):
                    dist = np.abs(longs[i] - ph_s)  # distance to spot centre

                    if dist < r_s:
                        Surface[j, i] += 1

    X, Y = np.meshgrid(longs * 180 / np.pi, tref)
    fig = plt.figure(1, figsize=(6, 6))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.98, bottom=0.1)
    ax = fig.add_subplot(111)
    cm = plt.cm.get_cmap('afmhot_r')
    spotmap = ax.contourf(X, Y, Surface / len(best_maps), 25, cmap=cm, vmax=(Surface.max() + 0.1) / len(best_maps),
                          vmin=-0.2 * (Surface.max() + 0.1) / len(best_maps))
    cb = plt.colorbar(spotmap, ax=ax)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Time [d]")

    ofilename = self.path / 'plots' / 'active_map.png'
    plt.savefig(ofilename, dpi=200)
    # plt.show()
    plt.close()


def plot_sphere(ax, radius=1.0, resolution=100):
    """Plot a sphere with the given radius."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='skyblue', alpha=0.3, edgecolor='none')


def lat_lon_to_cartesian(lat, lon, r=1.0):
    """Convert latitude and longitude (in radians) to cartesian coordinates."""
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def plot_objects_on_sphere(time_data, time_step=0, show_animation=False, save_animation=False):
    """
    Plot objects on a sphere for a given time step or create an animation.

    Parameters:
    -----------
    time_data : ndarray
        Array of shape (time_steps, num_obj, 3) where the last dimension contains
        [latitude, longitude, radius]
    time_step : int
        The time step to plot if not creating an animation
    show_animation : bool
        If True, create and show an animation
    save_animation : bool or str
        If True, save animation as 'sphere_animation.gif', or specify a filename
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere
    sphere_radius = 1.0
    plot_sphere(ax, radius=sphere_radius)

    # If creating a static plot
    if not show_animation:
        # Extract data for the given time step
        objects_data = time_data[time_step]

        # Plot each object
        for obj in objects_data:
            lat, lon, obj_radius = obj
            x, y, z = lat_lon_to_cartesian(lat, lon, sphere_radius)

            # Plot a marker with size proportional to the object radius
            # We scale by 100 for better visibility - adjust as needed
            marker_size = obj_radius * 100
            ax.scatter(x, y, z, s=marker_size, color='red', alpha=0.7)

        plt.title(f'Objects on Sphere at Time Step {time_step}')

    else:
        # For animation
        objects = []

        def init():
            return objects

        def animate(i):
            # Clear previous objects
            for obj in objects:
                obj.remove()
            objects.clear()

            # Extract data for the current time step
            current_data = time_data[i]

            # Plot each object
            for obj in current_data:
                lat, lon, obj_radius = obj
                x, y, z = lat_lon_to_cartesian(lat, lon, sphere_radius)

                # Plot a marker with size proportional to the object radius
                marker_size = obj_radius * 100
                scatter = ax.scatter(x, y, z, s=marker_size, color='red', alpha=0.7)
                objects.append(scatter)

            ax.set_title(f'Objects on Sphere at Time Step {i}')
            return objects

        ani = FuncAnimation(fig, animate, frames=len(time_data),
                            init_func=init, blit=True, interval=200)

        if save_animation:
            # Use GIF format instead of MP4 since it's more widely supported without additional dependencies
            filename = 'sphere_animation.gif' if save_animation is True else save_animation
            if not filename.endswith('.gif'):
                filename = filename.rsplit('.', 1)[0] + '.gif'

            print(f"\nSaving animation as {filename}")
            # Use PillowWriter which should be available by default
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=10)
            ani.save(filename, writer=writer)

    # Common settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    plt.tight_layout()
    plt.close()

    if show_animation:
        return ani


# Example usage with sample data
if __name__ == "__main__":
    # Create some sample data: (time_steps, num_obj, [lat, lon, radius])
    # This is just dummy data for demonstration
    np.random.seed(42)
    time_steps = 20
    num_objects = 5

    # Generate random movements for objects
    sample_data = np.zeros((time_steps, num_objects, 3))

    # Initialize starting positions
    lats = np.random.uniform(-np.pi / 2, np.pi / 2, num_objects)
    lons = np.random.uniform(0, 2 * np.pi, num_objects)
    radii = np.random.uniform(0.05, 0.15, num_objects)

    # Create trajectories
    for t in range(time_steps):
        for i in range(num_objects):
            # Make objects move in small increments
            if t > 0:
                lats[i] += np.random.uniform(-0.1, 0.1)
                lons[i] += np.random.uniform(-0.1, 0.1)
                # Keep latitude within bounds
                lats[i] = max(min(lats[i], np.pi / 2), -np.pi / 2)
                # Keep longitude within bounds
                lons[i] = lons[i] % (2 * np.pi)

            sample_data[t, i, 0] = lats[i]  # latitude
            sample_data[t, i, 1] = lons[i]  # longitude
            sample_data[t, i, 2] = radii[i]  # radius

    # Plot static view of time step 0
    # plot_objects_on_sphere(sample_data, time_step=0)

    # Create and display animation
    # plot_objects_on_sphere(sample_data, show_animation=True)

    # Save animation as GIF
    plot_objects_on_sphere(sample_data, show_animation=True, save_animation='sphere_objects.gif')