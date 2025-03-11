import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
import os
import json
import logging
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from Nujum.star import Star
import kiauhoku as kh
from Nujum.imf import sample_kroupa_imf
from Nujum.stellar_interpolator import interpolate_stellar_parameters
import scipy.stats as stats

# Constants
R_SUN = 6.96 * 1e10
M_SUN = 1.989 * 1e33
L_SUN = 3.846 * 1e33
AGE_SUN = 4.6
SIGMA = 5.67 * 1e-5
G = 6.67 * 1e-8
M_R_THRESH = 1
MAX_T = 10200
MIN_T = 2500
DATASET_DIR = r'C:\Users\Ilay\projects\simulations\dataset'
MODELS_ROOT = r'C:\Users\Ilay\projects\simulations\starsim\starsim'

os.makedirs(DATASET_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{DATASET_DIR}/dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_simdata(root, Nlc, sim_name='dataset'):
    """Generate simulation data and save distributions."""
    logger.info(f"Generating simulation data for {Nlc} light curves")

    incl = np.arccos(np.random.uniform(0, 1, Nlc))
    clen = np.random.normal(loc=10, scale=2.8,
                            size=Nlc)  # gives roughly the fraction with cycle < 4 years to be the same as detected by Reinhold
    cover = 10 ** np.random.uniform(low=-1, high=np.log10(3), size=Nlc)
    period1 = 10 ** np.random.uniform(low=1, high=np.log10(50), size=int(0.8 * Nlc))
    period2 = 10 ** np.random.uniform(low=0, high=np.log10(20), size=Nlc - int(0.8 * Nlc))
    period = np.concatenate((period1, period2))
    np.random.shuffle(period)
    theta_low = np.random.uniform(low=0, high=40, size=Nlc)
    theta_high = np.random.uniform(low=theta_low, high=80, size=Nlc)
    tau_evol = np.random.uniform(low=1, high=20, size=Nlc)
    butterfly = np.random.choice([True, False], size=Nlc, p=[0.8, 0.2])
    diffrot_shear = np.random.uniform(0, 1, size=Nlc)
    mass = sample_kroupa_imf(Nlc, m_min=0.3, m_max=2.0)
    feh = np.random.normal(loc=-0.03, scale=0.17, size=Nlc)  # distribution from Kepler stars
    alpha = np.random.uniform(0, 0.4, size=Nlc)
    ages = np.clip(np.random.normal(3, 1, size=Nlc), a_min=0.1, a_max=10)
    ar = (ages / AGE_SUN) ** (-0.5) / 2
    interp = interpolate_stellar_parameters(mass, feh, alpha, ages, grid_name='mist')
    convective_shift = np.random.normal(loc=1, scale=1, size=Nlc)
    teff = interp['Teff']
    logg = interp['logg']
    L = np.clip(interp['L'], a_min=None, a_max=100)
    R = interp['R']
    cdpp = 200 * L
    outlier_rate = np.random.uniform(0, 0.003, size=Nlc)
    flicker = np.random.uniform(0, 0.3, size=Nlc)
    np.random.shuffle(diffrot_shear)
    omega = 2 * np.pi / period  # rad / day

    # Stitch this all together and write the simulation properties to file
    sims = {}
    sims['mass'] = mass
    sims['age'] = ages
    sims['FeH'] = feh
    sims['alpha/H'] = alpha
    sims['Teff'] = teff
    sims['logg'] = logg
    sims['L'] = L
    sims['R'] = R
    sims["Activity Rate"] = ar
    sims["Cycle Length"] = clen
    sims["Cycle Overlap"] = cover
    sims["Inclination"] = incl
    sims["Spot Min"] = theta_low
    sims["Spot Max"] = theta_high
    sims["Period"] = period
    sims["Omega"] = omega
    sims["Shear"] = diffrot_shear
    sims["Decay Time"] = tau_evol
    sims["Butterfly"] = butterfly
    sims['convective_shift'] = convective_shift
    sims['CDPP'] = cdpp
    sims['Outlier Rate'] = outlier_rate
    sims['Flicker Time Scale'] = flicker

    sims = pd.DataFrame.from_dict(sims)
    sims.to_csv(os.path.join(root, "simulation_properties.csv"), float_format="%5.4f", index_label="Simulation Number")

    # Plot distributions
    plot_distributions(sims, root, sim_name)

    # Plot pair plot for key parameters
    plot_pairplot(sims, Nlc, root, sim_name)

    logger.info(f"simulation_properties.csv was saved in {root}")

    return sims


def plot_distributions(sims, root, sim_name):
    """Plot distributions of all parameters."""
    ncols = 4
    nrows = int(np.ceil(sims.shape[1] / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 18))
    axes = axes.flatten()

    # Plot each distribution
    for ax, (label, values) in zip(axes, sims.items()):
        sns.histplot(values, kde=False, ax=ax, color="khaki", bins=40)
        ax.set_title(label, fontsize=20)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    # Hide any unused subplots
    for ax in axes[len(sims.columns):]:
        ax.axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"{root}/{sim_name}_distributions.png", dpi=150)
    plt.close()


def plot_pairplot(sims, Nlc, root, sim_name):
    """Create and save pairplot of key parameters."""
    sims_reduce = sims[['mass', 'age', 'FeH', 'Teff', 'logg', 'L', 'R', 'alpha/H', 'Activity Rate']]
    sns.set(style="ticks", font_scale=3)
    pairplot = sns.pairplot(
        sims_reduce.sample(min(Nlc, 1000)),  # Limit sample size for faster plotting
        diag_kind="kde",  # KDE for diagonal plots
        plot_kws={"alpha": 0.5, "s": 15},  # Style for scatter plots
        diag_kws={"fill": True},  # Style for diagonal KDE plots
        corner=True,  # Show only the lower triangle
    )

    # Adjust layout
    pairplot.fig.suptitle("Pair Plot of Simulated Data", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{root}/{sim_name}_pairplot.png", dpi=150)
    plt.close()

def simulate_one(sim_row, sim_dir, idx, freq_rate=1 / 48, ndays=1000, wv_array=None, save=True):
    """
    Simulate a single star with spots and generate light curves and spectra.

    Parameters:
    -----------
    sim_row : pandas.Series
        Row from the simulation properties DataFrame
    sim_dir : str
        Directory to save outputs
    idx : int
        Index of the simulation
    freq_rate : float
        Sampling frequency in days
    ndays : int
        Number of days to simulate
    wv_array : numpy.ndarray
        Wavelength array for" spectra
    save : bool
        Whether to save outputs to files

    Returns:
    --------
    sm : StarSim object
        The simulated stellar model
    t_sampling : numpy.ndarray
        Time array for the light curve
    """
    if f"{idx}.pqt" in os.listdir(f"{sim_dir}/lc"):
        print(f"{idx}.pqt already exists in {sim_dir}/lc")
        return None, None
    try:
        # Create StarSim object and set parameters
        sm = Star(conf_file_path='starsim.conf')
        sm.set_stellar_parameters(sim_row)
        sm.models_root = MODELS_ROOT

        # Generate spot map
        sm.generate_spot_map(ndays=ndays)

        # Create time array for sampling
        t_sampling = np.linspace(0, ndays, int(ndays / freq_rate))

        # Compute forward model
        sm.compute_forward(t=t_sampling, wv_array=wv_array)

        # Extract results
        lc = sm.results['lc']
        spectra = sm.results['spectra']
        wavelength = sm.results['wvp']

        # Save config file with all parameters
        if save:
            config = {
                'stellar_params': {
                    'mass': float(sim_row['mass']),
                    'age': float(sim_row['age']),
                    'metallicity': float(sim_row['FeH']),
                    'alpha': float(sim_row['alpha/H']),
                    'Teff': float(sim_row['Teff']),
                    'logg': float(sim_row['logg']),
                    'luminosity': float(sim_row['L']),
                    'radius': float(sim_row['R']),
                    'inclination': float(sim_row['Inclination']),
                    'rotation_period': float(sim_row['Period']),
                    'differential_rotation': float(sim_row['Shear']),
                },
                'activity_params': {
                    'activity_rate': float(sim_row['Activity Rate']),
                    'cycle_length': float(sim_row['Cycle Length']),
                    'cycle_overlap': float(sim_row['Cycle Overlap']),
                    'spot_min_latitude': float(sim_row['Spot Min']),
                    'spot_max_latitude': float(sim_row['Spot Max']),
                    'spot_decay_time': float(sim_row['Decay Time']),
                    'butterfly_pattern': bool(sim_row['Butterfly']),
                },
                'simulation_params': {
                    'num_days': ndays,
                    'sampling_rate': freq_rate,
                    'num_points': len(t_sampling),
                    'evolution_law': 'gaussian',
                    'wavelength_range': [float(wv_array[0]), float(wv_array[-1])],
                }
            }

            with open(f"{sim_dir}/configs/{idx}.json", 'w') as f:
                json.dump(config, f, indent=4)

        # Plot every 100th sample
        if idx % 10 == 0:
            fig, axes = plt.subplots(2, 1, figsize=(16, 9))
            axes[0].plot(t_sampling, lc)
            axes[0].set_xlabel('Time [days]')
            axes[0].set_ylabel('Flux')
            axes[0].grid(True)
            axes[0].set_title('Light Curve')

            axes[1].plot(wv_array, spectra)
            axes[1].set_xlabel('Wavelength [Å]')
            axes[1].set_ylabel('Flux')
            axes[1].grid(True)
            axes[1].set_title('Spectra')

            fig.suptitle(f'period - {sm.rotation_period:.2f} days,'
                         f' inclination - {np.rad2deg(np.pi/2 - sm.inclination):.2f} deg')

            plt.tight_layout()
            plt.savefig(f'images/{idx}.png')
            plt.close()

        # Save data files
        if save:
            # Save light curve
            lightcurve = pd.DataFrame(np.c_[t_sampling, lc], columns=["time", "flux"])
            lightcurve.to_parquet(f"{sim_dir}/lc/{idx}.pqt")

            # Save LAMOST spectrum
            lamost_df = pd.DataFrame(np.c_[wv_array, spectra], columns=["wavelength", "flux"])
            lamost_df.to_parquet(f"{sim_dir}/lamost/{idx}.pqt")

            # Save spot map
            spots_map = pd.DataFrame(np.c_[sm.spot_map[:, :7]], columns=["init_time", "duration(days)",
                                                                  "colatitude", "longitude@ref_time",
                                                                  "coeff_rad_1", "coeff_rad_2", "coeff_rad_3"])
            spots_map.to_parquet(f"{sim_dir}/spots/{idx}.pqt")

        return sm, t_sampling

    except Exception as e:
        logger.error(f"Error in simulation {idx}: {str(e)}", exc_info=True)
        return None, None


def worker_function(args):
    """Worker function for multiprocessing."""
    row_idx, row, sim_dir, ndays, wv_array = args
    try:
        return simulate_one(row, sim_dir, idx=row_idx, ndays=ndays, wv_array=wv_array, save=True)
    except Exception as e:
        logger.error(f"Error in worker {row_idx}: {str(e)}", exc_info=True)
        return None, None


def main():
    """Main function to run the simulation."""
    # Create directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/lc", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/spots", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/lamost", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/configs", exist_ok=True)
    os.makedirs('images', exist_ok=True)

    # Check if simulation properties exist
    if os.path.exists(f"{DATASET_DIR}/simulation_properties.csv"):
        logger.info("Simulation properties already exist. Reading from file.")
        sims = pd.read_csv(f"{DATASET_DIR}/simulation_properties.csv")
    else:
        # Generate 5000 simulations
        sims = generate_simdata(DATASET_DIR, 5000)

    # Load wavelength array for LAMOST spectra
    wv_array = np.load('lamost_wv.npy')

    # Set the number of days to simulate
    ndays = 270

    # Start timing
    start_time = time.time()

    # Determine number of processes (use 75% of available cores)
    num_cpus = max(1, int(mp.cpu_count() * 0.4))
    # num_cpus = 1
    logger.info(f"Using {num_cpus} CPU cores for parallel processing")

    # Prepare arguments for multiprocessing
    args_list = [(i, row, DATASET_DIR, ndays, wv_array) for i, row in sims.iterrows()]

    # Run simulations in parallel
    with mp.Pool(processes=num_cpus) as pool:
        results = list(tqdm(pool.imap(worker_function, args_list), total=len(args_list), desc="Simulating"))

    # Calculate time taken
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Total simulation time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per simulation: {elapsed_time / len(sims):.2f} seconds")

    # Count successful simulations
    successful = sum(1 for r in results if r[0] is not None)
    logger.info(f"Successfully completed {successful} out of {len(sims)} simulations")


if __name__ == "__main__":
    main()