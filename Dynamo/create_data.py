import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
import os
import json
import logging
import multiprocessing as mp
from Dynamo.planet_params import generate_planet_parameters
from tqdm import tqdm
from Dynamo.star import Star
from Dynamo.imf import sample_kroupa_imf, sample_kepler_imf
from Dynamo.stellar_interpolator import interpolate_stellar_parameters
from Dynamo.phoenix import download_phoenix_models
import scipy.stats as stats
import argparse
from pathlib import Path

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


def generate_theta_with_linear_decay(ar, N):
    """
    Generate theta_high with linear decay of upper bound as ar increases.
    At ar=1, upper bound is 40 degrees.
    As ar increases, upper bound approaches 90 degrees.
    """
    # Linear decay function: starts at 40 when ar=1, approaches 90 as ar increases
    # Formula: upper_bound = 90 - (90-40)/ar = 90 - 50/ar
    upper_bound = 90 - (50 / ar)

    # Generate theta_low first (0 to 20 degrees)
    theta_low = np.random.uniform(low=0, high=20, size=N)

    # Generate theta_high with the calculated upper bound, ensuring it's always >= theta_low
    theta_high = np.random.uniform(low=theta_low, high=np.maximum(theta_low + 1, upper_bound), size=N)

    return theta_low, theta_high

def truncated_normal_dist(mean, std, lower_bound, upper_bound, size):
    a = (lower_bound - mean) / std
    b = (upper_bound - mean) / std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)


def age_activity_relation(age, saturation_age_gyr=0.1):
    """
    Calculate the age activity relation between star and saturation age
    relations taken from :
    "The coronal X-ray–age relation and its implications for the evaporation of exoplanets", Jackson et al.
    "An Improved Age-Activity Relationship for Cool Stars older than a Gigayear" Booth et al.
    :param age: age array in gyr
    :param saturation_age_gyr: stauration age (see Jackson et al.)
    :return: activity array
    """
    ar = (age / AGE_SUN) ** (-2.8)
    old_indices = np.where(age > 1)[0]
    transition_idx_up = np.argmin(age[old_indices] - 1) if len(old_indices) > 0 else 0
    transition_value_up = ar[old_indices][transition_idx_up] if len(old_indices) > 0 else 0
    
    young_indices = np.where(age <= 1)[0]
    if len(young_indices) > 0:
        ar[young_indices] = (age[young_indices] / AGE_SUN) ** (-1.2)
        transition_idx_low = np.argmin(1 - age[young_indices])
        transition_value_low = ar[young_indices][transition_idx_low]
        diff = transition_value_up - transition_value_low
        ar[young_indices] += diff
        
    saturation_idx = np.where(age < saturation_age_gyr)[0]
    closest_idx = np.argmin(np.abs(saturation_age_gyr - age))
    ar[saturation_idx] = ar[closest_idx]
    return ar



def calculate_cdpp(L, distance, Nlc):
    """
    Calculate CDPP based on Kepler magnitude.
    Parameters:
    -----------
    L : array
        Luminosity in solar units
    distance : array
        Distance in parsecs
    Nlc : int
        Number of light curves
        
    Returns:
    --------
    cdpp : float array
        Calculated CDPP in ppm
    distance : array
        Distance in parsecs
    """
    # 2. Calculate Apparent Magnitude
    # Absolute Bolometric Magnitude
    M_bol = 4.74 - 2.5 * np.log10(L)
    # Bolometric correction (Simplification: neglect BC for now, assume Kp ~ Bol)
    # Distance Modulus: m - M = 5 * log10(d / 10)
    m_Kp = M_bol + 5 * np.log10(distance / 10)

    # 3. Calculate CDPP (ppm) - Approximate Kepler noise model
    # Noise = Shot Noise + Read Noise + Jitter
    # Simplified power law roughly matching Kepler: ~30ppm at mag 12, ~100ppm at mag 14
    # log10(CDPP) ~ 0.2 * m + C
    # 30 = 10^(0.2*12 + C) -> 1.477 = 2.4 + C -> C = -0.92
    # Let's use a slightly more robust polynomial or just this power law for shot noise regime.
    # CDPP_12 = 30. ppm
    # CDPP = CDPP_12 * 10**(0.2 * (m_Kp - 12))
    # We add a noise floor of 10 ppm
    cdpp = 30.0 * 10**(0.2 * (m_Kp - 12.0)) + 10.0
    
    # Clip very noisy stars (e.g. > 2000 ppm)
    cdpp = np.clip(cdpp, 0, 5000)
    
    return cdpp


def generate_simdata(root, Nlc, logger, add_noise=False, sim_name='dataset'):
    """Generate simulation data and save distributions."""
    logger.info(f"Generating simulation data for {Nlc} light curves")

    incl = np.arccos(np.random.uniform(0, 1, Nlc))
    clen = np.random.normal(loc=10, scale=2.8,
                            size=Nlc)  # gives roughly the fraction with cycle < 4 years to be the same as detected by Reinhold
    cover = 10 ** np.random.uniform(low=-1, high=np.log10(3), size=Nlc)
    # tau_evol = np.random.uniform(low=1, high=20, size=Nlc)
    tau_evol = np.random.normal(loc=6, scale=2, size=Nlc) # spots lifetime in units of period. The choise of this distribution is arbitrary
    mask = tau_evol < 2
    tau_evol[mask] = np.random.uniform(low=2, high=4, size=np.count_nonzero(mask))
    butterfly = np.random.choice([True, False], size=Nlc, p=[0.8, 0.2])
    diffrot_shear = np.random.uniform(0, 0.4, size=Nlc)
    mass = sample_kepler_imf(Nlc, m_min=0.3, m_max=2.0)
    feh = np.random.normal(loc=-0.03, scale=0.17, size=Nlc)  # distribution from Kepler stars
    alpha = np.random.uniform(0, 0.4, size=Nlc)
    ages = truncated_normal_dist(2.5, 2, lower_bound=0.05, upper_bound=10, size=Nlc)
    ar = age_activity_relation(ages, saturation_age_gyr=0.2)
    # theta_low = np.random.uniform(low=0, high=20, size=Nlc)
    # theta_high = np.random.uniform(low=theta_low, high=40 * ar, size=Nlc)
    theta_low, theta_high = generate_theta_with_linear_decay(ar, Nlc)
    mask = theta_high > 90
    theta_high[mask] = np.random.uniform(40, 90, size=np.count_nonzero(mask))
    interp = interpolate_stellar_parameters(mass, feh, alpha, ages, grid_name='fastlaunch')
    convective_shift = np.random.normal(loc=1, scale=1, size=Nlc)
    teff = interp['Teff']
    logg = interp['logg']
    L = np.clip(interp['L'], a_min=None, a_max=1000)
    R = interp['R']
    if 'Prot' in interp.keys() and interp['Prot'] is not None:
        prot = interp['Prot']
        # replace very long periods with random values
        mask = prot > 100
        prot[mask] = np.random.uniform(35, 50, size=np.count_nonzero(mask))
    else:
        prot1 = 10 ** np.random.uniform(low=1, high=np.log10(50), size=int(0.8 * Nlc))
        prot2 = 10 ** np.random.uniform(low=0, high=np.log10(20), size=Nlc - int(0.8 * Nlc))
        prot = np.concatenate((prot1, prot2))
        np.random.shuffle(prot)
    # generate clean samples
    # 1. Assign distances (Log-Normal distribution matching from Kepler)
    # Mean ~1320 pc, Median ~1108 pc -> mu ~ 7.0, sigma ~ 0.6
    distance = np.random.lognormal(mean=7.0, sigma=0.6, size=Nlc)
    
    if add_noise:
        cdpp = calculate_cdpp(L, distance, Nlc)
    else:
        cdpp = 0
    # outlier_rate = np.random.uniform(0, 0.003, size=Nlc) # example of outlier rate
    outlier_rate = 0
    # flicker = np.random.uniform(0, 0.3, size=Nlc) # example of flicker
    flicker = 0
    np.random.shuffle(diffrot_shear)
    omega = 2 * np.pi / prot  # rad / day
    planet_params = generate_planet_parameters(mass, R, planet_prob=0.1)

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
    sims["Period"] = prot
    sims["Activity Rate"] = ar
    sims["Cycle Length"] = clen
    sims["Cycle Overlap"] = cover
    sims["Inclination"] = incl
    sims["Spot Min"] = theta_low
    sims["Spot Max"] = theta_high
    sims["Omega"] = omega
    sims["Shear"] = diffrot_shear
    sims["Decay Time"] = tau_evol
    sims["Butterfly"] = butterfly
    sims['convective_shift'] = convective_shift
    sims['Distance'] = distance
    sims['CDPP'] = cdpp
    sims['Outlier Rate'] = outlier_rate
    sims['Flicker Time Scale'] = flicker
    sims['simulate_planet'] = planet_params['simulate_planet']
    sims['planet_period'] = planet_params['period']
    sims['planet_radius'] = planet_params['radius']
    sims['planet_esinw'] = planet_params['esinw']
    sims['planet_ecosw'] = planet_params['ecosw']
    sims['planet_impact'] = planet_params['impact']
    sims['planet_spin_orbit_angle'] = planet_params['spin_orbit_angle']
    sims['planet_transit_t0'] = planet_params['transit_t0']
    sims['planet_semi_amplitude'] = planet_params['semi_amplitude']


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
    cols = ['mass', 'age', 'FeH', 'Teff', 'logg', 'L', 'R', 'Period', 'Spot Min', 'Spot Max', 'Activity Rate']
    sims_reduce = sims[cols]
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

def save_batch(batch_results, batch_idx, dataset_dir):
    """
    Save a batch of simulation results to chunked files.
    """
    if not batch_results:
        return

    lcs = []
    spots = []
    configs = {}
    spectra_storage = {}  # {spec_name: [list of dfs]}

    for res in batch_results:
        if res is None: continue
        idx = res['id']
        
        # LC
        # t_sampling, lc
        df_lc = pd.DataFrame(np.c_[res['t'], res['lc']], columns=['time', 'flux'])
        df_lc['sim_id'] = idx
        lcs.append(df_lc)

        # Spots
        # spot_map columns: init_time, duration, colatitude, longitude, r1, r2, r3
        # map is array
        if res['spots'] is not None and len(res['spots']) > 0:
            df_spots = pd.DataFrame(res['spots'][:, :7], columns=["init_time", "duration(days)",
                                                                  "colatitude", "longitude@ref_time",
                                                                  "coeff_rad_1", "coeff_rad_2", "coeff_rad_3"])
            df_spots['sim_id'] = idx
            spots.append(df_spots)

        # Config
        configs[idx] = res['config']

        # Spectra
        if res['spectra']:
            for name, (wv, flux) in res['spectra'].items():
                if name not in spectra_storage:
                    spectra_storage[name] = []
                
                # We can optimize space by not repeating wavelength if it's identical, 
                # but for simplicity and safety (as it might vary with vsini/doppler) we save it.
                # However, typically wavelength grid is constant for a given instrument.
                # Use float32 to save space?
                df_spec = pd.DataFrame({'wavelength': wv, 'flux': flux})
                df_spec['sim_id'] = idx
                spectra_storage[name].append(df_spec)

    # Save LCs
    if lcs:
        pd.concat(lcs).to_parquet(f"{dataset_dir}/lc/chunk_{batch_idx}.pqt")

    # Save Spots
    if spots:
        pd.concat(spots).to_parquet(f"{dataset_dir}/spots/chunk_{batch_idx}.pqt")

    # Save Configs
    with open(f"{dataset_dir}/configs/chunk_{batch_idx}.json", 'w') as f:
        json.dump(configs, f, indent=4)

    # Save Spectra
    for name, dfs in spectra_storage.items():
        if dfs:
            out_dir = f"{dataset_dir}/{name}"
            os.makedirs(out_dir, exist_ok=True)
            pd.concat(dfs).to_parquet(f"{out_dir}/chunk_{batch_idx}.pqt")
    
    # Clean up
    del lcs, spots, configs, spectra_storage


def simulate_one(models_root,
                 sim_row,
                 sim_dir,
                 idx,
                 logger,
                 freq_rate=1 / 48,
                 ndays=1000,
                 save=False, # Deprecated local save
                 plot_dir='images',
                 plot_every=np.inf,):
    """
    Simulate a single star with spots and return data.
    """
    # Check if already processed (skip logic would need to check chunks, but for now we assume new run or overwrite)
    # Skipping the individual file check usage since we are batching.
    
    try:
        # Create StarSim object and set parameters
        sm = Star(conf_file_path='starsim.conf')
        sm.set_stellar_parameters(sim_row)
        sm.set_planet_parameters(sim_row)
        sm.models_root = models_root

        # Generate spot map
        sm.generate_spot_map(ndays=ndays)

        # Create time array for sampling
        t_sampling = np.linspace(0, ndays, int(ndays / freq_rate))

        # Compute forward model
        sm.compute_forward(t=t_sampling)

        # Extract results
        lc = sm.results['lc']
        spectra = sm.results['spectra']
        wavelength = sm.results['wvp']

        # Construct Config Dict
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
                'rotation_period': float(sim_row['Period']),
                'differential_rotation': float(sim_row['Shear']),
                'distance': float(sim_row['Distance']),
            },
            'planet_params': {k:v for k, v in sim_row.items() if 'planet' in k} ,

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
                'num_spots': len(sm.spot_map),
                'sampling_rate': freq_rate,
                'num_points': len(t_sampling),
                'evolution_law': 'gaussian',
                'wavelength_range': [float(wavelength[0]), float(wavelength[-1])] if hasattr(wavelength, '__len__') and len(wavelength)>0 else [],
            }
        }

        # Plot every Nth sample
        if idx % plot_every == 0:
            fig, axes = plt.subplots(2, 1, figsize=(22, 12))
            axes[0].plot(t_sampling, lc)
            axes[0].set_xlabel('Time [days]')
            axes[0].set_ylabel('Flux')
            axes[0].set_title('Light Curve')

            axes[1].set_title('Spectra')
            
            # Helper to plot spectra
            spectra_dict = sm.results['spectra']
            if isinstance(spectra_dict, dict) and len(spectra_dict) > 0:
                for name, (wv, flx) in spectra_dict.items():
                    # print("plotting spectra: ", name, ' wv min/max: ', wv.min(), wv.max(), ' flx min/max: ', flx.min(), flx.max())
                    axes[1].plot(wv, flx, label=name, alpha=0.7)
                axes[1].legend()
            else:
                 if wavelength is not None and spectra is not None:
                    axes[1].plot(wavelength, spectra)

            teff = float(sim_row['Teff'])
            per = float(sim_row['Period'])
            fig.suptitle(f'Sample {idx} | Teff: {teff:.0f} K | Period: {per:.2f} d')
            plt.tight_layout()
            plt.savefig(f'{plot_dir}/{idx}.png')
            plt.close()

        # Return data structure for batch saving
        return {
            'id': idx,
            't': t_sampling,
            'lc': lc,
            'spectra': spectra,
            'spots': sm.spot_map,
            'config': config
        }

    except Exception as e:
        logger.error(f"Error in simulation {idx}: {str(e)}", exc_info=True)
        return None


def worker_function(args):
    """Worker function for multiprocessing."""
    models_root, row_idx, row, sim_dir, logger, ndays, plot_dir, plot_every = args
    try:
        return simulate_one(models_root, row, sim_dir,
                            idx=row_idx, logger=logger, ndays=ndays,
                            save=False, # Saving is now handled in batch
                            plot_dir=plot_dir, plot_every=plot_every)
    except Exception as e:
        logger.error(f"Error in worker {row_idx}: {str(e)}", exc_info=True)
        return None


def main():
    """Main function to run the simulation with command-line arguments."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run stellar spot simulation.')
    parser.add_argument('--models_root', type=str, default=None,
                        help='Root directory containing stellar models (default: ~/.dynamo')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Directory to store simulation outputs (default: dataset)')
    parser.add_argument('--plot_dir', type=str, default='images',
                        help='Directory to save plots (default: plots)'),
    parser.add_argument('--plot_every', type=int, default=100,
                        help='Create plots every N simulations (default: 100)')
    parser.add_argument('--num_simulations', type=int, default=1000,
                        help='Number of simulations to generate if not already existing (default: 1000)')
    parser.add_argument('--ndays', type=int, default=270,
                        help='Number of days to simulate (default: 1000)')
    parser.add_argument('--n_cpu', type=float, default=1,
                        help='Number of CPU cores to use (default: 1)')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add luminosity-dependent noise to light curves')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number of simulations per saved file chunk (default: 100)')

    # Parse arguments
    args = parser.parse_args()


    # Create directories
    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(f"{args.dataset_dir}/lc", exist_ok=True)
    os.makedirs(f"{args.dataset_dir}/spots", exist_ok=True)
    os.makedirs(f"{args.dataset_dir}/configs", exist_ok=True)
    os.makedirs('images', exist_ok=True)
    
    # Create spectra directories based on config
    try:
        import configparser
        conf = configparser.ConfigParser()
        conf.read('starsim.conf')
        if conf.has_option('general', 'spectra_names'):
            specs = conf.get('general', 'spectra_names').split(',')
            for s in specs:
                os.makedirs(f"{args.dataset_dir}/{s.strip()}", exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not parse starsim.conf to create spectra directories: {e}")
    os.makedirs('images', exist_ok=True)

    print(f"running create_data.py with {args.num_simulations} simulations and {args.n_cpu} CPU cores")
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{args.dataset_dir}/dataset_generation.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Check if simulation properties exist
    if os.path.exists(f"{args.dataset_dir}/simulation_properties.csv"):
        logger.info("Simulation properties already exist. Reading from file.")
        sims = pd.read_csv(f"{args.dataset_dir}/simulation_properties.csv")
    else:
        # Generate simulations
        logger.info(f"Generating {args.num_simulations} simulations.")
        sims = generate_simdata(args.dataset_dir, args.num_simulations, logger, add_noise=args.add_noise)
    
    feh_range = (sims['FeH'].min(), sims['FeH'].max())
    logg_range = (sims['logg'].min(), sims['logg'].max())
    teff_range = (sims['Teff'].min(), sims['Teff'].max())
    print("="*30, "\nDownloading relevant Phoenix spectra files\n", "="*30)
    download_phoenix_models(
        base_dir=args.models_root,
        metallicity_range=feh_range,
        temperature_range=teff_range,
        logg_range=logg_range,
        max_workers=int(args.n_cpu),
        delay_between_requests=0.4
    )

    if args.models_root is None:
        base_dir = Path.home() / '.dynamo'
    else:
        base_dir = Path(args.models_root)

    # Start timing
    start_time = time.time()

    # Determine number of processes
    num_cpus = max(1, int(args.n_cpu))
    logger.info(f"Using {num_cpus} CPU cores for parallel processing")

    # Prepare arguments for multiprocessing
    args_list = [(base_dir, i, row, args.dataset_dir, logger, args.ndays,
                  args.plot_dir, args.plot_every,)
                 for i, row in sims.iterrows()]

    # Run simulations
    batch_buffer = []
    chunk_counter = 0
    successful_count = 0
    
    # Define result processing function
    def process_result(res):
        nonlocal chunk_counter, successful_count
        if res is not None:
            batch_buffer.append(res)
            successful_count += 1
            
            if len(batch_buffer) >= args.batch_size:
                logger.info(f"Saving batch {chunk_counter} with {len(batch_buffer)} samples...")
                save_batch(batch_buffer, chunk_counter, args.dataset_dir)
                batch_buffer.clear()
                chunk_counter += 1

    if num_cpus == 1:
        # Run sequentially
        logger.info("Running in serial mode (n_cpu=1) - Ctrl+C enabled")
        for arg in tqdm(args_list, desc="Simulating"):
            process_result(worker_function(arg))
    else:
        # Run in parallel
        with mp.Pool(processes=num_cpus) as pool:
            try:
                for res in tqdm(pool.imap(worker_function, args_list), total=len(args_list), desc="Simulating"):
                    process_result(res)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                sys.exit(1)

    # Save remaining items in buffer
    if batch_buffer:
        logger.info(f"Saving final batch {chunk_counter} with {len(batch_buffer)} samples...")
        save_batch(batch_buffer, chunk_counter, args.dataset_dir)
        batch_buffer.clear()

    # Calculate time taken
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info(f"Total simulation time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per simulation: {elapsed_time / len(sims):.2f} seconds")

    # Count successful simulations
    logger.info(f"Successfully completed {successful_count} out of {len(sims)} simulations")


if __name__ == "__main__":
    main()