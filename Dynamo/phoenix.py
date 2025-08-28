import os
import requests
from pathlib import Path
from urllib.parse import urljoin
import time
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import concurrent.futures
import threading
from queue import Queue


def get_available_metallicities(base_url):
    """
    Get list of available metallicity directories from the Phoenix server.
    """
    try:
        response = requests.get(base_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all directory links
        available_metallicities = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('Z') and href.endswith('/'):
                # Extract metallicity value from directory name
                met_str = href.rstrip('/')
                try:
                    if met_str == 'Z-0.0':
                        available_metallicities.append(0.0)
                    elif met_str.startswith('Z+'):
                        available_metallicities.append(float(met_str[2:]))
                    elif met_str.startswith('Z-'):
                        available_metallicities.append(-float(met_str[2:]))
                except ValueError:
                    continue

        return sorted(available_metallicities)

    except Exception as e:
        print(f"Warning: Could not fetch available metallicities from server: {e}")
        # Fallback to known values
        return [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]


def download_phoenix_models(base_dir=None, metallicity_range=None, temperature_range=None,
                            logg_range=None, max_workers=4, max_retries=3,
                            delay_between_requests=0.5):
    """
    Download Phoenix SpecInt models from the online database using parallel downloads.

    Parameters:
    -----------
    base_dir : str, optional
        Base directory to download to. Defaults to ~/.dynamo/models/Phoenix_mu
    metallicity_range : tuple, optional
        (min, max) metallicity range. Default: (-4.0, 1.0)
    temperature_range : tuple, optional
        (min, max) temperature range in K. Default: (2300, 12000)
    logg_range : tuple, optional
        (min, max) log g range. Default: (-0.5, 6.0)
    max_workers : int
        Number of parallel download threads (default: 4)
    max_retries : int
        Maximum number of retry attempts per file
    delay_between_requests : float
        Delay in seconds between requests to be respectful to server
    """

    # Default parameters
    if base_dir is None:
        base_dir = Path.home() / '.dynamo' / 'models' / 'Phoenix_mu'
    else:
        base_dir = Path(base_dir)

    if metallicity_range is None:
        metallicity_range = (-4.0, 1.0)
    if temperature_range is None:
        temperature_range = (2300, 12000)
    if logg_range is None:
        logg_range = (-0.5, 6.0)

    # Create directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://phoenix.astro.physik.uni-goettingen.de/data/SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/"

    print(f"Downloading Phoenix models to: {base_dir}")
    print(f"Using {max_workers} parallel workers")
    print(f"Checking available metallicities from server...")

    # Get available metallicities from server
    available_metallicities = get_available_metallicities(base_url)
    print(f"Available metallicities on server: {available_metallicities}")

    # Filter based on requested range
    metallicities = []
    for met in available_metallicities:
        if metallicity_range[0] <= met <= metallicity_range[1]:
            metallicities.append(met)

    print(f"Downloading metallicity range: {metallicity_range} (values: {metallicities})")
    print(f"Temperature range: {temperature_range} K")
    print(f"Log g range: {logg_range}")

    # Collect all download tasks
    download_tasks = []

    for metallicity in metallicities:
        # Format metallicity for URL based on actual server structure
        if metallicity == 0.0:
            met_str = "Z-0.0"
        elif metallicity > 0:
            met_str = f"Z+{metallicity:.1f}"
        else:
            met_str = f"Z{metallicity:.1f}"

        met_url = urljoin(base_url, met_str + "/")
        print(f"\nGathering files for metallicity {met_str}...")

        try:
            # Get list of files for this metallicity
            file_list = get_file_list_from_url(met_url)
            if not file_list:
                print(f"No files found or unable to access {met_url}")
                continue

            # Filter files based on parameter ranges
            filtered_files = filter_files_by_parameters(file_list, temperature_range, logg_range)
            print(f"Found {len(filtered_files)} files matching criteria")

            # Add to download tasks
            for filename in filtered_files:
                file_url = urljoin(met_url, filename)
                local_path = base_dir / filename

                # Skip if already exists
                if not local_path.exists():
                    download_tasks.append((file_url, local_path, max_retries))

        except Exception as e:
            print(f"Error processing metallicity {met_str}: {e}")
            continue

    total_files = len(download_tasks)
    print(f"\nTotal files to download: {total_files}")

    if total_files == 0:
        print("No files to download (all already exist)")
        return

    # Parallel download with progress tracking
    download_stats = {
        'downloaded': 0,
        'failed': 0,
        'total': total_files
    }

    # Create progress bar
    progress_bar = tqdm(total=total_files, desc="Downloading files", unit="file")
    progress_lock = threading.Lock()

    # Rate limiting queue
    rate_limiter = Queue(maxsize=max_workers)
    for _ in range(max_workers):
        rate_limiter.put(True)

    def download_worker(task):
        file_url, local_path, retries = task

        # Rate limiting
        rate_limiter.get()
        try:
            success = download_file_with_retry(file_url, local_path, retries)

            with progress_lock:
                if success:
                    download_stats['downloaded'] += 1
                else:
                    download_stats['failed'] += 1
                progress_bar.update(1)

            # Delay to be respectful to server
            time.sleep(delay_between_requests)

        finally:
            rate_limiter.put(True)

        return success

    # Execute parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_worker, task) for task in download_tasks]

        # Wait for completion
        concurrent.futures.wait(futures)

    progress_bar.close()

    print(f"\nDownload summary:")
    print(f"  Downloaded: {download_stats['downloaded']} files")
    print(f"  Failed: {download_stats['failed']} files")
    print(f"  Already existed: {len(list(base_dir.glob('*.fits'))) - download_stats['downloaded']} files")
    print(f"  Total files in directory: {len(list(base_dir.glob('*.fits')))}")


def get_file_list_from_url(url):
    """
    Extract list of .fits files from a Phoenix directory URL.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all links to .fits files
        fits_files = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.fits') and 'SPECINT' in href:
                fits_files.append(href)

        return fits_files

    except Exception as e:
        print(f"Error accessing {url}: {e}")
        return []


def filter_files_by_parameters(file_list, temp_range, logg_range):
    """
    Filter Phoenix files based on temperature and log g ranges.

    Phoenix filenames have format: lte[TEMP]-[LOGG]-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits
    Example: lte03000-4.50-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits
    """
    filtered_files = []

    for filename in file_list:
        try:
            # Extract temperature and log g from filename
            match = re.match(r'lte(\d{5})-(\d+\.\d{2})([+-]?\d+\.\d+)\.PHOENIX.*\.fits', filename)
            if match:
                temp = int(match.group(1))
                logg = float(match.group(2))

                # Check if within specified ranges
                if (temp_range[0] <= temp <= temp_range[1] and
                        logg_range[0] <= logg <= logg_range[1]):
                    filtered_files.append(filename)

        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            continue

    return filtered_files


def download_file_with_retry(url, local_path, max_retries=3):
    """
    Download a single file with retry logic and progress bar.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            # Get file size for verification
            total_size = int(response.headers.get('content-length', 0))

            # Create temporary file first
            temp_path = local_path.with_suffix('.tmp')

            with open(temp_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=32768):  # 32KB chunks for large files
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Verify download completed successfully
            if total_size > 0 and downloaded != total_size:
                temp_path.unlink(missing_ok=True)
                raise ValueError(f"Incomplete download: {downloaded}/{total_size} bytes")

            # Verify file is a valid FITS file
            if not is_valid_fits_file(temp_path):
                temp_path.unlink(missing_ok=True)
                raise ValueError("Downloaded file is not a valid FITS file")

            # Move temp file to final location
            temp_path.rename(local_path)
            return True

        except Exception as e:
            temp_path = local_path.with_suffix('.tmp')
            temp_path.unlink(missing_ok=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    return False


def is_valid_fits_file(file_path):
    """
    Quick check if file is a valid FITS file by checking header.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(80)
            return header.startswith(b'SIMPLE  =')
    except:
        return False


def check_existing_models(base_dir=None):
    """
    Check what Phoenix models are already downloaded.
    """
    if base_dir is None:
        base_dir = Path.home() / '.dynamo' / 'models' / 'Phoenix_mu'
    else:
        base_dir = Path(base_dir)

    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return

    fits_files = list(base_dir.glob('*.fits'))
    print(f"Found {len(fits_files)} Phoenix model files in {base_dir}")

    if len(fits_files) > 0:
        # Extract parameters from existing files
        temperatures = []
        loggs = []

        for file_path in fits_files:
            match = re.match(r'lte(\d{5})-(\d+\.\d{2})-0\.0\.PHOENIX.*\.fits', file_path.name)
            if match:
                temperatures.append(int(match.group(1)))
                loggs.append(float(match.group(2)))

        if temperatures and loggs:
            print(f"Temperature range: {min(temperatures)} - {max(temperatures)} K")
            print(f"Log g range: {min(loggs):.2f} - {max(loggs):.2f}")
            print(f"Unique temperatures: {len(set(temperatures))}")
            print(f"Unique log g values: {len(set(loggs))}")


# Enhanced convenience functions for parallel downloading
def download_solar_metallicity_models(base_dir=None, max_workers=8):
    """Download only solar metallicity (Z=0.0) models with optimized settings."""
    download_phoenix_models(
        base_dir=base_dir,
        metallicity_range=(0.0, 0.0),
        max_workers=max_workers,
        delay_between_requests=0.3
    )


def download_main_sequence_models(base_dir=None, max_workers=6):
    """Download models suitable for main sequence stars with parallel optimization."""
    download_phoenix_models(
        base_dir=base_dir,
        metallicity_range=(-1.0, 0.5),
        temperature_range=(3000, 8000),
        logg_range=(4.0, 5.0),
        max_workers=max_workers,
        delay_between_requests=0.4
    )


def download_giant_star_models(base_dir=None, max_workers=6):
    """Download models suitable for giant stars."""
    download_phoenix_models(
        base_dir=base_dir,
        metallicity_range=(-2.0, 0.5),
        temperature_range=(3000, 6000),
        logg_range=(0.0, 3.5),
        max_workers=max_workers,
        delay_between_requests=0.4
    )


def estimate_download_size(metallicity_range=None, temperature_range=None, logg_range=None):
    """
    Estimate download size and number of files for given parameter ranges.
    """
    if metallicity_range is None:
        metallicity_range = (0.0, 0.0)
    if temperature_range is None:
        temperature_range = (2300, 12000)
    if logg_range is None:
        logg_range = (-0.5, 6.0)

    # Get available metallicities
    base_url = "https://phoenix.astro.physik.uni-goettingen.de/data/SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/"
    available_metallicities = get_available_metallicities(base_url)

    # Filter metallicities
    metallicities = [met for met in available_metallicities
                     if metallicity_range[0] <= met <= metallicity_range[1]]

    # Estimate temperature steps (every 100K from 2300-12000K)
    temp_steps = len(range(2300, 12001, 100))
    temp_in_range = max(1, int((temperature_range[1] - temperature_range[0]) / 100))

    # Estimate log g steps (every 0.5 from -0.5 to 6.0)
    logg_steps = len([g / 10 for g in range(-5, 61, 5)])
    logg_in_range = max(1, int((logg_range[1] - logg_range[0]) / 0.5))

    # Estimate files per metallicity (rough approximation)
    files_per_metallicity = temp_in_range * logg_in_range * 0.8  # 80% availability factor
    total_files = int(len(metallicities) * files_per_metallicity)

    # Estimate size (average 15MB per file)
    total_size_mb = total_files * 15
    total_size_gb = total_size_mb / 1024

    print(f"Download estimate:")
    print(f"  Metallicities: {len(metallicities)} ({metallicities})")
    print(f"  Temperature range: {temperature_range[0]}-{temperature_range[1]} K")
    print(f"  Log g range: {logg_range[0]}-{logg_range[1]}")
    print(f"  Estimated files: ~{total_files}")
    print(f"  Estimated size: ~{total_size_mb:.0f} MB ({total_size_gb:.1f} GB)")

    # Estimate download time
    download_time_minutes = total_files / 4 * 0.5  # 4 workers, ~0.5 min per file
    print(f"  Estimated download time: ~{download_time_minutes:.0f} minutes")

    return total_files, total_size_mb


def monitor_download_progress(base_dir=None, target_files=None):
    """
    Monitor download progress in real-time.
    """
    if base_dir is None:
        base_dir = Path.home() / '.dynamo' / 'models' / 'Phoenix_mu'
    else:
        base_dir = Path(base_dir)

    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return

    print("Monitoring download progress...")
    print("Press Ctrl+C to stop monitoring")

    try:
        last_count = 0
        start_time = time.time()

        while True:
            current_files = list(base_dir.glob('*.fits'))
            current_count = len(current_files)

            if current_count != last_count:
                elapsed = time.time() - start_time
                rate = current_count / elapsed * 60 if elapsed > 0 else 0

                print(f"\rFiles: {current_count}", end="")
                if target_files:
                    progress = current_count / target_files * 100
                    print(f"/{target_files} ({progress:.1f}%)", end="")
                print(f" | Rate: {rate:.1f} files/min", end="", flush=True)

                last_count = current_count

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. Current files: {current_count}")


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download Phoenix stellar atmosphere models')
    parser.add_argument('--estimate-only', action='store_true',
                        help='Only estimate download size without downloading')
    parser.add_argument('--solar-only', action='store_true',
                        help='Download only solar metallicity models')
    parser.add_argument('--main-sequence', action='store_true',
                        help='Download main sequence star models')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--base-dir', type=str,
                        help='Base directory for downloads')

    args = parser.parse_args()

    if args.estimate_only:
        if args.solar_only:
            estimate_download_size(metallicity_range=(0.0, 0.0))
        elif args.main_sequence:
            estimate_download_size(
                metallicity_range=(-1.0, 0.5),
                temperature_range=(3000, 8000),
                logg_range=(4.0, 5.0)
            )
        else:
            estimate_download_size()
    else:
        print("Starting Phoenix models download...")
        check_existing_models(args.base_dir)

        if args.solar_only:
            download_solar_metallicity_models(args.base_dir, args.workers)
        elif args.main_sequence:
            download_main_sequence_models(args.base_dir, args.workers)
        else:
            download_phoenix_models(base_dir=args.base_dir, max_workers=args.workers)