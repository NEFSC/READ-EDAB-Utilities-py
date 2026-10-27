import os
import sys
import re
import xarray as xr
import numpy as np
import h5netcdf
import warnings
import traceback
import concurrent.futures
from datetime import datetime
import time
import platform
import gc
import dask

# Local utilities (Adapted from stats_utilities)
from utilities.bootstrap.environment import bootstrap_environment
from utilities import get_default_climatology_range, extract_period_code, get_clim_period 
from utilities import get_prod_files, get_filepath, file_parser,get_default_product_dataset, get_default_dataset_product, corrupt_file_detector
from utilities import generate_logfile_path
from utilities import build_stat_attributes, build_product_attributes, get_default_metadata, get_lut_metadata, get_current_utc_timestamp
from utilities import get_geospatial_metadata, get_temporal_metadata, get_summary_metadata, get_reference_metadata, get_fill_value, get_source_metadata
env = bootstrap_environment(verbose=False)

xr.set_options(file_cache_maxsize=1) 

"""
ANOMALY_UTILITIES 

Purpose:
    This module creates gridded anomaly outputs by calculating the difference or ratio
    between statistical data and corresponding climatological baselines.

Main Functions:
    - run_anomaly_pipeline: Runs a multi-product, multi-period anomaly pipeline.
    - process_single_anomaly: Processes a single anomaly task (opens stat/clim files, computes anomaly, saves to NetCDF).
    - build_anomaly_map: Resolves output directories, pairs stat and climatology files, and builds a processing map.

Helper Functions:
    - compute_anomalies: Subtracts climatology from statistics and standardizes variable names.
    - boost_file_limits: Raises the system limit for open file handles (Linux/Mac only).
    - DualLogger: Intercepts standard print statements and writes them to both the terminal and a persistent log file.

References:
    
Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on October 13, 2026 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Oct 13, 2026 - KJWH: Initial code written usingn the stats_utilities structural framework
    Oct 16, 2026 - KJWH: RAM-Optimized Group Processing: Replaced process_single_anomaly with process_anomaly_group. Instead of opening climatology files for every single statistical task, the pipeline groups tasks by their target baseline, loads the climatology file into memory exactly once, and loops through all corresponding statistical files efficiently.
                         Log-Normal Ratio Anomalies: Added is_log_anomaly() to detect log-normally distributed products (like Chlorophyll) and compute their anomalies as a ratio (stat / clim) instead of a standard difference (stat - clim). Injected metadata attributes (units, comment, anomaly_method) to explicitly document this.
                         Targeted Statistics: Added the ability to manually target specific statistical variables for the anomaly calculation via stat_var (e.g., median or max), dynamically defaulting to median for log-products and mean for everything else.
                         Dynamic Baseline Matching: Integrated batch_anomaly_match to dynamically pair statistical files with clamped baseline files. Handled running means (like D3 or D8) by calculating the exact center-date of the time series and targeting the corresponding DOY climatology file.
                         Fixed Coordinate Alignment Crash: Fixed a Time coordinate is empty crash by explicitly squeezing and dropping the time dimension from the climatology file upon loading, preventing xarray from failing to align mismatched historical dates.
Missing Baseline Graceful Fails: Ensured that missing climatology files safely return "NA" rather than crashing the pipeline, allowing them to be specifically logged in the final summary.
    Oct 20, 2026 - KJWH: Removed xr.open_dataset() in build_stats_map() and using the native h5netcdf to read the file attributes
    Oct 21, 2026 - KJWH: Forced h5netcdf engine in all .to_netcdf() calls
                         Added a try/except block to the map builders that automatically detects unreadable/corrupted NetCDF files and deletes them
                         Also fixed the accidentally dropped var (variance) key
                         Moved the default product, dataset, and period routing logic to the very top of run_stats_pipeline
                         Implemented a new tracking system that distinguishes between files that are Created, Replaced (overwritten due to new inputs), and Skipped (up-to-date), along with tallies for deleted corrupt files and missing baselines.
                         Intercepted the logfile parameter to pass through generate_logfile_path(), allowing standardized log naming across all execution environments.
                         Updated documentation
    Oct 27, 2026 - KJWH: Fixed a ValueError in run_anomaly_pipeline by updating the early exits in build_anomaly_map to return three empty objects ({}, [], []) to match the expected unpacking signature.
    
"""

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def boost_file_limits():
    """
    Purpose:
        Raises the system limit for open file handles (Linux/Mac only).
    """
    if platform.system() != 'Windows':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_limit = min(4096, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))

def is_log_anomaly(var_name):
    """
    Purpose:
        Identifies whether a product is log-normally distributed and requires 
        a ratio-based anomaly calculation instead of a standard difference.

    Input Parameters:
        prod (str): The logical product name (e.g., 'CHL', 'SST').

    Returns:
        bool: True if the product requires log-anomaly (ratio) processing, False otherwise.
    """
    v_upper = str(var_name).upper()
    
    # 1. Explicitly exclude fractions (Standard Difference)
    if 'FMICRO' in v_upper or 'FNANO' in v_upper or 'FPICO' in v_upper:
        return False
        
    # 2. Define known log-normal indicators (Ratio)
    log_indicators = ['CHL', 'KD490', 'POC', 'PSC_MICRO', 'PSC_NANO', 'PSC_PICO', 'PP_']
    
    # Return True if any indicator is found in the variable name
    return any(ind in v_upper for ind in log_indicators)

def batch_anomaly_match(input_files, clim_files, climatology_range=None):
    """
    Purpose:
        Takes a list of input statistical files and matches them to their corresponding 
        climatology baseline files. Applies baseline clamping logic to enforce specific year ranges.

    Input Parameters:
        input_files (list): List of absolute paths to the statistical input NetCDF files.
        clim_files (list): List of absolute paths to the climatology baseline NetCDF files.
        climatology_range (tuple, optional): A two-element tuple (start_year, end_year) specifying the 
                                             climatology baseline. If None, it defaults to the dataset's standard.

    Helper Functions:
        - get_default_climatology_range: Fetches default dataset-specific baseline years.
        - file_parser: Extracts date, dataset, and grouping metadata from filenames.

    Returns:
        dict: A dictionary mapping each input file to its matched climatology file path. 
              Returns "NA" for the value if no match is found.
    """
   
    matched_pairs = {}
    
    for input_file in input_files:
        parsed_data = file_parser(input_file)[0]
        dataset_name = parsed_data.get('dataset', 'UNKNOWN').upper()
        
        # Get the dataset-specific baseline if no override is provided
        current_clim_range = climatology_range if climatology_range else get_default_climatology_range(dataset=dataset_name)
        y_clim_start, y_clim_end = current_clim_range
        
        expected_suffix = f"_{y_clim_start:04d}_{y_clim_end:04d}"
        
        metadata = parsed_data['period_details']['metadata']
        groupby = metadata.get('groupby')
        period_code = parsed_data['period_details']['period_code']
        
        # Extract dates to support center-date calculation
        start_date_str = parsed_data['start_date']
        end_date_str = parsed_data.get('end_date') or start_date_str
        
        if groupby == 'month':
            month = start_date_str[4:6] 
            target_period = f"MONTH_{month}{expected_suffix}"
        elif groupby == 'week':
            week = start_date_str[4:6]
            target_period = f"WEEK_{week}{expected_suffix}" 
        elif groupby in ['annual']:
            target_period = f"ANNUAL{expected_suffix}"
        elif groupby in ['year']:
            target_period = f"YEAR{expected_suffix}"
        elif period_code in ['D', 'D3', 'D7', 'D8', 'DD']:
            # Calculate the center date and extract the DOY for daily/running mean files
            start_dt = datetime.strptime(start_date_str, "%Y%m%d")
            end_dt = datetime.strptime(end_date_str, "%Y%m%d")
            center_dt = start_dt + (end_dt - start_dt) / 2
            doy = center_dt.timetuple().tm_yday
            
            target_period = f"DOY_{doy:03d}{expected_suffix}"
        else:
            print(f"Skipping {input_file}: Unsupported groupby '{groupby}'")
            continue

        match_found = False
        for clim_file in clim_files:
            if target_period in clim_file:
                matched_pairs[input_file] = clim_file
                match_found = True
                break  
                
        if not match_found:
            matched_pairs[input_file] = "NA"
            print(f"Warning: No climatology match (looking for '{target_period}') found for {os.path.basename(input_file)}")
            
    return matched_pairs

def build_anomaly_map(prod, period, output_dir=None,
                      dataset=None, daterange=None, climatology_range=None,
                      dataset_version=None, dataset_map=None,
                      map_subset=None, overwrite=False, 
                      data_type="ANOM", verbose=False, debug=False, **kwargs):
    """
    Purpose:
        Orchestrates the preparation phase of the anomaly pipeline by resolving output directories,
        pairing statistical files with their corresponding climatology files, and evaluating freshness (mtimes)
        to determine if processing is needed.

    Input Parameters:
        prod (str): The logical product name to process (e.g., 'SST', 'CHL').
        period (str): The time period code for the target anomalies (e.g., 'M', 'D8').
        output_dir (str, optional): The destination directory for the output files. Automatically resolved if None.
        dataset (str, optional): Override for the dataset name (e.g., 'OCCCI'). Defaults to the product's default.
        daterange (list or tuple, optional): Date boundaries for filtering input files.
        climatology_range (tuple, optional): Year boundaries (start, end) for the baseline.
        dataset_version (str, optional): Override for the dataset version.
        dataset_map (str, optional): Specific map folder structure to override default routing.
        map_subset (str, optional): The spatial region abbreviation to subset the data to (e.g., 'NES').
        overwrite (bool, optional): If True, flags all tasks for reprocessing regardless of existing file modification times.
        data_type (str, optional): The data type suffix appended to the output file names. Default is "ANOM".
        verbose (bool, optional): If True, prints progress updates and routing information.
        debug (bool, optional): If True, prints extensive troubleshooting information.

    Helper Functions:
        - batch_anomaly_match: Pairs stat files with their corresponding climatology files.
        - extract_period_code: Pulls exact full period codes from strings for naming.

    Returns:
        tuple:
            - dict: A dictionary mapping each target output period to its specific processing task definition.
            - list: A list of previously corrupted file paths that were successfully deleted.
            - list: A list of corrupted file paths that were locked by the OS and require manual deletion.
    """

    # Auto-resolve product-specific default map_subset
    if not map_subset:
        map_subset = 'NES' if 'PSC' in prod.upper() else 'NWA'
    
    # 1. Resolve Output Directory
    if output_dir is None:
        if verbose: print(f"🔍 Resolving output directory for {prod} ({period} Anomalies)...")
        output_dir = get_filepath(
            prod,
            dataset=dataset,
            dataset_version=dataset_version,
            dataset_map=dataset_map,
            map_subset=map_subset,         
            period=period,      
            data_type=data_type,
            getfilepath=True,
            make_dir=True,      
            verbose=debug
        )

    if not output_dir:
        raise ValueError("🛑 CRITICAL: output_dir resolved to None.")
    elif isinstance(output_dir, list):
        output_dir = output_dir[0]

    # 2. Find Input Files (Stats and Climatologies)
    if verbose: print(f"Searching for input {period} STATS files for {dataset}:{prod}")

    stat_files = get_prod_files(prod, dataset=dataset, dataset_version=dataset_version,
                                dataset_map=dataset_map, map_subset=map_subset,
                                period=period, data_type="STATS", verbose=debug)

    if verbose: print(f"Found {len(stat_files)} for input {period} STATS files for {dataset}:{prod}")
                     
    # Determine the correct climatology period string (e.g. 'M' maps to 'MONTH')
    clim_period = get_clim_period(period)
    
    if not clim_period:
        raise ValueError(f"⚠️ Cannot run anomalies on '{period}' because it has no mapped climatology baseline.")

    if verbose: print(f"Searching for {clim_period} CLIM files for {dataset}:{prod}")

    clim_files = get_prod_files(prod, dataset=dataset, dataset_version=dataset_version,
                                dataset_map=dataset_map, map_subset=map_subset,
                                period=clim_period, data_type="CLIM", verbose=debug)

    if verbose: print(f"Found {len(clim_files)} for climatology {period} CLIM files for {dataset}:{prod}")

    if not stat_files or not clim_files:
        if verbose: print(f"⚠️ Missing requisite STATS or CLIM files for {prod} ({period}). Returning empty map.")
        return {}, [], []

    fp = file_parser(stat_files[0])
    ds_name = fp[0].get('dataset', dataset or 'UNKNOWN') 

    # 3. Pair Stats with Climatologies based on time period mapping
    #period_sets = get_period_sets(period, files=stat_files, daterange=daterange)
    #pm = period_sets.get("period_map", {})
    file_matching = batch_anomaly_match(stat_files, clim_files,climatology_range=climatology_range)

    anomaly_map = {}
    total_tasks = len(file_matching)
    up_to_date_count = 0

    # Initialize corrupt file trackers
    deleted_corrupt_files = []
    manual_delete_files = []

    for stat_file, target_clim_file in file_matching.items():

        # Extract the specific output period from the stat_input filename (e.g., 'M_200005')
        out_period = extract_period_code(stat_file)['full_period']
        stat_input = [stat_file]  # Process single anomaly expects a list

        out_name = f"{out_period}-{map_subset}-{ds_name}-{prod}-{data_type}.nc"
        out_path = os.path.join(output_dir, out_name)

        is_up_to_date = False
        # Only check for up-to-date status if we actually have a climatology file
        if target_clim_file != "NA":
            all_inputs = stat_input + [target_clim_file]
            if os.path.exists(out_path) and not overwrite:
                out_mtime = os.path.getmtime(out_path)
                if all(out_mtime > os.path.getmtime(f) for f in all_inputs):
                    try:
                        check_corruption = False # 🎯 Initialize flag
                        
                        # Use native h5netcdf to read the input file attributes
                        with h5netcdf.File(out_path, 'r') as existing_ds:
                            stored_files_str = existing_ds.attrs.get('source_files', '')
                            
                            if isinstance(stored_files_str, bytes):
                                stored_files_str = stored_files_str.decode('utf-8')
                            elif isinstance(stored_files_str, np.ndarray):
                                stored_files_str = str(stored_files_str[0]) if stored_files_str.size > 0 else ""
                        
                        # 🎯 The 'with' block is over. existing_ds is safely CLOSED.
                                
                        if stored_files_str:
                            stored_basenames = set(f.strip() for f in str(stored_files_str).split(','))
                            current_basenames = set(os.path.basename(f) for f in all_inputs)
                            
                            if stored_basenames == current_basenames:
                                check_corruption = True
                            else:
                                is_up_to_date = False
                        else:
                            is_up_to_date = False
                            
                        # NOW it is safe to deep-scan for HDF corruption!
                        if check_corruption:
                            is_corrupt = corrupt_file_detector([out_path])
                            if is_corrupt:
                                raise RuntimeError(f"Corrupt data chunk detected: {is_corrupt[0][1]}")
                            is_up_to_date = True
                    except Exception:
                        if verbose: print(f"  ⚠️ Could not read {out_name}: {e}")
                        
                        #  Attempt to delete the corrupt/locked file
                        try:
                            os.remove(out_path)
                            deleted_corrupt_files.append(out_name) # 🎯 Track successful deletions
                            if verbose: print(f"  🗑️ Deleted corrupt file: {out_name}. Forcing recreation.")
                        except Exception as del_e:
                            manual_delete_files.append(out_name) # 🎯 Track locked files
                            if verbose: print(f"  🚨 Failed to delete corrupt file {out_name}: {del_e}. Manual removal may be required.")

                        is_up_to_date = False

        if is_up_to_date:
            up_to_date_count += 1
            
        anomaly_map[out_period] = {
            "stat_input": stat_input,
            "clim_input": target_clim_file,
            "output": out_path,
            "is_up_to_date": is_up_to_date,
            "file_exists": os.path.exists(out_path)
        }

    needs_update_count = len(anomaly_map) - up_to_date_count
    print("-" * 60)
    print(f"📊 ANOMALY SUMMARY: {prod} | {period} | {map_subset or 'NWA'}")
    print(f"✅ Up-to-date:  {up_to_date_count}/{len(anomaly_map)}")
    print(f"⏳ To process:  {needs_update_count} files")
    print("-" * 60)
    
    return anomaly_map, deleted_corrupt_files, manual_delete_files


def compute_anomaly(stat_da, clim_da, use_ratio):
    """
    Purpose:
        Calculates the anomaly between a statistical DataArray and a climatological baseline DataArray.
        Verifies that spatial dimensions match before calculating. Uses ratio for log-normal products 
        and simple difference for standard products.

    Input Parameters:
        stat_da (xr.DataArray): The statistical data array.
        clim_da (xr.DataArray): The climatological baseline data array.
        prod (str): The product name, used to determine the math method (ratio vs difference).

    Helper Functions:
        - is_log_anomaly: Determines if the product requires a ratio-based calculation.

    Returns:
        xr.DataArray: The computed anomaly array.
    """
    # Verify the core spatial dimensions match to prevent broadcasting errors
    if stat_da.sizes.get('lat') != clim_da.sizes.get('lat') or stat_da.sizes.get('lon') != clim_da.sizes.get('lon'):
        raise ValueError(
            f"Dimension mismatch! Stat (lat: {stat_da.sizes.get('lat')}, lon: {stat_da.sizes.get('lon')}) "
            f"does not match Clim (lat: {clim_da.sizes.get('lat')}, lon: {clim_da.sizes.get('lon')})."
        )
    
    if use_ratio:
        anom_da = stat_da / clim_da
    else:
        anom_da = stat_da - clim_da
        
    return anom_da


def process_anomaly_group(clim_file, task_list, prod, per, verbose, **kwargs):
    """
    Purpose:
        Processes a group of anomaly tasks that share a common climatology file.
        Opens and loads the climatology baseline into memory exactly once, 
        then iterates through all corresponding statistical files to calculate and save anomalies.

    Input Parameters:
        clim_file (str): The absolute path to the climatology NetCDF file.
        task_list (list): A list of tuples containing (out_period, task_dictionary) to be processed against the clim_file.
        prod (str): The product being processed (e.g., 'SST').
        per (str): The period code (e.g., 'M').
        verbose (bool): If True, prints progress updates.
        **kwargs: Additional arguments such as 'debug' and 'stat_var'.

    Helper Functions:
        - compute_anomaly: Performs the core mathematical anomaly calculation.
        - sanitize_for_netcdf: Cleans attributes and encodings before saving to disk.

    Returns:
        tuple: 
            - list: Periods successfully processed.
            - list: Periods that failed processing.
            - list: Corrupted files identified during processing.
    """
    boost_file_limits()
    
    def sanitize_for_netcdf(ds):
        for obj in [ds] + [ds[var] for var in ds.variables]:
            for k, val in list(obj.attrs.items()):
                if isinstance(val, dict):
                    obj.attrs[k] = val.get('default', str(val))
                elif val is None:
                    obj.attrs[k] = "None"
        return ds
        
    debug = kwargs.get('debug', False)
    successful_periods = []
    failed_periods = []
    corrupt_files_found = []
    
    ds_clim = None

    try:
        if verbose: print(f"  📖 Loading climatology baseline into RAM: {os.path.basename(clim_file)}")
        
        # Parse the climatology file to extract the baseline range
        clim_parsed = file_parser(clim_file)[0]
        clim_range = clim_parsed.get('period_details', {}).get('climatology_range')
        clim_range_str = f"{clim_range[0]}-{clim_range[1]}" if clim_range else "Unknown"
        
        # Open and fully load the climatology file into memory ONCE per group
        ds_clim = xr.open_dataset(clim_file, engine='h5netcdf')
        with dask.config.set(scheduler='single-threaded'):
            ds_clim = ds_clim.load()

        # Determine anomaly method and target statistic
        use_ratio = is_log_anomaly(prod)
        target_stat = kwargs.get('stat_var')
                
        if not target_stat:
            target_stat = 'median' if use_ratio else 'mean'        

        # Iterate through every statistical period assigned to this climatology
        for out_period, task in task_list:
            out_path = "Unknown"
            ds_stat, anom_ds = None, None
            
            try:
                # Safely unpack the task INSIDE the try block
                out_path = task.get('output', f"Unknown_Path_{out_period}")
                
                # Check for either key to be safe with older maps
                stat_files = task.get('stat_input', task.get('inputs', []))
                
                if not stat_files:
                    raise KeyError(f"No statistical input files found in map for {out_period}.")
                
                if debug: print(f"      ⚙️ Processing {out_period}...")
                
                # Load the statistical dataset
                ds_stat = xr.open_mfdataset(stat_files, combine="by_coords", engine='h5netcdf', parallel=False)
                anom_results = {}

                # Dynamically extract the base variables (e.g. 'PSC_micro' from 'PSC_micro_mean')
                stat_suffixes = ['_mean', '_median', '_max', '_min', '_std', '_sum', '_var', '_count']
                base_vars = set()
                for v in ds_stat.data_vars:
                    for suff in stat_suffixes:
                        if str(v).endswith(suff):
                            base_vars.add(str(v).replace(suff, ''))
                            break

                # Evaluate and process each base variable independently
                for base_var in base_vars:
                    use_ratio = is_log_anomaly(base_var)
                    
                    target_stat = kwargs.get('stat_var')
                    if not target_stat:
                        target_stat = 'median' if use_ratio else 'mean'
                        
                    target_var = f"{base_var}_{target_stat}"
                    
                    if target_var in ds_stat.data_vars and target_var in ds_clim.data_vars:
                        stat_da = ds_stat[target_var]
                        clim_da = ds_clim[target_var]

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            with dask.config.set(scheduler='single-threaded'):
                                # Pass the variable-specific math flag directly
                                anom_da = compute_anomaly(stat_da, clim_da, use_ratio)
                                anom_da = anom_da.load()
                        
                        out_var_name = f"{base_var}_anom"
                        anom_results[out_var_name] = anom_da
                    else:
                        if debug: print(f"      ⚠️ Climatology matching '{target_var}' not found. Skipping.")

                if not anom_results:
                    raise ValueError(f"❌ No valid variables computed for anomaly {out_path}.")
                
                anom_ds = xr.Dataset(anom_results, coords=ds_stat.coords)
                
                # Format Encoding and Attributes
                encoding = {}
                for out_var in anom_ds.data_vars:
                    var_data = anom_ds[out_var]
                    fill_value = get_fill_value(var_data)
                    var_masked = var_data.fillna(fill_value)

                    if np.issubdtype(var_masked.dtype, np.floating):
                        var_masked = var_masked.astype("float32")
                        encoding[out_var] = {"_FillValue": np.float32(fill_value), "dtype": "float32", "zlib": True, "complevel": 4}
                    else:
                        encoding[out_var] = {"_FillValue": fill_value, "zlib": True, "complevel": 4}

                    anom_ds[out_var] = var_masked
                    
                    base_var_name = out_var.replace("_anom", "")
                    var_attrs = build_stat_attributes(base_var_name, "anomaly", _FillValue=fill_value)

                    # Re-evaluate use_ratio for attributes injection
                    var_use_ratio = is_log_anomaly(base_var_name)
                    var_target_stat = kwargs.get('stat_var') or ('median' if var_use_ratio else 'mean')
                    
                    # Override units and add a comment if using the ratio method
                    if var_use_ratio:
                        var_attrs["units"] = "unitless"
                        var_attrs["comment"] = f"Anomaly calculated as a ratio (stat / clim) of the {var_target_stat}, resulting in a unitless value."
                    else:
                        var_attrs["comment"] = f"Anomaly calculated as a standard difference (stat - clim) of the {var_target_stat}."
                    # Remove _FillValue from attrs to prevent xarray conflict with encoding dictionary
                    var_attrs.pop("_FillValue", None)

                    anom_ds[out_var].attrs = var_attrs

                # Global Metadata
                attrs = get_default_metadata(sheet="General") | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
                attrs = attrs | get_geospatial_metadata(dataset=anom_ds) | get_temporal_metadata(ds=anom_ds)

                try:
                    # Force is_anomaly=True so base_prod_attrs fetches the new long_name_anomaly
                    base_prod_attrs = build_product_attributes(prod, is_anomaly=True)
                    # Directly use the long_name since it now inherently describes the anomaly
                    attrs["product_name"] = base_prod_attrs.get('long_name', f"{prod} Anomaly")
                except:
                    attrs["product_name"] = f"{prod} Anomaly"

                # Define the anomaly method used for global attributes
                anomaly_method = "ratio (stat / clim)" if is_log_anomaly(prod) else "difference (stat - clim)"
                attrs['anomaly_method'] = anomaly_method
                attrs['climatology_baseline'] = clim_range_str

                all_input_basenames = [os.path.basename(f) for f in stat_files] + [os.path.basename(clim_file)]
                attrs['source_files'] = ", ".join(all_input_basenames)
                attrs["history"] = f"{get_current_utc_timestamp()} Computed {prod} anomalies against climatological baselines ({clim_range_str}) using xarray via {anomaly_method} of the {target_stat}."                
                
                try:
                    ds_name = kwargs.get('dataset', 'UNKNOWN')
                    attrs = attrs | get_source_metadata(ds_name, source_prefix="source")
                except:
                    pass

                anom_ds.attrs = attrs
                anom_ds = sanitize_for_netcdf(anom_ds)
                
                # Save Output
                anom_ds.to_netcdf(out_path, format='NETCDF4', encoding=encoding, engine='h5netcdf')
                if verbose: print(f"    ✅ Saved: {os.path.basename(out_path)}")
                
                successful_periods.append(out_period)
                
            except Exception as e:
                print(f"❌ Error processing {out_path}: {e}")
                bad_files_info = corrupt_file_detector(stat_files)
                if bad_files_info:
                    corrupt_files_found.extend([f"{p} ({err})" for p, err in bad_files_info])
                if debug: traceback.print_exc()
                failed_periods.append(out_period)
                
            finally:
                for ds in [ds_stat, anom_ds]:
                    try:
                        if ds is not None: ds.close()
                    except: pass

        return successful_periods, failed_periods, corrupt_files_found
        
    except Exception as e:
        print(f"❌ FATAL ERROR loading climatology {os.path.basename(clim_file)}: {e}")
        # If the climatology file crashes, fail ALL tasks in this group
        if debug: traceback.print_exc()
        bad_clim = corrupt_file_detector([clim_file])
        if bad_clim:
            corrupt_files_found.extend([f"{p} ({err})" for p, err in bad_clim])
            
        failed_periods.extend([out_period for out_period, _ in task_list])
        return successful_periods, failed_periods, corrupt_files_found

    finally:
        try:
            if ds_clim is not None: ds_clim.close()
        except: pass
        gc.collect()


def run_anomaly_pipeline(prods, periods=None, **kwargs):
    """
    Purpose:
        Executes a multi-product, multi-period anomaly calculation pipeline. Resolves default datasets,
        orchestrates file mapping, groups tasks by climatology baselines to optimize memory utilization, and runs 
        the processing sequentially or in parallel.

    Input Parameters:
        prods (str or list of str, optional): Product name(s) to process. Infers default if None.
        periods (str or list of str, optional): Period code(s) to compute (e.g., 'M', 'A'). Defaults to ['M', 'A', 'SEASON'].
        **kwargs: Additional configuration parameters including:
            - dataset (str): Dataset name to use.
            - map_subset (str): Map region to process.
            - parallel_runs (int): Number of parallel workers to use.
            - stat_var (str): The specific statistic to use for the anomaly (e.g., 'mean', 'median').
            - logfile (str): Path to write logs, or "auto" to generate automatically.
            - verbose (bool): Enable standard logging outputs.
            - debug (bool): Enable detailed tracebacks and xarray warnings.

    Helper Functions:
        - build_anomaly_map: Creates the processing task list.
        - process_anomaly_group: Core processing function for grouped tasks.
        - generate_logfile_path: Creates standard logging path if requested.

    Returns:
        None
    """

    # Extract global dataset early to establish defaults
    dataset = kwargs.pop('dataset', None)
    
    # Fallback logic for products
    if not prods:
        if dataset:
            prods = [get_default_dataset_product(dataset.upper())]
        else:
            raise ValueError("Must provide either 'prods' or 'dataset' to infer defaults.")

    # Normalize prods input
    if isinstance(prods, str): prods = [prods.upper().strip()]

    # Fallback and normalization for periods
    if not periods: 
        periods = ['W', 'M', 'A']
    elif isinstance(periods, str):
        periods = [periods]
    
    # Handle logfile auto-generation
    logfile = kwargs.pop('logfile', None)
    if logfile in ["auto", True]:
        logfile = generate_logfile_path("anomaly", prods=prods, dataset=dataset, env_dict=env)    
    
    original_stdout = sys.stdout 

    if logfile:
        sys.stdout = DualLogger(logfile)
        print("\n" + "="*80)
        print(f"🟢 ANOMALY PIPELINE RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

    verbose = kwargs.pop('verbose', False)
    debug   = kwargs.get('debug', False) 
    should_log = verbose or bool(logfile)

    try:
        if isinstance(prods, str): prods = [prods.upper().strip()]
        if periods is None: periods = ['M', 'A']
        elif isinstance(periods, str): periods = [periods]

        verbose = kwargs.pop('verbose', False)
        debug   = kwargs.get('debug', False)
        parallel_runs = kwargs.pop('parallel_runs', 1) 

        if not debug:
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            np.seterr(all='ignore')
        else:
            warnings.filterwarnings('default', category=RuntimeWarning)
            np.seterr(all='warn')

        # 📋 Initialize trackers
        total_input_files = 0
        total_clim_files = set()
        skipped_tasks = []
        replaced_tasks = []
        created_tasks = []
        failed_tasks = []
        missing_climatologies = [] 
        all_corrupt_inputs = set()
        all_deleted_corrupt = [] 
        all_manual_delete = []   
                
        for prod in prods:
            current_dataset = dataset
            if not current_dataset:
                try:
                    current_dataset = get_default_product_dataset(prod)
                except KeyError:
                    if should_log: print(f"\n⚠️ No default dataset found for product '{prod}'. Skipping.")
                    continue
            
            check_corrupt = kwargs.get('check_corrupt', True)
            max_retries = kwargs.get('max_retries', 3)
            
            for period in periods:
                for attempt in range(1, max_retries + 1):
                    if should_log:
                        prefix = f"\n🚀 Starting anomaly pipeline" if attempt == 1 else f"\n🔄 Retry Attempt {attempt}/{max_retries}"
                        print(f"{prefix} for {prod} | Period: {period}")
                
                    anomaly_map, deleted_files, manual_delete_files = build_anomaly_map(prod, period, verbose=verbose, **kwargs)
                
                    # Aggregate the corrupt file lists
                    all_deleted_corrupt.extend(deleted_files)
                    all_manual_delete.extend(manual_delete_files)

                    if not anomaly_map: 
                        break # No tasks, break retry loop and go to next period
                        
                    # Group tasks by their climatology file
                    tasks_by_clim = {}
                
                    for out_period, task in anomaly_map.items():
                        if task['is_up_to_date']:
                            skipped_tasks.append(out_period)
                        elif task['clim_input'] == "NA":
                            missing_climatologies.append(out_period)
                            failed_tasks.append(f"{out_period} (Missing Climatology Baseline)")
                            if should_log: print(f"  ⚠️ Skipping {out_period}: No matching climatology file found.")
                        else:
                            clim_file = task['clim_input']
                            if clim_file not in tasks_by_clim:
                                tasks_by_clim[clim_file] = []
                            tasks_by_clim[clim_file].append((out_period, task))

                    if not tasks_by_clim: 
                        break

                    if parallel_runs == 1:
                        # Sequential execution by Climatology Group
                        for clim_file, task_list in tasks_by_clim.items():
                            successes, fails, corrupt_files = process_anomaly_group(clim_file, task_list, prod, period, verbose, **kwargs)
                            
                            if corrupt_files: all_corrupt_inputs.update(corrupt_files)
                            
                            # Route the successes into the correct trackers
                            for p in successes:
                                if anomaly_map[p]['file_exists']: 
                                    replaced_tasks.append(p)
                                else: 
                                    created_tasks.append(p)
                            for f in fails:
                                failed_tasks.append(f"{f} (Processing failed)")
                    else:
                        # Parallel execution by Climatology Group
                        if should_log: print(f"\n  ⚡ Launching {len(tasks_by_clim)} climatology groups across {parallel_runs} parallel workers...")
                        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_runs) as executor:
                            
                            future_to_clim = {
                                executor.submit(process_anomaly_group, clim_file, task_list, prod, period, verbose, **kwargs): clim_file
                                for clim_file, task_list in tasks_by_clim.items()
                            }
                            
                            for future in concurrent.futures.as_completed(future_to_clim):
                                try:
                                    successes, fails, corrupt_files = future.result()
                                    
                                    if corrupt_files: all_corrupt_inputs.update(corrupt_files)                                
                                    
                                    # Route the successes into the correct trackers
                                    for p in successes:
                                        if anomaly_map[p]['file_exists']: 
                                            replaced_tasks.append(p)
                                        else: 
                                            created_tasks.append(p)
                                    for f in fails:
                                        failed_tasks.append(f"{f} (Processing failed)")
                                except Exception as e:
                                    print(f"\n❌ [Parallel] Error processing group:\n  ⚠️  {e}")

                    # --- CORRUPT FILE CHECK & RETRY LOGIC ---
                    corrupt_this_run = False
                    if check_corrupt:
                        if should_log: print(f"  🔍 Verifying generated ANOMALY files for corruption...")
                        
                        # Collect all expected output files from this specific run
                        out_files = [t['output'] for t in anomaly_map.values() if t['output'] != "Unknown" and os.path.exists(t['output'])]
                        
                        if out_files:
                            bad_anoms = corrupt_file_detector(out_files)
                            if bad_anoms:
                                corrupt_this_run = True
                                if should_log: print(f"    🚨 Found {len(bad_anoms)} corrupt ANOMALY files.")
                                for bad_file, err in bad_anoms:
                                    if should_log: print(f"      🗑️ Deleting: {os.path.basename(bad_file)}")
                                    try:
                                        os.remove(bad_file)
                                        all_deleted_corrupt.append(os.path.basename(bad_file))
                                    except OSError:
                                        all_manual_delete.append(os.path.basename(bad_file))
                    
                    if not corrupt_this_run:
                        break # Break the retry loop! Everything is clean.
                    elif attempt == max_retries:
                        if should_log: print(f"  ❌ Reached max retries ({max_retries})")

        # --- FINAL SUMMARY REPORT ---
        print("\n" + "="*60)
        print("🏁 ANOMALY PIPELINE FINISHED SUMMARY")
        print("="*60)
        print(f"📥 Input files found:       {total_input_files}")
        print(f"🗺️  Climatology files found: {len(total_clim_files)}")
        print(f"⏭  Existing files skipped:  {len(skipped_tasks)}")
        print(f"🔄 Existing files replaced: {len(replaced_tasks)}")
        print(f"🆕 New files created:       {len(created_tasks)}")
        print(f"🗑️ Corrupt outputs deleted: {len(all_deleted_corrupt)}") 
        print(f"❌ Errors/Not created:      {len(failed_tasks)}")
        
        if failed_tasks:
            print(f"\n📂 FAILED TASKS:\n" + "\n".join([f"  • {task}" for task in failed_tasks]))
            
        # 🎯 Log out all files skipped due to missing baselines
        if missing_climatologies:
            print(f"\n⚠️ MISSING CLIMATOLOGIES ({len(missing_climatologies)}):")
            print("  The following periods could not be calculated because their baseline files are missing:")
            print("\n".join([f"  • {task}" for task in missing_climatologies]))
            
        if all_corrupt_inputs:
            print(f"\n🚨 CORRUPT FILES ({len(all_corrupt_inputs)}):")
            for err_str in sorted(all_corrupt_inputs):
                if " (" in err_str:
                    file_path, error_msg = err_str.rsplit(" (", 1)
                    print(f"  • {os.path.basename(file_path)} ({error_msg}")
                else:
                    print(f"  • {os.path.basename(err_str)}")
        
        # Output the manual deletion list
        if all_manual_delete:
            print(f"\n🚨 REQUIRES MANUAL DELETION ({len(all_manual_delete)}):")
            print("  The following files are corrupted but locked by the OS. Delete them manually:")
            for f in all_manual_delete:
                print(f"  • {f}")
        
        print("="*60)

    finally:
        if logfile:
            print(f"\n🔴 ANOMALY PIPELINE ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout.close()
            sys.stdout = original_stdout