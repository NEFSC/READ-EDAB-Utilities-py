import os
import xarray as xr
import numpy as np
import warnings
from datetime import datetime
import time
from contextlib import contextmanager
from collections import defaultdict
import resource
import platform
import traceback
import gc
import dask
from utilities.bootstrap.environment import bootstrap_environment
from utilities import get_period_info, get_period_sets, get_prod_files
from utilities import product_defaults, get_nc_prod, subset_dataset, file_parser, corrupt_file_detector
env = bootstrap_environment(verbose=False)

xr.set_options(file_cache_maxsize=1) # Don't keep any unnecessary files open in the background
"""
dask.config.set({'distributed.worker.memory.target': 0.6, 
                 'distributed.worker.memory.spill': 0.7,
                 'distributed.worker.memory.pause': 0.8,
                 'distributed.worker.memory.terminate': 0.95})
"""

"""
STATANOM_UTILITIES creates statistic and anomaly outputs from gridded satellite or modeled data. 
Main Functions:
    - compute_stats: Description
    - function2: Description
    - function3: Description

Helper Functions:
    - helper_function1: Description

References:
    
Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on September 08, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Sep 08, 2025 - KJWH: Initial code written
    Sep 29, 2025 - KJWH: Started building stats_period_map
    
"""

def boost_file_limits():
    """Raises the system limit for open file handles (Linux/Mac only)."""
    if platform.system() != 'Windows':
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to set it to 4096, but don't exceed the 'hard' system limit
        new_limit = min(4096, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
        print(f"🚀 System: File handle limit boosted to {new_limit}")


def get_groupby_hint(period_code):
    """
    Returns the appropriate xarray groupby hint or rolling strategy for a given statistical period code.

    Parameters:
    ----------
    period_code (str): The code representing the statistical period (e.g. 'D', 'W', 'M3', 'SEASON').

    Returns:
    -------
    dict
        Dictionary with keys:
            - method: 'groupby', 'rolling', or 'custom'
            - key: xarray groupby key or rolling window size
            - notes: optional clarification
    """
    info = get_period_info(period_code)
    if not info:
        return {'method': 'unknown', 'key': None, 'notes': f'Unrecognized period code: {period_code}'}

    _, _, is_range, is_running, is_climatology, iso_duration, groupby_key, *_ = info
    
    if is_running:
        return {'method': 'rolling', 'key': groupby_key, 'notes': f'{period_code} uses rolling window'}
    elif is_range:
        return {'method': 'custom', 'key': None, 'notes': f'{period_code} is a range period'}
    elif is_climatology:
        return {'method': 'groupby', 'key': groupby_key, 'notes': f'{period_code} is climatological'}
    elif groupby_key:
        return {'method': 'groupby', 'key': groupby_key, 'notes': f'{period_code} uses standard groupby'}
    else:
        return {'method': 'custom', 'key': None, 'notes': f'{period_code} requires custom logic'}

def apply_grouping(da, period_code, time_dim='time'):
    """
    Applies the appropriate grouping or rolling logic to a DataArray
    based on the period_code.
    """
    hint = get_groupby_hint(period_code)
    method = hint['method']
    key = hint['key']

    if method == 'groupby':
        grouped = da.groupby(eval(f"da.{key}"))
    elif method == 'rolling':
        grouped = da.rolling({time_dim: key}, center=True)
    elif method == 'custom':
        raise NotImplementedError(f"Custom logic required for period '{period_code}'")
    else:
        raise ValueError(f"Unknown grouping method for period '{period_code}'")

    return grouped

       
def build_stats_map(prod, period, output_dir=None,
                    dataset=None, daterange=None,climatology_range=None,
                    dataset_version=None, dataset_map=None,
                    subset=None, is_running=True, overwrite=False, 
                    data_type="STATS",verbose=False):
    """
    Unified stats pipeline:
      1) Resolve or Create Output Directory
      2) Find input files
      3) Generate period map via get_period_sets()
      4) Build stats map with mtime comparison for 'freshness'
    """
    
    # ---------------------------------------------------------
    # 1. Resolve Output Directory (if not provided)
    # ---------------------------------------------------------
    if output_dir is None:
        if verbose:
            print(f"🔍 Resolving output directory for {prod} ({period})...")
        
        # Use get_prod_files to 'discover' the new path.
        # Set make_dir=True to create the directory if it does not exist.
        output_dir = get_prod_files(
            prod,
            dataset=dataset,
            dataset_version=dataset_version,
            dataset_map=dataset_map,
            map_region=subset,         # Use 'subset' to override region (GLOBAL -> NES)
            period=period,      # This triggers Step 7 logic for CLIMS/STATS
            data_type=data_type,
            getfilepath=True,
            make_dir=True,      # Create the folder if it doesn't exist
            verbose=verbose
        )
        print(f"🔍 DEBUG: Output directory for {prod} ({period}) is {output_dir}")
    # ---------------------------------------------------------
    # 2. Determine expected input period (D, M, etc.)
    # ---------------------------------------------------------
    per_info = get_period_info(period)
    input_per = per_info["input_period_code"]

    # Disable running means if requested
    if not is_running and per_info["is_running_mean"]:
        per_info["is_running_mean"] = False

    # ---------------------------------------------------------
    # 3. Find input files
    # ---------------------------------------------------------
    search_period = None if input_per == "D" else input_per
    search_type = None if input_per == "D" else "STATS"
    
    if verbose:
        print(f"Searching for {input_per} files for {dataset}:{prod}")

    
    files = get_prod_files(prod,
        dataset=dataset,
        dataset_version=dataset_version,
        dataset_map=dataset_map,
        map_region=subset,
        period=search_period,
        data_type=search_type,
        verbose=verbose
    )

    if not files:
        if verbose:
            print(f"No {input_per} files found for {dataset}:{prod}")
        return {}

    # ---------------------------------------------------------
    # 4. Build period map using get_period_sets()
    # ---------------------------------------------------------
    period_sets = get_period_sets(period, files=files, daterange=daterange, climatology_range=climatology_range)
    pm = period_sets["period_map"]

    # ---------------------------------------------------------
    # 4. Build stats map with mtime comparison
    # ---------------------------------------------------------
    stats_map = {}
    total_tasks = len(pm)
    up_to_date_count = 0

    for out_period, info in pm.items():
        input_files = info["input_files"]

        # Build output filename
        out_name = f"{out_period}-{prod}-{data_type}.nc"
        print(f"DEBUG: output_dir={output_dir}, out_name={out_name}")
        out_path = os.path.join(output_dir, out_name)

        # Determine freshness
        is_up_to_date = False
        if os.path.exists(out_path) and not overwrite:
            out_mtime = os.path.getmtime(out_path)
            is_up_to_date = all(out_mtime > os.path.getmtime(f) for f in input_files) 

        if is_up_to_date:
            up_to_date_count += 1
        
        if verbose:
            status = "✓ up-to-date" if is_up_to_date else "⏳ needs update"
            print(f"{out_period}: {status} ({len(input_files)} inputs)")

        stats_map[out_period] = {
            "inputs": input_files,
            "output": out_path,
            "is_up_to_date": is_up_to_date
        }

    # ---------------------------------------------------------
    # 6. Final Summary Report
    # ---------------------------------------------------------
    needs_update_count = total_tasks - up_to_date_count
    completion_pct = (up_to_date_count / total_tasks * 100) if total_tasks > 0 else 0

    print("-" * 60)
    print(f"📊 STATS SUMMARY: {prod} | {period} | {subset or 'GLOBAL'}")
    print(f"📁 Output Dir: {output_dir}")
    print(f"✅ Up-to-date:  {up_to_date_count}/{total_tasks} ({completion_pct:.1f}%)")
    print(f"⏳ To process:  {needs_update_count} files")
    print("-" * 60)
    
    return stats_map


def process_single_stat(task, prod, per, verbose, **kwargs):
        """
        Processes a single statistical task: opens files, resolves product keys,
        applies spatial subsetting, computes statistics, and saves to NetCDF.

        Args:
            task (dict): Dictionary containing 'inputs' (list of paths) and 'output' (path).
            prod (str): The logical product name (e.g., 'SST').
            per (str): The time period label (e.g., 'M', 'W').
            verbose (bool): If True, prints progress updates.
            **kwargs: Includes 'debug', 'subset', and 'dataset' fallbacks.
        """
        
        boost_file_limits()
    
        @contextmanager
        def timer(label, debug=False):
            start = time.perf_counter()
            yield
            if debug:  # <--- Check this line!
                elapsed = time.perf_counter() - start
                print(f"    ⏱️ {label}: {elapsed:.2f}s")

        # 1. Extract variables from the inputs
        debug = kwargs.get('debug', False)
        subset = kwargs.get('subset')
        out_path = task['output']
        output_period = per
        input_files = task['inputs']
        max_retries = 3    

        # 2. Initialize to None so the finally block doesn't crash
        ds = None
        stats_ds = None

        # --- 🔄 RETRY LOOP ---
        for attempt in range(max_retries):

            # 3. Open the dataset
            try:                
                with timer(f"xr.open_mfdataset (Attempt {attempt+1})", debug=debug): # 'with' ensures file handles are released automatically, even on crash.
                    ds = xr.open_mfdataset(
                        task['inputs'], 
                        combine="by_coords",
                        decode_timedelta=True,
                        chunks={'time': 1, 'lat': 'auto', 'lon': 'auto'}, 
                        parallel=False,  # 🚨 CRITICAL: Parallel opening often crashes HDF5 on Mac
                        engine='netcdf4',
                        coords='minimal', # Don't waste RAM on redundant coords
                        compat='override' # Assume all files have the same grid (Fastest)
                    )
                        #decode_coords=True,
                        #chunks={'time': -1}, 
                        #parallel=True  # parallel=True to speed up coordinate coordinate decoding
                    
                break # All files opened successfully
            
            except Exception as e:
                if attempt < max_retries - 1:
                    if verbose: print(f"  ⚠️  Hiccup on {per} (Attempt {attempt+1}). Retrying in 2s...")
                    time.sleep(2)
                    
                else:
                    # All retries failed - identify the specific bad file
                    print(f"❌ FATAL: Could not open all files for {output_period} after {max_retries} attempts.")
                    corrupt_files = corrupt_file_detector(input_files)
                    if corrupt_files:
                        raise RuntimeError(f"Corrupt files found: {corrupt_files}")
                    else:
                        raise RuntimeError(f"Handle Error (No corruption detected): {e}")
        

        # 4. Resolve metadata and prep data
        try:
            # 4A. Resolve dataset name for product mapping
            try: 
                input_fp = file_parser(input_files[0])
                ds_name = input_fp[0]['dataset']
            except Exception as e:
                if debug: print(f"DEBUG: Failed to parse dataset from files: {e}")
                ds_name = kwargs.get('dataset') # Fallback

            # 4B. Resolve Variable Name (e.g., 'SST' -> 'sea_surface_temperature')
            prod_key = prod if prod in ds.data_vars else get_nc_prod(ds_name, prod)                
            da = ds[prod_key]
            if debug: print(f"dataset product key = {prod_key}")
                
            # 4C. Spatial Subsetting & Integrity Check
            if subset:
                if verbose: print(f"Subsetting to region: {subset}")
                da = subset_dataset(da, subset)
                if da.size == 0:
                    # This happens if the subset coordinates don't overlap with the file at all
                    print(f"⚠️ ERROR: Subset '{subset}' resulted in 0 pixels.")
                    if debug:
                        # Provide helpful context for why it failed
                        lon_min, lon_max = ds.lon.min().values, ds.lon.max().values
                        lat_min, lat_max = ds.lat.min().values, ds.lat.max().values
                        print(f"    DEBUG: File Extent: Lon({lon_min:.2f} to {lon_max:.2f}), Lat({lat_min:.2f} to {lat_max:.2f})")
                    return # Skip this task          
                if debug: print(f"    DEBUG: Subset successful. New Shape: {da.shape}")
            
            # 4D. Final Safety Check: Ensure there is data to process
            if da.size == 0:
                print(f"⚠️ Warning: DataArray is empty for {prod} after subsetting. Skipping.")
                return # Exit this task early
            
            # 5. Compute the statistics
            with timer("compute_stats (Graph Building)", debug=debug):
                stats_ds = compute_stats(da, prod, time_dim='time', label=per, **kwargs)
            
            # 6. Add dimensions to the stats dataset
            if 'time' not in stats_ds.coords:
                # Assign it as a coordinate so the file isn't "timeless"
                reference_time = da.time.values[0] # Use the first time from the original input 'da'
                stats_ds = stats_ds.expand_dims('time').assign_coords(time=[reference_time])
            
            # 6. Write the netcdf file to disk 
            with timer("stats_ds.to_netcdf (Disk I/O)", debug=debug):
                
                
                #stats_ds.to_netcdf(out_path,engine='netcdf4')
                with dask.config.set(scheduler='single-threaded'):
                    stats_ds.to_netcdf(out_path)
                if verbose: print(f"  ✅ Saved: {os.path.basename(out_path)}")
                if verbose: print("  🧊 Cooling down for 1.5s...")
                time.sleep(1.5)
                        
        except Exception as e:
            if debug: raise e # Crash and show traceback in debug mode
            print(f"❌ Error processing {out_path}: {e}")

        # 7. Clean up all data handles 
        finally: 
            try:
                if 'ds' in locals() and ds is not None: ds.close()
                if 'stats_ds' in locals() and stats_ds is not None: stats_ds.close()
            except:
                pass
        gc.collect()

def compute_stats(data, var_name, time_dim, label,stats=None, **kwargs):
    """
    Computes statistics on a DataArray.
    
    stats: list or str. 
           Options: 'mean', 'min', 'max', 'median', 'std', 'sum', 'var', 'count'
           Groupings: 'fast' (mean, min, max, count), 'full' (all)
    """
    
    # 1. Define the Available Library
    stats_library = {
        'mean':   lambda x: x.mean(dim=time_dim, skipna=True),
        'min':    lambda x: x.min(dim=time_dim, skipna=True),
        'max':    lambda x: x.max(dim=time_dim, skipna=True),
        'median': lambda x: x.median(dim=time_dim, skipna=True),
        'std':    lambda x: x.std(dim=time_dim, skipna=True),
        'sum':    lambda x: x.sum(dim=time_dim, skipna=True),
        'var':    lambda x: x.var(dim=time_dim, skipna=True),
        'count':  lambda x: x.notnull().sum(dim=time_dim)
    }

    # 2. Resolve the "Requested" stats
    if stats is None or stats == 'full':
        requested = list(stats_library.keys())
    elif stats == 'fast':
        requested = ['mean', 'min', 'max', 'count']
    elif isinstance(stats, str):
        requested = [stats]
    else:
        requested = stats # Assume it's a list: ['mean', 'std']

    # 3. Build the Dataset
    results = {}
    for s in requested:
        if s in stats_library:
            # Create a clean name: e.g., SST_M_mean
            col_name = f"{var_name}_{label}_{s}"
            results[col_name] = stats_library[s](data)
        else:
            print(f"⚠️ Warning: Statistics '{s}' is not recognized. Skipping.")

    return xr.Dataset(results)

def run_stats_pipeline(prods, periods=None, **kwargs):
                       #dataset=None, version=None, dataset_map=None,
                       #periods=None, daterange=None, climatology_range=None,
                       #subset=None,
                       #time_dim='time',
                       #is_running=True,
                       #verbose=False):
    
    
    """
    Runs a multi-product, multi-period stats pipeline.

    For each product in `prods` and each period code in `periods`, this function:
      1. Determines dataset defaults (via product_defaults).
      2. Locates daily (D) source or derived stats files (all other periods).
      3. Opens and subsets those files in time.
      4. Applies xarray grouping or rolling based on period code.
      5. Computes summary statistics (mean, min, max, etc.).
      6. Returns a dict of xr.Dataset results keyed by (prod, period).
      7. Adds metadata
      8. Saves output .nc files

    Parameters
    ----------
    prods (str or list of str): Single product name or list of product names (e.g. 'CHL', ['CHL','PSC']).
    dataset (str): Override dataset. Defaults to each product’s default from product_defaults() (optional).
    periods (list of str): Period codes to compute (e.g. ['D','W','M']) (optional). 
                           Defaults to ['W','WEEK','M','MONTH','A','ANNUAL','DOY'].
    subset (str): Regional subset (e.g. 'NES') (optional). Passed through to get_prod_files() as data_type or map override.
    daterange (str or tuple): Date range spec to subset time (optional). Passed to get_dates().
    time_dim (str): Name of the time dimension in your NetCDF files (default is 'time').
    version (str): Dataset version override (optional). Passed through to get_prod_files().
    dataset_map (str): Dataset map override (optional). Passed through to get_prod_files().
    verbose (bool): If True, prints progress and provenance (default is False).
        

    Returns
    -------
    results : dict
        {(prod, period): xr.Dataset} of computed statistics.
    """

    # Normalize inputs
    if isinstance(prods, str): prods = [prods.upper().strip()]
    if periods is None: periods = ['W','M','A']

    verbose = kwargs.pop('verbose', False)
    debug   = kwargs.get('debug', False) # Get it, but don't pop yet if others need it

    # --- 🛡️ THE WARNING FILTER ---
    # This block controls the "NaN chatter" from Dask/NumPy
    if not debug:
        # Hide "All-NaN slice" and "Divide by Zero" warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # Tell NumPy to be quiet about math errors in the background
        np.seterr(all='ignore')
        if verbose: print("🤫 Quiet Mode: RuntimeWarnings are hidden.")
    else:
        # If Debug is True, we want to see every single warning
        warnings.filterwarnings('default', category=RuntimeWarning)
        np.seterr(all='warn')
        print("🔍 Debug Mode: RuntimeWarnings are visible.")
    

    
    debug   = kwargs.pop('debug', False)  # 🛠 New Debug Flag
    dataset = kwargs.pop('dataset',None)
    subset  = kwargs.get('subset', 'GLOBAL')
    
    # 📋 Initialize an error log for the entire run
    failed_tasks = []
    failed_months = []   # Track which output files weren't created
    all_corrupt_inputs = [] # Track specific source files that are broken
    
    # Loop through prods
    for prod in prods:
        # Loop through the periods
        for period in periods:
            if verbose:
                print(f"\n🚀 Starting pipeline for {prod} | Period: {period}")
            
            # Create stats period map
            stats_map = build_stats_map(prod, period, 
                                        dataset=dataset,
                                        verbose=verbose,
                                        **kwargs
            )
            
                # --- Error Handling for Empty Maps ---
            if not stats_map:
                print(f"❌ ERROR: No processing tasks generated for {prod} ({period}).")
                print(f"   Check: Does {dataset} have {prod} data for the requested daterange?")
                print(f"   Check: Is the 'subset' ({subset}) correct for this dataset?")
                continue # Move to the next period/product
            
            # 2. Loop through the output tasks in the map
            for out_period, task in stats_map.items():
                                
                # SKIP if already up to date
                if task['is_up_to_date']:
                    if verbose: print(f"  ⏭ Skipping {out_period} (Up-to-date)")
                    continue
                
                # SKIP if there are no input files to process
                if not task['inputs'] or len(task['inputs']) == 0:
                    if verbose: print(f"  ⚠️ Skipping {out_period}: No source files found for this period.")
                    continue
                
                try:
                    process_single_stat(task, prod, period, verbose, **kwargs)
                
                except Exception as e:
                    error_info = f"{out_period}: {str(e)}"
                    failed_tasks.append(error_info)
                    
                    print(f"\n❌ Error on {out_period}:")
                    if debug:
                        # This prints the "Red Text" traceback without stopping the script
                        traceback.print_exc() 
                    else:
                        print(f"  ⚠️  {e}")
                        print(f"  ⏩ Skipping {out_period} and moving to next task.")

                if verbose:
                    print(f"  🛠 Finished attempt for {out_period} ({len(task['inputs'])} files)")    
                    

    # --- FINAL SUMMARY REPORT ---
    print("\n" + "="*60)
    print("🏁 PIPELINE FINISHED SUMMARY")
    print("="*60)
    
    if not failed_months and not all_corrupt_inputs:
        print("✨ Success! All files processed with 0 errors.")
    else:
        if failed_months:
            print(f"\n📂 OUTPUT FILES NOT CREATED ({len(failed_months)}):")
            for m in failed_months:
                print(f"  • {m}")

        if all_corrupt_inputs:
            print(f"\n🚨 CORRUPT INPUT FILES IDENTIFIED:")
            for err in all_corrupt_inputs:
                print(f"  • {err}")
            print("\n💡 ACTION: Delete the corrupt files listed above and re-run the pipeline.")
    
    print("="*60)



    

def get_climatology(ds, periods=['MONTH','WEEK','ANNUAL'],variable=None):
    """
    """
