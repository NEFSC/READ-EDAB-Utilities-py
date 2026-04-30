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
import concurrent.futures
from utilities.bootstrap.environment import bootstrap_environment
from utilities import get_period_info, get_period_sets, get_prod_files
from utilities import product_defaults, get_nc_prod, subset_dataset, file_parser, corrupt_file_detector
from utilities import build_stat_attributes, build_product_attributes, get_default_metadata, get_lut_metadata, get_current_utc_timestamp
from utilities import get_geospatial_metadata, get_temporal_metadata, get_summary_metadata, get_reference_metadata, get_fill_value,get_source_metadata

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

class DualLogger:
    """
    Intercepts standard print statements and writes them to 
    both the terminal and a persistent log file simultaneously.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Force write to disk immediately (crucial for parallel processing)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def boost_file_limits():
    """Raises the system limit for open file handles (Linux/Mac only)."""
    if platform.system() != 'Windows':
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to set it to 4096, but don't exceed the 'hard' system limit
        new_limit = min(4096, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
        print(f"🚀 System: File handle limit boosted to {new_limit}")

       
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

    # Check if the list is empty FIRST!
    if not files:
        if verbose: print(f"⚠️ No input files found for {prod} ({period}). Returning empty map.")
        return {} # Return an empty map so the pipeline cleanly skips it

    fp = file_parser(files[0])
    ds_name = fp[0].get('dataset') 

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
        out_name = f"{out_period}-{ds_name}-{prod}-{subset or 'GLOBAL'}-{data_type}.nc"
        out_path = os.path.join(output_dir, out_name)

        # Determine freshness
        is_up_to_date = False
        if os.path.exists(out_path) and not overwrite:
            out_mtime = os.path.getmtime(out_path)
            
            # Check 1: Does the output file have a newer mtime than ALL current inputs?
            mtime_ok = all(out_mtime > os.path.getmtime(f) for f in input_files)
            if mtime_ok:
                # Check 2: Does the internal list of files perfectly match our current list?
                try:
                    # Use xarray to quickly peek at the attributes (no decoding saves time/RAM)
                    import xarray as xr
                    with xr.open_dataset(out_path, engine='netcdf4', decode_times=False, decode_coords=False) as existing_ds:
                        stored_files_str = existing_ds.attrs.get('source_files', '')
                        
                    if stored_files_str:
                        # Convert both to sets so order doesn't matter
                        stored_basenames = set(f.strip() for f in stored_files_str.split(','))
                        current_basenames = set(os.path.basename(f) for f in input_files)
                        
                        # If the sets match perfectly, it is truly up-to-date!
                        is_up_to_date = (stored_basenames == current_basenames)
                    else:
                        # If the attribute doesn't exist (e.g., an old run), force an update
                        is_up_to_date = False
                        
                except Exception as e:
                    if verbose: print(f"  ⚠️ Could not read attributes of {out_name}: {e}. Forcing update.")
                    is_up_to_date = False

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

def preprocess_dataset(ds, prod, verbose=False):
    """
    Intercepts an xarray Dataset to perform product-specific math 
    and define the target variables before statistical aggregation.
    """
    target_vars = []
    
    if prod == 'PSC':
        if verbose: print("  🧪 Pre-computing absolute PSC concentrations...")
        
        # 1. Check if we are looking at raw data or derived data
        # If 'PSC_fmicro_mean' exists, skip because the math is already done
        if 'PSC_fmicro_mean' in ds.data_vars:
            target_vars = [v for v in ds.data_vars if v.endswith('_mean')]
            
        # Otherwise, this is a raw daily file. Do the math!
        elif all(v in ds.data_vars for v in ['PSC_fmicro', 'PSC_fnano', 'PSC_fpico', 'CHL']):
            # Calculate absolute fractions
            ds['PSC_micro'] = ds['PSC_fmicro'] * ds['CHL']
            ds['PSC_nano']  = ds['PSC_fnano']  * ds['CHL']
            ds['PSC_pico']  = ds['PSC_fpico']  * ds['CHL']
            
            # Explicitly define which variables to run stats on
            target_vars = [
                'PSC_fmicro', 'PSC_fnano', 'PSC_fpico', 
                'PSC_micro', 'PSC_nano', 'PSC_pico', 'CHL'
            ]
        else:
            if verbose: print("  ⚠️ Missing required base variables for PSC math.")

    else:
        # --- DEFAULT BEHAVIOR for standard products like SST, CHL, etc. ---
        mean_var_name = f"{prod}_mean"
        
        # Prioritize the 'mean' variable if we are reading derived stats
        if mean_var_name in ds.data_vars:
            target_vars = [mean_var_name]
        else:
            # Fallback to your original lookup logic
            # (Assuming get_nc_prod is available here, or pass ds_name into this function)
            target_vars = [prod if prod in ds.data_vars else get_nc_prod('dataset_name', prod)]

    # Store the target variables as an attribute so the pipeline knows what to do
    ds.attrs['pipeline_target_vars'] = target_vars
    return ds

def compute_stats(data, var_name, time_dim, label=None, stats=None, **kwargs):
    """
    Computes statistics on a DataArray.
    
    label: str or None. If provided, included in the output variable name.
    stats: list or str. 
           Options: 'mean', 'min', 'max', 'median', 'std', 'sum', 'var', 'count', 'gmean', 'gstd'
           Groupings: 'fast' (mean, min, max, count)
                      'full' (all standard linear stats)
                      'log'  (full + gmean, gstd)
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
        'count':  lambda x: x.notnull().sum(dim=time_dim),
        
        # Log-based statistics (Geometric Mean & Std) - Use .where(x > 0) to prevent log(0) or log(negative) errors
        'gmean':  lambda x: np.exp(np.log(x.where(x > 0)).mean(dim=time_dim, skipna=True)),
        'gstd':   lambda x: np.exp(np.log(x.where(x > 0)).std(dim=time_dim, skipna=True))
    }

    # Define standard grouping
    standard_stats = ['mean', 'min', 'max', 'median', 'std', 'sum', 'var', 'count']

    # 2. Resolve the "Requested" stats
    if stats is None or stats == 'full':
        requested = standard_stats
    elif stats == 'fast':
        requested = ['mean', 'min', 'max', 'count']
    elif stats == 'log':
        requested = standard_stats + ['gmean', 'gstd']
    elif isinstance(stats, str):
        requested = [stats]
    else:
        requested = stats # Assume it's a user-provided list

    # 3. Build the Dataset
    results = {}
    for s in requested:
        if s in stats_library:
            # Conditionally include the label in the variable name if provided
            if label:
                col_name = f"{var_name}_{label}_{s}"
            else:
                col_name = f"{var_name}_{s}"
            results[col_name] = stats_library[s](data)
        else:
            print(f"⚠️ Warning: Statistics '{s}' is not recognized. Skipping.")

    return xr.Dataset(results)

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
        input_files = task['inputs']
        max_retries = 3    

        # 2. Initialize to None so the finally block doesn't crash
        ds = None
        stats_ds = None
        input_files = task['inputs'].copy()
        corrupt_files_found = []

        # --- 🔄 RETRY LOOP ---
        for attempt in range(max_retries):
            if not input_files:
                if verbose: print(f"❌ FATAL: All input files for {per} are corrupt or missing.")
                return False, corrupt_files_found

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

                break # Success! All files opened successfully. Escape the retry loop
                
            except Exception as e:

                # If NOT on the last attempt, try to find the bad files and retry
                if attempt < max_retries - 1:
                    if verbose: print(f"  ⚠️  Hiccup on {per} (Attempt {attempt+1}). Retrying in 2s...")

                    # Find the bad files
                    bad_files_info = corrupt_file_detector(input_files)    
                    
                    if bad_files_info:
                        bad_paths = [info[0] for info in bad_files_info] # Isolate the paths so we can remove them
                        
                        # Format for the summary report: "filepath (Error)"
                        formatted_bad = [f"{p} ({err})" for p, err in bad_files_info]
                        corrupt_files_found.extend(formatted_bad)
                        
                        # 🎯 PURGE bad files from the input list for the next attempt
                        input_files = [f for f in input_files if f not in bad_paths]
                        
                        if verbose: print(f"  🗑️ Removed {len(bad_paths)} corrupt files. Retrying...")
                    else:
                        # It failed, but not because of file corruption (e.g., RAM or Dask issue)
                        if verbose: print(f"  ⚠️  No corrupt files detected. Retrying in 2s...")
                        time.sleep(2)
                else: 
                    pass # Let it fall out of the loop if the LAST attemp still failed
        
        # 🚨 SAFE EXIT: If ds is still None after all retries, fail cleanly.
        if 'ds' not in locals() or ds is None:
            if verbose: print(f"❌ FATAL: All retries exhausted. Could not open dataset for {per}.")
            return False, corrupt_files_found
        
        # 4. Resolve dataset, preprocess and loop through variables
        try:
            # 4A. Resolve dataset name for product mapping
            try:
                input_fp = file_parser(input_files[0])
                ds_name = input_fp[0].get('dataset') 
                ds_version = input_fp[0].get('dataset_version')
                
            except Exception as e:
                print(f"⚠️ Error with file parsing: {e}")                
                ds_name = kwargs.get('dataset', 'UNKNOWN') 
                ds_version = kwargs.get('dataset_version', 'UNKNOWN') 

            if ds_name: ds_name = ds_name.upper()

            # 4B. Preprocess data if needed (e.g. PSC files)
            with timer("preprocess_dataset", debug=debug):
                ds = preprocess_dataset(ds, prod, verbose=verbose)
                
            target_vars = ds.attrs.get('pipeline_target_vars', [])
            
            if not target_vars:
                raise ValueError(f"❌ No valid variables found to process for {prod}.")

            # 🎯 4C. Loop through target_vars
            computed_datasets = []
            
            for var_name in target_vars:
                if verbose: print(f"    📊 Extracting and preparing {var_name}...")

                # Extract the single DataArray safely
                try:
                    da = ds[var_name]
                except KeyError:
                    print(f"⚠️ Warning: '{var_name}' not found. Skipping this variable.")
                    continue

                # Spatial Subsetting & Integrity Check
                if subset:
                    if verbose: print(f"      Subsetting {var_name} to region: {subset}")
                    da = subset_dataset(da, subset)
                    if da.size == 0:
                        print(f"⚠️ Warning: Subset '{subset}' resulted in 0 pixels for {var_name}. Skipping.")
                        continue # Skip this variable but try the others!
                    
                if da.size == 0:
                    print(f"⚠️ Warning: DataArray is empty for {var_name}. Skipping.")
                    continue
                
                # Compute the statistics for this specific variable
                with timer(f"compute_stats ({var_name})", debug=debug):
                    # 🚨 Pass 'var_name' into compute_stats so it names the outputs correctly 
                    # (e.g. PSC_micro_mean instead of PSC_mean)
                    var_stats_ds = compute_stats(da, var_name, time_dim='time', **kwargs)
                
                computed_datasets.append(var_stats_ds)

            # Safety check: Did any variables successfully process?
            if not computed_datasets:
                print(f"⚠️ ERROR: No variables were successfully processed for {prod}.")
                return False, corrupt_files_found

            # 5. Merge dataset variables
            stats_ds = xr.merge(computed_datasets)

            # 6. Add variable attributes
            encoding = {}
            for out_var in stats_ds.data_vars:
                var = stats_ds[out_var]
                fill_value = get_fill_value(var)

                # Use fillna() instead of an undefined valid_mask
                var_masked = var.fillna(fill_value)

                # If dtype is float64 or float32, cast as float32
                if np.issubdtype(var_masked.dtype, np.floating):
                    var_masked = var_masked.astype("float32")
                    encoding[out_var] = {
                        "_FillValue": np.float32(fill_value),
                        "dtype": "float32",
                        "zlib": True,
                        "complevel": 4
                    }
                else:
                    encoding[out_var] = {
                        "_FillValue": fill_value,
                        "zlib": True,
                        "complevel": 4
                    }

                # Assign masked and casted data back
                stats_ds[out_var] = var_masked
                
                # 🎯 Extract the stat type (e.g., "PSC_micro_mean" -> "mean")
                stat_type = out_var.split('_')[-1]
                
                # Extract the base variable name for the attribute builder (e.g., "PSC_micro_mean" -> "PSC_micro")
                # This ensures build_stat_attributes looks up the specific fraction's metadata!
                base_var_name = out_var.replace(f"_{stat_type}", "")
                
                # Use the helper function to build stat-modified attributes using the specific variable name
                stats_ds[out_var].attrs = build_stat_attributes(base_var_name, stat_type, _FillValue=fill_value)
            
            # 7. Add dimensions to the stats dataset
            if 'time' not in stats_ds.coords:
                # Assign it as a coordinate so the file isn't "timeless"
                reference_time = da.time.values[0] 
                stats_ds = stats_ds.expand_dims('time').assign_coords(time=[reference_time])
            
            # 8. Add Global Metadata
            attrs = get_default_metadata(sheet="General")
            attrs = attrs | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
            
            # Pass the processed stats dataset to fetch accurate bounds and time ranges
            attrs = attrs | get_geospatial_metadata(dataset=stats_ds)
            attrs = attrs | get_temporal_metadata(ds=stats_ds)

            # Product Specific Attributes
            try:
                base_prod_attrs = build_product_attributes(prod)
                attrs["product_name"] = base_prod_attrs.get("long_name", prod)
            except ValueError as e:
                attrs["product_name"] = prod
                if debug: print(f"⚠️ {e}")

            try:
                attrs = attrs | get_summary_metadata(prod)
            except ValueError as e:
                if debug: print(f"⚠️ Skipping summary metadata: {e}")

            try:
                attrs["references"] = get_reference_metadata(prod, refs_only=True)
            except ValueError as e:
                if debug: print(f"⚠️ Skipping reference metadata: {e}")

            # 🎯 Refined History Statement
            input_files = task.get('inputs', [])
            num_files = len(input_files)
            
            if num_files > 0:
                input_basenames = [os.path.basename(f) for f in input_files]
                source_files_str = ", ".join(input_basenames)
                first_file = input_basenames[0]
                last_file = input_basenames[-1]
                file_stmt = f"Derived from {num_files} input files ranging from {first_file} to {last_file}."
            else:
                file_stmt = "No input files provided."

            new_history = (
                f"{get_current_utc_timestamp()} "
                f"{attrs['product_name']} statistics were calculated using xarray. "
                f"{file_stmt} Missing dates or masked pixels were excluded from the aggregation."
            )
            
            # Safely append to existing history if it exists
            attrs["history"] = f"{attrs['history']}\n{new_history}" if attrs.get("history") else new_history
            stats_ds.attrs['source_files'] = source_files_str  
                        
            # 🎯 Source Metadata
            try:
                # Make sure dataset and dataset_version variables are defined in your scope
                # Merges the flat dict returned by get_source_metadata into attrs
                source_meta = get_source_metadata(ds_name, dataset_version=ds_version, source_prefix="source")
                attrs = attrs | source_meta
            except Exception as e:
                if debug: print(f"⚠️ Warning extracting source metadata: {e}")
            
            # Assign global attributes
            stats_ds.attrs = attrs
                      
            # 6. Write the netcdf file to disk 
            with timer("stats_ds.to_netcdf (Disk I/O)", debug=debug):
                
                import warnings
                # 🎯 Temporarily mute the expected NaN math warnings during the save step
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    
                    # Force single-threaded execution to prevent Mac HDF5/NetCDF crashes
                    with dask.config.set(scheduler='single-threaded'):
                        stats_ds.to_netcdf(out_path)
                        
                if verbose: print(f"  ✅ Saved: {os.path.basename(out_path)}")
                if verbose: print("  🧊 Cooling down for 1.5s...")
                time.sleep(1.5)
                        
        except Exception as e:
            print(f"❌ Error processing {out_path}: {e}")
            if debug: 
                import traceback
                traceback.print_exc()
                #raise e # Crash and show traceback in debug mode
            return False, corrupt_files_found

        # 7. Clean up all data handles 
        finally: 
            try:
                if 'ds' in locals() and ds is not None: ds.close()
                if 'stats_ds' in locals() and stats_ds is not None: stats_ds.close()
            except:
                pass
        gc.collect()
        return True, corrupt_files_found


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
    verbose (bool): If True, prints progress and provenance (default is False).
        

    Returns
    -------
    results : dict
        {(prod, period): xr.Dataset} of computed statistics.
    """

    # Set up the Log file
    logfile = kwargs.pop('logfile', None)
    original_stdout = sys.stdout # Save the original terminal output

    if logfile:
        sys.stdout = DualLogger(logfile)
        print("\n" + "="*80)
        print(f"🟢 PIPELINE RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

    # Normalize inputs
    if isinstance(prods, str): prods = [prods.upper().strip()]
    if periods is None: 
        periods = ['W', 'M', 'A']
    elif isinstance(periods, str):
        periods = [periods]

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
    
    # Pull variables out of kwargs
    debug   = kwargs.pop('debug', False) 
    dataset = kwargs.pop('dataset',None)
    subset  = kwargs.get('subset', 'GLOBAL')
    parallel_runs = kwargs.pop('parallel_runs', 1) 
    
    # 📋 Initialize an error log for the entire run
    successful_tasks = []
    failed_tasks = []
    skipped_tasks = []
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
            
            # 🎯 Gather all tasks that actually need to be run
            tasks_to_run = []
            for out_period, task in stats_map.items():
                if task['is_up_to_date']:
                    if verbose: print(f"  ⏭ Skipping {out_period} (Up-to-date)")
                    skipped_tasks.append(out_period)
                    continue
                
                if not task['inputs'] or len(task['inputs']) == 0:
                    if verbose: print(f"  ⚠️ Skipping {out_period}: No source files found for this period.")
                    continue
                
                tasks_to_run.append((out_period, task))

            # If everything was up to date, skip to the next product/period
            if not tasks_to_run:
                continue

            # 🎯 Execute Tasks (Sequential or Parallel)
            if parallel_runs == 1:
                # --- SEQUENTIAL EXECUTION (Great for debugging) ---
                for out_period, task in tasks_to_run:
                    try:
                        success, corrupt_files = process_single_stat(task, prod, period, verbose, **kwargs)
                        if corrupt_files: all_corrupt_inputs.extend(corrupt_files)
                        
                        if success: successful_tasks.append(out_period)
                        else: failed_tasks.append(out_period)
                    except Exception as e:
                        failed_tasks.append(f"{out_period}: {str(e)}")
                        print(f"\n❌ Error on {out_period}:")
                        if debug:
                            import traceback
                            traceback.print_exc() 
                        else:
                            print(f"  ⚠️  {e}\n  ⏩ Skipping {out_period}.")
                    
                    if verbose: print(f"  🛠 Finished attempt for {out_period}")
            
            else:
                # --- PARALLEL EXECUTION ⚡ ---
                if verbose: print(f"\n  ⚡ Launching {len(tasks_to_run)} tasks across {parallel_runs} parallel workers...")
                
                with concurrent.futures.ProcessPoolExecutor(parallel_runs=parallel_runs) as executor:
                    # Submit all tasks to the pool
                    future_to_period = {
                        executor.submit(process_single_stat, task, prod, period, verbose, **kwargs): out_period
                        for out_period, task in tasks_to_run
                    }
                    
                    # Collect results as they finish (not necessarily in order!)
                    for future in concurrent.futures.as_completed(future_to_period):
                        out_period = future_to_period[future]
                        try:
                            success, corrupt_files = future.result()
                            if corrupt_files: all_corrupt_inputs.extend(corrupt_files)
                            
                            if success:
                                successful_tasks.append(out_period)
                                if verbose: print(f"  ✅ [Parallel] Finished {out_period}")
                            else:
                                failed_tasks.append(out_period)
                        except Exception as e:
                            failed_tasks.append(f"{out_period}: {str(e)}")
                            print(f"\n❌ [Parallel] Error on {out_period}:\n  ⚠️  {e}")

    # --- FINAL SUMMARY REPORT ---
    print("\n" + "="*60)
    print("🏁 PIPELINE FINISHED SUMMARY")
    print("="*60)
    
    total_processed = len(successful_tasks) + len(failed_tasks)

    if not failed_tasks and not all_corrupt_inputs:
        print("✨ Success! All files processed with 0 errors.")
    else:
        print(f"⚠️ Pipeline finished with ERRORS.")
        print(f"✅ Successfully processed: {len(successful_tasks)}")
        print(f"⏭  Skipped (up-to-date): {len(skipped_tasks)}")
        print(f"❌ Failed tasks: {len(failed_tasks)}")
        
        if failed_tasks:
            print(f"\n📂 FAILED TASKS (Output not created):")
            for task in failed_tasks:
                print(f"  • {task}")

        if all_corrupt_inputs:
            print(f"\n🚨 CORRUPT INPUT FILES IDENTIFIED:")
            for err in all_corrupt_inputs:
                print(f"  • {err}")
            print("\n💡 ACTION: Delete the corrupt files listed above and re-run the pipeline.")
    
    print("="*60)

    # Close the Log File and restore the terminal
    if logfile:
        print(f"\n🔴 PIPELINE RUN ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        sys.stdout.close()
        sys.stdout = original_stdout # Give control back to standard Python
    

def get_climatology(ds, periods=['MONTH','WEEK','ANNUAL'],variable=None):
    """
    """
