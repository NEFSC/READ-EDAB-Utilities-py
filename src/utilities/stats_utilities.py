import os
import sys
import re
import xarray as xr
import numpy as np
import pandas as pd
import h5netcdf
import warnings
import traceback
import concurrent.futures
from datetime import datetime
import time
from contextlib import contextmanager
import resource
import platform

import gc
import dask

# Local utilities
from utilities import bootstrap_environment
from utilities import get_period_info, get_period_sets
from utilities import get_prod_files, get_filepath, get_default_product_dataset, get_default_dataset_product, get_base_product
from utilities import generate_logfile_path
from utilities import get_nc_prod, subset_dataset, file_parser, corrupt_file_detector
from utilities import build_stat_attributes, build_product_attributes, get_default_metadata, get_lut_metadata, get_current_utc_timestamp
from utilities import get_geospatial_metadata, get_temporal_metadata, get_summary_metadata, get_reference_metadata, get_fill_value,get_source_metadata

env = bootstrap_environment(verbose=False)

xr.set_options(file_cache_maxsize=1) # Don't keep any unnecessary files open in the background

"""
STATS_UTILITIES 

Purpose:
    This module creates statistic outputs, including climatologies, from gridded satellite or modeled data. 

Main Functions:
    - run_stats_pipeline: Runs a multi-product, multi-period stats pipeline.
    - process_single_stat: Processes a single statistical task (opens files, applies spatial subsetting, computes statistics, and saves to NetCDF).
    - build_stats_map: Resolves output directories, finds input files, and builds a processing map indicating file 'freshness'.

Helper Functions:
    - compute_stats: Computes summary statistics across the time dimension for a given DataArray.
    - preprocess_dataset: Performs product-specific mathematical transformations and unit consistency enforcement.
    - boost_file_limits: Raises the system limit for open file handles (Linux/Mac only).
    - DualLogger: Intercepts standard print statements and writes them to both the terminal and a persistent log file.

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
    Oct 08, 2026 - KJWH: Updated documentation
    Oct 09, 2026 - KJWH: get_prod_files is now using the output period to add the data_type(STATS/CLIM) to the output file name
                         Bug fix: Added a second get_prod_files search when the input period is D and map_subset is provided (will default back to GLOBAL if no files are found)
    Oct 14, 2026 - KJWH: Added dataset to the get_period_sets() call to ensure the climatology_range is properly clamped to the dataset's start year if the input files are out of range of the default climatology_range
    Oct 15, 2026 - KJWH: Added calculations for skew (skewness) and kurt (kurtosis), as well as their log-transformed counterparts (gskew and gkurt), to the compute_stats() function.
    Oct 20, 2026 - KJWH: Replaced xr.open_dataset() in build_stats_map() and with the native h5netcdf to read the file attributes
    Oct 21, 2026 - KJWH: Forced h5netcdf engine in all .to_netcdf() calls
                         Added a try/except block to the map builders that automatically detects unreadable/corrupted NetCDF files and deletes them
                         Also fixed the accidentally dropped var (variance) key
                         Moved the default product, dataset, and period routing logic to the very top of run_stats_pipeline
                         Implemented a new tracking system that distinguishes between files that are Created, Replaced (overwritten due to new inputs), and Skipped (up-to-date), along with tallies for deleted corrupt files and missing baselines.
                         Intercepted the logfile parameter to pass through generate_logfile_path(), allowing standardized log naming across all execution environments.
                         Fixed Parallel Write Collisions: Identified and deleted a duplicate tasks_to_run.append line that was indiscriminately queueing "skipped" files for processing, causing multiple parallel workers to collide while trying to overwrite the exact same file.
                         Fixed a bug where ds_version was throwing a NameError in get_source_metadata by extracting it natively from file_parser alongside the dataset name.
                         Added .pop("_FillValue", None) to the var_attrs dictionary before assigning it to the dataset to prevent xarray from crashing when generating the .to_netcdf() encoding dictionaries.
    Oct 22, 2026 - KJWH: Added D7 period and weighted functionality for the 7-day running period
    Oct 27, 2026 - KJWH: Integrated get_base_product into preprocess_dataset to natively support direct subproduct targeting (e.g., PSC_micro)
                         Updated PSC family logic to preprocess_dataset to automatically rename raw Turner fractions (adding f prefix) and compute absolute concentrations on-the-fly using chlor_a.
                         Implemented a target_subprods filter at the end of preprocess_dataset to aggressively drop unrequested variables from memory before returning the dataset.
    Oct 29, 2026 - KWJH: Added verify_existing option that if False will skip the steps that open existing output files and speed up the processing



    
"""

class DualLogger:
    """
    Purpose:
        Intercepts standard print statements and writes them to both the terminal and a persistent log file simultaneously.
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
    """
    Purpose:
        Raises the system limit for open file handles (Linux/Mac only).
    """
    if platform.system() != 'Windows':
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Try to set it to 4096, but don't exceed the 'hard' system limit
        new_limit = min(4096, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
        #print(f"🚀 System: File handle limit boosted to {new_limit}")

       
def build_stats_map(prod, period, output_dir=None,
                    dataset=None, daterange=None,climatology_range=None,
                    dataset_version=None, dataset_map=None,
                    map_subset=None, is_running=True, overwrite=False, 
                    data_type=None,verbose=False, debug=False,
                    allow_nrt=False, verify_existing=True, **kwargs):
    """
    Purpose:
        Unified stats pipeline.

        This function orchestrates the preparation phase of the statistics pipeline by:
        1) Resolving and creating, if needed, the output directory.
        2) Finding valid daily input files.
        3) Generating a period map via get_period_sets().
        4) Building a stats processing map and evaluating file 'freshness' (mtime) to determine if reprocessing is required.
        5) Built in functionality to work with NRT and RAN processing level files

    Input Parameters:
        prod (str): The logical product name to process (e.g., 'SST', 'CHL', 'RRS').
        period (str): The time period code for the target statistics (e.g., 'M' for monthly, 'D8' for 8-day).
        output_dir (str, optional): The destination directory for the output files. If None, it is automatically resolved.
        dataset (str, optional): Override for the dataset name (e.g., 'OCCCI'). Defaults to the product's default.
        daterange (list or tuple, optional): A two-element sequence [start, end] specifying the date boundaries for filtering input files.
        climatology_range (list or tuple, optional): A two-element sequence [start_year, end_year] to filter years for climatology calculations.
        dataset_version (str, optional): Override for the dataset version (e.g., 'V6.0').
        dataset_map (str, optional): Specific map folder structure to override default routing (e.g., 'NES_4KM_DAILY').
        map_subset (str, optional): The spatial region abbreviation to subset the data to (e.g., 'NES'). Used for file naming and routing.
        is_running (bool, optional): Whether to treat the period as a running mean (if applicable). Default is True.
        overwrite (bool, optional): If True, flags all tasks for reprocessing regardless of existing file modification times. Default is False.
        data_type (str, optional): The data type (STATS/CLIM) suffix appended to the output file names. Default is None and it is inferred based on the output period.
        verbose (bool, optional): If True, prints progress updates and routing information. Default is False.
        debug (bool, optional): If True, prints extensive troubleshooting and traceback information. Default is False.
        allow_nrt(bool, optional): If True, allows near-real-time (NRT) files to be included in the climatology
        verify_existing(bool, optional): If True, skips the step to verify the input files in existing output files to speed up the processing

    Returns:
        dict: A dictionary mapping each target output period to its specific processing task definition.
        
        Example Structure:
        {
            'M_199710': {
                'inputs': ['/path/to/day1.nc', '/path/to/day2.nc', ...],
                'output': '/path/to/M_199710-OCCCI-RRS-NES-STATS.nc',
                'is_up_to_date': False
            }
        }
      
    """

    # Auto-resolve product-specific default map_subset
    if not map_subset:
        map_subset = 'NES' if 'PSC' in prod.upper() else 'NWA'
    
    # ---------------------------------------------------------
    # 1. Resolve Output Directory (if not provided)
    # ---------------------------------------------------------
    if output_dir is None:
        if verbose:
            print(f"🔍 Resolving output directory for {prod} ({period})...")
        
        # Get the output directory path based on product, dataset, and period. This will be used to store the computed statistics.
        # Set make_dir=True to create the directory if it does not exist.
        output_dir = get_filepath(
            prod,
            dataset=dataset,
            dataset_version=dataset_version,
            dataset_map=dataset_map,
            map_subset=map_subset,         # Use 'subset' to override region (GLOBAL -> NES)
            period=period,      # This triggers Step 7 logic for CLIMS/STATS
            getfilepath=True,
            make_dir=True,      # Create the folder if it doesn't exist
            verbose=verbose
        )

    if output_dir is None:
        raise ValueError("🛑 CRITICAL: output_dir resolved to None. get_filepath failed to generate a path.")
    elif isinstance(output_dir, list):
        if len(output_dir) == 1:
            output_dir = output_dir[0]
        elif len(output_dir) > 1:
            raise ValueError(f"🛑 CRITICAL: output_dir resolved to multiple paths: {output_dir}")
        else:
            raise ValueError("🛑 CRITICAL: output_dir resolved to an empty list.")
    if verbose: print(f"✅ Output directory is: {output_dir}")

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
    search_period = input_per
    
    if verbose:
        print(f"Searching for {input_per} files for {dataset}:{prod}")
    
    files = get_prod_files(prod,
        dataset=dataset,
        dataset_version=dataset_version,
        dataset_map=dataset_map,
        map_subset=map_subset,
        period=search_period,
        verbose=verbose
    )
    
    # Check if the list is empty FIRST!
    if not files:
        if search_period == 'D':
            # Search for the files again, but remove the map_subset filter so it will default to the GLOBAL files
            files = get_prod_files(prod,
                    dataset=dataset,
                    dataset_version=dataset_version,
                    dataset_map=dataset_map,
                    period=search_period,
                    verbose=verbose
                )
        if not files:
            if verbose: print(f"⚠️ No input files found for {prod} ({period}). Returning empty map.")
            return {} # Return an empty map so the pipeline cleanly skips it

    # ---------------------------------------------------------
    # 4 Exclude NRT from Climatologies (unless allow_nrt=True)
    # ---------------------------------------------------------
    is_clim = per_info.get("is_climatology", False)
    
    if is_clim and not allow_nrt:
        parsed_files = file_parser(files)
        filtered_files = []
        for f, p in zip(files, parsed_files):
            # Exclude if it came from an _NRT source dir or is a derived -NRT.nc file
            if p.get('processing_level') == 'NRT' or f.upper().endswith('-NRT.NC'):
                if verbose: print(f"  🚫 Excluding NRT file from climatology: {os.path.basename(f)}")
                continue
            filtered_files.append(f)
        files = filtered_files
        
        if not files:
            if verbose: print(f"⚠️ All available inputs were NRT and excluded for climatology {prod} ({period}).")
            return {}

    fp = file_parser(files[0])
    ds_name = fp[0].get('dataset') 

    # ---------------------------------------------------------
    # 5. Build period map using get_period_sets()
    # ---------------------------------------------------------
    period_sets = get_period_sets(period, files=files, daterange=daterange, climatology_range=climatology_range,dataset=ds_name)
    pm = period_sets["period_map"]

    # 4a. Get the output data_type (STATS/CLIM) based on the period
    if data_type is None:
        try:
            data_type = "CLIM" if is_clim else "STATS"
            if verbose:
                print(f"Inferred data_type='{data_type}' based on period '{period}'.")
        except Exception as e:
            data_type = "STATS"  
            if verbose:
                print(f"Could not infer data_type for period '{period}': {e}")

    # ---------------------------------------------------------
    # 6. Build stats map with mtime comparison
    # ---------------------------------------------------------
    stats_map = {}
    total_tasks = len(pm)
    up_to_date_count = 0
    nrt_replaced_count = 0
    # Initialize corrupt file trackers
    deleted_corrupt_files = []
    manual_delete_files = []

    for out_period, info in pm.items():
        input_files = info["input_files"]

        # Determine if the output needs the NRT suffix
        parsed_inputs = file_parser(input_files)
        has_nrt = any(p.get('processing_level') == 'NRT' or f.upper().endswith('-NRT.NC') for f, p in zip(input_files, parsed_inputs))

        # Build output filename
        out_suffix = f"{data_type}-NRT" if has_nrt else data_type
        out_name = f"{out_period}-{map_subset}-{ds_name}-{prod}-{out_suffix}.nc"

        if verbose: print(f"Checking {out_period}: {len(input_files)} input files -> {out_name}")
        out_path = os.path.join(output_dir, out_name)

        # Determine freshness
        is_up_to_date = False
        if os.path.exists(out_path) and not overwrite:
            out_mtime = os.path.getmtime(out_path)
            
            # Check 1: Does the output file have a newer mtime than ALL current inputs?
            mtime_ok = all(out_mtime > os.path.getmtime(f) for f in input_files)
            if mtime_ok:
                if verify_existing:
                    # Check 2: Does the internal list of files perfectly match our current list?
                    try:
                        check_corruption = False # Initialize flag
    
                        # Use native h5netcdf to read the file attributes
                        with h5netcdf.File(out_path, 'r') as existing_ds:
                            stored_files_str = existing_ds.attrs.get('source_files', '')
                            
                            # Safely decode if the library returns bytes instead of a string
                            if isinstance(stored_files_str, bytes):
                                stored_files_str = stored_files_str.decode('utf-8')
                            elif isinstance(stored_files_str, np.ndarray): # Handle numpy array strings
                                stored_files_str = str(stored_files_str[0]) if stored_files_str.size > 0 else ""
                                
                        if stored_files_str:
                            # Convert both to sets so order doesn't matter
                            stored_basenames = set(f.strip() for f in str(stored_files_str).split(','))
                            current_basenames = set(os.path.basename(f) for f in input_files)
                            
                            if stored_basenames == current_basenames:
                                check_corruption = True
                            else:
                                is_up_to_date = False
                        else:
                            is_up_to_date = False

                        if check_corruption:   
                            # The file claims to be up-to-date. Deep-scan it for HDF corruption!
                            is_corrupt = corrupt_file_detector([out_path])
                            if is_corrupt:
                                raise RuntimeError(f"Corrupt data chunk detected: {is_corrupt[0][1]}")
                            is_up_to_date = True
                        
                    except Exception as e:
                        if verbose: print(f"  ⚠️ Could not read attributes of {out_name}: {e}. Forcing update.")
                        # Attempt to delete the corrupt/locked file
                        try:
                            os.remove(out_path)
                            deleted_corrupt_files.append(out_name) # Track successful deletions
                            if verbose: print(f"  🗑️ Deleted corrupt file: {out_name}. Forcing recreation.")
                        except Exception as del_e:
                            manual_delete_files.append(out_name) # Track locked files
                            if verbose: print(f"  🚨 Failed to delete corrupt file {out_name}: {del_e}. Manual removal may be required.")
                        is_up_to_date = False
                else:
                    is_up_to_date = True

        if is_up_to_date:
            up_to_date_count += 1

        # Look for obsolete NRT files
        obsolete_nrt_file = None
        if not has_nrt:
            nrt_name = f"{out_period}-{map_subset}-{ds_name}-{prod}-{data_type}-NRT.nc"            
            nrt_path = os.path.join(output_dir, nrt_name)

            if os.path.exists(nrt_path):
                nrt_replaced_count += 1
                # Flag the file for deletion by the execution pipeline
                obsolete_nrt_file = nrt_path
        
        if verbose:
            status = "✓ up-to-date" if is_up_to_date else "⏳ needs update"
            print(f"{out_period}: {status} ({len(input_files)} inputs)")

        stats_map[out_period] = {
            "inputs": input_files,
            "output": out_path,
            "is_up_to_date": is_up_to_date,
            "file_exists": os.path.exists(out_path),
            "obsolete_nrt_file": obsolete_nrt_file
        }

    # ---------------------------------------------------------
    # 6. Final Summary Report
    # ---------------------------------------------------------
    needs_update_count = total_tasks - up_to_date_count
    completion_pct = (up_to_date_count / total_tasks * 100) if total_tasks > 0 else 0

    print("-" * 60)
    print(f"📊 STATS SUMMARY: {prod} | {period} | {map_subset}")
    print(f"📁 Output Dir: {output_dir}")
    print(f"✅ Up-to-date:  {up_to_date_count}/{total_tasks} ({completion_pct:.1f}%)")
    print(f"⏳ To process:  {needs_update_count} files")
    if nrt_replaced_count > 0: print(f"🗑️ NRT replaced: {nrt_replaced_count} files")
    print("-" * 60)
    
    return stats_map, deleted_corrupt_files, manual_delete_files

def preprocess_dataset(ds, prod, ds_name=None, verbose=False, **kwargs):
    """
    Purpose:
        Intercepts an xarray Dataset to perform product-specific mathematical transformations,
        enforce unit consistency (e.g., converting Kelvin to Celsius for SST), standardizes 
        variable names, and identifies the exact target variables for downstream statistical 
        computation (saved to the dataset's attributes as 'pipeline_target_vars').
    
    Input Parameters:
        ds (xarray.Dataset): The raw or partially processed input dataset to be evaluated.
        prod (str): The logical product name to process (e.g., 'SST', 'PSC', 'RRS').
        ds_name (str, optional): The overarching dataset name used for metadata or fallback variable lookups. Defaults to None.
        verbose (bool, optional): If True, prints detailed processing steps and transformation notifications to the terminal. Defaults to False.
        
    Returns:
        ds (xarray.Dataset): The fully preprocessed dataset containing the calculated or standardized variables and the required 'pipeline_target_vars' attribute.
    """
    
    # Auto-detect base product
    base_prod = get_base_product(prod)

    # If a subproduct (like PSC_micro) was passed directly, auto-configure the filter!
    target_subprods = kwargs.get('target_subprods', [])
    if base_prod != prod and not target_subprods:
        target_subprods = [prod]
        kwargs['target_subprods'] = target_subprods

    target_vars = []

    if base_prod == 'PSC':
        if verbose: print("  🦠 Detected Phytoplankton Size Class (PSC) product family...")

        # Check to see if processing an existing STATS file? (It will already have PSC_ prefixes)
        derived_vars = [v for v in ds.data_vars if str(v).upper().startswith('PSC_') and str(v).endswith('_mean')]
        if derived_vars:
            for v in derived_vars:
                base_name = v.replace('_mean', '')
                ds = ds.rename({v: base_name})
                target_vars.append(base_name)
        else: # Process raw daily files            
            # 1. Rename the raw fractions to standard names
            psc_keywords = ['micro', 'nano', 'pico']
            for v in ds.data_vars:
                v_str = str(v)
                v_lower = v_str.lower()
                if any(k in v_lower for k in psc_keywords):
                    # Strip any existing PSC prefix to avoid doubling it
                    clean_v = v_lower.replace('psc_', '')
                    
                    # Ensure it has the 'f' prefix for fraction
                    if not clean_v.startswith('f'):
                        clean_v = f"f{clean_v}"
                        
                    new_name = f"PSC_{clean_v}"
                    
                    if v_str != new_name:
                        if verbose: print(f"  🏷️ Renaming '{v_str}' -> '{new_name}'")
                        ds = ds.rename({v_str: new_name})
                        
                    if new_name not in target_vars:
                        target_vars.append(new_name)

            # 2. Calculate the Absolute Concentrations (PSC = Fraction * CHL)
            chl_var = None
            if 'chlor_a' in ds.data_vars:
                chl_var = 'chlor_a'
            elif 'CHL' in ds.data_vars:
                chl_var = 'CHL'
                
            if chl_var:
                if verbose: print("  🧮 Calculating absolute PSC concentrations using Chlorophyll-a...")
                
                fractions = [v for v in target_vars if 'fmicro' in v or 'fnano' in v or 'fpico' in v]
                
                for frac_var in fractions:
                    # Strip the 'f' to get the absolute name (e.g. PSC_fmicro -> PSC_micro)
                    abs_name = frac_var.replace('PSC_f', 'PSC_')
                    
                    # Calculate: Fraction * Chlorophyll
                    ds[abs_name] = ds[frac_var] * ds[chl_var]
                    
                    # Add metadata attributes for the new variable
                    ds[abs_name].attrs = ds[frac_var].attrs.copy()
                    ds[abs_name].attrs['long_name'] = ds[frac_var].attrs.get('long_name', '').replace('Fraction of ', '').replace('to total phytoplankton ', '')
                    ds[abs_name].attrs['units'] = ds[chl_var].attrs.get('units', 'mg m-3')
                    
                    target_vars.append(abs_name)
                    if verbose: print(f"    ✅ Generated {abs_name}")
            else:
                if verbose: print("  ⚠️ Warning: No Chlorophyll-a variable found. Cannot calculate absolute PSC concentrations.")

    # --- MULTI-WAVELENGTH PRODUCT LOGIC ---
    elif base_prod in ['RRS', 'APH', 'ADG', 'BBB']:
        if verbose: print(f"  🌈 Detected multi-wavelength product family: {prod}")
        
        # 1. Check if we are reading already-derived stats (e.g., RRS_412_mean)
        derived_vars = [v for v in ds.data_vars if v.upper().startswith(f"{prod}_") and v.endswith('_mean')]
        
        if derived_vars:
            for v in derived_vars:
                base_name = v.replace('_mean', '')
                if verbose: print(f"  🏷️ Renaming derived stat: '{v}' -> '{base_name}'")
                ds = ds.rename({v: base_name})
                target_vars.append(base_name)
                
        # 2. Otherwise, we are reading raw daily files. Find all wavelengths.
        else:
            # Regex match: Starts with product name (case-insensitive) + underscore + numbers
            # This grabs 'Rrs_412' but successfully ignores 'Rrs_412_bias'
            pattern = re.compile(rf"^{prod}_(\d+)$", re.IGNORECASE)
            
            raw_vars = [v for v in ds.data_vars if pattern.match(v)]
            
            if not raw_vars:
                if verbose: print(f"  ⚠️ No wavelength variables found for '{prod}' in this file.")
            
            # Standardize names to uppercase (e.g., 'Rrs_412' -> 'RRS_412')
            for v in raw_vars:
                clean_name = v.upper()
                if v != clean_name:
                    #if verbose: print(f"  🏷️ Standardizing variable name: '{v}' -> '{clean_name}'")
                    ds = ds.rename({v: clean_name})
                target_vars.append(clean_name)

    # --- PRIMARY PRODUCTIVITY (PPD) LOGIC ---
    elif base_prod == 'PPD':
        if verbose: print("  🌱 Detected Primary Productivity (PPD) product family...")
        
        # 1. Check if we are reading already-derived stats (e.g., PP_VGPM_mean)
        derived_vars = [v for v in ds.data_vars if v.startswith('PP_') and v.endswith('_mean')]
        
        if derived_vars:
            for v in derived_vars:
                base_name = v.replace('_mean', '')
                if verbose: print(f"  🏷️ Renaming derived stat: '{v}' -> '{base_name}'")
                ds = ds.rename({v: base_name})
                target_vars.append(base_name)
                
        # 2. Otherwise, we are reading raw daily files. Find all PP_ variables.
        else:
            raw_vars = [v for v in ds.data_vars if str(v).startswith('PP_')]            
            if not raw_vars:
                if verbose: print("  ⚠️ No 'PP_' variables found in this file.")
            
            target_vars.extend(raw_vars)
    
    # --- DEFAULT BEHAVIOR (Standard products or specific individual wavelengths) ---
    else:
        mean_var_name = f"{base_prod}_mean"
        
        if mean_var_name in ds.data_vars:
            if verbose: print(f"  🏷️ Renaming '{mean_var_name}' -> '{base_prod}'")
            ds = ds.rename({mean_var_name: base_prod})
            target_vars = [base_prod]
        else:            
            if base_prod in ds.data_vars:
                raw_var = base_prod
            else:
                case_matches = [v for v in ds.data_vars if v.upper() == base_prod.upper()]
                if case_matches:
                    raw_var = case_matches[0]
                else:
                    raw_var = get_nc_prod(ds_name, base_prod)

            if raw_var:
                if 'SST' in prod.upper() or 'TEMP' in prod.upper():
                    try:
                        max_val = float(ds[raw_var].max(skipna=True).values)
                        if not np.isnan(max_val):
                            if max_val > 100:
                                if verbose: print(f"  🌡️ Detected Kelvin (Max: {max_val:.1f}). Converting to Celsius...")
                                ds[raw_var] = ds[raw_var] - 273.15
                                ds[raw_var].attrs['units'] = 'Celsius' 
                            else:
                                if verbose: print(f"  🌡️ Detected Celsius. No conversion needed.")
                                ds[raw_var].attrs['units'] = 'Celsius' 
                    except Exception as e:
                        if verbose: print(f"  ⚠️ Could not verify temperature units: {e}") 
                
                if raw_var != prod and raw_var in ds.data_vars:
                    if verbose: print(f"  🏷️ Standardizing variable name: '{raw_var}' -> '{prod}'")
                    ds = ds.rename({raw_var: prod})
                    
                target_vars = [prod]

    # --- SUBPRODUCT FILTERING ---
    if target_subprods:
        upper_targets = [t.upper() for t in target_subprods]
        filtered_vars = [v for v in target_vars if v.upper() in upper_targets]
        if filtered_vars:
            target_vars = filtered_vars
            if verbose: print(f"  🎯 Subproduct filter applied. Keeping only: {target_vars}")

    ds.attrs['pipeline_target_vars'] = target_vars
    return ds


def compute_stats(data, var_name, time_dim, label=None, stats=None, weights=None, **kwargs):
    """
    Purpose:
        Computes summary statistics across the time dimension for a given DataArray.    
    
    Input Parameters
        data (xarray.DataArray) - The input data array to process.
        var_name (str) - The base variable name (e.g., 'SST') to prefix the output variables.
        time_dim (str, optional) - The dimension to reduce over (default is 'time').
        label (str or None) - If provided, included in the output variable name.
        stats (list or str) - The statistics to compute.
           Options: 'mean', 'min', 'max', 'median', 'std', 'sum', 'var', 'skew','kurt','count', 'gmean', 'gstd'
           Groupings: 'fast' (mean, min, max, count)
                      'full' (all standard linear stats)
                      'log'  (full + gmean, gstd, gskew, gkurt)
        weights (xarray.DataArray or None) - Weights for the weighted statistics.

    Returns
        xarray.Dataset - A dataset containing the computed statistical variables (mean, max, anomaly, etc.).
    
    """
    
    # 1. Define the Available Library
    if weights is not None:
        stats_library = {
            # Apply weights for mathematically supported statistics
            'mean':   lambda x: x.weighted(weights).mean(dim=time_dim, skipna=True),
            'sum':    lambda x: x.weighted(weights).sum(dim=time_dim, skipna=True),
            'var':    lambda x: x.weighted(weights).var(dim=time_dim, skipna=True),
            'std':    lambda x: x.weighted(weights).std(dim=time_dim, skipna=True),
            'skew':   lambda x: ((x - x.weighted(weights).mean(dim=time_dim, skipna=True))**3).weighted(weights).mean(dim=time_dim, skipna=True) / (x.weighted(weights).std(dim=time_dim, skipna=True)**3),
            'kurt':   lambda x: ((x - x.weighted(weights).mean(dim=time_dim, skipna=True))**4).weighted(weights).mean(dim=time_dim, skipna=True) / (x.weighted(weights).std(dim=time_dim, skipna=True)**4),
            'gmean':  lambda x: np.exp(np.log(x.where(x > 0)).weighted(weights).mean(dim=time_dim, skipna=True)),
            'gstd':   lambda x: np.exp(np.log(x.where(x > 0)).weighted(weights).std(dim=time_dim, skipna=True)),
            'gskew':  lambda x: ((np.log(x.where(x > 0)) - np.log(x.where(x > 0)).weighted(weights).mean(dim=time_dim, skipna=True))**3).weighted(weights).mean(dim=time_dim, skipna=True) / (np.log(x.where(x > 0)).weighted(weights).std(dim=time_dim, skipna=True)**3),
            'gkurt':  lambda x: ((np.log(x.where(x > 0)) - np.log(x.where(x > 0)).weighted(weights).mean(dim=time_dim, skipna=True))**4).weighted(weights).mean(dim=time_dim, skipna=True) / (np.log(x.where(x > 0)).weighted(weights).std(dim=time_dim, skipna=True)**4),
            
            # Unweighted fallbacks for everything else
            'min':    lambda x: x.min(dim=time_dim, skipna=True),
            'max':    lambda x: x.max(dim=time_dim, skipna=True),
            'median': lambda x: x.median(dim=time_dim, skipna=True),
            'count':  lambda x: x.notnull().sum(dim=time_dim),
        }
    else:
        # Standard unweighted library
        stats_library = {
            'mean':   lambda x: x.mean(dim=time_dim, skipna=True),
            'min':    lambda x: x.min(dim=time_dim, skipna=True),
            'max':    lambda x: x.max(dim=time_dim, skipna=True),
            'median': lambda x: x.median(dim=time_dim, skipna=True),
            'std':    lambda x: x.std(dim=time_dim, skipna=True),
            'sum':    lambda x: x.sum(dim=time_dim, skipna=True),
            'var':    lambda x: x.var(dim=time_dim, skipna=True),
            'skew':   lambda x: ((x - x.mean(dim=time_dim, skipna=True))**3).mean(dim=time_dim, skipna=True) / (x.std(dim=time_dim, skipna=True)**3),
            'kurt':   lambda x: ((x - x.mean(dim=time_dim, skipna=True))**4).mean(dim=time_dim, skipna=True) / (x.std(dim=time_dim, skipna=True)**4),
            'count':  lambda x: x.notnull().sum(dim=time_dim),
            'gmean':  lambda x: np.exp(np.log(x.where(x > 0)).mean(dim=time_dim, skipna=True)),
            'gstd':   lambda x: np.exp(np.log(x.where(x > 0)).std(dim=time_dim, skipna=True)),
            'gskew':  lambda x: ((np.log(x.where(x > 0)) - np.log(x.where(x > 0)).mean(dim=time_dim, skipna=True))**3).mean(dim=time_dim, skipna=True) / (np.log(x.where(x > 0)).std(dim=time_dim, skipna=True)**3),
            'gkurt':  lambda x: ((np.log(x.where(x > 0)) - np.log(x.where(x > 0)).mean(dim=time_dim, skipna=True))**4).mean(dim=time_dim, skipna=True) / (np.log(x.where(x > 0)).std(dim=time_dim, skipna=True)**4)
        }

    # Define standard grouping
    standard_stats = ['mean', 'min', 'max', 'median', 'std', 'sum', 'var', 'skew', 'kurt', 'count']

    # 2. Resolve the "Requested" stats
    if stats is None or stats == 'full':
        requested = standard_stats
    elif stats == 'fast':
        requested = ['mean', 'min', 'max', 'count']
    elif stats == 'log':
        requested = standard_stats + ['gmean', 'gstd', 'gskew', 'gkurt']
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
        Purpose:
            Processes a single statistical task: opens files, resolves product keys,
            applies spatial subsetting, computes statistics, and saves to NetCDF.

        Input Parameters:
            task (dict): Dictionary containing 'inputs' (list of paths) and 'output' (path).
            prod (str): The logical product name (e.g., 'SST').
            per (str): The time period label (e.g., 'M', 'W').
            verbose (bool): If True, prints progress updates.
            **kwargs: Includes 'debug', 'subset', and 'dataset' fallbacks.

        Returns:
            tuple: A tuple containing a boolean indicating success and a list of corrupt files found.
        """
        
        boost_file_limits()
    
        
        def sanitize_for_netcdf(ds):
            """
            Recursively scrubs all dictionaries and NoneTypes from a dataset's attributes.
            Must be run immediately before .to_netcdf()
            """
            for obj in [ds] + [ds[var] for var in ds.variables]:
                for k, val in list(obj.attrs.items()):
                    if isinstance(val, dict):
                        # Extract the default string if it's a schema dictionary
                        obj.attrs[k] = val.get('default', str(val))
                    elif val is None:
                        # Xarray also crashes if it tries to save a NoneType attribute
                        obj.attrs[k] = "None"
            return ds
        
        # 1. Extract variables from the inputs
        debug = kwargs.get('debug', False)
        map_subset = kwargs.get('map_subset')
        out_path = task['output']
        input_files = task['inputs']
        max_retries = 3    

        # 2. Initialize to None so the finally block doesn't crash
        ds = None
        stats_ds = None
        input_files = task['inputs'].copy()
        corrupt_files_found = []

        # --- 🔄 RETRY LOOP ---
        try:
            for attempt in range(max_retries):
                if not input_files:
                    print(f"❌ FATAL: All input files for {per} are corrupt or missing.")
                    return False, corrupt_files_found

                # 3. Open the dataset
                try:                
                    # Extract dataset name 
                    try:
                        input_fp = file_parser(input_files[0])
                        ds_name = input_fp[0].get('dataset') 
                        ds_version = input_fp[0].get('version')
                    except Exception:
                        ds_name = kwargs.get('dataset', 'UNKNOWN') 
                        ds_version = kwargs.get('dataset_version', None) 
                    if ds_name: ds_name = ds_name.upper()

                    # Route 1: Safe, manual loading for complex multi-wavelength products
                    if prod in ['RRS', 'APH', 'ADG', 'BBB']:
                        if debug: print(f"      📖 Opening complex dataset manually (Attempt {attempt+1})...")
                
                        daily_datasets = []   
                        files_to_remove = []  # Track files that fail during manual load     

                        for f_path in input_files:
                            if verbose: print(f" Loading file {f_path}")
                            try:
                                with xr.open_dataset(f_path, engine='h5netcdf') as single_ds:                                
                                    single_ds = preprocess_dataset(single_ds, prod, ds_name=ds_name, verbose=verbose)
                                    target_vars = single_ds.attrs.get('pipeline_target_vars', [])
                                    
                                    vars_to_drop = [v for v in single_ds.data_vars if v not in target_vars]
                                    if vars_to_drop:
                                        single_ds = single_ds.drop_vars(vars_to_drop)
                                    
                                    if map_subset:
                                        single_ds = subset_dataset(single_ds, map_subset)
                                        
                                    single_ds.load()
                                    daily_datasets.append(single_ds)
                            except Exception as file_err:
                                # Catch the exact file causing the HDF5 filter failure!
                                print(f"  ⚠️ Warning: Failed to read/load file {os.path.basename(f_path)}")
                                print(f"     Error: {file_err}")
                                files_to_remove.append(f_path)
                                
                                # Add to corrupt files list for your final summary report
                                corrupt_files_found.append(f"{f_path} (HDF5 filter read error)")
                                
                        # If files failed, trigger a retry by raising an exception so the retry loop catches it and cleans input_files
                        if files_to_remove:
                            raise RuntimeError(f"Encountered {len(files_to_remove)} unreadable/corrupt files during manual load.")

                        if not daily_datasets:
                            raise ValueError("All files in this period failed to load.")

                        if debug: print("      🔗 Concatenating all days into a single dataset...")
                        ds = xr.concat(daily_datasets, dim='time', combine_attrs='override')
                        ds.attrs['pipeline_target_vars'] = target_vars

                    # Route 2: High-speed parallel loading for standard products (SST, CHL, PSC, etc.)
                    else:
                        if debug: print(f"      📖 Opening standard dataset via mfdataset (Attempt {attempt+1})...")
                        ds = xr.open_mfdataset(
                            input_files,
                            combine="by_coords",
                            decode_timedelta=True,
                            chunks={'time': 1},              
                            parallel=False,                  
                            engine='h5netcdf',               
                            coords='minimal',                
                            data_vars='minimal',             
                            compat='override'                
                        )
                        
                        # 🎯 Apply preprocessing and subsetting to the opened standard dataset
                        if debug: print(f"      ⚙️ Preprocessing and subsetting standard dataset...")
                        
                        ds = preprocess_dataset(ds, prod, ds_name=ds_name, verbose=verbose)
                        target_vars = ds.attrs.get('pipeline_target_vars', [])
                        
                        vars_to_drop = [v for v in ds.data_vars if v not in target_vars]
                        if vars_to_drop:
                            ds = ds.drop_vars(vars_to_drop)
                            
                        if map_subset:
                            if verbose: print(f"      🗺️ Subsetting entire dataset to region: {map_subset}")
                            ds = subset_dataset(ds, map_subset)
                    
                except Exception as e:

                    # If NOT on the last attempt, try to find the bad files and retry
                    if attempt < max_retries - 1:
                        if verbose: print(f"  ⚠️  Hiccup on {per} (Attempt {attempt+1}). Retrying in 2s...")
                        if verbose: print(f"  🛑 ROOT CAUSE ERROR: {e}")
                        
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
            
            # 4. Loop through variables and compute stats
            try:
                target_vars = ds.attrs.get('pipeline_target_vars', [])
                if not target_vars:
                    raise ValueError(f"❌ No valid variables found to process for {prod}.")
                
                computed_datasets = []

                # 🎯 Wrap the loop to protect `da.size` from triggering multi-threaded HDF5 crashes in Route 2
                with dask.config.set(scheduler='single-threaded'):
                    for var_name in target_vars:
                        if verbose: print(f"    📊 Extracting and preparing {var_name}...")

                        # Extract the single DataArray safely
                        try:
                            da = ds[var_name]
                        except KeyError:
                            print(f"⚠️ Warning: '{var_name}' not found. Skipping this variable.")
                            continue

                        if da.size == 0:
                            print(f"⚠️ Warning: DataArray is empty for {var_name}. Skipping.")
                            continue
                        
                        # Determine the weights for specific periods (e.g. weighting the number of days in a month for the annual (A) period)
                        weights = None
                        monthly_weights = ['M3','A','JFM','AMJ','JAS','OND']

                        if debug: print(f" Checking period ({per}) if weights are needed for the stats caclulations")
                        
                        # Extract the number of days per month
                        if per in monthly_weights:
                            weights = da.time.dt.days_in_month
                            weights.name = "weights"
                            if verbose and var_name == target_vars[0]: # Only print once per file
                                print(f"      ⚖️ Applying days_in_month weights for {per} period...")
                        
                        # Apply Triangle Weights [1, 2, 1] to the 3-Day Running Mean
                        elif per == 'D3':
                            # Safety check: Ensure we actually found 3 days of inputs
                            if da.sizes['time'] == 3:
                                weights = xr.DataArray([1.0, 2.0, 1.0], dims=['time'], coords={'time': da.time})
                                weights.name = "weights"
                                if verbose and var_name == target_vars[0]:
                                    print(f"      ⚖️ Applying [1, 2, 1] central weights for {per} period...")
                            else:
                                if verbose and var_name == target_vars[0]:
                                    print(f"      ⚠️ {per} has {da.sizes['time']} inputs instead of 3. Falling back to unweighted mean.")
                        
                       # Apply Symmetric Tent Weights to the 7-Day Running Mean
                        elif per == 'D7':
                            if da.sizes['time'] == 7:
                                weights = xr.DataArray([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dims=['time'], coords={'time': da.time})
                                weights.name = "weights"
                                if verbose and var_name == target_vars[0]:
                                    print(f"      ⚖️ Applying [1, 2, 3, 4, 3, 2, 1] central weights for {per} period...")
                            else:
                                if verbose and var_name == target_vars[0]:
                                    print(f"      ⚠️ {per} has {da.sizes['time']} inputs instead of 7. Falling back to unweighted mean.")
                       
                        # Apply Symmetric Tent Weights to the 8-Day Running Mean
                        elif per == 'D8':
                            if da.sizes['time'] == 8:
                                weights = xr.DataArray([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0], dims=['time'], coords={'time': da.time})
                                weights.name = "weights"
                                if verbose and var_name == target_vars[0]:
                                    print(f"      ⚖️ Applying [1, 2, 3, 4, 4, 3, 2, 1] central weights for {per} period...")
                            else:
                                if verbose and var_name == target_vars[0]:
                                    print(f"      ⚠️ {per} has {da.sizes['time']} inputs instead of 8. Falling back to unweighted mean.")
                        
                        if debug: print(f" Computing stats for {var_name}")
                        # Compute the statistics for this specific variable
                        # 🚨 Pass 'var_name' into compute_stats so it names the outputs correctly (e.g. PSC_micro_mean instead of PSC_mean)
                        var_stats_ds = compute_stats(da, var_name, time_dim='time', weights=weights, **kwargs)
                
                        if debug: print(f" Adding computed stats for {var_name} to the main dataset")
                        computed_datasets.append(var_stats_ds)

                # Safety check: Did any variables successfully process?
                if not computed_datasets:
                    print(f"⚠️ ERROR: No variables were successfully processed for {prod}.")
                    return False, corrupt_files_found

                # 5. Merge dataset variables
                if debug: print(f"     Merging {len(computed_datasets)} computed variables into a single dataset...")
                stats_ds = xr.merge(computed_datasets)

                # Force Dask to finish all math and load into RAM imeediately after merging. 
                # This prevents any downstream metadata functions from secretly triggering multi-threaded HDF5 reads.
                if debug: print("     Computing stats into memory...")
                # Catch the expected NaN math warnings during computation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    
                    # Force single-threaded execution to prevent Mac HDF5 read crashes
                    with dask.config.set(scheduler='single-threaded'):
                        stats_ds = stats_ds.load()

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
                    traceback.print_exc()
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
                attrs['source_files'] = source_files_str  
                            
                # Source Metadata
                try:
                    # Make sure dataset and dataset_version variables are defined in your scope
                    # Merges the flat dict returned by get_source_metadata into attrs
                    source_meta = get_source_metadata(ds_name, dataset_version=ds_version, source_prefix="source")
                    attrs = attrs | source_meta
                except Exception as e:
                    if debug: print(f"⚠️ Warning extracting source metadata: {e}")
                
                # Assign global attributes
                stats_ds.attrs = attrs

                # Sanitize the dataset at the last possible millisecond
                stats_ds = sanitize_for_netcdf(stats_ds)
                
                # 6. Write the netcdf file to disk 
                if debug: print(f"      💾 Saving {out_path}")
                stats_ds.to_netcdf(out_path,format='NETCDF4', engine='h5netcdf')
                        
                print(f"  Saved: {os.path.basename(out_path)}")
                            
            except Exception as e:
                print(f"❌ Error processing {out_path}: {e}")
                if debug: 
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
            
            if verbose: print(f"  ✅ Successfully wrote: {os.path.basename(out_path)}")
            return True, corrupt_files_found
        except Exception as e:
            # The traceback catcher will ONLY print if a file completely fails.
            print(f"\n❌ FATAL ERROR processing: {os.path.basename(out_path)}")
            print(f"Details: {e}")
            
            # Only print the giant traceback if debug mode is on, to keep logs clean
            if debug:
                print("--- TRACEBACK ---")
                traceback.print_exc()
                print("-----------------\n")
                
            # Return False for failure!
            return False, corrupt_files_found
    


def run_stats_pipeline(prods=None, periods=None, **kwargs):
    """
    Purpose:
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

    Input Parameters
        prods (str or list of str): Single product name or list of product names (e.g. 'CHL', ['CHL','PSC']).
        dataset (str): Override dataset. Defaults to each product’s default from product_defaults() (optional).
        periods (list of str): Period codes to compute (e.g. ['D','W','M']) (optional). 
                               Defaults to ['W','WEEK','M','MONTH','A','ANNUAL','DOY'].
        map_subset (str): Regional subset (e.g. 'NES') (optional). Passed through to get_prod_files().
        daterange (str or tuple): Date range spec to subset time (optional). Passed to get_dates().
        time_dim (str): Name of the time dimension in your NetCDF files (default is 'time').
        version (str): Dataset version override (optional). Passed through to get_prod_files().
        verbose (bool): If True, prints progress and provenance (default is False).
        
    Returns
        results (dict): {(prod, period): xr.Dataset} of computed statistics.
    """

    dataset = kwargs.pop('dataset',None)
    # Fallback logic for products
    if not prods:
        if dataset:
            prods = [get_default_dataset_product(dataset.upper())]
        else:
            raise ValueError("Must provide either 'prods' or 'dataset' to infer default product.")

    # Normalize prods input
    if isinstance(prods, str): prods = [prods.upper().strip()]
    
    # Fallback and normalization of periods
    if periods is None: 
        periods = ['W', 'M', 'A']
    elif isinstance(periods, str):
        periods = [periods]

    # Handle logfile auto-generation
    logfile = kwargs.pop('logfile', None)
    if logfile in ["auto", True]:
        logfile = generate_logfile_path("stats", prods=prods, dataset=dataset, env_dict=env)    

    original_stdout = sys.stdout # Save the original terminal output

    if logfile:
        sys.stdout = DualLogger(logfile)
        print("\n" + "="*80)
        print(f"🟢 PIPELINE RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

    # Use try/finally to GUARANTEE the terminal output is restored, even if the code crashes
    try:
        verbose = kwargs.pop('verbose', False)
        debug   = kwargs.get('debug', False) # Get it, but don't pop yet if others need it
        should_log = verbose or bool(logfile)

        # --- 🛡️ THE WARNING FILTER ---
        # This block controls the "NaN chatter" from Dask/NumPy
        if not debug:
            warnings.filterwarnings('ignore', category=RuntimeWarning)  # Hide "All-NaN slice" and "Divide by Zero" warnings
            np.seterr(all='ignore') # Tell NumPy to be quiet about math errors in the background
            if verbose: print("🤫 Quiet Mode: RuntimeWarnings are hidden.")
        else: # If Debug is True, we want to see every single warning
            warnings.filterwarnings('default', category=RuntimeWarning)
            np.seterr(all='warn')
            print("🔍 Debug Mode: RuntimeWarnings are visible.")
        
        # Pull variables out of kwargs
        debug   = kwargs.get('debug', False) 
        map_subset  = kwargs.get('map_subset')
        parallel_runs = kwargs.pop('parallel_runs', 1) 
        allow_nrt = kwargs.pop('allow_nrt', False)
        verify_existing = kwargs.pop('verify_existing',False)
        
        # 📋 Initialize an error log for the entire run
        total_input_files = 0
        skipped_tasks = []
        replaced_tasks = []
        created_tasks = []
        failed_tasks = []
        all_corrupt_inputs = set()
        all_deleted_corrupt = [] 
        all_manual_delete = []   

        # Loop through prods
        for prod in prods:
            # Dynamic Dataset Resolution Per Product
            current_dataset = dataset
            if not current_dataset:
                try:
                    current_dataset = get_default_product_dataset(prod)
                except KeyError:
                    if should_log: print(f"\n⚠️ No default dataset found for product '{prod}'. Skipping.")
                    continue

            check_corrupt = kwargs.get('check_corrupt', True)
            max_retries = kwargs.get('max_retries', 3)
            
            # Loop through the periods
            for period in periods:
                for attempt in range(1, max_retries + 1):
                    if should_log:
                        prefix = f"\n🚀 Starting stats pipeline" if attempt == 1 else f"\n🔄 Retry Attempt {attempt}/{max_retries}"
                        print(f"{prefix} for {prod} | Period: {period}")
                                
                # Create stats period map
                    stats_map, deleted_files, manual_delete_files = build_stats_map(prod, period, dataset=dataset, verify_existing=verify_existing, verbose=verbose, allow_nrt=allow_nrt, **kwargs)

                    # Aggregate the corrupt file lists
                    all_deleted_corrupt.extend(deleted_files)
                    all_manual_delete.extend(manual_delete_files)
                    
                    # --- Error Handling for Empty Maps ---
                    if not stats_map:
                        print(f"❌ ERROR: No processing tasks generated for {prod} ({period}).")
                        print(f"   Check: Does {dataset} have {prod} data for the requested daterange?")
                        print(f"   Check: Is the 'map_subset' ({map_subset}) correct for this dataset?")
                        break # Move to the next period/product
                    
                    # Gather all tasks that actually need to be run
                    tasks_to_run = []
                    for out_period, task in stats_map.items():
                        total_input_files += len(task['inputs'])
                        file_name = os.path.basename(task['output'])

                        # 1. Existing file skipped
                        if task['is_up_to_date']:
                            if should_log: print(f"  ⏭  Skipped: {file_name} (Up-to-date)")
                            # Delete obsolete NRT file if the pure RAN file is already up-to-date
                            obsolete = task.get('obsolete_nrt_file')
                            if obsolete and os.path.exists(obsolete):
                                os.remove(obsolete)
                                if should_log: print(f"  🗑️ Deleted obsolete NRT file: {os.path.basename(obsolete)}")
                            skipped_tasks.append(out_period)

                        # 2 & 3. Queued for creation or replacement
                        elif not task['inputs']:
                            if should_log: print(f"  ⚠️  Failed: {file_name} (No source files found)")
                            failed_tasks.append(f"{out_period} (No inputs)")
                        else:
                            if task['file_exists']:
                                if should_log: print(f"  🔄 Queued for Replacement: {file_name} (New inputs detected)")
                            else:
                                if should_log: print(f"  🆕 Queued for Creation: {file_name}")
                            tasks_to_run.append((out_period, task))
                        
                    # If everything was up to date, skip to the next product/period
                    if not tasks_to_run:
                        break

                    if parallel_runs == 1:
                        # --- SEQUENTIAL EXECUTION ---
                        for out_period, task in tasks_to_run:
                            try:
                                success, corrupt_files = process_single_stat(task, prod, period, verbose, **kwargs)
                                
                                if corrupt_files: all_corrupt_inputs.update(corrupt_files)

                                if success: 
                                    # Delete obsolete NRT file if applicable
                                    obsolete = task.get('obsolete_nrt_file')
                                    if obsolete and os.path.exists(obsolete):
                                        os.remove(obsolete)
                                        if should_log: print(f"  🗑️ Deleted obsolete NRT file: {os.path.basename(obsolete)}")

                                    if task['file_exists']:
                                        replaced_tasks.append(out_period)
                                        if should_log: print(f"  ✅ Replaced: {file_name}")
                                    else:
                                        created_tasks.append(out_period)
                                        if should_log: print(f"  ✅ Created: {file_name}")
                                else: 
                                    failed_tasks.append(f"{out_period} (Processing failed)")
                                    if should_log: print(f"  ❌ Error creating: {file_name}")
                            except Exception as e:
                                failed_tasks.append(f"{out_period}: {str(e)}")
                                if should_log: print(f"\n❌ Error on {out_period}:")
                                if debug:
                                    traceback.print_exc() 
                                else:
                                    if should_log: print(f"  ⚠️  {e}\n  ⏩ Skipping {out_period}.")
                            
                            if verbose: print(f"  🛠 Finished attempt for {out_period}")
                    
                    else:
                        # --- PARALLEL EXECUTION ⚡ ---
                        if verbose: print(f"\n  ⚡ Launching {len(tasks_to_run)} tasks across {parallel_runs} parallel workers...")
                        
                        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_runs) as executor:
                            # Submit all tasks to the pool
                            future_to_period = {
                                executor.submit(process_single_stat, task, prod, period, verbose, **kwargs): out_period
                                for out_period, task in tasks_to_run
                            }
                            
                            # Collect results as they finish (not necessarily in order!)
                            for future in concurrent.futures.as_completed(future_to_period):
                                out_period = future_to_period[future]
                                task = stats_map[out_period]
                                file_name = os.path.basename(task['output'])
                                try:
                                    success, corrupt_files = future.result()
                                    if corrupt_files: all_corrupt_inputs.update(corrupt_files)                                
                                    if success:
                                        # Delete obsolete NRT file if applicable
                                        obsolete = task.get('obsolete_nrt_file')
                                        if obsolete and os.path.exists(obsolete):
                                            os.remove(obsolete)
                                            if should_log: print(f"  🗑️ [Parallel] Deleted obsolete NRT file: {os.path.basename(obsolete)}")

                                        if task['file_exists']:
                                            replaced_tasks.append(out_period)
                                            if should_log: print(f"  ✅ [Parallel] Replaced: {file_name}")
                                        else:
                                            created_tasks.append(out_period)
                                            if should_log: print(f"  ✅ [Parallel] Created: {file_name}")
                                    else:
                                        failed_tasks.append(f"{out_period} (Processing failed)")
                                        if should_log: print(f"  ❌ [Parallel] Error creating: {file_name}")
                                except Exception as e:
                                    failed_tasks.append(f"{out_period}: {str(e)}")
                                    if should_log: print(f"  ❌ [Parallel] Error creating: {file_name} - {str(e)}")
                    # --- CORRUPT FILE CHECK & RETRY LOGIC ---
                        corrupt_this_run = False
                        if check_corrupt:
                            if should_log: print(f"  🔍 Verifying generated STATS files for corruption...")
                            
                            # Collect all expected output files from this specific run
                            out_files = [t['output'] for t in stats_map.values() if t['output'] != "Unknown" and os.path.exists(t['output'])]
                            
                            if out_files:
                                bad_stats = corrupt_file_detector(out_files)
                                if bad_stats:
                                    corrupt_this_run = True
                                    if should_log: print(f"    🚨 Found {len(bad_stats)} corrupt STATS files.")
                                    for bad_file, err in bad_stats:
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
        print("🏁 STATS PIPELINE FINISHED SUMMARY")
        print("="*60)
        print(f"📥 Input files found:       {total_input_files}")
        print(f"⏭  Existing files skipped:  {len(skipped_tasks)}")
        print(f"🔄 Existing files replaced: {len(replaced_tasks)}")
        print(f"🆕 New files created:       {len(created_tasks)}")
        print(f"❌ Errors/Not created:      {len(failed_tasks)}")
        
        if failed_tasks:
            print(f"\n📂 FAILED TASKS DETAILS:")
            for task in failed_tasks:
                print(f"  • {task}")

        # Output the manual deletion list
        if all_manual_delete:
            print(f"\n🚨 REQUIRES MANUAL DELETION ({len(all_manual_delete)}):")
            print("  The following files are corrupted but locked by the OS. Delete them manually:")
            for f in all_manual_delete:
                print(f"  • {f}")

        if all_corrupt_inputs:
            print(f"\n🚨 CORRUPT FILES ({len(all_corrupt_inputs)}):")
            for err_str in sorted(all_corrupt_inputs):
                if " (" in err_str:
                    file_path, error_msg = err_str.rsplit(" (", 1)
                    print(f"  • {os.path.basename(file_path)} ({error_msg}")
                else:
                    print(f"  • {os.path.basename(err_str)}")
        print("="*60)
    finally:
        # Close the Log File and restore the terminal
        if logfile:
            print(f"\n🔴 PIPELINE RUN ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
            sys.stdout.close()
            sys.stdout = original_stdout # Give control back to standard Python
    