import os
import sys
import gc
import traceback
import warnings
import concurrent.futures
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
from scipy.stats import skew, kurtosis

from utilities import bootstrap_environment
env = bootstrap_environment(verbose=False)

from utilities import (
    get_filepath, get_prod_files, file_parser, corrupt_file_detector,
    get_period_info,
    generate_logfile_path, get_current_utc_timestamp, get_default_metadata,
    get_lut_metadata, get_source_metadata, get_reference_metadata,
    get_shapefile, get_subarea_geometries,
    get_default_product_dataset, is_log_product
)

"""
Purpose:
    DATAEXTRACTION_UTILITIES is a collection of utility functions for extracting data from gridded files using either coordinates or shapefiles.

Main Functions:
    - build_data_extraction_map: Builds a processing map evaluating file freshness, missing timestamps, and NRT replacement rules.
    
    

Helper Functions:
    
    
Copywrite: 
    Copyright (C) 2026, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on October 19, 2026 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Oct 19, 2026 - KJWH: Initial code written
                         Added clip_data
                         Added calculate_region_statistics
    Oct 21, 2026 - KJWH: Updates after testing
                         Added corrupt_file_detection
                         Now using shapefile functions 
                         Added stat_type parameter
                         Upated output file name and directory
                         Changed timestamp to date
                         Added geometric stats
                         Added ability to work with subareas within a shapefile
                         Updated metadata
                         Now saving the file as a csv (nccsv) with metadata
    Oct 27, 2026 - KJWH: Added functionality to work with fils that have multiple products (e.g. PSC and PP)
                         Completely overhauled variable resolution in process_single_extraction to dynamically loop through variable families, implementing case-insensitive exact matching and smart prefix-stripping for fallbacks.
                         Added logic to automatically strip input statistical suffixes (e.g., _mean, _median) from variable names before appending them to CSV column headers for cleaner formatting.
                         Fixed xarray and rioxarray object referencing errors by correctly applying CRS and spatial dimensions to the Dataset (ds) rather than a DataArray (da) prior to shapefile clipping.
                         Added a standard deviation check (> 1e-6) prior to calculating skewness and kurtosis to permanently suppress SciPy catastrophic cancellation precision warnings when clipping homogenous subareas.
                         Updated comprehensive docstrings outlining purpose, inputs, and returns for build_data_extraction_map, process_single_extraction, and run_extraction_pipeline.

"""

"""
FUNCTIONS to fix
    * search for multiple periods
    * set default periods
    * determine if a file will be saved by period
    * should there be a merged file if there are multiple periods???
    * fix incomplete stats
    * documentation
    * logging steps
    * optional shapefile subregions
"""

# Initialize DualLogger class
class DualLogger:
    """Intercepts standard print statements and writes to both terminal and log file."""
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


def build_data_extraction_map(prod, period, shapefile, map_subset=None, data_type="STATS",stat_type=None, dataset=None, dataset_version=None, 
                              output_dir=None, overwrite=False, verbose=False, **kwargs):
    """
    Purpose:
        Resolves the output directory, finds input files using get_prod_files, and builds a processing map evaluating file freshness, missing timestamps, and NRT replacement rules.
    
    Input Parameters:
        prod (str) - The logical product name to process (e.g., 'SST', 'PSC').
        period (str) - The time period code for the target statistics (e.g., 'W', 'M').
        shapefile (str) - The name or path of the shapefile used for the extraction region.
        map_subset (str, optional) - The spatial region abbreviation (e.g., 'NES'). Defaults to auto-resolving based on product.
        stat_type (str, optional) - The specific statistical suffix (e.g., 'mean'). Auto-detected if not provided.
        dataset (str, optional) - Override for the dataset name.
        dataset_version (str, optional) - Override for the dataset version.
        output_dir (str, optional) - The destination directory for output files. Auto-resolved if None.
        overwrite (bool, optional) - If True, flags all files for reprocessing regardless of existing modification times. Default is False.
        verbose (bool, optional) - If True, prints detailed progress and routing updates. Default is False.
        **kwargs - Additional keyword arguments passed to file discovery functions.

    Returns:
        tuple - (extraction_map, output_file, existing_df, target_var, stat_type) containing the processing dictionary, resolved output CSV path, existing pandas DataFrame, target variable string, and resolved statistic type.
    """

    # Auto-resolve product-specific default map_subset
    if not map_subset:
        map_subset = 'NES' if 'PSC' in prod.upper() else 'NWA'

    # 1. Find Input Files
    if verbose: print(f"🔍 Searching for {period} {data_type} files for {dataset}:{prod} (Grid: {map_subset})")
    input_files = get_prod_files(prod, dataset=dataset, dataset_version=dataset_version, period=period, map_subset=map_subset, data_type=data_type, verbose=verbose)

    if not input_files:
        return {}, None, pd.DataFrame()

    # Auto-detect stat_type if not explicitly provided
    if not stat_type:
        first_file_upper = os.path.basename(input_files[0]).upper()
        if "STATS" in first_file_upper:
            stat_type = 'mean'
        elif "ANOM" in first_file_upper or "ANOMS" in first_file_upper:
            # Route log-normal products to their specific log anomaly variable
            if is_log_product(prod):
                stat_type = 'anom_logdif'
                if verbose: print(f"  ℹ️ Log-normal product detected. Defaulting to '{stat_type}'.")
            else:
                stat_type = 'anom_dif'
        else:
            stat_type = "" # Leave blank for raw daily grids
    # Set target variable and filename safely
    target_var = f"{prod}_{stat_type}" if stat_type else prod
    if verbose: print(f"Using target_var: {target_var}")

    # 2. Parse inputs to extract the exact dataset and version from the physical files
    parsed_inputs = file_parser(input_files)
    first_meta = parsed_inputs[0]
    
    actual_dataset = first_meta.get('dataset', dataset or "UNKNOWN").upper()

    if output_dir is None:
        output_dir = get_filepath(
            prod, 
            data_type="EXTRACT",
            reference_file=input_files[0], 
            region_name=shapefile,
            make_dir=True
        )

    # 4. Define Output File Name
    suffix = f"-{data_type}-EXTRACT" if data_type != "STATS" else "-EXTRACT"
    out_name = f"{period}-{actual_dataset}-{prod}-{shapefile}{suffix}.csv"
    output_file = os.path.join(output_dir, out_name)
    if verbose: print(f"Creating/updating {output_file}")

    # 5. Read Existing Data (Ignoring '#' metadata lines)
    existing_df = pd.DataFrame()
    if os.path.exists(output_file) and not overwrite:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find where the actual data columns begin
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('Date,') or line.startswith('timestamp,'):
                    header_idx = i
                    break
                    
            existing_df = pd.read_csv(output_file, skiprows=header_idx)
        except Exception as e:
            if verbose: print(f"  ⚠️ Could not read existing extraction file '{out_name}': {e}. Forcing full rebuild.")

    # 6. Build Extraction Map
    extraction_map = {}
    up_to_date_count = 0
    nrt_replace_count = 0

    for file_path, meta in zip(input_files, parsed_inputs):
        file_name = os.path.basename(file_path)
        current_mtime = os.path.getmtime(file_path)
        
        is_nrt = meta.get('processing_level') == 'NRT' or file_name.upper().endswith('-NRT.NC')
        timestamp = meta.get('timestamp') or meta.get('start_date')
        
        needs_processing = True
        
        if not existing_df.empty and 'source_file' in existing_df.columns:
            match = existing_df[existing_df['source_file'] == file_name]
            
            if not match.empty:
                recorded_mtime = match.iloc[0].get('input_mtime', 0)
                
                # Compare modification times
                if current_mtime <= recorded_mtime:
                    needs_processing = False
            else:
                # If it's a new RAN file, check if it replaces an old NRT file for the same timestamp
                if not is_nrt and 'Date' in existing_df.columns:
                    nrt_match = existing_df[(existing_df['Date'] == timestamp) & (existing_df['is_nrt'] == True)]
                    if not nrt_match.empty:
                        nrt_replace_count += 1

        if not needs_processing:
            up_to_date_count += 1

        extraction_map[file_path] = {
            "source_file": file_name,
            "full_path": os.path.abspath(file_path),
            "input_mtime": current_mtime,
            "is_nrt": is_nrt,
            "timestamp": timestamp,
            "needs_processing": needs_processing
        }

    needs_update_count = len(input_files) - up_to_date_count
    
    print("-" * 60)
    print(f"📊 EXTRACTION SUMMARY: {prod} | {target_var} | {period} | {shapefile}")
    print(f" Output directory: {output_dir}")
    print(f"📄 Target File: {os.path.basename(output_file)}") 
    print(f"✅ Up-to-date:  {up_to_date_count}/{len(input_files)}")
    print(f"⏳ To process:  {needs_update_count} files")
    if nrt_replace_count > 0: print(f"🔄 NRT replaces: {nrt_replace_count}")
    print("-" * 60)

    return extraction_map, output_file, existing_df, target_var, stat_type

def process_single_extraction(file_path, target_var, shapefile=None, bbox=None, debug=False, **kwargs):
    """
    Purpose:
        Opens a single gridded NetCDF file, resolves target variables (including wildcard matching for families like PSC), clips the data to a specified shapefile or bounding box, calculates comprehensive regional statistics on valid pixels, and returns dictionary records for each subarea.
    
    Input Parameters:
        file_path (str) - The full physical path to the NetCDF file to be processed.
        target_var (str or list) - The specific variable(s) or variable base name(s) to extract.
        shapefile (str, optional) - The path to the shapefile to clip the data against.
        bbox (tuple, optional) - A bounding box (minx, miny, maxx, maxy) to clip the data against.
        debug (bool, optional) - If True, prints extensive troubleshooting and traceback information. Default is False.
        **kwargs - Additional keyword arguments including 'subareas' (list to filter), 'subarea_column' (str), and 'merge_all' (bool).

    Returns:
        tuple - (records, corrupt_files) where records is a list of dictionaries containing calculated statistics per subarea/variable, and corrupt_files is a list of any files that failed to read due to HDF5 corruption.
    """
    ds = None

    subareas = kwargs.get('subareas', None)
    subarea_column = kwargs.get('subarea_column', None)
    merge_all = kwargs.get('merge_all', False) 
    verbose = kwargs.get('verbose',False)
    
    try:
        if debug: print(f"    📖 Extracting from {os.path.basename(file_path)}...")
        
        # 1. Open with xarray
        ds = xr.open_dataset(file_path, engine='h5netcdf')
        
        # Resolve variables to extract
        target_vars = [target_var] if isinstance(target_var, str) else target_var

        actual_vars_to_extract = []
        for tv in target_vars:
            # 1. Case-insensitive exact match check!
            exact_match = [v for v in ds.data_vars if str(v).upper() == tv.upper()]
            if exact_match:
                actual_vars_to_extract.extend(exact_match)
            else:
                # 2. Identify what statistical suffix was requested (e.g. '_mean')
                expected_suffix = ""
                stat_suffixes = ['_mean', '_median', '_max', '_min', '_std', '_sum', '_var', '_count', '_skew', '_kurt', '_anom_logdif', '_anom_ratio', '_anom_dif', '_anom']
                for suff in stat_suffixes:
                    if tv.endswith(suff):
                        expected_suffix = suff
                        break
                        
                # 3. Smart Base Search (strip 'PSC_' prefix if present to avoid breaking on fractions)
                base_search = tv.replace(expected_suffix, '').upper() if expected_suffix else tv.upper()
                if base_search.startswith('PSC_'):
                    base_search = base_search.replace('PSC_', '')
                
                # 4. Match variables in the family that share the SAME statistical suffix
                for v in ds.data_vars:
                    v_str = str(v)
                    if base_search in v_str.upper():
                        if expected_suffix:
                            if v_str.endswith(expected_suffix):
                                actual_vars_to_extract.append(v_str)
                        else:
                            # For daily files (no stat suffix)
                            actual_vars_to_extract.append(v_str)
                
        # Fallback to everything if resolution fails
        if not actual_vars_to_extract:
            actual_vars_to_extract = list(ds.data_vars)

        records = []

        # Helper to calculate stats and append
        # Helper to calculate stats and append
        def append_stats(clipped_ds, subarea_name):
            # Base record for this spatial zone
            stats = {"subarea": subarea_name}
            
            # Loop through all requested variables in the file
            for var_name in actual_vars_to_extract:
                flat_data = clipped_ds[var_name].values.flatten()
                valid_pixels = flat_data[~np.isnan(flat_data)]
                n_valid = valid_pixels.size
                
                # 3. Clean the variable name by removing input stat suffixes for a cleaner CSV header
                clean_var = str(var_name)
                stat_suffixes = ['_mean', '_median', '_max', '_min', '_std', '_sum', '_var', '_count', '_skew', '_kurt', '_anom_logdif', '_anom_ratio', '_anom_dif', '_anom']
                for stat_suffix in stat_suffixes:
                    if clean_var.endswith(stat_suffix):
                        clean_var = clean_var.replace(stat_suffix, '')
                        break
                
                prefix = f"{clean_var}_"
                
                if n_valid == 0:
                    stats.update({
                        f"{prefix}N": 0, f"{prefix}Percent_Valid": 0.0, f"{prefix}Min": np.nan, 
                        f"{prefix}Max": np.nan, f"{prefix}Mean": np.nan, f"{prefix}Median": np.nan, 
                        f"{prefix}Stdev": np.nan, f"{prefix}CV": np.nan, f"{prefix}Geom_Mean": np.nan, 
                        f"{prefix}Geom_Stdev": np.nan, f"{prefix}Skewness": np.nan, f"{prefix}Kurtosis": np.nan,
                        f"{prefix}True_Ratio": np.nan, f"{prefix}Percent_Normal": np.nan
                    })
                else:
                    mean_val = np.mean(valid_pixels)
                    stdev_val = np.std(valid_pixels)
                    
                    # Safely calculate Geometric Stats (requires strictly positive numbers)
                    pos_pixels = valid_pixels[valid_pixels > 0]
                    if pos_pixels.size > 0:
                        log_pos = np.log(pos_pixels)
                        geom_mean = np.exp(np.mean(log_pos))
                        geom_stdev = np.exp(np.std(log_pos))
                    else:
                        geom_mean = np.nan
                        geom_stdev = np.nan
                        
                    # Use update() previous loop variables aren't erased
                    stats.update({
                        f"{prefix}N": n_valid,
                        f"{prefix}Percent_Valid": (n_valid / flat_data.size) * 100,
                        f"{prefix}Min": np.min(valid_pixels),
                        f"{prefix}Max": np.max(valid_pixels),
                        f"{prefix}Mean": mean_val,
                        f"{prefix}Median": np.median(valid_pixels),
                        f"{prefix}Stdev": stdev_val,
                        f"{prefix}CV": (stdev_val / mean_val) * 100 if mean_val != 0 else np.nan,
                        f"{prefix}Geom_Mean": geom_mean,
                        f"{prefix}Geom_Stdev": geom_stdev,
                        f"{prefix}Skewness": float(skew(valid_pixels)) if stdev_val > 1e-6 else np.nan,
                        f"{prefix}Kurtosis": float(kurtosis(valid_pixels)) if stdev_val > 1e-6 else np.nan
                    })
                    # Calculate the physical ratio from the log-difference
                    if str(var_name).endswith('_anom_logdif'):
                        true_ratio = 10 ** mean_val
                        stats.update({
                            f"{prefix}True_Ratio": true_ratio,
                            f"{prefix}Percent_Normal": true_ratio * 100
                        })
                    
            # Append the dictionary after the loop finishes all variables
            records.append(stats)

        # Ensure spatial dimensions are set for rioxarray on the entire Dataset
        if not ds.rio.crs: 
            ds = ds.rio.write_crs("EPSG:4326")
        if 'lon' in ds.coords and 'lat' in ds.coords: 
            ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        
        # Clip Data based on provided input
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            clipped = ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs="EPSG:4326")
            append_stats(clipped, "BBOX")
            
        elif shapefile is not None:
            # Fetch pre-filtered geometries from shapefile_utilities using ds.rio.crs
            subarea_geoms = get_subarea_geometries(shapefile, subarea_column=subarea_column, subareas=subareas, target_crs=ds.rio.crs, merge_all=merge_all, verbose=verbose)
            
            # Iterate through the returned dictionary
            for sa_name, geom in subarea_geoms.items():
                clipped = ds.rio.clip([geom], ds.rio.crs, drop=True) 
                append_stats(clipped, sa_name)

        else:
            # Fallback: Use the entire map_subset domain
            append_stats(ds, "ALL")

        return records, []

    except Exception as e:
        if debug: print(f"❌ ERROR extracting {os.path.basename(file_path)}: {e}") 
        corrupt_files = []
        error_str = str(e).lower()
        if "synchronously read data" in error_str or "filter returned failure" in error_str or "hdf" in error_str:
            corrupt_files.append(f"{file_path} (HDF5 Data Chunk/Filter Read Failure)")
        else:
            # Fallback to standard detector
            bad_files_info = corrupt_file_detector([file_path])
            if bad_files_info:
                for p, err in bad_files_info:
                    corrupt_files.append(f"{p} ({err})")
                    
        return None, corrupt_files
    finally:
        if ds is not None: ds.close()
        gc.collect()

def run_extraction_pipeline(prods=None, periods=None, shapefile=None, bbox=None, stat_type=None, data_type=None, **kwargs):
    """
    Purpose:
        Main orchestrator for extracting time-series regional statistics from gridded files to CSVs. It supports dynamic defaults, hierarchical file fallbacks, parallel execution, and metadata-rich ERDDAP NCCSV formatting.
    
    Input Parameters:
        prods (str or list, optional) - Single product name or list of product names (e.g., 'SST', ['CHL', 'PSC']). Inferred from dataset if None.
        periods (str or list, optional) - Period codes to process (e.g., 'W', ['W', 'M']). Defaults to ['W', 'M'].
        shapefile (str, optional) - The region name or path to a shapefile used for clipping.
        bbox (tuple, optional) - A bounding box (minx, miny, maxx, maxy) for rectangular extraction.
        stat_type (str, optional) - Specific statistical suffix to target (e.g., 'mean', 'median').
        **kwargs - Additional arguments including 'dataset', 'map_subset', 'logfile', 'verbose', 'debug', 'delete_corrupt', and 'parallel_runs'.

    Returns:
        None - Extracted outputs are written directly to disk as aggregated CSV files.
    """
    # 1. Extract global dataset early to establish defaults
    dataset = kwargs.pop('dataset', None)
    map_subset = kwargs.pop('map_subset', 'NWA')

    # Fallback and normalization for data_type
    if not data_type: 
        data_type = ['STATS']
    elif isinstance(data_type, str):
        data_type = [data_type.upper().strip()]
    else:
        data_type = [d.upper().strip() for d in data_type]
        
    # Fallback logic for products
    if not prods:
        if dataset:
            from dataset_utilities import get_default_dataset_product
            prods = [get_default_dataset_product(dataset.upper())]
        else:
            raise ValueError("Must provide either 'prods' or 'dataset' to infer defaults.")

    if isinstance(prods, str): prods = [prods.upper().strip()]
    
    # Fallback and normalization for periods
    if not periods: 
        periods = ['W', 'M']
    elif isinstance(periods, str):
        periods = [periods]

    # Resolve Region Name and Dynamic Shapefile Path
    if shapefile:
        resolved_shapefile = get_shapefile(shapefile, verbose=kwargs.get('verbose', False))
        if not resolved_shapefile:
            raise FileNotFoundError(f"❌ Could not resolve shapefile: {shapefile}")
        shapefile = resolved_shapefile
        region_name = Path(shapefile).stem
        extraction_type = "shapefile"
    elif bbox:
        region_name = kwargs.get('region_name', "BBOX")
        extraction_type = "bounding box"
    else:
        # Fallback to the full map_subset domain unless it is GLOBAL
        if map_subset == "GLOBAL":
            raise ValueError("❌ Refusing to extract full GLOBAL domain statistics. Please provide a shapefile, bbox, or a specific map_subset.")
        region_name = map_subset
        extraction_type = f"full {map_subset} domain"

    # Handle logfile auto-generation
    logfile = kwargs.pop('logfile', None)
    if logfile in ["auto", True]:
        logfile = generate_logfile_path("extract", prods=prods, dataset=dataset, env_dict=env)    

    original_stdout = sys.stdout 

    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        sys.stdout = DualLogger(logfile)
        print("\n" + "="*80)
        print(f"🟢 DATA EXTRACTION PIPELINE RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 Logging to: {logfile}")
        print("="*80)

    try:
        verbose = kwargs.pop('verbose', False)
        debug   = kwargs.get('debug', False)
        delete_corrupt = kwargs.get('delete_corrupt', False)
        should_log = verbose or bool(logfile)

        if not debug:
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            np.seterr(all='ignore')

        parallel_runs = kwargs.pop('parallel_runs', 1) 
        
        global_successful = 0
        global_failed = 0
        all_corrupt_inputs = set()
        updated_output_files = set()

        for prod in prods:
            # 4. Dynamic Dataset Resolution Per Product
            target_var = f"{prod}_{stat_type}"
            current_dataset = dataset
            if not current_dataset:
                try:
                    current_dataset = get_default_product_dataset(prod)
                except KeyError:
                    if should_log: print(f"\n⚠️ No default dataset found for product '{prod}'. Skipping.")
                    continue

            for period in periods:
                for dt in data_type:
                    # Sanitize explicit ANOM request for accurate file discovery
                    search_dt = "ANOMS" if dt in ["ANOM", "ANOMALY", "ANOMALIES"] else dt
                    
                    if should_log: print(f"\n🚀 Starting {search_dt} extraction for {prod} (Dataset: {current_dataset}) | Period: {period}")
                
                    try:
                        p_info = get_period_info(period)
                        p_code = period
                        p_desc = p_info.get('description', f"{period} period")
                    except:
                        p_code = period
                        p_desc = f"{period} period"
                    
                    # Catch target_var and resolved_stat_type from the map builder
                    extraction_map, output_file, existing_df, target_var, resolved_stat_type = build_data_extraction_map(
                        prod, period, region_name, dataset=current_dataset, map_subset=map_subset, verbose=verbose, stat_type=stat_type, data_type=data_type, **kwargs
                    )
                    
                    tasks_to_run = {k: v for k, v in extraction_map.items() if v['needs_processing']}
                    if not tasks_to_run:
                        continue

                    new_records = []
                    successful, failed = 0, 0
                    completed_tasks = 0

                    # --- 🎯 PARALLEL EXTRACTION ---
                    if parallel_runs > 1:
                        if should_log: print(f"  ⚡ Launching {len(tasks_to_run)} extractions across {parallel_runs} workers...")
                        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_runs) as executor:
                            future_to_file = {
                                # Pass explicit shapefile and bbox args
                                executor.submit(process_single_extraction, file_path, target_var, shapefile, bbox,**kwargs, verbose=verbose): file_path
                                for file_path in tasks_to_run.keys()
                            }
                            
                            for future in concurrent.futures.as_completed(future_to_file):
                                file_path = future_to_file[future]
                                info = tasks_to_run[file_path]
                                completed_tasks += 1
                                if should_log: print(f"    [{completed_tasks}/{len(tasks_to_run)}] Finished: {info['source_file']}")
                                try:
                                    stats, corrupt_files = future.result()
                                    if corrupt_files: all_corrupt_inputs.update(corrupt_files)
                                    
                                    if stats:
                                        for single_stats in stats:
                                            record = {
                                                "Date": info['timestamp'],
                                                "period_code": p_code,
                                                "period_descrption": p_desc,
                                                "source_file": info['source_file'],
                                                "input_mtime": info['input_mtime'],
                                                "is_nrt": info['is_nrt'],
                                                **single_stats
                                            }
                                            new_records.append(record)
                                        successful += 1
                                    else:
                                        failed += 1
                                except Exception as e:
                                    failed += 1
                                    if should_log: print(f"❌ [Parallel] Exception on {info['source_file']}: {e}")
                    
                    # --- SEQUENTIAL EXTRACTION ---
                    else:
                        for file_path, info in tasks_to_run.items():
                            completed_tasks += 1
                            if should_log: print(f"    [{completed_tasks}/{len(tasks_to_run)}] Processing: {info['source_file']}")
                            # Pass explicit shapefile and bbox args
                            stats, corrupt_files = process_single_extraction(file_path, target_var, shapefile, bbox, verbose=verbose, **kwargs)
                            if corrupt_files: all_corrupt_inputs.update(corrupt_files)
                            
                            if stats:
                                for single_stats in stats:
                                    record = {
                                        "Date": info['timestamp'],
                                        "period_code": p_code,
                                        "period_descrption": p_desc,
                                        "source_file": info['source_file'],
                                        "input_mtime": info['input_mtime'],
                                        "is_nrt": info['is_nrt'],
                                        **single_stats
                                    }
                                    new_records.append(record)
                                successful += 1
                            else:
                                failed += 1

                    global_successful += successful
                    global_failed += failed
                    
                    # --- SEQUENTIAL WRITE / MERGE ---
                    if new_records:
                        if should_log: print(f"  💾 Merging {len(new_records)} new records into master dataset...")
                        new_df = pd.DataFrame(new_records)
                        
                        if not existing_df.empty:
                            # Drop obsolete files (updated RANs replacing NRTs, or replaced source files)
                            updated_files = set(new_df['source_file'])
                            updated_timestamps = set(new_df['Date'].dropna())

                            existing_df = existing_df[
                                ~existing_df['source_file'].isin(updated_files) &
                                ~((existing_df['is_nrt'] == True) & (existing_df['Date'].isin(updated_timestamps)))
                            ]
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        else:
                            combined_df = new_df

                        # Sort chronologically (and by subarea if present)
                        if 'Date' in combined_df.columns:
                            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
                            sort_cols = ['Date', 'subarea'] if 'subarea' in combined_df.columns else ['Date']
                            combined_df = combined_df.sort_values(by=sort_cols).reset_index(drop=True)

                            # Format Date explicitly for ERDDAP
                            combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')

                        # Assemble global metadata
                        attrs = get_default_metadata(sheet="General") | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
                        stat_label = f" ({resolved_stat_type})" if resolved_stat_type else ""
                        attrs['title'] = f"Extracted data for {prod} {stat_label} - {region_name}"
                        attrs['product'] = prod
                        if resolved_stat_type:attrs['statistic_type'] = resolved_stat_type
                        attrs['region'] = region_name
                        attrs['Conventions'] = "COARDS, CF-1.6, ACDD-1.3, NCCSV-1.2"
                        attrs['history'] = f"{get_current_utc_timestamp()} Extracted regional time-series using {extraction_type}."

                        # Dynamic Temporal Details
                        if 'Date' in combined_df.columns:
                            dates = pd.to_datetime(combined_df['Date'])
                            attrs['time_coverage_start'] = dates.min().strftime('%Y-%m-%dT%H:%M:%SZ')
                            attrs['time_coverage_end'] = dates.max().strftime('%Y-%m-%dT%H:%M:%SZ')
                        
                        # Shapefile and Subarea details
                        if shapefile:
                            attrs['shapefile_source'] = str(os.path.basename(shapefile))
                        if 'subarea' in combined_df.columns:
                            attrs['subareas_included'] = ", ".join(combined_df['subarea'].dropna().unique().astype(str))
                        
                        try:
                            attrs = attrs | get_source_metadata(current_dataset, source_prefix="source")
                            attrs['references'] = get_reference_metadata(prod, refs_only=True)
                        except: pass

                        # Write ERDDAP NCCSV format
                        with open(output_file, 'w', encoding='utf-8') as f:
                            # 1. Write *GLOBAL* attributes
                            for k, v in attrs.items():
                                clean_v = str(v).replace('"', '""').replace('\n', ' | ')
                                f.write(f'*GLOBAL*,{k},"{clean_v}"\n')
                            
                            # 2. Write Variable attributes
                            f.write('Date,*DATA_TYPE*,String\n')
                            f.write('Date,standard_name,time\n')
                            f.write('period_code,*DATA_TYPE*,String\n')
                            f.write('period_desc,*DATA_TYPE*,String\n')
                            
                            if 'subarea' in combined_df.columns:
                                f.write('subarea,*DATA_TYPE*,String\n')
                            
                            # Declare stats variables as floats
                            for col in ['N', 'Percent_Valid', 'Min', 'Max', 'Mean', 'Median', 'Stdev', 'CV', 'Geom_Mean', 'Geom_Stdev', 'Skewness', 'Kurtosis', 'True_Ratio', 'Percent_Normal']:
                                if col in combined_df.columns:
                                    f.write(f'{col},*DATA_TYPE*,Float\n')
                        
                        # Append the dataframe below the header
                        combined_df.to_csv(output_file, mode='a', index=False)
                        
                        if should_log: print(f"  ✅ Saved/Updated: {os.path.basename(output_file)}")
                        updated_output_files.add(os.path.basename(output_file)) 

        # --- FINAL SUMMARY REPORT ---
        print("\n" + "="*60)
        print("🏁 DATA EXTRACTION FINISHED SUMMARY")
        print("="*60)
        print(f"✅ Extracted Records: {global_successful}")
        print(f"❌ Failed Files:      {global_failed}")

        # 🎯 Print the list of updated files
        if updated_output_files:
            print(f"\n💾 FILES SAVED/UPDATED ({len(updated_output_files)}):")
            for f in sorted(updated_output_files):
                print(f"  • {f}")

        if all_corrupt_inputs:
            print(f"\n🚨 CORRUPT SOURCE FILES ({len(all_corrupt_inputs)}):")
            for err_str in sorted(all_corrupt_inputs):
                if " (" in err_str:
                    file_path, error_msg = err_str.rsplit(" (", 1)
                    print(f"  • {os.path.basename(file_path)} ({error_msg}")
                    
                    # Execute deletion if requested
                    if delete_corrupt:
                        try:
                            os.remove(file_path)
                            print(f"    🗑️ -> DELETED")
                        except Exception as e:
                            print(f"    ⚠️ -> Failed to delete: {e}")
                else:
                    print(f"  • {os.path.basename(err_str)}")
        print("="*60)

    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ CRITICAL PIPELINE ERROR: {e}")
        if debug: traceback.print_exc()
            
    finally:
        if logfile and logfile not in ["auto", True]:
            print(f"\n🔴 PIPELINE RUN ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
            sys.stdout.close()
            sys.stdout = original_stdout       

