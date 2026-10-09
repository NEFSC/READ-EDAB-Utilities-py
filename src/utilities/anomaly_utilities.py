import os
import sys
import re
import xarray as xr
import numpy as np
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
from utilities import get_period_info, get_period_sets, get_clim_period, get_prod_files, get_filepath
from utilities import get_nc_prod, subset_dataset, file_parser, corrupt_file_detector
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
    if platform.system() != 'Windows':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_limit = min(4096, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))

def batch_anomaly_match(input_files, clim_files):
    """
    Takes a list of input files and matches them to climatology files.
    Returns a dictionary: {input_file: matched_clim_file}
    """
    matched_pairs = {}
    
    for input_file in input_files:
        # file_parser returns a list of dictionaries; grab the first one
        parsed_data = file_parser(input_file)[0]
        
        # Navigate the nested dictionary to find the temporal resolution
        metadata = parsed_data['period_details']['metadata']
        groupby = metadata.get('groupby')  # e.g., 'month'
        
        # Extract the date string (e.g., '20000201')
        start_date = parsed_data['start_date']
        
        # Construct the target climatology string
        if groupby == 'month':
            # Slice characters 4 and 5 from 'YYYYMMDD' to get the 2-digit month
            month = start_date[4:6] 
            target_period = f"MONTH_{month}"
            
        elif groupby == 'week':
            week = start_date[4:6]
            # Adjust slicing or dictionary keys based on how week is formatted
            target_period = f"WEEK_{week}" 
            
        elif groupby in ['annual']:
            target_period = "ANNUAL"

        elif groupby in ['year']:
            target_period = "YEAR"
            
        else:
            print(f"Skipping {input_file}: Unsupported groupby '{groupby}'")
            continue

        # Scan the climatology files for the target string
        match_found = False
        for clim_file in clim_files:
            if target_period in clim_file:
                matched_pairs[input_file] = clim_file
                match_found = True
                break  # Stop searching once a match is found
                
        if not match_found:
            print(f"Warning: No climatology match found for '{target_period}' ({input_file})")
            
    return matched_pairs

def build_anomaly_map(prod, period, output_dir=None,
                      dataset=None, daterange=None, climatology_range=None,
                      dataset_version=None, dataset_map=None,
                      map_subset=None, overwrite=False, 
                      data_type="ANOM", verbose=False, debug=False):
    """
    Purpose:
        Orchestrates the preparation phase of the anomaly pipeline by pairing 
        statistical files with their corresponding climatology files and evaluating freshness (mtimes).
    """
    
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
        return {}

    fp = file_parser(stat_files[0])
    ds_name = fp[0].get('dataset', dataset or 'UNKNOWN') 

    # 3. Pair Stats with Climatologies based on time period mapping
    period_sets = get_period_sets(period, files=stat_files, daterange=daterange)
    pm = period_sets.get("period_map", {})

    anomaly_map = {}
    total_tasks = len(pm)
    up_to_date_count = 0

    for out_period, info in pm.items():
        stat_inputs = info["input_files"]
        
        # Helper logic to find the exact matching climatology file for this period
        # (Assuming file_parser or a regex can match the period (e.g. Month 05) to the Climatology file)
        target_clim_file = next((f for f in clim_files if out_period.split('_')[-1] in f or out_period in f), None)
        
        if not target_clim_file:
            if verbose: print(f"⚠️ No matching climatology found for {out_period}. Skipping.")
            continue

        out_name = f"{out_period}-{ds_name}-{prod}-{map_subset or 'GLOBAL'}-{data_type}.nc"
        out_path = os.path.join(output_dir, out_name)

        is_up_to_date = False
        all_inputs = stat_inputs + [target_clim_file]

        if os.path.exists(out_path) and not overwrite:
            out_mtime = os.path.getmtime(out_path)
            if all(out_mtime > os.path.getmtime(f) for f in all_inputs):
                try:
                    with xr.open_dataset(out_path, engine='netcdf4', decode_times=False, decode_coords=False) as existing_ds:
                        stored_files_str = existing_ds.attrs.get('source_files', '')
                    if stored_files_str:
                        stored_basenames = set(f.strip() for f in stored_files_str.split(','))
                        current_basenames = set(os.path.basename(f) for f in all_inputs)
                        is_up_to_date = (stored_basenames == current_basenames)
                except Exception as e:
                    is_up_to_date = False

        if is_up_to_date:
            up_to_date_count += 1
            
        anomaly_map[out_period] = {
            "stat_inputs": stat_inputs,
            "clim_input": target_clim_file,
            "output": out_path,
            "is_up_to_date": is_up_to_date
        }

    needs_update_count = len(anomaly_map) - up_to_date_count
    print("-" * 60)
    print(f"📊 ANOMALY SUMMARY: {prod} | {period} | {map_subset or 'GLOBAL'}")
    print(f"✅ Up-to-date:  {up_to_date_count}/{len(anomaly_map)}")
    print(f"⏳ To process:  {needs_update_count} files")
    print("-" * 60)
    
    return anomaly_map


def compute_anomalies(stat_ds, clim_ds):
    """
    Purpose:
        Subtracts climatological dataset from the statistical dataset.
    """
    results = {}
    
    # Iterate through data variables in the stat dataset
    for var in stat_ds.data_vars:
        if var in clim_ds.data_vars:
            # Perform standard anomaly calculation
            anom_da = stat_ds[var] - clim_ds[var]
            
            # Rename for output formatting (e.g. SST_mean -> SST_anom)
            base_var = var.replace('_mean', '')
            out_var_name = f"{base_var}_anom"
            results[out_var_name] = anom_da
        else:
            print(f"⚠️ Climatology variable matching '{var}' not found. Skipping anomaly calculation.")

    return xr.Dataset(results)


def process_single_anomaly(task, prod, per, verbose, **kwargs):
    """
    Purpose:
        Processes a single anomaly task: opens stat and clim files,
        computes anomalies, applies metadata, and saves to NetCDF.
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
    out_path = task['output']
    stat_files = task['stat_inputs']
    clim_file = task['clim_input']
    corrupt_files_found = []
    
    ds_stat, ds_clim, anom_ds = None, None, None

    try:
        if debug: print(f"      📖 Opening stats and climatology inputs...")
        
        # Load datasets
        ds_stat = xr.open_mfdataset(stat_files, combine="by_coords", engine='h5netcdf', parallel=False)
        ds_clim = xr.open_dataset(clim_file, engine='h5netcdf')

        # Compute Anomalies
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with dask.config.set(scheduler='single-threaded'):
                anom_ds = compute_anomalies(ds_stat, ds_clim)
                anom_ds = anom_ds.load()

        if not anom_ds.data_vars:
            raise ValueError(f"❌ No valid variables computed for anomaly {out_path}.")

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
            
            # Attributes mapping
            base_var_name = out_var.replace("_anom", "")
            anom_ds[out_var].attrs = build_stat_attributes(base_var_name, "anomaly", _FillValue=fill_value)

        # Retain original time coordinates if lost during math
        if 'time' not in anom_ds.coords and 'time' in ds_stat.coords:
            anom_ds = anom_ds.assign_coords(time=ds_stat.time)

        # Global Metadata
        attrs = get_default_metadata(sheet="General") | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
        attrs = attrs | get_geospatial_metadata(dataset=anom_ds) | get_temporal_metadata(ds=anom_ds)

        try:
            base_prod_attrs = build_product_attributes(prod)
            attrs["product_name"] = f"{base_prod_attrs.get('long_name', prod)} Anomaly"
        except:
            attrs["product_name"] = f"{prod} Anomaly"

        all_input_basenames = [os.path.basename(f) for f in stat_files] + [os.path.basename(clim_file)]
        attrs['source_files'] = ", ".join(all_input_basenames)
        attrs["history"] = f"{get_current_utc_timestamp()} Computed {prod} anomalies against climatological baselines using xarray."

        try:
            ds_name = kwargs.get('dataset', 'UNKNOWN')
            attrs = attrs | get_source_metadata(ds_name, source_prefix="source")
        except:
            pass

        anom_ds.attrs = attrs
        anom_ds = sanitize_for_netcdf(anom_ds)
        
        # Save Output
        anom_ds.to_netcdf(out_path, format='NETCDF4', encoding=encoding)
        if verbose: print(f"  ✅ Saved: {os.path.basename(out_path)}")
        return True, corrupt_files_found
        
    except Exception as e:
        print(f"❌ Error processing {out_path}: {e}")
        bad_files_info = corrupt_file_detector(stat_files + [clim_file])
        if bad_files_info:
            corrupt_files_found.extend([f"{p} ({err})" for p, err in bad_files_info])
            
        if debug: traceback.print_exc()
        return False, corrupt_files_found

    finally:
        for ds in [ds_stat, ds_clim, anom_ds]:
            try:
                if ds is not None: ds.close()
            except: pass
        gc.collect()


def run_anomaly_pipeline(prods, periods=None, **kwargs):
    """
    Purpose:
        Executes a multi-product, multi-period anomaly calculation pipeline.
    """
    logfile = kwargs.pop('logfile', None)
    original_stdout = sys.stdout 

    if logfile:
        sys.stdout = DualLogger(logfile)
        print("\n" + "="*80)
        print(f"🟢 ANOMALY PIPELINE RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

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

        successful_tasks, failed_tasks, skipped_tasks = [], [], []
        all_corrupt_inputs = set()
                
        for prod in prods:
            for period in periods:
                if verbose: print(f"\n🚀 Starting anomaly pipeline for {prod} | Period: {period}")
                
                anomaly_map = build_anomaly_map(prod, period, verbose=verbose, **kwargs)
                if not anomaly_map: continue
                
                tasks_to_run = []
                for out_period, task in anomaly_map.items():
                    if task['is_up_to_date']:
                        skipped_tasks.append(out_period)
                    else:
                        tasks_to_run.append((out_period, task))

                if not tasks_to_run: continue

                if parallel_runs == 1:
                    for out_period, task in tasks_to_run:
                        success, corrupt_files = process_single_anomaly(task, prod, period, verbose, **kwargs)
                        if corrupt_files: all_corrupt_inputs.update(corrupt_files)
                        if success: successful_tasks.append(out_period)
                        else: failed_tasks.append(out_period)
                else:
                    if verbose: print(f"\n  ⚡ Launching {len(tasks_to_run)} parallel tasks...")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_runs) as executor:
                        future_to_period = {
                            executor.submit(process_single_anomaly, task, prod, period, verbose, **kwargs): out_period
                            for out_period, task in tasks_to_run
                        }
                        for future in concurrent.futures.as_completed(future_to_period):
                            out_period = future_to_period[future]
                            try:
                                success, corrupt_files = future.result()
                                if corrupt_files: all_corrupt_inputs.update(corrupt_files)                                
                                if success: successful_tasks.append(out_period)
                                else: failed_tasks.append(out_period)
                            except Exception as e:
                                failed_tasks.append(f"{out_period}: {str(e)}")

        # Final Summary
        print("\n" + "="*60)
        print("🏁 ANOMALY PIPELINE FINISHED")
        print(f"✅ Processed: {len(successful_tasks)} | ⏭ Skipped: {len(skipped_tasks)} | ❌ Failed: {len(failed_tasks)}")
        
        if failed_tasks:
            print(f"\n📂 FAILED TASKS:\n" + "\n".join([f"  • {task}" for task in failed_tasks]))
        if all_corrupt_inputs:
            print(f"\n🚨 CORRUPT FILES ({len(all_corrupt_inputs)}):\n" + "\n".join([f"  • {os.path.basename(err)}" for err in sorted(all_corrupt_inputs)]))
        print("="*60)

    finally:
        if logfile:
            print(f"\n🔴 ANOMALY PIPELINE ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout.close()
            sys.stdout = original_stdout