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

from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

from utilities import (
    get_filepath, get_prod_files, file_parser, corrupt_file_detector,
    generate_logfile_path, get_current_utc_timestamp, get_default_metadata,
    get_lut_metadata, get_source_metadata, get_reference_metadata,
    get_shapefile
)

"""
Purpose:
    DATAEXTRACTION_UTILITIES is a collection of utility functions for extracting data from gridded files using either coordinates or shapefiles.

Main Functions:
    - build_data_extraction_map: Builds a processing map evaluating file freshness, missing timestamps, and NRT replacement rules.
    - clip_data: Clips the data to a specified region
    - calculate_region_statistics: Computes the requested statistics on the valid pixels of the clipped data array
    - extract_gridded_statistics: 
    

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
    Oct 21, 2026 - KJWH: 

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

def build_data_extraction_map(prod, period, region_name, dataset=None, dataset_version=None, output_dir=None, overwrite=False, verbose=False, **kwargs):
    """
    Purpose:
        Resolves the output directory, finds input files using get_prod_files, and builds a processing map evaluating file freshness and NRT replacement rules.
    
    Input Parameters:

    Returns:
        tuple: (extraction_map, output_file_path, existing_dataframe)
    """

    """
    
    STEPS
    1) find the files
    2) get the dataset
    3) establish the output dir (make if needed)
    4) create the output file name
    5) search for file
    6) open if available
    7) compare input and output file names
    8) create summaries
    9) sort by date
    """

    # 1. Resolve Output Directory
    if output_dir is None:
        # Assuming EXTRACT outputs live alongside STATS/ANOMALY in the PRODUCTS tree
        output_dir = get_filepath(
            prod, dataset=dataset, dataset_version=dataset_version,
            map_subset=region_name, period=period, data_type="EXTRACT",
            getfilepath=True, make_dir=True, verbose=verbose
        )
        if isinstance(output_dir, list): output_dir = output_dir[0]

    if not output_dir:
        raise ValueError(f"🛑 CRITICAL: Could not resolve output directory for {prod} ({period}).")

    # Define the compiled time-series output file
    out_name = f"{period}-{region_name}-{dataset}-{prod}-EXTRACT.nc"
    output_file = os.path.join(output_dir, out_name)


    # 2. Find Input Files
    if verbose: print(f"🔍 Searching for {period} files for {dataset}:{prod}")
    input_files = get_prod_files(prod, dataset=dataset, dataset_version=dataset_version, period=period, map_subset=map_subset, verbose=verbose)
    
    if not input_files:
        return {}, output_file, pd.DataFrame()

    # 3. Read Existing Data (to evaluate freshness/NRT replacement)
    existing_df = pd.DataFrame()
    if os.path.exists(output_file) and not overwrite:
        try:
            with xr.open_dataset(output_file, engine='h5netcdf') as existing_ds:
                existing_df = existing_ds.to_dataframe().reset_index()
        except Exception as e:
            if verbose: print(f"  ⚠️ Could not read existing extraction file '{out_name}': {e}. Forcing full rebuild.")

    # 4. Build Extraction Map
    extraction_map = {}
    parsed_inputs = file_parser(input_files)
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
                if not is_nrt and 'timestamp' in existing_df.columns:
                    nrt_match = existing_df[(existing_df['timestamp'] == timestamp) & (existing_df['is_nrt'] == True)]
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
    print(f"📊 EXTRACTION SUMMARY: {prod} | {period} | {region_name}")
    print(f"✅ Up-to-date:  {up_to_date_count}/{len(input_files)}")
    print(f"⏳ To process:  {needs_update_count} files")
    if nrt_replace_count > 0: print(f"🔄 NRT replaces: {nrt_replace_count}")
    print("-" * 60)

    return extraction_map, output_file, existing_df


def process_single_extraction(file_path, prod, shapefile=None, bbox=None, debug=False):
    """
    Purpose:
        Opens a single gridded file, clips it to the specified region/coordinates, 
        calculates statistics on the valid pixels, and returns a dictionary record.
    """
    ds = None
    try:
        if debug: print(f"    📖 Extracting from {os.path.basename(file_path)}...")
        
        # 1. Open with xarray
        ds = xr.open_dataset(file_path, engine='h5netcdf')
        
        # Determine variable to extract
        target_var = prod if prod in ds.data_vars else [v for v in ds.data_vars if prod in v.upper()][0]
        da = ds[target_var]

        # Ensure spatial dimensions are set for rioxarray
        if not da.rio.crs: da = da.rio.write_crs("EPSG:4326")
        if 'lon' in da.coords and 'lat' in da.coords: da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

        records = []

        # Helper to calculate stats and append
        def append_stats(clipped_da, subarea_name):
            flat_data = clipped_da.values.flatten()
            valid_pixels = flat_data[~np.isnan(flat_data)]
            n_valid = valid_pixels.size
            
            if n_valid == 0:
                stats = {"N": 0, "Percent_Valid": 0.0, "Min": np.nan, "Max": np.nan, 
                         "Mean": np.nan, "Median": np.nan, "Stdev": np.nan, "CV": np.nan}
            else:
                mean_val = np.mean(valid_pixels)
                stdev_val = np.std(valid_pixels)
                stats = {
                    "N": n_valid,
                    "Percent_Valid": (n_valid / flat_data.size) * 100,
                    "Min": np.min(valid_pixels),
                    "Max": np.max(valid_pixels),
                    "Mean": mean_val,
                    "Median": np.median(valid_pixels),
                    "Stdev": stdev_val,
                    "CV": (stdev_val / mean_val) * 100 if mean_val != 0 else np.nan
                }
            stats['subarea'] = subarea_name
            records.append(stats)

        # 🎯 Clip Data based on provided input
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs="EPSG:4326")
            append_stats(clipped, "BBOX")
            
        elif shapefile is not None:
            gdf = gpd.read_file(shapefile)
            if gdf.crs != da.rio.crs:
                gdf = gdf.to_crs(da.rio.crs)
                
            # Iterate through Subareas
            if subarea_column and subarea_column in gdf.columns:
                target_subareas = subareas if subareas else gdf[subarea_column].unique()
                for sa in target_subareas:
                    sa_gdf = gdf[gdf[subarea_column] == sa]
                    if sa_gdf.empty: continue
                    clipped = da.rio.clip(sa_gdf.geometry, da.rio.crs, drop=True)
                    append_stats(clipped, str(sa))
            else:
                # Extract the entire shapefile as one polygon
                clipped = da.rio.clip(gdf.geometry, da.rio.crs, drop=True)
                append_stats(clipped, "ALL")
        else:
            # Fallback: Use the entire map_subset domain
            append_stats(da, "ALL")

        return records, []

    except Exception as e:
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

def run_extraction_pipeline(prods=None, periods=None, shapefile=None, bbox=None, subareas=None, **kwargs):
    """
    Purpose:
        Main orchestrator for extracting time-series regional statistics. 
        It supports dynamic defaults, parallel extraction, and robust HDF5 integration.
    """
    # 1. Extract global dataset early to establish defaults
    dataset = kwargs.pop('dataset', None)
    
    # 2. Fallback logic for products
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

    # Resolve map_subset for finding input files
    map_subset = kwargs.get('map_subset', 'NWA')

    # Resolve Region Name and Dynamic Shapefile Path
    if shapefile:
        resolved_shapefile = get_shapefile(shapefile, env, verbose=kwargs.get('verbose', False))
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

        for prod in prods:
            # 🎯 4. Dynamic Dataset Resolution Per Product
            current_dataset = dataset
            if not current_dataset:
                from product_utilities import get_default_product_dataset
                try:
                    current_dataset = get_default_product_dataset(prod)
                except KeyError:
                    if should_log: print(f"\n⚠️ No default dataset found for product '{prod}'. Skipping.")
                    continue

            for period in periods:
                if should_log: print(f"\n🚀 Starting extraction for {prod} (Dataset: {current_dataset}) | Period: {period}")
                
                extraction_map, output_file, existing_df = build_data_extraction_map(
                    prod, period, region_name, dataset=current_dataset, map_subset=map_subset, verbose=verbose, **kwargs
                )
                
                tasks_to_run = {k: v for k, v in extraction_map.items() if v['needs_processing']}
                if not tasks_to_run:
                    continue

                new_records = []
                successful, failed = 0, 0

                # --- 🎯 PARALLEL EXTRACTION ---
                if parallel_runs > 1:
                    if should_log: print(f"  ⚡ Launching {len(tasks_to_run)} extractions across {parallel_runs} workers...")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_runs) as executor:
                        future_to_file = {
                            # Pass explicit shapefile and bbox args
                            executor.submit(process_single_extraction, file_path, prod, shapefile, bbox, debug): file_path
                            for file_path in tasks_to_run.keys()
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_file):
                            file_path = future_to_file[future]
                            info = tasks_to_run[file_path]
                            try:
                                stats, corrupt_files = future.result()
                                if corrupt_files: all_corrupt_inputs.update(corrupt_files)
                                
                                if stats:
                                    record = {
                                        "timestamp": info['timestamp'],
                                        "source_file": info['source_file'],
                                        "input_mtime": info['input_mtime'],
                                        "is_nrt": info['is_nrt'],
                                        **stats
                                    }
                                    new_records.append(record)
                                    successful += 1
                                else:
                                    failed += 1
                            except Exception as e:
                                failed += 1
                                if debug: print(f"❌ [Parallel] Exception on {info['source_file']}: {e}")
                
                # --- SEQUENTIAL EXTRACTION ---
                else:
                    for file_path, info in tasks_to_run.items():
                        # Pass explicit shapefile and bbox args
                        stats, corrupt_files = process_single_extraction(file_path, prod, shapefile, bbox, debug)
                        if corrupt_files: all_corrupt_inputs.update(corrupt_files)
                        
                        if stats:
                            record = {
                                "timestamp": info['timestamp'],
                                "source_file": info['source_file'],
                                "input_mtime": info['input_mtime'],
                                "is_nrt": info['is_nrt'],
                                **stats
                            }
                            new_records.append(record)
                            successful += 1
                        else:
                            failed += 1

                global_successful += successful
                global_failed += failed
                
                # --- 🎯 SEQUENTIAL WRITE / MERGE ---
                if new_records:
                    if should_log: print(f"  💾 Merging {len(new_records)} new records into master dataset...")
                    new_df = pd.DataFrame(new_records)
                    
                    if not existing_df.empty:
                        # Drop obsolete files (updated RANs replacing NRTs, or replaced source files)
                        updated_files = set(new_df['source_file'])
                        updated_timestamps = set(new_df['timestamp'].dropna())

                        existing_df = existing_df[
                            ~existing_df['source_file'].isin(updated_files) &
                            ~((existing_df['is_nrt'] == True) & (existing_df['timestamp'].isin(updated_timestamps)))
                        ]
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        combined_df = new_df

                    # Sort chronologically
                    if 'timestamp' in combined_df.columns:
                        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                        combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)

                    # Convert to xarray dataset
                    final_ds = combined_df.set_index('timestamp').to_xarray() if 'timestamp' in combined_df.columns else combined_df.to_xarray()
                    
                    # Apply global metadata
                    attrs = get_default_metadata(sheet="General") | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
                    attrs['title'] = f"Regional Statistics for {prod} - {region_name}"
                    attrs['product'] = prod
                    attrs['region'] = region_name
                    attrs['history'] = f"{get_current_utc_timestamp()} Extracted regional time-series using {extraction_type}."                    
                    
                    try:
                        attrs = attrs | get_source_metadata(current_dataset, source_prefix="source")
                        attrs['references'] = get_reference_metadata(prod, refs_only=True)
                    except: pass
                    
                    final_ds.attrs = attrs

                    # Scrub for NetCDF compliance and write to disk
                    for var in final_ds.variables:
                        for k, val in list(final_ds[var].attrs.items()):
                            if isinstance(val, dict): final_ds[var].attrs[k] = str(val)
                            elif val is None: final_ds[var].attrs[k] = "None"
                            
                    final_ds.to_netcdf(output_file, format='NETCDF4', engine='h5netcdf')
                    if should_log: print(f"  ✅ Updated: {os.path.basename(output_file)}")

        # --- FINAL SUMMARY REPORT ---
        print("\n" + "="*60)
        print("🏁 DATA EXTRACTION FINISHED SUMMARY")
        print("="*60)
        print(f"✅ Extracted Records: {global_successful}")
        print(f"❌ Failed Files:      {global_failed}")

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

def load_and_clip_data(file_path, region_input, input_type="coords", crs="EPSG:4326"):
    """
    Loads gridded data and clips it to a specified region.
    """
    # Load the dataset and ensure it has a spatial dimension
    da = rioxarray.open_rasterio(file_path, masked=True)
    
    # Handle Coordinate Bounding Box (minx, miny, maxx, maxy)
    if input_type == "coords":
        minx, miny, maxx, maxy = region_input
        clipped = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=crs)
        
    # Handle Shapefile / GeoJSON
    elif input_type == "shapefile":
        gdf = gpd.read_file(region_input)
        # Reproject shapefile to match raster CRS if necessary
        if gdf.crs != da.rio.crs:
            gdf = gdf.to_crs(da.rio.crs)
        clipped = da.rio.clip(gdf.geometry, da.rio.crs, drop=True)
        
    else:
        raise ValueError("input_type must be 'coords' or 'shapefile'")
        
    return clipped

def calculate_region_statistics(clipped_da):
    """
    Computes the requested statistics on the valid pixels of the clipped data array.
    """
    # Flatten the array and remove NaNs/missing values for clean numpy calculations
    flat_data = clipped_da.values.flatten()
    valid_pixels = flat_data[~np.isnan(flat_data)]
    
    total_pixels = flat_data.size
    n_valid = valid_pixels.size
    
    if n_valid == 0:
        return {"N": 0, "Percent Valid": 0.0, "Min": np.nan, "Max": np.nan, 
                "Mean": np.nan, "Median": np.nan, "Stdev": np.nan, "CV": np.nan}

    mean_val = np.mean(valid_pixels)
    stdev_val = np.std(valid_pixels)
    
    stats = {
        "N": n_valid,
        "Percent Valid": (n_valid / total_pixels) * 100,
        "Min": np.min(valid_pixels),
        "Max": np.max(valid_pixels),
        "Mean": mean_val,
        "Median": np.median(valid_pixels),
        "Stdev": stdev_val,
        "CV": (stdev_val / mean_val) * 100 if mean_val != 0 else np.nan
    }
    
    return stats

def extract_gridded_statistics(file_path, region_input, env, input_type="coords", verbose=False):
    """
    Main pipeline updated to utilize the dynamic shapefile locator.
    """
    try:
        # Pre-process the shapefile path if necessary
        if input_type == "shapefile":
            resolved_shapefile = get_shapefile(region_input, env, verbose=verbose)
            if not resolved_shapefile:
                raise FileNotFoundError(f"Could not resolve shapefile for input: {region_input}")
            
            # Reassign region_input to the verified, absolute path
            region_input = resolved_shapefile
            
        # 1. Extract and Clip 
        # (Assuming load_and_clip_data and calculate_region_statistics from the previous step)
        clipped_data = load_and_clip_data(file_path, region_input, input_type)
        
        # 2. Compute Statistics
        stats = calculate_region_statistics(clipped_data)
        
        # 3. Compile Metadata
        metadata = {
            "Source File": os.path.basename(file_path),
            "Full Path": os.path.abspath(file_path),
            "Region Input": region_input if input_type == "coords" else os.path.basename(region_input),
            "Input Type": input_type,
            "Resolution": clipped_data.rio.resolution(),
            "Raster CRS": str(clipped_data.rio.crs)
        }
        
        return {**metadata, **stats}
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def run_regional_extraction_pipeline(prod, region_input, env, period='D', input_type="shapefile", 
                                     dataset=None, dataset_version=None, output_dir=None, 
                                     overwrite=False, verbose=False, **kwargs):
    """
    Main workflow pipeline integrating dynamic file mapping and statistical extraction.
    """
    region_name = Path(region_input).stem
    
    # 1. Build Data Extraction Map and Resolve Output Path
    extraction_map, output_file = build_data_extraction_map(
        prod=prod,
        region_name=region_name,
        env=env,
        period=period,
        dataset=dataset,
        dataset_version=dataset_version,
        output_dir=output_dir,
        overwrite=overwrite,
        verbose=verbose,
        **kwargs
    )

    if not extraction_map:
        return None

    # 2. Filter tasks
    tasks_to_process = [path for path, info in extraction_map.items() if info['needs_processing']]

    if not tasks_to_process:
        print(f"✅ All records up to date in output: {os.path.basename(output_file)}")
        return output_file

    print(f"🚀 Processing {len(tasks_to_process)} file(s) for {prod} | Region: {region_name}")

    # 3. Extract Statistics
    new_records = []
    for file_path in tasks_to_process:
        info = extraction_map[file_path]
        try:
            clipped_da = load_and_clip_data(file_path, region_input, input_type=input_type, env=env, verbose=verbose)
            stats = calculate_region_statistics(clipped_da)

            record = {
                "source_file": info['source_file'],
                "full_path": info['full_path'],
                "input_mtime": info['input_mtime'],
                "is_nrt": info['is_nrt'],
                "timestamp": info['timestamp'],
                "product": prod,
                "region": region_name,
                **stats
            }
            new_records.append(record)
        except Exception as e:
            print(f"❌ Error processing {info['source_file']}: {e}")

    if not new_records:
        print("⚠️ No new records calculated.")
        return output_file

    new_df = pd.DataFrame(new_records)

    # 4. Incremental Output Update (Replace NRT / Append / Merge)
    if os.path.exists(output_file) and not overwrite:
        existing_ds = xr.open_dataset(output_file)
        existing_df = existing_ds.to_dataframe().reset_index()
        existing_ds.close()

        updated_files = set(new_df['source_file'])
        updated_timestamps = set(new_df['timestamp'].dropna())

        existing_df = existing_df[
            ~existing_df['source_file'].isin(updated_files) &
            ~((existing_df['is_nrt'] == True) & (existing_df['timestamp'].isin(updated_timestamps)))
        ]

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    if 'timestamp' in combined_df.columns:
        combined_df = combined_df.sort_values(by='timestamp').reset_index(drop=True)

    # 5. Export to NetCDF
    final_ds = combined_df.set_index('timestamp').to_xarray() if 'timestamp' in combined_df.columns else combined_df.to_xarray()
    final_ds.attrs['title'] = f"Regional Statistics for {prod} - {region_name}"
    final_ds.attrs['product'] = prod
    final_ds.attrs['region'] = region_name
    final_ds.attrs['last_updated'] = get_current_utc_timestamp()

    final_ds.to_netcdf(output_file, format='NETCDF4')
    print(f"✅ Saved updated output to: {output_file}")

    return output_file