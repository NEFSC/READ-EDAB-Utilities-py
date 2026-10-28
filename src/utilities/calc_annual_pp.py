import os
import sys
import gc
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import traceback
import concurrent.futures
from datetime import datetime

from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

from utilities import (
    pixel_area,
    get_prod_files, get_filepath, file_parser,
    get_default_metadata, get_current_utc_timestamp, get_lut_metadata, get_source_metadata, get_geospatial_metadata, get_temporal_metadata,
    get_default_product_dataset, get_default_dataset_product
)


"""
Calculate the total annual Primary Productivity (mapped)

REQUIRED INPUTS:
    pp (): Daily satellite georeference primary production (gC m⁻² d⁻¹)

RETURNS:
    pp_total   : Total annual primary production (gC m⁻² y⁻¹)
    

REFERENCES:
    

COPYRIGHT: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

AUTHOR:
    This program was written on Oct 27, 2026 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882

MODIFICATION HISTORY:
    Oct 27, 2026 - KJWH: Initial code written and adapted IDL code
    

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


def build_annual_pp_map(prod, input_period, dataset, map_subset, output_dir=None, overwrite=False, verbose=False, **kwargs):
    """
    Purpose:
        Discovers all available input files for the specified period, groups them by year, 
        and builds a processing map evaluating if the annual output file needs creation/updating.
    """
    if output_dir is None:
        output_dir = get_filepath(prod, dataset=dataset, period='A', data_type='ANNUAL', 
                                  map_subset=map_subset, getfilepath=True, make_dir=True)

    if verbose: print(f"🔍 Searching for {input_period} STATS files for {dataset}:{prod}...")
    
    input_files = get_prod_files(prod, dataset=dataset, period=input_period, 
                                 map_subset=map_subset, data_type="STATS", verbose=verbose)
    
    if not input_files:
        if verbose: print(f"⚠️ No files found for {prod} ({input_period}).")
        return {}

    # Parse metadata to group files by year
    parsed_files = file_parser(input_files)
    years_dict = {}
    
    for f_path, meta in zip(input_files, parsed_files):
        # Extract the year from the parsed timestamp
        timestamp = meta.get('timestamp') or meta.get('start_date')
        if not timestamp: continue
        
        year = pd.to_datetime(timestamp).year
        if year not in years_dict:
            years_dict[year] = []
        years_dict[year].append(f_path)

    stats_map = {}
    up_to_date = 0

    for year, files_for_year in years_dict.items():
        # Ensure chronological order
        files_for_year.sort()
        
        out_name = f"A_{year}-{map_subset}-{dataset}-{prod}-ANNUAL.nc"
        out_path = os.path.join(output_dir, out_name)
        
        is_up_to_date = False
        if os.path.exists(out_path) and not overwrite:
            out_mtime = os.path.getmtime(out_path)
            if all(out_mtime > os.path.getmtime(f) for f in files_for_year):
                is_up_to_date = True
                up_to_date += 1

        stats_map[year] = {
            "inputs": files_for_year,
            "output": out_path,
            "is_up_to_date": is_up_to_date,
            "file_exists": os.path.exists(out_path)
        }

    if verbose:
        print(f"📊 MAP SUMMARY: {len(years_dict)} total years found. {up_to_date} up-to-date, {len(years_dict) - up_to_date} to process.")
        
    return stats_map


def gap_fill_data(da, dim='time', method='linear', limit=None, verbose=False):
    """
    Purpose:
        Interpolates missing data along a specified dimension.
    """
    if verbose: print(f"    🩹 Applying {method} gap-filling along '{dim}'...")
    return da.interpolate_na(dim=dim, method=method, limit=limit)


def compute_annual_sum(da, input_period, gap_fill=True, min_valid_fraction=0.75, verbose=False):
    """
    Purpose:
        Computes the total annual sum from rate data.
        1. Counts valid observations BEFORE gap filling.
        2. Applies interpolation (optional).
        3. Multiplies by the specific time-weight (e.g., days in month).
        4. Sums across the year.
        5. Masks pixels that did not meet the valid observation threshold.
    """
    # 1. Count True Valid Observations (Before any interpolation)
    valid_obs_count = da.notnull().sum(dim='time', skipna=True)
    
    # 2. Gap Fill
    if gap_fill:
        da = gap_fill_data(da, dim='time', method='linear', verbose=verbose)

    # 3. Apply Time Weights (Scale daily rates to period totals)
    if input_period == 'M':
        weights = da.time.dt.days_in_month
        expected_obs = 12
    elif input_period in ['W', 'D7']:
        weights = xr.DataArray(np.full(da.sizes['time'], 7), dims=['time'], coords={'time': da.time})
        expected_obs = 52
    elif input_period == 'D8':
        weights = xr.DataArray(np.full(da.sizes['time'], 8), dims=['time'], coords={'time': da.time})
        expected_obs = 46
    else: # Daily
        weights = xr.DataArray(np.full(da.sizes['time'], 1), dims=['time'], coords={'time': da.time})
        days_in_year = 366 if da.time.dt.is_leap_year[0] else 365
        expected_obs = days_in_year

    # Multiply rates by days to get total carbon for that specific period
    period_totals = da * weights
    
    # 4. Sum across the year
    annual_sum_m2 = period_totals.sum(dim='time', skipna=True)

    # 5. Apply the Data Quality Mask
    min_obs_required = expected_obs * min_valid_fraction
    valid_mask = valid_obs_count >= min_obs_required
    
    annual_sum_m2 = annual_sum_m2.where(valid_mask)

    return annual_sum_m2, valid_obs_count


def process_single_year(year, task, prod, input_period, stat_type, dataset, map_subset, verbose, **kwargs):
    """
    Purpose:
        Processes a single year: opens all period files, calculates the annual sum, 
        integrates with pixel area, and writes the resulting NetCDF.
    """
    input_files = task['inputs']
    out_path = task['output']
    debug = kwargs.get('debug', False)
    ds = None
    
    try:
        if verbose: print(f"  📖 Opening {len(input_files)} files for {year}...")
        
        ds = xr.open_mfdataset(input_files, combine='by_coords', engine='h5netcdf', parallel=False)
        
        # Resolve target variables dynamically
        if prod.upper() == 'PPD':
            target_vars = [v for v in ds.data_vars if v.upper().startswith('PP_') and v.lower().endswith(stat_type.lower())]
        else:
            target_var = f"{prod}_{stat_type}"
            if target_var in ds.data_vars:
                target_vars = [target_var]
            else:
                target_vars = [v for v in ds.data_vars if prod.upper() in v.upper() and stat_type.upper() in v.upper()]
        
        if not target_vars:
            raise ValueError(f"Could not locate any variables matching '{prod}' and '{stat_type}'.")

        lat, lon = ds.lat.values, ds.lon.values
        out_ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        if debug: print(f"    🗺️ Computing spatial integration (Pixel Area)...")
        area_km2 = pixel_area(lat, lon)
        area_m2 = area_km2 * 1e6 
        out_ds["pixel_area_km2"] = area_km2

        gap_fill = kwargs.get('gap_fill', True)
        min_valid_fraction = kwargs.get('min_valid_fraction', 0.75)

        for var_name in target_vars:
            if verbose: print(f"    📊 Processing {var_name}...")
            da = ds[var_name]
            
            annual_sum_m2, valid_obs_count = compute_annual_sum(
                da, input_period, gap_fill=gap_fill, min_valid_fraction=min_valid_fraction, verbose=debug
            )
            
            annual_total_absolute = annual_sum_m2 * area_m2
            base_var = var_name.replace(f"_{stat_type}", "")
            
            out_ds[f"{base_var}_Annual_Total"] = (["lat", "lon"], annual_total_absolute.values.astype("float32"))
            out_ds[f"{base_var}_Annual_Per_m2"] = (["lat", "lon"], annual_sum_m2.values.astype("float32"))
            out_ds[f"{base_var}_Valid_Periods"] = (["lat", "lon"], valid_obs_count.values.astype("int16"))
            
            out_ds[f"{base_var}_Annual_Total"].attrs = {
                "long_name": f"{base_var} Total Annual Primary Productivity (derived from {input_period} {stat_type}s)",
                "units": "gC yr-1",
                "description": "Absolute carbon production per pixel per year."
            }
            out_ds[f"{base_var}_Annual_Per_m2"].attrs = {
                "long_name": f"{base_var} Annual Primary Productivity per Square Meter (derived from {input_period} {stat_type}s)",
                "units": "gC m-2 yr-1",
                "description": "Normalized carbon production rate."
            }
            out_ds[f"{base_var}_Valid_Periods"].attrs = {
                "long_name": f"{base_var} Valid {input_period} Observations",
                "units": "count",
                "description": f"Number of valid input periods contributing to the annual sum (Threshold: >={min_valid_fraction*100}%)."
            }

        ref_time = pd.to_datetime(f"{year}-07-01") 
        out_ds = out_ds.expand_dims('time').assign_coords(time=[ref_time])

        attrs = get_default_metadata(sheet="General") | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
        attrs = attrs | get_geospatial_metadata(dataset=out_ds) | get_temporal_metadata(ds=out_ds)
        
        attrs['title'] = f"Annual Total Primary Productivity for {year}"
        attrs['product'] = prod
        attrs['history'] = f"{get_current_utc_timestamp()} Computed annual sums using {input_period} inputs. Gap filling={'Enabled' if gap_fill else 'Disabled'}."
        
        try:
            attrs = attrs | get_source_metadata(dataset, source_prefix="source")
        except: pass

        out_ds.attrs = attrs

        if debug: print(f"    💾 Saving {os.path.basename(out_path)}...")
        out_ds.to_netcdf(out_path, format='NETCDF4', engine='h5netcdf')
        
        return True, None

    except Exception as e:
        if debug: traceback.print_exc()
        return False, str(e)
    finally:
        if ds is not None: ds.close()
        gc.collect()


def run_annual_pp_pipeline(prods=None, input_periods=None, stat_type='median', **kwargs):
    """
    Purpose:
        Orchestrates the Annual Primary Productivity workflow across multiple products and years.
    """
    dataset = kwargs.pop('dataset', None)
    map_subset = kwargs.get('map_subset', 'NWA')
    
    if not prods: prods = ['PPD']
    if isinstance(prods, str): prods = [prods.upper().strip()]

    if not input_periods: input_periods = ['M']
    elif isinstance(input_periods, str): input_periods = [input_periods]

    logfile = kwargs.pop('logfile', None)
    original_stdout = sys.stdout 
    if logfile:
        sys.stdout = DualLogger(logfile)
        print("\n" + "="*80)
        print(f"🟢 ANNUAL PP PIPELINE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

    try:
        verbose = kwargs.get('verbose', False)
        parallel_runs = kwargs.get('parallel_runs', 1)
        
        skipped, created, replaced, failed = [], [], [], []

        for prod in prods:
            current_dataset = dataset or get_default_product_dataset(prod)
            
            for input_period in input_periods:
                print(f"\n🚀 Starting Annual Pipeline for {prod} (Dataset: {current_dataset}) | Input: {input_period}")
                
                stats_map = build_annual_pp_map(prod, input_period, current_dataset, map_subset, **kwargs)
                tasks_to_run = [(yr, task) for yr, task in stats_map.items() if not task['is_up_to_date']]
                
                for yr, task in stats_map.items():
                    if task['is_up_to_date']: skipped.append(yr)

                if not tasks_to_run:
                    continue

                if parallel_runs > 1:
                    if verbose: print(f"  ⚡ Launching {len(tasks_to_run)} years across {parallel_runs} workers...")
                    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_runs) as executor:
                        future_map = {
                            executor.submit(process_single_year, yr, task, prod, input_period, stat_type, current_dataset, map_subset, verbose, **kwargs): yr
                            for yr, task in tasks_to_run
                        }
                        for future in concurrent.futures.as_completed(future_map):
                            yr = future_map[future]
                            success, err = future.result()
                            if success:
                                if stats_map[yr]['file_exists']: replaced.append(yr)
                                else: created.append(yr)
                            else:
                                failed.append(f"{yr} ({err})")
                else:
                    for yr, task in tasks_to_run:
                        success, err = process_single_year(yr, task, prod, input_period, stat_type, current_dataset, map_subset, verbose, **kwargs)
                        if success:
                            if task['file_exists']: replaced.append(yr)
                            else: created.append(yr)
                        else:
                            failed.append(f"{yr} ({err})")

        print("\n" + "="*60)
        print("🏁 ANNUAL PP PIPELINE FINISHED")
        print("="*60)
        print(f"⏭  Skipped (Up-to-date): {len(skipped)}")
        print(f"🔄 Replaced:             {len(replaced)}")
        print(f"🆕 Created:              {len(created)}")
        print(f"❌ Failed:               {len(failed)}")
        if failed:
            for f in failed: print(f"  • {f}")
        print("="*60)

    finally:
        if logfile:
            print(f"\n🔴 PIPELINE RUN ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout.close()
            sys.stdout = original_stdout

    # Check input data

    # Create monthly means

    # Find all valid data - IDL code first subset by region and then filled in any missing data with the monthly mean of the region, but that won't work if this code is just returning the summed PP for the entire map.

    

    # Replicate monthly data based on the number of days in the month

    # Sum all data within a year

    # Return annual sums