import xarray as xr
import numpy as np
from datetime import datetime
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

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
    info = period_info().get(period_code)
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

def compute_stats(grouped, var_name, time_dim, label):
    return xr.Dataset({
        f"{var_name}_{label}_mean":   grouped.mean(dim=time_dim, skipna=True),
        f"{var_name}_{label}_min":    grouped.min(dim=time_dim, skipna=True),
        f"{var_name}_{label}_max":    grouped.max(dim=time_dim, skipna=True),
        f"{var_name}_{label}_median": grouped.median(dim=time_dim, skipna=True),
        f"{var_name}_{label}_std":    grouped.std(dim=time_dim, skipna=True),
        f"{var_name}_{label}_sum":    grouped.sum(dim=time_dim, skipna=True),
        f"{var_name}_{label}_var":    grouped.var(dim=time_dim, skipna=True),
    })
       
def stats_period_map(prod, period, dataset=None, dates=None, version=None, dataset_map=None, subset=None, is_running=True, verbose=False):
    """
    Constructs a stats period file path map using get_prod_files.

    Parameters:
        prod (str): Product to conduct the stats on.
        period (str): The period code for the output stats file.
        dataset (str): Override dataset. Defaults to each product’s default from product_defaults() (optional).
        dates (list of str): List of input dates (YYYYMMDD) to process.
        version (str): Dataset version override passed through to get_prod_files (optional)).
        dataset_map (str): Dataset map override passed through to get_prod_files (optional).
        subset (str): The subset region (e.g. NES, NWA) to subset the data 
        is_running (bolean): To override the D3 and D8 running stats.
        verbose (bool): If True, prints progress and provenance (default is False).

    Returns:
        dict: Mapping of output period dates
    """
    from utilities import get_prod_files, get_period_info, parse_dataset_info, get_file_dates
    per_map = get_period_info(period)
    input_per = per_map['input_period_code']

    # Turn off the is_running_mean option in in the period map if the default is to create a running_mean (e.g. D3 and D8 periods)
    if not is_running:
        if per_map['is_running_mean']:
            per_map['is_running_mean'] = False
    
    # Find the input files
    if verbose:
        print(f"Searching for {input_per} files for {dataset}:{prod}")
    if input_per in ['D','DD']:
        files = get_prod_files(prod,dataset=dataset,dataset_version=version,dataset_map=dataset_map,period=None,data_type=None,verbose=verbose)
    else:
        files = get_prod_files(prod,dataset=dataset,dataset_version=version,dataset_map=dataset_map,period=input_per,data_type="STATS",verbose=verbose)

    if not files:
        print(f"No {input_per} files found for {dataset}:{prod}")
        return []

    # Get the dates of the input files
    ds_parsed = parse_dataset_info(files[0])
    infile_dates = get_file_dates(files)

    # Create output periods based on the range of input dates


import os
from datetime import datetime
from collections import defaultdict

def build_stat_file_map(files, period_code, output_dir, date_format="%Y%m%d", verbose=False):
    """
    Builds a map of output period → (input files, output path, is_up_to_date)

    Parameters
    ----------
    files : list of str
        Input file paths.
    period_code : str
        Output period code (e.g. 'D3', 'M3', 'SEA').
    output_dir : str
        Directory where stat files will be written.
    date_format : str
        Format of date strings in filenames.
    verbose : bool
        If True, print status messages.

    Returns
    -------
    dict
        Mapping of period → dict with keys:
            - 'inputs': list of input file paths
            - 'output': output stat file path
            - 'is_up_to_date': bool
    """
    # Extract start dates from input files
    dates = get_source_file_dates(files, format="yyyymmdd", placeholder=None)
    file_map = {d: f for d, f in zip(dates, files) if d}

    # Get output periods
    period_spans = get_period_sets(period_code, dates=dates, running=True)

    stat_map = {}
    for period in period_spans:
        _, start, end = period.split("_")
        inputs = [
            f for d, f in file_map.items()
            if start <= d <= end
        ]
        if not inputs:
            continue

        # Build output filename
        filename = f"{period}-STAT.nc"
        output_path = os.path.join(output_dir, filename)

        # Check freshness
        if os.path.exists(output_path):
            output_mtime = os.path.getmtime(output_path)
            input_mtimes = [os.path.getmtime(f) for f in inputs]
            is_up_to_date = all(output_mtime > m for m in input_mtimes)
        else:
            is_up_to_date = False

        if verbose:
            status = "✅ up-to-date" if is_up_to_date else "⏳ needs update"
            print(f"{period}: {status} ({len(inputs)} files)")

        stat_map[period] = {
            "inputs": inputs,
            "output": output_path,
            "is_up_to_date": is_up_to_date
        }

    return stat_map




def run_stats_pipeline(prods,
                       dataset=None,
                       periods=None,
                       subset=None,
                       daterange=None,
                       time_dim='time',
                       version=None,
                       dataset_map=None,
                       is_running=True,
                       verbose=False
):
    
    from utilities import product_defaults, get_nc_prod, get_dates
    """
    Runs a multi-product, multi-period stats pipeline.

    For each product in `prods` and each period code in `periods`, this function:
      1. Determines dataset defaults (via product_defaults).
      2. Locates raw (D/'DD') or derived stats files (all other periods).
      3. Opens and subsets those files in time.
      4. Applies xarray grouping or rolling based on period code.
      5. Computes summary statistics (mean, min, max, etc.).
      6. Returns a dict of xr.Dataset results keyed by (prod, period).

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
    if isinstance(prods, str):
        prods = [prods]
    if periods is None:
        periods = ['W','WEEK','M','MONTH','A','ANNUAL','DOY']
    
    # Load PERIOD and PRODUCT defaults
    period_map = period_info()
    prods_map = product_defaults()
        
    # Loop through prods
    for prod in [prods]:
        prod = prod.upper().strip()

        # If prod not provided use the dataset default
        if not dataset:
            dataset = prods_map[prod][1]
        # Loop through the periods
        for per in periods:
            
            # Create stats period map
            stats_map = stats_period_map(prod, per, dataset=dataset, dataset_version=version,dataset_map=dataset_map, subset=subset, is_running=is_running, verbose=verbose)
            
            
            
            # Open dataset
            ds = xr.open_mfdataset(files, combine="by_coords")
            ds = xr.decode_cf(ds)

            if time_dim not in ds.dims and time_dim not in ds.coords:
                raise ValueError(
                    f"Time dimension '{time_dim}' not found in {dataset}:{prod} "
                    f"({datetime.now()})"
                )

            # Check that the time resolution matches the input period

            # Subset the data based on the input daterange
            if daterange:
                dates = get_dates(daterange)
                ds = ds.sel({time_dim: slice(min(dates), max(dates))})

            # Check that the prod exists in the dataset and extract the dataset product
            if prod not in ds.data_vars:
                ncprod = get_nc_prod(ds_name, prod)
                if ncprod in ds.data_vars:
                    prod_key = ncprod
                else:
                    raise ValueError(
                        f"Variable '{prod}' not in dataset. "
                        f"Available: {list(ds.data_vars)} ({datetime.now()})"
                    )
            else:
                prod_key = prod

            da = ds[prod_key]

            # Apply grouping
            grouped = apply_grouping(da, per, time_dim)
            
            # Compute stats
            stats = compute_stats(grouped, prod, time_dim, label=per)

            if verbose:
                print(f"✅ Completed stats for {prod}, period '{per}'")
    
            return stats



    

def get_climatology(ds, periods=['MONTH','WEEK','ANNUAL'],variable=None):
    """
    """
