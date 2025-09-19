import xarray as xr
import numpy as np
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)
#from utilities import dataset_info_map


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
    
"""
def period_info():
    """
    Returns a dictionary mapping period codes to metadata for statistical processing.
    Each entry contains:
        0) Input time to xarray.groupby for stat calculations
        1) Input period code (to stats) to create the period
        2) Is it a range period?
        3) Is it a running mean period?
        4) Is it a climatological period?
        5) ISO 8601 duration
        6) GroupBy hint (xarray expression or None)
        7) File period format
        8) Description
    """
    return {
        'D':      ('time',  'D',  False, False, False, 'P1D', 'time', 'D_YYYYMMDD', 'Daily'),
        'DD':     ('time',  'D',  True,  False, False, 'PnD', 'time', 'DD_YYYYMMDD_YYYYMMDD', 'Range of daily data'),
        'DOY':    ('doy',   'D',  False, False, True,  '',    'time.dt.dayofyear', 'DOY_YYYY_YYYY', 'Climatological day of year'),
        'D3':     ('day3',  'D',  False, True,  False, 'P3D',  None, 'D3_YYYYMMDD_YYYYMMDD', 'Running 3-day'),
        'DD3':    ('day3',  'D',  True,  True,  False, 'P3D',  None, 'DD3_YYYYMMDD_YYYYMMDD', 'Range of running 3-day data'),
        'D8':     ('day8',  'D',  False, True,  False, 'P8D',  None, 'D8_YYYYMMDD_YYYYMMDD', 'Running 8-day'),
        'DD8':    ('day8',  'D',  True,  True,  False, 'P8D',  None, 'DD8_YYYYMMDD_YYYYMMDD', 'Range of running 8-day data'),
        'W':      ('week',  'D',  False, False, False, 'P1W', 'time.dt.isocalendar().week', 'W_YYYYWW', 'Weekly'),
        'WW':     ('week',  'D',  True,  False, False, 'P1W', 'time.dt.isocalendar().week', 'WW_YYYYWW_YYYYWW', 'Range of weekly data'),
        'WEEK':   ('week',  'W',  False, False, True,  '',    'time.dt.isocalendar().week', 'WEEK_YYYYWW_YYYYWW', 'Climatological week'),
        'M':      ('month', 'D',  False, False, False, 'P1M', 'time.dt.month', 'M_YYYYMM', 'Monthly'),
        'MM':     ('month', 'D',  True,  False, False, 'P1M', 'time.dt.month', 'MM_YYYYMM_YYYYMM', 'Range of monthly data'),
        'M3':     ('month3','M',  False, True,  False, 'P3M',  None, 'M3_YYYYMM_YYYYMM', 'Running 3-month'),
        'MM3':    ('month3','M',  True,  True,  False, 'P3M',  None, 'MM3_YYYYMM_YYYYMM', 'Range of running 3-month'),
        'MONTH':  ('month', 'M',  False, False, True,  'P1M', 'time.dt.month', 'MONTH_YYYYMM_YYYYMM', 'Climatological month'),
        'SEA':    ('sea',   'M',  False, False, False, 'P3M', 'time.dt.month', 'SEA_YYYYMM_YYYYMM', 'Seasonal (JFM)'),
        'SEASON': ('season','SEA',False, False, True,  'P3M', 'time.dt.season', 'SEASON_YYYYMM_YYYYMM', 'Climatological season'),
        'A':      ('annual' ,'M', False, False, False, 'P1Y', 'time.dt.year', 'A_YYYY', 'Annual (monthly inputs)'),
        'AA':     ('annual' ,'M', True,  False, False, 'P1Y', 'time.dt.year', 'AA_YYYY_YYYY', 'Range of annual (monthly inputs)'),
        'ANNUAL': ('annual' ,'A', False, False, True,  'P1Y', 'time.dt.year', 'ANNUAL_YYYY_YYYY', 'Climatological annual'),
        'Y':      ('year',  'D',  False, False, False, 'P1Y', 'time.dt.year', 'Y_YYYY_YYYY', 'Yearly (daily inputs)'),
        'YY':     ('year',  'D',  True,  False, False, 'P1Y', 'time.dt.year', 'YY_YYYY_YYYY', 'Range of yearly (daily inputs)'),
        'YEAR':   ('year',  'Y',  False, False, True,  'PnY', 'time.dt.year', 'YEAR_YYYY_YYYY', 'Climatological year'),
    }

def get_period_info(period_code):
    """
    Returns a dictionary of named fields for a given period code.

    Parameters:
        period_code (str): The key from the period_info() map (e.g. 'W', 'M3', 'SEASON')

    Returns:
        dict: A dictionary with named metadata fields, or raises KeyError if not found.
    """
    period_map = period_info()

    if period_code not in period_map:
        raise KeyError(f"Period code '{period_code}' not found in period_info()")

    (
        groupby,
        input_period_code,
        is_range,
        is_running_mean,
        is_climatology,
        iso_duration,
        groupby_hint,
        file_format,
        description
    ) = period_map[period_code]

    return {
        'groupby': groupby,
        'input_period_code': input_period_code,
        'is_range': is_range,
        'is_running_mean': is_running_mean,
        'is_climatology': is_climatology,
        'iso_duration': iso_duration,
        'groupby_hint': groupby_hint,
        'file_format': file_format,
        'description': description
    }
    
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
    
def stats_wrapper(ds, periods=['W','M'], var_name='chlorophyll', time_dim='time'):
    """
    Compute statistics (mean, min, max, median, std) from a gridded xarray Dataset or DataArray.

    Parameters:
    ----------
    ds (xr.Dataset or xr.DataArray) : Input dataset containing chlorophyll data.
    var_name (str): Name of the variable to compute stats on.
    periods (str): Name of the output period(s) to calculate

    Returns:
    -------
    xr.Dataset: Dataset containing period statistics.
    """
    # Defensive checks
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    if time_dim not in ds.dims and time_dim not in ds.coords:
        raise ValueError(f"Time dimension '{time_dim}' not found.")

    da = ds[var_name] if isinstance(ds, xr.Dataset) else ds

    # Ensure datetime type
    da[time_dim] = xr.decode_cf(da).indexes[time_dim].to_datetimeindex()

    for per in periods:

        grouped = apply_grouping(da, per, time_dim)
        stats = compute_stats(grouped, var_name, time_dim, label=per)



def run_stats_pipeline(dataset,prods=None,periods=['W','WEEK','M','MONTH','A','ANNUAL','DOY'],version=None, verbose=False):
    from utilities import get_prod_files,dataset_defaults
    """
    Runs the full stats pipeline
        - Finds the input files
        - 
    """
    # --- Step 1: Get standard PERIOD_CODE information and DATASETS defaults
    period_map = period_info()
    dataset_map = dataset_defaults()

    # If prod not provided use the dataset default
    if not prods:
        prods = dataset_map[dataset][2]
        
    # Loop through prods
    for prod in [prods]:
        # Loop through the periods
        for per in periods:
            per_map = get_period_info(per)
            input_per = per_map['input_period_code']
            print(f"Searching for {input_per} files for {dataset}:{prod}")
            if input_per == 'D':
                files = get_prod_files(prod,dataset=dataset)
            else:
                files = get_prod_files(prod,dataset=dataset,data_type="STATS",period=input_per)
    return files

def get_climatology(ds, periods=['MONTH','WEEK','ANNUAL'],variable=None):
    """
    """
