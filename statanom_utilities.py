import xarray as xr
import numpy as np

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
def period_defaults():
    """
    Creates the period code map for the default period codes
        1) Input time to xarray.groupby for stat calculations
        2) Input period code (to stats) to create the period
        3) Is it a range period? (a period with multiple days/weeks/months in a single file)
        4) Is it a running mean period? 
        5) Is it a climatological period?
        6) Period duration (using ISO standards)
        7) Description
        8) File period format
    Parameters:
        No inputs
        
    Returns:
        Dictionary of the period codes and associated time information
    """

    period_map = {
        'D':     ('daily','',False, False, False, 'P1D','daily','D_YYYYMMDD'),
        'DD':    ('daily','D',True, False, False, 'PnD','Range of daily files','DD_YYYYMMDD_YYYYMMDD'),
        'DOY':   ('','D',False, False, True,'','climatoligical day of year','DOY_YYYY_YYYY'),
        'D3':    ('','D',False, True, False,'P3D','3-day','D3_YYYYMMDD_YYYYMMDD'),
        'DD3':   ('','',True, True, False,'3-day','DD3_YYYYMMDD_YYYYMMDD'),
        'D8':    ('','D',False, True, False,'','D8_YYYYMMDD_YYYYMMDD'),
        'DD8':   ('','D',True, True, False,'','D8_YYYYMMDD_YYYYMMDD'),
        'W':     ('weekly','D',False, False, False, 'P1W', 'weekly', 'W_YYYYWW'),
        'WW':    ('','D',True, False, False,'','','WW_YYYYWW_YYYYWW'),
        'WEEK':  ('','W',False, False, True,'','','WEEK_YYYYWW_YYYYWW'),
        'M':     ('','D',False, False, False,'','','M_YYYYMM'),
        'MM':    ('','D',True, False, False,'','','MM_YYYYMM_YYYYMM'),
        'M3':    ('','M',False, True, False,'','','M3_YYYYMM_YYYYMM'),
        'MM3':   ('','M',True, True, False,'','','MM3_YYYYMM_YYYYMM'),
        'MONTH': ('','M',False, False, True,'','','MONTH_YYYYMM_YYYYMM'),
        'SEA':   ('','M',False, False, False,'','','SEA_YYYYMM_YYYYMM'),
        'SEASON':('','SEA',False, False, True,'','','SEASON_YYYYMM_YYYYMM'),
        'A':     ('','M',False, False, False,'','','A_YYYY'),
        'SA':    ('','M',True, False, False,'','','AA_YYYY_YYYY'),
        'ANNUAL':('','A',False, False, True,'','','ANNUAL_YYYY_YYYY'),
        'Y':     ('','D',False, False, False,'','','Y_YYYY_YYYY'),
        'YEAR':  ('','Y',False, False, True,'','','YEAR_YYYY_YYYY'),

    }

def compute_stats(grouped, label):
    """
        return xr.Dataset({
            f"{var_name}_{label}_mean": grouped.mean(dim=time_dim, skipna=True),
            f"{var_name}_{label}_min": grouped.min(dim=time_dim, skipna=True),
            f"{var_name}_{label}_max": grouped.max(dim=time_dim, skipna=True),
            f"{var_name}_{label}_median": grouped.median(dim=time_dim, skipna=True),
            f"{var_name}_{label}_std": grouped.std(dim=time_dim, skipna=True)
            f"{var_name}_{label}_sum": grouped.sum(dim=time_dim, skipna=True)
            f"{var_name}_{label}_var": grouped.var(dim=time_dim, skipna=True)
        })
    """
def get_stats(ds, periods=['W','M'], var_name='chlorophyll', time_dim='time'):
    """
    Compute weekly and monthly statistics (mean, min, max, median, std) 
    from a gridded chlorophyll xarray Dataset or DataArray.

    Parameters:
    ----------
    ds : xr.Dataset or xr.DataArray
        Input dataset containing chlorophyll data.
    var_name : str
        Name of the variable to compute stats on.
    time_dim : str
        Name of the time dimension.

    Returns:
    -------
    xr.Dataset
        Dataset containing weekly and monthly statistics.
    """
    # Defensive checks
    if var_name not in ds:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    if time_dim not in ds.dims and time_dim not in ds.coords:
        raise ValueError(f"Time dimension '{time_dim}' not found.")

    da = ds[var_name] if isinstance(ds, xr.Dataset) else ds

    # Ensure datetime type
    da[time_dim] = xr.decode_cf(da).indexes[time_dim].to_datetimeindex()

    # Group by week and month
    weekly = da.groupby(f"{time_dim}.week")
    monthly = da.groupby(f"{time_dim}.month")

    

    weekly_stats = compute_stats(weekly, "weekly")
    monthly_stats = compute_stats(monthly, "monthly")

    # Merge and return
    return xr.merge([weekly_stats, monthly_stats])

def get_climatology(ds, periods=['MONTH','WEEK','ANNUAL'],variable=None):
    """
    """
