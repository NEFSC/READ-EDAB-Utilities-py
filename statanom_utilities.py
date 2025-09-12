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
