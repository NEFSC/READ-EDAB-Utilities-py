import os
import xarray as xr
import pandas as pd
from date_utilities import get_source_file_dates
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

"""
Purpose:
    NETCDF_UTILITIES is a collection of utility functions for working with netcdf files.

Main Functions:
    - add_time_dim: Adds the time dimension to a netcdf file if missing
    - 

Helper Functions:
    - 
    
Copywrite: 
    Copyright (C) 2026, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on July 28, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Jul 28, 2026 - KJWH: Initial functions created
    

"""



def add_time_dim(ds):
    """
    Checks if the 'time' dimension exists in a netcdf file. If not, extracts the date 
    from the filename and adds it as a dimension.
    """
    # 1. Check if 'time' is already a dimension
    if 'time' not in ds.dims:
        
        # 2. Extract the original filename from xarray's encoding dictionary
        # When opening files, xarray stores the file path in ds.encoding['source']
        filepath = ds.encoding.get('source', '')
        
        if filepath:
            # 3. Extract the date from the filename
            # Note: adjust the arguments of get_source_file_dates based on your exact function signature
            extracted_dates = get_source_file_dates([filepath]) 
            date_str = extracted_dates[0] 
            
            # 4. Convert the string to a pandas datetime object
            timestamp = pd.to_datetime(date_str)
            
            # 5. Expand dimensions to include time
            ds = ds.expand_dims(time=[timestamp])
            
    return ds

# --- How to use this with multiple files ---

# Assuming get_prod_files returns a list of file paths:
file_list = get_prod_files('POC')

# Open all files, preprocess each one to add 'time', and concatenate along the new 'time' dimension
ds_combined = xr.open_mfdataset(
    file_list, 
    preprocess=add_time_dim, 
    combine='nested',      # Use 'nested' since we are enforcing the concatenation structure
    concat_dim='time',     # Tell xarray to stack the files along the 'time' dimension
    engine='netcdf4'       # Explicitly declaring the engine can prevent backend ambiguity
)

print(ds_combined)