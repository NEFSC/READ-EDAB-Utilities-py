import numpy as np
import xarray as xr
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)


def total_annual_pp(pp):
    """
    Calculate the total annual Primary Productivity (mapped)

    REQUIRED INPUTS:
        pp         : xarray — Daily satellite georeference primary production (gC m⁻² d⁻¹)

    RETURNS:
        pp_total   : Total annual primary production (gC m⁻² y⁻¹)
        
    
    REFERENCES:
        

    COPYRIGHT: 
        Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
        Northeast Fisheries Science Center, Narragansett Laboratory.
        This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
        This routine is provided AS IS without any express or implied warranties whatsoever.

    AUTHOR:
      This program was written on July 08, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
    MODIFICATION HISTORY:
        July 08, 2025 - KJWH: Initial code written and adapted IDL code
        
    
    """

    # Check input data

    # Create monthly means

    # Find all valid data - IDL code first subset by region and then filled in any missing data with the monthly mean of the region, but that won't work if this code is just returning the summed PP for the entire map.

    

    # Replicate monthly data based on the number of days in the month

    # Sum all data within a year

    # Return annual sums