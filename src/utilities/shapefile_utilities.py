import os
import glob
from pathlib import Path
import xarray as xr
import geopandas as gpd

from utilities.bootstrap.environment import bootstrap_environment
from utilities import parse_dataset_info, extract_period_code, get_period_info, get_period_dates, get_source_file_dates
from utilities import product_defaults, dataset_defaults, get_dataset_products, resolve_dataset_grid, get_daterange
env = bootstrap_environment(verbose=False)

"""
Purpose:
    SHAPEFILE_UTILITIES is a collection of utility functions for finding and handling shapefiles.

Main Functions:
    - get_shapefile: Finds the shapefile path
    - get_shapefile_subareas: Loads a shapefile and returns a list of subarea names from a specified column
    

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
                         Added get_shapefile
    Oct 21, 2026 - KJWH: Added get_shapefile_subareas

"""

def get_shapefile(shapefile_input, env, verbose=False):
    """
    Locates the correct .shp file within the env['shapefile_path'] directory structure.
    Handles case-insensitive matching for both directories and files.
    """
    # Extract the base name of the input (e.g., '/path/to/MyShape.shp' -> 'myshape')
    base_name = Path(shapefile_input).stem.lower()
    
    # Get the base shapefile path from the environment dictionary
    shape_root = env.get('shapefile_path')
    
    if not shape_root or not shape_root.exists() or not shape_root.is_dir():
        if verbose:
            print(f"Error: Invalid or missing 'shapefile_path' in env: {shape_root}")
        return None

    # Step 1: Find the matching subdirectory (case-insensitive)
    available_dirs = [d for d in shape_root.iterdir() if d.is_dir()]
    matched_dir = None
    
    for d in available_dirs:
        if d.name.lower() == base_name:
            matched_dir = d
            break
            
    if not matched_dir:
        if verbose:
            print(f"Directory for '{base_name}' not found in {shape_root}.")
            print("Available shapefile directories:")
            for d in available_dirs:
                print(f"  - {d.name}")
        return None

    # Step 2: Find the matching .shp file inside the matched directory (case-insensitive)
    available_files = [f for f in matched_dir.iterdir() if f.is_file()]
    matched_shp = None
    
    for f in available_files:
        # Match both the .shp extension and the basename
        if f.suffix.lower() == '.shp' and f.stem.lower() == base_name:
            matched_shp = f
            break
            
    if not matched_shp:
        if verbose:
            print(f"Correct .shp file not found in directory: {matched_dir.name}")
            print("Files present in directory:")
            for f in available_files:
                print(f"  - {f.name}")
        return None

    # Return the resolved full path as a string (or you can return the Path object)
    return str(matched_shp)

def get_shapefile_subareas(shapefile_input, env, subarea_column=None, verbose=False):
    """
    Purpose:
        Loads a shapefile and returns a list of subarea names from a specified column.
        If no column is provided, it attempts to auto-detect a common name column.
    """
    shapefile_path = get_shapefile(shapefile_input, env, verbose=verbose)
    if not shapefile_path:
        return None
        
    gdf = gpd.read_file(shapefile_path)
    
    if subarea_column:
        if subarea_column not in gdf.columns:
            raise ValueError(f"Column '{subarea_column}' not found in shapefile.")
        return gdf[subarea_column].unique().tolist()
        
    # Auto-detect common name columns
    common_names = ['name', 'region', 'stratum', 'id', 'subarea', 'zone']
    for col in gdf.columns:
        if col.lower() in common_names:
            if verbose: print(f"  🔍 Auto-detected subarea column: '{col}'")
            return gdf[col].unique().tolist()
            
    # Fallback: just return indices as strings if multiple features exist
    if len(gdf) > 1:
        if verbose: print("  ⚠️ No standard name column found. Using index values.")
        return gdf.index.astype(str).tolist()
        
    return [] # Single feature, no subareas needed