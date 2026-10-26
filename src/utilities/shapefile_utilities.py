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
    - get_subarea_geometries: Loads a shapefile, optionally projects it to a target CRS, and returns a dictionary of {subarea_name: geometry}.
    

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
    Oct 22, 2026 - KJWH: Updated get_shapefile_subareas functionality (now looking for string variables)
                         Added get_subarea_geometries()



"""

def get_shapefile(shapefile_input, verbose=False):
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

def get_shapefile_subareas(shapefile_input, subarea_column=None, verbose=False):
    """
    Purpose:
        Loads a shapefile and returns a list of subarea names from a specified column.
        If no column is provided, it attempts to auto-detect a common name column.
    """
    shapefile_path = get_shapefile(shapefile_input, verbose=verbose)
    if not shapefile_path:
        return None

    gdf = gpd.read_file(shapefile_path)

    if not subarea_column:
        # IDL Replicator: Find the first string/object column
        string_cols = gdf.select_dtypes(include=['object', 'string']).columns
        if len(string_cols) > 0:
            subarea_column = string_cols[0]
            if verbose: print(f"ℹ️ Auto-detected string column: '{subarea_column}'")
        else:
            raise ValueError("No string columns found in this shapefile to use as subarea names.")
    else:
        # Validation: Ensure the provided column actually exists
        if subarea_column not in gdf.columns:
            raise ValueError(f"Provided subarea_column '{subarea_column}' does not exist. Available columns: {list(gdf.columns)}")
            
    # Return the unique names (e.g., ['MAB', 'GOM', 'SS', 'GB'])
    return gdf[subarea_column].unique().tolist()

def get_subarea_geometries(shapefile, subarea_column=None, subareas=None, target_crs=None, merge_all=False):
    """
    Loads a shapefile and returns a dictionary of geometries.
    - merge_all=True: Merges the entire shapefile into a single "ALL" geometry.
    - merge_all=False: Returns every subarea individually (Default).
      If subarea_column is not provided, auto-detects the first string column.
    """
    gdf = gpd.read_file(shapefile)
    
    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    geometries = {}
    
    if merge_all:
        geometries["ALL"] = gdf.geometry.union_all()
        return geometries
        
    # Auto-detect the first text column (Mimicking IDL's "FIRST STRING ATTRIBUTE")
    if not subarea_column:
        string_cols = [col for col in gdf.columns if gdf[col].dtype in ['object', 'string'] and col != 'geometry']
        if string_cols:
            subarea_column = string_cols[0]
            print(f"  ℹ️ Auto-detected subarea column: '{subarea_column}'")

    # Group and extract by the subarea column
    if subarea_column and subarea_column in gdf.columns:
        target_subareas = subareas if subareas else gdf[subarea_column].unique()
        for sa in target_subareas:
            sa_gdf = gdf[gdf[subarea_column] == sa]
            if not sa_gdf.empty:
                geometries[str(sa)] = sa_gdf.geometry.union_all()
    else:
        # Fallback: Extract every row independently by index if no string columns exist at all
        for idx, row in gdf.iterrows():
            geometries[f"Feature_{idx}"] = row.geometry
            
    return geometries