import xesmf as xe
import os
import hashlib
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
from .file_utilities import parse_dataset_info
from .import_utilities import get_python_dir
from .date_utilities import get_dates


def hash_grid(ds):
    """Hash lat/lon coordinates for reproducible weight filenames."""
    lat = ds["lat"].values
    lon = ds["lon"].values
    key = f"{lat.tobytes()}{lon.tobytes()}"
    return hashlib.md5(key.encode()).hexdigest()
    
def get_regrid_weights(target_path, source_path,
                       weights_dir=None,
                       method='bilinear',
                       overwrite=False, 
                       target_label=None,
                       source_label=None
):
    """
        Use Xesmf to create the regridding weights file based on the input datasets and return the full path of the weigths file.

        Parameters:
        - target_input: path to the dataset that the source datasets will be regridded to
        - source_input: path to the dataset that will be regridded
        - weights_dir: directory to store weight files
        - method: interpolation type (for Xesmf)
        - overwrite: if True, regenerate weight files
        - target_label, source_labels: optional strings to override auto labels

        Returns:
        - the full path of the regrid weights file
    """

    # Get the output directory for the weight files
    if not weights_dir:
        pypath = get_python_dir(resources=True)
        weights_dir = Path(pypath) / "regrid_weights"
    os.makedirs(weights_dir, exist_ok=True)

    if target_label is None:
        try:
            target_parser = parse_dataset_info(target_path)
            target_label = f"{target_parser['dataset']}_{target_parser['dataset_map']}_{target_parser['product']}"
        except Exception:
            target_label = f"CUSTOM_TARGET"

    if source_label is None: 
        try:
            source_parser = parse_dataset_info(source_path)
            source_label = f"{source_parser['dataset']}_{source_parser['dataset_map']}_{source_parser['product']}"
        except Exception:
            source_parser = "CUSTOM_SOURCE"
    
    if os.path.isfile(target_path) and target_path.endswith(".nc"):
       target_ds = xr.open_dataset(target_path)  # Single NetCDF file
    else:
        target_ds = xr.open_mfdataset(os.path.join(target_path, "*.nc"), combine='by_coords') # Directory containing multiple NetCDF files
        
    if "lat" not in target_ds or "lon" not in target_ds:
        raise ValueError("Target dataset must include 'lat' and 'lon' coordinates.")
    
    if os.path.isfile(target_path) and target_path.endswith(".nc"):
       source_ds = xr.open_dataset(source_path)  # Single NetCDF file
    else:
        source_ds = xr.open_mfdataset(os.path.join(source_path, "*.nc"), combine='by_coords') # Directory containing multiple NetCDF files

    if "lat" not in source_ds or "lon" not in source_ds:
        raise ValueError(f"Source dataset {source_path} missing lat/lon coordinates.")

    source_hash = hash_grid(source_ds)
    target_hash = hash_grid(target_ds)
    weights_file = os.path.join(weights_dir, f"regrid_weights_{target_label}_{source_hash}_to_{source_label}_{target_hash}.nc")

    if overwrite or not os.path.exists(weights_file):
        target_grid = target_ds[["lat", "lon"]]
        
        print(f"🛠️ Creating weights file: {weights_file}")
        print("(⏳ This step may take several minutes, but will only need to be done once for a given dataset combination.⏳)")
        regridder = xe.Regridder(source_ds, target_grid, method, filename=weights_file, reuse_weights=False)
        regridder.to_netcdf(weights_file) # 💾 Save the weights file
    
    return weights_file

def regrid_dataset(
    weights_file,
    source_ds: xr.Dataset,
    target_ds: xr.Dataset,
    method: str = "bilinear",
    variables: list[str] = None
) -> xr.Dataset:
    """
    Regrid selected variables from source_ds onto the grid of target_ds.
    
    Parameters
    ----------
    source_ds : xr.Dataset
        Dataset to regrid.
    target_ds : xr.Dataset
        Dataset whose grid defines the output.
    variables : list of str, optional
        Variables to regrid. If None, use all data_vars.
    method : str
        Regridding method: "bilinear", "nearest_s2d", etc.
    reuse_weights : bool
        Whether to cache and reuse weights between calls.
    output_path : str
        If provided, save the result to NetCDF at this path.
    
    Returns
    -------
    regridded : xr.Dataset
        Dataset with regridded variables.
    """

    if weights_file is None:
        raise ValueError("Weights file is required to regrid the data.")
    
    # Get the source data variables
    if variables is None:
        variables = list(source_ds.data_vars)
    elif isinstance(variables, str):
        variables = [variables]
    elif not isinstance(variables, (list, tuple)):
        raise TypeError(f"`variables` must be a list of strings, not {type(variables).__name__}")

    
    target_grid = target_ds[["lat", "lon"]]
    #check_weights_valid(weights_file, source_ds, target_grid)

    regridder = xe.Regridder(source_ds, target_grid, method, filename=weights_file, reuse_weights=True)

    regridded = xr.Dataset()
    for var in variables:
        regridded[var] = regridder(source_ds[var])
        regridded[var].attrs.update(source_ds[var].attrs)

    for coord in target_ds.coords:
        if coord not in regridded.coords:
            regridded[coord] = target_ds[coord]

    regridded.attrs.update(target_ds.attrs)
    return regridded

def regrid_wrapper(target_path, source_path, source_vars=None, daterange=None):
    
    # 1. Get the weights file
    weight_file = get_regrid_weights(target_path, source_path)

    # 2. Open the datasets    
    if os.path.isfile(target_path) and target_path.endswith(".nc"):
        target_ds = xr.open_dataset(target_path)  # Single NetCDF file
    else:
        target_ds = xr.open_mfdataset(os.path.join(target_path, "*.nc"), combine='by_coords') # Directory containing multiple NetCDF files
    
    if os.path.isfile(target_path) and target_path.endswith(".nc"):
        source_ds = xr.open_dataset(source_path)  # Single NetCDF file
    else:
        source_ds = xr.open_mfdataset(os.path.join(source_path, "*.nc"), combine='by_coords') # Directory containing multiple NetCDF files
    
    # 3 Subset dataset by time (daterange) and products if needed    
    if daterange:
        dates = get_dates(daterange,format='datetime')
        source_ds = source_ds.sel(time=dates,method='nearest',tolerance=np.timedelta64(1, 'D'))
        
    if source_vars:
        source_ds = source_ds[source_vars]

    # 4. Regrid datasets
    regridded_ds = regrid_dataset(weights_file=weight_file,target_ds=target_ds,source_ds=source_ds,variables=source_vars)

    return regridded_ds