import xesmf as xe
import os
import hashlib
import glob
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd

from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

from utilities import parse_dataset_info
from utilities import get_dates

def map_subset_defaults():
    """
    Returns the default longitude and latitude coordinates for map subsets

    Parameters:
        No inputs
    
    Returns:
        Dictionary of default dataset specific data types and products
    """

    # The default source data location and product for each dataset
    subset_map_info = {
        'NES': {'lat_min':34.0,'lat_max':46.0,'lon_min':-79.0,'lon_max':-61.0},
        'NWA': {'lat_min':22.5,'lat_max':48.5,'lon_min':-82.5,'lon_max':-51.5},
        'GLOBAL': {'lat_min':-90.0,'lat_max':90.0,'lon_min':-180.0,'lon_max':-180.0}
    }

    return subset_map_info

def subset_dataset(ds, region, lat_name="lat", lon_name="lon"):
    """
    Subsets an xarray Dataset or DataArray using predefined lat/lon bounds.

    Parameters:
        ds          : xarray.Dataset or xarray.DataArray
        region.     : str — key from map_subset_defaults() (e.g., 'NES', 'NWA')
        lat_name    : str — name of latitude coordinate in ds
        lon_name    : str — name of longitude coordinate in ds

    Returns:
        Subsetted xarray object
    """
    bounds = map_subset_defaults().get(region)
    if bounds is None:
        raise ValueError(f"Region '{region}' not found in subset map.")

    lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
    lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]

    # Handle longitude wraparound if needed
    if ds[lon_name].max() > 180:
        lon_min = (lon_min + 360) if lon_min < 0 else lon_min
        lon_max = (lon_max + 360) if lon_max < 0 else lon_max

    # Handle latitude direction
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:  # descending
        lat_slice = slice(lat_max, lat_min)
    else:  # ascending
        lat_slice = slice(lat_min, lat_max)

    return ds.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})

def compute_dynamic_chunks(ds: xr.Dataset, target_chunks: int = 32) -> dict:
    """
    Compute dynamic chunk sizes for a dataset with shape (time, lat, lon),
    aiming for ~target_chunks total across spatial dimensions.
    """
    dims = list(ds.dims)
    shape = tuple(ds.sizes[dim] for dim in dims)

    # Default to full chunk along non-spatial dims
    chunks = {}

    # Identify spatial dims (assumes lat/lon naming)
    spatial_dims = [dim for dim in dims if dim.lower() in ("lat", "latitude", "lon", "longitude")]

    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dimensions, got {spatial_dims}")

    lat_dim, lon_dim = spatial_dims
    lat_size, lon_size = ds.sizes[lat_dim], ds.sizes[lon_dim]

    spatial_total = lat_size * lon_size
    chunk_area = spatial_total // target_chunks
    aspect_ratio = lon_size / lat_size

    chunk_lat = int((chunk_area / aspect_ratio) ** 0.5)
    chunk_lon = int(chunk_area / chunk_lat)

    chunks[lat_dim] = min(chunk_lat, lat_size)
    chunks[lon_dim] = min(chunk_lon, lon_size)

    # Preserve full chunking for other dims (e.g. time)
    for dim in dims:
        if dim not in chunks:
            chunks[dim] = ds.sizes[dim]

    return chunks

def hash_grid(ds):
    """Hash lat/lon coordinates for reproducible weight filenames."""
    lat = ds["lat"].values
    lon = ds["lon"].values
    key = f"{lat.tobytes()}{lon.tobytes()}"
    return hashlib.md5(key.encode()).hexdigest()
    
def load_dataset(path_or_paths):
    """
    Loads a NetCDF dataset from a file, directory, or list of files.
    Returns an xarray.Dataset.
    """
    if isinstance(path_or_paths, (list, tuple)):
        # List of files
        return xr.open_mfdataset(path_or_paths, combine='by_coords')

    if os.path.isfile(path_or_paths) and path_or_paths.endswith(".nc"):
        # Single file
        return xr.open_dataset(path_or_paths)

    if os.path.isdir(path_or_paths):
        # Directory of files
        nc_files = sorted(glob.glob(os.path.join(path_or_paths, "*.nc")))
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found in directory: {path_or_paths}")
        return xr.open_mfdataset(nc_files, combine='by_coords')

    raise ValueError(f"Invalid input: {path_or_paths}")

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
        pypath = env["workflow_resources"]
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
    
    target_ds = load_dataset(target_path)
    source_ds = load_dataset(source_path)
  
    if "lat" not in target_ds or "lon" not in target_ds:
        raise ValueError("Target dataset must include 'lat' and 'lon' coordinates.")
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
    if any(dim in target_ds.chunks and target_ds.chunks[dim] is not None for dim in target_ds.dims):
        target_chunks = {
            dim: target_ds.chunks[dim][0] if dim in target_ds.chunks else None
            for dim in target_ds.dims}
    else:
        target_chunks = compute_dynamic_chunks(target_ds)

    regridder = xe.Regridder(source_ds, target_grid, method, filename=weights_file, reuse_weights=True)

    # 2. Regrid variables
    regridded = xr.Dataset()
    for var in variables:
        regridded_var = regridder(source_ds[var])  # Don't pre-chunk source
        regridded_var = regridded_var.chunk(target_chunks)  # Apply target chunking here
        regridded_var.attrs.update(source_ds[var].attrs)
        regridded[var] = regridded_var

    # 3. Add missing coords
    for coord in target_ds.coords:
        if coord not in regridded.coords:
            regridded[coord] = target_ds[coord]

    # 4. Apply dynamic chunking AFTER regridding
    regridded = regridded.chunk(target_chunks)

    # 5. Copy global attrs
    regridded.attrs.update(target_ds.attrs)
    return regridded

def regrid_wrapper(target_path, source_path, source_vars=None, daterange=None):
    """
    Wrapper function to regrid source dataset to target dataset grid.
    Parameters:
    - target_path: path to target dataset (file, dir, or list of files)
    - source_path: path to source dataset (file, dir, or list of files)
    - source_vars: list of variable names to regrid (default: all)
    - daterange: tuple of (start_date, end_date) to subset source data by time
    Returns:
    - regridded xarray.Dataset
    """

    # 1. Get the weights file
    weight_file = get_regrid_weights(target_path, source_path)

    # 2. Open the datasets    
    target_ds = load_dataset(target_path)
    source_ds = load_dataset(source_path)

    # 3 Subset dataset by time (daterange) and products if needed    
    if daterange:
        dates = get_dates(daterange,format='datetime')
        source_ds = source_ds.sel(time=dates,method='nearest',tolerance=np.timedelta64(1, 'D'))
        
    if source_vars:
        source_ds = source_ds[source_vars]

    # 4. Regrid datasets
    regridded_ds = regrid_dataset(weights_file=weight_file,target_ds=target_ds,source_ds=source_ds,variables=source_vars)

    return regridded_ds