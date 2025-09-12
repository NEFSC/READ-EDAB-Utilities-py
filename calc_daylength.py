import numpy as np
import xarray as xr
import os
from pathlib import Path
from utilities import parse_dataset_info
from bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

def _validate_daylength_grid(ds):
    assert "daylength" in ds, "Missing 'daylength' variable in dataset."
    assert ds["day"].max() == 366, "Dataset must include all 366 days (leap year inclusive)."
    assert ds["daylength"].shape[0] == 366, "'daylength' must have 366 entries along the day axis."


def daylength_calculation(latitude,doy):
    """
    Compute daylength (hours of daylight) for given latitude(s) and day-of-year(s).

    Parameters:
        latitude (float or np.ndarray): Latitude(s) in degrees (-90 to 90)
        doy (int or np.ndarray): Day-of-year(s), 1–366

    Returns:
        np.ndarray: Daylength in hours
    """
    # Check that the input data are np arrays
    if not isinstance(doy,np.ndarray):
        doy = np.array(doy,dtype=float)

    if not isinstance(latitude,np.ndarray):
        latitude = np.array(latitude,dtype=float)

    # Check the input array sizes
    if doy.size > 1 and latitude.size > 1 and doy.size != latitude.size:
        raise ValueError("Input latitude and DOY arrays must have the same length when both are arrays.")

    # Check bounds of the input data and replace with NA if invalid
    if np.any((doy < 1) | (doy > 366)):
        raise ValueError("DOY must be between 1 and 366.")

    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("Latitude must be between -90 and 90 degrees.") 
    
    dtor = np.pi/180.0 # Convert degrees to radians
    angle = dtor * 360.0 * (doy-1)/365.25 # Using 365.25 to account for leap years
    
    declination = (0.39637 - 22.9133*np.cos(angle)
                   + 4.02543*np.sin(angle)
                   -0.3872*np.cos(2*angle)
                   +0.052*np.sin(2*angle))

    radians = -1.0 * np.tan(latitude * dtor) * np.tan(declination * dtor)
    radians = radians.clip(min=-1.0, max=1.0)  # Ensure radians is at least -1.0 and most 1.0
    
    factor = 0.133333
    day_length = factor * np.arccos(radians) / dtor
    
    return day_length

def get_daylength(latitude_grid_path, daylength_dir=None,dayofyear=None):
    """
    Compute daylength (hours of daylight) for each day of the year (1–366)
    and each latitude in the input grid.

    Parameters:
        latitude_grid_path (str): Path to input dataset containing latitude grid
        daylength_dir (str or Path, optional): Directory to cache daylength files
        dayofyear (int, list[int], np.ndarray, or xarray.DataArray, optional): 
            Day-of-year(s) to extract. If None, returns full (day, lat) grid.

    Returns:
        xarray.DataArray: Daylength values with shape depending on `doy`:
            - (lat,) if single day
            - (day, lat) if multiple days
            - (time, lat) if `dayofyear` is a DataArray with time dimension
    """

    if not isinstance(latitude_grid_path, (str, Path)):
        raise TypeError("latitude_grid_path must be a string or Path object.")

    if not daylength_dir:
        pypath = env["workflow_resources"]
        daylength_dir = Path(pypath) / "daylength"
    os.makedirs(daylength_dir, exist_ok=True)

    target_parser = parse_dataset_info(latitude_grid_path)
    target_label = f"{target_parser['dataset']}_{target_parser['dataset_map']}"

    cache_path = Path(daylength_dir) / f"daylength_{target_label}.nc"

    # Load or compute
    if cache_path.exists():
        da = xr.open_dataset(cache_path)
    else:
        lat_ds = xr.open_mfdataset(os.path.join(latitude_grid_path,'*.nc'),combine='by_coords')
        latitudes = lat_ds["lat"].values
        days = np.arange(1, 367)

        # Compute full grid
        daylength = np.empty((366, len(latitudes)))
        for i, doy_val in enumerate(days):
            daylength[i, :] = daylength_calculation(latitudes, doy_val)

        da = xr.Dataset(
            {"daylength": (["day", "lat"], daylength)},
            coords={"day": days, "lat": latitudes}
        )
        da.attrs["units"] = "hours"
        da.attrs["description"] = "Daylength computed from solar geometry"
        
        da.to_netcdf(cache_path)
        print(f"💾 Cached daylength grid: {cache_path}")

    try:
        _validate_daylength_grid(da)
    except AssertionError as e:
        raise ValueError(f"Invalid daylength dataset: {e}")


    # Return full or partial daylength
    if dayofyear is None:
        return da["daylength"]
    else:
        valid_days = da["day"].values
        if isinstance(dayofyear, (np.ndarray, xr.DataArray)):
            dayofyear = xr.where(dayofyear > valid_days.max(), valid_days.max(), dayofyear)
        elif isinstance(dayofyear, int):
            dayofyear = min(dayofyear, valid_days.max())
        
        return da.sel(day=dayofyear)["daylength"]


def get_daylength_grid(latitude_grid_path):
    """
    Broadcast daylength values to match the shape of input_ds (time, lat, lon).

    Parameters:
        latitude_grid_path (str): Path to input dataset containing latitude grid

    Returns:
        xarray.DataArray: Daylength values with shape (time, lat, lon)
    """
    
    # Get the daylength
    daylength_ds = get_daylength(latitude_grid_path)

    input_ds = xr.open_mfdataset(os.path.join(latitude_grid_path,'*.nc'))
    
    # Extract coordinates
    time = input_ds["time"]
    lat = input_ds["lat"]
    lon = input_ds["lon"]

    # Convert time to day-of-year
    doy = xr.DataArray(time.dt.dayofyear, dims="time")

    # Select daylength values for each DOY
    daylength_per_day = daylength_ds.sel(day=doy)

    # Broadcast to full grid
    daylength_3d = daylength_per_day.expand_dims(lon=lon).transpose("time", "lat", "lon")

    return daylength_3d