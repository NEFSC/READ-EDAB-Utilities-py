import numpy as np
import xarray as xr
import os
from pathlib import Path
from utilities import parse_dataset_info
from utilities import get_python_dir
from utilities import get_dates


def daylength_calculation(latitude,doy):
    
    # Check that the input data are np arrays
    if not isinstance(doy,np.ndarray):
        doy = np.array(doy,dtype=float)

    if not isinstance(latitude,np.ndarray):
        latitude = np.array(latitude,dtype=float)

    # Check the input array sizes
    if doy.size > 1 and latitude.size > 1 and doy.size != latitude.size:
        print("Error: Input latitude and DOY arrays have different lengths")

    # Check bounds of the input data and replace with NA if invalid
    doy = np.where((doy < 1) | (doy > 366), np.nan, doy)
    latitude = np.where((latitude < -90.0) | (latitude > 90.0), np.nan, latitude)
        
    
    dtor = np.pi/180.0 # Convert degrees to radians
    angle = dtor * 360.0 * (doy-1)/365
    
    declination = (0.39637 - 22.9133*np.cos(angle)
                   + 4.02543*np.sin(angle)
                   -0.3872*np.cos(2*angle)
                   +0.052*np.sin(2*angle))

    radians = -1.0 * np.tan(latitude * dtor) * np.tan(declination * dtor)
    radians = radians.clip(min=-1.0, max=1.0)  # Ensure radians is at least -1.0 and most 1.0
    
    factor = 0.133333
    day_length = factor * np.arccos(radians) / dtor
    
    return day_length

def get_daylength(latitude_grid_path, daylength_dir=None,doy=None):

    if not daylength_dir:
        pypath = get_python_dir(resources=True)
        daylength_dir = Path(pypath) / "daylength"
    os.makedirs(daylength_dir, exist_ok=True)

    target_parser = parse_dataset_info(latitude_grid_path)
    target_label = f"{target_parser['dataset']}_{target_parser['dataset_map']}"

    cache_path = Path(daylength_dir) / f"daylength_{target_label}.nc"

    # Load or compute
    if cache_path.exists():
        ds = xr.open_dataset(cache_path)
    else:
        lat_ds = xr.open_mfdataset(os.path.join(latitude_grid_path,'*.nc'))
        latitudes = lat_ds["lat"].values
        days = np.arange(1, 366)

        # Compute full grid
        daylength = np.empty((365, len(latitudes)))
        for i, doy_val in enumerate(days):
            daylength[i, :] = daylength_calculation(latitudes, doy_val)

        ds = xr.Dataset(
            {"daylength": (["day", "lat"], daylength)},
            coords={"day": days, "lat": latitudes}
        )
        ds.to_netcdf(cache_path)
        print(f"💾 Cached daylength grid: {cache_path}")

    # Return full or partial daylength
    if doy is None:
        return ds["daylength"]
    else:
        return ds.sel(day=doy)["daylength"]

    