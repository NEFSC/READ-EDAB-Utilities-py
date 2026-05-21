import numpy as np
import xarray as xr
from pyproj import Geod
from joblib import Parallel, delayed
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)


def pixel_area(lat, lon, land_mask=None):

    """
    PURPOSE:
        Computes the geodesic pixel area using WGS84 ellipsoid based on 1D or 2D lat/lon arrays.

    REQUIRED INPUTS:
        lat (np.ndarray): Latitude array (1D or 2D)
        lon (np.ndarray): Longitude array (1D or 2D)

    RETURNS:
        pixel_area (xr.DataArray): Pixel area in square kilometers (km²) with same shape as input grida map
        
    
    REFERENCES:
        

    COPYRIGHT: 
        Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
        Northeast Fisheries Science Center, Narragansett Laboratory.
        This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
        This routine is provided AS IS without any express or implied warranties whatsoever.

    AUTHOR:
      This program was written on July 08, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
    MODIFICATION HISTORY:
        July 08, 2025 - KJWH: Initial code written and adapted from IDL code maps_pixearea.pro
        July 16, 2025 - KJWH: Updated to work with 1D and 2D input arrays
        
    
    """
    
    geod = Geod(ellps="WGS84")

    # Step 1: Expand lat/lon to 2D grid if needed
    if lat.ndim == 1 and lon.ndim == 1:
        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

        # Estimate pixel edge coordinates
        lat_edges = (lat[:-1] + lat[1:]) / 2
        lon_edges = (lon[:-1] + lon[1:]) / 2

        lat_bounds = np.concatenate(([lat[0] - (lat_edges[0] - lat[0])], lat_edges, [lat[-1] + (lat[-1] - lat_edges[-1])]))
        lon_bounds = np.concatenate(([lon[0] - (lon_edges[0] - lon[0])], lon_edges, [lon[-1] + (lon[-1] - lon_edges[-1])]))

        lat_b, lon_b = np.meshgrid(lat_bounds, lon_bounds, indexing="ij")
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d = lat
        lon2d = lon

        # NotImplementedError("2D edge estimation for curvilinear grids is not yet implemented.")
    else:
        raise ValueError("Latitude and longitude arrays must be either both 1D or both 2D.")


    # Step 2: Compute pixel areas in parallel
    n_lat = lat2d.shape[0]
    n_lon = lat2d.shape[1]
    n_pixel_lat = lat2d.shape[0] - 1
    n_pixel_lon = lat2d.shape[1] - 1
    
    area_km2 = np.empty((n_pixel_lat, n_pixel_lon))

    # Step 3: Apply a land mask (if provided)
    if land_mask is not None:
    # Convert to NumPy array if it's an xarray object
        if hasattr(land_mask, "values"):
            land_mask = land_mask.values
    
        # Interpolate to match the pixel area grid shape
        if land_mask.shape == (n_lat, n_lon):
            mask_pixel = land_mask[:n_pixel_lat, :n_pixel_lon]
        else:
            raise ValueError("land_mask shape must match lat/lon grid shape.")
    else:
        mask_pixel = np.zeros((n_pixel_lat, n_pixel_lon), dtype=bool)

    # Function to compute the pixel area (so that it can be run in parallel)
    def compute_pixel_area(i, j):
        lats_box = [lat2d[i, j], lat2d[i, j + 1], lat2d[i + 1, j + 1], lat2d[i + 1, j]]
        lons_box = [lon2d[i, j], lon2d[i, j + 1], lon2d[i + 1, j + 1], lon2d[i + 1, j]]
        area, _ = geod.polygon_area_perimeter(lons_box, lats_box)
        return i, j, abs(area) / 1e6  # km²

    # Step 4: Run the area calculation in parallel
    tasks = [(i, j) for i in range(lat2d.shape[0] - 1) for j in range(lon2d.shape[1] - 1)]
    results = Parallel(n_jobs=-1)(delayed(compute_pixel_area)(i, j) for i, j in tasks)
    for i, j, area in results:
        area_km2[i, j] = area

    # Step 5: Interpolate edges to match full grid
    area_full = np.empty((n_lat, n_lon))
    area_full[:n_pixel_lat, :n_pixel_lon] = area_km2
    area_full[n_pixel_lat:, :] = area_full[n_pixel_lat-1:n_pixel_lat, :]  # last row
    area_full[:, n_pixel_lon:] = area_full[:, n_pixel_lon-1:n_pixel_lon]  # last column
    area_full[n_pixel_lat - 1, n_pixel_lon - 1] = (area_full[n_pixel_lat - 1, n_pixel_lon - 2] + area_full[n_pixel_lat - 2, n_pixel_lon - 1]) / 2

    # Step 6: Package result as xarray DataArray
    return xr.DataArray(
        area_full,
        coords={"lat": lat2d[:, 0], "lon": lon2d[0, :]},
        dims=["lat", "lon"],
        name="pixel_area_km2",
        attrs={"units": "km²", "description": "Geodesic pixel area with full grid resolution"}
    )

    


    