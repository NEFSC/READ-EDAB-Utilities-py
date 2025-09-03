import pandas as pd
import xarray as xr
import os
import numpy as np
import netCDF4

from datetime import datetime

from utilities import regrid_wrapper
from utilities import get_daylength
from utilities import get_nc_prod
from utilities import parse_dataset_info
from utilities import get_fill_value
from utilities import build_product_attributes


def build_pp_date_map(dates=None, get_date_prod="CHL", chl_dataset=None, sst_dataset=None, par_dataset=None, verbose=False):
    """
    Constructs a date→(CHL, SST, PAR, PPD) file path mapping using get_prod_files.

    Parameters:
        dates (list of str): List of dates (YYYYMMDD) to process.
        get_date_prod (str): Which product to use for date filtering.
        chl_dataset, sst_dataset, par_dataset (str): Dataset identifiers.   
    Returns:
        dict: Mapping of date → (chl_path, sst_path, par_path,ppd_output_path)
    """
    
    import os
    from utilities import get_source_file_dates, get_prod_files, get_dates, make_product_output_dir,parse_dataset_info
    
    # Get files
    chl_files = get_prod_files("CHL", dataset=chl_dataset)
    sst_files = get_prod_files("SST", dataset=sst_dataset)
    par_files = get_prod_files("PAR", dataset=par_dataset)
    
    # Extract metadata from first CHL file
    if not chl_files:
        raise ValueError("No CHL files found to derive metadata.")
    info = parse_dataset_info(chl_files[0])

    # Extract valid date strings
    chl_dates = get_source_file_dates(chl_files, format="yyyymmdd", placeholder=None)
    sst_dates = get_source_file_dates(sst_files, format="yyyymmdd", placeholder=None)
    par_dates = get_source_file_dates(par_files, format="yyyymmdd", placeholder=None)

    # Build reverse maps (date → file)
    chl_map = {d: f for d, f in zip(chl_dates, chl_files) if d}
    sst_map = {d: f for d, f in zip(sst_dates, sst_files) if d}
    par_map = {d: f for d, f in zip(par_dates, par_files) if d}
    #ppd_map = {d: f for d, f in zip(ppd_dates, ppd_files) if d}
    # Build PPD file map
    output_dir = make_product_output_dir('CHL','PPD',dataset=chl_dataset)
    ppd_map = {}
    for date in chl_dates:
        filename = f"D_{date}-{info['dataset']}-{info['version']}-{info['dataset_map']}-PPD.nc"
        ppd_map[date] = os.path.join(output_dir, filename)

    # Normalize get_date_prod
    prod_key = (get_date_prod or "CHL").upper()

    # Get date list
    target_dates = get_dates(dates)
    if target_dates is None:
        if prod_key == "CHL":
            target_dates = sorted(chl_map.keys())
        elif prod_key == "SST":
            target_dates = sorted(sst_map.keys())
        elif prod_key == "PAR":
            target_dates = sorted(par_map.keys())
        else:
            raise ValueError(f"Invalid get_date_prod value: '{get_date_prod}'")

     # Build mapping for dates present in CHL and all other inputs
    pp_data_map = {}
    for date in target_dates:
        if date in chl_map and date in sst_map and date in par_map:
            chl_path = chl_map[date]
            sst_path = sst_map[date]
            par_path = par_map[date]
            ppd_path = os.path.join(output_dir, f"D_{date}-{info['dataset']}-{info['version']}-{info['dataset_map']}-PPD.nc")

            # Check if PPD file exists and is newer than all inputs
            if os.path.exists(ppd_path):
                ppd_mtime = os.path.getmtime(ppd_path)
                input_mtimes = [os.path.getmtime(p) for p in [chl_path, sst_path, par_path]]
                is_up_to_date = all(ppd_mtime > m for m in input_mtimes)
            else:
                is_up_to_date = False

            pp_data_map[date] = (chl_path, sst_path, par_path, ppd_path, is_up_to_date)
        else:
            print(f"⚠ Missing data for {date}: CHL {'✅' if date in chl_map else '❌'}  SST {'✅' if date in sst_map else '❌'}  PAR {'✅' if date in par_map else '❌'}")

    print(f"📅 Mapped {len(pp_data_map)} dates with complete product files.")
    return pp_data_map

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def make_pp_files(pp_data_map, overwrite=False):
    """
    Create primary productivity files using the pp_data_map. Skip up-to-date outputs unless overwrite=True.

    Parameters:
        pp_data_map (dict): Input and output file names from build_pp_date_map
        overwrite (bool): If True, regenerate the output file even if output is newer
    """
    for date, (chl, sst, par, ppd, is_up_to_date) in pp_data_map.items():
        if is_up_to_date and not overwrite:
            print(f"⏩ Skipping {date}: {os.path.basename(ppd)} is up-to-date.")
            continue

        print(f"🛠️ Processing {date} → {os.path.basename(ppd)}")
        process_daily_pp(date,chl,sst,par,ppd)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def vgpm_models(chl,sst,par,day_length):
    """
    Calculate Primary Productivity using the Behrenfeld-Falkowski VGPM Model (1997) and a modified Eppley-based formulation.

    INPUTS:
        chl         : float or np.ndarray — Satellite chlorophyll (mg m⁻³)
        sst         : float or np.ndarray — Sea surface temperature (°C)
        par         : float or np.ndarray — Surface PAR (Einstein m⁻² d⁻¹)
        day_length  : float or np.ndarray — Day length (hours)

    RETURNS:
        pp_eppley        : Daily PP using Eppley-based Pb_opt (gC m⁻² d⁻¹)
        pp_vgpm          : Daily PP using VGPM Pb_opt (gC m⁻² d⁻¹)
        kdpar            : Diffuse attenuation coefficient for PAR (m⁻¹)
        chl_euphotic     : Areal chlorophyll within euphotic layer (mg m⁻²)
        euphotic_depth   : Euphotic depth (m)
    
    REFERENCES:
        VGPM Reference:
        Behrenfeld, M.J. and P.G. Falkowski. 1997. Photosynthetic Rates Derived from Satellite-based Chlorophyll Concentration. Limnol. Oceanogr., 42[1]:1-20.

        Pbopt/temperature Reference:
        Antoine, D., J.-M. Andre, and A. Morel. 1996. Oceanic primary production. 2. Estimation at global
           scale from satellite (coastal zone color scanner) chlorophyll. Global Biogeochem. Cycles 10:57- 69.

    COPYRIGHT: 
        Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
        Northeast Fisheries Science Center, Narragansett Laboratory.
        This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
        This routine is provided AS IS without any express or implied warranties whatsoever.

    AUTHOR:
      This program was written on June 02, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
    MODIFICATION HISTORY:
        June 02, 2025 - KJWH: Initial code written and adapted from John E. O'Reilly's PP_VGPM2.pro (IDL) code
        June 23, 2025 - KJWH: Modified to return both VGPM and Eppley
    
    """

    def wrap_output(data, template):
        if isinstance(template, xr.DataArray):
            return xr.DataArray(data, dims=template.dims, coords=template.coords)
        elif isinstance(template, np.ndarray):
            return data
        elif np.isscalar(template):
            return np.asscalar(data) if np.ndim(data) == 0 else data
        else:
            return data  # fallback

    # Convert inputs to arrays if they're scalars
    if np.isscalar(chl):
        chl = np.asarray(chl)
    if np.isscalar(sst):
        sst = np.asarray(sst)
    if np.isscalar(par):
        par = np.asarray(par)
    if np.isscalar(day_length):
        day_length = np.asarray(day_length)

    # Ensure arrays and mask invalid inputs
    chl = np.where((chl > 0) & ~np.isnan(chl), chl, np.nan)
    
    # Calculate total chlorophyll within the euphotic zone (mg m⁻²)
    chl_euphotic = np.where(
        chl < 1.0,
        38.0 * chl**0.425,
        40.2 * chl**0.507
    )  
    
    # Calculate depth of euphotic layer (m)
    zeu = np.where(
        chl_euphotic <= 10.0,
        200.0 * chl_euphotic**-0.293,
        568.2 * chl_euphotic**-0.746
    )

    # Calculate Kd_PAR (m⁻¹)
    #kdpar = -np.log(0.01) / zeu
    #kdpar = xr.where(zeu > 0, -np.log(0.01) / zeu, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        kdpar = xr.where(zeu > 0, -np.log(0.01) / zeu, np.nan)

    
    # Eppley formulation for Pb_opt (gC gChl⁻¹ h⁻¹)
    Pb_opt_eppley = 1.54 * 10.0**(0.0275 * sst - 0.07)

    # Behrenfeld VGPM polynomial formulation for Pb_opt
    sst_valid = np.where(sst > 1.e-5, sst, np.nan)
    coeffs = [1.2956, 2.749e-1, 6.17e-2, -2.05e-2, 2.462e-3, -1.348e-4, 3.4132e-6, -3.27e-8]
    sst_poly = sum(c * sst_valid**i for i, c in enumerate(coeffs))

    Pb_opt_vgpm = np.where(
        sst > 28.5, 4.0,
        np.where(sst < -1.0, 1.13, sst_poly)
    )
    
    # Light limitation factor
    light_lim = par / (par + 4.1)
    scalar = 0.66125 * 0.001 # (gC m⁻² d⁻¹) and unit conversion

    # Final primary production cac;iatopm (gC m⁻² d⁻¹)
    pp_eppley = scalar * Pb_opt_eppley * light_lim * chl * zeu * day_length
    pp_vgpm   = scalar * Pb_opt_vgpm   * light_lim * chl * zeu * day_length
        
    return (
        wrap_output(pp_eppley, chl),
        wrap_output(pp_vgpm, chl),
        wrap_output(kdpar, chl),
        wrap_output(chl_euphotic, chl),
        wrap_output(zeu, chl)
    )

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def process_daily_pp(chl, sst, par, daylength, output_pp_file):
    """
    Process daily CHL, SST, and PAR inputs:
    - Checks if output exists or needs update
    - Regrids SST and PAR
    - Computes or loads daylength
    - Saves processed output
    """
    
    # 1️⃣ Check the chl, sst, par and daylength inputs
    
    # 2️⃣  Create a mask of missing data
    def generate_valid_mask(*arrays):
        mask = arrays[0].notnull()
        for arr in arrays[1:]:
            mask = mask & arr.notnull()  # ✅ safe: creates new object each time
        return mask
    valid_mask = generate_valid_mask(chl, sst, par)
    assert valid_mask.shape == chl.shape, f"valid_mask shape mismatch: expected {chl.shape}, got {valid_mask.shape}"
    
    if valid_mask.shape != chl.shape:
        raise ValueError("valid_mask shape mismatch: expected {}, got {}".format(chl.shape, valid_mask.shape))

    chl_valid = chl.where(valid_mask)
    sst_valid = sst.where(valid_mask)
    par_valid = par.where(valid_mask)
    day_length_valid = daylength.where(valid_mask)

    # 3️⃣ Calculate PP on non-missing data and create dataset    
    pp_eppley, pp_vgpm, kdpar, chl_euphotic, zeu = vgpm_models(
        chl_valid, sst_valid, par_valid, day_length_valid)
        
    # 4️⃣ Create output dataset
    def ensure_shape(var, template):
        if isinstance(var, xr.DataArray):
            var = var.data
        if var.shape != template.shape:
            return np.broadcast_to(var, template.shape)
        return var
    
    ds_out = xr.Dataset({
        "PP_Eppley": (chl.dims, ensure_shape(pp_eppley, chl)),
        "PP_VGPM": (chl.dims, ensure_shape(pp_vgpm, chl)),
        "Kd_PAR": (chl.dims, ensure_shape(kdpar, chl)),
        "Chl_Euphotic": (chl.dims, ensure_shape(chl_euphotic, chl)),
        "Z_eu": (chl.dims, ensure_shape(zeu, chl))
    }, coords=chl.coords)

    # 5️⃣ Add variable attributes
    encoding = {}
    for var_name in ds_out.data_vars:
        var = ds_out[var_name]
        fill_value = get_fill_value(var)

        # Mask invalid pixels
        var_masked = var.where(valid_mask, fill_value)

        # If dtype is float64 or float32, cast as float32
        if np.issubdtype(var_masked.dtype, np.floating):
            var_masked = var_masked.astype("float32")
            encoding[var_name] = {
                "_FillValue": np.float32(fill_value),
                "dtype": "float32",
                "zlib": True,
                "complevel": 4
            }
        else:
            encoding[var_name] = {
                "_FillValue": fill_value,
                "zlib": True,
                "complevel": 4
            }

        # Assign masked and casted data back
        ds_out[var_name] = var_masked
        ds_out[var_name].attrs = build_product_attributes(var_name)
        

    # 6️⃣ Add Global metadata
    ds_out.attrs['creation_date'] = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')

    #ds_out.attrs = chl_ds.attrs.copy()

    

    # 🔟 Save results
    ds_out.to_netcdf(output_pp_file,encoding=encoding)
    print(f"💾 Saved: {output_pp_file}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_pp_pipeline(
    chl_dataset=None,
    sst_dataset=None,
    par_dataset=None,
    daterange=None,
    daylength_dir=None,
    overwrite=False,
    verbose=True
):
    """
    Runs the full primary productivity pipeline:
    - Builds date→file map
    - Regrids SST and PAR in batch
    - Processes daily CHL/SST/PAR into PPD outputs

    Parameters:
        chl_dataset, sst_dataset, par_dataset (str): Dataset identifiers
        daterange (list of str): Dates to process (YYYYMMDD)
        overwrite (bool): If True, reprocess even if output is up-to-date
        verbose (bool): If True, print progress
    """
    from utilities import build_pp_date_map, parse_dataset_info
    import xarray as xr
    import pandas as pd
    import os

    # 1️⃣ Build date→file map
    pp_data_map = build_pp_date_map(
        dates=daterange,
        chl_dataset=chl_dataset,
        sst_dataset=sst_dataset,
        par_dataset=par_dataset,
        verbose=verbose
    )

    # 2️⃣ Filter dates to run
    dates_to_run = [
        date for date, (_, _, _, _, up_to_date) in pp_data_map.items()
        if overwrite or not up_to_date
    ]
    if verbose:
        print(f"🚀 Processing {len(dates_to_run)} dates...")

    if not dates_to_run:
        print("✅ All outputs are up-to-date. Nothing to run.")
        return

    # 3️⃣ Collect input files
    chl_files = [pp_data_map[d][0] for d in dates_to_run]
    sst_files = [pp_data_map[d][1] for d in dates_to_run]
    par_files = [pp_data_map[d][2] for d in dates_to_run]
    output_files = [pp_data_map[d][3] for d in dates_to_run]

    # 4️⃣ Get input data information
    chl_info = parse_dataset_info(chl_files[0])
    sst_info = parse_dataset_info(sst_files[0])
    par_info = parse_dataset_info(par_files[0])

    chl_nc_var = get_nc_prod(chl_info['dataset'],'CHL')
    sst_nc_var = get_nc_prod(sst_info['dataset'],'SST')
    par_nc_var = get_nc_prod(par_info['dataset'],'PAR')

    # 5️⃣ Batch regrid SST and PAR
    sst_ds = regrid_wrapper(chl_files, sst_files, source_vars=[sst_nc_var], daterange=dates_to_run)
    par_ds = regrid_wrapper(chl_files, par_files, source_vars=[par_nc_var], daterange=dates_to_run)

    # 6️⃣ Daily loop
    for date, chl_file, output_file in zip(dates_to_run, chl_files, output_files):
        
        chl_ds = xr.open_dataset(chl_file)
        chl = chl_ds[chl_nc_var]
        sst = sst_ds.sel(time=pd.to_datetime(date))[sst_nc_var]
        par = par_ds.sel(time=pd.to_datetime(date))[par_nc_var]

        chl_date = pd.to_datetime(chl["time"].values[0]).date()
        sst_date = pd.to_datetime(sst["time"].values[0]).date()
        par_date = pd.to_datetime(par["time"].values[0]).date()

        if chl_date == sst_date:
            sst["time"] = chl["time"]
        else:
            raise ValueError(f"Date mismatch: chl={chl_date}, sst={sst_date}")

        if chl_date == par_date:
            par["time"] = chl["time"]
        else:
            raise ValueError(f"Date mismatch: chl={chl_date}, sst={sst_date}")

        chl, sst, par = xr.align(chl, sst, par, join="exact")  # or "inner" if needed

        # Compute or load daylength
        doy = pd.Timestamp(date).dayofyear
        daylength_ds = get_daylength(chl_file, daylength_dir=daylength_dir, dayofyear=doy)

        process_daily_pp(date, chl, sst, par, daylength_ds,output_file)

    print("🎉 Pipeline complete.")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
