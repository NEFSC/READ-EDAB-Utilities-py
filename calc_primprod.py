import pandas as pd
import xarray as xr
import os
import sys
import time
import numpy as np
from datetime import datetime

from utilities import regrid_wrapper, subset_dataset
from utilities import get_daylength
from utilities import get_nc_prod
from utilities import get_fill_value

from utilities import build_product_attributes
from utilities import get_reference_metadata
from utilities import get_geospatial_metadata
from utilities import get_lut_metadata
from utilities import get_temporal_metadata
from utilities import get_summary_metadata
from utilities import get_default_metadata
from utilities import get_source_metadata
from utilities import parse_dataset_info
from utilities import get_source_file_dates, get_prod_files, get_dates, make_product_output_dir


"""
CALC_PRIMPROD create daily net primary productivity (PPD) files from daily satellite chlorophyll (CHL), sea surface temperature (SST), and photosynthetic active radiation (PAR) inputs.
It uses the Behrenfeld-Falkowski VGPM Model (1997) and a modified Eppley-based formulation to compute primary productivity.
Main Functions:
    - run_pp_pipeline: Main function to run the full pipeline
    - build_pp_date_map: Constructs a date→(CHL, SST, PAR, PPD) file path mapping and determines which files need to be processed
    - process_daily_pp: Runs the vgpm_models function using daily CHL, SST, PAR and Daylength inputs, adds metadata, and saves the data as a netcdf file
    - vgpm_models: Calculates primary productivity using VGPM and Eppley models

Helper Functions:
    - validate_inputs: Validates that input data arrays are xarray.DataArray and have matching shapes

References:
    Antoine, D., Andre, J.-M., Morel, A., 1996. Oceanic primary production 2.Estimation at global scale from satellite (coastal zone color scanner) chlorophyll. Global Biogeochemical Cycles 10, 57–69.
    Behrenfeld, M.J., Falkowski, P.G., 1997. Photosynthetic rates derived from satellite-based chlorophyll concentration. Limnology and Oceanography 42, 1–20
    Eppley, R.W., 1972. Temperature and phytoplankton growth in the sea. Fishery Bulletin 70, 1063–1085
    Kirk, J.T.O., 1994. Light and Photosynthesis in Aquatic Ecosystems. Cambridge University Press, Cambridge, UK
    Morel, A., 1991. Light and marine photosynthesis: a spectral model with geochemical and climatological implications. Progress in Oceanography 96, 263–306
    Morel, A., Berthon, J.-F., 1989. Surface pigments, algal biomass profiles, and potential production of the euphotic layer: Relationships reinvestigated in view of remote-sensing applications. Limnology and Oceanography 34, 1545–1562. https://doi.org/10.4319/lo.1989.34.8.1545.

Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on June 02, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Jun 02, 2025 - KJWH: Initial code written and adapted from John E. O'Reilly's PP_VGPM2.pro (IDL) code
    Jun 23, 2025 - KJWH: Modified to return both VGPM and Eppley
    Sep 02, 2025 - KJWH: Added run_pp_pipeline function to run the full pipeline
                         Modified process_daily_pp to require xarray.DataArrays instead of filepaths
    Sep 03, 2025 - KJWH: Added metadata functions to process_daily_pp
                         Updated documentation
                         Added validate_inputs function to check input data types and shapes
"""

def validate_inputs(chl, sst, par, daylength):
    expected_type = xr.DataArray
    inputs = {"CHL": chl, "SST": sst, "PAR": par}
    
    # 1️⃣ Type check
    for name, arr in {**inputs, "Daylength": daylength}.items():
        if not isinstance(arr, expected_type):
            raise TypeError(f"{name} must be an xarray.DataArray, got {type(arr)}")

    # 2️⃣ Shape and dims check for CHL/SST/PAR
    ref_shape = chl.shape
    ref_dims = chl.dims
    for name, arr in inputs.items():
        if arr.shape != ref_shape:
            raise ValueError(f"{name} shape mismatch: expected {ref_shape}, got {arr.shape}")
        if arr.dims != ref_dims:
            raise ValueError(f"{name} dims mismatch: expected {ref_dims}, got {arr.dims}")

    # 3️⃣ Daylength check: must be 1D over latitude
    if daylength.ndim != 1:
        raise ValueError(f"Daylength must be 1D over latitude, got shape {daylength.shape}")
    
    lat_dim = next((dim for dim in daylength.dims if 'lat' in dim.lower()), None)
    if lat_dim is None:
        raise ValueError("Daylength must have a latitude dimension.")

    chl_lat_dim = next((dim for dim in chl.dims if 'lat' in dim.lower()), None)
    if chl_lat_dim and not np.array_equal(daylength[lat_dim], chl[chl_lat_dim]):
        raise ValueError("Daylength latitude values do not match CHL latitude values.")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_pp_date_map(dates=None, get_date_prod="CHL", chl_dataset=None, sst_dataset=None, par_dataset=None, subset=None, verbose=False):
    """
    Constructs a date→(CHL, SST, PAR, PPD) file path mapping using get_prod_files.

    Parameters:
        dates (list of str): List of dates (YYYYMMDD) to process.
        get_date_prod (str): Which product to use for date filtering.
        chl_dataset, sst_dataset, par_dataset (str): Dataset identifiers.   
        subset (str): The subset region (e.g. NES, NWA) to subset the data 
    Returns:
        dict: Mapping of date → (chl_path, sst_path, par_path,ppd_output_path)
    """
        
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
    output_dir = make_product_output_dir('CHL','PPD',dataset=chl_dataset,subset=subset)
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

            if is_up_to_date:
                print(f"✅ PSC is current for {date}")
            else:
                print(f"⏳ Need to process PSC for {date}")

            pp_data_map[date] = (chl_path, sst_path, par_path, ppd_path, is_up_to_date)
        else:
            print(f"⚠ Missing data for {date}: CHL {'✅' if date in chl_map else '❌'}  SST {'✅' if date in sst_map else '❌'}  PAR {'✅' if date in par_map else '❌'}")

    print(f"📅 Mapped {len(pp_data_map)} dates with complete product files.")
    return pp_data_map

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
def process_daily_pp(chl, sst, par, daylength, output_pp_file, history=None, sourceinfo=None):
    """
    Process daily CHL, SST, PAR and Daylength inputs:
    - Checks if data are valid and creates a valid data mask
    - Computes primary productivity using VGPM and VGPM-EPPLEY models
    - Adds global and product metadata
    - Saves processed output as a netcdf file
    """
    # 1️⃣ Check the chl, sst, par and daylength inputs
    validate_inputs(chl, sst, par, daylength)

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
    start_vgpm = time.perf_counter()
    pp_eppley, pp_vgpm, kdpar, chl_euphotic, zeu = vgpm_models(
        chl_valid, sst_valid, par_valid, day_length_valid)
    elapsed_vgpm = (time.perf_counter() - start_vgpm) / 60
    print(f"    Finished running the VGPM models after {elapsed_vgpm:.2f} minutes.")

        
    # 4️⃣ Create output dataset
    start_meta = time.perf_counter()
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
    # Global Attributes
    attrs = get_default_metadata(sheet="General")
    attrs = attrs | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
  
    # Geospatial Attributes
    attrs = attrs | get_geospatial_metadata(dataset=chl)

    # Temporal Attributes
    attrs = attrs | get_temporal_metadata(ds=ds_out)

    # Product Specific Attributes
    attrs["product_name"] = build_product_attributes("PPD")["long_name"]
    attrs = attrs | get_summary_metadata("PPD")
    attrs["references"] = get_reference_metadata(['VGPM_EPPLEY','KIRK','ZEU'],refs_only=True)

    # Source Metadata
    if history: attrs["history"] = history
    chl_source = get_source_metadata(sourceinfo["CHL"]["dataset"],dataset_version=sourceinfo["CHL"]["version"],source_prefix="source_chl")
    sst_source = get_source_metadata(sourceinfo["SST"]["dataset"],dataset_version=sourceinfo["SST"]["version"],source_prefix="source_sst")
    par_source = get_source_metadata(sourceinfo["PAR"]["dataset"],dataset_version=sourceinfo["PAR"]["version"],source_prefix="source_par")
    attrs = attrs | chl_source | sst_source | par_source
    
    # Add metadta to the output dataset
    ds_out.attrs = attrs
    elapsed_meta = (time.perf_counter() - start_meta) / 60
    print(f"    Finished building the dataset after {elapsed_meta:.2f} minutes.")

    # 7️⃣ Save results
    print (f"    Starting to write {output_pp_file}")
    start_netcdf = time.perf_counter()
    ds_out.to_netcdf(output_pp_file,encoding=encoding)
    elapsed_netcdf = (time.perf_counter() - start_netcdf) / 60
    print(f"    Finished writing {output_pp_file} after {elapsed_netcdf:.2f} minutes.")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_pp_pipeline(chl_dataset=None,
                    sst_dataset=None,
                    par_dataset=None,
                    daterange=None,
                    subset=None,
                    daylength_dir=None,
                    overwrite=False,
                    verbose=True,
                    logfile=None
                    ):
    """
    Runs the full primary productivity pipeline:
    - Builds date→file map
    - Regrids SST and PAR in batch
    - Processes daily CHL/SST/PAR into PPD outputs

    Parameters:
        chl_dataset, sst_dataset, par_dataset (str): Dataset identifiers
        daterange (list of str): Dates to process (YYYYMMDD)
        subset (str): The subset region (e.g. NES, NWA) to subset the data 
        overwrite (bool): If True, reprocess even if output is up-to-date
        verbose (bool): If True, print progress
        logfile (str): Optional file to log the progress
    """
    
    if logfile:
        print(f"logging progress in {logfile}")
        log = open(logfile, "w")
        sys.stdout = log
        sys.stderr = log


    # 1️⃣ Build date→file map
    pp_data_map = build_pp_date_map(dates=daterange,
                                    chl_dataset=chl_dataset,
                                    sst_dataset=sst_dataset,
                                    par_dataset=par_dataset,
                                    subset=subset,
                                    verbose=verbose)

    # 2️⃣ Filter dates to run
    dates_to_run = [
        date for date, (_, _, _, _, up_to_date) in pp_data_map.items()
        if overwrite or not up_to_date
    ]
    
    total_dates = len(dates_to_run)  # or however you're batching

    if verbose:
        print(f"🚀 Processing {total_dates} dates...")

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

    chl_source = get_source_metadata(chl_info["dataset"],dataset_version=chl_info["version"],source_prefix="source_chl")
    sst_source = get_source_metadata(sst_info["dataset"],dataset_version=sst_info["version"],source_prefix="source_sst")
    par_source = get_source_metadata(par_info["dataset"],dataset_version=par_info["version"],source_prefix="source_par")
    
    merged_info = {
        chl_info["product"]: chl_info,
        sst_info["product"]: sst_info,
        par_info["product"]: par_info
    }

    chl_nc_var = get_nc_prod(chl_info['dataset'],'CHL')
    sst_nc_var = get_nc_prod(sst_info['dataset'],'SST')
    par_nc_var = get_nc_prod(par_info['dataset'],'PAR')

    # 5️⃣ Batch regrid SST and PAR
    print(f"Regridding SST and PAR files to CHL grid...")
    sst_ds = regrid_wrapper(chl_files, sst_files, source_vars=[sst_nc_var], daterange=dates_to_run)
    par_ds = regrid_wrapper(chl_files, par_files, source_vars=[par_nc_var], daterange=dates_to_run)

    # 6️⃣ Daily loop
    print(f"Writing PPD files to {os.path.dirname(output_files[0])}")
    for i, (date, chl_file, sst_file, par_file, output_file) in enumerate(
        zip(dates_to_run, chl_files, sst_files, par_files, output_files), start=1):
        
        chl_ds = xr.open_dataset(chl_file)
        chl = chl_ds[chl_nc_var]
        
        target_time = pd.to_datetime(date)

        sst = sst_ds.sel(time=target_time, method='nearest', tolerance=pd.Timedelta('1D'), drop=False)[sst_nc_var]        
        par = par_ds.sel(time=target_time, method='nearest', tolerance=pd.Timedelta('1D'), drop=False)[par_nc_var]        
        
        chl_date = pd.to_datetime(chl["time"].values[0]).date()
        sst_date = pd.to_datetime(sst["time"].values).date()
        par_date = pd.to_datetime(par["time"].values).date()

        if chl_date == sst_date:
            sst = sst.expand_dims(time=chl["time"])
            sst["time"] = chl["time"]
        else:
            raise ValueError(f"Date mismatch: chl={chl_date}, sst={sst_date}")

        if chl_date == par_date:
            par = par.expand_dims(time=chl["time"])
            par["time"] = chl["time"]
        else:
            raise ValueError(f"Date mismatch: chl={chl_date}, sst={sst_date}")

        # Subset CHL and SST if requested
        if subset:
            if verbose: print(f"Subsetting to region: {subset}")
            chl = subset_dataset(chl, subset)
            sst = subset_dataset(sst, subset)
            par = subset_dataset(par, subset)
        
        chl, sst, par = xr.align(chl, sst, par, join="exact")  # or "inner" if needed

        # Compute or load daylength
        doy = pd.Timestamp(date).dayofyear
        daylength_ds = get_daylength(chl_file, daylength_dir=daylength_dir, dayofyear=doy)

         # Build the ppd history
        history = " ".join([f"{build_product_attributes('PPD')['long_name']} is calculated using the VGPM {get_reference_metadata('VGPM')[0]['citation']} and VGPM-EPPLEY {get_reference_metadata('VGPM_EPPLEY')[0]['citation']} models. ",
                f"The input chlorophyll file ({os.path.basename(chl_file)}) is from the {chl_source['source_chl_title']}. ",
                f"The input sea surface temperature file ({os.path.basename(sst_file)}) is from the {sst_source['source_sst_title']}. ",
                f"The input photosynthetic active radiation file ({os.path.basename(par_file)}) is from the {par_source['source_par_title']}. ",
                f"The SST and PAR data were regridded to the CHL grid using xESMF bilinear regridding {get_reference_metadata('xesmf')[0]['citation']}.",
                f"Day length was calculated according to Kirk (1994)"
        ])
        
        # 7️⃣ Process daily PP
        
        start_time = time.perf_counter()
        process_daily_pp(chl, sst, par, daylength_ds,output_file,history=history,sourceinfo=merged_info)
        elapsed = (time.perf_counter() - start_time) / 60
        if verbose: print(f"Finished writing {os.path.basename(output_file)}, file {i} of {total_dates} ({round((i / total_dates) * 100)}%) after {elapsed:.2f} minutes.")

    print("🎉 Primary productivity pipeline complete.")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
