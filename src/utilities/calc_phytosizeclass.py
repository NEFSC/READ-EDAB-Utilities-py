import pandas as pd
import xarray as xr
import os
import sys
import time
import gc
import numpy as np
from datetime import datetime
import xarray as xr
import pandas
import numpy as np

from utilities import bootstrap_environment
env = bootstrap_environment(verbose=False)

from utilities import regrid_wrapper, subset_dataset
from utilities import get_nc_prod
from utilities import build_product_attributes, get_default_metadata, get_lut_metadata, get_current_utc_timestamp
from utilities import get_geospatial_metadata, get_temporal_metadata, get_summary_metadata, get_reference_metadata, get_fill_value, get_source_metadata
from utilities import parse_dataset_info
from utilities import get_source_file_dates, get_prod_files, get_dates, get_filepath
from utilities import generate_logfile_path


"""
CALC_PHYTOSIZECLASS create daily phytoplankton size class files from daily satellite chlorophyll (CHL) and sea surface temperature (SST) inputs.
It uses the Northeast U.S. regionally tuned phytoplankton size class model based on Turner et al. (2021) to compute the micro, nano, and picoplankton fractions of total chlorophyll.
Main Functions:
    - run_psc_pipeline: Main function to run the full pipeline
    - build_psc_date_map: Constructs a date → (CHL, SST, PSC) file path mapping and determines which files need to be processed
    - process_daily_psc: Runs the psc_models function using daily CHL and SST inputs, adds metadata, and saves the data as a netcdf file
    - psc_turner: Calculates the micro, nano, and picoplankton fractions of total chlorophyll using the Turner et al. (2021) model

Helper Functions:
    - validate_inputs: Validates that input data arrays are xarray.DataArray and have matching shapes

Notes:
    To calculate phytoplankton size class chlorophyll contribution, multiply total chlorophyll with each size class fraction.

References:
    Turner, K.J., Mouw, C.B., Hyde, K.J.W., Morse, R., Ciochetto, A.B., 2021. Optimization and assessment of phytoplankton size class algorithms for ocean color data on the Northeast U.S. continental shelf. Remote Sensing of Environment 267, 112729. https://doi.org/10.1016/j.rse.2021.112729

Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on August 06, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Aug 06, 2024 - KJWH: Initial code written
    Aug 08, 2024 - KJWH: Updated how the coefficients are extracted and the data are returned
    Sep 11, 2025 - KJWH: Converted to a full pipeline modeled after calc_primprod.py
                         Updated documentation
    Oct 15, 2026 - KJWH: Added DualLogger and updated logfile output
    Oct 21, 2026 - KJWH: Forced h5netcdf engine in all .to_netcdf() calls

"""

class DualLogger:
    """
    Intercepts standard print statements and writes them to both the terminal 
    and a persistent log file simultaneously.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def validate_inputs(chl, sst):
    expected_type = xr.DataArray
    inputs = {"CHL": chl, "SST": sst}
    
    # 1️⃣ Type check
    for name, arr in {**inputs}.items():
        if not isinstance(arr, expected_type):
            raise TypeError(f"{name} must be an xarray.DataArray, got {type(arr)}")

    # 2️⃣ Shape and dims check for CHL/SST
    ref_shape = chl.shape
    ref_dims = chl.dims
    for name, arr in inputs.items():
        if arr.shape != ref_shape:
            raise ValueError(f"{name} shape mismatch: expected {ref_shape}, got {arr.shape}")
        if arr.dims != ref_dims:
            raise ValueError(f"{name} dims mismatch: expected {ref_dims}, got {arr.dims}")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_psc_date_map(dates=None, get_date_prod="CHL", chl_dataset=None, sst_dataset=None, map_subset=None, verbose=False):
    """
    Constructs a date→(CHL, SST, PSC) file path mapping using get_prod_files.

    Parameters:
        dates (list of str): List of dates (YYYYMMDD) to process.
        get_date_prod (str): Which product to use for date filtering.
        chl_dataset, sst_dataset (str): Dataset identifiers. 
        map_subset (str): The subset region (e.g. NES, NWA) to subset the data 
    Returns:
        dict: Mapping of date → (chl_path, sst_path, psc_output_path)
    """

    # Get files
    chl_files = get_prod_files("CHL", dataset=chl_dataset)
    sst_files = get_prod_files("SST", dataset=sst_dataset)
    
    # Extract metadata from first CHL file
    if not chl_files:
        raise ValueError("No CHL files found to derive metadata.")
    
    # Extract valid date strings
    chl_dates = get_source_file_dates(chl_files, format="yyyymmdd", placeholder=None)
    sst_dates = get_source_file_dates(sst_files, format="yyyymmdd", placeholder=None)

    # Build reverse maps (date → file)
    chl_map = {d: f for d, f in zip(chl_dates, chl_files) if d}
    sst_map = {d: f for d, f in zip(sst_dates, sst_files) if d}

    # Build PSC file map
    #output_dir = make_product_output_dir('CHL','PSC',dataset=chl_dataset,subset=subset)
    output_dir = get_filepath('PSC',dataset=chl_dataset,map_subset=map_subset,getfilepath=True,make_dir=True)
    
    parsed_info = parse_dataset_info(output_dir)
    info = parsed_info[0] if isinstance(parsed_info, list) else parsed_info

    psc_map = {}
    for date in chl_dates:
        filename = f"D_{date}-{map_subset or 'GLOBAL'}-{info['dataset']}-PSC-Turner.nc"
        psc_map[date] = os.path.join(output_dir, filename)

    # Normalize get_date_prod
    prod_key = (get_date_prod or "CHL").upper()

    # Get date list
    target_dates = get_dates(dates)
    if target_dates is None:
        if prod_key == "CHL":
            target_dates = sorted(chl_map.keys())
        elif prod_key == "SST":
            target_dates = sorted(sst_map.keys())
        else:
            raise ValueError(f"Invalid get_date_prod value: '{get_date_prod}'")

     # Build mapping for dates present in CHL and all other inputs
    psc_data_map = {}
    up_to_date_count = 0  # Initialize the tracker
    for date in target_dates:
        if date in chl_map and date in sst_map:
            chl_path = chl_map[date]
            sst_path = sst_map[date]
            psc_path = psc_map[date] #os.path.join(output_dir, f"D_{date}-{info['dataset']}-{info['version']}-{info['dataset_map']}-PSC-Turner.nc")

            # Check if PSC file exists and is newer than all inputs
            if os.path.exists(psc_path):
                psc_mtime = os.path.getmtime(psc_path)
                input_mtimes = [os.path.getmtime(p) for p in [chl_path, sst_path]]
                is_up_to_date = all(psc_mtime > m for m in input_mtimes)
            else:
                is_up_to_date = False

            if is_up_to_date:
                up_to_date_count += 1  # Increment for existing files
                if verbose: print(f"✅ PSC is current for {date}")
            else:
                if verbose: print(f"⏳ Need to process PSC for {date}")

            psc_data_map[date] = (chl_path, sst_path, psc_path, is_up_to_date)
        else:
            if verbose: print(f"⚠ Missing data for {date}: CHL {'✅' if date in chl_map else '❌'}  SST {'✅' if date in sst_map else '❌'} ")

    # --- FINAL MAP SUMMARY ---
    needs_update_count = len(psc_data_map) - up_to_date_count
    print("-" * 60)
    print(f"📊 PHYTOPLANKTON SIZE CLASS SUMMARY: PSC | DAILY | {map_subset or 'GLOBAL'}")
    print(f"✅ Up-to-date:  {up_to_date_count}/{len(psc_data_map)}")
    print(f"⏳ To process:  {needs_update_count} files")
    print("-" * 60)
    return psc_data_map

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def psc_turner(chl,sst):
    """
    Calculate the phytoplankton size class fractions of total chlorophyll according to Turner et al. (2021).

    INPUTS:
        chl         : float or np.ndarray — Satellite chlorophyll (mg m⁻³)
        sst         : float or np.ndarray — Sea surface temperature (°C)

    RETURNS:
        fmicro      : The fraction of total chlorophyll associated with microplankton
        fnano       : The fraction of total chlorophyll associated with nanoplankton
        fpico       : The fraction of total chlorophyll associated with picoplankton
        total_chl   : Total chlorophyll (mg m⁻³) (the same values as the input chlorophyll)
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

    # Ensure arrays and mask invalid inputs
    chl = np.where((chl > 0) & ~np.isnan(chl), chl, np.nan)
    
    # ===> Get the SST look-up file (can be updated with new LUT versions)
    lutdir = env["lookuptable_path"]
    lutpath = os.path.join(lutdir,'TURNER_PSIZE_SST_LUT_VER1.csv')
       
    sstlut = pandas.read_csv(lutpath,index_col='SST')
    sstlut = sstlut.to_xarray()
    
    # ===> Find the coefficients from the LUT based on the input SST
    sst_coeffs = sstlut.sel({"SST": sst}, method="nearest")

    # ===> Calculate the phytoplankton size class fractions
    fpico = (sst_coeffs.COEFF3 * (1 - np.exp(-1 * (sst_coeffs.COEFF4 / sst_coeffs.COEFF3) * chl))) / chl
    fnanopico = (sst_coeffs.COEFF1 * (1 - np.exp(-1 * (sst_coeffs.COEFF2 / sst_coeffs.COEFF1) * chl))) / chl
    fnano = fnanopico - fpico
    fmicro = (chl - (sst_coeffs.COEFF1 * (1 - np.exp(-1 * (sst_coeffs.COEFF2 / sst_coeffs.COEFF1) * chl)))) / chl

    # ===> Build the PSC dataset
    total_chl = chl
    
    return (
        wrap_output(fmicro, chl),
        wrap_output(fnano, chl),
        wrap_output(fpico, chl),
        wrap_output(total_chl, chl),
    )

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def process_daily_psc(chl, sst, output_psc_file, history=None, sourceinfo=None):
    """
    Process daily CHL and SST inputs:
    - Checks if data are valid and creates a valid data mask
    - Computes phytoplankton size class using the Turner model
    - Adds global and product metadata
    - Saves processed output as a netcdf file
    """
    # 1️⃣ Check the chl and sst
    validate_inputs(chl, sst)

    # 2️⃣  Create a mask of missing data
    def generate_valid_mask(*arrays):
        mask = arrays[0].notnull()
        for arr in arrays[1:]:
            mask = mask & arr.notnull()  # ✅ safe: creates new object each time
        return mask
    valid_mask = generate_valid_mask(chl, sst)
    assert valid_mask.shape == chl.shape, f"valid_mask shape mismatch: expected {chl.shape}, got {valid_mask.shape}"
    
    if valid_mask.shape != chl.shape:
        raise ValueError("valid_mask shape mismatch: expected {}, got {}".format(chl.shape, valid_mask.shape))

    chl_valid = chl.where(valid_mask)
    sst_valid = sst.where(valid_mask)

    # 3️⃣ Calculate PSC on non-missing data and create dataset    
    fmicro, fnano, fpico, total_chl = psc_turner(chl_valid, sst_valid)
        
    # 4️⃣ Create output dataset
    def ensure_shape(var, template):
        if isinstance(var, xr.DataArray):
            var = var.data
        if var.shape != template.shape:
            return np.broadcast_to(var, template.shape)
        return var
    
    ds_out = xr.Dataset({
        "PSC_fmicro": (chl.dims, ensure_shape(fmicro, chl)),
        "PSC_fnano": (chl.dims, ensure_shape(fnano, chl)),
        "PSC_fpico": (chl.dims, ensure_shape(fpico, chl)),
        "CHL": (chl.dims, ensure_shape(total_chl, chl))
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
    attrs["product_name"] = build_product_attributes("PSC")["long_name"]
    attrs = attrs | get_summary_metadata("PSC")
    attrs["references"] = get_reference_metadata('PSC_TURNER',refs_only=True)

    # Source Metadata
    if history: attrs["history"] = history
    chl_source = get_source_metadata(sourceinfo["CHL"]["dataset"],dataset_version=sourceinfo["CHL"]["version"],source_prefix="source_chl")
    sst_source = get_source_metadata(sourceinfo["SST"]["dataset"],dataset_version=sourceinfo["SST"]["version"],source_prefix="source_sst")
    attrs = attrs | chl_source | sst_source
    
    # Add metadta to the output dataset
    ds_out.attrs = attrs

    # 7️⃣ Save results
    ds_out.to_netcdf(output_psc_file,encoding=encoding,format='NETCDF4', engine='h5netcdf')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def process_single_day(date, chl_file, sst_file, output_file, sst_ds, chl_nc_var, sst_nc_var, map_subset, merged_info, verbose, debug):
    """Standalone helper function to process a single day of PSC for parallel execution."""
    try:
        chl_ds = xr.open_dataset(chl_file)
        chl = chl_ds[chl_nc_var]
        
        target_time = pd.to_datetime(date)
        
        # Force the slice into memory immediately to detach from massive array
        sst = sst_ds.sel(time=target_time, method='nearest', tolerance=pd.Timedelta('1D'), drop=False)[sst_nc_var].load()         
              
        chl_date = pd.to_datetime(chl["time"].values[0]).date()
        sst_date = pd.to_datetime(sst["time"].values).date()

        if chl_date == sst_date:
            sst = sst.expand_dims(time=chl["time"])
            sst["time"] = chl["time"]
        else:
            raise ValueError(f"Date mismatch: chl={chl_date}, sst={sst_date}")

        # Subset CHL and SST if requested
        if map_subset:
            chl = subset_dataset(chl, map_subset)
            sst = subset_dataset(sst, map_subset)
        
        chl, sst = xr.align(chl, sst, join="exact")

        chl_source = get_source_metadata(merged_info["CHL"]["dataset"],dataset_version=merged_info["CHL"]["version"],source_prefix="source_chl")
        sst_source = get_source_metadata(merged_info["SST"]["dataset"],dataset_version=merged_info["SST"]["version"],source_prefix="source_sst")

        history = " ".join([f"{build_product_attributes('PSC')['long_name']} is calculated using the Northeast U.S. regionally tuned phytoplankton size class model based on Turner et al. {get_reference_metadata('PSC_TURNER')[0]['year']}. ",
                f"The input chlorophyll file ({os.path.basename(chl_file)}) is from the {chl_source['source_chl_title']}. ",
                f"The input sea surface temperature file ({os.path.basename(sst_file)}) is from the {sst_source['source_sst_title']}. ",
                f"The SST data were regridded to the CHL grid using xESMF bilinear regridding {get_reference_metadata('xesmf')[0]['citation']}."
        ])
        
        start_time = time.perf_counter()
        process_daily_psc(chl, sst, output_file, history=history, sourceinfo=merged_info)
        elapsed = (time.perf_counter() - start_time) / 60
        
        return date, True, elapsed, None
        
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return date, False, 0, str(e)
        
    finally:
        # Force RAM garbage collection
        try:
            chl_ds.close()
        except: pass
        gc.collect()


def run_psc_pipeline(chl_dataset=None,
                    sst_dataset=None,
                    daterange=None,
                    map_subset='NES',
                    parallel_runs=1,
                    overwrite=False,
                    verbose=False,
                    debug=False,
                    logfile=None
                    ):
    """
    Runs the full phytoplankton size class pipeline:
    - Builds date→file map
    - Regrids SST in batch
    - Processes daily CHL/SST into PSC outputs

    Parameters:
        chl_dataset, sst_dataset (str): Dataset identifiers
        daterange (list of str): Dates to process (YYYYMMDD)
        map_subset (str): The subset region (e.g. NES, NWA) to subset the data 
        overwrite (bool): If True, reprocess even if output is up-to-date
        verbose (bool): If True, print progress
        logfile (str): Optional file to log the progress
    
    Example:
        run_psc_pipeine() # Uses the default datasets and will process the full daterange
        run_psc_pipeline(sst_dataset="CORALSST",daterange=['19980101','19980131']) # specifies a dataset and a daterange

    """
    map_subset = map_subset or "NES"  # Default to NES if None provided

    original_stdout = sys.stdout 

    if logfile in ["auto", True]:
        logfile = generate_logfile_path("phytosize", prods=['PSC'], dataset=chl_dataset, env_dict=env)    
        
    should_log = verbose or bool(logfile)

    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        sys.stdout = DualLogger(logfile)
        print("\n" + "="*80)
        print(f"🟢 PHYTOPLANKTON SIZE CLASS PIPELINE RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 Logging to: {logfile}")
        print("="*80)

    try:
        # 1️⃣ Build date→file map
        if should_log: print(f"\n🚀 Starting Phytoplankton Size Class Pipeline...")
        psc_data_map = build_psc_date_map(dates=daterange,
                                          chl_dataset=chl_dataset,
                                          sst_dataset=sst_dataset,
                                          map_subset=map_subset,
                                          verbose=verbose)

        # 2️⃣ Filter dates to run and Track Progress
        dates_to_run = []
        skipped_tasks = []
        created_tasks = []
        failed_tasks = []
        
        for date, info in psc_data_map.items():
            chl_path, sst_path, psc_path, up_to_date = info
            file_name = os.path.basename(psc_path)
            
            if up_to_date and not overwrite:
                skipped_tasks.append(date)
                if should_log: print(f"  ⏭  Skipped: {file_name} (Up-to-date)")
            else:
                dates_to_run.append(date)
                if should_log: print(f"  🆕 Queued for Creation: {file_name}")
        
        total_dates = len(dates_to_run)

        if not dates_to_run:
            if should_log: print("✅ All outputs are up-to-date. Nothing to run.")
            
            print("\n" + "="*60)
            print("🏁 PHYTOPLANKTON SIZE CLASS PIPELINE FINISHED SUMMARY")
            print("="*60)
            print(f"📥 Input dates found:       {len(psc_data_map)}")
            print(f"⏭  Existing files skipped:  {len(skipped_tasks)}")
            print(f"🆕 New files created:       0")
            print("="*60)
            return

        if should_log: print(f"\n🚀 Processing {total_dates} dates...")

        # 3️⃣ Collect input files
        chl_files = [psc_data_map[d][0] for d in dates_to_run]
        sst_files = [psc_data_map[d][1] for d in dates_to_run]
        output_files = [psc_data_map[d][2] for d in dates_to_run]

        # 4️⃣ Get input data information
        chl_parse = parse_dataset_info(chl_files[0])
        sst_parse = parse_dataset_info(sst_files[0])

        chl_info = chl_parse[0] if isinstance(chl_parse, list) else chl_parse
        sst_info = sst_parse[0] if isinstance(sst_parse, list) else sst_parse

        merged_info = {
            chl_info["product"]: chl_info,
            sst_info["product"]: sst_info,
        }

        chl_source = get_source_metadata(chl_info["dataset"],dataset_version=chl_info["version"],source_prefix="source_chl")
        sst_source = get_source_metadata(sst_info["dataset"],dataset_version=sst_info["version"],source_prefix="source_sst")

        chl_nc_var = get_nc_prod(chl_info['dataset'],'CHL')
        sst_nc_var = get_nc_prod(sst_info['dataset'],'SST')

        # 5️⃣ Batch regrid SST
        if should_log: print(f"Regridding SST files to CHL grid...")
        sst_ds = regrid_wrapper(chl_files, sst_files, source_vars=[sst_nc_var], daterange=dates_to_run)
        
        # 6️⃣ Execute Daily Loop (Sequential or Parallel)
        import concurrent.futures
        if should_log: print(f"Writing PSC files to {os.path.dirname(output_files[0])}")

        if parallel_runs == 1:
            for i, (date, chl_file, sst_file, output_file) in enumerate(zip(dates_to_run, chl_files, sst_files, output_files), start=1):
                date, success, elapsed, error = process_single_day(date, chl_file, sst_file, output_file, sst_ds, chl_nc_var, sst_nc_var, map_subset, merged_info, verbose, debug)
                
                if success:
                    created_tasks.append(date)
                    if should_log: print(f"  ✅ Created: {os.path.basename(output_file)}, file {i} of {total_dates} ({round((i / total_dates) * 100)}%) after {elapsed:.2f} minutes.")
                else:
                    failed_tasks.append(f"{date}: {error}")
                    if should_log: print(f"  ❌ Error creating {os.path.basename(output_file)}: {error}")
        else:
            if should_log: print(f"\n  ⚡ Launching {total_dates} daily tasks across {parallel_runs} parallel workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_runs) as executor:
                future_to_date = {
                    executor.submit(process_single_day, date, chl_file, sst_file, output_file, sst_ds, chl_nc_var, sst_nc_var, map_subset, merged_info, verbose, debug): date
                    for date, chl_file, sst_file, output_file in zip(dates_to_run, chl_files, sst_files, output_files)
                }
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_date):
                    completed_count += 1
                    date, success, elapsed, error = future.result()
                    out_filename = os.path.basename(psc_data_map[date][2])
                    
                    if success:
                        created_tasks.append(date)
                        if should_log: print(f"  ✅ [Parallel] Created: {out_filename}, file {completed_count} of {total_dates} ({round((completed_count / total_dates) * 100)}%) after {elapsed:.2f} minutes.")
                    else:
                        failed_tasks.append(f"{date}: {error}")
                        if should_log: print(f"  ❌ [Parallel] Error creating {out_filename}: {error}")

        # --- FINAL SUMMARY REPORT ---
        print("\n" + "="*60)
        print("🏁 PHYTOPLANKTON SIZE CLASS PIPELINE FINISHED SUMMARY")
        print("="*60)
        print(f"📥 Input dates found:       {len(psc_data_map)}")
        print(f"⏭  Existing files skipped:  {len(skipped_tasks)}")
        print(f"🆕 New files created:       {len(created_tasks)}")
        print(f"❌ Errors/Not created:      {len(failed_tasks)}")
        
        if failed_tasks:
            print(f"\n📂 FAILED TASKS DETAILS:")
            for task in failed_tasks:
                print(f"  • {task}")
        print("="*60)

    finally:
        if logfile and logfile not in ["auto", True]:
            print(f"\n🔴 PIPELINE RUN ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
            sys.stdout.close()
            sys.stdout = original_stdout
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
