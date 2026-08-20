import os
import glob
import xarray as xr
import pandas as pd
import traceback
import gc
from datetime import datetime

from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)
from utilities import (
    subset_dataset, 
    get_dataset_products, parse_dataset_info, get_prod_files, get_source_file_dates,
    get_default_metadata, 
    get_lut_metadata, 
    get_geospatial_metadata, 
    get_temporal_metadata, 
    build_product_attributes, 
    get_summary_metadata, 
    get_reference_metadata, 
    get_source_metadata, 
    get_current_utc_timestamp
)

"""
PROCESS_GLOBCOLOUR prepares raw GlobColour daily netCDF files for downstream analysis.
The primary objectives are to append a missing time dimension, subset the global spatial grid to a 
predefined regional map (e.g., Northwest Atlantic - NWA), standardize file nomenclature, and inject 
comprehensive CF-compliant metadata.

Main Functions:
    - run_globcolour_dataset: Orchestrates the preprocessing pipeline across all products within the dataset.

Helper Functions:
    - build_globcolour_map: Scans input directories and determines which files require processing based on modification times.
    - preprocess_globcolour: Performs the core xarray operations (time dimension addition, subsetting, metadata injection) and saves the output.

References:
    GlobColour Product Documentation (http://www.globcolour.info)
    
Copywrite: 
    Copyright (C) 2026, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Aug 19, 2026 - KJWH: Initial code written for GlobColour preprocessing and subsetting workflow.
"""

def build_globcolour_map(input_files, output_dir, overwrite=False, verbose=False):
    """
    Compares a list of input files against the target output directory to determine 
    which files are missing or out-of-date. It builds a processing task map and extracts
    dataset metadata needed for standardized file naming.
    
    Parameters
    ----------
    input_files : list
        List of absolute file paths to the raw source NetCDF files.
    output_dir : str
        Path to the directory where processed files should be saved.
    overwrite : bool, optional
        If True, forces all files to be marked for processing regardless of modification time.
    verbose : bool, optional
        If True, prints progress and summary statistics.
        
    Returns
    -------
    tuple
        - processing_map (dict): A dictionary mapping original filenames to their task parameters.
        - dataset_info (dict): Parsed metadata components (dataset, version, grid, product) for the directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Parse dataset info dynamically from the output directory
    parsed_info = parse_dataset_info(output_dir)
    dataset_info = parsed_info[0] if isinstance(parsed_info, list) else parsed_info

    # Extract dates using your custom utility
    dates = get_source_file_dates(input_files, format="yyyymmdd", placeholder=None)

    processing_map = {}
    up_to_date_count = 0

    # Zip files and dates together to build the map
    for in_path, date_str in zip(input_files, dates):
        if not date_str:
            if verbose: print(f"  ⚠️ Skipping {os.path.basename(in_path)}: Could not parse date.")
            continue
            
        filename = os.path.basename(in_path)
            
        # Construct standard output filename using parsed info
        out_name = f"D_{date_str}-{dataset_info['dataset']}-{dataset_info['version']}-{dataset_info['dataset_grid']}-{dataset_info['product']}.nc"
        out_path = os.path.join(output_dir, out_name)
        
        is_up_to_date = False
        
        # Freshness Check
        if os.path.exists(out_path) and not overwrite:
            if os.path.getmtime(out_path) > os.path.getmtime(in_path):
                is_up_to_date = True
                
        if is_up_to_date:
            up_to_date_count += 1
            
        processing_map[filename] = {
            "input": in_path,
            "output": out_path,
            "date_str": date_str,
            "is_up_to_date": is_up_to_date
        }
        
    if verbose:
        print(f"  📁 Output Dir:  {output_dir}")
        print(f"  ✅ Up-to-date:  {up_to_date_count}/{len(input_files)}")
        print(f"  ⏳ To process:  {len(input_files) - up_to_date_count} files")
        
    return processing_map, dataset_info


def preprocess_globcolour(task, subset_map, dataset_info, verbose=False, debug=False):
    """
    Processes a single GlobColour file by extracting the date to create a time dimension,
    spatially subsetting to the target region, and injecting CF-compliant global metadata.
    
    Parameters
    ----------
    task : dict
        A dictionary containing 'input', 'output', and 'date_str' for the file to process.
    dataset_info : dict
        Dictionary containing parsed dataset attributes (dataset, version, product, etc.).
    subset_map : str
        The regional string identifier (e.g., 'NWA') to subset the data.
    verbose : bool, optional
        If True, prints step-by-step progress.
    debug : bool, optional
        If True, prints detailed traceback information upon failure.
        
    Returns
    -------
    bool
        True if the file was processed and saved successfully, False otherwise.
    """
    in_path = task['input']
    out_path = task['output']
    date_str = task['date_str']
    ds = None
    
    try:
        if debug: print(f"    📖 Opening {os.path.basename(in_path)}...")
        ds = xr.open_dataset(in_path, engine='h5netcdf')

        # Save a copy of the attributes for every data variable in the file
        orig_var_attrs = {var: ds[var].attrs.copy() for var in ds.data_vars}
        
        # Add missing time dimension
        if 'time' not in ds.coords:
            parsed_date = pd.to_datetime(date_str, format='%Y%m%d')
            if debug: print(f"      🕒 Adding time dimension for {parsed_date.date()}...")
            ds = ds.expand_dims('time').assign_coords(time=[parsed_date])
            
        # Subset to predefined region
        if subset_map and subset_map != "GLOB":
            if debug: print(f"      🗺️ Subsetting to region: {subset_map}...")
            ds = subset_dataset(ds, subset_map)
            
        # 3. Add Global metadata
        if debug: print(f"      🏷️ Updating metadata attributes...")
        
        # Base Attributes
        attrs = get_default_metadata(sheet="General")
        attrs = attrs | get_lut_metadata(add_program="Ecosystem Dynamics and Assessment Branch")
      
        # Geospatial & Temporal Attributes
        attrs = attrs | get_geospatial_metadata(dataset=ds)
        attrs = attrs | get_temporal_metadata(ds=ds)

        # Copy specific global attributes
        keys_to_copy = [
            "product_level", "product_version", "product_type",
            "parameter_code", "spatial_resolution", "parameter",
            "parameter_algo_list", "publication", "sensor_name_list", "grid_type"
        ]
        for k in keys_to_copy:
            if k in ds.attrs:
                attrs[k] = ds.attrs[k]

        # Product Specific Attributes[cite: 3]
        try:
            attrs["product_name"] = build_product_attributes(dataset_info['product'])["long_name"]
            attrs = attrs | get_summary_metadata(dataset_info['product'])
            attrs["references"] = get_reference_metadata(dataset_info['product'], refs_only=True)
        except Exception as e:
            if debug: print(f"      ⚠️ Product metadata warning: {e}")

        # Modification History[cite: 3]
        new_history = f"{get_current_utc_timestamp()} Added time coordinate and subset to {subset_map} from global file {os.path.basename(in_path)}."
        attrs["history"] = f"{ds.attrs.get('history', '')}\n{new_history}".strip()
        
        # Source Metadata[cite: 3]
        try:
            source_meta = get_source_metadata(dataset_info["dataset"], dataset_version=dataset_info["version"], source_prefix=f"source",ds_attrs=ds.attrs)
            attrs = attrs | source_meta
        except Exception as e:
            if debug: print(f"      ⚠️ Source metadata warning: {e}")

        # Assign updated attributes
        ds.attrs = attrs

        # Restore original variable attributes
        for var, v_attrs in orig_var_attrs.items():
            if var in ds.data_vars:
                ds[var].attrs = v_attrs

        # 4. Save output
        if verbose: print(f"      💾 Saving {os.path.basename(out_path)}...")
        ds.to_netcdf(out_path, format='NETCDF4')
        
        return True

    except Exception as e:
        print(f"❌ Error processing {in_path}: {e}")
        if debug: traceback.print_exc()
        return False
        
    finally:
        if ds is not None: ds.close()
        gc.collect()




def run_globcolour_dataset(subset_map="NWA", overwrite=False, verbose=True, debug=False, dry_run=False):
    """
    High-level orchestrator that discovers all GlobColour products, fetches their global 
    daily files, evaluates processing needs, and executes the preprocessing loop for each product.
    
    Parameters
    ----------
    subset_map : str, optional
        The regional string identifier (default is 'NWA') to subset the dataset.
    overwrite : bool, optional
        If True, forces the regeneration of all files.
    verbose : bool, optional
        If True, prints detailed pipeline progress.
    debug : bool, optional
        If True, prints tracebacks for any failed processing tasks.
    dry_run : bool, optional
        If True, simulates the processing step, printing the intended input and output paths
        without reading the data or writing to disk.
    """

    if verbose: 
        mode = "[DRY RUN] " if dry_run else ""
        print(f"🚀 {mode} Initializing GLOBCOLOUR preprocessing pipeline...")
    
    # Fetch the dictionary of products and paths
    prod_dict = get_dataset_products('GLOBCOLOUR')
    
    # Parse the nested dictionary structure
    # e.g., {'SOURCE': {'GLOBAL_4KM_DAILY': {'CHL': '/path/...', 'PAR': '/path/...'}}}
    for source_key, source_level in prod_dict.items():
        # Only process raw SOURCE files
        if source_key != 'SOURCE':
            continue
        for res_key, res_level in source_level.items():
            # Only process the GLOBAL and DAILY base grids
            if 'GLOBAL' not in res_key or 'DAILY' not in res_key:
                continue
            for prod_name, input_dir in res_level.items():
                
                if verbose:
                    print("=" * 60)
                    print(f"🧪 Processing product: {prod_name} ({source_key} -> {res_key})")
                    print("=" * 60)

                # 1. Fetch input files using get_prod_files looking for GLOBAL files
                input_files = get_prod_files(prod_name, dataset='GLOBCOLOUR', map_subset='GLOBAL', verbose=debug)

                if not input_files:
                    if verbose: print(f"⚠️ No input files found for {prod_name}. Skipping.")
                    continue

                # 2. Define dynamic output directory
                output_dir = input_dir.replace("GLOBAL", subset_map)

                # 3. Build the task map
                proc_map, dataset_info = build_globcolour_map(input_files, output_dir, overwrite, verbose)
                
                # 4. Execute the preprocessing tasks
                successful, failed = 0, 0
                for filename, task in proc_map.items():
                    if task['is_up_to_date']:
                        continue

                    if dry_run:
                        if verbose:
                            print(f"    🌵 [DRY RUN] Would process: {filename}")
                            print(f"       -> {os.path.basename(task['output'])}")
                        successful += 1
                        continue
                        
                    success = preprocess_globcolour(
                        task=task, 
                        dataset_info=dataset_info, 
                        subset_map=subset_map, 
                        verbose=verbose, 
                        debug=debug
                    )
                    
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        
                if verbose: print(f"🏁 {prod_name} COMPLETE: {successful} processed, {failed} failed.")

# Example usage:
# run_globcolour_workflow("/path/to/raw/globcolour", "/path/to/processed/globcolour", region="NES")