import os
import glob
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)
from utilities import dataset_defaults, get_dataset_products, parse_dataset_info, resolve_dataset_map, get_period_info

"""
Purpose:
    PRODUCT_UTILITIES is a collection of utility functions for handling "product" specific tasks.

Main Functions:
    - product_defaults: Returns the default/primary dataset and datatype location for each product
    - netcdf_product_defaults: Returns the data product name found in the original source netcdf files
    - get_nc_prod: Returns internal variable name and metadata for a given dataset and product.
    - get_prod_files: Get the files for the specified product.
    - make_product_output_dir: Create the output path by replacing dataset_type and product, and create the directory if it does not exist.

Helper Functions:
    - validate_inputs: Validates that input data arrays are xarray.DataArray and have matching shapes
    
Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on August 01, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Sep 18, 2025 - KJWH: Moved product specific functions from file_utilities to dataset_utitlities
                         Overhauled get_prod_files to by more dynamic and find STATS files based on the period code

"""
def product_defaults():
    """
    Returns the default/primary dataset and datatype location for each product

    Parameters:
        No inputs
    
    Returns:
        Dictionary of default dataset specific data types and products
    """

    # The default product name, dataset and source data location product
    prod_info_map = {
        'CHL': ('CHL','OCCCI', 'SOURCE'),
        'CHLOR_A': ('CHLOR_A','OCCCI','PRODUCT'),
        'SST': ('SST','ACSPO', 'SOURCE'),
        'PPD': ('PPD','OCCCI', 'PRODUCT'),
        'PSC': ('PSC','OCCCI', 'PRODUCT'),
        'RRS': ('RRS','OCCCI', 'SOURCE'),
        'PAR': ('PAR','GLOBCOLOUR','SOURCE'),
        'IPAR': ('IPAR','PACE','SOURCE'),
        'AVW': ('AVW','OCCCI','PRODUCT'),
        'KD': ('KD','OCCCI','SOURCE'),
        'IOP': ('IOP','OCCCI','SOURCE'),
        'MOANA': ('MOANA','PACE','SOURCE'),
        'CARBON': ('CARBON','PACE','SOURCE'),
        'FLH': ('FLH','PACE','SOURCE'),
        'CHL_TEMP': ('CHL','GLOBCOLOUR','SOURCE'),
        'SST_TEMP': ('SST','ACSPONRT', 'SOURCE'),
        'CHL_FRONTS': ('CHL_FRONTS','OCCCI', 'PRODUCT'),
        'SST_FRONTS': ('SST_FRONTS','ACSPO', 'PRODUCT'),
        'FRONTS': ('SST_FRONTS','ACSPO', 'PRODUCT'),
        'BTEMP': ('BTEMP','GLORYS','SOURCE')
    }

    return prod_info_map

def netcdf_product_defaults():
    """
    Returns the data product name found in the original source netcdf files

    Parameters:
        No inputs
    
    Returns:
        Dictionary of default dataset specific data types and products
    """

    # The default product name, dataset and source data location product
    return {
        'ACSPO': {
            'SST': 'sea_surface_temperature',
            'SST_GRADMAG': 'sst_gradient_magnitude',
            'SST_GRADDIR': 'sst_gradient_direction',
            'SST_BIAS': 'sses_bias'
        },
        'ACSPO_NRT': {
            'SST': 'sea_surface_temperature',
            'SST_GRADMAG': 'sst_gradient_magnitude',
            'SST_GRADDIR': 'sst_gradient_direction',
            'SST_BIAS': 'sses_bias'
        },
        'CORALSST': {
            'SST': 'analysed_sst',
        },
        'OISST': {
            'SST': 'SST',
        },
        'MUR': {
            'SST': 'analysed_sst',
        },
        'AVHRR': {
            'SST': 'sea_surface_temperature',
        },
        'GLOBCOLOUR': {
            'PAR': 'PAR_mean',
            'CHL': 'CHL1_mean',
        },
        'OCCCI': {
            'CHL': 'chlor_a',
        },
        'PACE': {
            'CHL': 'chlor_a',
            'PAR': 'par_day_planar_above',
            'IPAR': 'ipar_plana_above',
            'RRS': 'Rrs',
            'AVW': 'avw',
            'MOANA_PRO': 'prococcus_moana',
            'MOANA_SYN': 'synococcus_moana',
            'MOANA_PICO': 'picoeuk_moana',
            'PYTHO_CARBON': 'carbon_phyto',
            'FLH': 'nflh'
        },
        'GLORYS': {
            'BTEMP': 'bottomT',
            'BSAL': 'bottomS',
        },
        'TESTDATASET': {
            'SST_MEAN': 'sst_mean',
            'SST_MAX': 'sst_max',
        }
    }

#--------------------------------------------------------------------------------------
def get_nc_prod(dataset,product):
    """
    Returns internal variable name and metadata for a given dataset and product.

    Parameters:
        dataset_name (str): Dataset key (e.g., 'GLOBCOLOUR')
        product_name (str): Product key (e.g., 'CHL1')

    Returns:
        dict with keys like 'var_name', 'source', 'frequency', or None if not found
    """
    dataset_name = dataset.upper()
    product_name = product.upper()

    dataset_map = netcdf_product_defaults()

    # Check if dataset exists
    if dataset_name not in dataset_map:
        print(f"[ERROR] Dataset '{dataset_name}' not found.")
        return None
    
    product_map = dataset_map[dataset_name]

    # Exact match
    if product_name in product_map:
        return product_map[product_name]

    fuzzy_matches = []

    # Fuzzy match: product_name vs product keys
    for key in product_map:
        if product_name in key or key in product_name:
            fuzzy_matches.append((key, product_map[key]))

    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0][1]
    elif len(fuzzy_matches) > 1:
        print(f"[ERROR] Ambiguous product name '{product_name}'. Multiple matches found:")
        for key, val in fuzzy_matches:
            print(f"  - Product key: {key}, internal name: {val}")
        return None
    else:
        print(f"[ERROR] No match found for product '{product_name}' in dataset '{dataset_name}'.")
        return None


def get_prod_files(prod, dataset=None, period=None, getfilepath=False, make_dir=False, verbose=False, **kwargs):
    """
    Retrieves NetCDF files for a specified product from a structured dataset directory.

    This function resolves the correct path based on product metadata, dataset defaults,
    and optional period codes (e.g. 'M' for monthly stats). It supports fallback logic
    when dataset_map is not provided, and redirects to *_STATS folders for derived products.

    Parameters:
    ----------
    prod: str
        Product name (e.g. 'CHL', 'PSC', 'SST').
    dataset: str, optional
        Dataset name (e.g. 'OCCCI', 'ACSPO'). Defaults to product's default dataset.
    dataset_version: str, optional
        Version string (e.g. 'V6.0'). Defaults to dataset default.
    dataset_type: str, optional
        Type of dataset folder (e.g. 'SOURCE' or 'PRODUCTS'). Defaults to product default in product_defaults().
    dataset_map: str, optional
        Specific map folder (e.g. 'NES_4KM_DAILY'). If not provided, will be auto-resolved.
    map_region: str, optional
        Map region (e.g. 'NES', 'GLOBAL'). Used for fallback resolution.
    resolution: str, optional
        Resolution in km (e.g. '4', '25'). Used for fallback resolution.
    type: str, optional
        Data type (e.g. 'DAILY', 'STATS', 'ANOMS'). Used for fallback resolution.
    prod_type: str, optional
        Subfolder under product (e.g. 'ANOMALY', 'CLIMATOLOGY').
    period: str, optional
        Period code (e.g. 'M', 'D3', 'YEAR'). Used to redirect to *_STATS folders.
    getfilepath: bool, optional
        If True, returns the resolved path instead of listing files.
    verbose: bool, optional
        If True, prints detailed resolution steps.

    Returns:
    -------
    list or str
        List of matching NetCDF files, or path string if getfilepath=True.

    Raises:
    -------
    ValueError
        If product is found in multiple folders or resolution fails.
    """
    
    def verbose_trace(msg, verbose=True):
        if verbose:
            print(msg)

    # --- Step 1: Normalize product name and resolved product metadata ---
    prod = prod.upper().strip()
    prod_info_map = product_defaults()
    if prod not in prod_info_map:
        print(f"❌ Product '{prod}' not found in prod_info_map.")
        return None
    actual_prod, default_dataset, default_type = prod_info_map[prod]

    # --- Step 2: Extract parameters from kwards ---
    dataset = dataset.upper().strip() if dataset else default_dataset # Use provided dataset or fallback to default
    dataset_type = kwargs.get('dataset_type', default_type).upper() # User provided dataset_type or fallback to default from prod_info_map
    dataset_version = kwargs.get('dataset_version') 
    dataset_type = kwargs.get('dataset_type', default_type).upper()
    dataset_map = kwargs.get('dataset_map')
    resolution = kwargs.get('resolution')
    map_region = kwargs.get('map_region')
    data_type = kwargs.get('data_type')
    prod_type = kwargs.get('prod_type')


    # --- Step 3: Initialize a provenance dictionary to track decisions for debugging ---
    provenance_log = {
        'product': prod,
        'dataset': dataset,
        'dataset_type': dataset_type,
        'dataset_version': dataset_version,
        'period': period,
        'map_region':map_region,
        'resolution_steps': []
    }
    
    # --- Step 4: Resolve dataset defaults ---
    dataset_info_map = dataset_defaults()
    if dataset not in dataset_info_map:
        print(f"❌ Dataset '{dataset}' not found in dataset_info_map.")
        return None

    default_version, default_map, default_product = dataset_info_map[dataset]
    dataset_version = dataset_version or default_version
    provenance_log['resolution_steps'].append(f"Loaded dataset defaults → version: {default_version}, map: {default_map}, default_product: {default_product}")
    
    # --- Step 5: Load dataset structure ---
    dataset_products = get_dataset_products(dataset)
    if not dataset_products:
        print(f"⚠ No product structure found for dataset '{dataset}'.")
        return None
    
    if actual_prod == default_product:
        provenance_log['resolution_steps'].append(f"Using default dataset_map: {default_map}")
    else:
        provenance_log['resolution_steps'].append(f"Searching all maps for product '{actual_prod}'")

    if dataset_type in dataset_products:
        filtered_structure = {dataset_type: dataset_products[dataset_type]}
    else:
        filtered_structure = dataset_products
    

    # --- Step 6: Resolve dataset_map
    if not kwargs.get('dataset_map'):
        dataset_map, path = resolve_dataset_map(
        filtered_structure,
        prod=actual_prod,
        default_map=default_map,
        period=period,
        data_type=data_type,
        provenance_log=provenance_log,
        verbose=verbose
    )
    
    # --- Step 7: Handle PRODUCTS/STATS/CLIMS/ANOMS Logic ---
    transformed_path = False
    # 🎯 TRIGGER LOGIC: Transform if writing outputs or looking for derived periods
    is_derived_period = period and period not in ['D', 'DD']
    if getfilepath or is_derived_period or data_type == 'ANOMS':    
        try:
            # Get period_info dictionary/tuple
            p_info = get_period_info(period) if period else {}
            folder_name = p_info.get('folder_name')
            # Determine Suffix dynamically based entirely on period_info
            if folder_name:
                suffix = folder_name.upper() # e.g., 'MONTHLY', 'CLIMATOLOGY', 'WEEKLY'
            else:
                # Fail-safe: Use data_type if provided, otherwise default to 'DERIVED'
                suffix = data_type.upper() if data_type else 'DERIVED'
            
            # Manually override for anomalies if requested
            if data_type == 'ANOMS' or (period and 'ANOM' in period):
                suffix = 'ANOMS'

            # --- Reconstruction Logic ---
            parts = dataset_map.split('_') if dataset_map else ['GLOBAL', '4KM']
            
            # Use map_region/resolution overrides if provided, else keep original
            region = map_region.upper() if map_region else parts[0]
            
            if resolution:
                res_val = str(resolution).upper()
                resolution_str = res_val if 'KM' in res_val else f"{res_val}KM"
            else:
                resolution_str = parts[1] if len(parts) > 1 else '4KM'
            
            # 🎯 NEW MAP: e.g., NES_2KM_MONTHLY or NES_2KM_CLIMATOLOGY
            new_map = f"{region}_{resolution_str}_{suffix}"

            # --- Transform the physical path ---
            res_info = parse_dataset_info(path)
            if res_info:
                info = res_info[0]
                # Replace the old map string with the new elegant one
                path = path.replace(info["dataset_map"], new_map)
                
                # Ensure we redirect to /PRODUCTS/ for any derived data 
                # OR if we are subsetting a region (e.g. GLOBAL -> NES)
                if suffix != 'DAILY' or region != parts[0]:
                    path = path.replace(f'/{info["dataset_type"]}/', "/PRODUCTS/")
            
            dataset_map = new_map 
            transformed_path = True
            verbose_trace(f"📂 Transformed path to {suffix}: {path}", verbose)
        except Exception as e:
            verbose_trace(f"⚠ Path transformation failed: {e}", verbose)
    """
    is_derived_type = data_type in ['STATS', 'ANOMS', 'CLIMS']
    
    #if getfilepath and (kwargs.get('map_region') or kwargs.get('resolution')):
    if getfilepath or is_derived_period or is_derived_type:    
        try:
            # Get suffix from period_info
            p_info = get_period_info(period) if period else None
            
            # 🎯 Safely check for climatology (handles both dicts and tuples)
            is_clim = False
            if isinstance(p_info, dict):
                is_clim = p_info.get('is_climatology', False)
            elif isinstance(p_info, tuple) and len(p_info) > 4:
                is_clim = p_info[4]
            
            # Determine Suffix (Daily, Stats, or Climatology)
            if is_clim: 
                suffix = 'CLIMS'
            elif period in ['D', 'DD']:
                suffix = 'DAILY'
            elif is_derived_type:
                suffix = data_type
            else:
                suffix = 'STATS'

            # Manually override for anomalies if requested
            if data_type == 'ANOMS' or (period and 'ANOM' in period):
                suffix = 'ANOMS'

            # --- Reconstruction Logic ---
            parts = dataset_map.split('_') if dataset_map else ['GLOBAL', '4KM']
            
            # Use map_region/resolution overrides if provided, else keep original
            region = map_region.upper() if map_region else parts[0]
            
            if resolution:
                res_val = str(resolution).upper()
                resolution_str = res_val if 'KM' in res_val else f"{res_val}KM"
            else:
                resolution_str = parts[1] if len(parts) > 1 else '4KM'
            
            new_map = f"{region}_{resolution_str}_{suffix}"

            # --- Transform the physical path ---
            res_info = parse_dataset_info(path)
            if res_info:
                info = res_info[0]
                # Replace the map (e.g., GLOBAL_4KM_DAILY -> NES_2KM_STATS)
                path = path.replace(info["dataset_map"], new_map)
                
                # Ensure we redirect to /PRODUCTS/ for any derived data 
                # OR if we are subsetting a region (e.g. GLOBAL -> NES)
                if suffix != 'DAILY' or region != parts[0]:
                    path = path.replace(f'/{info["dataset_type"]}/', "/PRODUCTS/")
            
            dataset_map = new_map 
            transformed_path = True
            verbose_trace(f"📂 Transformed path to {suffix}: {path}", verbose)
        except Exception as e:
            verbose_trace(f"⚠ Path transformation failed: {e}", verbose)
    """ 
    
    if not dataset_map or not path:
        print(f"⚠ No valid dataset_map found for product '{prod}' with data_type='{data_type}'")
        return []


    # --- Step 8: Search for product path ---
    candidate_types = [dataset_type] if dataset_type in dataset_products else list(dataset_products.keys())    

    if not transformed_path:
        matching_paths = {}
        for dtype in candidate_types:
            for map_key, prod_dict in dataset_products[dtype].items():
                verbose_trace(f"🔍 Checking {dtype}/{map_key} for product '{actual_prod}'", verbose)
                if dataset_map and dataset_map not in map_key:
                    continue
                if actual_prod in prod_dict:
                    path = prod_dict[actual_prod]
                    if prod_type:
                        path = os.path.join(path, prod_type)
                    matching_paths[dtype] = path
        # --- Step 8.5: Handle ambiguous or missing paths ---
        if not matching_paths:
            verbose_trace(f"⚠ Product '{prod}' not found in dataset '{dataset}' under type '{dataset_type}' and map '{dataset_map}'")
            return []
        
        if len(matching_paths) > 1:
            raise ValueError(f"❌ Ambiguous product location: '{prod}' found in multiple types ({list(matching_paths.keys())}) for dataset '{dataset}'.")

        resolved_type, path = next(iter(matching_paths.items()))
        verbose_trace(f"✅ Found '{prod}' in '{resolved_type}' → {path}",verbose)
    
    provenance_log['resolution_steps'].append(f"Resolved product path → {path}")
    provenance_log['final_path'] = path

    # --- Step 9a: Make and return path or search for files ---
    dir_exists = os.path.isdir(path)
    if make_dir and not dir_exists:
        os.makedirs(path, exist_ok=True)
        dir_exists = True
        verbose_trace(f"🛠 Created directory: {path}", verbose)
    
    if getfilepath:
        # New Safety Check: If we are returning the path but it doesn't exist
        if not dir_exists:
            print(f"⚠ Warning: Resolved path does not exist on disk: {path}")
            print("  (Hint: Set make_dir=True to automatically create this directory structure)")
        
        return path

    # --- Step 9b: Or search for files ---
    search_pattern = f"{period}_*.nc" if period else "*.nc"
    verbose_trace(f"🔍 Searching for .nc files in: {path}",verbose)

    if verbose:
            print("🧾 Provenance log:")
            for step in provenance_log['resolution_steps']:
                print("  •", step)

    nc_files = glob.glob(os.path.join(path, search_pattern))
    if not nc_files:
        if period:
            print(f"⚠ No .nc files found in: {path} for period code {period}")
        else:
            print(f"⚠ No .nc files found in: {path}")
        return []

    verbose_trace(f"📦 Found {len(nc_files)} .nc files in: {path}")
    nc_files.sort()

    return nc_files

                                                           
