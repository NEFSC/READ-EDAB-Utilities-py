import os
import glob
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)
from utilities import dataset_defaults, get_dataset_products, parse_dataset_info, resolve_dataset_map

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
        'FRONTS': ('SST_FRONTS','ACSPO', 'PRODUCT')
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


def get_prod_files(prod,
                   dataset=None,
                   dataset_version=None,
                   dataset_type=None,
                   dataset_map=None, 
                   map=None,
                   resolution=None,
                   data_type=None,
                   prod_type=None,
                   period=None,
                   getfilepath=False,
                   verbose=False):
    """
    Retrieves NetCDF files for a specified product from a structured dataset directory.

    This function resolves the correct path based on product metadata, dataset defaults,
    and optional period codes (e.g. 'M' for monthly stats). It supports fallback logic
    when dataset_map is not provided, and redirects to *_STATS folders for derived products.

    Parameters:
    ----------
    prod : str
        Product name (e.g. 'CHL', 'PSC', 'SST').
    dataset : str, optional
        Dataset name (e.g. 'OCCCI', 'ACSPO'). Defaults to product's default dataset.
    dataset_version : str, optional
        Version string (e.g. 'V6.0'). Defaults to dataset default.
    dataset_type : str, optional
        Type of dataset folder (e.g. 'SOURCE', 'EDAB_PRODUCTS'). Defaults to product default.
    dataset_map : str, optional
        Specific map folder (e.g. 'NES_4KM_DAILY'). If not provided, will be auto-resolved.
    map : str, optional
        Map region (e.g. 'NES', 'GLOBAL'). Used for fallback resolution.
    resolution : str, optional
        Resolution in km (e.g. '4', '25'). Used for fallback resolution.
    type : str, optional
        Data type (e.g. 'DAILY', 'STATS', 'ANOMS'). Used for fallback resolution.
    prod_type : str, optional
        Subfolder under product (e.g. 'ANOMALY', 'CLIMATOLOGY').
    period : str, optional
        Period code (e.g. 'M', 'D3', 'YEAR'). Used to redirect to *_STATS folders.
    getfilepath : bool, optional
        If True, returns the resolved path instead of listing files.
    verbose : bool, optional
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

    # --- Internal utilities ---
    def resolve_stats_path(input_path, output_product=None, new_dataset_type='EDAB_PRODUCTS', subset='GLOBAL'):
        """
        Transforms a source path into a stats output path by replacing dataset_type,
        product name, and map prefix.
        """
        info = parse_dataset_info(input_path)
        new_path = input_path
        new_path = new_path.replace(info["dataset_type"], new_dataset_type)
        if output_product:
            new_path = new_path.replace(info["product"], output_product)
        for old_prefix in ["MAPPED", "BINNED"]:
            if info["dataset_map"].startswith(old_prefix):
                dmap = info['dataset_map'].replace(old_prefix, subset, 1)
                new_path = new_path.replace(info['dataset_map'], dmap)
        return new_path

    def subset_from_map(dataset_map):
        return dataset_map.split('_')[0] if dataset_map else 'GLOBAL'
    
    def verbose_trace(msg, verbose=True):
        if verbose:
            print(msg)

    # --- Step 1: Normalize product name ---
    prod = prod.upper().strip()

    # --- Step 2: Resolve product metadata ---
    prod_info_map = product_defaults()
    if prod not in prod_info_map:
        print(f"❌ Product '{prod}' not found in prod_info_map.")
        return None

    actual_prod, default_dataset, default_type = prod_info_map[prod]
    dataset = dataset.upper().strip() if dataset else default_dataset
    dataset_type = dataset_type.upper().strip() if dataset_type else default_type

    # --- Step 3: Initialize a provenance dictionary to track decisions for debugging ---
    provenance_log = {
        'product': prod,
        'dataset': dataset,
        'dataset_type': dataset_type,
        'dataset_version': dataset_version,
        'period': period,
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

    # --- Step 6: Resolve dataset_map
    if not dataset_map:
        dataset_map, path = resolve_dataset_map(
        dataset_products=dataset_products,
        actual_prod=actual_prod,
        default_map=default_map,
        period=period,
        data_type=data_type,
        provenance_log=provenance_log,
        verbose=verbose
    )
    if not dataset_map or not path:
        print(f"⚠ No valid dataset_map found for product '{prod}' with data_type='{data_type}'")
        return []

    # --- Step 7: Redirect to *_STATS if period is provided ---
    if period and dataset_map:
        if period not in ['D', 'DD']:
            try:
                map_parts = dataset_map.split('_')
                map_name = map_parts[0]
                resolution = map_parts[1].replace('KM','').replace('km','')
                dataset_map = f"{map_name}_{resolution}KM_STATS"
                verbose_trace(f"📊 Redirected dataset_map to stats: {dataset_map}",verbose)
                provenance_log['resolution_steps'].append(f"Redirected dataset_map to STATS → {dataset_map}")

                # 🔁 Override dataset_type to EDAB_PRODUCTS for derived outputs
                dataset_type = 'EDAB_PRODUCTS'
                provenance_log['resolution_steps'].append(f"Redirected dataset_type to 'EDAB_PRODUCTS' for derived stats")
            except Exception as e:
                print(f"⚠ Failed to parse dataset_map '{dataset_map}' → {e}")
        provenance_log['resolution_steps'].append(f"Redirected dataset_map to STATS → {dataset_map}")

    # --- Step 8: Search for product path ---
    candidate_types = [dataset_type] if dataset_type in dataset_products else list(dataset_products.keys())    

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
    
    provenance_log['resolution_steps'].append(f"Resolved product path → {path}")
    provenance_log['final_path'] = path

    # --- Step 9: Handle ambiguous or missing paths ---
    if not matching_paths:
        verbose_trace(f"⚠ Product '{prod}' not found in dataset '{dataset}' under type '{dataset_type}' and map '{dataset_map}'")
        return []
    
    if len(matching_paths) > 1:
        raise ValueError(f"❌ Ambiguous product location: '{prod}' found in multiple types ({list(matching_paths.keys())}) for dataset '{dataset}'.")

    resolved_type, path = next(iter(matching_paths.items()))
    verbose_trace(f"✅ Found '{prod}' in '{resolved_type}' → {path}",verbose)

    # --- Step 10: Confirm resolved map exists ---
    resolved_map = next(
        (maptype for maptype in dataset_products[resolved_type] if dataset_map in maptype),
        None
    )
    if not resolved_map:
        verbose_trace(f"⚠ Map type '{dataset_map}' not found under '{resolved_type}' in '{dataset}'.",verbose)
        return None

    # --- Step 11: Return path or search for files ---
    if getfilepath:
        return path

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

                                                           


def make_product_output_dir(input_product,output_product,dataset=None,new_dataset_type='EDAB_PRODUCTS',subset='GLOBAL',create_dir=True):
    """
    Create the output path by replacing dataset_type and product,
    and create the directory if it does not exist.

    Parameters:
    - input_product (str): Original product name (e.g., 'CHL').
    - output_product (str): Replacement for product name (e.g. 'PPD' or 'PSC').
    - dataset (str): Dataset name for the original product (e.g. 'OCCCI').
    - new_dataset_type (str): Replacement for dataset_type (default: 'EDAB_PRODUCTS').
    - subset (str): Indicates the map subset region (default: GLOBAL)
    - create (bool): Whether to create the directory if it doesn't exist.

    Returns:
    - str: Transformed directory path.

    Raises:
    - ValueError: If original or new product name is missing or invalid.
    """
    
    defaults = product_defaults()

    # Validate input product
    if input_product.upper() not in defaults:
        raise ValueError(f"Input product '{input_product}' not found in defaults.")

    # Validate output product
    if output_product.upper() not in defaults:
        raise ValueError(f"Output product '{output_product}' not found in defaults.")    

    original_path = get_prod_files(input_product, dataset=dataset, getfilepath=True)
    info = parse_dataset_info(original_path)

    new_path = original_path
    new_path = new_path.replace(info["dataset_type"], new_dataset_type)
    new_path = new_path.replace(info["product"], output_product)
    for old_prefix in ["MAPPED", "BINNED"]:
        if info["dataset_map"].startswith(old_prefix):
            dmap = info['dataset_map'].replace(old_prefix, subset, 1)
            new_path = new_path.replace(info['dataset_map'],dmap)

    if create_dir:
        os.makedirs(new_path, exist_ok=True)

    return new_path