import os
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

"""
Purpose:
    DATASET_UTILITIES is a collection of utility functions for handling dataset specific tasks.

Main Functions:
    - dataset_defaults: Returns the default/primary datatype location and product for each "source" dataset
    - get_default_dataset_product: Returns the default product (e.g., SST, CHL) for a given dataset.
    - get_datasets_source: Checks for the first valid input data directory from a given dictionary.
    - get_dataset_products: Searches for products within the given SOURCE and PRODUCTS paths for a dataset
    - parse_dataset_info: Extracts structured metadata from a full dataset path.
    - resolve_dataset_map: Resolves the appropriate dataset_map folder for a given product within a dataset structure.
    - get_dataset_vars: Determines which variables to regrid from a source dataset.

Helper Functions:
    - validate_inputs: Validates that input data arrays are xarray.DataArray and have matching shapes
    
Copywrite: 
    Copyright (C) 2025, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on September 18, 2025 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Sep 18, 2025 - KJWH: Moved dataset specific functions from file_utilities to dataset_utitlities
    Mar 16, 2026 - KJWH: Updated parse_dataset_info(). 
                            • It can now parse mixed file types 
                            • It now caches dataset information to reduce overhead when working with a large number of files
                            • It validates the dataset by matching it with the default datasets 

"""

def dataset_defaults():
    """
    Returns the default/primary version, datatype and main product EDAB datasets

    Parameters:
        No inputs
    
    Returns:
        Dictionary of default dataset specific data types and products
    """

    # The default source data location and product for each dataset
    dataset_info_map = {
        'ACSPO': ('V2.8.1','GLOBAL_2KM', 'SST'),
        'ACSPONRT': ('V2.8.1','GLOBAL_2KM', 'SST'),
        'AVHRR': ('V5.3','GLOBAL_4KM', 'SST'),
        'CORALSST': ('V3.1','GLOBAL_5KM', 'SST'),
        'GLOBCOLOUR': ('V4.2.1','GLOBAL_4KM', 'CHL'),
        'GLORYS': ('V12','GLOBAL_9KM','BTEMP'),
        'MUR': ('V4.1','GLOBAL_1KM', 'SST'),
        'OCCCI': ('V6.0','GLOBAL_4KM', 'CHL'),
        'OISST': ('V2','GLOBAL_25KM', 'SST'),
        'PACE': ('V3.1','GLOBAL_4KM','CHL')
    }

    return dataset_info_map

def get_default_dataset_product(dataset_name):
    """
    Return the default product (e.g., SST, CHL) for a given dataset.

    Parameters:
        dataset_name (str): Name of the dataset (e.g., 'OCCCI', 'ACSPO').

    Returns:
        The default product name for the specified dataset (str).

    Raises
        KeyError - If the dataset is unknown.
        ValueError - If the dataset entry does not contain a product field.
    """
    info = dataset_defaults()
    if dataset_name not in info:
        raise KeyError(f"Unknown dataset '{dataset_name}'. "
                       f"Available datasets: {list(info.keys())}")
    entry = info[dataset_name]
    if len(entry) < 3:
        raise ValueError(
            f"Dataset '{dataset_name}' does not contain a product entry. "
            f"Expected tuple of length ≥ 3, got: {entry}"
        )
    return entry[2]

def get_datasets_source(preferred=None,verbose=False):
    """
    Checks for the first valid input data directory from a given dictionary.
    
    Parameters:
        preferred (str, optional): A label indicating the preferred data source 
                                   (e.g. 'network', 'laptop', 'server').
        
    Returns:
        str: Path to the first existing data directory.

    Raises:
        FileNotFoundError: If none of the provided directories exist.
    """
    
    # List of input directories in the ordered preference (e.g. if two are available, the first will be choosen)
    input_dirs = {
        "laptop": r"/Users/kimberly.hyde/Documents/nadata/DATASETS/",
        "network": r"/Volumes/EDAB_Datasets/",
        "server": r"/mnt/EDAB_Datasets/"
    }

    # If preferred is specified, try it first
    if preferred:
        if preferred in input_dirs:
            path = input_dirs[preferred]
            if os.path.exists(path):
                if verbose:
                    print(f"✓ Using specified input directory: [{preferred}] → {path}")
                return path
            else:
                print(f"✗ Preferred input source '{preferred}' not available — falling back to defaults.")
        else:
            print(f"⚠ Preferred label '{preferred}' not recognized. Valid options: {list(input_dirs.keys())}")

    # Try remaining options
    for label, path in input_dirs.items():
        if os.path.exists(path):
            if verbose:
                print(f"✓ Using default input data directory: [{label}] → {path}")
            return path

    raise FileNotFoundError("No valid input data directory found.")

#--------------------------------------------------------------------------------------
def get_dataset_dirs(dataset=None,verbose=True):
    """
    Search for subfolders within a directory and return a dictionary mapping
    folder names to their full paths.

    Parameters:
        root_dir (str): The directory to search.
        verbose (bool): Whether to print status messages.

    Returns:
        dict: Dictionary with subfolder names as keys and full paths as values.
              If dataset is provided, returns only its matching subfolders.

    """
    root_dir = env["dataset_path"]
    dataset_info_map = dataset_defaults()
    results = {}
    for dirpath, dirnames, _ in os.walk(root_dir): 
        for dirname in dirnames:
            if dirname.startswith("SOURCE") or dirname.endswith("PRODUCTS"):
                full_path = os.path.join(dirpath, dirname)
               
                # Get top-level dataset name
                relative_parts = os.path.relpath(full_path, root_dir).split(os.sep)
                dataset_name = relative_parts[0] if relative_parts else os.path.basename(root_dir)

                # Skip datasets not in dataset_info_map if no specific dataset is requested
                if not dataset and dataset_name not in dataset_info_map:
                    continue

                # Filter by dataset if specified
                if dataset and dataset_name != dataset:
                    continue
                
                # Initialize nested dict if needed
                if dataset_name not in results:
                    results[dataset_name] = {}

                results[dataset_name][dirname] = full_path
                        
    if dataset:
        if dataset not in results:
            raise ValueError(f"No matching folders found for dataset: {dataset}")
        return results[dataset]

    if verbose:
        print("No dataset specified. Returning available datasets from dataset_info_map.")

    return results

#--------------------------------------------------------------------------------------
def get_dataset_products(dataset, dataset_map=None, verbose=False):
    """
    Searches within the given SOURCE and OUTPUT paths for a dataset,
    organizes the results into:
        {
            source_type: {
                map_type: {
                    product: path
                }
            }
        }

    Parameters:
        dataset (str): Top-level dataset folder name.
        dataset_map (str, optional): Specific map type folder to scan (e.g., 'GLOBAL_4KM_DAILY','NES_2KM_STATS').

    Returns:
        dict: Nested dictionary grouped by source_type → map_type → product.

            {
                "SOURCE": {
                    "GLOBAL_4KM_DAILY": {
                        "CHL": "/.../GLOBAL_4KM_DAILY/CHL",
                        "RRS": "/.../GLOBAL_4KM_DAILY/RRS"
                    },
                }
                "EDAB_PRODUCTS": {
                    "GLOBAL_4KM_DAILY": {}
                        "PPD": "/.../GLOBAL_4KM_DAILY/PPD",
                    "NES_4KM_DAILY": {}
                        "PSC": "/.../NES_4KM_STATS/PSC",
                    }
                },
                ...
            }
    """
    # First, get all available SOURCE paths
    all_sources = get_dataset_dirs(verbose=False)
    
    if dataset not in all_sources:
        print(f"❌ {dataset}' not found in available sources.")
        return None
    
    # Extract nested SOURCE and OUTPUT paths
    # Dynamically find the first key containing 'SOURCE' and 'OUTPUT'
    source_data_path = next(
        (path for key, path in all_sources[dataset].items() if "SOURCE" in key),
        None
        )    
    output_data_path = next(
        (path for key, path in all_sources[dataset].items() if "OUTPUT" in key),
        None
        )    

    if not source_data_path and not output_data_path:
        print(f"❌ No valid SOURCE or OUTPUT path found for '{dataset}'.")
        return None
    
    if not source_data_path:
        if verbose:
            print(f"⚠ No 'SOURCE' path found for '{dataset}' – proceeding with OUTPUT only.")

    if not output_data_path:
        if verbose:
            print(f"⚠ No 'OUTPUT' path found for '{dataset}' – proceeding with SOURCE only.")

    results = {}
    
    def scan_data_path(base_path, tag):
        """
        Returns: {map_type: {product: path}}
        """
        tag_results = {}

        for map_type in os.listdir(base_path):
            if map_type.startswith('.'):
                continue

            maptype_path = os.path.join(base_path, map_type)
            if not os.path.isdir(maptype_path):
                continue

            # Filter if user specified a single map type
            if dataset_map and dataset_map not in map_type:
                continue

            products = {
                product: os.path.join(maptype_path, product)
                for product in os.listdir(maptype_path)
                if not product.startswith('.') and os.path.isdir(os.path.join(maptype_path, product))
            }

            if products:
                tag_results[map_type] = products

        return tag_results

    # Scan both folders if available
    # Scan each source key and preserve its label exactly
    has_valid_path = False

    for source_type, base_path in all_sources[dataset].items():
        if not os.path.isdir(base_path):
            continue

        source_results = scan_data_path(base_path, source_type)

        if source_results:
            results[source_type] = source_results
            has_valid_path = True

    if not has_valid_path:
        print(f"❌ No valid SOURCE or OUTPUT path found for '{dataset}'.")
        return None

    return results

def resolve_dataset_map(dataset_products,
                        prod=None,
                        data_type=None,
                        default_map=None,
                        period=None,
                        verbose=False,
                        provenance_log=None):
    """
    Resolves the appropriate dataset_map folder for a given product within a dataset structure.

    This function searches across all dataset types (e.g., 'SOURCE', 'PRODUCTS') and identifies
    which map folder (e.g., 'NES_4KM_DAILY', 'GLOBAL_4KM_STATS') contains the requested product.
    It supports filtering by explicit data_type (e.g., 'DAILY', 'STATS', 'ANOMS') and fallback logic
    based on period codes (e.g., 'M', 'D3', 'YEAR').

    Parameters
    ----------
    dataset_products : dict
        Nested dictionary of dataset structure returned by get_dataset_products().
        Format: {dataset_type: {map_folder: {product: path}}}
    prod : str
        The product name to locate (e.g., 'CHL', 'PSC').
    data_type : str, optional
        Explicit map type to filter by (e.g., 'DAILY', 'STATS', 'ANOMS').
        If provided, only folders ending in _{data_type.upper()} will be considered.
    default_map : str, optional
        Default map folder name (e.g., 'GLOBAL_4KM_DAILY'). Used for provenance logging.
    period : str, optional
        Period code (e.g., 'D', 'M', 'D3'). Used to infer map type if data_type is not provided.
    verbose : bool, optional
        If True, prints resolution steps and decisions.
    provenance_log : dict, optional
        Dictionary to append resolution steps for debugging and audit trails.

    Returns
    -------
    tuple
        (dataset_map, path) if resolved successfully.
        (None, None) if no matching folder is found or ambiguity cannot be resolved.

    Raises
    ------
    ValueError
        If multiple folders contain the product and no resolution strategy succeeds.
    """

    if prod == None:
        prod = 'CHL'

    candidate_maps = []
    for dtype in dataset_products:
        for map_key, prod_dict in dataset_products[dtype].items():
            if prod in prod_dict:
                candidate_maps.append((map_key, prod_dict[prod]))

    if not candidate_maps:
        return None, None

    # --- Handle multiple matches ---
    if len(candidate_maps) > 1:

        # ✅ If data_type is explicitly provided, filter strictly by it
        if data_type:
            typed_maps = [m for m in candidate_maps if m[0].endswith(f"_{data_type.upper()}")]
            if typed_maps:
                selected_map = typed_maps[0]
                if provenance_log is not None:
                    provenance_log['resolution_steps'].append(f"Filtered by data_type='{data_type}' → {selected_map[0]}")
                if verbose:
                    print(f"📦 Selected map by data_type → {selected_map[0]}")
                return selected_map
            else:
                if provenance_log is not None:
                    provenance_log['resolution_steps'].append(f"No map folder found for data_type='{data_type}'")
                if verbose:
                    print(f"⚠ No map folder found for data_type='{data_type}'")
                return None, None

        # ✅ Otherwise, fall back to period-based logic
        daily_maps = [m for m in candidate_maps if m[0].endswith('_DAILY')]
        stats_maps = [m for m in candidate_maps if m[0].endswith('_STATS')]

        if period in [None, 'D', 'DD'] and daily_maps:
            if provenance_log is not None:
                provenance_log['resolution_steps'].append(f"Defaulted to DAILY map → {daily_maps[0][0]}")
            if verbose:
                print(f"📦 Defaulted to DAILY map → {daily_maps[0][0]}")
            return daily_maps[0]

        elif period not in ['D', 'DD'] and stats_maps:
            if provenance_log is not None:
                provenance_log['resolution_steps'].append(f"Defaulted to STATS map for period '{period}' → {stats_maps[0][0]}")
            if verbose:
                print(f"📊 Defaulted to STATS map → {stats_maps[0][0]}")
            return stats_maps[0]

        # ❌ Ambiguity remains — raise error
        maps_found = [m[0] for m in candidate_maps]
        raise ValueError(f"❌ Ambiguous product location: '{prod}' found in multiple maps ({maps_found}). Please specify dataset_map.")

    # ✅ Single match — return directly
    if verbose:
        print(f"📦 Auto-resolved dataset_map → {candidate_maps[0][0]}")
    return candidate_maps[0]



def parse_dataset_info(pathlist, base=None):
    """
    Extract structured metadata from a full dataset path.
    
    Parameters:
        path: str, full path to a dataset directory or file
        base: str, optional base path to strip (e.g. from get_datasets_source())

    Returns:
        dict: metadata including source name, version, resolution, product

    Updates:

    """
   
    dataset_info = dataset_defaults()

    if base is None:
        base = env["dataset_path"]
    
    base = os.path.abspath(base)
    
    # Use a cache to avoid re-parsing the same directory structure multiple times
    dir_cache = {}
    results = []

    # Ensure we are working with a list
    paths = pathlist if isinstance(pathlist, (list, tuple)) else [pathlist]

    for path in paths:
        # Get directory name - this is the fastest way to categorize files
        dirname = os.path.dirname(path)
        if dirname not in dir_cache:
            # Only perform the expensive string splitting when we see a new directory
            abs_dir = os.path.abspath(dirname)
            relative = abs_dir[len(base):].lstrip(os.sep) if abs_dir.startswith(base) else abs_dir
            parts = relative.split(os.sep)

            if parts and parts[0].upper() in dataset_info:
                dir_cache[dirname] = {
                    "dataset": parts[0] if len(parts) > 0 else "UNKNOWN",
                    "version": parts[1] if len(parts) > 1 else "UNKNOWN",
                    "dataset_type": parts[2] if len(parts) > 2 else "UNKNOWN",
                    "dataset_map": parts[3] if len(parts) > 3 else "UNKNOWN",
                    "product": parts[4] if len(parts) > 4 else "UNKNOWN"
                }
            else:
                dir_cache[dirname] = {
                    "dataset": "UNKNOWN",
                    "version": "UNKNOWN",
                    "dataset_type": "UNKNOWN",
                    "dataset_map": "UNKNOWN",
                    "product": "UNKNOWN"
                }
        results.append(dir_cache[dirname])
    return results

"""
def get_dataset_vars(src_ds, requested=None, name=None, min_ndim=2):
    
    Determine which variables to regrid from a source dataset.

    Parameters
    ----------
    src_ds : xarray.Dataset
        The source dataset containing variables to regrid.
    requested : list or None
        List of variable names to regrid. If None or invalid, defaults to all suitable variables.
    name : str or None
        Optional name used in diagnostics (e.g., filename or label).
    min_ndim : int
        Minimum number of dimensions a variable must have to be considered regriddable.

    Returns
    -------
    list
        Valid variable names to regrid.
    
    name = name or "source"
    available = set(src_ds.data_vars)

    if requested is None:
        print(f" No variables specified for '{name}', defaulting to all regriddable variables.")
        return [v for v in available if src_ds[v].ndim >= min_ndim]

    filtered = [v for v in requested if v in available and src_ds[v].ndim >= min_ndim]
    if not filtered:
        print(f"⚠️ No matching variables found in '{name}'. Defaulting to all regriddable variables.")
        return [v for v in available if src_ds[v].ndim >= min_ndim]

    missing = set(requested) - available
    if missing:
        print(f"⚠️ Variables not found in '{name}': {', '.join(sorted(missing))}")

    return filtered  
"""