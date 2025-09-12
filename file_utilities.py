import os
import glob
from pathlib import Path
import stat
from bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

"""
FILE_UTILITIES is a collection of utility functions for handling file paths, directories, and permissions.

Main Functions:
    - dataset_defaults: Returns the default/primary datatype location and product for each "source" dataset
    - product_defaults: Returns the default/primary dataset and datatype location for each product
    - netcdf_product_defaults: Returns the data product name found in the original source netcdf files
    - get_nc_prod: Returns internal variable name and metadata for a given dataset and product.

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
    Aug 01, 2025 - KJWH: Initial code written
    Sep 10, 2025 - KJWH: Updated documentation
"""



def dataset_defaults():
    """
    Returns the default/primary datatype location and product for each "source" dataset

    Parameters:
        No inputs
    
    Returns:
        Dictionary of default dataset specific data types and products
    """

    # The default source data location and product for each dataset
    dataset_info_map = {
        'ACSPO': ('V2.8.1','MAPPED_2KM', 'SST'),
        'ACSPONRT': ('V2.8.1','MAPPED_2KM', 'SST'),
        'AVHRR': ('V5.3','MAPPED_4KM', 'SST'),
        'CORALSST': ('V3.1','MAPPED_5KM', 'SST'),
        'GLOBCOLOUR': ('V4.2.1','MAPPED_4KM', 'CHL'),
        'MUR': ('V4.1','MAPPED_1KM', 'SST'),
        'OCCCI': ('V6.0','MAPPED_4KM', 'CHL'),
        'OISST': ('V2','MAPPED_25KM', 'SST'),
        'PACE': ('V3.1','MAPPED_4KM','CHL')
    }

    return dataset_info_map

#--------------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------------
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
        dataset_map (str, optional): Specific map type folder to scan (e.g., 'MAPPED_4KM_DAILY').

    Returns:
        dict: Nested dictionary grouped by source_type → map_type → product.

            {
                "SOURCE": {
                    "MAPPED_4KM_DAILY": {
                        "CHL": "/.../MAPPED_4KM_DAILY/CHL",
                        "RRS": "/.../MAPPED_4KM_DAILY/RRS"
                    },
                }
                "EDAB_PRODUCTS": {
                    "MAPPED_4KM_DAILY": {}
                        "PPD": "/.../MAPPED_4KM_DAILY/PPD",
                        "PSC": "/.../MAPPED_4KM_DAILY/PSC",
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
    # Dynamically find the first key containing 'SOURCE'
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

#--------------------------------------------------------------------------------------
def get_prod_files(prod,
                   dataset=None,
                   dataset_version=None,
                   dataset_type=None,
                   dataset_map=None,
                   prod_type=None,
                   period=None,
                   getfilepath=False,
                   verbose=False):
    """
    Get the files for the specified product
    
    Parameters:
        prod (str, required): The name of the product to search for the files (e.g. CHL, SST)
        dataset (str, optional): The name of a dataset (e.g. 'OCCCI', 'ACSPO', 'GLOBCOLOUR') to 
            find the requested product. If none provided, will use the product default.
        dataset_type(str,optional): The name of the dataset type (e.g. 'SOURCE','EDAB_PRODUCTS') to
            search for the requested product. If none provided, will use the product default.
        dataset_map (str, optional): The dataset map type to search for the files (e.g. MAPPED_4KM_DAILY)
            If none is provided, will use the product defualt.
        
    Returns:
        Returns a list of files if found

    Raises:
        DirNotFoundError: If expected directory does not exist.
    """

    # Ensure the input dataset is capitalized
    prod = prod.upper().strip()

    # Get the default source data location and product for each dataset
    prod_info_map = product_defaults()
    if prod not in prod_info_map:
        print(f"❌ Product '{prod}' not found in prod_info_map.")
        return None

    actual_prod, default_dataset, default_type = prod_info_map[prod]
    dataset = dataset.upper().strip() if dataset else default_dataset
    dataset_type = dataset_type.upper().strip() if dataset_type else default_type
    #prod_type = prod_type.upper().strip() if prod_type else default_prod_type

    # Load default dataset info (version, map type)
    dataset_info_map = dataset_defaults()
    if dataset not in dataset_info_map:
        print(f"❌ Dataset '{dataset}' not found in dataset_info_map.")
        return None

    default_version, default_map, default_product = dataset_info_map[dataset]
    dataset_version = dataset_version or default_version
    dataset_map = dataset_map or default_map
    
    # Get available structure from get_dataset_products
    dataset_products = get_dataset_products(dataset)

    if not dataset_products:
        print(f"⚠ No product structure found for dataset '{dataset}'.")
        return None

    # Try locating product path. If dataset is specified, search both SOURCE and PRODUCT types
    # Match folder keys using flexible rules
    candidate_types = [
        key for key in dataset_products.keys()
        if key.startswith("SOURCE") or key.endswith("PRODUCTS")
    ]
    matching_paths = {}
    
    for dtype in candidate_types:
        if dtype not in dataset_products:
            continue

        # Try to resolve map type from folder keys using partial match
        resolved_map = next(
            (maptype for maptype in dataset_products[dtype] if dataset_map in maptype),
            None
        )

        if not resolved_map:
            continue

        # Check if product exists under this type/map
        prod_dict = dataset_products[dtype][resolved_map]
        if actual_prod in prod_dict:
            path = prod_dict[actual_prod]
            if prod_type:
                path = os.path.join(path, prod_type)
            matching_paths[dtype] = path

    # Handle results
    if not matching_paths:
        print(f"⚠ Product '{prod}' not found in any type under dataset '{dataset}'.")
        return None

    if len(matching_paths) > 1:
        raise ValueError(f"❌ Ambiguous product location: '{prod}' found in multiple types ({list(matching_paths.keys())}) for dataset '{dataset}'.")

    # Use the single resolved path
    resolved_type, path = next(iter(matching_paths.items()))
    if verbose:
        print(f"✅ Found '{prod}' in '{resolved_type}' → {path}")

    if not resolved_type:
        print(f"⚠ Dataset type '{dataset_type}' not found in products for '{dataset}'.")
        return None
        
    # Try to resolve map type from folder keys using partial match
    resolved_map = next(
        (maptype for maptype in dataset_products[resolved_type] if dataset_map in maptype),
        None
    )

    if not resolved_map:
        if verbose:
            print(f"⚠ Map type '{dataset_map}' not found under '{resolved_type}' in '{dataset}'.")
        return None

    try:
        path = dataset_products[resolved_type][resolved_map][actual_prod]
        if verbose:
            print(f"✅ Found path for '{prod}' → {path}")
    except KeyError:
        print(f"⚠ Product '{prod}' not found under {resolved_type}/{resolved_map} in '{dataset}'.")
        return None
    
    if getfilepath:
        return path
    
    if period:
        search_pattern = f"*{period}*.nc"
    else:
        search_pattern = "*.nc"
    if verbose:
        print(f"🔍 Searching for .nc files in: {path}")    
    
    if verbose: print(f"🔍 Searching for .nc files in: {path}")
    nc_files = glob.glob(os.path.join(path, "*.nc"))

    if not nc_files:
        print(f"⚠ No .nc files found in: {path}")
        return []

    if verbose:
        print(f"📦 Found {len(nc_files)} .nc files in: {path}")
    nc_files.sort()
    return nc_files
    

#--------------------------------------------------------------------------------------
def parse_dataset_info(path, base=None):
    """
    Extract structured metadata from a full dataset path.
    
    Parameters:
        path: str, full path to a dataset directory or file
        base: str, optional base path to strip (e.g. from get_datasets_source())

    Returns:
        dict: metadata including source name, version, resolution, product
    """
   
    if base is None:
        base = env["dataset_path"]
    
    # Handle list of files
    if isinstance(path, (list, tuple)):
        if len(path) == 0:
            raise ValueError("Empty path list provided.")
        dirnames = {os.path.dirname(p) for p in path}
        if len(dirnames) > 1:
            raise ValueError(f"Files span multiple directories: {dirnames}")
        path = path[0]  # Use first file after validation

    # Normalize to relative path
    path = os.path.abspath(path)
    base = os.path.abspath(base)
    relative = path[len(base):].lstrip(os.sep) if path.startswith(base) else path

    parts = relative.split(os.sep)
    
    if len(parts) < 4:
        raise ValueError(f"Insufficient path depth: {path}")

    return {
        "dataset": parts[0],       # OCCCI, CORALSST, etc.
        "version": parts[1],      # V6.0, V3.1, etc.
        "dataset_type": parts[2],  # SOURCE, EDAB_PRODUCTS, etc.
        "dataset_map": parts[3], # MAPPED_4KM_DAILY, etc.
        "product": parts[4] if len(parts) > 4 else "UNKNOWN"  # CHL, SST, PAR
    }

def get_dataset_vars(src_ds, requested=None, name=None, min_ndim=2):
    """
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
    """
    name = name or "source"
    available = set(src_ds.data_vars)

    if requested is None:
        print(f"ℹ️ No variables specified for '{name}', defaulting to all regriddable variables.")
        return [v for v in available if src_ds[v].ndim >= min_ndim]

    filtered = [v for v in requested if v in available and src_ds[v].ndim >= min_ndim]
    if not filtered:
        print(f"⚠️ No matching variables found in '{name}'. Defaulting to all regriddable variables.")
        return [v for v in available if src_ds[v].ndim >= min_ndim]

    missing = set(requested) - available
    if missing:
        print(f"⚠️ Variables not found in '{name}': {', '.join(sorted(missing))}")

    return filtered  

def set_file_permissions(filepath,desired_permissions=0o664):
    """
    Checks the permissions of a file and changes them if they don't match
    the desired permissions.

    Args:
        filepath (str): The path to the file.
        desired_permissions (int): The desired permissions in octal format (e.g., 0o664).
    """
    try:
        # Get current file mode
        current_mode = os.stat(filepath).st_mode
        # Extract only the permission bits
        current_permissions = stat.S_IMODE(current_mode)

        print(f"Current permissions for {filepath}: {oct(current_permissions)}")

        if current_permissions != desired_permissions:
            print(f"Permissions do not match. Changing to {oct(desired_permissions)}...")
            os.chmod(filepath, desired_permissions)
            print("Permissions updated successfully.")
        else:
            print("Permissions already match the desired settings.")
        
        file_info = os.stat(filepath)
        permissions_mode = file_info.st_mode
        return stat.filemode(permissions_mode)
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except PermissionError:
        print(f"Error: Permission denied when accessing or modifying {filepath}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    
def file_make(input_files, output_file):
    output_path = Path(output_file)
    if not output_path.exists():
        print(f"↪ Output file missing: {output_file}, recreate")
        return True

    output_mtime = output_path.stat().st_mtime
    for file in input_files:
        if os.path.getmtime(file) > output_mtime:
            print(f"↪ {file} newer than output file: {output_file}, recreate")
            return True

    return False
   

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