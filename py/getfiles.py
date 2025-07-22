import os
import glob
from pathlib import Path

def dataset_defaults_source():
    """
    Returns the default/primary datatype location and product for each "source" dataset

    Parameters:
        No inputs
    
    Returns:
        Dictionary of default dataset specific data types and products
    """

    # The default source data location and product for each dataset
    dataset_info_map = {
        'ACSPO': ('V2.8.1','MAPPED_2KM_DAILY', 'SST'),
        'ACSPONRT': ('V2.8.1','MAPPED_2KM_DAILY', 'SST'),
        'AVHRR': ('V5.3','MAPPED_4KM_DAILY', 'SST'),
        'CORALSST': ('V3.1','MAPPED_5KM_DAILY', 'SST'),
        'GLOBCOLOUR': ('V4.2.1','MAPPED_4KM_DAILY', 'CHL1'),
        'MUR': ('V4.1','MAPPED_1KM_DAILY', 'SST'),
        'OCCCI': ('V6.0','MAPPED_4KM_DAILY', 'CHL'),
        'OISST': ('V2','MAPPED_25KM_DAILY', 'SST')
    }

    return dataset_info_map

def product_defaults():
    """
    Returns the default/primary dataset and datatype location for each product

    Parameters:
        No inputs
    
    Returns:
        Dictionary of default dataset specific data types and products
    """

    # The default source data location and product for each dataset
    prod_info_map = {
        'CHL': ('OCCCI', 'SOURCE_DATA',''),
        'SST': ('ACSPO', 'SOURCE_DATA',''),
        'PPD': ('OCCCI', 'OUTPUTS','DAILY'),
        'PSC': ('OCCCI', 'OUTPUTS','DAILY'),
        'PAR': ('GLOBCOLOUR','SOURCE_DATA',''),
        'CHL_TEMP': ('GLOBCOLOUR','SOURCE_DATA',''),
        'SST_TEMP': ('ACSPONRT', 'SOURCE_DATA',''),
        'CHL_FRONTS': ('OCCCI', 'OUTPUTS','DAILY'),
        'SST_FRONTS': ('ACSPO', 'OUTPUTS','DAILY')
    }

    return prod_info_map

def get_datasets_source(preferred=None):
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
        "network": r"/Volumes/EDAB_Datasets/",
        "laptop": r"/Users/kimberly.hyde/Documents/nadata/DATASETS_SOURCE/",
        "server": r"/mnt/EDAB_Datasets/"
    }

    # If preferred is specified, try it first
    if preferred:
        if preferred in input_dirs:
            path = input_dirs[preferred]
            if os.path.exists(path):
                print(f"✓ Using specified input directory: [{preferred}] → {path}")
                return path
            else:
                print(f"✗ Preferred input source '{preferred}' not available — falling back to defaults.")
        else:
            print(f"⚠ Preferred label '{preferred}' not recognized. Valid options: {list(input_dirs.keys())}")

    # Try remaining options
    for label, path in input_dirs.items():
        if os.path.exists(path):
            print(f"✓ Using default input data directory: [{label}] → {path}")
            return path

    raise FileNotFoundError("No valid input data directory found.")

def get_dataset_dirs(dataset=None):
    """
    Search for subfolders within a directory and return a dictionary mapping
    folder names to their full paths.

    Parameters:
        root_dir (str): The directory to search.

    Returns:
        dict: Dictionary with subfolder names as keys and full paths as values.
    """
    root_dir = get_datasets_source()
    results = {}
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if dirname.endswith("SOURCE_DATA"):
                full_path = os.path.join(dirpath, dirname)
                
                # Get top-level folder name by splitting relative to root
                relative_parts = os.path.relpath(full_path, root_dir).split(os.sep)
                top_level = relative_parts[0] if relative_parts else os.path.basename(root_dir)

                if dataset:
                    if top_level == dataset:
                        return full_path
            
                results[top_level] = full_path
            if dirname.startswith("OUTPUT"):
                full_path = os.path.join(dirpath, dirname)

                # Get top-level folder name by splitting relative to root
                relative_parts = os.path.relpath(full_path, root_dir).split(os.sep)
                top_level = relative_parts[0] if relative_parts else os.path.basename(root_dir)

                if dataset:
                    if top_level == dataset:
                        return full_path
            
                # Add to results dict appropriately
                if top_level in results:
                    # If already a list, append; otherwise, convert to list
                    if isinstance(results[top_level], list):
                        results[top_level].append(full_path)
                    else:
                        results[top_level] = [results[top_level], full_path]
                else:
                    results[top_level] = full_path
                        
    if dataset:
        raise ValueError(f"No SOURCE_DATA folder found for top-level: {dataset}")
    
    return results

import os

def get_dataset_products(dataset, dataset_map=None):
    """
    Searches within the given SOURCE_DATA path and identifies available datatypes and products.

    Parameters:
        dataset (str): Top-level dataset folder name returned by find_source_data_subfolders.
        dataset_map (str, optional): Specific dataset map type folder to search within (e.g. 'MAPPED_4KM_DAILY').

    Returns:
        dict: Dictionary of the form:
              {
                  "MAPPED_4KM_DAILY": {
                      "CHL": "/path/to/MAPPED_4KM_DAILY/CHL",
                      "SST": "/path/to/MAPPED_4KM_DAILY/SST"
                  },
                  ...
              }
    """
    # First, get all available SOURCE_DATA paths
    all_sources = get_dataset_dirs()
    
    if dataset not in all_sources:
        raise ValueError(f"❌ No SOURCE_DATA folder found for dataset: '{dataset}'")
    
    source_data_path = all_sources[dataset]
    results = {}

     # Define inner helper to collect product folders
    def get_valid_products(path):
        return {
            entry: os.path.join(path, entry)
            for entry in os.listdir(path)
            if not entry.startswith('.') and os.path.isdir(os.path.join(path, entry))
        }

    # If user specifies a particular dataset_map
    if dataset_map:
        dtype_path = os.path.join(source_data_path, dataset_map)
        if dataset_map.startswith('.') or not os.path.isdir(dtype_path):
            print(f"⚠ Skipping invalid or hidden dataset type: '{dataset_map}'")
            return results

        print(f"🔍 Scanning specified datatype: {dataset_map}")
        results[dataset_map] = get_valid_products(dtype_path)

        
    else:
        # Loop through all datatypes within the dataset
        for dtype in os.listdir(source_data_path):
            if dtype.startswith('.'):
                continue  # Skip hidden/system folders
            
            dtype_path = os.path.join(source_data_path, dtype)
            if not os.path.isdir(dtype_path):
                continue

            print(f"🔍 Scanning datatype: {dtype}")
            results[dtype] = get_valid_products(dtype_path)

    return results

def get_prod_path(dataset,dataset_version=None,dataset_type=None,dataset_map=None,prod=None,period=None):
    """
    Get the downloaded source path for a specified dataset.
    
    Parameters:
        dataset (str,required): The name of a specified dataset (e.g. 'OCCCI', 'ACSPO', 'GLOBCOLOUR').
        dataset_map (str, optional): The dataset map type to search for the files (e.g. MAPPED_4KM_DAILY)
            If none is provided, will use the primary dataset type as default
        prod (str, optional): The name of the product to search for the files (e.g. CHL, SST)
            If none is provided, will default to the primary product as default
        
    Returns:
        str: The path for the specified dataset, dataset_map and prod

    Raises:
        DirNotFoundError: If expected directory does not exist.
    """

    # Ensure the input dataset is capitalized
    dataset = dataset.upper()

    # Get the default source data location and product for each dataset
    dataset_info = dataset_defaults_source()

    # If dataset_map or prod name are not provided, use default
    if dataset_version is None or dataset_map is None or prod is None:
        info = dataset_info.get(dataset) # type: ignore
        if info:
            dataset_version = info[0] if dataset_version is None else dataset_version
            dataset_map = info[1] if dataset_map is None else dataset_map
            prod = info[2] if prod is None else prod
    
    # If the period is not provided, use product default
    if period is None:
        prod_info = product_defaults()
        pinfo = prod_info.get(prod.upper())
        period = pinfo[2] if period is None else period.upper()
    

    # Get the dataset_map and products found within a dataset
    dataset_prods = get_dataset_products(dataset)

    # Extract the filepath for the specified dataset_map and product
    filepath = dataset_prods.get(dataset_map, {}).get(prod)
    
    # Return filepath
    return Path(filepath)

def get_prod_files(prod,dataset=None,dataset_version=None,dataset_type=None,dataset_map=None,period=None):
    """
    Get the files for the specified product
    
    Parameters:
        prod (str, required): The name of the product to search for the files (e.g. CHL, SST)
        dataset (str, optional): The name of a dataset (e.g. 'OCCCI', 'ACSPO', 'GLOBCOLOUR') to 
            find the requested product. If none provided, will use the product default.
        dataset_type(str,optional): The name of the dataset type (e.g. 'SOURCE_DATA','OUTPUT_DATA') to
            search for the requested product. If none provided, will use the product default.
        dataset_map (str, optional): The dataset map type to search for the files (e.g. MAPPED_4KM_DAILY)
            If none is provided, will use the product defualt.
        
    Returns:
        Returns a list of files if found

    Raises:
        DirNotFoundError: If expected directory does not exist.
    """

    # Ensure the input dataset is capitalized
    prod = prod.upper()

    # Get the default source data location and product for each dataset
    prod_info = product_defaults()
    info = prod_info.get(prod) # type: ignore
        
    # If dataset_map or period are not provided, use default
    if dataset is None or dataset_type is None or period is None:         
        if info:
            dataset = info[0] if dataset is None else dataset.upper()
            dataset_type = info[1] if dataset_type is None else dataset_type
            period = info[2] if period is None else period.upper()
    
    # If the dataset version, type or map are not provided use the dataset specific defaults
    if dataset_version is None or dataset_map is None:
            dataset_info = dataset_defaults_source()
            dinfo = dataset_info.get(dataset)
            dataset_version = dinfo[0] if dataset_version is None else dataset_version
            dataset_map = dinfo[1] if dataset_map is None else dataset_map
            
    
    print(f"Searching for {period} {prod} files in dataset {dataset} (version={dataset_version}, type={dataset_type}, map={dataset_map})")

    # Get the dataset path
    filepath = get_prod_path(dataset,dataset_version=dataset_version,dataset_type=dataset_type,dataset_map=dataset_map,prod=prod,period=period)
    return filepath

    # Find the .nc files
    print(f"Searching {filepath} for .nc files...")
    files = list(filepath.glob("*.nc"))
    
    # Return files
    if not files:
        files = []
        raise FileNotFoundError(f"No .nc files found in directory: {filepath}")
    else:
        print(f"Found {len(files)} .nc files in directory: {filepath}")
        return files
    
    
def get_output_prod_files(dataset,dataset_map=None,prod=None):
    """
    Get the source path for a specified locally created dataset.
    
    Parameters:
        dataset (str): The name of a specified dataset (e.g. 'OCCCI', 'ACSPO', 'GLOBCOLOUR').
        dataset_map (str, optional): The dataset map type to search for the files (e.g. MAPPED_4KM_DAILY)
            If none is provided, will use the primary dataset type as default
        prod (str, optional): The name of the product to search for the files (e.g. CHL, SST)
            If none is provided, will default to the primary product as default
        
    Returns:
        str: The path for the specified output dataset, dataset_map and prod

    Raises:
        DirNotFoundError: If expected directory does not exist.
    """

    # Ensure the input dataset is capitalized
    dataset = dataset.upper()

    # Get the default source data location and product for each dataset
    dataset_info = dataset_defualts()
    
    # If dataset_map or prod name are not provided, use default
    if dataset_map is None or prod is None:
        info = dataset_info.get(dataset) # type: ignore
        if info:
            dataset_map = info[0] if dataset_map is None else dataset_map
            prod = info[1] if prod is None else prod
    
    # Get the dataset path
    filepath = get_prod_path(dataset,dataset_map=dataset_map,prod=prod)
    
    # Find the .nc files
    print(f"Searching {filepath} for .nc files...")
    files = list(filepath.glob("*.nc"))
    
    # Return files
    if not files:
        files = []
        raise FileNotFoundError(f"No .nc files found in directory: {filepath}")
    else:
        print(f"Found {len(files)} .nc files in directory: {filepath}")
        return files
    

