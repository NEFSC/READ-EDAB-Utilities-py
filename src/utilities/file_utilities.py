import os
import glob
from pathlib import Path
import stat
from utilities.bootstrap.environment import bootstrap_environment
from utilities import parse_dataset_info, extract_period_code, get_period_info, get_period_dates, get_source_file_dates
from utilities import product_defaults, dataset_defaults, get_dataset_products, resolve_dataset_grid, get_daterange
env = bootstrap_environment(verbose=False)

"""
Purpose:
    FILE_UTILITIES is a collection of utility functions for handling file paths, directories, and permissions.

Main Functions:
    - get_file_dates: Extracts dates from a list of filenames
    - file_make: Checks to see if a file exists and if it needs to be remade based on the mtimes of the input files
    - set_file_permissions: Checks the permissions of a file and changes them if they don't match the desired permissions.
    - corrupt_file_detector: Searches for corrupt files in a list of files
    - get_filepath: Resolves the exact directory path for a product, optionally creating it if it doesn't exist.
    - get_prod_files: Retrieves a list of NetCDF files for a specified product, optionally filtering by date range.

Helper Functions:
    - verbose_trace: Internal function for conditional debug printing.
    
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
    Sep 25, 2025 - KJWH: Added get_file_dates
    Mar 20, 2026 - KJWH: Added corrupt_file_detector
"""
import os
from pathlib import Path

def file_parser(files): 
    """
    Parses a list of input file paths into structured metadata. 
        • Uses parse_dataset_info() to extract dataset specific information
        • Uses get_file_dates() to extract dates from the file names
        • uses extract_period_code() to get period specific information

    Parameters:
        files (list): List of strings containing full paths to .nc or similar files.

    Returns:
        list[dict]: A list of dictionaries where each entry contains:
            - full_file_path: The original path.
            - directory/file_name/extension: Basic OS path info.
            - dataset/version/product: Validated info from the directory structure.
            - start_date/end_date: Temporal info from the file string.
            - period_details: Nested dict of period codes and associated static info.

    """
    if not files:
        return{}
    # If 'files' is a single string, wrap it in a list
    if isinstance(files, str):
        files = [files]

    # 1. Extract Dataset
    dataset_metadata = parse_dataset_info(files)
    
    # 2. Extract Date Ranges for each file
    file_dates = get_file_dates(files)

    # 3. Extract period specific information if available
    perinfo = extract_period_code(files)
    percode = [info['code'] for info in perinfo]
    fullperiod = [info['full_period'] for info in perinfo]
        
    # 4. Loop through files to add information
    results = []
    
    # Using zip allows us to process the parallel lists efficiently
    for i, file_path in enumerate(files):        
        directory, full_name = os.path.split(file_path)
        file_name, extension = os.path.splitext(full_name)
        idataset = dataset_metadata[i]
        file_entry = {
            'full_file_path': file_path,
            'directory': directory,
            'file_name': file_name,
            'extension': extension,
            **idataset,
            'start_date': file_dates[i][0],
            'end_date': file_dates[i][1],
        }

        # Add period info if it was found in Step 3        
        if percode is not None:
            icode = percode[i]
            if icode is not None:
                period_info = get_period_info(icode)
                period_dates = get_period_dates(fullperiod[i])
                file_entry['period_details'] = {
                    'period_code': icode,
                    'period_full_code': fullperiod[i],
                    'metadata': get_period_info(icode)
                }
        else:
            file_entry['period_details'] = None
        
        results.append(file_entry)

    return results




def get_file_dates(files, source_format="yyyymmdd", period_format="%Y%m%d", placeholder=None):
    """
    Extracts (start,end) dates from a list of filenames, dispatching to either:
        • Uses get_period_dates() in batch for files starting with a known period code.
        • Uses get_source_file_dates() in batch for all others.
        • Preserves input order.
        • Returns (placeholder, placeholder) for unmatched files.

    Parameters
        files (list of str or Path): Paths or basenames of your files.
        source_format (str): Format passed to get_source_file_dates(). default "yyyymmdd"
        period_format (str): Format passed to extract_period_dates(). default "%Y%m%d"
        placeholder (any): Value to use for start/end when extraction fails. Default is NA, but can be set to '' or None as needed.

    
    Returns a dict mapping each input filename to a (start, end) tuple:
        - Raw SOURCE files yield (date, date)
        - PERIOD files yield (start_date, end_date)
        - Files that fail both extractors map to (placeholder, placeholder)

    
    Returns
    -------
        list of tuple: [(start_str, end_str)] aligned with the input file order
    """
    
    from utilities import period_info, get_period_dates, get_source_file_dates
    
    period_codes = set(period_info().keys())

    # Track which files are period-coded vs source
    period_files = []
    source_files = []
    period_indices = []
    source_indices = []

    for i, f in enumerate(files):
        fname = str(f)
        token = Path(fname).name.split("_", 1)[0]
        if token in period_codes:
            period_files.append(fname)
            period_indices.append(i)
        else:
            source_files.append(fname)
            source_indices.append(i)

    # Initialize output
    results = [None] * len(files)

    # Batch process period-coded files
    if period_files:
        period_ranges = get_period_dates(period_files, format=period_format)
        for i, rng in zip(period_indices, period_ranges):
            results[i] = rng if rng else (placeholder, placeholder)

    # Batch process source files
    if source_files:
        source_dates = get_source_file_dates(source_files, format=source_format, placeholder=placeholder)
        for i, date in zip(source_indices, source_dates):
            if date and date != placeholder:
                results[i] = (date, date)
            else:
                results[i] = (placeholder, placeholder)

    return results

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

def corrupt_file_detector(file_list):
    """Deep-dives into a list of files to find specific 'bad' ones."""
    issues = []
    for f in file_list:
        # Check 1: Does it exist?
        if not os.path.exists(f):
            # 🎯 CHANGED: Now appending a tuple (filepath, error_reason)
            issues.append((f, "File Missing"))
            continue
        
        # Check 2: Can we read it? (Permissions)
        if not os.access(f, os.R_OK):
            # 🎯 CHANGED: Tuple
            issues.append((f, "Permission Denied"))
            continue

        # Check 3: Is it a valid NetCDF?
        try:
            with xr.open_dataset(f, engine='netcdf4', decode_timedelta=False) as test:
                # Trigger a load of the coordinates to ensure it's not truncated
                _ = test.coords.to_dataset().load()
        except Exception as e:
            # 🎯 CHANGED: Tuple
            issues.append((f, f"Corrupt/Invalid: {str(e)}"))
            
    return issues

def get_filepath(prod, dataset=None, period='D',make_dir=False, verbose=False, **kwargs):
    """
    Resolves the exact directory path for a product.
    If the product is new, uses the dataset's default product (e.g., CHL) as a template.
    Optionally creates the directory if make_dir=True.
    """
    def verbose_trace(msg):
        if verbose: print(f"DEBUG [PATH - {prod}]: {msg}")

    # --- 1. Setup & Defaults ---
    prod = prod.upper().strip()
    prod_info_map = product_defaults()
    if prod not in prod_info_map:
        raise ValueError(f"Product '{prod}' not found in product defaults.")
    
    actual_prod, default_dataset, default_type = prod_info_map[prod]
    dataset = dataset.upper().strip() if dataset else default_dataset

    dataset_grid = kwargs.get('dataset_grid')
    map_subset = kwargs.get('map_subset')   
    data_type = kwargs.get('data_type')
    dataset_type = kwargs.get('dataset_type', default_type).upper()
    period = (period or 'D').upper()

    dataset_info_map = dataset_defaults()
    _, default_grid, default_product = dataset_info_map[dataset]

    dataset_products = get_dataset_products(dataset)
    filtered_structure = {dataset_type: dataset_products[dataset_type]} if dataset_type in dataset_products else dataset_products

    # --- 2. Try to find the product in the dictionary ---
    verbose_trace("Attempting to find existing grid...")
    resolved_grid, path = resolve_dataset_grid(
        filtered_structure, 
        prod=actual_prod, 
        default_grid=dataset_grid or default_grid,
        period=period,
        data_type=data_type,
        verbose=verbose
    )

    # --- 3. Fallback: Template a new path if it doesn't exist ---
    if not path:
        verbose_trace(f"'{actual_prod}' not found. Templating from '{default_product}'...")
        
        #  Get the filepath for the default daily product
        template_path = get_filepath(
            default_product, 
            dataset=dataset, 
            period='D', 
            verbose=verbose
        )

        if not template_path:
            raise ValueError(f"Critical error: Could not find base template for {default_product}")

        # Swap product name
        base_dir, _ = os.path.split(template_path)
        path = os.path.join(base_dir, actual_prod)
        
        # Reroute the folder based on the product defaults (e.g. SOURCE -> PRODUCTS)
        if "/SOURCE/" in path and dataset_type != "SOURCE":
            path = path.replace("/SOURCE/", f"/{dataset_type}/")
            
        resolved_grid = default_grid

    # --- 4. Transform path for Subsets or Derived Periods ---
    is_derived = period and period.upper() not in ['D', 'DD']
    needs_subset = map_subset and resolved_grid and not resolved_grid.startswith(map_subset.upper())

    if is_derived or data_type == 'ANOMS' or needs_subset:
        verbose_trace("Transforming path for derived product or subset...")
        p_info = get_period_info(period) if period else {}
        suffix = p_info.get('folder_name', data_type or 'DERIVED').upper()
        if data_type == 'ANOMS' or (period and 'ANOM' in period):
            suffix = 'ANOMS'

        parts = resolved_grid.split('_') if resolved_grid else ['GLOBAL', '4KM']
        region = map_subset.upper() if map_subset else parts[0]
        resolution_str = kwargs.get('resolution', parts[1] if len(parts) > 1 else '4KM')
        if isinstance(resolution_str, (int, float)): 
            resolution_str = f"{resolution_str}KM"

        new_grid = f"{region}_{resolution_str}_{suffix}"
        
        # Replace the grid folder name in the physical string
        res_info = parse_dataset_info(path)
        old_grid = res_info.get("dataset_grid", res_info.get("dataset_map")) if res_info else resolved_grid        
        if old_grid:
            path = path.replace(old_grid, new_grid)
        
        # Ensure derived data goes to PRODUCTS
        if "/SOURCE/" in path:
            path = path.replace("/SOURCE/", "/PRODUCTS/")

    # --- 5. Create Directory if requested ---
    if make_dir:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            verbose_trace(f"🛠 Created directory: {path}")
        else:
            verbose_trace(f"Directory already exists: {path}")

    return path

def get_prod_files(prod, dataset=None, period='D', verbose=False, **kwargs):
    """
    Retrieves a list of NetCDF files for a specified product.
    Relies on get_filepath() to resolve the correct directory.

    Required Parameters:
        prod (str): Product name (e.g. 'CHL', 'PSC', 'SST').

    Optional Parameters:
        dataset (str): Dataset name (e.g. 'OCCCI', 'ACSPO'). Defaults to a product's default dataset.
    """
    def verbose_trace(msg):
        if verbose: print(f"DEBUG [FILE - {prod}]: {msg}")
    

    # 1. Get the directory path (Passing all kwargs down)
    path = get_filepath(prod, dataset=dataset, period=period, verbose=verbose, **kwargs)
    
    if (not path or not os.path.isdir(path)) and kwargs.get('map_subset'):
        verbose_trace(f"Subset directory not found. Falling back to base grid...")
        
        # Remove map_subset and try again
        fallback_kwargs = kwargs.copy()
        fallback_kwargs.pop('map_subset')
        path = get_filepath(prod, dataset=dataset, period=period, verbose=verbose, **fallback_kwargs)

    # 2. Build the search pattern
    if "/SOURCE/" in path:
        search_pattern = "*.nc"
        verbose_trace("SOURCE folder detected. Dropping period prefix for file search.")
    else:
        search_pattern = f"{period}_*.nc"
    verbose_trace(f"Searching for '{search_pattern}' in: {path}")

    # 3. Glob the files
    nc_files = glob.glob(os.path.join(path, search_pattern))
    if not nc_files:
        verbose_trace("⚠ No .nc files found.")
        return []

    # 4. Subset by date range if provided
    daterange = kwargs.get('daterange')
    if daterange:
        std_daterange = get_daterange(daterange)
        if std_daterange:
            start_str = str(daterange[0]).replace("-", "")
            end_str = str(daterange[1]).replace("-", "")
            
            file_dates = get_source_file_dates(nc_files, format="yyyymmdd")
            
            filtered_files = []
            for i, f_date in enumerate(file_dates):
                if f_date is not None:
                    if start_str <= f_date <= end_str:
                        filtered_files.append(nc_files[i])
                else:
                    filtered_files.append(nc_files[i])
            nc_files = filtered_files
            
    verbose_trace(f"📦 Found {len(nc_files)} files.")
    nc_files.sort()
    return nc_files