import os
from pathlib import Path
import stat
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

"""
Purpose:
    FILE_UTILITIES is a collection of utility functions for handling file paths, directories, and permissions.

Main Functions:
    - get_file_dates: Extracts dates from a list of filenames
    - file_make: Checks to see if a file exists and if it needs to be remade based on the mtimes of the input files
    - set_file_permissions: Checks the permissions of a file and changes them if they don't match the desired permissions.

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
    Sep 25, 2025 - KJWH: Added get_file_dates
"""
import os
from pathlib import Path

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

