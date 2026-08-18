from importlib.resources import path
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)
from utilities import dataset_defaults, get_dataset_products, parse_dataset_info, resolve_dataset_grid, get_period_info, get_source_file_dates, get_daterange

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
        'CHLOR_A': ('CHLOR_A','OCCCI','PRODUCTS'),
        'SST': ('SST','ACSPO', 'SOURCE'),
        'PPD': ('PPD','OCCCI', 'PRODUCTS'),
        'PSC': ('PSC','OCCCI', 'PRODUCTS'),
        'RRS': ('RRS','OCCCI', 'SOURCE'),
        'PAR': ('PAR','GLOBCOLOUR','SOURCE'),
        'IPAR': ('IPAR','PACE','SOURCE'),
        'AVW': ('AVW','OCCCI','PRODUCTS'),
        'KD': ('KD','OCCCI','SOURCE'),
        'IOP': ('IOP','OCCCI','SOURCE'),
        'MOANA': ('MOANA','PACE','SOURCE'),
        'CARBON': ('CARBON','PACE','SOURCE'),
        'FLH': ('FLH','PACE','SOURCE'),
        'CHL_TEMP': ('CHL1','GLOBCOLOUR','SOURCE'),
        'PIC': ('PIC','GLOBCOLOUR','SOURCE'),
        'POC': ('POC','GLOBCOLOUR','SOURCE'),
        'SST_TEMP': ('SST','ACSPONRT', 'SOURCE'),
        'CHL_FRONTS': ('CHL_FRONTS','OCCCI', 'PRODUCTS'),
        'SST_FRONTS': ('SST_FRONTS','ACSPO', 'PRODUCTS'),
        'FRONTS': ('SST_FRONTS','ACSPO', 'PRODUCTS'),
        'BTEMP': ('BTEMP','GLORYS','SOURCE'),
        'DO': ('DO','MOM6','SOURCE'),
        'ARAG': ('ARAG','MOM6','SOURCE')
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
            'PIC': 'PIC_mean',
            'POC': 'POC_mean',
        },
        'OCCCI': {
            'CHL': 'chlor_a',
            'RRS_412': 'Rrs_412',
            'RRS_443': 'Rrs_443',
            'RRS_490': 'Rrs_490',
            'RRS_510': 'Rrs_510',
            'RRS_560': 'Rrs_560',
            'RRS_665': 'Rrs_665',
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
        },
        'MOM6': {
            'DO': 'btm_o2',
            'ARAG': 'aragonite_saturation'
        }
    }

#--------------------------------------------------------------------------------------
def get_nc_prod(dataset,product):
    """
    Returns internal variable name and metadata for a given dataset and product.

    Parameters:
        dataset (str): Dataset key (e.g., 'GLOBCOLOUR')
        product(str): Product key (e.g., 'CHL1')

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
                                                           
