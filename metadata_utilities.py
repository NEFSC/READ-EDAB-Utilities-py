from utilities import get_python_dir
from utilities import dataset_defaults
import pandas as pd
import xarray as xr
import numpy as np
import os
import math
from typing import Optional, Dict, Any
from datetime import datetime

ISO_RESOLUTION_MAP = {
                "daily": "P1D",
                "weekly": "P1W",
                "monthly": "P1M",
                "seasonal": "P3M",
                "annual": "P1Y"
            }

def get_current_utc_timestamp() -> str:
    """ Returns current UTC time in ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ' """
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

def rename_source_prefix(metadata: dict, old_prefix: str = "source_", new_prefix: str = "source_sst_") -> dict:
    """
    Renames all keys in the metadata dictionary that start with `old_prefix` to use `new_prefix`.
    Preserves all other keys and values.
    """
    renamed = {}
    for key, value in metadata.items():
        if key.startswith(old_prefix):
            new_key = new_prefix + key[len(old_prefix):]
            renamed[new_key] = value
        else:
            renamed[key] = value
    return renamed

def format_iso_duration(days: float) -> str:
    """Convert float days to ISO 8601 duration string."""
    whole_days = int(np.floor(days))
    fractional_day = days - whole_days
    seconds = int(round(fractional_day * 86400))  # 86400 seconds in a day

    if whole_days == 0 and seconds == 0:
        return "P1D"  # fallback minimum duration

    duration = "P"
    if whole_days > 0:
        duration += f"{whole_days}D"
    if seconds > 0:
        duration += f"T{seconds}S"
    return duration

def get_metadata_table(metapath=None,sheet=None) -> dict:
    """
    Reads an Excel file with multiple sheets containing metadata mappings.
    If `sheet` is specified, returns {sheet_name: {attribute_name: value, ...}} for that sheet.
    If `sheet` is None, returns all sheets as a dictionary. 
    Raises an error if the requested sheet is not found.  
    """

    if metapath is None:
        dir = get_python_dir(resources=True)
        metapath = os.path.join(dir,'metadata','EDAB_metadata.xlsx')

    xls = pd.ExcelFile(metapath)
    sheet_map = {s.lower(): s for s in xls.sheet_names}

    def parse_sheet(sheet_name):
        df = xls.parse(sheet_name)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        if df.empty or df.dropna(how="all").empty:
            return {}

        if {"attribute", "required", "default"}.issubset(df.columns):
            return {
                str(row["attribute"]).strip(): {
                    "required": bool(row.get("required", False)),
                    "default": str(row.get("default", "")).strip() if pd.notna(row.get("default")) else None
                }
                for _, row in df.iterrows()
                if pd.notna(row.get("attribute"))
            }
        else:
            first_col = df.columns[0]
            # Return all rows as index-keyed dict
            return {
                i: {col: row[col] for col in df.columns}
                for i, row in df.iterrows()
                if pd.notna(row.get(first_col))
            }

    if sheet:
        if sheet not in xls.sheet_names:
            available = ", ".join(xls.sheet_names)
            raise ValueError(f"Sheet '{sheet}' not found. Available sheets: {available}")
        return parse_sheet(sheet)

    return {sheet_name: parse_sheet(sheet_name) for sheet_name in xls.sheet_names}


def load_all_metadata(metadata_path=None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Loads all metadata attributes and lookup tables from an Excel file.
    Returns a nested dictionary:
    {
        sheet_name: {
            attribute_name: {
                "required": bool,
                "default": str or None
            }
            # OR for LUTs:
            attribute_name: str, int, float, list, etc.
        }
    }
    Example Usage:
        metadata_flags = load_all_metadata()
        required_flags = metadata_flags.get("TimeCoverage", {}) # Access required flags for TimeCoverage
        program_codes = metadata_flags.get("LUT_Programs", {}).get("Programs", []) # Access lookup values for LUT_Region
    """
    all_sheets = get_metadata_table(metapath=metadata_path)  # {sheet_name: {attribute_name: value_dict}}
    metadata = {}

    for sheet_name, sheet_dict in all_sheets.items():
        rows = []
        for attr, attr_dict in sheet_dict.items():
            if isinstance(attr_dict, dict):
                rows.append({
                    "attribute": attr,
                    "required": bool(attr_dict.get("required", False)),
                    "default": attr_dict.get("default", None)
                })
            elif isinstance(attr_dict, (str, int, float)):
                rows.append({
                    "attribute": attr,
                    "required": False,
                    "default": attr_dict
                })
            else:
                # Unexpected type — preserve raw value
                rows.append({
                    "attribute": attr,
                    "required": False,
                    "default": None
                })

        df = pd.DataFrame(rows)
        df.columns = [str(col).lower().strip() for col in df.columns]

        # If it's a LUT sheet, preserve full row structure
        if sheet_name.lower().startswith("lut_"):
            metadata[sheet_name] = {
                row["attribute"]: {
                    k: v for k, v in row.items() if k != "attribute"
                }
                for _, row in df.iterrows()
                if pd.notna(row.get("attribute"))
            }
        else:
            metadata[sheet_name] = {
                str(row["attribute"]).strip(): {
                    "required": bool(row.get("required", False)),
                    "default": str(row.get("default", "")).strip() if pd.notna(row.get("default")) else None
                }
                for _, row in df.iterrows()
                if pd.notna(row.get("attribute"))
            }

    return metadata

def get_default_metadata(metadata_path: Optional[str] = None,sheet: str = "Global") -> Dict[str, Any]:
    """
    Returns a dictionary of default values for all attributes marked as required in the given sheet.
    """
    metadata_definitions = load_all_metadata(metadata_path)

    if sheet not in metadata_definitions:
        available = ", ".join(metadata_definitions.keys())
        raise KeyError(f"❌ Sheet '{sheet}' not found. Available sheets: {available}")

    sheet_meta = metadata_definitions[sheet]
    required_defaults = {
        attr: meta.get("default")
        for attr, meta in sheet_meta.items()
        if meta.get("required") is True and meta.get("default") not in [None, "", False]
    }

    return required_defaults

def extract_metadata_from_attrs(attrs: dict, keys: list, round_digits: int = 5) -> dict:

    metadata = {}
    for key in keys:
        value = attrs.get(key)
        if value is None:
            continue
        try:
            if isinstance(value, (float, int, np.number)):
                metadata[key] = round(float(value), round_digits)
            elif isinstance(value, str):
                parsed = float(value)
                metadata[key] = round(parsed, round_digits)
            else:
                metadata[key] = value
        except (ValueError, TypeError):
            metadata[key] = value
    return metadata

def get_lut_metadata(metadata: dict=None,
                                 metapath: str = None,
                                 add_program: str = None,
                                 add_project: str = None,
                                 add_contributor: str = None) -> dict:
    """
    Dynamically injects all available program/project metadata fields using lookup tables.
    Prepends 'program_' or 'project_' to each attribute name.
    Uses get_metadata_table() to load LUT_Programs and LUT_Projects sheets.
    Raises ValueError if the requested program/project is not found.
    Only fills fields that are missing in the input metadata.
    """
    if metadata is None:
        metadata = {}

    # Load lookup tables
    all_sheets = get_metadata_table(metapath=metapath)
    try:
        lut_programs = all_sheets["LUT_Programs"]
        lut_projects = all_sheets["LUT_Projects"]
        lut_contributors = all_sheets["LUT_Contributors"]
    except KeyError as e:
        raise ValueError(f"Missing required sheet: {e}")

    def normalize_lut(lut):
        # Normalize keys and values
        return {
            str(name).strip(): {
                "email": str(info.get("email")).strip() if pd.notna(info.get("email")) else None,
                "url": str(info.get("url")).strip() if pd.notna(info.get("url")) else None
            }
            for name, info in lut.items()
        }

    programs = normalize_lut(lut_programs)
    projects = normalize_lut(lut_projects)
    contributors = normalize_lut(lut_contributors)

    # Inject program metadata
    if add_program:
        if add_program not in programs:
            available = ", ".join(programs.keys())
            raise ValueError(f"Program '{add_program}' not found in LUT_Programs. Available: {available}")
        for key, value in programs[add_program].items():
            meta_key = f"program_{key}"
            if value and meta_key not in metadata:
                metadata[meta_key] = value
        metadata.setdefault("program_name", add_program)

    # Inject project metadata
    if add_project:
        if add_project not in projects:
            available = ", ".join(projects.keys())
            raise ValueError(f"Project '{add_project}' not found in LUT_Projects. Available: {available}")
        for key, value in projects[add_project].items():
            meta_key = f"project_{key}"
            if value and meta_key not in metadata:
                metadata[meta_key] = value
        metadata.setdefault("project_name", add_project)
    
    if add_contributor:
        if add_contributor not in contributors:
            available = ", ".join(contributors.keys())
            raise ValueError(f"Contributor '{add_contributor}' not found in LUT_Projects. Available: {available}")
        for key, value in contributors[add_contributor].items():
            meta_key = f"contributor_{key}"
            if value and meta_key not in metadata:
                metadata[meta_key] = value
        metadata.setdefault("contributor_name", add_project)

    return metadata

def get_geospatial_metadata(
    use_inputdata_path: Optional[str] = None,
    use_current_data: Optional[xr.Dataset] = None,
    default_depth: float = 0.0,
    round_digits: int = 5
    ) -> Dict[str, Any]:
    """
    Generate geospatial metadata either by copying from an existing NetCDF file
    or deriving from the current xarray Dataset.
    Returns a dictionary of metadata attributes.
    If vertical coordinate is missing, assumes surface data at default_depth.
    """
    metadata = {}

    if use_inputdata_path:
        with xr.open_dataset(use_inputdata_path) as ds:
            geospatial_keys = [k for k in ds.attrs if k.startswith("geospatial_")]
            metadata = extract_metadata_from_attrs(ds.attrs, geospatial_keys, round_digits)
        return metadata

    elif use_current_data:
        ds = use_current_data
        lat_name = next((v for v in ds.coords if 'lat' in v.lower()), None)
        lon_name = next((v for v in ds.coords if 'lon' in v.lower()), None)
        vert_name = next((v for v in ds.coords if 'lev' in v.lower() or 'z' in v.lower()), None)

        if not lat_name or not lon_name:
            raise ValueError("Could not infer coordinate names for latitude and longitude.")

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        metadata.update({
            'geospatial_lat_min': round(float(np.min(lat)), round_digits),
            'geospatial_lat_max': round(float(np.max(lat)), round_digits),
            'geospatial_lat_resolution': round(float(np.mean(np.diff(np.sort(lat)))), round_digits),
            'geospatial_lat_units': ds[lat_name].attrs.get('units', 'decimal degrees north'),

            'geospatial_lon_min': round(float(np.min(lon)), round_digits),
            'geospatial_lon_max': round(float(np.max(lon)), round_digits),
            'geospatial_lon_resolution': round(float(np.mean(np.diff(np.sort(lon)))), round_digits),
            'geospatial_lon_units': ds[lon_name].attrs.get('units', 'decimal degrees east'),
        })

        if vert_name:
            vert = ds[vert_name].values
            metadata.update({
                'geospatial_vertical_min': round(float(np.min(vert)), round_digits),
                'geospatial_vertical_max': round(float(np.max(vert)), round_digits),
                'geospatial_vertical_units': ds[vert_name].attrs.get('units', 'meters'),
                'geospatial_vertical_positive': ds[vert_name].attrs.get('positive', 'down'),
            })
        else:
            metadata.update({
                'geospatial_vertical_min': round(default_depth, round_digits),
                'geospatial_vertical_max': round(default_depth, round_digits),
                'geospatial_vertical_units': 'meters',
                'geospatial_vertical_positive': 'down',
            })

        return metadata

    else:
        raise ValueError("Either use_inputdata_path or use_current_data must be provided.")

def get_temporal_metadata(
    ds: Optional[xr.Dataset] = None,
    include_dates: bool = True,
    include_time_coverage: bool = True,
    resolution_mode: str = "auto"  # options: "auto", "daily", "weekly", "monthly", "seasonal", "annual"
) -> Dict[str, Any]:
    """
    Generates temporal metadata including date stamps and time coverage.
    """
    metadata = {}

    if include_dates:
        metadata.update({
            "date_created": get_current_utc_timestamp(),
            # You can optionally add placeholders for others
            "date_issued": None,
            "date_metadata_modified": None,
            "date_modified": None
        })

    if include_time_coverage and ds is not None:
        time_name = next((v for v in ds.coords if 'time' in v.lower()), None)
        if not time_name:
            raise ValueError("Time coordinate not found in dataset.")

        time_vals = ds[time_name].values
        if len(time_vals) == 0:
            raise ValueError("Time coordinate is empty.")        
        start = np.min(time_vals)
        end = np.max(time_vals)
        duration = (np.datetime64(end, 'D') - np.datetime64(start, 'D')).astype(int)
        if duration < 1:
            duration = 1

        # Determine resolution
        if resolution_mode == "auto":
            diffs = np.diff(np.sort(time_vals))
            if len(diffs) == 0 or np.isnan(np.mean(diffs)):
                resolution_days = 1  # fallback
            else:
                resolution_days = np.mean(diffs) / np.timedelta64(1, 'D')
        else:
            
            
            metadata["time_coverage_resolution"] = iso_resolution_map.get(resolution_mode.lower(), "P1D")
            resolution_days = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30,
                "seasonal": 90,
                "annual": 365
            }.get(resolution_mode.lower())

            if resolution_days is None:
                raise ValueError(f"Unsupported resolution_mode: {resolution_mode}")
            resolution_iso = ISO_RESOLUTION_MAP.get(resolution_mode.lower(), format_iso_duration(resolution_days))

        metadata.update({
            "time_coverage_start": str(np.datetime64(start, 'D')),
            "time_coverage_end": str(np.datetime64(end, 'D')),
            "time_coverage_duration": format_iso_duration(duration_days),
            "time_coverage_resolution": resolution_iso
        })

    return metadata

from typing import Dict, List, Tuple


def get_reference_metadata(models: list, metapath: str = None) -> dict:
    """
    Retrieves reference metadata for onen or more input model from LUT_References.
    Dynamically uses column headers as keys.
    """
    def normalize_lut(lut_references):
        """
        Converts LUT_References into a dict keyed by model name.
        Handles both DataFrame and dict-of-dict formats.
        """
        normalized = {}

        if isinstance(lut_references, pd.DataFrame):
            for _, row in lut_references.iterrows():
                model = str(row.get("model", "")).strip()
                if model:
                    normalized[model] = row.to_dict()
        elif isinstance(lut_references, dict):
            for entry in lut_references.values():
                model = str(entry.get("model", "")).strip()
                if model:
                    normalized[model] = entry
        else:
            raise TypeError("LUT_References must be a DataFrame or dict of dicts")

        return normalized
    
    lut_refs = get_metadata_table(metapath=metapath, sheet="LUT_References")

    if not isinstance(models, list):
        models = [models]

    # Normalize keys for lookup
    normalized_refs = normalize_lut(lut_refs)

    results = []
    for model in models:
        model_key = str(model).strip()
        if model_key not in normalized_refs:
            available = ", ".join(sorted(normalized_refs.keys()))
            raise ValueError(f"Model '{model_key}' not found in LUT_References. Available: {available}")

        entry = normalized_refs[model_key]
        clean_entry = {
            k: str(v).strip() if isinstance(v, str) else v
            for k, v in entry.items()
            if pd.notna(v)
        }
        results.append(clean_entry)

    return results

def get_source_metadata(dataset: str, dataset_version: str = None, source_prefix: str = "source", metapath: str = None) -> dict:
    """
    Retrieves metadata for a given dataset and version from LUT_Datasets.
    - Uses nested dict structure: {dataset: {version: metadata_dict}}
    - Applies default version if none provided.
    - Supports legacy version aliasing.
    - Returns keys prefixed with 'source_'.
    """
    def is_nan(val):
        return isinstance(val, float) and math.isnan(val)


    # Load and reshape
    lut_raw = get_metadata_table(metapath=metapath, sheet="LUT_Datasets")
    lut_nested = {}
    for row in lut_raw.values():
        ds = str(row.get("dataset", "")).strip().upper()
        if not ds:
            continue

        # Get the version and treat NaN or empty string as missing
        ver_raw = row.get("version", "")
        ver_clean = str(ver_raw).strip().upper() if ver_raw is not None else ""
        ver_key = ver_clean if ver_clean and not (isinstance(ver_raw, float) and math.isnan(ver_raw)) else "__NOVERSION__"

        if ver_key in lut_nested.get(ds, {}):
            raise ValueError(f"Duplicate entry for dataset '{ds}', version '{ver_key}' in LUT_Datasets")

        metadata = {k: v for k, v in row.items() if k not in ["dataset", "version"]}
        lut_nested.setdefault(ds, {})[ver_key] = metadata

    dataset_key = dataset.strip()
    dataset_upper = dataset_key.upper()

    if dataset_key not in lut_nested:
        raise ValueError(f"Dataset '{dataset}' not found. Available: {', '.join(sorted(lut_nested.keys()))}")

    version_dict = lut_nested[dataset_key]

    # Use default version if none provided
    if not dataset_version:
        version_keys = list(version_dict.keys())

        # If only one version exists, use it
        if len(version_keys) == 1:
            dataset_version = version_keys[0]

        # If multiple versions exist, try dataset_defaults
        else:
            default_entry = dataset_defaults().get(dataset_upper)
            if default_entry:
                dataset_version = default_entry[0]
            else:
                raise ValueError(
                    f"Multiple versions found for dataset '{dataset}' but no default specified. "
                    f"Available: {', '.join(sorted(version_keys))}"
                )
    # Return requested version if available
    if dataset_version in version_dict:
        metadata = version_dict[dataset_version]
        # Filter out NaN values and prefix keys
        filtered = {
            f"{source_prefix}_{k}": v
            for k, v in metadata.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        # Include the version explicitly
        filtered["source_version"] = dataset_version
        return filtered


    # Raise error if version not found
    available_versions = sorted(version_dict.keys())
    raise ValueError(f"Version '{dataset_version}' not found for dataset '{dataset}'. Available: {', '.join(available_versions)}")

def get_lut_products(metapath: str = None) -> dict:
    """
    Parses LUT_Products sheet into a dict keyed by short_name.
    Returns: {short_name: {attribute: value, ...}}
    """
    lut_raw = get_metadata_table(metapath=metapath, sheet="LUT_Products")
    lut_clean = {}

    for row in lut_raw.values():
        short_name = str(row.get("short_name", "")).strip().upper()
        if not short_name:
            continue

        # Drop NaNs and normalize keys
        metadata = {
            str(k).strip(): v
            for k, v in row.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }

        lut_clean[short_name] = metadata

    return lut_clean

def build_product_attributes(product: str, metapath: str = None,  _FillValue=None) -> dict:
    """
    Builds a metadata dictionary for a given product short_name.
    - Uses 'Products' sheet to determine required attributes
    - Pulls values from 'LUT_Products' sheet
    - Includes optional attributes if present
    - Allows _FillValue override
    - Raises message for missing required attributes with note to add manually
    """
    product_meta = get_metadata_table(metapath=metapath, sheet="Products")
    lut_products = get_lut_products(metapath)

    product_key = product.strip().upper()
    product_row = None

    # Find matching product in LUT_Products
    for row in lut_products.values():
        short_name = str(row.get("short_name", "")).strip().upper()
        if short_name == product_key:
            product_row = row
            break

    if not product_row:
        raise ValueError(f"Product '{product}' not found in LUT_Products")

    attributes = {}
    missing_required = []

    print(product_meta.items())
    for attr, meta in product_meta.items():
        attr_name = str(attr).strip()
        is_required = str(meta.get("required", "")).strip().upper() == "TRUE"

        # Special handling for _FillValue override
        if attr_name == "_FillValue" and _FillValue is not None:
            attributes["_FillValue"] = _FillValue
        elif attr_name in product_row and pd.notna(product_row[attr_name]):
            attributes[attr_name] = product_row[attr_name]
        elif is_required:
            missing_required.append(attr_name)

    if missing_required:
        for attr in missing_required:
            print(f"⚠️ Required attribute '{attr}' is missing for product '{product}' and should be added manually to LUT_Products.")

    return attributes

