from datetime import datetime, date, timedelta
import numpy as np
from pathlib import Path
import re
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)


def format_date(dt, fmt):
    if fmt == "yyyymmdd":
        return dt.strftime("%Y%m%d")
    elif fmt == "yyyy":
        return dt.strftime("%Y")
    elif fmt == "yyyy-mm-dd":
        return dt.strftime("%Y-%m-%d")
    elif fmt == "yyyymmddhhmmss":
        return dt.strftime("%Y%m%d%H%M%S")
    elif fmt == "datetime":
        if isinstance(dt, datetime):
            return dt.date()
        elif isinstance(dt, date):
            return dt
        else:
            raise TypeError(f"Expected date or datetime, got {type(dt)}")
    else:
        raise ValueError(f"Unknown format: {fmt}")
    
def get_dates(dates,format='yyyymmdd'):
    """
    Create normalized date lists from different date inputs and output a list of dates in multiple formats

    Input can be:
        dates (list): Accepts list of dates, range, year(s), or datetime objects.
            - List of date strings (yyyymmdd or yyyy-mm-dd)
            - List of datetime.date objects
            - List of integers: [2023] or [2023, 2025]
            - List of one or two strings: ['20250101'] or ['20250101', '20250630'] or ['2025', '2026']
    
        format (str): Output format. Options:
            - 'yyyymmdd'       → default
            - 'yyyy-mm-dd'
            - 'yyyymmddhhmmss' → used if any input contains time info
            - 'datetime'       → returns datetime.date 
    Returns:
        list: Formatted date strings or date objects

    """
    def to_dt(d):
        if isinstance(d, int): return date(d, 1, 1)
        if isinstance(d, str):
            # Try to coerce year-like strings to int
            if d.isdigit() and len(d) == 4:
                return date(int(d), 1, 1)
            for fmt in ("%Y%m%d%H%M%S", "%Y%m%d", "%Y-%m-%d"):
                try:
                    return datetime.strptime(d, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Could not parse string: {d}")
        if isinstance(d, (datetime, date)): return d
        raise TypeError(f"Unsupported date type: {type(d)}")

    if dates is None:
        return None

    # Normalize input to list
    if not isinstance(dates, list):
        dates = [dates]

    # Coerce year-like strings to int
    dates = [int(d) if isinstance(d, str) and d.isdigit() and len(d) == 4 else d for d in dates]
    
    # Range of years
    if all(isinstance(d, int) for d in dates):
        start = date(dates[0], 1, 1)
        end = date(dates[-1], 12, 31)

    # Handle two dates (any type)
    elif len(dates) == 2 and all(isinstance(d, (str, int, date, datetime)) for d in dates):
        start, end = sorted([to_dt(d) for d in dates])
    
    # Treat as list of individual dates
    else:
        return [format_date(to_dt(d), format) for d in dates]

    # Build range
    step = timedelta(days=1)
    current = start
    out = []
    while current <= end:
        out.append(format_date(current, format))
        current += step
    return out



def get_source_file_dates(files, format="yyyymmdd", placeholder=None):
    """
    Extract standardized date strings from source data filenames.
    
    Args:
        files (list): List of filenames to extract dates from.
        format (str): Output date format. Options: 'yyyymmdd', 'yyyy-mm-dd', 'datetime', etc.
        placeholder: Value to return when a filename doesn't match any known pattern.
                    Default is None, but can be set to '' or 'NA' as needed.
    
    Returns:
        list: One formatted date per input filename (or placeholder if no match)
    """
    
    # Get dates from files
    def extract_raw_date(filename):
        
        # Define common patterns to match the date portion across different formats
        patterns = [
            r"^(\d{14})",                                   # e.g. ACSPO, MUR
            r"[-_](\d{8})(?=[-_\.])",                       # e.g. CORALSST, OCCCI, Globcolour
            r"\.mean\.(\d{4})\.nc$",                        # e.g. OISST
        ]

        basename = Path(filename).name
        for p in patterns:
            m = re.search(p, basename)
            if m:
                return m.group(1)
        return None
    
    def parse_date_string(s):
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d", "%Y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None
    
    out = []
    for f in files:
        raw = extract_raw_date(f)
        if not raw:
            out.append(placeholder)
            continue
        dt = parse_date_string(raw)
        if not dt:
            out.append(placeholder)
            continue
        out.append(format_date(dt, format))
    return out

def get_current_utc_timestamp() -> str:
    """ Returns current UTC time in ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ' """
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

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