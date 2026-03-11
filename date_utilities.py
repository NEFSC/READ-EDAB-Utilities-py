from datetime import datetime, date, timedelta
import numpy as np
from pathlib import Path
import re
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

"""
Purpose:
    DATE_UTILITIES is a collection of utility functions for handling file paths, directories, and permissions.

Main Functions:
    - format_date: 
    - get_dates: Create normalized date lists from different date inputs and output a list of dates in multiple formats
    - get_source_file_dates: Extract standardized date strings from source data filenames.
    - get_current_utc_timestamp: Returns current UTC time in ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ' 
    - format_iso_duration: Convert float days to ISO 8601 duration string.


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
    Sep 18, 2025 - KJWH: Updated documentation
"""


def format_date(dt, fmt):
    if fmt == "yyyymmdd":
        return dt.strftime("%Y%m%d")
    elif fmt == "yyyy":
        return dt.strftime("%Y")
    elif fmt == "yyyy-mm-dd":
        return dt.strftime("%Y-%m-%d")
    elif fmt == "yyyymmddhhmmss":
        return dt.strftime("%Y%m%d%H%M%S")
    elif fmt == "tuple":
        return (dt.year, dt.month, dt.day)
    elif fmt == "datetime":
        if isinstance(dt, datetime):
            return dt.date()
        elif isinstance(dt, date):
            return dt
        else:
            raise TypeError(f"Expected date or datetime, got {type(dt)}")
    try:
        return dt.strftime(fmt)
    except Exception:
        raise ValueError(f"Unknown format: {fmt}")
    
def get_dates(
        dates,
        output="days",   # "days", "doys", "weeks", "months", "seasons", "years"
        day_format='yyyymmdd',
        period_format="tuple", # "datetime", "bounds"
):
    """
    Generate normalized date outputs from a wide variety of input formats.

    This function accepts flexible date specifications (single dates, lists of dates,
    year ranges, or explicit start/end pairs) and returns a sequence of days, weeks,
    months, seasons, or years in a consistent, schema‑driven format.

    Parameters
    ----------
    dates : object or list
        Input date specification. Accepted forms include:
        • A single date string ("YYYYMMDD" or "YYYY‑MM‑DD")
        • A single datetime.date or datetime.datetime object
        • A single integer year (e.g., 2023)
        • A list of date strings
        • A list of datetime/date objects
        • A list of integer years: [2023] or [2023, 2025]
        • A two‑element list defining a range:
                ["20250101", "20250630"]
                [2025, 2026]
                [datetime(2025,1,1), datetime(2026,12,31)]

    output : str, optional
        Specifies the type of period to generate. Options:
        • "days"    → daily sequence (default)
        • "doys"    → day of year
        • "weeks"   → ISO week periods
        • "months"  → calendar months
        • "seasons" → climatological seasons (JFM, AMJ, JAS, OND)
        • "years"   → calendar years

    day_format : str, optional
        Controls how *daily* outputs are formatted. Only used when output="days".
        Options:
        • "yyyymmdd"       → default
        • "yyyy-mm-dd"
        • "yyyymmddhhmmss"
        • "datetime"       → return datetime objects

    period_format : str, optional
        Controls how *non‑day* periods (weeks, months, seasons, years) are represented.
        Options:
        • "tuple"    → canonical tuple representation (default)
                doy     → (year, day_of_year)
                weeks   → (year, week)
                months  → (year, month)
                seasons → (year, "JFM"/"AMJ"/"JAS"/"OND")
                years   → (year,)
        • "datetime" → return the start date of each period as a datetime
        • "bounds"   → return (start_date, end_date) as "YYYYMMDD" strings

    Returns
    -------
    list
        A list of formatted days or period representations, depending on the
        `output`, `day_format`, and `period_format` settings.

    Notes
    -----
    • When `dates` contains more than two items and output != "days", the function
    interprets the input as a collection of individual dates and uses the min/max
    to define the range.

    • Seasons follow fixed 3‑month climatological groupings:
        JFM = Jan–Mar
        AMJ = Apr–Jun
        JAS = Jul–Sep
        OND = Oct–Dec

    • All returned datetime objects are timezone‑naive and represent local calendar dates.
    """

    # -----------------------------
    # Helper functions: 
    # -----------------------------
    # Parse a single input
    def to_dt(d):
        if isinstance(d, int):
            return date(d, 1, 1)

        if isinstance(d, str):
            # Year-like string
            if d.isdigit() and len(d) == 4:
                return date(int(d), 1, 1)

            for fmt in ("%Y%m%d%H%M%S", "%Y%m%d", "%Y-%m-%d"):
                try:
                    return datetime.strptime(d, fmt)
                except ValueError:
                    pass
            raise ValueError(f"Could not parse string: {d}")

        if isinstance(d, (datetime, date)):
            return d

        raise TypeError(f"Unsupported date type: {type(d)}")

    def apply_period_format(start_dt, end_dt, tuple_value):
        if period_format == "tuple":
            return tuple_value
        if period_format == "datetime":
            return start_dt
        if period_format == "bounds":
            return (start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
        raise ValueError(f"Unknown period_format: {period_format}")

    # -----------------------------
    # Normalize input list
    # -----------------------------
    if dates is None:
        return None

    if not isinstance(dates, list):
        dates = [dates]

    # Coerce year-like strings to int
    dates = [int(d) if isinstance(d, str) and d.isdigit() and len(d) == 4 else d
             for d in dates]

    # -----------------------------
    # Determine start/end
    # -----------------------------
    if all(isinstance(d, int) for d in dates):
        # Year range
        start = date(dates[0], 1, 1)
        end   = date(dates[-1], 12, 31)

    elif len(dates) == 2:
        # Two endpoints
        start, end = sorted([to_dt(d) for d in dates])

    else:
        # List of individual dates
        parsed = [to_dt(d) for d in dates]
        if output == "days":
            # Return exactly the dates provided
            return [format_date(d, day_format) for d in parsed]
        
        # For weeks/months/seasons/years, treat as range
        start = min(parsed)
        end   = max(parsed)

    # Ensure datetime
    if isinstance(start, date) and not isinstance(start, datetime):
        start = datetime(start.year, start.month, start.day)
    if isinstance(end, date) and not isinstance(end, datetime):
        end = datetime(end.year, end.month, end.day)

    # -----------------------------
    # Output: DAYS
    # -----------------------------
    if output == "days":
        out = []
        cur = start
        while cur <= end:
            out.append(format_date(cur, day_format))
            cur += timedelta(days=1)
        return out

    # -----------------------------
    # Output: DOYS
    # -----------------------------
    if output == "doys":
        out = []
        cur = start
        while cur <= end:
            year = cur.year
            doy = cur.timetuple().tm_yday
            tup = (year, doy)
            # DOY uses period_format just like other non-day periods
            out.append(apply_period_format(cur, cur, tup))
            cur += timedelta(days=1)

        return out

    # -----------------------------
    # Output: WEEKS
    # -----------------------------
    if output == "weeks":
        # Snap to Monday
        cur = start - timedelta(days=start.isocalendar().weekday)
        end_week = end - timedelta(days=end.isocalendar().weekday)

        out = []
        while cur <= end_week:
            iso = cur.isocalendar()
            tup = (iso.year, iso.week)
            out.append(apply_period_format(cur, cur + timedelta(days=6), tup))
            cur += timedelta(days=7)
        return out

    # -----------------------------
    # Output: MONTHS
    # -----------------------------
    if output == "months":
        cur = start.replace(day=1)
        end_m = end.replace(day=1)

        out = []
        while cur <= end_m:
            tup = (cur.year, cur.month)
            # compute bounds
            if cur.month == 12:
                next_month = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                next_month = cur.replace(month=cur.month + 1, day=1)
            end_dt = next_month - timedelta(days=1)
            out.append(apply_period_format(cur, end_dt, tup))
            cur = next_month
        return out

    # -----------------------------
    # Output: SEASONS
    # -----------------------------
    if output == "seasons":

        def season_start_month(m):
            if m in (1,2,3): return 1
            if m in (4,5,6): return 4
            if m in (7,8,9): return 7
            return 10

        cur = start.replace(month=season_start_month(start.month), day=1)
        end_s = end.replace(month=season_start_month(end.month), day=1)

        out = []
        while cur <= end_s:
            m = cur.month
            if m == 1:  code = "JFM"
            elif m == 4: code = "AMJ"
            elif m == 7: code = "JAS"
            else:        code = "OND"

            # bounds
            if m == 10:
                next_season = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                next_season = cur.replace(month=m + 3, day=1)
            end_dt = next_season - timedelta(days=1)

            tup = (cur.year, code)
            out.append(apply_period_format(cur, end_dt, tup))

            cur = next_season
        return out

    # -----------------------------
    # Output: YEARS
    # -----------------------------
    if output == "years":
        out = []
        for y in range(start.year, end.year + 1):
            s = datetime(y, 1, 1)
            e = datetime(y, 12, 31)
            tup = (y,)
            out.append(apply_period_format(s, e, tup))
        return out

    # -----------------------------
    # Unknown output
    # -----------------------------
    raise ValueError(f"Unknown output type: {output}")
   

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
    """ 
      Returns current UTC time in ISO 8601 format: 'YYYY-MM-DDTHH:MM:SSZ' 
    """
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

def format_iso_duration(days: float) -> str:
    """
    Convert float days to ISO 8601 duration string.
    """
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