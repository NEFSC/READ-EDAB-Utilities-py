import re
import calendar
import datetime
from datetime import datetime, timedelta, date
from pathlib import Path
from dateutil.relativedelta import relativedelta
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)

"""
Purpose:
    PERIOD_UTILITIES is a collection of utility functions for handling "period code" specific tasks.

Main Functions:
    - period_regex_debug: Period debug regex matching for a given filename against all known period codes.
    - period_info: Returns a dictionary mapping period codes to metadata for statistical processing.
    - get_period_info: Returns a dictionary of named fields for a given period code from the period_info dictionary
    - get_period_dates: Extracts the period encoded in a filename (e.g. 'M_202204' or 'DD3_20220401_20220403') and returns the start and end dates of that period.
    - get_period_sets: Generate period tokens (e.g. M_YYYYMM, W_YYYYWW) for a given date range or list of dates.
    

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
    Sep 25, 2025 - KJWH: Moved period specific functions from statanom_utilities to period_utitlities
    Sep 26, 2025 - KJWH: Created get_period_dates
    Sep 29, 2025 - KJWH: Finalized get_period_dates function
                         Created get_period_sets function and period_regex_debug

"""

def period_info():
    """
    Returns a dictionary mapping period codes to metadata for statistical processing.
    Each entry contains:
        0) Input time to xarray.groupby for stat calculations
        1) Input period code (to stats) to create the period
        2) Is it a range period?
        3) Is it a running mean period?
        4) Is it a climatological period?
        5) ISO 8601 duration
        6) GroupBy hint (xarray expression or None)
        7) File period format
        8) Description
    """
    return {
        'D':      ('time',  'D',  False, False, False, 'P1D', 'time', 'D_YYYYMMDD', 'Daily'),
        'DD':     ('time',  'D',  True,  False, False, 'PnD', 'time', 'DD_YYYYMMDD_YYYYMMDD', 'Range of daily data'),
        'DOY':    ('doy',   'D',  False, False, True,  '',    'time.dt.dayofyear', 'DOY_YYYY_YYYY', 'Climatological day of year'),
        'D3':     ('day3',  'D',  False, True,  False, 'P3D',  None, 'D3_YYYYMMDD_YYYYMMDD', 'Running 3-day'),
        'DD3':    ('day3',  'D',  True,  True,  False, 'P3D',  None, 'DD3_YYYYMMDD_YYYYMMDD', 'Range of running 3-day data'),
        'D8':     ('day8',  'D',  False, True,  False, 'P8D',  None, 'D8_YYYYMMDD_YYYYMMDD', 'Running 8-day'),
        'DD8':    ('day8',  'D',  True,  True,  False, 'P8D',  None, 'DD8_YYYYMMDD_YYYYMMDD', 'Range of running 8-day data'),
        'W':      ('week',  'D',  False, False, False, 'P1W', 'time.dt.isocalendar().week', 'W_YYYYWW', 'Weekly'),
        'WW':     ('week',  'D',  True,  False, False, 'P1W', 'time.dt.isocalendar().week', 'WW_YYYYWW_YYYYWW', 'Range of weekly data'),
        'WEEK':   ('week',  'W',  False, False, True,  '',    'time.dt.isocalendar().week', 'WEEK_YYYY_YYYY', 'Climatological week'),
        'M':      ('month', 'D',  False, False, False, 'P1M', 'time.dt.month', 'M_YYYYMM', 'Monthly'),
        'MM':     ('month', 'D',  True,  False, False, 'P1M', 'time.dt.month', 'MM_YYYYMM_YYYYMM', 'Range of monthly data'),
        'M3':     ('month3','M',  False, True,  False, 'P3M',  None, 'M3_YYYYMM_YYYYMM', 'Running 3-month'),
        'MM3':    ('month3','M',  True,  True,  False, 'P3M',  None, 'MM3_YYYYMM_YYYYMM', 'Range of running 3-month'),
        'MONTH':  ('month', 'M',  False, False, True,  'P1M', 'time.dt.month', 'MONTH_YYYY_YYYY', 'Climatological month'),
        'SEA':    ('sea',   'M',  False, False, False, 'P3M', 'time.dt.month', 'SEA_YYYYMM_YYYYMM', 'Seasonal (JFM)'),
        'SEASON': ('season','SEA',False, False, True,  'P3M', 'time.dt.season', 'SEASON_YYYYMM_YYYYMM', 'Climatological season'),
        'A':      ('annual' ,'M', False, False, False, 'P1Y', 'time.dt.year', 'A_YYYY', 'Annual (monthly inputs)'),
        'AA':     ('annual' ,'M', True,  False, False, 'P1Y', 'time.dt.year', 'AA_YYYY_YYYY', 'Range of annual (monthly inputs)'),
        'ANNUAL': ('annual' ,'A', False, False, True,  'P1Y', 'time.dt.year', 'ANNUAL_YYYY_YYYY', 'Climatological annual'),
        'Y':      ('year',  'D',  False, False, False, 'P1Y', 'time.dt.year', 'Y_YYYY_YYYY', 'Yearly (daily inputs)'),
        'YY':     ('year',  'D',  True,  False, False, 'P1Y', 'time.dt.year', 'YY_YYYY_YYYY', 'Range of yearly (daily inputs)'),
        'YEAR':   ('year',  'Y',  False, False, True,  'PnY', 'time.dt.year', 'YEAR_YYYY_YYYY', 'Climatological year'),
    }

def period_regex_debug(filename, period_map=None):
    """
    Period debug regex matching for a given filename against all known period codes.

    Parameters
    ----------
    filename : str
        The filename to test.
    period_map : dict, optional
        Precomputed map from get_period_info(). If None, will call get_period_info().

    Returns
    -------
    list of tuple
        [(code, matched, match_obj or None)]
    """
    if period_map is None:
        period_map = get_period_info()

    base = Path(filename).name
    token = base.split("_", 1)[0]

    results = []
    for code, info in period_map.items():
        pattern = info['regex']
        m = pattern.match(base)
        results.append((code, bool(m), m))

    return results

def get_period_info(period_code=None):
    """
    Returns a dictionary of named fields for a given period code and a compiled regex pattern for a given period code.

    Parameters:
        period_code (str, optional): The key from the period_info() map (e.g. 'W', 'M3', 'SEASON')

    Returns:
        dict: 
        - If the period code is provided, returns a dictionary with named metadata fields and a compiled regex pattern.
        - If None, returns a metadata dictionary for all period codes
    
    Errors:
        Raises KeyError if not found.
    """
    def segment_length(seg):
        return (
            seg.count("Y") +
            seg.count("M") +
            seg.count("D") +
            seg.count("W")  # Weeks counted as days (7)
        )
    
    raw_map = period_info()

    def build_info(code, info):
        (
            groupby,
            input_period_code,
            is_range,
            is_running_mean,
            is_climatology,
            iso_duration,
            groupby_hint,
            file_format,
            description
        ) = info

        # Build regex pattern
        parts = file_format.split("_")[1:]
        rx_parts = []
        for i, seg in enumerate(parts):
            length = segment_length(seg)
            label = "start" if i == 0 else "end"
            rx_parts.append(rf"(?P<{label}>\d{{{length}}})")

        if is_range or is_climatology:
            pattern = re.compile(rf"^{code}_" + "_".join(rx_parts) + r"(?:[_\-\.]|$)")
        else:
            pattern = re.compile(rf"^{code}_" + rx_parts[0] + r"(?:[_\-\.]|$)")

        return {
            'groupby': groupby,
            'input_period_code': input_period_code,
            'is_range': is_range,
            'is_running_mean': is_running_mean,
            'is_climatology': is_climatology,
            'iso_duration': iso_duration,
            'groupby_hint': groupby_hint,
            'file_format': file_format,
            'description': description,
            'regex': pattern
        }

    if period_code:
        if period_code not in raw_map:
            raise KeyError(f"Period code '{period_code}' not found in period_info()")
        return build_info(period_code, raw_map[period_code])
    else:
        return {code: build_info(code, info) for code, info in raw_map.items()}




def get_period_dates(files,format="%Y%m%d",placeholder="NA"):
    """
    Extracts the period encoded in a filename (e.g. 'M_202204' or 'DD3_20220401_20220403') and returns the start and end dates of that period.

    The function uses the period_info() table and regex patterns, then:
      - For range codes (e.g. DD, DD3, MM3): reads both start/end dates directly
      - For single‐period codes (D, M, A, etc.): computes end date via calendar or ISO-week logic
      - Weekly codes (W, WW) use ISO calendar weeks

    Parameters
        filename (str): Name or path of the file containing a period segment (e.g. 'M_202204.nc' or 'WW_202201_202203.nc'.
        format (str): Python date‐format to return (default "%Y%m%d") (optional)

    Returns
        (start_str, end_str) - (tuple of str): Start and end dates of the period, formatted per default_format.

    Raises
        ValueError
            If no known period pattern is found in `filename`.
    """
    def parse_date_segment(s, is_end=False):
        """Helper to parse YYYYMMDD / YYYYMM / YYYY into year, month, day"""
        if len(s) == 8:
            return int(s[:4]), int(s[4:6]), int(s[6:8])
        elif len(s) == 6:
            y, m = int(s[:4]), int(s[4:6])
            d = calendar.monthrange(y, m)[1] if is_end else 1
            return y, m, d
        elif len(s) == 4:
            return int(s), 12 if is_end else 1, 31 if is_end else 1
        else:
            raise ValueError(f"Unrecognized date segment: {s}")

    # 1. Cache period metadata and precompile regexes
    period_map = get_period_info()

    # 2. Process each file
    results = []
    for fname in files:
        base = Path(fname).name
        token = base.split("_", 1)[0]

        info = period_map.get(token)
        if not info:
            results.append((placeholder, placeholder))
            continue

        pattern = info['regex']
        m = pattern.match(base)
        if not m:
            results.append((placeholder, placeholder))
            continue
        
        gd = m.groupdict()
        if info["is_climatology"]: #  in ("DOY", "WEEK", "MONTH", "SEASON", "ANNUAL", "YEAR")
            # Use start year only for now
            sy = int(gd['start'])
            ey = int(gd.get('end', sy))
            start = date(sy, 1, 1)
            end = date(ey, 12, 31)
            results.append((start.strftime(format), end.strftime(format)))
            continue
        
        try:
            # Parse start
            s = gd['start']
            sy, sm, sd = parse_date_segment(s)
            start = date(sy, sm, sd)

            # Special handlers first
            if token == "W":
                s = gd['start']
                year = int(s[:4])
                week = int(s[4:6])
                start = date.fromisocalendar(year, week, 1)
                end   = date.fromisocalendar(year, week, 7)
                results.append((start.strftime(format), end.strftime(format)))
                continue

            elif token == "WW":
                s = gd['start']
                e = gd['end']
                try:
                    sy, sw = int(s[:4]), int(s[4:6])
                    ey, ew = int(e[:4]), int(e[4:6])
                    start = date.fromisocalendar(sy, sw, 1)
                    end   = date.fromisocalendar(ey, ew, 7)
                    results.append((start.strftime(format), end.strftime(format)))
                except ValueError as err:
                    print(f"WW parsing failed: {err}")
                    results.append((placeholder, placeholder))
                continue

            # Then handle generic calendar-style ranges
            if 'end' in gd and gd['end']:
                e = gd['end']
                ey, em, ed = parse_date_segment(e, is_end=True)
                end = date(ey, em, ed)
            else:
                # Fallback logic... compute end from input_code
                if token == 'D':
                    end = start
                elif token == 'M':
                    last = calendar.monthrange(sy, sm)[1]
                    end = date(sy, sm, last)
                elif token in ('A', 'Y'):
                    year = int(gd['start'])
                    start = date(year, 1, 1)
                    end = date(year, 12, 31)                                
                else:
                    end = start

            results.append((start.strftime(format), end.strftime(format)))
        except Exception:
            results.append((placeholder, placeholder))

    return results


def get_period_sets(code, start=None, end=None, dates=None, date_format="%Y%m%d", running=True):
    """
    Generate period tokens (e.g. M_YYYYMM, W_YYYYWW) for a given date range or list of dates.

    Parameters
        code (str): Period code (e.g. 'M', 'W', 'A', 'Y').
        start (str or datetime, optional): Start date in YYYYMMDD format or datetime object.
        end (str or datetime, optional): End date in YYYYMMDD format or datetime object.
        dates (list of str or datetime, optional): List of dates to infer range from.
        date_format (str, optional): Format of input strings. Default is '%Y%m%d'.
        running (bool, optional): If True, use sliding windows. If False, use fixed-step blocks.

    Returns
        list of str: List of period tokens like ['M_202005', 'M_202006', ...]
    """
   
    if dates:
        parsed = [
            d if isinstance(d, datetime) else datetime.strptime(d, date_format)
            for d in dates
        ]
        start_dt = min(parsed)
        end_dt = max(parsed)
    elif start and end:
        start_dt = start if isinstance(start, datetime) else datetime.strptime(start, date_format)
        end_dt = end if isinstance(end, datetime) else datetime.strptime(end, date_format)
    else:
        raise ValueError("Provide either start and end, or a list of dates.")

    if start_dt > end_dt:
        raise ValueError("Start date must be before or equal to end date.")

    current = start_dt
    periods = []

    if code in ('D3', 'D8'):
        step = int(code[1:])
        delta = timedelta(days=step)
        print(f"End date is {end_dt}")
        if running:
            current = start_dt
            while current + delta - timedelta(days=1) <= end_dt:
                s = current.strftime(date_format)
                e = (current + delta - timedelta(days=1)).strftime(date_format)
                periods.append(f"{code}_{s}_{e}")
                current += timedelta(days=1)
        else:
            current = start_dt
            while current <= end_dt:
                s = current.strftime(date_format)
                e = current + delta - timedelta(days=1)
                if e <= end_dt:
                    periods.append(f"{code}_{s}_{e.strftime(date_format)}")
                current += delta
        return periods

    if code == "M3":
        if running:
            current = start_dt
            while True:
                end_block = current + relativedelta(months=3) - timedelta(days=1)
                if end_block > end_dt:
                    break
                s = current.strftime(date_format)
                e = end_block.strftime(date_format)
                periods.append(f"{code}_{s}_{e}")
                current += timedelta(days=1)
        else:
            current = start_dt.replace(day=1)
            while current <= end_dt:
                end_block = current + relativedelta(months=3) - timedelta(days=1)
                if end_block <= end_dt:
                    s = current.strftime(date_format)
                    e = end_block.strftime(date_format)
                    periods.append(f"{code}_{s}_{e}")
                current += relativedelta(months=3)
        return periods
    
    if code == "SEA":
        current = datetime(start_dt.year, 1, 1)  # anchor to January
        while current <= end_dt:
            end_block = current + relativedelta(months=3) - timedelta(days=1)
            if end_block <= end_dt:
                s = current.strftime(date_format)
                e = end_block.strftime(date_format)
                periods.append(f"{code}_{s}_{e}")
            current += relativedelta(months=3)
        return periods

    while current <= end_dt:
        if code == "W":
            iso_year, iso_week, _ = current.isocalendar()
            period = f"{code}_{iso_year:04d}{iso_week:02d}"
            current += timedelta(days=7 - current.weekday())  # jump to next Monday
        
        elif code == "M":
            period = f"{code}_{current.strftime('%Y%m')}"
            current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        
        elif code == "A":
            period = f"{code}_{current.year}"
            current = datetime(current.year + 1, 1, 1)
        
        elif code == "Y":
            period = f"{code}_{current.year}"
            current = datetime(current.year + 1, 1, 1)
        
        else:
            raise ValueError(f"Unsupported period code: {code}")
        periods.append(period)

    return sorted(periods)

    