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
    - get_token_lengths: Return canonical token length mapping for period parsing.
    - period_info: Returns a dictionary mapping period codes to metadata for statistical processing.
    - get_period_info: Returns a dictionary of named fields for a given period code from the period_info dictionary
    - check_period: Validates a single period string
    - get_period_dates: Extracts the period encoded in a filename (e.g. 'M_202204' or 'DD3_20220401_20220403') and returns the start and end dates of that period.
    - get_period_sets: Generate period tokens (e.g. M_YYYYMM, W_YYYYWW) for a given date range or list of dates.
    

Helper Functions:
    - build_regex (in period_info): Build a strict anchored regex based on file_format tokens.
    - build_info (in period_info): Build the full info dictionary for a given period code, including the compiled regex.
    - _fail (in check_period): Helper to return a standardized failure response.
    - seg_value_for (in get_period_dates): Map token -> segment value (1-based occurrence).
    - parse_yyyymmdd (in get_period_dates): Parses a date segment (YYYYMMDD) into a date object.
    - parse_yyyymm (in get_period_dates): Parses a date segment (YYYYMM) into start and end date objects.
    - token_indices (in get_period_dates): Convenience function to get token index list for a token.
    - fail (in get_period_dates): Helper to record failure and optionally diagnostics.

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
    Feb 24, 2026 - KJWH: Fixed D3 and D8 parsing logic and regex error
                         Added SEA and SEASON seasonal parsing logic and fixed regex error

"""

def is_leap_year(year):
    """Return True if year is a leap year."""
    year = int(year)
    return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))

# Canonical token length map used by build_regex(), check_period(), validators, etc.
def get_token_lengths():
    """Return canonical token length mapping for period parsing."""
    return {
        "YYYY": 4,
        "YYYYMM": 6,
        "YYYYMMDD": 8,
        "YYYYWW": 6,
        "WW": 2,
        "MM": 2,
        "DD": 2,
        "DDD": 3,
        "DOY": (3, 4, 4),
    }


def period_info(help=False):
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
        7) File period format (period code plus the date format)
        8) Date format (e.g. a single start date or start and end dates)
        9) Number of date segments
        10) Period description
        11) Span type
    """
    fields = {
        "0) groupby": "xarray groupby key used for statistical aggregation",
        "1) input_period_code": "base period used to compute this period",
        "2) is_range": "True if the token encodes a user-defined range period",
        "3) is_running_mean": "True if the period is a running window period",
        "4) is_climatology": "True if the period is a climatological period",
        "5) iso_duration": "ISO 8601 duration string",
        "6) groupby_hint": "xarray expression for grouping (or None)",
        "7) file_format": "filename pattern for this period (period code plus the date format)",
        "8) date_format": "description of date format in filename (e.g. YYYY, YYYYMM, YYYYMMDD_YYYYMMDD)",
        "9) n_date_segments": "number of date segments in the filename (e.g. 1 for YYYYMM, 2 for YYYYMMDD_YYYYMMDD)",
        "10) description": "human-readable description of the type of data associate with the period",
        "11) span_type": "categorization of the period span (e.g. 'Daily', 'Running 3-day', 'Seasonal block')",
    }

    if help:
        return fields
        
    return {
        'D':      ('time','D',False,False,False,'P1D','time','D_YYYYMMDD','YYYYMMDD',1,'Daily','day'),
        'DD':     ('time','D',True, False,False,'PnD','time','DD_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of daily data','day'),
        'DOY':    ('doy','D',False,False,True,'','time.dt.dayofyear','DOY_DDD_YYYY_YYYY','DOY',3,'Climatological day of year','climatology_doy'),
        'DOYS':   ('doy','D',False,False,True,'','time.dt.dayofyear','DOYS_YYYY_YYYY','YYYY',2,'Combined climatological days of year','climatology_doys'),

        'D3':     ('day3','D',False,True, False,'P3D',None,'D3_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Running 3-day','running'),
        'DD3':    ('day3','D',True, True, False,'P3D',None,'DD3_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of running 3-day data','range'),

        'D8':     ('day8','D',False,True, False,'P8D',None,'D8_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Running 8-day','running'),
        'DD8':    ('day8','D',True, True, False,'P8D',None,'DD8_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of running 8-day data','range'),

        'W':     ('week','D',False,False,False,'P1W','time.dt.isocalendar().week','W_YYYYWW','YYYYWW',1,'Weekly (ISO week format)','week'),
        'WW':    ('week','D',True, False,False,'P1W','time.dt.isocalendar().week','WW_YYYYWW_YYYYWW','YYYYWW',2,'Range of weekly data','range'),
        'WEEK':  ('week','W',False,False,True,'','time.dt.isocalendar().week','WEEK_WW_YYYY_YYYY','YYYY',3,'Climatological ISO week','climatology_week'),
        'WEEKS': ('week','W',False,False,True,'','time.dt.isocalendar().week','WEEKS_YYYY_YYYY','YYYY',2,'Combined climatological ISO week','climatology_week'),

        'M':      ('month','D',False,False,False,'P1M','time.dt.month','M_YYYYMM','YYYYMM',1,'Monthly','month'),
        'MM':     ('month','D',True, False,False,'P1M','time.dt.month','MM_YYYYMM_YYYYMM','YYYYMM',2,'Range of monthly data','range'),
        'MONTH':  ('month','M',False,False,True,'P1M','time.dt.month','MONTH_MM_YYYY_YYYY','YYYY',3,'Climatological month','climatology_month'),
        'MONTHS': ('month','M',False,False,True,'P1M','time.dt.month','MONTHS_YYYY_YYYY','YYYY',2,'Combined climatological month','climatology_month'),

        'M3':   ('month3','M',False,True, False,'P3M',None,'M3_YYYYMM_YYYYMM','YYYYMM',2,'Running 3-month','running'),
        'MM3':  ('month3','M',True, True, False,'P3M',None,'MM3_YYYYMM_YYYYMM','YYYYMM',2,'Range of running 3-month','range'),

        'JFM': ('sea','M',False,False,False,'P3M','time.dt.season','JFM_YYYY','YYYY',1,'Seasonal block JFM','season'),
        'AMJ': ('sea','M',False,False,False,'P3M','time.dt.season','AMJ_YYYY','YYYY',1,'Seasonal block AMJ','season'),
        'JAS': ('sea','M',False,False,False,'P3M','time.dt.season','JAS_YYYY','YYYY',1,'Seasonal block JAS','season'),
        'OND': ('sea','M',False,False,False,'P3M','time.dt.season','OND_YYYY','YYYY',1,'Seasonal block OND','season'),
        'SEA': ('sea','M',False,False,False,'P3M','time.dt.season','SEA_YYYY','YYYY',1,'Seasonal blocks for a single year','season'),

        'SJFM': ('sea','SEAJFM',False,False,True,'P3M','time.dt.season','SEASONJFM_YYYY_YYYY','YYYY',2,'Climatological seasonal JFM','climatology_season'),
        'SAMJ': ('sea','SEAAMJ',False,False,True,'P3M','time.dt.season','SEASONAMJ_YYYY_YYYY','YYYY',2,'Climatological seasonal AMJ','climatology_season'),
        'SJAS': ('sea','SEAJAS',False,False,True,'P3M','time.dt.season','SEASONJAS_YYYY_YYYY','YYYY',2,'Climatological seasonal JAS','climatology_season'),
        'SOND': ('sea','SEAOND',False,False,True,'P3M','time.dt.season','SEASONOND_YYYY_YYYY','YYYY',2,'Climatological seasonal OND','climatology_season'),
        'SEASON': ('season','SEA',False,False,True,'P3M','time.dt.season','SEASON_YYYY_YYYY','YYYY',2,'Combined climatological season','climatology_season'),

        'A':      ('annual','M',False,False,False,'P1Y','time.dt.year','A_YYYY','YYYY',1,'Annual (monthly inputs)','year'),
        'AA':     ('annual','M',True, False,False,'P1Y','time.dt.year','AA_YYYY_YYYY','YYYY',2,'Range of annual','range'),
        'ANNUAL': ('annual','A',False,False,True,'P1Y','time.dt.year','ANNUAL_YYYY_YYYY','YYYY',2,'Climatological annual','climatology_year'),

        'Y':      ('year','D',False,False,False,'P1Y','time.dt.year','Y_YYYY','YYYY',1,'Yearly (daily inputs)','year'),
        'YY':     ('year','D',True, False,False,'P1Y','time.dt.year','YY_YYYY_YYYY','YYYY',2,'Range of yearly','range'),
        'YEAR':   ('year','Y',False,False,True,'PnY','time.dt.year','YEAR_YYYY_YYYY','YYYY',2,'Climatological year','climatology_year'),
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
    
    def build_regex(code, file_format, date_format, n_segments):
        """
        Build a strict anchored regex based on file_format tokens.
        Returns (compiled_pattern, parts_list).
        Group naming rules:
        - 'doy' for a leading DDD/DOY token
        - 'week' only if exactly one week-like token appears in the pattern
        - otherwise positional names 'start','mid','end' (unique per pattern)
        """

        TOKEN_LENGTHS = get_token_lengths()   # once at module import

        # Extract tokens after the leading code
        parts = file_format.split("_")[1:]
        if len(parts) < n_segments:
            raise ValueError(f"file_format '{file_format}' for {code} has fewer segments ({len(parts)}) than n_segments={n_segments}")
        parts = parts[:n_segments]

        # validate tokens
        for seg in parts:
            if seg not in TOKEN_LENGTHS:
                raise ValueError(f"Unknown segment token '{seg}' in file_format for {code}")

        rx_parts = []
        for i, seg in enumerate(parts):
            length = TOKEN_LENGTHS[seg]
            if isinstance(length, tuple):
                # file_format tokens should be scalar lengths; mixed-length tuples are for date_format metadata (e.g., DOY)
                raise ValueError(f"Unexpected mixed-length token '{seg}' in file_format for {code}")
            # positional unique group names
            gname = f"seg{i}"
            rx_parts.append(rf"(?P<{gname}>\d{{{length}}})")

        pattern = rf"^{re.escape(code)}_" + "_".join(rx_parts) + r"$"
        return re.compile(pattern), parts
    
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
            date_format,
            n_date_segments,
            description,
            span_type
        ) = info

        pattern, parts = build_regex(code, file_format, date_format, n_date_segments)

        return {
            "groupby": groupby,
            "input_period_code": input_period_code,
            "is_range": is_range,
            "is_running_mean": is_running_mean,
            "is_climatology": is_climatology,
            "iso_duration": iso_duration,
            "groupby_hint": groupby_hint,
            "file_format": file_format,
            "date_format": date_format,
            "n_date_segments": n_date_segments,
            "description": description,
            "span_type":span_type,
            "regex": pattern,
            "file_format_parts": parts,
        }
    
    raw_map = period_info()
    if period_code:
        if period_code not in raw_map:
            raise KeyError(f"Period code '{period_code}' not found in period_info()")
        return build_info(period_code, raw_map[period_code])

    return {code: build_info(code, info) for code, info in raw_map.items()}



def check_period(period_base, diagnostics=False, placeholder="NA"):
    """
    Validate a single period string and return a structured result.

    Parameters
    - period_base (str): e.g. "D_20200101" or "DOY_234_2015_2025"
    - diagnostics (bool): if True, include an 'error' message on failure
    - placeholder (str): value used for placeholder results on failure

    Returns (dict):
    - On success: {
          "ok": True,
          "matched_code": str,
          "info": dict,                # entry from get_period_info()
          "parts": list[str],          # file_format parts in order
          "segments": list[str],       # captured segment strings (seg0, seg1, ...)
          "match_obj": re.Match
      }
    - On failure: {
          "ok": False,
          "error": str,                # human message (only if diagnostics True)
      }
    """
    # helper to return failure
    def _fail(msg):
        return {"ok": False, "error": msg} if diagnostics else {"ok": False}

    # load supporting metadata (must exist in your module)
    period_map = get_period_info()
    TOKEN_LENGTHS = get_token_lengths()

    # normalize input
    if not isinstance(period_base, str) or not period_base:
        return _fail("Invalid input period string")

    base = Path(period_base).stem  # strip extension if present
    codes_sorted = sorted(period_map.keys(), key=lambda x: -len(x))

    # 1. quick prefix check
    prefix = base.split("_")[0]
    if prefix not in period_map:
        return _fail("Unknown or invalid period code")

    # 2. quick raw-part count check (fast fail)
    raw_parts = base.split("_")[1:]
    expected_n = period_map[prefix]["n_date_segments"]
    if len(raw_parts) < expected_n:
        return _fail(f"{prefix} expects {expected_n} segment(s), got {len(raw_parts)}")

    # 3. find matched_code via regex fullmatch
    matched_code = None
    match_obj = None
    for code in codes_sorted:
        info2 = period_map[code]
        m = info2["regex"].fullmatch(base)
        if m:
            matched_code = code
            match_obj = m
            break

    if not matched_code:
        return _fail("Invalid format for this period code")

    # 4. ensure matched_code equals prefix
    if matched_code != prefix:
        return _fail(f"Filename prefix {prefix} does not match matched period code {matched_code}")

    # 5. prepare info, parts, segments
    info = period_map[matched_code]
    gd = match_obj.groupdict()

    parts = info.get("file_format_parts", info["file_format"].split("_")[1:])
    parts = parts[: info["n_date_segments"]]

    segments = []
    for i, token in enumerate(parts):
        gname = f"seg{i}"
        if gname not in gd:
            return _fail(f"{matched_code} regex did not produce expected group '{gname}'")
        segments.append(gd[gname])

    # defensive: ensure segment count matches expectation
    if len(segments) != info["n_date_segments"]:
        return _fail(f"{matched_code} expects {info['n_date_segments']} segment(s), got {len(segments)}")

    # 6. per-segment length validation (validate each segment against its token)
    for idx, token in enumerate(parts):
        seg_val = segments[idx]
        if token not in TOKEN_LENGTHS:
            return _fail(f"Unknown token '{token}' in file_format for {matched_code}")
        expected = TOKEN_LENGTHS[token]
        if isinstance(expected, tuple):
            # If you have DOY-style mixed-length tokens, handle them explicitly elsewhere.
            return _fail(f"Unexpected mixed-length token '{token}' in file_format for {matched_code}")
        else:
            if len(seg_val) != expected:
                return _fail(f"{matched_code} expects segment '{token}' length {expected}, got {len(seg_val)}")

    # 7. numeric check for all segments
    for seg_val in segments:
        if not seg_val.isdigit():
            return _fail(f"{matched_code} segments must be numeric")

    # success: return normalized pieces for semantic handling
    return {
        "ok": True,
        "matched_code": matched_code,
        "info": info,
        "parts": parts,
        "segments": segments,
        "match_obj": match_obj,
    }

def extract_period_code(filename):
    """
    Extract the period code, the full period, and date segments from a filename.
    Assumes the period code is the first token in the filename. 
    
    Parameters:
        filename (str): The filename to extract the period code from.

    Returns:
        The extracted period code (str), the full period (str) and the date segments, or (None,None,None) if no valid code is found.
    """

    TOKEN_REGEX = re.compile(
        r"^([A-Z0-9]+)"          # period code (M, D3, DOY, etc.)
        r"_(\d{3,8})"            # first numeric segment
        r"(?:_(\d{3,8}))?"       # optional second segment
        r"(?:_(\d{3,8}))?"       # optional third segment (DOY)
    )

    base = Path(filename).name

    m = TOKEN_REGEX.match(base)
    if not m:
        return None, None, None

    code = m.group(1)
    segs = [g for g in m.groups()[1:] if g]  # numeric segments

    full_token = code + "_" + "_".join(segs)
    valid = check_period(full_token)

    if not valid.get("ok"):
        return None, None, None

    return code, full_token, segs

def get_period_dates(files,format="%Y%m%d",placeholder="NA", diagnostics=False):
    """
    Extracts the period encoded in a filename (e.g. 'M_202204' or 'DD3_20220401_20220403') and returns the start and end dates of that period.

    The function uses the period_info() table and regex patterns, then:
      - For range codes (e.g. DD, DD3, MM3): reads both start/end dates directly
      - For single‐period codes (D, M, A, etc.): computes end date via calendar or ISO-week logic
      - Weekly codes (W, WW, WEEK) use ISO calendar weeks

    Parameters
      - filename (str): Name or path of the file containing a period segment (e.g. 'M_202204.nc' or 'WW_202201_202203.nc'.
      - format (str): Python date‐format to return (default "%Y%m%d") (optional)
      - placeholder: Value to return when no valid period is found (default "NA") (optional)
      - diagnostics (bool): If True, print debug info about regex matching and parsing (default False) (optional)

    Helper functions
      - seg_value_for (in get_period_dates): Map token -> segment value (1-based occurrence).
      - parse_yyyymmdd (in get_period_dates): Parses a date segment (YYYYMMDD) into a date object.
      - parse_yyyymm (in get_period_dates): Parses a date segment (YYYYMM) into start and end date objects.
      - token_indices (in get_period_dates): Convenience function to get token index list for a token.
      - fail (in get_period_dates): Helper to record failure and optionally diagnostics.
    
    
    Returns
        (start_str, end_str) - (tuple of str): Start and end dates of the period, formatted per default_format.

    Raises
        ValueError
            If no known period pattern is found in `filename`.
    """
    
    # 0. Create helper functions
    # helper function: map token -> segment value (1-based occurrence)
    def seg_value_for(token, occurrence=1):
        count = 0
        for idx, t in enumerate(parts):
            if t == token:
                count += 1
                if count == occurrence:
                    return segments[idx]
        raise ValueError(f"Token {token} occurrence {occurrence} not found in parts")

    # helper function: parse YYYYMMDD -> date
    def parse_yyyymmdd(s):
        try:
            y = int(s[0:4]); m = int(s[4:6]); d = int(s[6:8])
            return date(y, m, d)
        except Exception:
            raise ValueError("Invalid YYYYMMDD")

    # helper function: parse YYYYMM -> (start_date, end_date)
    def parse_yyyymm(s):
        try:
            y = int(s[0:4]); m = int(s[4:6])
            start = date(y, m, 1)
            # compute last day of month
            if m == 12:
                end = date(y, 12, 31)
            else:
                end = date(y, m+1, 1) - timedelta(days=1)
            return start, end
        except Exception:
            raise ValueError("Invalid YYYYMM")

    # helper function (convenience): get token index list for a token
    def token_indices(token):
        return [i for i, t in enumerate(parts) if t == token]
 
    # 1. Cache token and period metadata and precompile regexes
    TOKEN_LENGTHS = get_token_lengths()   # once at module import
    period_map = get_period_info()

    original_input = files
    if isinstance(files, str):
        files = [files]

    results = []
    errors = []
    placeholder = placeholder if 'placeholder' in globals() else "NA"

    # 2. Process each file
    for fname in files:
        
        # Helper functgion: fail() bound to this fname so diagnostics record the correct file
        def fail(msg):
            results.append((placeholder, placeholder))
            if diagnostics:
                errors.append((fname, msg))
            return True
        
        base = Path(fname).stem
        percode,fullperiod,segments = extract_period_code(base)
        if not percode:
            if diagnostics:
                errors.append(f"Invalid period token in '{fname}'")
            results.append((placeholder, placeholder))
            continue

        # 3. Check that the the input period is valid
        res = check_period(fullperiod, diagnostics=diagnostics, placeholder=placeholder)
        if not res.get("ok"):
            # record the error message if diagnostics requested
            if diagnostics:
                errors.append((fname, res.get("error", "Invalid format")))
            # results already not appended by check_period; append placeholder for consistency
            results.append((placeholder, placeholder))
            continue

        # 4. Unpack normalized pieces for semantic handling
        matched_code = res["matched_code"]
        parts = res["parts"]
        segments = res["segments"]

        # 5. Period specific handlers
        # DAILY single date: D
        if matched_code == "D":
            try:
                s = seg_value_for("YYYYMMDD", 1)
            except ValueError:
                if fail("D file_format missing YYYYMMDD token"):
                    continue
            try:
                dt = parse_yyyymmdd(s)
            except ValueError:
                if fail("Invalid date in D"):
                    continue
            results.append((dt.strftime("%Y%m%d"), dt.strftime("%Y%m%d")))
            continue

        # DAILY range: DD
        if matched_code == "DD":
            try:
                s = seg_value_for("YYYYMMDD", 1)
                e = seg_value_for("YYYYMMDD", 2)
            except ValueError:
                if fail("DD file_format missing YYYYMMDD tokens"):
                    continue
            try:
                ds = parse_yyyymmdd(s)
                de = parse_yyyymmdd(e)
            except ValueError:
                if fail("Invalid date in DD"):
                    continue
            if ds > de:
                if fail("DD start must be <= end"):
                    continue
            results.append((ds.strftime("%Y%m%d"), de.strftime("%Y%m%d")))
            continue

        # Climatological day-of-year: DOY (DOY_DDD_YYYY_YYYY)
        if matched_code == "DOY":
            try:
                doy_str = None
                if token_indices("DDD"):
                    doy_str = seg_value_for("DDD", 1)
                elif token_indices("DOY"):
                    doy_str = seg_value_for("DOY", 1)
                else:
                    raise ValueError
            except ValueError:
                if fail("DOY file_format does not contain DDD/DOY token"):
                    continue
            try:
                sy_str = seg_value_for("YYYY", 1)
                ey_str = seg_value_for("YYYY", 2)
            except ValueError:
                if fail("DOY file_format missing start/end year tokens"):
                    continue
            if not (doy_str.isdigit() and sy_str.isdigit() and ey_str.isdigit()):
                if fail("DOY segments must be numeric"):
                    continue
            doy_i = int(doy_str); sy = int(sy_str); ey = int(ey_str)
            if doy_i < 1 or doy_i > 366:
                if fail("Invalid day of year"):
                    continue
            if sy > ey:
                if fail("DOY start year must be <= end year"):
                    continue
            try:
                start_date = date(sy, 1, 1) + timedelta(days=doy_i - 1)
                end_date = date(ey, 12, 31)
            except Exception:
                if fail("Invalid DOY date conversion"):
                    continue
            results.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue

        # Climatological DOYs combined: DOYS (DOYS_YYYY_YYYY)
        if matched_code == "DOYS":
            try:
                sy_str = seg_value_for("YYYY", 1)
                ey_str = seg_value_for("YYYY", 2)
            except ValueError:
                if fail("Invalid DOYS format"):
                    continue
            if not (sy_str.isdigit() and ey_str.isdigit()):
                if fail("DOYS segments must be numeric"):
                    continue
            sy = int(sy_str); ey = int(ey_str)
            if sy > ey:
                if fail("DOYS start year must be <= end year"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(ey,12,31).strftime("%Y%m%d")))
            continue

        # Daily running windows: D3, DD3, D8, DD8
        if matched_code in ("D3", "DD3", "D8", "DD8"):
            expected_span = 3 if matched_code.endswith("3") else 8

            try:
                s = seg_value_for("YYYYMMDD", 1)
                e = seg_value_for("YYYYMMDD", 2)
            except ValueError:
                if fail(f"{matched_code} file_format missing YYYYMMDD tokens"):
                    continue
            try:
                ds = parse_yyyymmdd(s)
                de = parse_yyyymmdd(e)
            except ValueError:
                if fail(f"Invalid date in {matched_code}"):
                    continue
            if ds > de:
                if fail(f"{matched_code} start must be <= end"):
                    continue
            # Enforce fixed-span only for D3 and D8 
            if matched_code in ("D3", "D8"):
                span_days = (de - ds).days + 1
                if span_days != expected_span:
                    if fail(f"{matched_code} must span exactly {expected_span} days"):
                        continue
            results.append((ds.strftime("%Y%m%d"), de.strftime("%Y%m%d")))
            continue

        # Weekly (individual): W (YYYYWW)
        if matched_code == "W":
            token = segments[0]
            try:
                w_year = int(token[:4])
                w_week = int(token[4:])
            except Exception:
                if fail("W file_format missing or malformed YYYYWW token"):
                    continue

            # basic ISO week range check
            if w_week < 1 or w_week > 53:
                if fail("Invalid ISO week"):
                    continue

            # convert ISO week to start (Monday) and end (Sunday)
            try:
                start_date = date.fromisocalendar(w_year, w_week, 1)
                end_date = date.fromisocalendar(w_year, w_week, 7)
            except ValueError:
                if fail("Invalid ISO week for the specified year"):
                    continue

            results.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue

        # Weekly range: WW (YYYYWW_YYYYWW)
        if matched_code == "WW":
            try:
                s = seg_value_for("YYYYWW", 1)
                e = seg_value_for("YYYYWW", 2)
            except ValueError:
                if fail("WW file_format missing YYYYWW tokens"):
                    continue
            try:
                sy = int(s[:4]); sw = int(s[4:])
                ey = int(e[:4]); ew = int(e[4:])
            except Exception:
                if fail("WW segments must be numeric YYYYWW"):
                    continue
            # validate weeks
            if sw < 1 or sw > 53 or ew < 1 or ew > 53:
                if fail("Invalid ISO week in WW"):
                    continue
            # compute start = Monday of start YYYYWW, end = Sunday of end YYYYWW
            try:
                start_date = date.fromisocalendar(sy, sw, 1)
                end_date = date.fromisocalendar(ey, ew, 7)
            except ValueError:
                if fail("Invalid ISO week for the specified year in WW"):
                    continue
            if start_date > end_date:
                if fail("WW start must be <= end"):
                    continue
            results.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue

        # Climatological WEEK (ISO week): WEEK_WW_YYYY_YYYY 
        if matched_code == "WEEK":
            # Defensive sanity check: check_period should have validated token counts/lengths
            #if len(parts) != 3 or len(segments) != 3:
            #    if fail(f"{matched_code} expects 3 segment(s), got {len(parts)}"):
            #        continue
            # segments: [WW, YYYY_start, YYYY_end]
            try:
                w_week = int(segments[0])
                sy = int(segments[1])
                ey = int(segments[2])
            except Exception:
                if fail("WEEK file_format missing or malformed WW/ YYYY tokens"):
                    continue

            # basic checks
            if w_week < 1 or w_week > 53:
                if fail("Invalid ISO week"):
                    continue
            if sy > ey:
                if fail("WEEK start year must be <= end year"):
                    continue

            # compute start (Monday of week in start year) and end (Sunday of week in end year)
            try:
                start_date = date.fromisocalendar(sy, w_week, 1)
                end_date = date.fromisocalendar(ey, w_week, 7)
            except ValueError:
                if fail("Invalid ISO week for the specified year"):
                    continue

            results.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue

        # WEEKS (climatological weeks combined) - two YYYY tokens
        if matched_code == "WEEKS":
            try:
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail("Invalid format for this period code"):
                    continue
            if sy > ey:
                if fail("WEEKS start year must be <= end year"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(ey,12,31).strftime("%Y%m%d")))
            continue

        # Monthly (individual): M (M_YYYYMM)
        if matched_code == "M":
            try:
                m_str = seg_value_for("YYYYMM", 1)
            except ValueError:
                if fail("M file_format missing YYYYMM token"):
                    continue
            try:
                s, e = parse_yyyymm(m_str)
            except ValueError:
                if fail("Invalid month in M"):
                    continue
            results.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
            continue

        # Monthly range: MM (MM_YYYYMM_YYYYMM)
        if matched_code == "MM":
            try:
                s = seg_value_for("YYYYMM", 1)
                e = seg_value_for("YYYYMM", 2)
            except ValueError:
                if fail("MM file_format missing YYYYMM tokens"):
                    continue
            try:
                ds, _ = parse_yyyymm(s)
                _, de = parse_yyyymm(e)
            except ValueError:
                if fail("Invalid month in MM"):
                    continue
            if ds > de:
                if fail("MM end must be >= start"):
                    continue
            results.append((ds.strftime("%Y%m%d"), de.strftime("%Y%m%d")))
            continue
        
        # Climatological month: MONTH (MONTH_MM_YYYY_YYYY)
        if matched_code == "MONTH":
            try:
                mm = seg_value_for("MM", 1)
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail("MONTH file_format missing expected tokens"):
                    continue
            if not mm.isdigit():
                if fail("MONTH segments must be numeric"):
                    continue
            m_i = int(mm)
            if m_i < 1 or m_i > 12:
                if fail("Invalid month in MONTH"):
                    continue
            if sy > ey:
                if fail("MONTH start year must be <= end year"):
                    continue
            # start = first day of month in start year; end = last day of month in end year
            try:
                start_date = date(sy, m_i, 1)
                if m_i == 12:
                    end_date = date(ey, 12, 31)
                else:
                    end_date = date(ey, m_i+1, 1) - timedelta(days=1)
            except Exception:
                if fail("Invalid MONTH date conversion"):
                    continue
            results.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue

        # MONTHS (climatological months combined): MONTHS(MONTHS_YYYY_YYYY)
        if matched_code == "MONTHS":
            try:
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail("Invalid format for this period code"):
                    continue
            if sy > ey:
                if fail("MONTHS start year must be <= end year"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(ey,12,31).strftime("%Y%m%d")))
            continue

        # Monthly running windows: M3 (exact 3-month span), MM3 (any span, only start <= end enforced)
        if matched_code in ("M3", "MM3"):
            expected_months = 3

            # Defensive sanity check: check_period() should have validated token counts and lengths
            #if len(parts) != 2 or len(segments) != 2:
            #    if fail(f"{matched_code} expects 2 segment(s), got {len(parts)}"):
            #        continue

            # parse start and end months (segments are validated numeric strings)
            s = segments[0]
            e = segments[1]
            try:
                ds, _ = parse_yyyymm(s)
                _, de = parse_yyyymm(e)
            except ValueError:
                if fail(f"Invalid month in {matched_code}"):
                    continue

            if ds > de:
                if fail(f"{matched_code} start must be <= end"):
                    continue

            # compute end_date as last day of end month
            if de.month == 12:
                end_date = date(de.year, 12, 31)
            else:
                end_date = date(de.year, de.month + 1, 1) - timedelta(days=1)

            if matched_code == "M3":
                # enforce exact 3-month span
                months_span = (de.year - ds.year) * 12 + (de.month - ds.month) + 1
                if months_span != expected_months:
                    if fail("M3 must span exactly 3 months"):
                        continue

            # MM3 accepts any span (only start <= end enforced)
            results.append((ds.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue


        # Seasons (single year): JFM/AMJ/JAS/OND/SEA (SEA_YYYY))
        if matched_code in ("JFM","AMJ","JAS","OND","SEA"):
            try:
                sy = int(seg_value_for("YYYY", 1))
            except ValueError:
                if fail(f"{matched_code} Invalid format for this period code"):
                    continue
            # Map season to month ranges
            season_map = {
                "JFM": (1,3),
                "AMJ": (4,6),
                "JAS": (7,9),
                "OND": (10,12),
                "SEA": (1,12),  # SEA for all 4 seasons within a single file
            }
            if matched_code not in season_map:
                if fail(f"Unknown seasonal code: {matched_code}"):
                    continue
            mstart, mend = season_map[matched_code]
            try:
                start_date = date(sy, mstart, 1)
                if mend == 12:
                    end_date = date(sy, 12, 31)
                else:
                    end_date = date(sy, mend+1, 1) - timedelta(days=1)
            except Exception:
                if fail(f"Invalid date for {matched_code}"):
                    continue
            results.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue

        # Seasonal climatologies: SJFM, SAMJ, SJAS, SOND, SEASON (SEASON_YYYY_YYYY)
        if matched_code in ("SJFM","SAMJ","SJAS","SOND","SEASON"):
            try:
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail(f"{matched_code} Invalid format for this period code"):
                    continue
            if sy > ey:
                if fail(f"{matched_code} start year must be <= end year"):
                    continue
            # map to start of sy season to end of ey season (use same season_map)
            season_map = {
                "SJFM": (1,3),
                "SAMJ": (4,6),
                "SJAS": (7,9),
                "SOND": (10,12),
                "SEASON": (1,12),
            }
            mstart, mend = season_map[matched_code]
            try:
                start_date = date(sy, mstart, 1)
                if mend == 12:
                    end_date = date(ey, 12, 31)
                else:
                    end_date = date(ey, mend+1, 1) - timedelta(days=1)
            except Exception:
                if fail(f"Invalid date for {matched_code}"):
                    continue
            results.append((start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")))
            continue

        # Annual (single year): A (A_YYYY)
        if matched_code == "A":
            try:
                sy = int(seg_value_for("YYYY", 1))
            except ValueError:
                if fail("A Invalid format for this period code"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(sy,12,31).strftime("%Y%m%d")))
            continue
        
        # Annual range: AS (AS_YYYY_YYYY)
        if matched_code == "AA":
            try:
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail("AA Invalid format for this period code"):
                    continue
            if sy > ey:
                if fail("AA start year must be <= end year"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(ey,12,31).strftime("%Y%m%d")))
            continue

        # Annual climatology: (ANNUAL_YYYY_YYYY)
        if matched_code == "ANNUAL":
            try:
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail("ANNUAL Invalid format for this period code"):
                    continue
            if sy > ey:
                if fail("ANNUAL start year must be <= end year"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(ey,12,31).strftime("%Y%m%d")))
            continue

        # Yearly (single): Y (Y_YYYY) - note Y inputs are daily files
        if matched_code == "Y":
            try:
                sy = int(seg_value_for("YYYY", 1))
            except ValueError:
                if fail("Y Invalid format for this period code"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(sy,12,31).strftime("%Y%m%d")))
            continue

        # Yearly range: Y (Y_YYYY_YYYY)
        if matched_code == "YY":
            try:
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail("YY Invalid format for this period code"):
                    continue
            if sy > ey:
                if fail("YY start year must be <= end year"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(ey,12,31).strftime("%Y%m%d")))
            continue

        # Yearly climatology: YEAR (YEAR_YYYY_YYYY)
        if matched_code == "YEAR":
            try:
                sy = int(seg_value_for("YYYY", 1))
                ey = int(seg_value_for("YYYY", 2))
            except ValueError:
                if fail("YEAR Invalid format for this period code"):
                    continue
            if sy > ey:
                if fail("YEAR start year must be <= end year"):
                    continue
            results.append((date(sy,1,1).strftime("%Y%m%d"), date(ey,12,31).strftime("%Y%m%d")))
            continue

        # If we reach here, matched_code not handled explicitly
        if fail("Unknown or unhandled period code"):
            continue

    # --- Final return normalization ---
    if isinstance(original_input, str):
        # Single token → return a 2‑tuple
        if diagnostics:
            return results[0], errors
        return results[0]

    # Otherwise: list input → return list of tuples
    return (results, errors) if diagnostics else results
    

def get_period_sets(
        period_code, 
        files=None,
        start=None, 
        end=None, 
        dates=None, 
        date_format="%Y%m%d",
        running=True,
        climatology_range=(1991,2020),
        diagnostics=False
):
    """
    Generate output period tokens (e.g. M_YYYYMM, W_YYYYWW) and input mappings for a given date range, list of dates, or list of periods.

    Parameters
        - period_code (str): Period code to generate (e.g. 'M', 'W', 'A', 'Y').
        - files (list[str], optional): list of filenames that start with period tokens (e.g., 'M_202201-chl.nc').
        - start (str or datetime, optional): Start date in YYYYMMDD format or datetime object.
        - end (str or datetime, optional): End date in YYYYMMDD format or datetime object.
        - dates (list[str|datetime], optional): list of dates to infer range from.
        - period_tokens (list[str], optional): list of existing period tokens (e.g., ['M_200001', ...]).
        - date_format (str): format used for input date strings and for token construction.
        - running (bool): for running-window codes (D3, M3), whether to produce sliding windows.
        - climatology_range (tuple): default climatology years (start_year, end_year).
        - diagnostics (bool): if True, return (outputs, diagnostics_list).

    Returns
        - a period map that show the list of input periods/dates/files for each output period
    
    Helper functions
        - to_dt(x): Convert input string or datetime to datetime object.
        - token_prefix_from_file(fname): Extract token prefix and matched token from filename using get_period_info() regexes.
        - run_get_period_dates(period_token): Get (start_dt, end_dt) for a single token or filename by calling get_period_dates().
        - iso_week_tokens_between(s_dt, e_dt): Generate ISO week tokens between two datetimes.
        - month_tokens_between(s_dt, e_dt): Generate month tokens between two datetimes.
        - year_tokens_between(s_dt, e_dt, prefix): Generate year tokens between two datetimes with a given prefix.
        - running_daily_windows(code_key, s_dt, e_dt): Generate daily running window tokens between two datetimes for a given code key (e.g. D3).
        - running_monthly_windows(code_key, s_dt, e_dt): Generate monthly running window tokens between two datetimes for a given code key (e.g. M3).
    
    Raises:
      - ValueError for invalid inputs or unsupported requests.
    """
    # 0. Helper functions 
    def to_dt(x):
        if isinstance(x, datetime):
            return x
        return datetime.strptime(x, date_format)
    
    def run_get_period_dates(period_token, date_format="%Y%m%d", diagnostics=False):
        """
        Return (start_dt, end_dt) for a single token or filename by calling get_period_dates().
        Raises ValueError on parse failure; returns datetimes on success.
        """
    
        results = get_period_dates(period_token, format=date_format, placeholder=None, diagnostics=diagnostics)
        # get_period_dates returns either [(start,end)] or (results, errors) when diagnostics=True
        if diagnostics:
            results, errors = results
            if errors:
                raise ValueError(f"Parsing failed for {period_token}: {errors}")
        else:
            pass # results is a list of tuples

        if not results or results[0] is None:
            raise ValueError(f"Could not parse token/file: {period_token}")

        start_str, end_str = results[0]
        if start_str in (None, "NA") or end_str in (None, "NA"):
            raise ValueError(f"Invalid token/file span: {period_token}")

        start_dt = datetime.strptime(start_str, date_format)
        end_dt   = datetime.strptime(end_str, date_format)
        return start_dt, end_dt
    
    def iso_week_tokens_between(s_dt, e_dt):
        toks = []
        cur = s_dt - timedelta(days=s_dt.weekday())  # Monday
        while cur <= e_dt:
            y, w, _ = cur.isocalendar()
            toks.append(f"W_{y:04d}{w:02d}")
            cur += timedelta(days=7)
        return sorted(dict.fromkeys(toks))

    def month_tokens_between(s_dt, e_dt):
        toks = []
        cur = s_dt.replace(day=1)
        while cur <= e_dt:
            toks.append(f"M_{cur.strftime('%Y%m')}")
            cur = (cur + relativedelta(months=1)).replace(day=1)
        return toks

    def year_tokens_between(s_dt, e_dt, prefix):
        toks = []
        y = s_dt.year
        while y <= e_dt.year:
            toks.append(f"{prefix}_{y:04d}")
            y += 1
        return toks

    def running_daily_windows(code_key, s_dt, e_dt):
        step = int(code_key[1:])
        delta = timedelta(days=step)
        toks = []
        if running:
            cur = s_dt
            while cur + delta - timedelta(days=1) <= e_dt:
                s = cur.strftime(date_format)
                e = (cur + delta - timedelta(days=1)).strftime(date_format)
                toks.append(f"{code_key}_{s}_{e}")
                cur += timedelta(days=1)
        else:
            cur = s_dt
            while cur <= e_dt:
                end_block = cur + delta - timedelta(days=1)
                if end_block <= e_dt:
                    toks.append(f"{code_key}_{cur.strftime(date_format)}_{end_block.strftime(date_format)}")
                cur += delta
        return toks

    def running_monthly_windows(code_key, s_dt, e_dt):
        toks = []
        if running:
            cur = s_dt
            while True:
                end_block = cur + relativedelta(months=3) - timedelta(days=1)
                if end_block > e_dt:
                    break
                toks.append(f"{code_key}_{cur.strftime(date_format)}_{end_block.strftime(date_format)}")
                cur += timedelta(days=1)
        else:
            cur = s_dt.replace(day=1)
            while cur <= e_dt:
                end_block = cur + relativedelta(months=3) - timedelta(days=1)
                if end_block <= e_dt:
                    toks.append(f"{code_key}_{cur.strftime(date_format)}_{end_block.strftime(date_format)}")
                cur += relativedelta(months=3)
        return toks

    def generate_periods(span_type, start_dt, end_dt, climatology_range):
        """
        Dispatch to the correct span generator based on span_type.
        Returns a list of period tuples.
        """

        if span_type == "climatology_doy":
            return generate_climatology_doy(start_dt, end_dt)
        
        if span_type == "week":
            return generate_weeks(start_dt, end_dt)

        if span_type == "month":
            return generate_months(start_dt, end_dt)

        if span_type == "season":
            return generate_seasons(start_dt, end_dt)

        if span_type == "year":
            return generate_annual(start_dt, end_dt)

        # --- Climatology ---
        y1, y2 = climatology_range

        if span_type == "climatology_month":
            return generate_climatology_months(y1, y2)

        if span_type == "climatology_doys":
            return generate_climatology_doys(y1, y2)

        if span_type == "climatology_season":
            return generate_climatology_seasons(y1, y2)

        raise NotImplementedError(f"Unknown span type '{span_type}'.")

    def generate_climatology_doy(clim_start, clim_end):
        """
        Generate (doy,) tuples for climatological DOY.
        Always 365 (or 366) subperiods.
        """
        days = 366 if is_leap_year(clim_start) else 365
        return [(d,) for d in range(1, days + 1)]
    
    def generate_climatology_doys(clim_start, clim_end):
        # No subperiods — one combined file
        return [("ALL_DOYS",)]
    
    def generate_weeks(start_dt, end_dt):
        """
        Generate (year, week) tuples for all ISO weeks between start_dt and end_dt.
        """
        start = datetime.strptime(start_dt, "%Y%m%d")
        end   = datetime.strptime(end_dt, "%Y%m%d")

        # Snap start to Monday of its ISO week
        start_iso = start.isocalendar()
        current = start - timedelta(days=start_iso.weekday)

        # Snap end to Monday of its ISO week
        end_iso = end.isocalendar()
        end_week_start = end - timedelta(days=end_iso.weekday)

        periods = []

        while current <= end_week_start:
            iso = current.isocalendar()
            year, week = iso.year, iso.week
            periods.append((year, week))
            current += timedelta(days=7)

        return periods
    
    def generate_climatology_week(clim_start, clim_end):
        # ISO weeks: 52 or 53
        weeks = 53 if iso_year_has_53_weeks(clim_start) else 52
        return [(w,) for w in range(1, weeks + 1)]
    
    def generate_climatology_weeks(clim_start, clim_end):
        return [("ALL_WEEKS",)]

    def generate_months(start_dt, end_dt):
        """
        Generate (year, month) tuples for all months between start_dt and end_dt.
        start_dt and end_dt are strings in YYYYMMDD format.
        """
        # Parse into datetime objects
        start = datetime.strptime(start_dt, "%Y%m%d")
        end   = datetime.strptime(end_dt, "%Y%m%d")

        # Snap to first day of start month
        current = start.replace(day=1)

        # Snap end to first day of its month
        end_month = end.replace(day=1)

        periods = []

        while current <= end_month:
            periods.append((current.year, current.month))

            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        return periods
    
    def generate_climatology_month(clim_start, clim_end):
        """
        Generate (month,) tuples for climatological months.
        Always 12 subperiods: 1–12.
        """
        return [(m,) for m in range(1, 13)]
    
    def generate_climatology_months(clim_start, clim_end):
        return [("ALL_MONTHS",)]

    def generate_seasons(start_dt, end_dt):
        """
        Generate (year, season_code) tuples for JFM, AMJ, JAS, OND.
        """
        start = datetime.strptime(start_dt, "%Y%m%d")
        end   = datetime.strptime(end_dt, "%Y%m%d")

        # Snap to first month of its season
        def season_start_month(m):
            if m in (1,2,3): return 1
            if m in (4,5,6): return 4
            if m in (7,8,9): return 7
            return 10

        current = start.replace(month=season_start_month(start.month), day=1)
        end_season = end.replace(month=season_start_month(end.month), day=1)

        periods = []

        while current <= end_season:
            m = current.month
            if m == 1:  code = "JFM"
            elif m == 4: code = "AMJ"
            elif m == 7: code = "JAS"
            else:        code = "OND"

            periods.append((current.year, code))

            # Advance 3 months
            if m == 10:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=m + 3)

        return periods
    
    def generate_climatology_seasons(clim_start, clim_end):
        """
        Generate (season_code,) tuples for climatological seasons.
        """
        return [("JFM",), ("AMJ",), ("JAS",), ("OND",)]
    
    def generate_climatology_seasons(clim_start, clim_end):
        return [("ALL_SEASONS",)]

    def generate_annual(start_dt, end_dt):
        """
        Generate (annual,) tuples for all years between start_dt and end_dt.
        """
        start = datetime.strptime(start_dt, "%Y%m%d")
        end   = datetime.strptime(end_dt, "%Y%m%d")

        periods = []
        for year in range(start.year, end.year + 1):
            periods.append((year,))
        return periods
    
    def generate_years(start_dt, end_dt):
        return generate_annual(start_dt, end_dt)
    
    def generate_climatology_annual(clim_start, clim_end):
        return [("ANNUAL",)]
    
    def generate_climatology_year(clim_start, clim_end):
        return [("YEAR",)]

    def format_period_token(target_code, period_tuple, period_map, climatology_range=None):
        """
        Format an output period token using the file_format template
        from period_info().
        """
        info = period_map[target_code]
        template = info['file_format']
        span_type = info['span_type']

        parts = list(period_tuple)

        # --- WEEK ---
        if span_type == "week":
            year, week = parts
            return (template
                    .replace("YYYY", f"{year:04d}")
                    .replace("WW", f"{week:02d}"))

        # --- MONTH ---
        if span_type == "month":
            year, month = parts
            return (template
                    .replace("YYYY", f"{year:04d}")
                    .replace("MM", f"{month:02d}"))

        # --- SEASON ---
        if span_type == "season":
            year, season_code = parts
            return (template
                    .replace("YYYY", f"{year:04d}")
                    .replace("SEA", season_code)
                    .replace("SEASON", season_code))

        # --- YEAR (A, Y) ---
        if span_type == "year":
            year = parts[0]
            return template.replace("YYYY", f"{year:04d}")

        # --- CLIMATOLOGY ---
        if span_type.startswith("climatology"):

            if climatology_range is None:
                raise ValueError("Climatology range must be provided to format_period_token().")

            y1, y2 = climatology_range

            # --- Special case: DOY climatology ---
            if span_type == "climatology_doy":
                doy = int(parts[0])   # period_tuple = (doy,)
                return (template
                        .replace("DDD", f"{doy:03d}")
                        .replace("YYYY", f"{y1:04d}")
                        .replace("YYYY2", f"{y2:04d}"))

            # --- Special case: DOYS (combined) ---
            if span_type == "climatology_doys":
                return (template
                        .replace("YYYY", f"{y1:04d}")
                        .replace("YYYY2", f"{y2:04d}"))

            # --- All other climatologies (month, week, season, annual) ---
            sub = parts[-1]
            token = (template
                    .replace("YYYY", f"{y1:04d}")
                    .replace("YYYY2", f"{y2:04d}"))

            if isinstance(sub, int):
                if "MM" in template:
                    token = token.replace("MM", f"{sub:02d}")
                if "DDD" in template:
                    token = token.replace("DDD", f"{sub:03d}")
            else:
                token = token.replace("SEA", sub).replace("SEASON", sub)

            return token

        raise NotImplementedError(f"Span type '{span_type}' not implemented.")


    def map_inputs_to_period(span_type, period_tuple, spans):
        """
        Dispatch to the correct input-mapping function based on span_type.
        """
        if span_type == "doy":
            return map_inputs_to_doy(period_tuple, spans)
        
        if span_type == "week":
            return map_inputs_to_week(period_tuple, spans)

        if span_type == "month":
            return map_inputs_to_month(period_tuple, spans)

        if span_type == "season":
            return map_inputs_to_season(period_tuple, spans)

        if span_type == "year":
            return map_inputs_to_year(period_tuple, spans)

        # --- Climatology ---
        if span_type == "climatology_doy":
            return map_inputs_to_climatology_doy(period_tuple, spans)
        if span_type == "climatology_doys":
            return map_inputs_to_climatology_doys(period_tuple, spans)

        if span_type == "climatology_week":
            return map_inputs_to_climatology_week(period_tuple, spans)
        if span_type == "climatology_weeks":
            return map_inputs_to_climatology_weeks(period_tuple, spans)

        if span_type == "climatology_month":
            return map_inputs_to_climatology_month(period_tuple, spans)
        if span_type == "climatology_months":
            return map_inputs_to_climatology_months(period_tuple, spans)

        if span_type == "climatology_season":
            return map_inputs_to_climatology_season(period_tuple, spans)
        if span_type == "climatology_seasons":
            return map_inputs_to_climatology_seasons(period_tuple, spans)

        if span_type == "climatology_year":
            return map_inputs_to_climatology_year(period_tuple, spans)

        raise NotImplementedError(f"Unknown span type '{span_type}'.")
   
    def map_inputs_to_climatology_doy(period_tuple, spans):
        """
        period_tuple = (doy,)
        spans = [(sdt, edt, token), ...]
        Match all inputs whose DOY equals the target DOY.
        """
        doy = period_tuple[0]
        matched = []

        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            # DOY climatology uses the *start date* DOY
            if s.timetuple().tm_yday == doy:
                matched.append(token)

        return matched
    
    def map_inputs_to_climatology_doys(period_tuple, spans):
        """
        DOYS: include all input tokens.
        """
        return [token for _, _, token in spans]

    def map_inputs_to_week(period_tuple, spans):
        year, week = period_tuple

        # Compute Monday of the ISO week
        # ISO week: Monday = 1, Sunday = 7
        # Use the ISO calendar trick: week 1 always contains Jan 4
        week_start = datetime.fromisocalendar(year, week, 1)
        week_end   = datetime.fromisocalendar(year, week, 7)

        matched = []
        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            e = datetime.strptime(edt, "%Y%m%d")
            if e >= week_start and s <= week_end:
                matched.append(token)

        return matched
    
    def map_inputs_to_climatology_week(period_tuple, spans):
        """
        period_tuple = (week,)
        Match all inputs whose ISO week equals the target week.
        """
        week = period_tuple[0]
        matched = []

        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            if s.isocalendar().week == week:
                matched.append(token)

        return matched
    
    def map_inputs_to_climatology_weeks(period_tuple, spans):
        return [token for _, _, token in spans]
    
    def map_inputs_to_month(period_tuple, spans):
        year, month = period_tuple

        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(days=1)

        matched = []
        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            e = datetime.strptime(edt, "%Y%m%d")
            if e >= month_start and s <= month_end:
                matched.append(token)

        return matched
    
    def map_inputs_to_climatology_month(period_tuple, spans):
        """
        period_tuple = (month,)
        Match all inputs whose month equals the target month.
        """
        month = period_tuple[0]
        matched = []

        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            if s.month == month:
                matched.append(token)

        return matched

    def map_inputs_to_climatology_months(period_tuple, spans):
        return [token for _, _, token in spans]

    def map_inputs_to_season(period_tuple, spans):
        year, season_code = period_tuple

        if season_code == "JFM":
            months = (1, 2, 3)
        elif season_code == "AMJ":
            months = (4, 5, 6)
        elif season_code == "JAS":
            months = (7, 8, 9)
        else:  # OND
            months = (10, 11, 12)

        season_start = datetime(year, months[0], 1)

        if season_code == "OND":
            season_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            season_end = datetime(year, months[-1] + 1, 1) - timedelta(days=1)

        matched = []
        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            e = datetime.strptime(edt, "%Y%m%d")
            if e >= season_start and s <= season_end:
                matched.append(token)

        return matched
   
    def map_inputs_to_climatology_season(period_tuple, spans):
        """
        period_tuple = (season_code,)
        Match all inputs whose month falls in the season.
        """
        season_code = period_tuple[0]

        if season_code == "JFM":
            months = (1, 2, 3)
        elif season_code == "AMJ":
            months = (4, 5, 6)
        elif season_code == "JAS":
            months = (7, 8, 9)
        else:  # OND
            months = (10, 11, 12)

        matched = []
        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            if s.month in months:
                matched.append(token)

        return matched

    def map_inputs_to_climatology_seasons(period_tuple, spans):
        return [token for _, _, token in spans]
    
    def map_inputs_to_year(period_tuple, spans):
        year = period_tuple[0]

        year_start = datetime(year, 1, 1)
        year_end   = datetime(year, 12, 31)

        matched = []
        for sdt, edt, token in spans:
            s = datetime.strptime(sdt, "%Y%m%d")
            e = datetime.strptime(edt, "%Y%m%d")
            if e >= year_start and s <= year_end:
                matched.append(token)

        return matched

    def map_inputs_to_climatology_annual(period_tuple, spans):
        return [token for _, _, token in spans]
    
    def map_inputs_to_climatology_year(period_tuple, spans):
        return [token for _, _, token in spans]
    
    """
    STEP 1: Normalize inputs and validate the requested output period code.
    """
    # --- Establish climatology range  if the input is None ---
    if climatology_range is None:
        climatology_range = (1991, 2020)
    
    # --- Normalize period_code to a single code (not iterable) ---
    if isinstance(period_code, str):
        target_code = period_code
    else:
        raise ValueError("period_code must be a single string (e.g. 'M', 'W', 'A').")

    # --- Load period metadata ---
    period_map = get_period_info()

    if target_code not in period_map:
        raise ValueError(f"Unsupported target period code: '{target_code}'")

    target_info = period_map[target_code]
    expected_input_code = target_info["input_period_code"]

    # --- Prepare canonical input list ---
    input_items = []      # canonical list of tokens or filenames
    input_prefix = None   # the prefix of the input items (D, M, W, etc.)
    start_dt = None
    end_dt = None
    diagnostics_list = [] # collects warnings, non-fatal errors

    """
    STEP 2: Determine input type and extract input date span.
    """

    # --- Initialize spans for Step 3 ---
    spans = []
    input_items = []

    # --- Case 1: Input file strings -------------------------------------
    if files:
        prefixes = set()

        for tok in files:
            results, errors = get_period_dates(tok, diagnostics=True)
            if errors:
                raise ValueError(errors[0])
            
            sdt, edt = results
            spans.append((sdt, edt, tok))
            prefixes.add(tok.split("_", 1)[0])
            input_items.append(tok)

        if len(prefixes) > 1:
            raise ValueError(f"Mixed input period prefixes not supported: {sorted(prefixes)}")

        input_prefix = next(iter(prefixes))
        if input_prefix != expected_input_code:
            raise ValueError(
                f"Cannot generate '{target_code}' from '{input_prefix}' inputs; "
                f"expected input period code is '{expected_input_code}'.")
        
        start_dt = min(s[0] for s in spans)
        end_dt   = max(s[1] for s in spans)

    # --- Case 2: Raw dates list ---------------------------------------------
    elif dates:
        # Parse all dates into datetime objects
        parsed = [d if isinstance(d, datetime) else to_dt(d) for d in dates]
        start_dt_dt = min(parsed)
        end_dt_dt   = max(parsed)

        # Convert to canonical strings for Step 3
        start_dt = start_dt_dt.strftime("%Y%m%d")
        end_dt   = end_dt_dt.strftime("%Y%m%d")

        # Build spans and input_items
        cur = start_dt_dt
        while cur <= end_dt_dt:
            sdt = cur.strftime("%Y%m%d")
            edt = sdt
            token = f"D_{sdt}"
            spans.append((sdt, edt, token))
            input_items.append(token)
            cur += timedelta(days=1)

        input_prefix = "D"

    # --- Case 3: Start/end date range ---------------------------------------
    elif start and end:
        start_dt_dt = start if isinstance(start, datetime) else to_dt(start)
        end_dt_dt   = end   if isinstance(end,   datetime) else to_dt(end)

        if start_dt_dt > end_dt_dt:
            raise ValueError("Start date must be before or equal to end date.")

        start_dt = start_dt_dt.strftime("%Y%m%d")
        end_dt   = end_dt_dt.strftime("%Y%m%d")

        # Build spans and input_items
        cur = start_dt_dt
        while cur <= end_dt_dt:
            sdt = cur.strftime("%Y%m%d")
            edt = sdt
            token = f"D_{sdt}"
            spans.append((sdt, edt, token))
            input_items.append(token)
            cur += timedelta(days=1)

        input_prefix = "D"

    else:
        raise ValueError("Provide either period_tokens, files, or start/end/dates.")

    # --- Enforce raw-date restriction ---------------------------------------
    if input_prefix == "D" and expected_input_code != "D":
        raise ValueError(
            f"Cannot generate '{target_code}' from raw dates; "
            f"its input_period_code is '{expected_input_code}' (only 'D' allowed)."
        )

    """Step 3: Generate output periods for the full date span """

    info = period_map[target_code]
    span_type = info["span_type"]
    periods = generate_periods(span_type, start_dt, end_dt, climatology_range)

    period_map_out = {}

    for p in periods:
        token = format_period_token(target_code, p, period_map, climatology_range)
        inputs = map_inputs_to_period(span_type, p, spans)
        period_map_out[token] = {"inputs": inputs}




    #periods = generate_periods(span_type, start_dt, end_dt, climatology_range)
    #tokens  = [format_period_token(target_code, p) for p in periods]
    #inputs  = [map_inputs_to_period(info["input_period_code"], p, input_items)]

    return {
        "target_code": target_code,
        "expected_input_code": expected_input_code,
        "input_items": input_items,
        "input_prefix": input_prefix,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "period_map": period_map_out,   # <-- NEW
        "diagnostics": diagnostics_list,
    }

    
    # -------------------------
    # Produce outputs
    # -------------------------
    outputs = []

    # Helper to map a subperiod token to its input D_ tokens or to files/period_tokens
    def inputs_for_subperiod(sub_tok):
        """Return list of input items (D_ tokens or filenames or period tokens) that fall into sub_tok span."""
        try:
            sdt, edt = parse_token_span(sub_tok)
        except Exception:
            return []
        # if original inputs were files, return filenames that map into this span
        if files:
            matched_files = []
            for f in files:
                pfx, gd, base = token_prefix_from_file(f)
                if not pfx:
                    continue
                # reconstruct token for file
                info = get_period_info()[pfx]
                if info["n_date_segments"] == 1:
                    tok = f"{pfx}_{gd['start']}"
                else:
                    tok = f"{pfx}_{gd['start']}_{gd['end']}"
                try:
                    fs, fe = parse_token_span(tok)
                except Exception:
                    continue
                if fs <= edt and fe >= sdt:
                    matched_files.append(f)
            return matched_files

        # if inputs were period_tokens, return those tokens that overlap
        if period_tokens:
            matched = []
            for tok in period_tokens:
                try:
                    ts, te = parse_token_span(tok)
                except Exception:
                    continue
                if ts <= edt and te >= sdt:
                    matched.append(tok)
            return matched

        # otherwise inputs were raw dates -> return D_ tokens within span
        matched = []
        cur = sdt
        while cur <= edt:
            matched.append(f"D_{cur.strftime(date_format)}")
            cur += timedelta(days=1)
        return matched

    # 1) If target is climatology and inputs are period_tokens (or user explicitly wants climatology),
    #    produce climatology token(s). Use C_ prefix for climatology outputs to distinguish.
    if target_info["is_climatology"]:
        # determine climatology years: default or user-specified via climatology_range
        cy_start, cy_end = climatology_range
        # If inputs were period_tokens/files and span narrower/wider, allow overriding climatology_range
        if period_tokens or files:
            # if user provided explicit start/end via args, use them; otherwise use span years
            if start is None and end is None and not dates:
                cy_start = start_dt.year
                cy_end = end_dt.year
        # Build main climatology token with C_ prefix to distinguish
        # e.g., C_MONTH_1991_2020
        main_token = f"C_{code}_{cy_start:04d}_{cy_end:04d}"
        # Build subperiods inside the climatology:
        subperiods = []
        # For MONTH climatology: subperiods are MONTH_01 ... MONTH_12 (or M_YYYYMM? choose canonical)
        if code == "MONTH":
            for mm in range(1, 13):
                sub_tok = f"M_{mm:02d}_{cy_start:04d}_{cy_end:04d}"  # e.g., M_01_1991_2020
                # inputs for each monthly climatology are all D_ dates for that month across years
                inputs = []
                for y in range(cy_start, cy_end + 1):
                    # build daily tokens for that month/year
                    last = calendar.monthrange(y, mm)[1]
                    for d in range(1, last + 1):
                        inputs.append(f"D_{y:04d}{mm:02d}{d:02d}")
                subperiods.append({"period": sub_tok, "inputs": inputs})
        elif code == "DOY":
            # DOY climatology: 1..366
            for doy in range(1, 367):
                sub_tok = f"DOY_{doy:03d}_{cy_start:04d}_{cy_end:04d}"
                inputs = []
                for y in range(cy_start, cy_end + 1):
                    try:
                        dt = date.fromordinal(date(y,1,1).toordinal() + doy - 1)
                        inputs.append(f"D_{dt.strftime('%Y%m%d')}")
                    except Exception:
                        # skip invalid (e.g., doy 366 in non-leap years)
                        continue
                subperiods.append({"period": sub_tok, "inputs": inputs})
        elif code == "SEASON":
            # SEASON climatology: full seasonal cycle; subperiods could be JFM, AMJ, JAS, OND across climatology years
            seasons = [("JFM", 1), ("AMJ", 4), ("JAS", 7), ("OND", 10)]
            for name, start_month in seasons:
                sub_tok = f"SEASON_{name}_{cy_start:04d}_{cy_end:04d}"
                inputs = []
                for y in range(cy_start, cy_end + 1):
                    sm = start_month
                    em = sm + 2
                    ey = y
                    if em > 12:
                        em -= 12
                        ey = y + 1
                    # collect daily inputs for the 3-month block
                    cur = datetime(y, sm, 1)
                    last_day = calendar.monthrange(ey, em)[1]
                    end_dt = datetime(ey, em, last_day)
                    while cur <= end_dt:
                        inputs.append(f"D_{cur.strftime('%Y%m%d')}")
                        cur += timedelta(days=1)
                subperiods.append({"period": sub_tok, "inputs": inputs})
        elif code == "WEEK":
            # WEEK climatology: all ISO weeks across climatology years
            toks = []
            for y in range(cy_start, cy_end + 1):
                start_iso = date.fromisocalendar(y, 1, 1)
                last_iso = date(y, 12, 28).isocalendar().week
                for w in range(1, last_iso + 1):
                    toks.append(f"W_{y:04d}{w:02d}")
            # map each week to its daily inputs
            for wk in toks:
                inputs = []
                sdt, edt = parse_token_span(wk)
                cur = sdt
                while cur <= edt:
                    inputs.append(f"D_{cur.strftime('%Y%m%d')}")
                    cur += timedelta(days=1)
                subperiods.append({"period": wk, "inputs": inputs})
        else:
            # generic fallback: produce subperiods at the target input_period_code granularity
            ip = target_info["input_period_code"]
            if ip == "D":
                for y in range(cy_start, cy_end + 1):
                    cur = datetime(y, 1, 1)
                    endy = datetime(y, 12, 31)
                    while cur <= endy:
                        subperiods.append({"period": f"D_{cur.strftime(date_format)}", "inputs": [f"D_{cur.strftime(date_format)}"]})
                        cur += timedelta(days=1)
            elif ip == "M":
                for y in range(cy_start, cy_end + 1):
                    for m in range(1, 13):
                        sub_tok = f"M_{y:04d}{m:02d}"
                        inputs = []
                        last = calendar.monthrange(y, m)[1]
                        for d in range(1, last + 1):
                            inputs.append(f"D_{y:04d}{m:02d}{d:02d}")
                        subperiods.append({"period": sub_tok, "inputs": inputs})
            else:
                # leave empty
                pass

        outputs.append({"output_period": main_token, "subperiods": subperiods, "notes": f"Climatology {cy_start}-{cy_end}"})
        if diagnostics:
            diagnostics_list.append(("climatology", f"Produced climatology {main_token} with {len(subperiods)} subperiods"))
        return (outputs, diagnostics_list) if diagnostics else outputs

    # 2) If target is a range token (is_range True) -> produce main range token and subperiods
    if target_info["is_range"]:
        # format start/end according to target_info['date_format']
        df = target_info["date_format"]
        fmt_map = {"YYYY": "%Y", "YYYYMM": "%Y%m", "YYYYMMDD": "%Y%m%d", "YYYYWW": "%Y%W"}
        if df not in fmt_map:
            raise ValueError(f"Unsupported date_format for range token: {df}")
        fmt = fmt_map[df]
        start_str = start_dt.strftime(fmt)
        end_str = end_dt.strftime(fmt)
        main_token = f"{code}_{start_str}_{end_str}"

        # subperiods: produce the canonical subperiods at the input_period_code granularity
        subperiods = []
        ip = target_info["input_period_code"]
        if ip == "D":
            cur = start_dt
            while cur <= end_dt:
                sub_tok = f"D_{cur.strftime(date_format)}"
                subperiods.append({"period": sub_tok, "inputs": [sub_tok]})
                cur += timedelta(days=1)
        elif ip == "M":
            for tok in month_tokens_between(start_dt, end_dt):
                inputs = []
                # daily inputs for that month
                y = int(tok.split("_")[1][:4]); m = int(tok.split("_")[1][4:6])
                last = calendar.monthrange(y, m)[1]
                for d in range(1, last + 1):
                    inputs.append(f"D_{y:04d}{m:02d}{d:02d}")
                subperiods.append({"period": tok, "inputs": inputs})
        elif ip == "W":
            for tok in iso_week_tokens_between(start_dt, end_dt):
                sdt, edt = parse_token_span(tok)
                inputs = []
                cur = sdt
                while cur <= edt:
                    inputs.append(f"D_{cur.strftime(date_format)}")
                    cur += timedelta(days=1)
                subperiods.append({"period": tok, "inputs": inputs})
        else:
            # fallback: attempt to produce D_ tokens
            cur = start_dt
            while cur <= end_dt:
                sub_tok = f"D_{cur.strftime(date_format)}"
                subperiods.append({"period": sub_tok, "inputs": [sub_tok]})
                cur += timedelta(days=1)

        outputs.append({"output_period": main_token, "subperiods": subperiods, "notes": f"Range {start_str} to {end_str}"})
        if diagnostics:
            diagnostics_list.append(("range", f"Produced range {main_token} with {len(subperiods)} subperiods"))
        return (outputs, diagnostics_list) if diagnostics else outputs

    # 3) Otherwise produce list of individual tokens of the requested code across the span
    code = period_code
    if code == "D":
        toks = []
        cur = start_dt
        while cur <= end_dt:
            toks.append(f"D_{cur.strftime(date_format)}")
            cur += timedelta(days=1)
        for t in toks:
            outputs.append({"output_period": t, "subperiods": [{"period": t, "inputs": [t]}], "notes": ""})
        return (outputs, diagnostics_list) if diagnostics else outputs

    if code == "W":
        toks = iso_week_tokens_between(start_dt, end_dt)
        for t in toks:
            # map to daily inputs
            sdt, edt = parse_token_span(t)
            inputs = []
            cur = sdt
            while cur <= edt:
                inputs.append(f"D_{cur.strftime(date_format)}")
                cur += timedelta(days=1)
            outputs.append({"output_period": t, "subperiods": [{"period": t, "inputs": inputs}], "notes": ""})
        return (outputs, diagnostics_list) if diagnostics else outputs

    if code == "M":
        toks = month_tokens_between(start_dt, end_dt)
        for t in toks:
            y = int(t.split("_")[1][:4]); m = int(t.split("_")[1][4:6])
            inputs = []
            last = calendar.monthrange(y, m)[1]
            for d in range(1, last + 1):
                inputs.append(f"D_{y:04d}{m:02d}{d:02d}")
            outputs.append({"output_period": t, "subperiods": [{"period": t, "inputs": inputs}], "notes": ""})
        return (outputs, diagnostics_list) if diagnostics else outputs

    if code in ("D3", "D8"):
        toks = running_daily_windows(code, start_dt, end_dt)
        for t in toks:
            # each running window is itself composed of D_ inputs
            m = re.match(rf"^{code}_(\d{{8}})_(\d{{8}})$", t)
            if not m:
                continue
            s = datetime.strptime(m.group(1), date_format)
            e = datetime.strptime(m.group(2), date_format)
            inputs = []
            cur = s
            while cur <= e:
                inputs.append(f"D_{cur.strftime(date_format)}")
                cur += timedelta(days=1)
            outputs.append({"output_period": t, "subperiods": [{"period": t, "inputs": inputs}], "notes": ""})
        return (outputs, diagnostics_list) if diagnostics else outputs

    if code == "M3":
        toks = running_monthly_windows(code, start_dt, end_dt)
        for t in toks:
            m = re.match(rf"^{code}_(\d{{8}})_(\d{{8}})$", t)
            if not m:
                continue
            s = datetime.strptime(m.group(1), date_format)
            e = datetime.strptime(m.group(2), date_format)
            inputs = []
            cur = s
            while cur <= e:
                inputs.append(f"D_{cur.strftime(date_format)}")
                cur += timedelta(days=1)
            outputs.append({"output_period": t, "subperiods": [{"period": t, "inputs": inputs}], "notes": ""})
        return (outputs, diagnostics_list) if diagnostics else outputs

    if code == "SEA":
        toks = year_tokens_between(start_dt, end_dt, "SEA")
        for t in toks:
            # SEA_YYYY contains JFM, AMJ, JAS, OND; map to daily inputs for the year
            y = int(t.split("_")[1])
            inputs = []
            cur = datetime(y, 1, 1)
            endy = datetime(y, 12, 31)
            while cur <= endy:
                inputs.append(f"D_{cur.strftime(date_format)}")
                cur += timedelta(days=1)
            # subperiods: list the four seasons as subperiod tokens
            seasons = []
            for sm in (1, 4, 7, 10):
                em = sm + 2
                ey = y
                if em > 12:
                    em -= 12
                    ey += 1
                sdt = datetime(y, sm, 1)
                last = calendar.monthrange(ey, em)[1]
                edt = datetime(ey, em, last)
                # name like SEA_JFM_YYYY
                name = {1: "JFM", 4: "AMJ", 7: "JAS", 10: "OND"}[sm]
                sub_tok = f"SEA_{name}_{y:04d}"
                sub_inputs = []
                cur2 = sdt
                while cur2 <= edt:
                    sub_inputs.append(f"D_{cur2.strftime(date_format)}")
                    cur2 += timedelta(days=1)
                seasons.append({"period": sub_tok, "inputs": sub_inputs})
            outputs.append({"output_period": t, "subperiods": seasons, "notes": "SEA contains JFM, AMJ, JAS, OND"})
        return (outputs, diagnostics_list) if diagnostics else outputs

    if code in ("A", "Y"):
        toks = year_tokens_between(start_dt, end_dt, code)
        for t in toks:
            y = int(t.split("_")[1])
            inputs = []
            cur = datetime(y, 1, 1)
            endy = datetime(y, 12, 31)
            while cur <= endy:
                inputs.append(f"D_{cur.strftime(date_format)}")
                cur += timedelta(days=1)
            outputs.append({"output_period": t, "subperiods": [{"period": t, "inputs": inputs}], "notes": ""})
        return (outputs, diagnostics_list) if diagnostics else outputs

    # unsupported
    raise ValueError(f"Unsupported or non-generatable target code: {code}")

"""
   # 1. Parse input dates
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

    # 2. Regect if climatology code
    climatology_codes = {"SEASON", "MONTH", "WEEK", "ANNUAL", "YEAR", "DOY"}
    if code in climatology_codes:
        raise ValueError(f"{code} is a climatology code and cannot be generated by get_period_sets().")

    current = start_dt
    periods = []

    # 4. Special Non-climatology handler
        # --- ISO week single-period -----------------------------------------

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
"""
    