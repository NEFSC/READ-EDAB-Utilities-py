import re
import calendar
import datetime
from datetime import datetime, timedelta, date
from pathlib import Path
import numpy as np
from dateutil.relativedelta import relativedelta
from utilities.bootstrap.environment import bootstrap_environment
env = bootstrap_environment(verbose=False)
from utilities import get_dates

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
    }

    if help:
        return fields
        
    return {
        'D':      ('time','D',False,False,False,'P1D','time','D_YYYYMMDD','YYYYMMDD',1,'Daily'),
        'DD':     ('time','D',True, False,False,'PnD','time','DD_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of daily data'),
        'DOY':    ('doy','D',False,False,True,'','time.dt.dayofyear','DOY_DDD_YYYY_YYYY','DOY',3,'Climatological day of year'),
        'DOYS':   ('doy','DOY',False,False,True,'','time.dt.dayofyear','DOYS_YYYY_YYYY','YYYY',2,'Combined climatological days of year'),

        'D3':     ('day3','D',False,True, False,'P3D',None,'D3_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Running 3-day'),
        'DD3':    ('day3','D3',True, True, False,'P3D',None,'DD3_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of running 3-day data'),

        'D8':     ('day8','D',False,True, False,'P8D',None,'D8_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Running 8-day'),
        'DD8':    ('day8','D8',True, True, False,'P8D',None,'DD8_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of running 8-day data'),

        'W':     ('week','D',False,False,False,'P1W','time.dt.isocalendar().week','W_YYYYWW','YYYYWW',1,'Weekly (ISO week format)'),
        'WW':    ('week','W',True, False,False,'P1W','time.dt.isocalendar().week','WW_YYYYWW_YYYYWW','YYYYWW',2,'Range of weekly data'),
        'WEEK':  ('week','W',False,False,True,'','time.dt.isocalendar().week','WEEK_WW_YYYY_YYYY','YYYY',3,'Climatological ISO week'),
        'WEEKS': ('week','WEEK',False,False,True,'','time.dt.isocalendar().week','WEEKS_YYYY_YYYY','YYYY',2,'Combined climatological ISO week'),

        'M':      ('month','D',False,False,False,'P1M','time.dt.month','M_YYYYMM','YYYYMM',1,'Monthly'),
        'MM':     ('month','M',True, False,False,'P1M','time.dt.month','MM_YYYYMM_YYYYMM','YYYYMM',2,'Range of monthly data'),
        'MONTH':  ('month','M',False,False,True,'P1M','time.dt.month','MONTH_MM_YYYY_YYYY','YYYY',3,'Climatological month'),
        'MONTHS': ('month','MONTH',False,False,True,'P1M','time.dt.month','MONTHS_YYYY_YYYY','YYYY',2,'Combined climatological month'),

        'M3':   ('month3','M',False,True, False,'P3M',None,'M3_YYYYMM_YYYYMM','YYYYMM',2,'Running 3-month'),
        'MM3':  ('month3','M3',True, True, False,'P3M',None,'MM3_YYYYMM_YYYYMM','YYYYMM',2,'Range of running 3-month'),

        'JFM': ('sea','M',False,False,False,'P3M','time.dt.season','JFM_YYYY','YYYY',1,'Seasonal block JFM'),
        'AMJ': ('sea','M',False,False,False,'P3M','time.dt.season','AMJ_YYYY','YYYY',1,'Seasonal block AMJ'),
        'JAS': ('sea','M',False,False,False,'P3M','time.dt.season','JAS_YYYY','YYYY',1,'Seasonal block JAS'),
        'OND': ('sea','M',False,False,False,'P3M','time.dt.season','OND_YYYY','YYYY',1,'Seasonal block OND'),
        'SEA': ('sea','M',False,False,False,'P3M','time.dt.season','SEA_YYYY','YYYY',1,'Seasonal blocks for a single year'),

        'SJFM': ('sea','JFM',False,False,True,'P3M','time.dt.season','SJFM_YYYY_YYYY','YYYY',2,'Climatological seasonal JFM'),
        'SAMJ': ('sea','AMJ',False,False,True,'P3M','time.dt.season','SAMJ_YYYY_YYYY','YYYY',2,'Climatological seasonal AMJ'),
        'SJAS': ('sea','JAS',False,False,True,'P3M','time.dt.season','SJAS_YYYY_YYYY','YYYY',2,'Climatological seasonal JAS'),
        'SOND': ('sea','OND',False,False,True,'P3M','time.dt.season','SOND_YYYY_YYYY','YYYY',2,'Climatological seasonal OND'),
        'SEASON': ('season','SEA',False,False,True,'P3M','time.dt.season','SEASON_YYYY_YYYY','YYYY',2,'Combined climatological season'),

        'A':      ('annual','M',False,False,False,'P1Y','time.dt.year','A_YYYY','YYYY',1,'Annual (monthly inputs)'),
        'AA':     ('annual','A',True, False,False,'P1Y','time.dt.year','AA_YYYY_YYYY','YYYY',2,'Range of annual'),
        'ANNUAL': ('annual','A',False,False,True,'P1Y','time.dt.year','ANNUAL_YYYY_YYYY','YYYY',2,'Climatological annual'),

        'Y':      ('year','D',False,False,False,'P1Y','time.dt.year','Y_YYYY','YYYY',1,'Yearly (daily inputs)'),
        'YY':     ('year','Y',True, False,False,'P1Y','time.dt.year','YY_YYYY_YYYY','YYYY',2,'Range of yearly'),
        'YEAR':   ('year','Y',False,False,True,'PnY','time.dt.year','YEAR_YYYY_YYYY','YYYY',2,'Climatological year'),
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

def extract_period_code(filenames):
    """
    Extract the period code, the full period, and date segments from a filename.
    Assumes the period code is the first token in the filename. 
    
    Parameters:
        filenames (str or list of str): The filename(s) to extract the period code from.

    Returns:
        The extracted period code (str), the full period (str) and the date segments, or (None,None,None) if no valid code is found.
        If input is a string, return a single tuple: (code, full_token, segs, years)
        If input is a list, return lists of tuples: (codes[], full_tokens[], seg_lists[], years[])
    """
    
    # Normalize to list
    if isinstance(filenames, (list, tuple)):
        codes = []
        fulls = []
        seglists = []
        yearslist = []
        for fn in filenames:
            code, full, segs, years = extract_period_code(Path(fn).name)  # recursive call
            codes.append(code)
            fulls.append(full)
            seglists.append(segs)
            yearslist.append(years)
        return codes, fulls, seglists, yearslist

    # --- Single filename case ---
    filename = filenames
    name = Path(filename).name
    base = name.split("-", 1)[0]

    for code, info in get_period_info().items():
        pattern = info["regex"]
        m = pattern.match(base)
        if m:
            segs = [m.group(g) for g in m.groupdict()]
            parts = info["file_format_parts"]
            years = []
            for part, seg in zip(parts, segs):
                if part == "YYYY":
                    years.append(int(seg))
                elif part.startswith("YYYY"):  # YYYYMM, YYYYWW, YYYYMMDD
                    years.append(int(seg[:4]))
            return code, base, segs, years

    print("ERROR: extract_period_code FAILED for:", repr(filename))
    return None, None, None

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
        percode,fullperiod,segments,years = extract_period_code(base)
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

def generate_periods(period_code, start_dt, end_dt, climatology_range=None,range_by_year=False):
    """
    Generate all period tokens for a given period code and date range.

    This function dispatches to the correct span generator based on the
    period's span_type (daily, weekly, monthly, seasonal, running window,
    climatology, or range). It returns a list of formatted period tokens
    consistent with the period_map specification.

    Parameters:
    period_code : str
        The period code to generate. Supported categories include:
        • Daily blocks:
            D, DD, DOY, DOYS
        • Weekly blocks:
            W, WEEK, WW, WEEKS
        • Monthly blocks:
            M, MONTH, MM, MONTHS
        • Seasonal blocks:
            JFM, AMJ, JAS, OND, SEA, SEASON
        • Annual blocks:
            A, Y, ANNUAL, YEAR
        • Running windows:
            D3, D8, M3
        • Range periods:
            DD, DD3, DD8, WW, MM, MM3

    start_dt, end_dt : str (YYYYMMDD)
        The inclusive date range over which periods should be generated.

    climatology_range : tuple(int, int), optional
        The (start_year, end_year) range used for climatology period codes.
        If None, climatology periods are not generated.

    range_by_year : bool, default False
        If True, range-type periods (DD, DD3, DD8, WW, MM, MM3) are split
        into one token per calendar year. If False, a single full-range
        token is produced.

    Returns:
    list of str
        A list of formatted period tokens. The exact token format is
        determined by the period's file_format template in period_map.

    Helper Functions:
        split_range_by_year → Split a continuous date range into one (start, end) pair per calendar year.
            
    Notes:
    • Running windows (D3, D8, M3) extend beyond the input end date by
      the window length (2 days, 7 days, or 2 months respectively).

    • Climatology periods ignore the input date range and instead iterate
      over the full climatology_range.

    • Seasonal container code SEA produces one token per year (SEA_YYYY),
      while individual seasons (JFM, AMJ, JAS, OND) produce one token per
      season per year.

    • ISO week logic is fully ISO‑8601 compliant. Weekly range periods
      (WW) use ISO week 1 and the last ISO week of each year.
    """

    def split_range_by_year(start_dt, end_dt):
        """
        Split a continuous date range into one (start, end) pair per calendar year.

        This helper is used by range-type period codes (DD, DD3, DD8, WW, MM, MM3)
        when range_by_year=True. It preserves the original start and end dates
        within each year boundary.

        Example:
        split_range_by_year("19980101", "20011231") returns:
            [
                ("19980101", "19981231"),
                ("19990101", "19991231"),
                ("20000101", "20001231"),
                ("20010101", "20011231"),
            ]

        Notes:
        • The first segment begins at start_dt, not January 1.
        • The last segment ends at end_dt, not December 31.
        • Intermediate years span full calendar years.
        • Returned values are always YYYYMMDD strings.
            start = datetime.strptime(start_dt, "%Y%m%d")
            end   = datetime.strptime(end_dt, "%Y%m%d")
        """
        out = []
        # Ensure datetime
        if isinstance(start_dt, str):
            start_dt = datetime.strptime(start_dt, "%Y%m%d")
        if isinstance(end_dt, str):
            end_dt = datetime.strptime(end_dt, "%Y%m%d")
        
        y = start_dt.year
        while y <= end_dt.year:
            year_start = datetime(y, 1, 1)
            year_end   = datetime(y, 12, 31)
            s = max(start_dt, year_start)
            e = min(end_dt, year_end)
            out.append((s, e)) #out.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
            y += 1
        return out
    
    
    """ Main function """
    info = get_period_info(period_code)
    template = info['file_format']

    # Normalize start_dt and end_dt to datetime
    if isinstance(start_dt, str):
        start_dt = datetime.strptime(start_dt, "%Y%m%d")
    if isinstance(end_dt, str):
        end_dt = datetime.strptime(end_dt, "%Y%m%d")

    if period_code == "D":
        dates = get_dates([start_dt, end_dt],output="days",day_format="tuple")
        out = []
        for (year, month, day) in dates:
            token = (template
                        .replace("YYYY", f"{year:04d}")
                        .replace("MM", f"{month:02d}")
                        .replace("DD", f"{day:02d}"))
            out.append(token)
        return out
    
    if period_code == "W":
        dates = get_dates([start_dt, end_dt],output="weeks")
        out = []
        for (year, week) in dates:
            token = (template
                        .replace("YYYY", f"{year:04d}")
                        .replace("WW", f"{week:02d}"))
            out.append(token)
        return out
    
    if period_code == "M":
        dates = get_dates([start_dt, end_dt], output="months", period_format="tuple")
        out = []
        for (year, month) in dates:
            token = (template
                    .replace("YYYY", f"{year:04d}")
                    .replace("MM", f"{month:02d}"))
            out.append(token)
        return out

    if period_code in {"JFM","AMJ","JAS","OND"}:
        dates = get_dates([start_dt, end_dt], output="seasons", period_format="tuple")
        out = []
        for (year, season_code) in dates:
            if season_code != period_code:
                continue  # skip other seasons
            token = (template
                    .replace("YYYY", f"{year:04d}")
                    .replace("SEA", season_code)
                    .replace("SEASON", season_code))
            out.append(token)
        return out

    if period_code in {"A","Y"}:
        dates = get_dates([start_dt, end_dt], output="years", period_format="tuple")
        out = []
        for (year,) in dates:
            token = template.replace("YYYY", f"{year:04d}")
            out.append(token)
        return out

    # --- Running periods (D3, D8, M3) ---
    if period_code in {"D3","D8"}:
        window = 3 if period_code == "D3" else 8
        days = get_dates([start_dt, end_dt], output="days", day_format="datetime")
        out = []
        for s in days:
            e = s + relativedelta(days=+window-1)
            token = (template
                    .replace("YYYYMMDD", "{START}", 1)
                    .replace("YYYYMMDD", "{END}", 1)
                    .replace("{START}", s.strftime("%Y%m%d"))
                    .replace("{END}", e.strftime("%Y%m%d")))
            out.append(token)
        return out
    
    if period_code == "M3":
        months = get_dates([start_dt, end_dt],output="months",period_format="datetime")
        out = []
        for s in months:
            # Compute the end month using relativedelta
            e = s + relativedelta(months=+2)

            token = (template
                    .replace("YYYYMM", "{START}", 1)
                    .replace("YYYYMM", "{END}", 1)
                    .replace("{START}", f"{s.year:04d}{s.month:02d}")
                    .replace("{END}", f"{e.year:04d}{e.month:02d}"))
            out.append(token)
        return out

    # --- Range periods (DD, DD3, DD8, WW, MM, MM3) ---
    if period_code in {"DD", "DD3", "DD8", "WW", "MM", "MM3"}:
        # Determine ranges (full or split by year)
                
        if range_by_year:
            ranges = split_range_by_year(start_dt, end_dt)
        else:
            ranges = [(start_dt, end_dt)]

        out = []
        for (s_dt, e_dt) in ranges:
            #s_dt = datetime.strptime(s, "%Y%m%d")
            #e_dt = datetime.strptime(e, "%Y%m%d")

            # --- Extend end date for DD3/DD8 ---
            if period_code == "DD3":
                e_dt = e_dt + relativedelta(days=+2)
            elif period_code == "DD8":
                e_dt = e_dt + relativedelta(days=+7)

            # --- Convert to YYYYMMDD strings ---
            s_str = s_dt.strftime("%Y%m%d")
            e_str = e_dt.strftime("%Y%m%d")

            # --- Weekly range (WW) ---
            if period_code == "WW":
                # Determine ISO start and end week
                s_iso_year, s_iso_week, _ = s_dt.isocalendar()
                e_iso_year, e_iso_week, _ = e_dt.isocalendar()

                # -----------------------------
                # range_by_year = False
                # -----------------------------
                if not range_by_year:
                    token = f"WW_{s_iso_year:04d}{s_iso_week:02d}_{e_iso_year:04d}{e_iso_week:02d}"
                    out.append(token)
                    return out

                # -----------------------------
                # range_by_year = True
                # -----------------------------
                for year in range(s_iso_year, e_iso_year + 1):

                    # First ISO week of this ISO year
                    first_week = 1

                    # Last ISO week of this ISO year
                    last_week = date(year, 12, 28).isocalendar()[1]

                    # Clip to actual range
                    if year == s_iso_year:
                        first_week = s_iso_week
                    if year == e_iso_year:
                        last_week = e_iso_week

                    token = f"WW_{year:04d}{first_week:02d}_{year:04d}{last_week:02d}"
                    out.append(token)

                return out

            # --- Monthly range (MM/MM3) ---
            if period_code in {"MM", "MM3"}:
                s_month = f"{s_dt.year:04d}{s_dt.month:02d}"
                if period_code == "MM3":
                    e_dt = e_dt + relativedelta(months=+2)

                e_month = f"{e_dt.year:04d}{e_dt.month:02d}"
                token = (template
                        .replace("YYYYMM", "{START}", 1)
                        .replace("YYYYMM", "{END}", 1)
                        .replace("{START}", s_month)
                        .replace("{END}", e_month))
                out.append(token)
                continue

            # --- Daily range (DD, DD3, DD8) ---
            token = (template
                    .replace("YYYYMMDD", "{START}", 1)
                    .replace("YYYYMMDD", "{END}", 1)
                    .replace("{START}", s_str)
                    .replace("{END}", e_str))
            out.append(token)
        return out
    
    # --- Full Season (SEA) ---
    if period_code == "SEA":
        # Determine the year range from the input dates
        start_year = start_dt.year
        end_year   = end_dt.year

        out = []
        for y in range(start_year, end_year + 1):
            out.append(f"SEA_{y:04d}")

        return out
    
    # --- Climatology ---
    #clim_start = datetime(y1, 1, 1)
    #clim_end = datetime(y2, 12, 31)
    y1, y2 = climatology_range
    y_clim_start, y_clim_end = climatology_range
    clim_template = (template.replace("YYYY_YYYY","YYYY_YYYY2"))        

    # Actual data years
    data_start_year = start_dt.year
    data_end_year   = end_dt.year

    # Clamp data_start_year to climatology range
    y1 = max(data_start_year, y_clim_start)
    y2 = min(data_end_year,y_clim_end)

    if period_code == "DOY":
        out = []
        for doy in range(1, 366):   # always 365 days
            token = (clim_template
                    .replace("YYYY2", f"{y2:04d}")
                    .replace("YYYY",  f"{y1:04d}")
                    .replace("DDD",   f"{doy:03d}"))
            out.append(token)

        return out
    
    if period_code == "WEEK":
        out = []
        for w in range(1, 53):   # ISO weeks 1–52 (53 optional)
            token = (clim_template
                    .replace("WW", f"{w:02d}")
                    .replace("YYYY2", f"{y2:04d}")
                    .replace("YYYY",  f"{y1:04d}"))
            out.append(token)
        return out
    
    if period_code == "MONTH":
        out = []
        for m in range(1, 13):
            token = (clim_template
                    .replace("MM", f"{m:02d}")
                    .replace("YYYY2", f"{y2:04d}")
                    .replace("YYYY",  f"{y1:04d}"))
            out.append(token)
        return out
    
    # --- Full Climatology ---
    if period_code in {"DOYS","WEEKS","MONTHS","SJFM","SAMJ","SJAS","SOND","SEASON","ANNUAL","YEAR"}:
        out = (clim_template
                .replace("YYYY2", f"{y2:04d}")
                .replace("YYYY",  f"{y1:04d}"))
        return [out] 

    raise NotImplementedError(f"Unknown span type '{span_type}'.")

def map_inputs_to_period(period_code, output_period, spans, vspans, range_by_year):
    """
    Map validated input spans to a single output period token.

    This function determines which input files (represented as validated
    spans) contribute to a given output period. The mapping logic depends
    on the period_code, not the span_type, because different period codes
    with the same span_type (e.g., M, MM, M3) have different semantics.

    Parameters
    ----------
    period_code : str
        The output period code being generated (e.g., M, M3, D, D3, JFM, SEA).

    output_period : tuple or datetime
        The canonical representation of the output period, as produced by
        generate_periods(). For example:
            • (year, month) for monthly periods
            • (year, week) for weekly periods
            • (start_dt, end_dt) for range periods
            • (year, season_code) for seasonal periods

    spans : list of dict
        Validated input spans, each containing:
            {
                "code": input_period_code,
                "start": datetime,
                "end": datetime,
                "token": original_token_string
            }

    Returns
    -------
    list of str
        A list of input tokens that contribute to the output period.
    """

    # ------------------------------------------------------------
    # Normalize output_period → date or (date,date)
    # ------------------------------------------------------------
    def normalize_op(op):
        if isinstance(op, tuple):
            out = []
            for x in op:
                if hasattr(x, "date"):
                    out.append(x.date())
                elif isinstance(x, str) and len(x) == 8 and x.isdigit():
                    out.append(date(int(x[:4]), int(x[4:6]), int(x[6:8])))
                else:
                    out.append(x)
            return tuple(out)
        else:
            if hasattr(op, "date"):
                return op.date()
            elif isinstance(op, str) and len(op) == 8 and op.isdigit():
                return date(int(op[:4]), int(op[4:6]), int(op[6:8]))
            return op

    def season_bounds(year, season_code):
        """
        Return (start_date, end_date) for a climatological 3‑month season.
        Seasons follow standard meteorological definitions:
            JFM = Jan–Mar
            AMJ = Apr–Jun
            JAS = Jul–Sep
            OND = Oct–Dec
        """
        if season_code == "JFM":
            return date(year, 1, 1), date(year, 3, 31)

        if season_code == "AMJ":
            return date(year, 4, 1), date(year, 6, 30)

        if season_code == "JAS":
            return date(year, 7, 1), date(year, 9, 30)

        if season_code == "OND":
            return date(year, 10, 1), date(year, 12, 31)

        raise ValueError(f"Unknown season code: {season_code}")

    def climatology_matches(period_code, output_period, span):
        in_code = span["code"]
        token   = span["token"]

        if in_code == "DOYS":
            return True

        _, _, segs, _ = extract_period_code(token)
        in_key = segs[0]
        out_key = str(output_period[0])
        return in_key == out_key

    output_period = normalize_op(output_period)

    # Shorthand
    starts = vspans["starts"]
    ends   = vspans["ends"]
    tokens = vspans["tokens"]

    # ------------------------------------------------------------
    # 1. DAILY (D)
    # ------------------------------------------------------------
    if period_code == "D":
        day = output_period.toordinal()
        mask = (starts <= day) & (ends >= day)
        return tokens[mask].tolist()

    # ------------------------------------------------------------
    # 2. RUNNING WINDOW AND RANGE PERIODS (DD, D3, DD3, D8, DD8)
    # ------------------------------------------------------------
    if period_code in {"DD", "D3", "DD3", "D8", "DD8"}:
        start_day, end_day = output_period
        s = start_day.toordinal()
        e = end_day.toordinal()

        mask = (vspans["starts"] <= e) & (vspans["ends"] >= s)

        # Year clamp applies ONLY to DD3 and DD8
        if period_code in {"DD3", "DD8"} and range_by_year:
            target_year = start_day.year
            mask &= (vspans["years"] == target_year)

        return vspans["tokens"][mask].tolist()

    # ------------------------------------------------------------
    # 3. WEEKLY (W)
    # ------------------------------------------------------------
    if period_code == "W":
        iso_year, iso_week = output_period
        mask = (vspans["iso_years"] == iso_year) & (vspans["iso_weeks"] == iso_week)
        return vspans["tokens"][mask].tolist()
    
    # ------------------------------------------------------------
    # 5. WEEK RANGE (WW)
    # ------------------------------------------------------------
    if period_code == "WW":
        (sy, sw), (ey, ew) = output_period
        span_keys = vspans["iso_years"] * 100 + vspans["iso_weeks"]         # Convert ISO year/week to a comparable integer key
        start_key = sy * 100 + sw
        end_key   = ey * 100 + ew
        mask = (span_keys >= start_key) & (span_keys <= end_key)
        return vspans["tokens"][mask].tolist()    
    
    # ------------------------------------------------------------
    # 6. MONTHLY (M)
    # ------------------------------------------------------------
    if period_code == "M":
        year, month = output_period
        mask = (vspans["years"] == year) & (vspans["months"] == month)
        return vspans["tokens"][mask].tolist()

    # ------------------------------------------------------------
    # 7. MONTH RANGE (MM, MM3)
    # ------------------------------------------------------------
    if period_code in {"MM", "MM3"}:
        (sy, sm), (ey, em) = output_period
        span_keys = vspans["years"] * 100 + vspans["months"]
        start_key = sy * 100 + sm
        end_key   = ey * 100 + em

        mask = (span_keys >= start_key) & (span_keys <= end_key)

        # Year clamp applies ONLY to MM3
        if period_code == "MM3" and range_by_year:
            mask &= (vspans["years"] == sy)

        return vspans["tokens"][mask].tolist()
    
    # ------------------------------------------------------------
    # 8. SEASONS (JFM, AMJ, JAS, OND)
    # ------------------------------------------------------------
    if period_code in {"JFM", "AMJ", "JAS", "OND"}:
        year, season_code = output_period
        start_dt, end_dt = season_bounds(year, season_code)
        s = start_dt.toordinal()
        e = end_dt.toordinal()
        mask = (vspans["starts"] <= e) & (vspans["ends"] >= s)
        return vspans["tokens"][mask].tolist()

    # ------------------------------------------------------------
    # 9. SEASON CONTAINER (SEA)
    # ------------------------------------------------------------
    if period_code == "SEA":
        year = output_period
        s = date(year, 1, 1).toordinal()
        e = date(year, 12, 31).toordinal()
        mask = (vspans["starts"] <= e) & (vspans["ends"] >= s)
        return vspans["tokens"][mask].tolist()

    # ------------------------------------------------------------
    # 10. ANNUAL (A, Y)
    # ------------------------------------------------------------
    if period_code in {"A", "Y"}:
        year = output_period
        s = date(year, 1, 1).toordinal()
        e = date(year, 12, 31).toordinal()
        mask = (vspans["starts"] <= e) & (vspans["ends"] >= s)

        # Extract the tokens for this year
        tokens_for_year = vspans["tokens"][mask]

        if period_code == "A":
            # Annual periods require exactly 12 monthly inputs
            monthly_mask = [
                (tok.startswith("M_") and tok[2:6] == f"{year}")
                for tok in vspans["tokens"]
            ]

            # Extract months for this year
            months = [tok[2:8] for tok in vspans["tokens"][monthly_mask]]

            # Require exactly 12 unique YYYYMM values
            if len(set(months)) != 12:
                return []
            return tokens_for_year.tolist()
        
        # If there are *zero* daily inputs, exclude the year
        if len(tokens_for_year) == 0:
            return []

        # Otherwise include it (even if some days are missing)
        return tokens_for_year.tolist()

    # ------------------------------------------------------------
    # CLIMATOLOGY (DOY, DOYS, WEEK, WEEKS, MONTH, MONTHS, SEASON)
    # ------------------------------------------------------------
    if period_code == "DOY":
        doy, y1, y2 = output_period   # e.g., (365, 1997, 2020)
        mask = (
            (vspans["doys"] == doy) &
            (vspans["years"] >= y1) &
            (vspans["years"] <= y2)
        )
        return vspans["tokens"][mask].tolist()

    if period_code == "DOYS":
        y1, y2 = output_period 
        mask = [
            tok.startswith("DOY_") and
            y1 <= int(tok.split("_")[2]) <= y2
            for tok in vspans["tokens"]
        ]
        return vspans["tokens"][mask].tolist()

    if period_code == "WEEK":
        week, y1, y2 = output_period
        mask = (
            (vspans["iso_weeks"] == week) &
            (vspans["iso_years"] >= y1) &
            (vspans["iso_years"] <= y2)
        )
        return vspans["tokens"][mask].tolist()
    
    if period_code == "WEEKS":
        y1, y2 = output_period
        mask = (vspans["iso_years"] >= y1) & (vspans["iso_years"] <= y2)
        return vspans["tokens"][mask].tolist()

    if period_code == "MONTH":
        month, y1, y2 = output_period
        mask = (
            (vspans["months"] == month) &
            (vspans["years"] >= y1) &
            (vspans["years"] <= y2)
        )
        return vspans["tokens"][mask].tolist()
    
    if period_code == "MONTHS":
        y1, y2 = output_period
        mask = (vspans["years"] >= y1) & (vspans["years"] <= y2)
        return vspans["tokens"][mask].tolist()

    if period_code == "SEASON":
        season_code = output_period[0]
        # Climatology seasons ignore year → use dummy year
        start_dt, end_dt = season_bounds(2001, season_code)
        s = start_dt.toordinal()
        e = end_dt.toordinal()
        mask = (vspans["starts"] <= e) & (vspans["ends"] >= s)
        return vspans["tokens"][mask].tolist()
    
    if period_code == "ANNUAL":
        start_year, end_year = output_period  
        mask = [
            tok.startswith("A_") and start_year <= int(tok[2:6]) <= end_year
            for tok in vspans["tokens"]
        ]
        return vspans["tokens"][mask].tolist()
    
    if period_code == "YEAR":
        start_year, end_year = output_period  
        mask = [
            tok.startswith("Y_") and start_year <= int(tok[2:6]) <= end_year
            for tok in vspans["tokens"]
        ]
        return vspans["tokens"][mask].tolist()


def get_period_sets(
        period_code, 
        files=None,
        start=None, 
        end=None, 
        dates=None, 
        date_format="%Y%m%d",
        range_by_year=False,
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
        - range_by_year (bool): to create range periods (DD, DD3, DD8, WW, MM, MM3) in yearly chunks.
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
    
    def canonical_output_period(period_code, token):
        """
        Convert a formatted output token into the canonical structure needed
        by map_inputs_to_period(), using existing helpers only.
        """
        
        code, full_token, segs, years = extract_period_code(token)
        if code is None:
            print("ERROR: canonical_output_period rejecting token:", repr(token))
            raise ValueError(f"Invalid period token: {token}")
        # 1. Get the canonical date range for ALL period codes
        sdt, edt = get_period_dates(full_token)

        if sdt == "NA" or edt == "NA":
            return ("NA", "NA")
        # Always convert to datetime objects
        start_dt = sdt if isinstance(sdt, datetime) else to_dt(sdt)
        end_dt   = edt if isinstance(edt, datetime) else to_dt(edt)

        # 2. Convert to the structure expected by mapping helpers
        if period_code == "D":
            return start_dt.date()
        
        if period_code in { "DD", "DD3", "DD8", "D3", "D8","M3"}:
            return (start_dt, end_dt)
        
        if period_code == "DOY":
            _, doy_str, y1_str, y2_str = token.split("_")
            return (int(doy_str), int(y1_str), int(y2_str))

        if period_code == "DOYS":
            _, y1_str, y2_str = token.split("_")
            return (int(y1_str), int(y2_str))

        if period_code == "W":
            y, w, _ = start_dt.isocalendar()
            return (y, w)

        if period_code == "WW":
            sy, sw, _ = start_dt.isocalendar()
            ey, ew, _ = end_dt.isocalendar()
            return ((sy, sw), (ey, ew))
        
        if period_code == "WEEK":
            _, week_str, y1_str, y2_str = token.split("_")
            return (int(week_str), int(y1_str), int(y2_str))
        
        if period_code == "M":
            return (start_dt.year, start_dt.month)

        if period_code in {"MM", "MM3"}:
            return ((start_dt.year, start_dt.month),
                    (end_dt.year, end_dt.month))
        
        if period_code == "MONTH":
            _, month_str, y1_str, y2_str = token.split("_")
            return (int(month_str), int(y1_str), int(y2_str))

        if period_code in {"JFM", "AMJ", "JAS", "OND"}:
            return (start_dt.year, period_code)
        
        if period_code in {"SEA", "A", "Y"}:
            return start_dt.year
        
        if period_code in {"WEEKS","MONTHS","SJFM","SAMJ","SJAS","SOND","SEASON"}:
            _, y1_str, y2_str = token.split("_")
            return (int(y1_str), int(y2_str))
        
        if period_code == "ANNUAL":
            _, y1_str, y2_str = token.split("_")
            return (int(y1_str), int(y2_str))

        if period_code == "YEAR":
            _, y1_str, y2_str = token.split("_")
            return (int(y1_str), int(y2_str))

        return token

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

    """ STEP 2: Determine input type and extract input date span."""
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
            #print(f"token={tok}, results={results}")
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

    # Normalize spans into dicts
    normalized_spans = []
    for (sdt, edt, tok) in spans:
        normalized_spans.append({
            "code": input_prefix,
            "start": to_dt(sdt),
            "end":   to_dt(edt),
            "token": tok
        })

    # Vectorize spans
    span_starts = np.array([s["start"].toordinal() for s in normalized_spans])
    span_ends   = np.array([s["end"].toordinal()   for s in normalized_spans])
    span_tokens = np.array([s["token"]             for s in normalized_spans])

    # Extract start dates for categorical keys
    span_start_dates = np.array([s["start"] for s in normalized_spans])

    # ISO week keys
    span_iso_years  = np.array([d.isocalendar()[0] for d in span_start_dates])
    span_iso_weeks  = np.array([d.isocalendar()[1] for d in span_start_dates])

    # Month keys
    span_years  = np.array([d.year  for d in span_start_dates])
    span_months = np.array([d.month for d in span_start_dates])

    # DOY keys
    span_doys = np.array([d.timetuple().tm_yday for d in span_start_dates],dtype=int)

    # Package into vectorized spans
    vectorized_spans = {
        "starts": span_starts,
        "ends": span_ends,
        "tokens": span_tokens,
        "iso_years": span_iso_years,
        "iso_weeks": span_iso_weeks,
        "years": span_years,
        "months": span_months,
        "doys": span_doys,
    }
    
    # Clamp data_start_year to climatology range
    if target_info['is_climatology']: 
        y_clim_start, y_clim_end = climatology_range
        codes, fulls, seglists, yearslists = extract_period_code([tok for _,_,tok in spans])
        all_years = [y for years in yearslists for y in years]
        data_start_year = min(all_years)
        data_end_year   = max(all_years)        
        y1 = max([data_start_year, y_clim_start])
        y2 = min([data_end_year,y_clim_end])
        start_dt = f"{y1:04d}0101"
        end_dt   = f"{y2:04d}1231"
    
     # Generate output periods
    periods = generate_periods(target_code,start_dt,end_dt,climatology_range=climatology_range,range_by_year=range_by_year)
    period_map_out = {}

    if target_code in {"SEA","SEASON"}:
        
        # Determine the year from the input tokens
        _, _, _, yearslists = extract_period_code([tok for _,_,tok in spans])
        all_years = [y for years in yearslists for y in years]

        start_year = min(all_years)
        end_year   = max(all_years)
        
        period_map = {}
        if target_code == "SEA":
            seasonal_codes = ["JFM", "AMJ", "JAS", "OND"]
        
            # Map each seasonal token to its monthly inputs
            for year in range(start_year, end_year + 1):
                seasonal_tokens = [f"{s}_{year:04d}" for s in seasonal_codes]

                # Map each seasonal block to its monthly inputs
                seasonal_map = {}
                for s_tok in seasonal_tokens:
                    s_code, _, _, _ = extract_period_code(s_tok)
                    output_period = canonical_output_period(s_code, s_tok)
                    inputs = map_inputs_to_period(
                        s_code,
                        output_period,
                        spans,
                        vectorized_spans,
                        range_by_year
                    )

                    seasonal_map[s_tok] = {"inputs": inputs}

                period_map[f"SEA_{year:04d}"] = seasonal_map

            return {
                "target_code": "SEA",
                "period_map": period_map
            }
        elif target_code == "SEASON":
            seasonal_clim_codes = ["SJFM", "SAMJ", "SJAS", "SOND"]
            # Build each seasonal climatology block
            for season in seasonal_clim_codes:
                s_tok = f"{season}_{y1:04d}_{y2:04d}"

                # Convert token → canonical output period
                s_code, _, _, _ = extract_period_code(s_tok)
                output_period = canonical_output_period(s_code, s_tok)

                # Map inputs to this seasonal climatology block
                inputs = map_inputs_to_period(
                    s_code,
                    output_period,
                    spans,
                    vectorized_spans,
                    range_by_year
                )

                period_map[s_tok] = {"inputs": inputs}

            # Wrap into the SEASON container
            return {
                "target_code": "SEASON",
                "period_map": {
                    f"SEASON_{y1:04d}_{y2:04d}": period_map
                }
            }


    for p in periods:        
        # Convert token → canonical representation
        output_period = canonical_output_period(target_code, p)
        inputs = map_inputs_to_period(target_code, output_period, normalized_spans, vectorized_spans,range_by_year)
        period_map_out[p] = {"inputs": inputs}
        
    return {
        "target_code": target_code,
        "expected_input_code": expected_input_code,
        "input_items": input_items,
       # "input_prefix": input_prefix,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "period_map": period_map_out,   
        "diagnostics": diagnostics_list,
    }

    
    
    