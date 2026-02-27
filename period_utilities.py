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
    - get_period_dates: Extracts the period encoded in a filename (e.g. 'M_202204' or 'DD3_20220401_20220403') and returns the start and end dates of that period.
    - get_period_sets: Generate period tokens (e.g. M_YYYYMM, W_YYYYWW) for a given date range or list of dates.
    

Helper Functions:
    - build_regex (in period_info): Build a strict anchored regex based on file_format tokens.
    - build_info (in period_info): Build the full info dictionary for a given period code, including the compiled regex.
    - _fail (in check_period): Helper to return a standardized failure response.
    - seg_value_for (in get_period_dates): Map token -> segment value (1-based occurrence).
    - parse_yyyymmdd (in get_period_dates): Parses a date segment (YYYYMMDD) into a date object.
    - parse_yyyymm (in get_period_dates): Parses a date segment (YYYYMM) into start and end date objects.
    - parse_yyyy (in get_period_dates): Parses a date segment (YYYY) into start and end date objects.
    - parse_iso_week_to_range (in get_period_dates): Parses ISO week codes into start and end date objects.
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
        "10) description": "human-readable description of the type of data associate with the period"
    }

    if help:
        return fields
        
    return {
        'D':      ('time','D',False,False,False,'P1D','time','D_YYYYMMDD','YYYYMMDD',1,'Daily'),
        'DD':     ('time','D',True, False,False,'PnD','time','DD_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of daily data'),
        'DOY':    ('doy','D',False,False,True,'','time.dt.dayofyear','DOY_DDD_YYYY_YYYY','DOY',3,'Climatological day of year'),
        'DOYS':   ('doy','D',False,False,True,'','time.dt.dayofyear','DOYS_YYYY_YYYY','YYYY',2,'Combined climatological days of year'),

        'D3':     ('day3','D',False,True, False,'P3D',None,'D3_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Running 3-day'),
        'DD3':    ('day3','D',True, True, False,'P3D',None,'DD3_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of running 3-day data'),

        'D8':     ('day8','D',False,True, False,'P8D',None,'D8_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Running 8-day'),
        'DD8':    ('day8','D',True, True, False,'P8D',None,'DD8_YYYYMMDD_YYYYMMDD','YYYYMMDD',2,'Range of running 8-day data'),

        'W':      ('week','D',False,False,False,'P1W','time.dt.isocalendar().week','W_YYYYWW','YYYYWW',1,'Weekly (ISO week format)'),
        'WW':     ('week','D',True, False,False,'P1W','time.dt.isocalendar().week','WW_YYYYWW_YYYYWW','YYYYWW',2,'Range of weekly data (ISO week format)'),
        'WEEK':   ('week','W',False,False,True,'','time.dt.isocalendar().week','WEEK_WW_YYYY_YYYY','YYYY',3,'Climatological ISO week'),
        'WEEKS':  ('week','W',False,False,True,'','time.dt.isocalendar().week','WEEKS_YYYY_YYYY','YYYY',2,'Combined climatological ISO week'),

        'M':      ('month','D',False,False,False,'P1M','time.dt.month','M_YYYYMM','YYYYMM',1,'Monthly'),
        'MM':     ('month','D',True, False,False,'P1M','time.dt.month','MM_YYYYMM_YYYYMM','YYYYMM',2,'Range of monthly data'),
        'MONTH':  ('month','M',False,False,True,'P1M','time.dt.month','MONTH_MM_YYYY_YYYY','YYYY',3,'Climatological month'),
        'MONTHS': ('month','M',False,False,True,'P1M','time.dt.month','MONTHS_YYYY_YYYY','YYYY',2,'Combined climatological month'),

        'M3':     ('month3','M',False,True, False,'P3M',None,'M3_YYYYMM_YYYYMM','YYYYMM',2,'Running 3-month'),
        'MM3':    ('month3','M',True, True, False,'P3M',None,'MM3_YYYYMM_YYYYMM','YYYYMM',2,'Range of running 3-month'),

        'JFM': ('sea','M',False,False,False,'P3M','time.dt.season','JFM_YYYY','YYYY',1,'Seasonal block JFM'),
        'AMJ': ('sea','M',False,False,False,'P3M','time.dt.season','AMJ_YYYY','YYYY',1,'Seasonal block AMJ'),
        'JAS': ('sea','M',False,False,False,'P3M','time.dt.season','JAS_YYYY','YYYY',1,'Seasonal block JAS'),
        'OND': ('sea','M',False,False,False,'P3M','time.dt.season','OND_YYYY','YYYY',1,'Seasonal block OND'),
        'SEA': ('sea','M',False,False,False,'P3M','time.dt.season','SEA_YYYY','YYYY',1,'Seasonal blocks for a single year (JFM/AMJ/JAS/OND)'),

        'SJFM': ('sea','SEAJFM',False,False,True,'P3M','time.dt.season','SEASONJFM_YYYY_YYYY','YYYY',2,'Climatological seasonal JFM'),
        'SAMJ': ('sea','SEAAMJ',False,False,True,'P3M','time.dt.season','SEASONAMJ_YYYY_YYYY','YYYY',2,'Climatological seasonal AMJ'),
        'SJAS': ('sea','SEAJAS',False,False,True,'P3M','time.dt.season','SEASONJAS_YYYY_YYYY','YYYY',2,'Climatological seasonal JAS'),
        'SOND': ('sea','SEAOND',False,False,True,'P3M','time.dt.season','SEASONOND_YYYY_YYYY','YYYY',2,'Climatological seasonal OND'),
        'SEASON': ('season','SEA',False,False,True,'P3M','time.dt.season','SEASON_YYYY_YYYY','YYYY',2,'Combined climatological season'),

        'A':      ('annual','M',False,False,False,'P1Y','time.dt.year','A_YYYY','YYYY',1,'Annual (monthly inputs)'),
        'AA':     ('annual','M',True, False,False,'P1Y','time.dt.year','AA_YYYY_YYYY','YYYY',2,'Range of annual (monthly inputs)'),
        'ANNUAL': ('annual','A',False,False,True,'P1Y','time.dt.year','ANNUAL_YYYY_YYYY','YYYY',2,'Climatological annual'),

        'Y':      ('year','D',False,False,False,'P1Y','time.dt.year','Y_YYYY','YYYY',1,'Yearly (daily inputs)'),
        'YY':     ('year','D',True, False,False,'P1Y','time.dt.year','YY_YYYY_YYYY','YYYY',2,'Range of yearly (daily inputs)'),
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
            description
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

from pathlib import Path
from datetime import date, timedelta

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
      - parse_yyyy (in get_period_dates): Parses a date segment (YYYY) into start and end date objects.
      - parse_iso_week_to_range (in get_period_dates): Parses ISO week codes into start and end date objects.
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

    # helper function: parse YYYY -> (start_date, end_date)
    def parse_yyyy(s):
        try:
            y = int(s)
            return date(y, 1, 1), date(y, 12, 31)
        except Exception:
            raise ValueError("Invalid YYYY")

    # helper function: parse YYYYWW (year+week) or WW (week only)
    def parse_iso_week_to_range(year, week):
        # returns (monday_date, sunday_date) for given ISO year/week
        try:
            start = date.fromisocalendar(year, week, 1)
            end = date.fromisocalendar(year, week, 7)
            return start, end
        except Exception:
            raise ValueError("Invalid ISO week for the specified year")

    # helper function (convenience): get token index list for a token
    def token_indices(token):
        return [i for i, t in enumerate(parts) if t == token]
 
    # 1. Cache token and period metadata and precompile regexes
    TOKEN_LENGTHS = get_token_lengths()   # once at module import
    period_map = get_period_info()
    codes_sorted = sorted(period_map.keys(), key=lambda x: -len(x))

    if isinstance(files, str):
        files = [files]

    results = []
    errors = []
    placeholder = placeholder if 'placeholder' in globals() else "NA"

    # 2. Process each file
    for fname in files:
        base = Path(fname).stem

        # Helper functgion: fail() bound to this fname so diagnostics record the correct file
        def fail(msg):
            results.append((placeholder, placeholder))
            if diagnostics:
                errors.append((fname, msg))
            return True

        # 3. Check that the the input period is valid
        res = check_period(base, diagnostics=diagnostics, placeholder=placeholder)
        if not res.get("ok"):
            # record the error message if diagnostics requested
            if diagnostics:
                errors.append((fname, res.get("error", "Invalid format")))
            # results already not appended by check_period; append placeholder for consistency
            results.append((placeholder, placeholder))
            continue

        # 4. Unpack normalized pieces for semantic handling
        matched_code = res["matched_code"]
        #info = res["info"]
        parts = res["parts"]
        segments = res["segments"]
        #match_obj = res["match_obj"]


        # 4. Period specific handlers
        
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
            # these are single-year seasonal blocks: file_format usually YYYY
            try:
                sy = int(seg_value_for("YYYY", 1))
            except ValueError:
                if fail(f"{matched_code} Invalid format for this period code"):
                    continue
            # map season to month ranges
            season_map = {
                "JFM": (1,3),
                "AMJ": (4,6),
                "JAS": (7,9),
                "OND": (10,12),
                "SEA": (1,12),  # SEA for full-year blocks (if you want different semantics adjust)
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

    return (results, errors) if diagnostics else results
    

def get_period_sets(
        period_code, 
        start=None, 
        end=None, 
        dates=None, 
        period_tokens=None,
        files=None, 
        date_format="%Y%m%d",
        running=True,
        climatology_range=(1991,2020),
        diagnostics=False
):
    """
    Generate output period tokens (e.g. M_YYYYMM, W_YYYYWW) and input mappings for a given date range, list of dates, or list of periods.

    Parameters
        - period_code (str): Period code to generate (e.g. 'M', 'W', 'A', 'Y').
        - start (str or datetime, optional): Start date in YYYYMMDD format or datetime object.
        - end (str or datetime, optional): End date in YYYYMMDD format or datetime object.
        - dates (list[str|datetime], optional): list of dates to infer range from.
        - period_tokens (list[str], optional): list of existing period tokens (e.g., ['M_200001', ...]).
        - files (list[str], optional): list of filenames that start with period tokens (e.g., 'M_202201-chl.nc').
        - date_format (str): format used for input date strings and for token construction.
        - running (bool): for running-window codes (D3, M3), whether to produce sliding windows.
        - climatology_range (tuple): default climatology years (start_year, end_year).
        - diagnostics (bool): if True, return (outputs, diagnostics_list).

    Returns
        - list of str: List of period tokens like ['M_202005', 'M_202006', ...]
    
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
    
    def token_prefix_from_file(fname):
        """Return the token prefix and the matched token (if any) using get_period_info() regexes."""
        base = Path(fname).name
        for pfx, info in get_period_info().items():
            m = info["regex"].match(base)
            if m:
                return pfx, m.groupdict(), base
        return None, None, base
    
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


    # 1. Input normalization
    diagnostics_list = []
    input_prefix = None
    input_items = []  # canonical list of inputs (dates -> D_ tokens; period_tokens -> tokens; files -> filenames)

    # 2. Get start and end date ranges (from either the input period tokens, the input files, or the start/end/dates )
    
    # 2A. Look for input period tokens
    if period_tokens:
        # parse tokens to spans and set input_prefix
        spans = []
        prefixes = set()
        for tok in period_tokens:
            try:
                sdt, edt = parse_token_span(tok)
            except Exception as e:
                raise ValueError(f"Failed to parse input token '{tok}': {e}")
            spans.append((sdt, edt, tok))
            prefixes.add(tok.split("_", 1)[0])
            input_items.append(tok)
        if len(prefixes) > 1:
            raise ValueError(f"Mixed input period prefixes not supported: {sorted(prefixes)}")
        input_prefix = next(iter(prefixes))
        start_dt = min(s[0] for s in spans)
        end_dt = max(s[1] for s in spans)
    
    #2B. Look for input files
    elif files:
        # files should start with period tokens; map files to tokens
        spans = []
        prefixes = set()
        for f in files:
            pfx, gd, base = token_prefix_from_file(f)
            if not pfx:
                diagnostics_list.append((f, "No period token found at start of filename"))
                continue
            prefixes.add(pfx)
            # reconstruct token from matched groups (use same formatting as regex expects)
            info = get_period_info()[pfx]
            # build token string from groups
            if info["n_date_segments"] == 1:
                token_str = f"{pfx}_{gd['start']}"
            else:
                token_str = f"{pfx}_{gd['start']}_{gd['end']}"
            try:
                sdt, edt = parse_token_span(token_str)
            except Exception as e:
                diagnostics_list.append((f, f"Failed to parse token from filename: {e}"))
                continue
            spans.append((sdt, edt, token_str, f))
            input_items.append(f)  # keep file-level mapping
        if not spans:
            raise ValueError("No valid period tokens found in provided files.")
        if len(prefixes) > 1:
            raise ValueError(f"Mixed input period prefixes not supported for files: {sorted(prefixes)}")
        input_prefix = next(iter(prefixes))
        start_dt = min(s[0] for s in spans)
        end_dt = max(s[1] for s in spans)
    #2C. Otherwise look for a list of dates or start and end dates
    else:
        # 2Ci List of dates
        if dates:
            parsed = [d if isinstance(d, datetime) else to_dt(d) for d in dates]
            start_dt = min(parsed)
            end_dt = max(parsed)
            # convert dates to D_ tokens for mapping
            cur = start_dt
            while cur <= end_dt:
                input_items.append(f"D_{cur.strftime(date_format)}")
                cur += timedelta(days=1)
            input_prefix = "D"
        # 2Cii Start and end dates
        elif start and end:
            start_dt = start if isinstance(start, datetime) else to_dt(start)
            end_dt = end if isinstance(end, datetime) else to_dt(end)
            if start_dt > end_dt:
                raise ValueError("Start date must be before or equal to end date.")
            cur = start_dt
            while cur <= end_dt:
                input_items.append(f"D_{cur.strftime(date_format)}")
                cur += timedelta(days=1)
            input_prefix = "D"
        else:
            raise ValueError("Provide either period_tokens, files, or start/end/dates.")

    
    # 3. Validate target code
    info_map = get_period_info()
    if period_code not in info_map:
        raise ValueError(f"Unsupported target period code: {period_code}")
    target_info = info_map[period_code]

    # If inputs were raw dates, only allow codes whose input_period_code == 'D'
    if input_prefix is None:
        input_prefix = "D"
    if input_prefix == "D" and target_info["input_period_code"] != "D":
        raise ValueError(f"Cannot generate '{code}' from raw dates; its input_period_code is '{target_info['input_period_code']}' (only 'D' allowed).")

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
    