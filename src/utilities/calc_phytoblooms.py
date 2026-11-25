import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.integrate import trapezoid
import os
import sys
import traceback
from datetime import datetime

from utilities import (
    bootstrap_environment,
    get_extracted_files,
    generate_logfile_path,
    get_current_utc_timestamp,
    read_nccsv,
)

env = bootstrap_environment(verbose=False)

"""
Purpose:
    CALC_PHTYOBLOOMS calculated the phytoplankton bloom metrics at both the extracted time series and gridded data levels. 
    Phytoplankton blooms are determined using a combination of a threshold and rate of change criteria. 
    Bloom metrics include initiation, peak day, termination, duration and total bloom biomass.

Main Functions:
    - 
    
    

Helper Functions:
    
    
Copywrite: 
    Copyright (C) 2026, Department of Commerce, National Oceanic and Atmospheric Administration, National Marine Fisheries Service,
    Northeast Fisheries Science Center, Narragansett Laboratory.
    This software may be used, copied, or redistributed as long as it is not sold and this copyright notice is reproduced on each copy made.
    This routine is provided AS IS without any express or implied warranties whatsoever.

Author:
    This program was written on November 24, 2026 by Kimberly J. W. Hyde, Northeast Fisheries Science Center | NOAA Fisheries | U.S. Department of Commerce, 28 Tarzwell Dr, Narragansett, RI 02882
  
Modification History
    Nov 24, 2026 - KJWH: Initial code written
"""

def _resolve_single_file(file_input, label="file"):
    """Accepts a single path string or a 1-element list/tuple of paths."""
    if isinstance(file_input, (list, tuple)):
        if len(file_input) == 0:
            raise FileNotFoundError(f"No {label} found (empty list passed).")
        if len(file_input) > 1:
            raise ValueError(f"Expected 1 {label}, but received {len(file_input)}: {file_input}")
        return file_input[0]
    return file_input


def classify_bloom_season(doy):
    """Classifies a bloom by its peak Day-of-Year (DOY 1–366)."""
    if 1 <= doy <= 59:
        return 'Winter'
    elif 60 <= doy <= 152:
        return 'Spring'
    elif 153 <= doy <= 244:
        return 'Summer'
    else:
        return 'Fall'


def integrate_chl(dates, chl_vals, threshold=None):
    """
    Computes trapezoidal integrated chlorophyll (mg m^-3 * days) over valid points.
    If `threshold` is provided, values below the threshold are zeroed out prior to integration.
    """
    mask = ~np.isnan(chl_vals)
    if mask.sum() < 2:
        return 0.0
    t_days = (dates[mask] - dates[mask][0]) / np.timedelta64(1, 'D')
    y = chl_vals[mask].copy()
    if threshold is not None:
        y = np.where(y >= threshold, y, 0.0)
    return float(trapezoid(y, t_days))

def smooth_chl_series(series, method='savgol', window=15, poly=3, deriv=0, frac=0.00117):
    """
    Purpose:
        Smooths a 1D chlorophyll pandas Series while preserving original NaNs.
        Supports 'savgol' (default: 15-day window, polyorder 3) and 'lowess'.

    Input Parameters:
        verbose (bool): If True, prints progress updates.
        

    Returns:
        
    """

    y = series.astype(float).copy()
    orig_nan_mask = y.isna()

    if (~orig_nan_mask).sum() < max(window, 3):
        return pd.Series(np.nan, index=series.index)

    method_lower = method.lower()
    if method_lower == 'savgol':
        # SavGol requires no NaNs; linearly interpolate gaps then restore original NaNs
        y_filled = y.interpolate(method='linear').bfill().ffill()
        smoothed = savgol_filter(y_filled.values, window_length=window, polyorder=poly, deriv=deriv)
        out = pd.Series(smoothed, index=series.index)
        out[orig_nan_mask] = np.nan
        return out

    elif method_lower == 'lowess':
        valid_mask = ~orig_nan_mask
        smoothed = lowess(
            endog=y[valid_mask].values,
            exog=np.arange(len(y))[valid_mask],
            frac=frac,
            return_sorted=False
        )
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out[valid_mask] = smoothed
        return out

def detect_threshold_blooms(chl_file, clim_file, var_name='CHL_Median', thld=0.1,
                            smooth_method='savgol', window=15, poly=3, frac=0.00117,
                            save_output=True, **roc_kwargs):
    """
    Reads extracted chlorophyll timeseries and ANNUAL climatology NCCSVs, smooths the
    target CHL column and computes its daily rate of change (ROC) per subarea, flags
    threshold exceedances ('is_bloom'), runs ROC bloom event detection ('bloom_id'),
    and optionally writes the updated daily NCCSV.
    """
    chl_file = _resolve_single_file(chl_file, label="chlorophyll file")
    clim_file = _resolve_single_file(clim_file, label="climatology file")

    meta_lines, df = read_nccsv(chl_file)
    _, df_clim = read_nccsv(clim_file)

    if var_name not in df.columns:
        raise ValueError(f"Column '{var_name}' not found in {chl_file}.")
    if var_name not in df_clim.columns:
        raise ValueError(f"Column '{var_name}' not found in {clim_file}.")

    # Sort chronologically per subarea before smoothing
    df = df.sort_values(['subarea', 'Date']).reset_index(drop=True)

    # 1. Build BOTH median and threshold lookups (ROC trough snapping needs the raw median)
    clim_med_lookup = df_clim.groupby('subarea')[var_name].median().to_dict()
    clim_thld_lookup = {sa: med * (1.0 + thld) for sa, med in clim_med_lookup.items()}

    # 2. Define and initialize all new daily columns right here
    smooth_col = f"{var_name}_smoothed"
    roc_col = f"{var_name}_ROC"          # <--- GOES HERE
    df[smooth_col] = np.nan
    df[roc_col] = np.nan
    df['is_bloom'] = False
    df['bloom_id'] = 0

    all_events = []

    # 3. Smooth, compute ROC, and detect bloom events per subarea
    for sa, group in df.groupby('subarea'):
        idx = group.index
        smoothed_vals = smooth_chl_series(
            group[var_name], method=smooth_method, window=window, poly=poly, deriv=0, frac=frac
        )
        roc_vals = smooth_chl_series(
            group[var_name], method=smooth_method, window=window, poly=poly, deriv=1, frac=frac
        )

        df.loc[idx, smooth_col] = smoothed_vals
        df.loc[idx, roc_col] = roc_vals

        if sa in clim_thld_lookup and not np.isnan(clim_thld_lookup[sa]):
            df.loc[idx, 'is_bloom'] = smoothed_vals > clim_thld_lookup[sa]

            events_df, bloom_ids = detect_bloom_events_roc(
                group['Date'], smoothed_vals, roc_vals,
                clim_med=clim_med_lookup[sa],
                clim_thld=clim_thld_lookup[sa],
                **roc_kwargs
            )
            df.loc[idx, 'bloom_id'] = bloom_ids

            if not events_df.empty:
                events_df.insert(0, 'subarea', sa)
                all_events.append(events_df)

    # Combine all subarea event tables into a single DataFrame
    events_all_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    # Restore standard (Date, subarea) ordering
    df = df.sort_values(['Date', 'subarea']).reset_index(drop=True)

    # Build output filename
    dirname, basename = os.path.split(chl_file)
    out_name = basename.replace('-CHL-STATS-', '-PHYTOBLOOMS-METRICS-')
    out_path = os.path.join(dirname, out_name)

    if save_output:
        df_out = df.copy()
        df_out['Date'] = df_out['Date'].dt.strftime('%Y-%m-%d')

        # 4. Append NCCSV data types for ALL new columns
        new_meta = list(meta_lines)
        new_meta.append(f"{smooth_col},*DATA_TYPE*,Float\n")
        new_meta.append(f"{roc_col},*DATA_TYPE*,Float\n")
        new_meta.append("is_bloom,*DATA_TYPE*,Boolean\n")
        new_meta.append("bloom_id,*DATA_TYPE*,Int\n")
        new_meta.append("*END_METADATA*\n")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.writelines(new_meta)
        df_out.to_csv(out_path, mode='a', index=False)

    return df, events_all_df, clim_thld_lookup, out_path

def detect_bloom_events_roc(dates, smoothed_chl, roc, clim_med, clim_thld,
                            window_for_peak=10, peak_distance=14, prominence=0.02,
                            event_distance=21, init_term_window=5,
                            trough_length=3, search_window=90):
    """
    Purpose:
        Identifies discrete phytoplankton bloom events and their initiation, peak,
        and termination timing using a combined Threshold + Rate-of-Change (ROC) method.

    Parameters:
        dates (array-like): Chronologically sorted datetime sequence for one subarea.
        smoothed_chl (array-like): 1D smoothed chlorophyll concentrations (e.g., SavGol deriv=0).
        roc (array-like): 1D daily rate of change of chlorophyll (e.g., SavGol deriv=1).
        clim_med (float): Climatological annual median CHL for the subarea.
        clim_thld (float): Climatological threshold (typically clim_med * 1.10).
        window_for_peak (int): Min consecutive days above threshold for a peak to be valid (default 10).
        peak_distance (int): Min days between consecutive peaks in find_peaks (default 14).
        prominence (float): Min vertical prominence of a peak relative to surroundings (default 0.02).
        event_distance (int): Min days between peaks to be treated as separate events (default 21).
        init_term_window (int): Consecutive days of positive/negative ROC below threshold
                                required to flag initiation/termination (default 5).
        trough_length (int): Days on either side of a local minimum to qualify as a trough (default 3).
        search_window (int): Max days after the last peak to search for termination (default 90).

    Returns:
        tuple: (events_df, bloom_id_array)
            - events_df (pd.DataFrame): One row per bloom event with timing and peak metrics.
            - bloom_id_array (np.ndarray): Integer array matching `dates` length (0 = no bloom,
              1..N = active Bloom_ID for that day).
    """
    chl = np.asarray(smoothed_chl, dtype=float)
    roc_arr = np.asarray(roc, dtype=float)
    dates_arr = pd.to_datetime(dates).to_numpy()
    n = len(chl)
    bloom_id_array = np.zeros(n, dtype=int)

    if np.isnan(chl).all():
        return pd.DataFrame(), bloom_id_array

    # ------------------------------------------------------------------
    # STEP 1: Peak Detection (above threshold + minimum streak duration)
    # ------------------------------------------------------------------
    chl_series = pd.Series(chl)
    is_above = chl_series > clim_thld

    # Assign a unique ID to each contiguous True/False streak and compute its length
    streak_ids = (is_above != is_above.shift()).cumsum()
    streak_lengths = is_above.groupby(streak_ids).transform('sum')

    peak_locs, _ = find_peaks(chl, distance=peak_distance, prominence=prominence)
    valid_peaks = [
        int(p) for p in peak_locs
        if bool(is_above.iloc[p]) and streak_lengths.iloc[p] >= window_for_peak
    ]
    if not valid_peaks:
        return pd.DataFrame(), bloom_id_array

    # ------------------------------------------------------------------
    # STEP 2: Group Nearby Peaks into Single Bloom Events
    # ------------------------------------------------------------------
    raw_events = []
    current_event = [valid_peaks[0]]
    for i in range(1, len(valid_peaks)):
        prev_p, curr_p = valid_peaks[i - 1], valid_peaks[i]
        between = chl_series.iloc[prev_p:curr_p]

        # Check if CHL dropped below threshold for at least `window_for_peak` consecutive days
        dropped_below = False
        if len(between) >= window_for_peak:
            dropped_below = (
                (between < clim_thld)
                .rolling(window=window_for_peak)
                .sum()
                .eq(window_for_peak)
                .any()
            )

        if (curr_p - prev_p < event_distance) or not dropped_below:
            current_event.append(curr_p)
        else:
            raw_events.append(current_event)
            current_event = [curr_p]
    raw_events.append(current_event)

    # ------------------------------------------------------------------
    # STEP 3: Pre-compute ROC Initiation/Termination Candidates & Troughs
    # ------------------------------------------------------------------
    is_below_thld = chl < clim_thld
    is_below_med = chl <= clim_med

    # Initiation: sustained positive growth (ROC >= 0) while still below threshold
    init_cond = pd.Series((roc_arr >= 0) & is_below_thld)
    init_roll = init_cond.rolling(window=init_term_window).sum()
    init_days = init_roll[init_roll == init_term_window].index.to_numpy()

    # Termination: sustained decline (ROC < 0) after dropping below threshold
    term_cond = pd.Series((roc_arr < 0) & is_below_thld)
    term_roll = term_cond.rolling(window=init_term_window).sum()
    term_days = term_roll[term_roll == init_term_window].index.to_numpy()

    # Local troughs: lower than `trough_length` days on both sides AND <= climatological median
    trough_mask = (
        (chl_series < chl_series.shift(trough_length)) &
        (chl_series < chl_series.shift(-trough_length)) &
        is_below_med
    )
    trough_days = trough_mask[trough_mask].index.to_numpy()

    # ------------------------------------------------------------------
    # STEP 4: Resolve Initiation & Termination Bounds per Event
    # ------------------------------------------------------------------
    merged_events = []
    last_start, last_end = 0, 0
    max_idx = n - 1

    for ev_peaks in raw_events:
        ev_first, ev_last = ev_peaks[0], ev_peaks[-1]
        end_window = min(max_idx, ev_last + search_window)

        # --- Initiation ---
        start_idx = last_end
        cand_init = init_days[(init_days < ev_first) & (init_days >= start_idx)]
        if len(cand_init) > 0 and last_end >= last_start:
            # Back up to the first day of the qualifying 5-day growth window
            poss_start = int(cand_init[-1] - init_term_window + 1)
            win_troughs = trough_days[(trough_days >= last_end) & (trough_days <= ev_first)]
            if len(win_troughs) > 0:
                start_idx = int(win_troughs[np.abs(win_troughs - poss_start).argmin()])
            else:
                start_idx = poss_start

        # --- Termination ---
        end_idx = ev_last
        cand_term = term_days[(term_days > ev_last) & (term_days <= end_window)]
        win_troughs = trough_days[(trough_days >= ev_last) & (trough_days <= end_window)]
        if len(cand_term) > 0:
            poss_end = int(cand_term[0] - init_term_window + 1)
            if len(win_troughs) > 0:
                end_idx = int(win_troughs[np.abs(win_troughs - poss_end).argmin()])
            else:
                end_idx = poss_end
        elif len(win_troughs) > 0:
            # Fallback: first trough below median if ROC termination wasn't met
            end_idx = int(win_troughs[0])

        # --- Merge if overlapping with previous event ---
        is_duplicate = (
            len(merged_events) > 0
            and start_idx == merged_events[-1]['start_idx']
            and end_idx == merged_events[-1]['end_idx']
        )
        is_overlapping = (last_end > 0) and (ev_first <= last_end)

        if is_duplicate or is_overlapping:
            merged_events[-1]['peaks'].extend(ev_peaks)
            merged_events[-1]['end_idx'] = max(merged_events[-1]['end_idx'], end_idx)
        else:
            merged_events.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'peaks': list(ev_peaks)
            })

        last_start = merged_events[-1]['start_idx']
        last_end = merged_events[-1]['end_idx']

    # ------------------------------------------------------------------
    # STEP 5: Assemble Event Metrics & Daily Bloom_ID Mask
    # ------------------------------------------------------------------
    rows = []
    for bloom_id, ev in enumerate(merged_events, start=1):
        s, e, pks = ev['start_idx'], ev['end_idx'], ev['peaks']
        bloom_id_array[s:e + 1] = bloom_id

        # Peak CHL and Max ROC within the bloom bounds
        max_pk = max(pks, key=lambda p: chl[p])
        roc_slice = roc_arr[s:e + 1]
        max_roc_idx = s + int(np.nanargmax(roc_slice)) if not np.isnan(roc_slice).all() else s

        # First day crossing above threshold & last day dropping below threshold
        thld_s = pks[0]
        for j in range(s + 1, pks[0] + 1):
            if chl[j] >= clim_thld:
                thld_s = j
                break

        thld_e = e
        if e > pks[-1]:
            for j in range(e, pks[-1] - 1, -1):
                if chl[j] >= clim_thld:
                    thld_e = min(j + 1, max_idx)
                    break

        s_ts = pd.Timestamp(dates_arr[s])
        pk_ts = pd.Timestamp(dates_arr[max_pk])
        e_ts = pd.Timestamp(dates_arr[e])
        bio_year = pk_ts.year if pk_ts.month >= 7 else pk_ts.year - 1

        ev_dates = dates_arr[s:e + 1]
        ev_chl = chl[s:e + 1]
        
        rows.append({
            'Bloom_ID': bloom_id,
            'Year': pk_ts.year,
            'Biological_Year': bio_year,
            'Season': classify_bloom_season(pk_ts.dayofyear),
            'Start_Date': s_ts.strftime('%Y-%m-%d'),
            'Peak_Date': pk_ts.strftime('%Y-%m-%d'),
            'End_Date': e_ts.strftime('%Y-%m-%d'),
            'Start_DOY': s_ts.dayofyear,
            'Peak_DOY': pk_ts.dayofyear,
            'End_DOY': e_ts.dayofyear,
            'First_Exceed_Date': pd.Timestamp(dates_arr[thld_s]).strftime('%Y-%m-%d'),
            'Last_Drop_Date': pd.Timestamp(dates_arr[thld_e]).strftime('%Y-%m-%d'),
            'Duration_Days': int(e - s),
            'Duration_Above_Thld': int(thld_e - thld_s),
            'Num_Peaks': len(pks),
            'Max_CHL': float(chl[max_pk]),
            'Max_ROC': float(roc_arr[max_roc_idx]),
            'Max_ROC_Date': pd.Timestamp(dates_arr[max_roc_idx]).strftime('%Y-%m-%d'),
            'Integrated_CHL': integrate_chl(ev_dates, ev_chl),
            'Integrated_CHL_Above_Thld': integrate_chl(ev_dates, ev_chl, threshold=clim_thld),
        })

    return pd.DataFrame(rows), bloom_id_array

def compute_annual_bloom_metrics(df, events_df, var_name='CHL_Median', year_range=None):
    """
    Purpose:
        Computes annual (calendar and biological year) phytoplankton bloom summary
        metrics per subarea from the daily bloom DataFrame and event table.

    Parameters:
        df (pd.DataFrame): Daily DataFrame returned by detect_threshold_blooms()
                           (must contain 'Date', 'subarea', '{var_name}_smoothed',
                           'is_bloom', and 'bloom_id').
        events_df (pd.DataFrame): Event-level DataFrame returned by detect_threshold_blooms().
        var_name (str): Base CHL column name (default 'CHL_Median').
        year_range (tuple, optional): (start_year, end_year) inclusive. Defaults to all
                                      years present in `df`.

    Returns:
        pd.DataFrame: One row per (subarea, Year) with calendar and biological year metrics.
    """
    smooth_col = f"{var_name}_smoothed"
    records = []

    for sa, grp in df.groupby('subarea'):
        grp = grp.sort_values('Date').reset_index(drop=True)
        ev_sa = events_df[events_df['subarea'] == sa].copy() if not events_df.empty else pd.DataFrame()

        if year_range is None:
            years = sorted(grp['Date'].dt.year.unique())
        else:
            years = list(range(year_range[0], year_range[1] + 1))

        dates_all = grp['Date'].to_numpy()
        chl_all = grp[smooth_col].to_numpy(dtype=float)
        bloom_id_all = grp['bloom_id'].to_numpy(dtype=int)
        is_bloom_all = grp['is_bloom'].to_numpy(dtype=bool)

        for yr in years:
            # --- Calendar Year (Jan 1 - Dec 31) ---
            cal_mask = (dates_all >= np.datetime64(f"{yr}-01-01")) & (dates_all <= np.datetime64(f"{yr}-12-31"))
            cal_dates, cal_chl = dates_all[cal_mask], chl_all[cal_mask]
            cal_bid, cal_thld = bloom_id_all[cal_mask], is_bloom_all[cal_mask]

            ev_yr = ev_sa[ev_sa['Year'] == yr] if not ev_sa.empty else pd.DataFrame()
            n_blooms_cal = len(ev_yr)
            n_spring = int((ev_yr['Season'] == 'Spring').sum()) if not ev_yr.empty else 0
            n_fall = int((ev_yr['Season'] == 'Fall').sum()) if not ev_yr.empty else 0

            bloom_days_cal = int((cal_bid > 0).sum())
            above_thld_days_cal = int(((cal_bid > 0) & cal_thld).sum())
            pct_above_cal = (above_thld_days_cal / bloom_days_cal * 100.0) if bloom_days_cal > 0 else 0.0

            tot_int_cal = integrate_chl(cal_dates, cal_chl)
            bloom_int_cal = sum(
                integrate_chl(cal_dates[cal_bid == bid], cal_chl[cal_bid == bid])
                for bid in np.unique(cal_bid[cal_bid > 0])
            )

            # --- Biological Year (Jul 1 of yr - Jun 30 of yr+1) ---
            bio_mask = (dates_all >= np.datetime64(f"{yr}-07-01")) & (dates_all <= np.datetime64(f"{yr+1}-06-30"))
            bio_dates, bio_chl = dates_all[bio_mask], chl_all[bio_mask]
            bio_bid, bio_thld = bloom_id_all[bio_mask], is_bloom_all[bio_mask]

            n_blooms_bio = int((ev_sa['Biological_Year'] == yr).sum()) if not ev_sa.empty else 0
            bloom_days_bio = int((bio_bid > 0).sum())
            above_thld_days_bio = int(((bio_bid > 0) & bio_thld).sum())
            pct_above_bio = (above_thld_days_bio / bloom_days_bio * 100.0) if bloom_days_bio > 0 else 0.0

            tot_int_bio = integrate_chl(bio_dates, bio_chl)
            bloom_int_bio = sum(
                integrate_chl(bio_dates[bio_bid == bid], bio_chl[bio_bid == bid])
                for bid in np.unique(bio_bid[bio_bid > 0])
            )

            records.append({
                'subarea': sa,
                'Year': yr,
                'Num_Blooms_Cal': n_blooms_cal,
                'Num_Spring_Blooms': n_spring,
                'Num_Fall_Blooms': n_fall,
                'Bloom_Days_Cal': bloom_days_cal,
                'Bloom_Days_Above_Thld_Cal': above_thld_days_cal,
                'Pct_Bloom_Days_Above_Thld_Cal': round(pct_above_cal, 2),
                'Total_Integrated_CHL_Cal': round(tot_int_cal, 2),
                'Bloom_Integrated_CHL_Cal': round(bloom_int_cal, 2),
                'Pct_CHL_In_Blooms_Cal': round((bloom_int_cal / tot_int_cal * 100.0) if tot_int_cal > 0 else 0.0, 2),
                'Num_Blooms_Bio': n_blooms_bio,
                'Bloom_Days_Bio': bloom_days_bio,
                'Bloom_Days_Above_Thld_Bio': above_thld_days_bio,
                'Pct_Bloom_Days_Above_Thld_Bio': round(pct_above_bio, 2),
                'Total_Integrated_CHL_Bio': round(tot_int_bio, 2),
                'Bloom_Integrated_CHL_Bio': round(bloom_int_bio, 2),
                'Pct_CHL_In_Blooms_Bio': round((bloom_int_bio / tot_int_bio * 100.0) if tot_int_bio > 0 else 0.0, 2),
            })

    return pd.DataFrame(records)

def _write_nccsv(df, out_path, base_meta_lines, title_suffix="", history_note=""):
    """
    Purpose: 
        Writes a DataFrame to an ERDDAP-compatible NCCSV file, reusing the *GLOBAL*
        metadata lines from the source file and auto-generating *DATA_TYPE* lines
        for all columns in `df`.
    """
    global_lines = []
    for line in base_meta_lines:
        if not line.startswith('*GLOBAL*,'):
            continue
        if title_suffix and line.startswith('*GLOBAL*,title,'):
            clean_title = line.strip().rstrip('"') + f' ({title_suffix})"\n'
            global_lines.append(clean_title)
        elif history_note and line.startswith('*GLOBAL*,history,'):
            clean_hist = line.strip().rstrip('"') + f' | {get_current_utc_timestamp()} {history_note}"\n'
            global_lines.append(clean_hist)
        else:
            global_lines.append(line)

    # Auto-generate *DATA_TYPE* metadata lines for every column
    var_lines = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            nc_type = "Boolean"
        elif pd.api.types.is_integer_dtype(dtype):
            nc_type = "Int"
        elif pd.api.types.is_float_dtype(dtype):
            nc_type = "Float"
        else:
            nc_type = "String"
        var_lines.append(f"{col},*DATA_TYPE*,{nc_type}\n")
        if col == 'Date':
            var_lines.append("Date,standard_name,time\n")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(global_lines)
        f.writelines(var_lines)
        f.write('*END_METADATA*\n')
    df.to_csv(out_path, mode='a', index=False)


def run_phytobloom_pipeline(regions, datasets='OCCCI', periods='D7', prod='CHL',
                            var_name='CHL_Median', thld=0.1,
                            smooth_method='savgol', window=15, poly=3, frac=0.00117,
                            year_range=None, overwrite=False,
                            logfile=None, verbose=False, debug=False, **roc_kwargs):
    """
    Purpose:
        Orchestrates the end-to-end phytoplankton bloom detection pipeline for one or
        more regions, datasets, and temporal periods.

    For each (dataset, region, period) combination:
      1. Locates the extracted timeseries CSV (default: D7) and ANNUAL climatology CSV.
      2. Checks file freshness (skips if outputs are newer than inputs and overwrite=False).
      3. Runs `detect_threshold_blooms()` (smoothing, ROC, threshold mask, event detection).
      4. Runs `compute_annual_bloom_metrics()` (calendar & biological year summaries).
      5. Saves three ERDDAP NCCSV files in the same directory as the source CSV:
           - ...-PHYTOBLOOMS-METRICS-...  (daily timeseries + smoothed CHL, ROC, is_bloom, bloom_id)
           - ...-PHYTOBLOOMS-EVENTS-...   (one row per discrete bloom event)
           - ...-PHYTOBLOOMS-ANNUAL-...   (annual calendar & biological year summaries)

    Parameters:
        regions (str or list of str): Extracted region/shapefile name(s) (e.g., 'NES_EPU_NOESTUARIES').
        datasets (str or list of str): Dataset name(s) (default: 'OCCCI').
        periods (str or list of str): Input timeseries period(s) (default: 'D7').
        prod (str): Source chlorophyll product code (default: 'CHL').
        var_name (str): Target CHL column to smooth and evaluate (default: 'CHL_Median').
        thld (float): Fractional threshold above annual median (default: 0.1 = +10%).
        smooth_method (str): 'savgol' or 'lowess' (default: 'savgol').
        window (int): SavGol window length in days (default: 15).
        poly (int): SavGol polynomial order (default: 3).
        frac (float): LOWESS smoothing fraction (default: 0.00117).
        year_range (tuple, optional): (start_year, end_year) for annual summaries.
        overwrite (bool): Reprocess even if outputs are up-to-date (default: False).
        logfile (str or bool, optional): Path to log file, or True/'auto' to auto-generate.
        verbose (bool): Print detailed progress messages (default: False).
        debug (bool): Print full tracebacks on error (default: False).
        **roc_kwargs: Optional overrides passed to `detect_bloom_events_roc()`
                      (e.g., window_for_peak, peak_distance, prominence, event_distance).

    Returns:
        dict: Nested dictionary keyed by (dataset, region, period) containing:
              {'daily_df', 'events_df', 'annual_df', 'thresholds', 'output_files'}.
    """
    # Normalize list inputs
    if isinstance(regions, str): regions = [regions]
    if isinstance(datasets, str): datasets = [datasets]
    if isinstance(periods, str): periods = [periods]

    # Setup optional logging
    if logfile in ["auto", True]:
        logfile = generate_logfile_path("phytoblooms", prods=[prod], dataset=datasets[0], env_dict=env)

    original_stdout = sys.stdout
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        sys.stdout = DualLogger(logfile)
        print("\n" + "=" * 80)
        print(f"🟢 PHYTOBLOOM PIPELINE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    results = {}
    processed_count = 0
    skipped_count = 0
    failed_tasks = []

    try:
        for ds_name in datasets:
            for region in regions:
                # 1. Locate ANNUAL climatology file once per (dataset, region)
                clim_files = get_extracted_files(ds_name, region, prod=prod, period='ANNUAL')
                if not clim_files:
                    msg = f"{ds_name} | {region} (Missing ANNUAL climatology CSV)"
                    print(f"⚠️ Skipping {msg}")
                    failed_tasks.append(msg)
                    continue
                clim_file = clim_files[0]

                for period in periods:
                    chl_files = get_extracted_files(ds_name, region, prod=prod, period=period)
                    if not chl_files:
                        msg = f"{ds_name} | {region} | {period} (No extracted {prod} CSV found)"
                        print(f"⚠️ Skipping {msg}")
                        failed_tasks.append(msg)
                        continue

                    for chl_file in chl_files:
                        dirname, basename = os.path.split(chl_file)
                        daily_out = os.path.join(dirname, basename.replace(f'-{prod}-STATS-', '-PHYTOBLOOMS-METRICS-'))
                        events_out = os.path.join(dirname, basename.replace(f'-{prod}-STATS-', '-PHYTOBLOOMS-EVENTS-'))
                        annual_out = os.path.join(dirname, basename.replace(f'-{prod}-STATS-', '-PHYTOBLOOMS-ANNUAL-'))

                        # 2. Freshness check
                        all_exist = all(os.path.exists(p) for p in [daily_out, events_out, annual_out])
                        if all_exist and not overwrite:
                            latest_in = max(os.path.getmtime(chl_file), os.path.getmtime(clim_file))
                            earliest_out = min(os.path.getmtime(p) for p in [daily_out, events_out, annual_out])
                            if earliest_out > latest_in:
                                if verbose:
                                    print(f"  ⏭️ Up-to-date: {os.path.basename(daily_out)}")
                                skipped_count += 1
                                continue

                        print(f"\n🚀 Processing Phytoblooms: {ds_name} | {region} | {period}")
                        try:
                            meta_lines, _ = read_extracted_nccsv(chl_file)

                            # 3. Daily smoothing, ROC, threshold flags, and event detection
                            daily_df, events_df, thlds, daily_path = detect_threshold_blooms(
                                chl_file, clim_file,
                                var_name=var_name, thld=thld,
                                smooth_method=smooth_method, window=window, poly=poly, frac=frac,
                                save_output=True, **roc_kwargs
                            )

                            # 4. Annual (calendar & biological year) summary metrics
                            annual_df = compute_annual_bloom_metrics(
                                daily_df, events_df, var_name=var_name, year_range=year_range
                            )

                            # 5. Save Event and Annual NCCSVs
                            if not events_df.empty:
                                _write_nccsv(
                                    events_df, events_out, meta_lines,
                                    title_suffix="Phytoplankton Bloom Events",
                                    history_note=f"Detected {len(events_df)} ROC bloom events (thld=+{int(thld*100)}%)."
                                )
                            if not annual_df.empty:
                                _write_nccsv(
                                    annual_df, annual_out, meta_lines,
                                    title_suffix="Annual Phytoplankton Bloom Summary",
                                    history_note="Computed calendar and biological year bloom metrics."
                                )

                            results[(ds_name, region, period)] = {
                                'daily_df': daily_df,
                                'events_df': events_df,
                                'annual_df': annual_df,
                                'thresholds': thlds,
                                'output_files': {
                                    'daily': daily_path,
                                    'events': events_out,
                                    'annual': annual_out,
                                }
                            }
                            processed_count += 1
                            print(f"  ✅ Saved: {os.path.basename(daily_path)}")
                            print(f"  ✅ Saved: {os.path.basename(events_out)} ({len(events_df)} events)")
                            print(f"  ✅ Saved: {os.path.basename(annual_out)} ({len(annual_df)} rows)")

                        except Exception as e:
                            failed_tasks.append(f"{ds_name} | {region} | {period}: {e}")
                            print(f"  ❌ Error processing {basename}: {e}")
                            if debug:
                                traceback.print_exc()

        # --- Final Summary ---
        print("\n" + "=" * 60)
        print("🏁 PHYTOBLOOM PIPELINE SUMMARY")
        print("=" * 60)
        print(f"✅ Processed: {processed_count}")
        print(f"⏭️ Skipped:   {skipped_count}")
        print(f"❌ Failed:    {len(failed_tasks)}")
        if failed_tasks:
            for ft in failed_tasks:
                print(f"  • {ft}")
        print("=" * 60)

    finally:
        if logfile:
            print(f"\n🔴 PHYTOBLOOM PIPELINE ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80 + "\n")
            sys.stdout.close()
            sys.stdout = original_stdout

    return results