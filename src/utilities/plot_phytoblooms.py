import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress


SEASON_COLORS = {
    'Spring': '#2ca02c',   # green
    'Summer': '#1f77b4',   # blue
    'Fall':   '#ff8c00',   # dark orange (matches original Fall plot)
    'Winter': '#6a5acd',   # slate blue
}

BLOOM_METRIC_SPECS = [
    ('Start_DOY_Unwrapped', 'Bloom Initiation',       'DOY',                                         True),
    ('Peak_DOY_Unwrapped',  'Bloom Peak',             'DOY',                                         True),
    ('End_DOY_Unwrapped',   'Bloom Termination',      'DOY',                                         True),
    ('Duration_Days',       'Duration',               'Duration (Days)',                             False),
    ('Integrated_CHL',      'Integrated Chlorophyll', r'Integrated Chlorophyll ($mg/m^3 \cdot days$)', False),
    ('Max_CHL',             'Maximum Chlorophyll',    r'Chlorophyll concentration ($mg/m^3$)',        False),
]


def _prepare_seasonal_events(events_df, subarea, season, year_col='Year',
                             year_range=None, dominant_only=False):
    """
    Filters `events_df` to a single (subarea, season) and computes continuous
    (unwrapped) DOY values so Fall/Winter blooms crossing Dec 31 plot > 365.
    """
    sub = events_df[
        (events_df['subarea'] == subarea) &
        (events_df['Season'].str.lower() == season.lower())
    ].copy()

    if sub.empty:
        return sub

    if year_range is not None:
        sub = sub[(sub[year_col] >= year_range[0]) & (sub[year_col] <= year_range[1])]

    if sub.empty:
        return sub

    # Optional: keep only the single largest bloom per year
    if dominant_only:
        sub = (
            sub.sort_values('Integrated_CHL', ascending=False)
               .drop_duplicates(subset=[year_col])
               .sort_values(year_col)
        )

    # Unwrap DOYs relative to Start_Date's calendar year so cross-year blooms stay continuous
    s_dt = pd.to_datetime(sub['Start_Date'])
    p_dt = pd.to_datetime(sub['Peak_Date'])
    e_dt = pd.to_datetime(sub['End_Date'])

    ref_year_start = pd.to_datetime(s_dt.dt.year.astype(str) + '-01-01')
    sub['Start_DOY_Unwrapped'] = (s_dt - ref_year_start).dt.days + 1
    sub['Peak_DOY_Unwrapped']  = (p_dt - ref_year_start).dt.days + 1
    sub['End_DOY_Unwrapped']   = (e_dt - ref_year_start).dt.days + 1

    # For Fall blooms that actually start in Jan of the following year, shift by +365
    if season.lower() == 'fall':
        jan_shift = sub['Start_DOY_Unwrapped'] < 180
        sub.loc[jan_shift, ['Start_DOY_Unwrapped', 'Peak_DOY_Unwrapped', 'End_DOY_Unwrapped']] += 365

    return sub.sort_values(year_col).reset_index(drop=True)


def compute_bloom_trends(events_df, seasons=('Spring', 'Fall'), year_col='Year',
                         year_range=None, alpha=0.05, dominant_only=False):
    """
    Computes OLS linear trends (slope, R^2, p-value, significance) for all six
    bloom metrics across every subarea and requested season.

    Returns:
        pd.DataFrame: Summary table of trend statistics.
    """
    rows = []
    subareas = sorted(events_df['subarea'].dropna().unique())

    for subarea in subareas:
        for season in seasons:
            sub = _prepare_seasonal_events(
                events_df, subarea, season,
                year_col=year_col, year_range=year_range, dominant_only=dominant_only
            )
            for col, label, _, _ in BLOOM_METRIC_SPECS:
                valid = sub[[year_col, col]].dropna()
                n = len(valid)
                if n < 3 or valid[year_col].nunique() < 2:
                    rows.append({
                        'subarea': subarea, 'Season': season, 'Metric': label,
                        'N': n, 'Slope_Per_Year': np.nan, 'Slope_Per_Decade': np.nan,
                        'Intercept': np.nan, 'R_Value': np.nan, 'R_Squared': np.nan,
                        'P_Value': np.nan, 'Std_Err': np.nan, 'Significant': False,
                    })
                    continue

                res = linregress(valid[year_col].astype(float), valid[col].astype(float))
                rows.append({
                    'subarea': subarea,
                    'Season': season,
                    'Metric': label,
                    'N': n,
                    'Slope_Per_Year': res.slope,
                    'Slope_Per_Decade': res.slope * 10.0,
                    'Intercept': res.intercept,
                    'R_Value': res.rvalue,
                    'R_Squared': res.rvalue ** 2,
                    'P_Value': res.pvalue,
                    'Std_Err': res.stderr,
                    'Significant': bool(res.pvalue < alpha),
                })

    return pd.DataFrame(rows)


def plot_seasonal_bloom_trends(events_df, subarea, season='Fall', year_col='Year',
                               year_range=None, alpha=0.05, dominant_only=False,
                               color=None, save_path=None, show=True):
    """
    Generates a 2x3 panel plot of seasonal phytoplankton bloom metrics with OLS
    trend lines, 95% confidence bands, and p-value annotations (bold red if p < alpha).
    """
    sub = _prepare_seasonal_events(
        events_df, subarea, season,
        year_col=year_col, year_range=year_range, dominant_only=dominant_only
    )
    if sub.empty:
        print(f"⚠️ No {season} bloom events found for '{subarea}'. Skipping plot.")
        return None

    plot_color = color or SEASON_COLORS.get(season.capitalize(), '#ff8c00')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
    fig.suptitle(f"{subarea} {season.capitalize()} Bloom Metrics", fontsize=20, y=0.98)

    # Determine shared DOY y-limits across the top row
    doy_vals = pd.concat([
        sub['Start_DOY_Unwrapped'], sub['Peak_DOY_Unwrapped'], sub['End_DOY_Unwrapped']
    ]).dropna()
    doy_ymin = max(0, np.floor((doy_vals.min() - 25) / 50) * 50)
    doy_ymax = np.ceil((doy_vals.max() + 75) / 50) * 50  # extra headroom for p-value text

    # Year ticks (every year minor/tick, label every 3 years like original notebook)
    yr_min = int(year_range[0]) if year_range else int(sub[year_col].min())
    yr_max = int(year_range[1]) if year_range else int(sub[year_col].max())
    all_years = np.arange(yr_min, yr_max + 1)
    # Start 3-year labels at 1998 if in range, else yr_min
    label_start = 1998 if yr_min <= 1998 <= yr_max else yr_min
    labeled_years = set(range(label_start, yr_max + 1, 3))

    for idx, (col, title, ylabel, is_doy) in enumerate(BLOOM_METRIC_SPECS):
        ax = axes[idx // 3, idx % 3]
        valid = sub[[year_col, col]].dropna()

        if len(valid) >= 2:
            sns.regplot(
                data=valid, x=year_col, y=col, ax=ax,
                ci=95, color=plot_color,
                scatter_kws={'s': 45, 'alpha': 0.9},
                line_kws={'linewidth': 2.2},
            )
            res = linregress(valid[year_col].astype(float), valid[col].astype(float))
            is_sig = res.pvalue < alpha
            ax.text(
                0.05, 0.95, f"p-value = {res.pvalue:.4f}",
                transform=ax.transAxes, va='top', ha='left',
                fontsize=14,
                fontweight='bold' if is_sig else 'normal',
                color='red' if is_sig else 'black',
            )
        else:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha='center', va='center')

        ax.set_title(title, fontsize=16, pad=8)
        ax.set_ylabel(ylabel if (not is_doy or idx == 0) else '', fontsize=12)
        ax.set_xlabel('Year' if idx >= 3 else '', fontsize=12)

        if is_doy:
            ax.set_ylim(doy_ymin, doy_ymax)
            if doy_ymin < 365 < doy_ymax:
                ax.axhline(365, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
        else:
            # Add a bit of top headroom so p-value text doesn't collide with points
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.12 * (ymax - ymin))

        ax.set_xticks(all_years)
        if idx >= 3:
            ax.set_xticklabels(
                [str(y) if y in labeled_years else '' for y in all_years],
                rotation=70, fontsize=11
            )
        ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def run_bloom_trend_analysis(events_df, seasons=('Spring', 'Fall'), year_col='Year',
                             year_range=None, alpha=0.05, dominant_only=False,
                             output_dir=None, show=True):
    """
    Runs trend statistics across all subareas & seasons and generates the 6-panel
    diagnostic figure for each combination.
    """
    trends_df = compute_bloom_trends(
        events_df, seasons=seasons, year_col=year_col,
        year_range=year_range, alpha=alpha, dominant_only=dominant_only
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        trends_csv = os.path.join(output_dir, "phytobloom_trend_summary.csv")
        trends_df.to_csv(trends_csv, index=False)
        print(f"📊 Saved trend summary table: {trends_csv}")

    for subarea in sorted(events_df['subarea'].dropna().unique()):
        for season in seasons:
            save_path = None
            if output_dir:
                safe_sa = str(subarea).replace(' ', '_')
                save_path = os.path.join(output_dir, f"{safe_sa}_{season}_Bloom_Trends.png")

            plot_seasonal_bloom_trends(
                events_df, subarea=subarea, season=season,
                year_col=year_col, year_range=year_range,
                alpha=alpha, dominant_only=dominant_only,
                save_path=save_path, show=show
            )

    return trends_df