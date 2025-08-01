from datetime import datetime, date, timedelta

def format_date(dt, fmt):
    if fmt == "yyyymmdd":
        return dt.strftime("%Y%m%d")
    elif fmt == "yyyy-mm-dd":
        return dt.strftime("%Y-%m-%d")
    elif fmt == "yyyymmddhhmmss":
        return dt.strftime("%Y%m%d%H%M%S")
    elif fmt == "datetime":
        return dt.date()
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
            - List of two strings: ['20250101', '20250630']
    
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

    # Range of years
    if all(isinstance(d, int) for d in dates):
        start = date(dates[0], 1, 1)
        end = date(dates[-1], 12, 31)

    # Range of strings
    elif len(dates) == 2 and all(isinstance(d, (str, int, date, datetime)) for d in dates):
        start = to_dt(dates[0])
        end = to_dt(dates[1])

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