import pandas as pd
import numpy as np
import os

RAW_DIR  = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
PROC_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed"
os.makedirs(PROC_DIR, exist_ok=True)

def strip_tz(series):
    s = pd.to_datetime(series, utc=True)
    return s.dt.tz_localize(None)

def classify_flare(class_type):
    if pd.isna(class_type):
        return 'Q'
    c = str(class_type)[0].upper()
    if c == 'X': return 'X'
    if c == 'M': return 'M'
    if c == 'C': return 'C'
    if c == 'B': return 'B'
    return 'Q'

def build_catalog_from_donki():
    print("=" * 60)
    print("KARN — Building Catalog from NASA DONKI")
    print("=" * 60)

    # Load DONKI flare catalog
    donki_path = os.path.join(RAW_DIR, "donki_flare_catalog.csv")
    donki = pd.read_csv(donki_path)

    print(f"Loaded {len(donki)} DONKI flare events")

    # Clean timestamps
    donki['beginTime'] = strip_tz(donki['beginTime'])
    donki['peakTime']  = strip_tz(donki['peakTime'])
    donki['endTime']   = strip_tz(donki['endTime'])

    # Add classification
    donki['flare_class'] = donki['classType'].apply(classify_flare)

    print(f"\nFull DONKI breakdown:")
    print(donki['flare_class'].value_counts())

    # Build clean catalog
    catalog_rows = []
    event_id = 0

    for _, row in donki.iterrows():
        flare_class = row['flare_class']
        if flare_class not in ['C', 'M', 'X']:
            continue

        catalog_rows.append({
            'event_id':    event_id,
            'start_time':  row['beginTime'],
            'peak_time':   row['peakTime'],
            'end_time':    row['endTime'],
            'peak_flux':   0.0,  # DONKI doesn't have flux values
            'flare_class': flare_class,
            'duration_min': (
                (row['endTime'] - row['beginTime']).seconds / 60
                if pd.notna(row['endTime']) and pd.notna(row['beginTime'])
                else 30.0
            ),
            'source': 'DONKI'
        })
        event_id += 1

    catalog = pd.DataFrame(catalog_rows)

    print(f"\nClean catalog: {len(catalog)} events")
    print(catalog['flare_class'].value_counts())

    # Save flare-only catalog
    cat_path = os.path.join(PROC_DIR, "flare_catalog_v2.csv")
    catalog.to_csv(cat_path, index=False)
    print(f"Saved: {cat_path}")

    return catalog

def add_quiet_periods_v2(catalog, n_quiet=300):
    """
    Generate quiet Sun windows synthetically.
    Since we don't have flux data for 2 years,
    we create quiet windows in gaps between flares.
    """
    print(f"\nAdding {n_quiet} quiet Sun periods...")

    catalog = catalog.sort_values('start_time').reset_index(drop=True)

    quiet_events = []
    event_id     = len(catalog)

    # Find gaps between flares that are at least 2 hours long
    for i in range(len(catalog) - 1):
        current_end = pd.Timestamp(catalog.iloc[i]['end_time'])
        next_start  = pd.Timestamp(catalog.iloc[i + 1]['start_time'])

        if pd.isna(current_end) or pd.isna(next_start):
            continue

        gap_minutes = (next_start - current_end).seconds / 60

        # If gap is more than 2 hours — it's a quiet period
        if gap_minutes >= 120:
            quiet_start = current_end + pd.Timedelta('30min')
            quiet_end   = quiet_start + pd.Timedelta('60min')

            quiet_events.append({
                'event_id':    event_id,
                'start_time':  quiet_start,
                'peak_time':   quiet_start + pd.Timedelta('30min'),
                'end_time':    quiet_end,
                'peak_flux':   0.0,
                'flare_class': 'Q',
                'duration_min': 60.0,
                'source': 'synthetic'
            })
            event_id += 1

            if len(quiet_events) >= n_quiet:
                break

    quiet_df     = pd.DataFrame(quiet_events)
    full_catalog = pd.concat([catalog, quiet_df], ignore_index=True)
    full_catalog  = full_catalog.sort_values('start_time').reset_index(drop=True)

    full_path = os.path.join(PROC_DIR, "full_catalog_v2.csv")
    full_catalog.to_csv(full_path, index=False)

    print(f"\nFull catalog v2:")
    print(full_catalog['flare_class'].value_counts())
    print(f"Total events: {len(full_catalog)}")
    print(f"Saved: {full_path}")

    return full_catalog

if __name__ == "__main__":
    catalog = build_catalog_from_donki()
    full    = add_quiet_periods_v2(catalog, n_quiet=300)

    print("\n" + "=" * 60)
    print("Catalog v2 ready")
    print("=" * 60)
    print(f"\nDate range: {full['start_time'].min()} → {full['start_time'].max()}")
    print(f"\nSample:")
    print(full.head(10).to_string())