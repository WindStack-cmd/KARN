import pandas as pd
import numpy as np
import os

RAW_DIR  = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
PROC_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed"
os.makedirs(PROC_DIR, exist_ok=True)

# Flare class thresholds (W/m²)
THRESHOLDS = {
    'X': 1e-4,
    'M': 1e-5,
    'C': 1e-6,
    'B': 1e-7,
}

def classify_flare(peak_flux):
    if peak_flux >= THRESHOLDS['X']:
        return 'X'
    elif peak_flux >= THRESHOLDS['M']:
        return 'M'
    elif peak_flux >= THRESHOLDS['C']:
        return 'C'
    elif peak_flux >= THRESHOLDS['B']:
        return 'B'
    else:
        return 'Q'

def strip_timezone(df, col):
    """Remove timezone info from a datetime column safely."""
    df[col] = pd.to_datetime(df[col])
    if df[col].dt.tz is not None:
        df[col] = df[col].dt.tz_localize(None)
    return df

def build_flare_catalog(df):
    print("Building flare catalog...")

    # Work with soft X-ray only
    df_soft = df[df['energy'] == '0.1-0.8nm'].copy()
    df_soft = strip_timezone(df_soft, 'time_tag')
    df_soft = df_soft.sort_values('time_tag').reset_index(drop=True)
    df_soft['flux'] = pd.to_numeric(df_soft['flux'], errors='coerce')
    df_soft = df_soft.dropna(subset=['flux'])

    # Rolling background
    df_soft['background'] = df_soft['flux'].rolling(60, min_periods=1).quantile(0.10)
    df_soft['flux_deriv']  = df_soft['flux'].diff()

    C_THRESHOLD = THRESHOLDS['C']

    events     = []
    in_flare   = False
    flare_start = None
    flare_peak_flux = 0
    flare_peak_time = None
    event_id   = 0

    for idx, row in df_soft.iterrows():
        flux = row['flux']
        time = row['time_tag']

        if not in_flare:
            if flux >= C_THRESHOLD:
                in_flare        = True
                flare_start     = time
                flare_peak_flux = flux
                flare_peak_time = time
        else:
            if flux > flare_peak_flux:
                flare_peak_flux = flux
                flare_peak_time = time

            if flux < C_THRESHOLD:
                in_flare     = False
                flare_class  = classify_flare(flare_peak_flux)

                if flare_class in ['C', 'M', 'X']:
                    events.append({
                        'event_id':    event_id,
                        'start_time':  flare_start,
                        'peak_time':   flare_peak_time,
                        'end_time':    time,
                        'peak_flux':   flare_peak_flux,
                        'flare_class': flare_class,
                        'duration_min': (time - flare_start).seconds / 60
                    })
                    event_id += 1

    catalog = pd.DataFrame(events)

    if len(catalog) == 0:
        print("No flares found — data window may be too short")
        return catalog

    # Strip timezone from all time columns
    for col in ['start_time', 'peak_time', 'end_time']:
        catalog = strip_timezone(catalog, col)

    catalog_path = os.path.join(PROC_DIR, "flare_catalog.csv")
    catalog.to_csv(catalog_path, index=False)

    print("\n" + "=" * 60)
    print(f"Flare catalog saved: {catalog_path}")
    print(f"Total flares found: {len(catalog)}")
    print(f"\nBreakdown by class:")
    print(catalog['flare_class'].value_counts())
    print(f"\nDate range: {catalog['start_time'].min()} → {catalog['start_time'].max()}")
    print("=" * 60)
    print("\nSample catalog entries:")
    print(catalog.head(10))

    return catalog

def add_quiet_periods(df, catalog, n_quiet=50):
    print("\nAdding quiet Sun periods...")

    df_soft = df[df['energy'] == '0.1-0.8nm'].copy()
    df_soft = strip_timezone(df_soft, 'time_tag')
    df_soft = df_soft.sort_values('time_tag').reset_index(drop=True)
    df_soft['flux'] = pd.to_numeric(df_soft['flux'], errors='coerce')

    # Strip timezone from catalog too
    for col in ['start_time', 'peak_time', 'end_time']:
        catalog = strip_timezone(catalog, col)

    quiet_events = []
    event_id     = len(catalog)

    window_size = 60
    step        = 30

    times  = df_soft['time_tag'].values
    fluxes = df_soft['flux'].values

    i     = 0
    found = 0

    while i < len(df_soft) - window_size and found < n_quiet:
        window_flux = fluxes[i:i + window_size]
        window_max  = np.nanmax(window_flux)

        if window_max < THRESHOLDS['C']:
            quiet_events.append({
                'event_id':    event_id,
                'start_time':  pd.Timestamp(times[i]),
                'peak_time':   pd.Timestamp(times[i + window_size // 2]),
                'end_time':    pd.Timestamp(times[i + window_size - 1]),
                'peak_flux':   float(window_max),
                'flare_class': 'Q',
                'duration_min': window_size
            })
            event_id += 1
            found    += 1
            i        += window_size
        else:
            i += step

    quiet_df     = pd.DataFrame(quiet_events)
    full_catalog = pd.concat([catalog, quiet_df], ignore_index=True)
    full_catalog  = full_catalog.sort_values('start_time').reset_index(drop=True)

    full_path = os.path.join(PROC_DIR, "full_catalog.csv")
    full_catalog.to_csv(full_path, index=False)

    print(f"\nFull catalog (flares + quiet): {len(full_catalog)} events")
    print(full_catalog['flare_class'].value_counts())
    print(f"Saved: {full_path}")

    return full_catalog

if __name__ == "__main__":
    flux_path = os.path.join(RAW_DIR, "goes_flux_combined.csv")

    if not os.path.exists(flux_path):
        print("ERROR: goes_flux_combined.csv not found")
        print("Run download_historical.py first")
        exit()

    print("Loading flux data...")
    df = pd.read_csv(flux_path)
    df['time_tag'] = pd.to_datetime(df['time_tag'])
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['time_tag'].min()} → {df['time_tag'].max()}")

    catalog = build_flare_catalog(df)

    if len(catalog) > 0:
        full_catalog = add_quiet_periods(df, catalog)
        print("\nDay 2 complete — catalog ready for ML")
        print(f"Total events in full catalog: {len(full_catalog)}")