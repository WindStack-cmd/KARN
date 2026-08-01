import requests
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta

RAW_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_noaa_archive_month(year, month):
    """
    Fetch monthly GOES X-ray data from NOAA SWPC archive.
    Archive format: https://services.swpc.noaa.gov/json/goes/primary/
    """
    month_str = f"{year}-{month:02d}"
    filename  = os.path.join(RAW_DIR, f"goes_{year}_{month:02d}.csv")

    if os.path.exists(filename):
        print(f"  Already downloaded: {filename}")
        df = pd.read_csv(filename)
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        return df

    # NOAA archive for historical monthly files
    url = f"https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

    print(f"  Fetching {month_str}...")
    try:
        response = requests.get(url, timeout=30)
        raw      = response.text.strip().replace('][', ',')
        data     = json.loads(raw)
        df       = pd.DataFrame(data)
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        df['flux']     = pd.to_numeric(df['flux'], errors='coerce')
        df.to_csv(filename, index=False)
        print(f"  Saved {len(df)} rows → {filename}")
        return df
    except Exception as e:
        print(f"  Failed: {e}")
        return None

def fetch_goes_archive_bulk():
    """
    Pull GOES X-ray data from multiple NOAA endpoints to maximize dataset size.
    Then download historical flare event lists for ground truth labels.
    """
    print("=" * 60)
    print("KARN — Bulk Historical Data Download")
    print("=" * 60)

    all_frames = []

    # NOAA provides multiple time window endpoints
    endpoints = {
        "7day":  "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
        "3day":  "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json",
        "6hour": "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
        "1day":  "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json",
    }

    for label, url in endpoints.items():
        try:
            print(f"Fetching {label} endpoint...")
            response = requests.get(url, timeout=30)
            raw      = response.text.strip().replace('][', ',')
            data     = json.loads(raw)
            df       = pd.DataFrame(data)
            df['time_tag'] = pd.to_datetime(df['time_tag'])
            df['flux']     = pd.to_numeric(df['flux'], errors='coerce')
            all_frames.append(df)
            print(f"  Got {len(df)} rows")
            time.sleep(1)
        except Exception as e:
            print(f"  Failed {label}: {e}")

    # NOAA historical flare event list — this is the key
    # Contains ALL flares going back years with exact timestamps and classes
    print("\nFetching NOAA historical flare event lists...")
    flare_endpoints = [
        "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
        "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json",
    ]

    all_flares = []
    for url in flare_endpoints:
        try:
            response = requests.get(url, timeout=30)
            raw      = response.text.strip().replace('][', ',')
            data     = json.loads(raw)
            flare_df = pd.DataFrame(data)
            all_flares.append(flare_df)
            print(f"  Got {len(flare_df)} flare events from {url.split('/')[-1]}")
            time.sleep(1)
        except Exception as e:
            print(f"  Failed: {e}")

    # Save flare catalog
    if all_flares:
        combined_flares = pd.concat(all_flares, ignore_index=True)
        combined_flares = combined_flares.drop_duplicates()
        flare_path      = os.path.join(RAW_DIR, "noaa_flare_catalog.csv")
        combined_flares.to_csv(flare_path, index=False)
        print(f"\nFlare catalog: {len(combined_flares)} events saved")
        print(combined_flares[['time_tag', 'current_class', 'max_class']].head(10))

    # Combine all flux data
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['time_tag', 'energy'])
        combined = combined.sort_values('time_tag').reset_index(drop=True)

        out_path = os.path.join(RAW_DIR, "goes_flux_combined.csv")
        combined.to_csv(out_path, index=False)

        print("\n" + "=" * 60)
        print(f"Combined flux saved: {out_path}")
        print(f"Total rows:   {len(combined)}")
        print(f"Date range:   {combined['time_tag'].min()} → {combined['time_tag'].max()}")

        # Show class distribution potential
        soft = combined[combined['energy'] == '0.1-0.8nm'].copy()
        soft['flux'] = pd.to_numeric(soft['flux'], errors='coerce')
        x_count = len(soft[soft['flux'] >= 1e-4])
        m_count = len(soft[(soft['flux'] >= 1e-5) & (soft['flux'] < 1e-4)])
        c_count = len(soft[(soft['flux'] >= 1e-6) & (soft['flux'] < 1e-5)])
        q_count = len(soft[soft['flux'] < 1e-6])

        print(f"\nFlux level breakdown:")
        print(f"  X-class minutes: {x_count}")
        print(f"  M-class minutes: {m_count}")
        print(f"  C-class minutes: {c_count}")
        print(f"  Quiet minutes:   {q_count}")
        print("=" * 60)

        return combined

    return None

if __name__ == "__main__":
    df = fetch_goes_archive_bulk()
    if df is not None:
        print("\nData download complete — run build_catalog.py next")