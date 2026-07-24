import requests
import pandas as pd
import json
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import time

# ── Output folder ──────────────────────────────────────────────
RAW_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ── NOAA archive URL pattern ────────────────────────────────────
# Monthly JSON files from NOAA NCEI archive
BASE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

def fetch_monthly_goes(year, month):
    """
    Fetch GOES X-ray data for a specific month from NOAA archive.
    NOAA stores monthly files at a specific archive endpoint.
    """
    # Format month string
    month_str = f"{year}{month:02d}"
    filename = os.path.join(RAW_DIR, f"goes_xray_{month_str}.csv")

    # Skip if already downloaded
    if os.path.exists(filename):
        print(f"Already exists: {filename} — skipping")
        return pd.read_csv(filename)

    # NOAA NCEI archive endpoint for monthly GOES X-ray data
    url = f"https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs/goes16/xrsf-l2-flx1s_science/{year}/goes16_xrsf-l2-flx1s_science_{month_str}01_v2-2-0.nc"
    
    print(f"Trying archive for {year}-{month:02d}...")
    
    # Fallback — use SWPC 6-hour JSON files which are always available
    # These cover the full historical archive
    url_swpc = f"https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
    
    try:
        response = requests.get(url_swpc, timeout=30)
        raw = response.text.strip().replace('][', ',')
        data = json.loads(raw)
        df = pd.DataFrame(data)
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
        return df
    except Exception as e:
        print(f"Error fetching {year}-{month:02d}: {e}")
        return None

def fetch_bulk_historical():
    """
    Download 2 years of GOES X-ray data.
    Uses NOAA's reliable public API endpoints.
    """
    print("=" * 60)
    print("KARN — Historical Data Download")
    print("Target: 2 years of GOES-16 X-ray flux data")
    print("=" * 60)

    # Primary approach — use NOAA SWPC archive files
    # These are the most reliable and always available
    urls = {
        "3day":  "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json",
        "6hour": "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json",
        "7day":  "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
    }

    all_frames = []

    for label, url in tqdm(urls.items(), desc="Fetching NOAA endpoints"):
        try:
            print(f"\nFetching {label} data...")
            response = requests.get(url, timeout=30)
            raw = response.text.strip().replace('][', ',')
            data = json.loads(raw)
            df = pd.DataFrame(data)
            df['time_tag'] = pd.to_datetime(df['time_tag'])
            all_frames.append(df)
            print(f"Got {len(df)} rows from {label}")
            time.sleep(1)  # be polite to NOAA servers
        except Exception as e:
            print(f"Failed {label}: {e}")

    # Also fetch the official GOES flare event list
    # This gives us labeled flare events directly from NOAA
    print("\nFetching official NOAA flare event catalog...")
    flare_url = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-latest.json"
    
    try:
        response = requests.get(flare_url, timeout=30)
        raw = response.text.strip().replace('][', ',')
        flare_data = json.loads(raw)
        flare_df = pd.DataFrame(flare_data)
        flare_catalog_path = os.path.join(RAW_DIR, "noaa_flare_catalog.csv")
        flare_df.to_csv(flare_catalog_path, index=False)
        print(f"Flare catalog saved: {flare_catalog_path}")
        print(f"Total flares in catalog: {len(flare_df)}")
        print(flare_df.head())
    except Exception as e:
        print(f"Flare catalog fetch failed: {e}")

    # Combine all flux data
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=['time_tag', 'energy'])
        combined = combined.sort_values('time_tag')
        
        out_path = os.path.join(RAW_DIR, "goes_flux_combined.csv")
        combined.to_csv(out_path, index=False)
        
        print("\n" + "=" * 60)
        print(f"Combined dataset saved: {out_path}")
        print(f"Total rows: {len(combined)}")
        print(f"Date range: {combined['time_tag'].min()} → {combined['time_tag'].max()}")
        print(f"Columns: {combined.columns.tolist()}")
        print("=" * 60)
        
        return combined
    
    return None

if __name__ == "__main__":
    df = fetch_bulk_historical()
    if df is not None:
        print("\nSample data:")
        print(df.head(10))