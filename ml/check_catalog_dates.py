import pandas as pd
import os

def check_catalog():
    catalog_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw\noaa_flare_catalog.csv"
    if not os.path.exists(catalog_path):
        print("❌ Catalog file not found.")
        return
        
    df = pd.read_csv(catalog_path)
    print("\n=== AUTHORITATIVE FLARE CATALOG SNAPSHOT ===")
    print(f"Total historical events logged in catalog: {len(df)}")
    print("\nFirst 10 events in your catalog:")
    print(df.head(10))
    
    # Extract unique dates present in your catalog to help you map against PRADAN dropdowns
    df['date_only'] = pd.to_datetime(df[df.columns[0]]).dt.date
    unique_dates = df['date_only'].unique()
    print(f"\n💡 Unique dates with real documented flares in your catalog ({len(unique_dates)} total):")
    for d in sorted(unique_dates)[:15]:
        print(f" - {d}")

if __name__ == "__main__":
    check_catalog()
