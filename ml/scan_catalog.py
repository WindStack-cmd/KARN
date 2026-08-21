import os
import pandas as pd

cat_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\full_catalog.csv"

if os.path.exists(cat_path):
    df = pd.read_csv(cat_path)
    print("--- Space Weather Catalog Audit ---")
    print(f"• Total Recorded Events in Catalog: {len(df)}")
    print(f"• Column Names found: {list(df.columns)}")
    
    # Identify key date and classification columns dynamically
    start_col = [c for c in df.columns if 'start' in c.lower()][0]
    class_col = [c for c in df.columns if 'class' in c.lower() or 'label' in c.lower()][0]
    
    df[start_col] = pd.to_datetime(df[start_col]).dt.tz_localize(None)
    
    # Check if any flares are recorded for August 2026
    august_flares = df[df[start_col].dt.strftime('%Y-%m') == '2026-08']
    print(f"• Real Flare Events found in August 2026: {len(august_flares)}")
    
    print("\n--- Sample Flare Entries from the Catalog ---")
    print(df[[start_col, class_col]].head(10).to_string(index=False))
else:
    print(f"❌ Cannot find catalog file at: {cat_path}")
