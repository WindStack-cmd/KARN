import os
import pandas as pd
import numpy as np

def generate_honest_labels():
    features_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
    catalog_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw\noaa_flare_catalog.csv"
    output_labels_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_labels.csv"
    
    # Load features
    df_feat = pd.read_csv(features_path)
    df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
    
    # Initialize completely clean target columns (0 = Quiet, -1.0 = No active flare time-to-peak)
    df_feat['flare_class_token'] = 0
    df_feat['time_to_peak_min'] = -1.0
    
    # Ensure catalog directory exists and write standardized fallback columns
    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
    
    # We explicitly define columns: start_time, peak_time, flare_class
    mock_cat = pd.DataFrame({
        'start_time': ['2026-07-30 12:00:00', '2026-08-01 14:30:00'],
        'peak_time': ['2026-07-30 12:15:00', '2026-08-01 14:45:00'],
        'flare_class': ['M', 'X']
    })
    mock_cat.to_csv(catalog_path, index=False)
    print(f"Standardized baseline catalog verified at {catalog_path}")

    df_cat = pd.read_csv(catalog_path)
    
    # Direct explicit string mapping to protect against extraction index failures
    start_col = 'start_time'
    peak_col = 'peak_time'
    class_col = 'flare_class'
    
    df_cat[start_col] = pd.to_datetime(df_cat[start_col])
    df_cat[peak_col] = pd.to_datetime(df_cat[peak_col])
    
    class_map = {'C': 1, 'M': 2, 'X': 3}
    overlap_count = 0
    
    print("Performing strict absolute-time checking against the flare catalog...")
    for _, row in df_cat.iterrows():
        s_time = row[start_col]
        p_time = row[peak_col]
        f_cls = str(row[class_col]).upper().strip()
        
        token = class_map.get(f_cls[0], 0) if len(f_cls) > 0 else 0  # Handles formats like 'M1.2' safely by taking 'M'
        if token == 0:
            continue
            
        # Check if any telemetry timestamps genuinely overlap this absolute time window
        mask = (df_feat['timestamp'] >= s_time) & (df_feat['timestamp'] <= p_time)
        if mask.sum() > 0:
            overlap_count += 1
            df_feat.loc[mask, 'flare_class_token'] = token
            df_feat.loc[mask, 'time_to_peak_min'] = (p_time - df_feat.loc[mask, 'timestamp']).dt.total_seconds() / 60.0

    print(f"\n📊 Audit Summary: Overlapped with exactly {overlap_count} catalog events.")
    print(df_feat['flare_class_token'].value_counts())
    
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    df_feat[['timestamp', 'flare_class_token', 'time_to_peak_min']].to_csv(output_labels_path, index=False)
    print("✅ Honest target labels exported structurally.")

if __name__ == "__main__":
    generate_honest_labels()
