import os
import pandas as pd

def generate_honest_labels():
    features_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
    catalog_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\official\noaa_ncei_goes_flare_report_2026.csv"
    output_labels_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_labels.csv"
    
    df_feat = pd.read_csv(features_path)
    df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
    
    # Initialize clean targets across all stacked partitions
    df_feat['flare_class_token'] = 0
    df_feat['time_to_peak_min'] = -1.0
    
    # Authoritative catalog definitions
    df_cat = pd.read_csv(catalog_path)
    required_columns = {'flare_id', 'start_time', 'time', 'end_time', 'flare_class'}
    missing_columns = required_columns.difference(df_cat.columns)
    if missing_columns:
        raise ValueError(f"Official NOAA/NCEI catalog is missing columns: {sorted(missing_columns)}")

    start_col = 'start_time'
    peak_col = 'time'
    end_col = 'end_time'
    class_col = 'flare_class'
    
    df_cat[start_col] = pd.to_datetime(df_cat[start_col], utc=True, errors='coerce')
    df_cat[peak_col] = pd.to_datetime(df_cat[peak_col], utc=True, errors='coerce')
    df_cat[end_col] = pd.to_datetime(df_cat[end_col], utc=True, errors='coerce')
    df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'], utc=True, errors='raise')
    
    class_map = {'C': 1, 'M': 2, 'X': 3}
    overlap_count = 0
    
    print("Performing multi-day absolute-time checking against the flare catalog...")
    for _, row in df_cat.dropna(subset=[start_col, peak_col]).iterrows():
        s_time = row[start_col]
        p_time = row[peak_col]
        f_cls = str(row[class_col]).upper().strip()
        
        token = class_map.get(f_cls[:1], 0) if f_cls else 0
        if f_cls and f_cls[:1] not in class_map:
            continue
        
        if token == 0:
            continue
            
        mask = (df_feat['timestamp'] >= s_time) & (df_feat['timestamp'] <= p_time)
        if mask.sum() > 0:
            overlap_count += 1
            df_feat.loc[mask, 'flare_class_token'] = token
            df_feat.loc[mask, 'time_to_peak_min'] = (p_time - df_feat.loc[mask, 'timestamp']).dt.total_seconds() / 60.0

    print(f"\nOfficial NOAA/NCEI overlap summary: {overlap_count} C/M/X events.")
    print(df_feat['flare_class_token'].value_counts())
    
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    df_feat[['timestamp', 'flare_class_token', 'time_to_peak_min']].to_csv(output_labels_path, index=False)
    print("Official NOAA/NCEI target labels exported.")

if __name__ == "__main__":
    generate_honest_labels()
