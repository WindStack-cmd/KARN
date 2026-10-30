import os
import pandas as pd


def run_scientific_audit():
    features_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
    catalog_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\official\noaa_ncei_goes_flare_report_2026.csv"

    feat_df = pd.read_csv(features_path)
    feat_df['timestamp'] = pd.to_datetime(feat_df['timestamp'], errors='coerce', utc=True)
    telemetry_start = feat_df['timestamp'].min()
    telemetry_end = feat_df['timestamp'].max()

    cat_df = pd.read_csv(catalog_path)
    cat_df['start_time'] = pd.to_datetime(cat_df['start_time'], errors='coerce', utc=True)
    cat_df['peak_time'] = pd.to_datetime(cat_df['time'], errors='coerce', utc=True)
    cat_df = cat_df[['flare_id', 'start_time', 'peak_time', 'flare_class']].dropna(subset=['start_time', 'peak_time']).copy()
    cat_df = cat_df.rename(columns={'flare_id': 'event_id'})

    rows = []
    for _, row in cat_df.iterrows():
        flare_class = str(row['flare_class']).strip().upper()
        if flare_class == 'Q':
            continue
        overlaps = bool((row['start_time'] <= telemetry_end) and (row['peak_time'] >= telemetry_start))
        rows.append({
            'event_id': row['event_id'],
            'catalog_start': row['start_time'],
            'catalog_peak': row['peak_time'],
            'flare_class': flare_class,
            'telemetry_start': telemetry_start,
            'telemetry_end': telemetry_end,
            'overlaps': overlaps,
            'reason': 'actual overlap with the telemetry window' if overlaps else 'outside telemetry window; no true temporal overlap'
        })

    audit_df = pd.DataFrame(rows)
    print("=============================================")
    print("KARN SCIENTIFIC INTEGRITY AUDIT REPORT")
    print("=============================================")
    print(f"Telemetry window: {telemetry_start} -> {telemetry_end}")
    print(f"Real non-Q catalog events overlapping telemetry: {int(audit_df['overlaps'].sum())}")
    print(audit_df[['event_id', 'catalog_start', 'catalog_peak', 'flare_class', 'telemetry_start', 'telemetry_end', 'overlaps', 'reason']].to_string(index=False))
    print("=============================================")


if __name__ == "__main__":
    run_scientific_audit()
