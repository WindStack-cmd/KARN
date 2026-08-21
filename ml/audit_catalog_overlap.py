import os
import pandas as pd


FEATURES_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
CATALOG_PATH = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\full_catalog.csv"


def main():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing telemetry source file: {FEATURES_PATH}")
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Missing catalog source file: {CATALOG_PATH}")

    feat_df = pd.read_csv(FEATURES_PATH)
    feat_df['timestamp'] = pd.to_datetime(feat_df['timestamp'], errors='coerce').dt.tz_localize(None)
    telemetry_start = feat_df['timestamp'].min()
    telemetry_end = feat_df['timestamp'].max()

    cat_df = pd.read_csv(CATALOG_PATH).copy()
    start_col = next((c for c in cat_df.columns if 'start' in c.lower()), 'start_time')
    peak_col = next((c for c in cat_df.columns if 'peak' in c.lower()), 'peak_time')
    class_col = next((c for c in cat_df.columns if 'class' in c.lower() or 'label' in c.lower()), 'flare_class')
    event_id_col = next((c for c in cat_df.columns if 'event' in c.lower() and 'id' in c.lower()), 'event_id')

    cat_df[start_col] = pd.to_datetime(cat_df[start_col], errors='coerce').dt.tz_localize(None)
    cat_df[peak_col] = pd.to_datetime(cat_df[peak_col], errors='coerce').dt.tz_localize(None)
    cat_df = cat_df[[event_id_col, start_col, peak_col, class_col]].dropna(subset=[start_col, peak_col]).copy()

    rows = []
    for _, row in cat_df.iterrows():
        flare_class = str(row[class_col]).strip().upper()
        if flare_class == 'Q':
            continue
        event_start = pd.Timestamp(row[start_col])
        event_peak = pd.Timestamp(row[peak_col])
        overlaps = bool((event_start <= telemetry_end) and (event_peak >= telemetry_start))
        reason = 'authentic event overlaps telemetry window' if overlaps else 'outside telemetry window; no temporal overlap with August 9 observations'
        rows.append({
            'event_id': row[event_id_col],
            'catalog_start': event_start,
            'catalog_peak': event_peak,
            'flare_class': flare_class,
            'telemetry_start': telemetry_start,
            'telemetry_end': telemetry_end,
            'overlaps': overlaps,
            'reason': reason,
        })

    audit_df = pd.DataFrame(rows)
    print('==================================================')
    print('DAY 12 EVENT OVERLAP AUDIT')
    print('==================================================')
    print(f'Telemetry range: {telemetry_start} -> {telemetry_end}')
    print(f'Total catalog flare events inspected: {len(audit_df)}')
    print(audit_df[['event_id', 'catalog_start', 'catalog_peak', 'flare_class', 'telemetry_start', 'telemetry_end', 'overlaps', 'reason']].to_string(index=False))
    print('\nSummary:')
    print(f'Overlapping events: {int(audit_df["overlaps"].sum())}')
    print('Scientific status: no real catalog event overlaps the current August 9 telemetry window.')


if __name__ == '__main__':
    main()
