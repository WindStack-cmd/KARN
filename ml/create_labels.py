import os
import numpy as np
import pandas as pd


def generate_scientific_targets():
    features_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
    catalog_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\full_catalog.csv"
    output_labels_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_labels.csv"

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features source file: {features_path}")
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Missing catalog source file: {catalog_path}")

    feat_df = pd.read_csv(features_path)
    feat_df['timestamp'] = pd.to_datetime(feat_df['timestamp']).dt.tz_localize(None)
    feat_df = feat_df.sort_values('timestamp').reset_index(drop=True)

    telemetry_start = feat_df['timestamp'].min()
    telemetry_end = feat_df['timestamp'].max()

    feat_df['flare_label'] = 0
    feat_df['flare_class'] = 0
    feat_df['time_to_peak'] = -1.0
    feat_df['event_id'] = 'QUIET_SUN_BACKGROUND'

    cat_df = pd.read_csv(catalog_path).copy()
    start_col = next((c for c in cat_df.columns if 'start' in c.lower()), 'start_time')
    peak_col = next((c for c in cat_df.columns if 'peak' in c.lower()), 'peak_time')
    class_col = next((c for c in cat_df.columns if 'class' in c.lower() or 'label' in c.lower()), 'flare_class')
    event_id_col = next((c for c in cat_df.columns if 'event' in c.lower() and 'id' in c.lower()), 'event_id')

    cat_df[start_col] = pd.to_datetime(cat_df[start_col], errors='coerce').dt.tz_localize(None)
    cat_df[peak_col] = pd.to_datetime(cat_df[peak_col], errors='coerce').dt.tz_localize(None)
    cat_df = cat_df[[start_col, peak_col, class_col, event_id_col]].dropna(subset=[start_col, peak_col]).copy()

    class_map = {'C': 1, 'M': 2, 'X': 3}
    overlapping_events = []
    for _, event in cat_df.iterrows():
        event_start = pd.Timestamp(event[start_col])
        event_peak = pd.Timestamp(event[peak_col])
        flare_class = str(event[class_col]).strip().upper()
        if flare_class == 'Q':
            continue
        if event_start <= telemetry_end and event_peak >= telemetry_start:
            overlapping_events.append((event, event_start, event_peak, flare_class))

    print('==================================================')
    print('KARN DAY 12 ABSOLUTE-TIME GROUND TRUTH AUDIT')
    print('==================================================')
    print(f'Telemetry range: {telemetry_start} -> {telemetry_end}')
    print(f'Catalog events: {len(cat_df)}')
    print(f'Events overlapping telemetry: {len(overlapping_events)}')
    print(f'Events outside telemetry: {len(cat_df) - len(overlapping_events)}')
    print('Synthetic events: 0')
    print('Timestamp shifting: 0')
    print('Catalog timestamp modification: 0')

    if not overlapping_events:
        print('WARNING: This telemetry window contains no temporally overlapping authentic catalog flare events.')
        print('The current August 9 dataset is unsuitable for supervised flare-classification performance evaluation until correctly dated flare-overlapping telemetry is available.')
        print('No synthetic flare labels were generated.')

    for event, event_start, event_peak, flare_class in overlapping_events:
        event_id = str(event[event_id_col])
        class_token = class_map.get(flare_class[0], 0)
        if class_token == 0:
            continue

        forecast_start = event_start - pd.Timedelta(minutes=30)
        mask = (feat_df['timestamp'] >= forecast_start) & (feat_df['timestamp'] <= event_peak)
        mask &= (feat_df['timestamp'] >= telemetry_start) & (feat_df['timestamp'] <= telemetry_end)

        if not mask.any():
            continue

        feat_df.loc[mask, 'flare_label'] = 1
        feat_df.loc[mask, 'event_id'] = f'REAL_CATALOG_{event_id}'
        feat_df.loc[mask, 'flare_class'] = np.maximum(feat_df.loc[mask, 'flare_class'].to_numpy(), class_token)

        rem_minutes = (event_peak - feat_df.loc[mask, 'timestamp']).dt.total_seconds() / 60.0
        current_ttp = feat_df.loc[mask, 'time_to_peak']
        feat_df.loc[mask, 'time_to_peak'] = np.where((current_ttp == -1.0) | (rem_minutes < current_ttp), rem_minutes, current_ttp)

    dist = feat_df['flare_class'].value_counts().to_dict()
    print(f'Total labeled rows: {len(feat_df)}')
    print(f'Quiet rows: {dist.get(0, 0)}')
    print(f'C rows: {dist.get(1, 0)}')
    print(f'M rows: {dist.get(2, 0)}')
    print(f'X rows: {dist.get(3, 0)}')

    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    output_cols = ['timestamp', 'flare_label', 'flare_class', 'time_to_peak', 'event_id']
    feat_df[output_cols].to_csv(output_labels_path, index=False)
    print(f'Ground-truth labels saved to: {output_labels_path}')

    if not overlapping_events:
        print('This dataset is scientifically valid only as an all-quiet no-overlap telemetry record. It is not suitable for supervised solar flare training/evaluation.')


if __name__ == '__main__':
    generate_scientific_targets()
