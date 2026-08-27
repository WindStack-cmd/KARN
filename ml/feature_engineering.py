import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def compute_temporal_features():
    input_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_synced_timeline.csv"
    output_features_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
    
    master_df = pd.read_csv(input_path)
    master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
    
    all_featured_partitions = []
    
    # Process rolling windows independently per day folder block to guarantee zero cross-day leakage
    for part_id, df in master_df.groupby('partition_id'):
        print(f"Computing multi-resolution feature vectors for: {part_id}")
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        time_deltas = df['timestamp'].diff().dt.total_seconds().fillna(1.0)
        dt = np.where(time_deltas <= 0, 1.0, time_deltas)
        
        df['solexs_rate'] = df['solexs_flux'].diff().fillna(0) / dt
        df['hel1os_rate'] = df['hel1os_flux'].diff().fillna(0) / dt
        df['soft_hard_ratio'] = df['solexs_flux'] / (df['hel1os_flux'] + 1e-5)
        
        window_sizes = {'5m': 300, '30m': 1800, '1h': 3600, '6h': 21600}
        for label, size in window_sizes.items():
            if len(df) < size:
                size = max(5, len(df) // 4)
            df[f'solexs_mean_{label}'] = df['solexs_flux'].rolling(window=size, closed='left').mean()
            df[f'solexs_std_{label}'] = df['solexs_flux'].rolling(window=size, closed='left').std()
            df[f'hel1os_mean_{label}'] = df['hel1os_flux'].rolling(window=size, closed='left').mean()
            df[f'hel1os_std_{label}'] = df['hel1os_flux'].rolling(window=size, closed='left').std()
            
        df = df.bfill().ffill()
        
        h_median = df['hel1os_flux'].median()
        h_mad = np.median(np.abs(df['hel1os_flux'] - h_median))
        h_mad = 1.0 if h_mad == 0 else h_mad
        df['anomaly_flag'] = ((df['hel1os_flux'] - h_median) / h_mad > 15.0).astype(int)
        df['quality_flag'] = df['solexs_flux'].isna().astype(int)
        
        all_featured_partitions.append(df)
        
    final_features_df = pd.concat(all_featured_partitions, axis=0).reset_index(drop=True)
    
    # Safe numerical scaling checks across array shapes
    feature_cols = [c for c in final_features_df.columns if c not in ['timestamp', 'time_seconds', 'partition_id', 'anomaly_flag', 'quality_flag']]
    for col in feature_cols:
        final_features_df[col] = final_features_df[col].replace([np.inf, -np.inf], np.nan).interpolate(method='linear').fillna(0)
        
    # Time-split partition scaling execution
    split_idx = int(0.8 * len(final_features_df))
    train_df = final_features_df.iloc[:split_idx].copy()
    test_df = final_features_df.iloc[split_idx:].copy()
    
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])
    
    train_scaled = scaler.transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    train_scaled_df = pd.DataFrame(train_scaled, columns=[f"{c}_scaled" for c in feature_cols], index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=[f"{c}_scaled" for c in feature_cols], index=test_df.index)
    
    meta_cols = ['timestamp', 'time_seconds', 'anomaly_flag', 'quality_flag']
    final_train = pd.concat([train_df[meta_cols], train_scaled_df], axis=1)
    final_test = pd.concat([test_df[meta_cols], test_scaled_df], axis=1)
    
    final_dataset = pd.concat([final_train, final_test], axis=0).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    final_dataset.to_csv(output_features_path, index=False)
    print(f"🚀 Success! Multi-day feature engineering dataset built with: {len(final_dataset.columns)} columns across {len(final_dataset)} samples.")

if __name__ == "__main__":
    compute_temporal_features()
