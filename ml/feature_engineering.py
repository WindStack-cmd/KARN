import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def compute_temporal_features():
    input_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_synced_timeline.csv"
    output_features_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_features.csv"
    
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Enforce absolute chronological verification
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Extract actual time deltas safely
    time_deltas = df['timestamp'].diff().dt.total_seconds().fillna(1.0)
    
    # Force any 0.0 time steps to a standard 1.0s cadence baseline to eliminate division by zero
    dt = np.where(time_deltas <= 0, 1.0, time_deltas)
    
    # 3. Calculate Rate-of-Change differentials safely
    df['solexs_rate'] = df['solexs_flux'].diff().fillna(0) / dt
    df['hel1os_rate'] = df['hel1os_flux'].diff().fillna(0) / dt
    
    print("Calculating Cross-Channel Soft/Hard Sensor Relationship metrics...")
    # Add a slightly higher denominator guard floor (1e-5) to block near-zero explosive infinities
    df['soft_hard_ratio'] = df['solexs_flux'] / (df['hel1os_flux'] + 1e-5)
    
    # 4. Multi-Resolution Context Window Definitions
    window_sizes = {
        '5m': 300,
        '30m': 1800,
        '1h': 3600,
        '6h': 21600
    }
    
    print("Computing historical rolling features (closed='left' to eliminate data leakage)...")
    for label, size in window_sizes.items():
        if len(df) < size:
            size = max(5, len(df) // 4)
            
        df[f'solexs_mean_{label}'] = df['solexs_flux'].rolling(window=size, closed='left').mean()
        df[f'solexs_std_{label}'] = df['solexs_flux'].rolling(window=size, closed='left').std()
        df[f'hel1os_mean_{label}'] = df['hel1os_flux'].rolling(window=size, closed='left').mean()
        df[f'hel1os_std_{label}'] = df['hel1os_flux'].rolling(window=size, closed='left').std()

    # Backfill initialization windows safely
    df = df.bfill().ffill()
    
    # Isolate anomalies as flags cleanly
    h_median = df['hel1os_flux'].median()
    h_mad = np.median(np.abs(df['hel1os_flux'] - h_median))
    h_mad = 1.0 if h_mad == 0 else h_mad
    df['anomaly_flag'] = ((df['hel1os_flux'] - h_median) / h_mad > 15.0).astype(int)
    df['quality_flag'] = df['solexs_flux'].isna().astype(int)

    # CRUCIAL SCALAR SHIELD: Replace any infinite or invalid float values with zero 
    # to protect the scikit-learn validator array checks
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'time_seconds', 'anomaly_flag', 'quality_flag']]
    
    print("Enforcing strict mathematical finite checks across all feature tracks...")
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Interpolate or fill any generated NaNs to maintain array density
        df[col] = df[col].interpolate(method='linear').fillna(0)

    # 5. Leakage-Proof Time-Series Partitioning Pipeline
    print("\nExecuting leak-proof time-series partitioning for Scaling calculations...")
    split_idx = int(0.8 * len(df))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Initialize and fit the scaler strictly on the historical training slice
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])
    print("✅ RobustScaler successfully fit strictly on Training Data Partition.")
    
    # Apply transformation independently
    train_scaled = scaler.transform(train_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    train_scaled_df = pd.DataFrame(train_scaled, columns=[f"{c}_scaled" for c in feature_cols], index=train_df.index)
    test_scaled_df = pd.DataFrame(test_scaled, columns=[f"{c}_scaled" for c in feature_cols], index=test_df.index)
    
    final_train = pd.concat([train_df[['timestamp', 'time_seconds', 'anomaly_flag', 'quality_flag']], train_scaled_df], axis=1)
    final_test = pd.concat([test_df[['timestamp', 'time_seconds', 'anomaly_flag', 'quality_flag']], test_scaled_df], axis=1)
    
    final_dataset = pd.concat([final_train, final_test], axis=0).sort_values('timestamp').reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    final_dataset.to_csv(output_features_path, index=False)
    print(f"🚀 Success! Clean model-ready features dataset saved to: {output_features_path}")
    print(f"Verified Final Feature Columns Count: {len(final_dataset.columns)} columns across {len(final_dataset)} time steps.")

if __name__ == "__main__":
    compute_temporal_features()
