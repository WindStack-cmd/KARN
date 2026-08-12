import os
import pandas as pd
import numpy as np

def run_quality_audit():
    input_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_synced_timeline.csv"
    output_report = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\data_quality_report.csv"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing synchronization source: {input_path}")
        
    print(f"Loading timeline array for quality sweep: {os.path.basename(input_path)}")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Audit 1: Chronological sorting checks
    is_sorted = df['timestamp'].is_monotonic_increasing
    
    # Audit 2: Telemetry Data Gaps / Time Diff calculations
    time_diffs = df['timestamp'].diff().dt.total_seconds()
    gaps_detected = (time_diffs > 1.0).sum() # Flags instances where sensor gaps skip seconds
    
    # Audit 3: Flag extreme Cosmic Ray / Particle strikes using statistical Z-Score thresholding
    h_median = df['hel1os_flux'].median()
    h_mad = np.median(np.abs(df['hel1os_flux'] - h_median)) # Median Absolute Deviation
    h_mad = 1.0 if h_mad == 0 else h_mad # Prevent divide-by-zero bounds
    
    # Flag rows where counts jump to extreme outliers (like your 650-count spike at 16:00)
    df['particle_outlier_flag'] = ((df['hel1os_flux'] - h_median) / h_mad > 15.0).astype(int)
    
    # Audit 4: Detect missing or invalid negative measurements
    df['invalid_data_flag'] = ((df['solexs_flux'] < 0) | (df['hel1os_flux'] < 0) | df['solexs_flux'].isna() | df['hel1os_flux'].isna()).astype(int)
    
    report_df = pd.DataFrame({
        'Metric': ['Chronological_Order_Valid', 'Total_Time_Steps', 'Missing_Data_Rows', 'Detected_Telemetry_Gaps', 'Particle_Strikes_Flagged'],
        'Value': [str(is_sorted), len(df), int(df['invalid_data_flag'].sum()), int(gaps_detected), int(df['particle_outlier_flag'].sum())]
    })
    
    report_df.to_csv(output_report, index=False)
    print(f"📊 Quality Audit Complete! Matrix report exported to: {output_report}")
    print(report_df)

if __name__ == "__main__":
    run_quality_audit()
