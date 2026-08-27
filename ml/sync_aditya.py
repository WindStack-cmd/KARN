import os
import pandas as pd
from read_fits import extract_solexs_fits, extract_hel1os_fits

def run_synchronization_pipeline():
    base_raw_dir = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
    processed_dir = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed"
    
    day_folders = [d for d in os.listdir(base_raw_dir) if os.path.isdir(os.path.join(base_raw_dir, d)) and d.startswith('day_')]
    print(f"Initializing Aditya-L1 Multi-Day Processing Engine... Found directories: {day_folders}\n")
    
    all_timelines = []
    
    for day in sorted(day_folders):
        print(f"=== Processing Sub-Timeline Partition: {day} ===")
        day_path = os.path.join(base_raw_dir, day)
        solexs_path = os.path.join(day_path, "solexs_sample.fits.gz")
        hel1os_path = os.path.join(day_path, "hel1os_sample.fits")
        
        try:
            solexs_df = extract_solexs_fits(solexs_path)
            hel1os_df = extract_hel1os_fits(hel1os_path)
            
            # Align time tracking shapes on standard datetime objects, ignoring minor second rounding shifts
            solexs_df['timestamp_round'] = solexs_df['timestamp'].dt.round('1s')
            hel1os_df['timestamp_round'] = hel1os_df['timestamp'].dt.round('1s')
            
            synced_df = pd.merge(solexs_df, hel1os_df, on='timestamp_round', how='outer', suffixes=('_sol', '_hel'))
            synced_df = synced_df.sort_values('timestamp_round').reset_index(drop=True)
            
            # Unify master tracking metrics
            synced_df['timestamp'] = synced_df['timestamp_round']
            synced_df['time_seconds'] = (synced_df['timestamp'] - synced_df['timestamp'].min()).dt.total_seconds()
            
            synced_df['solexs_flux'] = synced_df['solexs_flux'].interpolate(method='linear').ffill().bfill()
            synced_df['hel1os_flux'] = synced_df['hel1os_flux'].interpolate(method='linear').ffill().bfill()
            
            final_day_df = synced_df[['timestamp', 'time_seconds', 'solexs_flux', 'hel1os_flux']].copy()
            final_day_df['partition_id'] = day
            
            all_timelines.append(final_day_df)
            print(f"✅ Partition {day} successfully synced with {len(final_day_df)} time steps.")
            
        except Exception as e:
            print(f"❌ Critical Processing Error in {day}: {e}")
            continue
            
    if len(all_timelines) == 0:
        print("❌ No active partitions loaded. Pipeline halted.")
        return
        
    master_timeline = pd.concat(all_timelines, axis=0).reset_index(drop=True)
    output_path = os.path.join(processed_dir, "aditya_synced_timeline.csv")
    os.makedirs(processed_dir, exist_ok=True)
    master_timeline.to_csv(output_path, index=False)
    print(f"\n🚀 Master Combined Pipeline Matrix Exported: {output_path} ({len(master_timeline)} total time steps)")

if __name__ == "__main__":
    run_synchronization_pipeline()
