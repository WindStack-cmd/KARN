import os
import pandas as pd
import matplotlib.pyplot as plt
from read_fits import extract_solexs_fits, extract_hel1os_fits

def run_synchronization_pipeline():
    base_raw_dir = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
    solexs_path = os.path.join(base_raw_dir, "solexs_sample.fits.gz")
    hel1os_path = os.path.join(base_raw_dir, "hel1os_sample.fits")
    
    print("Initializing Aditya-L1 Multi-Channel Processing Engine...\n")
    
    try:
        # Extract individual instrument data frames using the exact columns discovered
        solexs_df = extract_solexs_fits(solexs_path)
        hel1os_df = extract_hel1os_fits(hel1os_path)
    except Exception as e:
        print(f"\n❌ Diagnostic Processing Failure: {e}")
        return

    print("\nSynchronizing dual instruments along relative second timeline tracks...")
    
    # Merge timelines on matching time seconds
    # Rounding seconds to the nearest whole integer ensures perfect alignment points
    solexs_df['time_seconds'] = solexs_df['time_seconds'].round()
    hel1os_df['time_seconds'] = hel1os_df['time_seconds'].round()
    
    synced_df = pd.merge(solexs_df, hel1os_df, on='time_seconds', how='outer')
    synced_df = synced_df.sort_values('time_seconds').reset_index(drop=True)
    
    # Handle the master datetime column backfilling
    synced_df['timestamp'] = synced_df['timestamp'].ffill().bfill()
    
    print("Applying astronomical interpolation to resolve sensor gaps...")
    synced_df['solexs_flux'] = synced_df['solexs_flux'].interpolate(method='linear').ffill().bfill()
    synced_df['hel1os_flux'] = synced_df['hel1os_flux'].interpolate(method='linear').ffill().bfill()
    
    # Clean up and order the final dataset rows
    final_df = synced_df[['timestamp', 'time_seconds', 'solexs_flux', 'hel1os_flux']]
    
    output_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_synced_timeline.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\n✅ Success! Clean synchronized dataset exported to: {output_path}")
    
    # Generate the pristine scientific plot
    print("Generating validation plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    ax1.plot(final_df['timestamp'], final_df['solexs_flux'], color='#F97316', label='SoLEXS (Soft X-Ray Channel)')
    ax1.set_ylabel("Photon Counts (RATE)")
    ax1.set_title("ADITYA-L1 CORE DUAL-CHANNEL TELEMETRY TIMELINE (AUGUST 09, 2026)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    ax2.plot(final_df['timestamp'], final_df['hel1os_flux'], color='#3B82F6', label='HEL1OS (Full Band 18-160 keV Hard X-Ray)')
    ax2.set_ylabel("Photon Count Rates (CTR)")
    ax2.set_xlabel("Observation Time (UTC Tracker)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("aditya_validation_plot.png")
    print("Verification plot exported successfully to root directory as aditya_validation_plot.png")
    plt.show()

if __name__ == "__main__":
    run_synchronization_pipeline()
