import os
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from astropy.io import fits

def extract_solexs_fits(file_path):
    print(f"Parsing SoLEXS Level-1 FITS: {os.path.basename(file_path)}")
    with fits.open(file_path) as hdul:
        # Access Extension 1 ('RATE') directly
        data = hdul[1].data
        
        # Extract headers. SoLEXS 'TIME' column is seconds from an observation reference time.
        # We will keep raw TIME for merging or convert using the header reference epoch.
        time_array = np.array(data['TIME'], dtype=float)
        counts_array = np.array(data['COUNTS'], dtype=float)
        
        # Convert relative observation seconds into structural timeline indexes
        df = pd.DataFrame({
            'time_seconds': time_array,
            'solexs_flux': counts_array
        })
        return df.sort_values('time_seconds').reset_index(drop=True)

def extract_hel1os_fits(file_path):
    print(f"Parsing HEL1OS Level-1 FITS: {os.path.basename(file_path)}")
    with fits.open(file_path) as hdul:
        # Load Extension 5 (The full broad spectrum 18 keV - 160 keV hard X-ray channel)
        data = hdul[5].data
        
        # ISOT holds clean string timestamps like '2026-08-09T12:00:04'
        isot_time = np.array(data['ISOT'], dtype=str)
        # CTR holds the clean photon count tracking rates
        ctr_rate = np.array(data['CTR'], dtype=float)
        
        # Map into a data frame, converting string stamps into datetime objects
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(isot_time),
            'hel1os_flux': ctr_rate
        })
        
        # Strip timezones and compute relative second offsets matching today's target date base
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        start_time = df['timestamp'].min()
        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
        
        return df[['time_seconds', 'timestamp', 'hel1os_flux']].sort_values('time_seconds').reset_index(drop=True)
