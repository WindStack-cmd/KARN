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
        
        # SoLEXS 'TIME' column is UNIX epoch seconds
        time_array = np.array(data['TIME'], dtype=float)
        counts_array = np.array(data['COUNTS'], dtype=float)
        
        df = pd.DataFrame({
            'time_seconds': time_array,
            'solexs_flux': counts_array
        })
        # Convert UNIX epoch seconds into datetime UTC timestamps matching HEL1OS
        df['timestamp'] = pd.to_datetime(df['time_seconds'], unit='s')
        
        return df[['time_seconds', 'timestamp', 'solexs_flux']].sort_values('time_seconds').reset_index(drop=True)

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
        
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        # Convert timestamp to absolute UNIX epoch seconds (keeping it absolute, consistent with SoLEXS)
        df['time_seconds'] = (df['timestamp'] - pd.Timestamp("1970-01-01")).dt.total_seconds()
        
        return df[['time_seconds', 'timestamp', 'hel1os_flux']].sort_values('time_seconds').reset_index(drop=True)
