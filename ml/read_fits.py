import os
import pandas as pd
import numpy as np
from astropy.io import fits

def extract_solexs_fits(file_path):
    print(f"Parsing SoLEXS Level-1 FITS: {os.path.basename(file_path)}")
    with fits.open(file_path) as hdul:
        # Access data and header explicitly from Extension 1 (the 'RATE' BinTableHDU)
        data = hdul[1].data
        header = hdul[1].header
        
        # Pull observation reference time from header cards, defaulting to 2026-08-01 if blank
        ref_date = pd.to_datetime(header.get('DATE-OBS', '2026-08-01')).normalize()
        
        time_array = np.array(data['TIME'], dtype=float)
        counts_array = np.array(data['COUNTS'], dtype=float)
        
        df = pd.DataFrame({
            'time_seconds': time_array,
            'solexs_flux': counts_array
        })
        
        # Build clean absolute timestamps by adding the relative seconds to the observation epoch
        df['timestamp'] = ref_date + pd.to_timedelta(df['time_seconds'], unit='s')
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        return df.sort_values('time_seconds').reset_index(drop=True)

def extract_hel1os_fits(file_path):
    print(f"Parsing HEL1OS Level-1 FITS: {os.path.basename(file_path)}")
    with fits.open(file_path) as hdul:
        # Access Extension 5 (Broad broad-spectrum 18 keV - 160 keV hard X-ray band)
        data = hdul[5].data
        
        isot_time = np.array(data['ISOT'], dtype=str)
        ctr_rate = np.array(data['CTR'], dtype=float)
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(isot_time),
            'hel1os_flux': ctr_rate
        })
        
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        start_time = df['timestamp'].min()
        df['time_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
        
        return df[['time_seconds', 'timestamp', 'hel1os_flux']].sort_values('time_seconds').reset_index(drop=True)
