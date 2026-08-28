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
        
        # TIME is measured in seconds from FITS MJDREF, not from DATE-OBS.  These
        # files specify MJDREF=40587 (Unix epoch), so adding TIME to DATE-OBS
        # previously produced the erroneous year-2083 timeline.
        mjd_ref_i = header.get('MJDREFI')
        mjd_ref_f = header.get('MJDREFF', 0.0)
        if mjd_ref_i is None:
            raise ValueError("SoLEXS RATE extension has no MJDREFI timing reference")
        mjd_ref = float(mjd_ref_i) + float(mjd_ref_f)
        
        time_array = np.array(data['TIME'], dtype=float)
        counts_array = np.array(data['COUNTS'], dtype=float)
        
        df = pd.DataFrame({
            'time_seconds': time_array,
            'solexs_flux': counts_array
        })
        
        # MJD is a defined FITS time coordinate: MJDREF + TIME / seconds-per-day.
        df['timestamp'] = pd.to_datetime(mjd_ref + df['time_seconds'] / 86400.0, unit='D', origin='1858-11-17', utc=True).dt.tz_localize(None)
        
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
