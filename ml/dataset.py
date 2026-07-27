import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SolarFlareDataset(Dataset):
    def __init__(self, catalog_path, flux_path, window_size=60):
        self.window_size = window_size
        
        # Ingest structural datasets
        self.catalog = pd.read_csv(catalog_path)
        raw_flux = pd.read_csv(flux_path)
        
        # Clean timestamps, enforce chronological order, and strip timezones (.dt.tz_localize(None))
        raw_flux['time_tag'] = pd.to_datetime(raw_flux['time_tag']).dt.tz_localize(None)
        raw_flux = raw_flux.sort_values('time_tag').reset_index(drop=True)
        
        print("Pivoting data channels into separate Soft and Hard X-ray tracks...")
        pivoted_df = raw_flux.pivot_table(
            index='time_tag', 
            columns='energy', 
            values='flux', 
            aggfunc='mean'
        ).reset_index()
        
        # Clean column names to prevent channel mismatch errors
        pivoted_df = pivoted_df.rename(columns={
            '0.1-0.8nm': 'soft_flux',
            '0.05-0.4nm': 'hard_flux'
        })
        
        # Backfill any missing entries to maintain sequence continuity
        pivoted_df = pivoted_df.ffill().bfill()
        self.flux_df = pivoted_df
        
        self.X, self.y = self._build_tensor_windows()
        
    def _build_tensor_windows(self):
        X_list, y_list = [], []
        
        for _, row in self.catalog.iterrows():
            # Force catalog timestamps to also be tz-naive for flawless array comparison
            peak_time = pd.to_datetime(row['peak_time']).tz_localize(None)
            is_flare = 1 if str(row['flare_class']) != 'nan' and str(row['flare_class']) != 'QUIET' else 0
            
            # Find closest matching timestamp sequence index within the pivoted dataset
            idx_list = self.flux_df.index[self.flux_df['time_tag'] <= peak_time]
            if len(idx_list) == 0:
                continue
            end_idx = idx_list[-1]
            start_idx = end_idx - self.window_size
            
            if start_idx < 0:
                continue
                
            # Slice trailing sequence context array
            window_slice = self.flux_df.iloc[start_idx:end_idx]
            
            soft_chan = window_slice['soft_flux'].values
            hard_chan = window_slice['hard_flux'].values
            
            # Verify arrays are completely filled to maintain fixed shape dimensions (60, 2)
            if len(soft_chan) == self.window_size and len(hard_chan) == self.window_size:
                features = np.stack([soft_chan, hard_chan], axis=1)
                X_list.append(features)
                y_list.append(is_flare)
        
        if len(X_list) == 0:
            raise ValueError("Data pipeline Error: 0 windows generated. Check timestamp overlapping bounds.")
            
        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
