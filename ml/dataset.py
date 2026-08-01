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
        
        # Clean timestamps and strip timezones
        raw_flux['time_tag'] = pd.to_datetime(raw_flux['time_tag']).dt.tz_localize(None)
        raw_flux = raw_flux.sort_values('time_tag').reset_index(drop=True)
        
        print("Pivoting data channels into separate Soft and Hard X-ray tracks...")
        pivoted_df = raw_flux.pivot_table(
            index='time_tag', 
            columns='energy', 
            values='flux', 
            aggfunc='mean'
        ).reset_index()
        
        pivoted_df = pivoted_df.rename(columns={
            '0.1-0.8nm': 'soft_flux',
            '0.05-0.4nm': 'hard_flux'
        })
        
        pivoted_df = pivoted_df.ffill().bfill()
        self.flux_df = pivoted_df
        
        # Map class strings to numerical integers
        # Q = 0, C = 1, M = 2, X = 3
        self.class_map = {'QUIET': 0, 'C': 1, 'M': 2, 'X': 3}
        
        self.X, self.y = self._build_tensor_windows()
        
    def _build_tensor_windows(self):
        X_list, y_list = [], []
        
        for _, row in self.catalog.iterrows():
            peak_time = pd.to_datetime(row['peak_time']).tz_localize(None)
            
            # Extract the explicit flare class string from your catalog file
            f_class = str(row['flare_class']).strip().upper()
            
            # Fallback handling to protect against malformed rows
            if f_class not in self.class_map or pd.isna(row['flare_class']):
                f_token = 0  # Default to Quiet if unknown
            else:
                f_token = self.class_map[f_class]
            
            idx_list = self.flux_df.index[self.flux_df['time_tag'] <= peak_time]
            if len(idx_list) == 0:
                continue
            end_idx = idx_list[-1]
            start_idx = end_idx - self.window_size
            
            if start_idx < 0:
                continue
                
            window_slice = self.flux_df.iloc[start_idx:end_idx]
            soft_chan = window_slice['soft_flux'].values
            hard_chan = window_slice['hard_flux'].values
            
            if len(soft_chan) == self.window_size and len(hard_chan) == self.window_size:
                features = np.stack([soft_chan, hard_chan], axis=1)
                X_list.append(features)
                y_list.append(f_token)
        
        if len(X_list) == 0:
            raise ValueError("Data pipeline Error: 0 windows generated. Check timestamps.")
            
        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
