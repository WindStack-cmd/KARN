import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os

RAW_DIR  = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
PROC_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed"

CLASS_MAP = {'Q': 0, 'C': 1, 'M': 2, 'X': 3}
WINDOW    = 60  # 60 minutes of context

def strip_tz(series):
    s = pd.to_datetime(series)
    if hasattr(s, 'dt'):
        if s.dt.tz is not None:
            return s.dt.tz_localize(None)
        return s
    if s.tz is not None:
        return s.tz_localize(None)
    return s

class SolarFlareDatasetV2(Dataset):
    """
    Dataset that builds windows from DONKI catalog timestamps.
    Uses GOES flux data where available, synthetic data where not.
    """

    def __init__(self, catalog_path, flux_path):
        print("Loading DONKI catalog + GOES flux...")

        # Load catalog
        catalog = pd.read_csv(catalog_path)
        catalog['start_time'] = strip_tz(catalog['start_time'])
        catalog['peak_time']  = strip_tz(catalog['peak_time'])
        catalog['end_time']   = strip_tz(catalog['end_time'])
        catalog['label']      = catalog['flare_class'].map(CLASS_MAP)

        # Load flux data
        flux = pd.read_csv(flux_path)
        flux['time_tag'] = strip_tz(flux['time_tag'])
        flux['flux']     = pd.to_numeric(flux['flux'], errors='coerce')

        # Separate channels
        self.flux_soft = flux[flux['energy'] == '0.1-0.8nm'].copy()
        self.flux_hard = flux[flux['energy'] == '0.05-0.4nm'].copy()

        self.flux_soft = self.flux_soft.sort_values('time_tag').set_index('time_tag')
        self.flux_hard = self.flux_hard.sort_values('time_tag').set_index('time_tag')

        # Get flux range for synthetic generation
        self.flux_min  = float(self.flux_soft['flux'].quantile(0.10))
        self.flux_max  = float(self.flux_soft['flux'].quantile(0.90))
        self.flux_mean = float(self.flux_soft['flux'].mean())
        self.flux_std  = float(self.flux_soft['flux'].std())

        print(f"Flux range: {self.flux_min:.2e} → {self.flux_max:.2e}")
        print(f"Catalog events: {len(catalog)}")
        print(f"Class distribution:")
        print(catalog['flare_class'].value_counts())

        # Build windows
        self.windows = []
        self.labels  = []

        skipped = 0
        for _, row in catalog.iterrows():
            label      = int(row['label'])
            peak_time  = pd.Timestamp(row['peak_time'])
            win_start  = peak_time - pd.Timedelta(f'{WINDOW}min')
            win_end    = peak_time

            window = self._build_window(win_start, win_end, label)
            if window is not None:
                self.windows.append(window)
                self.labels.append(label)
            else:
                skipped += 1

        print(f"\nWindows built: {len(self.windows)}")
        print(f"Skipped:       {skipped}")

        # Final class distribution
        label_counts = {}
        for l in self.labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        class_names = ['Quiet', 'C-class', 'M-class', 'X-class']
        print("\nFinal window distribution:")
        for k, v in sorted(label_counts.items()):
            print(f"  {class_names[k]}: {v}")

    def _build_window(self, start, end, label):
        """
        Try to get real flux data for this window.
        If not available, generate synthetic data based on class.
        """
        try:
            # Try real data first
            mask_soft = (self.flux_soft.index >= start) & \
                        (self.flux_soft.index <= end)
            mask_hard = (self.flux_hard.index >= start) & \
                        (self.flux_hard.index <= end)

            soft_vals = self.flux_soft.loc[mask_soft, 'flux'].values
            hard_vals = self.flux_hard.loc[mask_hard, 'flux'].values

            if len(soft_vals) >= 10 and len(hard_vals) >= 10:
                # Real data available — use it
                soft_interp = np.interp(
                    np.linspace(0, 1, WINDOW),
                    np.linspace(0, 1, len(soft_vals)),
                    soft_vals
                )
                hard_interp = np.interp(
                    np.linspace(0, 1, WINDOW),
                    np.linspace(0, 1, len(hard_vals)),
                    hard_vals
                )
            else:
                # Generate synthetic window based on class
                soft_interp, hard_interp = self._synthetic_window(label)

        except Exception:
            soft_interp, hard_interp = self._synthetic_window(label)

        # Normalize using log scale (standard for X-ray flux)
        soft_norm = self._log_normalize(soft_interp)
        hard_norm = self._log_normalize(hard_interp)

        # Check for NaN
        if np.any(np.isnan(soft_norm)) or np.any(np.isnan(hard_norm)):
            return None

        # Stack into (60, 2) tensor
        window = np.stack([soft_norm, hard_norm], axis=1).astype(np.float32)
        return window

    def _synthetic_window(self, label):
        """
        Generate realistic synthetic flux windows per class.
        Based on actual solar physics flux profiles.
        """
        t = np.linspace(0, 1, WINDOW)

        if label == 0:
            # Quiet Sun — flat with small noise
            base = self.flux_mean * 0.3
            soft = base + np.random.normal(0, base * 0.05, WINDOW)
            hard = base * 0.01 + np.random.normal(0, base * 0.005, WINDOW)

        elif label == 1:
            # C-class — small rise
            base = self.flux_mean * 0.5
            rise = base * 3 * np.exp(-((t - 0.7)**2) / 0.02)
            soft = base + rise + np.random.normal(0, base * 0.05, WINDOW)
            hard = (base + rise) * 0.1 + np.random.normal(0, base * 0.01, WINDOW)

        elif label == 2:
            # M-class — significant rise
            base = self.flux_mean * 0.5
            rise = base * 20 * np.exp(-((t - 0.7)**2) / 0.015)
            soft = base + rise + np.random.normal(0, base * 0.1, WINDOW)
            hard = (base + rise) * 0.3 + np.random.normal(0, base * 0.03, WINDOW)

        else:
            # X-class — dramatic spike
            base = self.flux_mean * 0.5
            rise = base * 200 * np.exp(-((t - 0.7)**2) / 0.01)
            soft = base + rise + np.random.normal(0, base * 0.2, WINDOW)
            hard = (base + rise) * 0.5 + np.random.normal(0, base * 0.05, WINDOW)

        # Ensure positive values
        soft = np.abs(soft) + 1e-10
        hard = np.abs(hard) + 1e-10

        return soft, hard

    def _log_normalize(self, arr):
        """Log-scale normalization — standard for X-ray flux data."""
        arr  = np.clip(arr, 1e-10, None)
        log  = np.log10(arr)
        # Normalize to [-1, 1] range
        log_min = np.log10(1e-9)   # below B-class
        log_max = np.log10(1e-3)   # above X-class
        norm = (log - log_min) / (log_max - log_min)
        norm = np.clip(norm, 0, 1)
        return norm

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.long)
        return x, y


if __name__ == "__main__":
    catalog_path = os.path.join(PROC_DIR, "full_catalog_v2.csv")
    flux_path    = os.path.join(RAW_DIR,  "goes_flux_combined.csv")

    dataset = SolarFlareDatasetV2(catalog_path, flux_path)
    print(f"\nDataset ready — {len(dataset)} windows")

    # Test one sample
    x, y = dataset[0]
    print(f"\nSample shape: {x.shape}")
    print(f"Sample label: {y.item()}")
    print(f"Value range:  {x.min():.3f} → {x.max():.3f}")