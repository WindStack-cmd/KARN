import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

RAW_DIR  = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\raw"
PROC_DIR = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed"

def strip_tz(series):
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s

def plot_flare_vs_quiet(catalog, flux_df, n_samples=4):

    # Clean flux dataframe
    df_soft = flux_df[flux_df['energy'] == '0.1-0.8nm'].copy()
    df_soft['time_tag'] = strip_tz(df_soft['time_tag'])
    df_soft = df_soft.sort_values('time_tag').reset_index(drop=True)
    df_soft['flux'] = pd.to_numeric(df_soft['flux'], errors='coerce')
    df_soft = df_soft.dropna(subset=['flux'])
    df_soft = df_soft.set_index('time_tag')

    print(f"Flux index range: {df_soft.index.min()} → {df_soft.index.max()}")

    # Clean catalog timestamps
    catalog['start_time'] = strip_tz(catalog['start_time'])
    catalog['peak_time']  = strip_tz(catalog['peak_time'])
    catalog['end_time']   = strip_tz(catalog['end_time'])

    # Get flare and quiet samples
    flares = catalog[catalog['flare_class'].isin(['M', 'X'])]
    if len(flares) == 0:
        flares = catalog[catalog['flare_class'] == 'C']
    flares = flares.head(n_samples)

    quiets = catalog[catalog['flare_class'] == 'Q'].head(n_samples)

    print(f"Flare samples: {len(flares)}")
    print(f"Quiet samples: {len(quiets)}")

    fig = plt.figure(figsize=(18, 10), facecolor='#0A0A14')
    fig.suptitle('KARN — Flare vs Quiet Sun Windows\n(What the LSTM needs to learn)',
                 color='#FFB347', fontsize=14, y=0.98)

    gs = gridspec.GridSpec(n_samples, 2, figure=fig, hspace=0.6, wspace=0.3)

    def plot_window(ax, start, end, color, title):
        ax.set_facecolor('#0F0F1E')
        ax.tick_params(colors='#555', labelsize=7)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#333')

        try:
            # Get data in window
            mask = (df_soft.index >= start) & (df_soft.index <= end)
            window = df_soft.loc[mask, 'flux']

            print(f"  Window {start} → {end}: {len(window)} points")

            if len(window) < 2:
                ax.text(0.5, 0.5, 'No data in window',
                        ha='center', va='center',
                        color='#555', fontsize=8,
                        transform=ax.transAxes)
            else:
                ax.plot(range(len(window)), window.values,
                        color=color, linewidth=1.5)
                ax.set_yscale('log')
                # Add flare class lines
                ax.axhline(y=1e-5, color='#FF8C00',
                           linestyle='--', alpha=0.4, linewidth=0.8)
                ax.axhline(y=1e-6, color='#FFD700',
                           linestyle='--', alpha=0.4, linewidth=0.8)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}',
                    ha='center', va='center',
                    color='#555', fontsize=7,
                    transform=ax.transAxes)

        ax.set_title(title, color=color, fontsize=8, pad=4)

    # Plot flare windows
    for i, (_, event) in enumerate(flares.iterrows()):
        if i >= n_samples:
            break
        ax = fig.add_subplot(gs[i, 0])
        peak = pd.Timestamp(event['peak_time'])
        start = peak - pd.Timedelta('60min')
        end   = peak + pd.Timedelta('60min')
        title = f"{event['flare_class']}-class | {str(event['peak_time'])[:16]}"
        plot_window(ax, start, end, '#FF8C00', title)
        if i == 0:
            ax.set_title(f'FLARE WINDOWS\n{title}',
                        color='#FF8C00', fontsize=8, pad=4)

    # Plot quiet windows
    for i, (_, event) in enumerate(quiets.iterrows()):
        if i >= n_samples:
            break
        ax = fig.add_subplot(gs[i, 1])
        start = pd.Timestamp(event['start_time'])
        end   = pd.Timestamp(event['end_time'])
        title = f"QUIET | {str(event['start_time'])[:16]}"
        plot_window(ax, start, end, '#4A9EFF', title)
        if i == 0:
            ax.set_title(f'QUIET WINDOWS\n{title}',
                        color='#4A9EFF', fontsize=8, pad=4)

    plt.savefig(os.path.join(PROC_DIR, 'eda_flare_vs_quiet.png'),
                dpi=150, bbox_inches='tight', facecolor='#0A0A14')
    print("\nEDA plot saved: eda_flare_vs_quiet.png")
    plt.show()

def print_catalog_stats(catalog):
    print("\n" + "=" * 50)
    print("KARN FLARE CATALOG STATISTICS")
    print("=" * 50)
    print(f"Total events:  {len(catalog)}")
    print(f"\nBy class:")
    print(catalog['flare_class'].value_counts())
    flares_only = catalog[catalog['flare_class'] != 'Q']
    if len(flares_only) > 0:
        print(f"\nMean flare duration: {flares_only['duration_min'].mean():.1f} min")
        print(f"Max peak flux:       {catalog['peak_flux'].max():.2e} W/m²")
    print("=" * 50)

if __name__ == "__main__":
    catalog_path = os.path.join(PROC_DIR, "full_catalog.csv")
    flux_path    = os.path.join(RAW_DIR,  "goes_flux_combined.csv")

    if not os.path.exists(catalog_path):
        print("ERROR: Run build_catalog.py first")
        exit()

    print("Loading data...")
    catalog = pd.read_csv(catalog_path)
    flux_df = pd.read_csv(flux_path)

    print_catalog_stats(catalog)
    plot_flare_vs_quiet(catalog, flux_df, n_samples=4)