import requests
import pandas as pd
import matplotlib.pyplot as plt

# GOES-16 X-ray flux — last 7 days, updated every minute
URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

def fetch_goes_data():
    print("Fetching GOES-16 X-ray data from NOAA...")
    response = requests.get(URL)
    
    # Fix malformed JSON from NOAA
    raw = response.text.strip()
    # Sometimes NOAA returns concatenated JSON arrays — fix it
    raw = raw.replace('][', ',')
    
    import json
    data = json.loads(raw)
    df = pd.DataFrame(data)
    print(f"Total data points: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    return df
def plot_xray(df):
    df['time_tag'] = pd.to_datetime(df['time_tag'])

    df_long  = df[df['energy'] == '0.1-0.8nm'].copy()
    df_short = df[df['energy'] == '0.05-0.4nm'].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor='#0A0A14')

    for ax in axes:
        ax.set_facecolor('#0F0F1E')
        ax.tick_params(colors='#666')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Soft X-ray
    axes[0].plot(df_long['time_tag'], df_long['flux'],
                 color='#FF8C00', linewidth=1, label='Soft X-ray (0.1-0.8nm)')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Flux (W/m²)', color='#888')
    axes[0].set_title('GOES-16 Solar X-ray Flux — Last 7 Days',
                      color='#FFB347', pad=10)
    axes[0].legend(facecolor='#1A1A2E', labelcolor='#CCC')

    # Flare class reference lines
    for level, label, color in [
        (1e-4, 'X', '#FF3B3B'),
        (1e-5, 'M', '#FF8C00'),
        (1e-6, 'C', '#FFD700'),
        (1e-7, 'B', '#4CAF50'),
    ]:
        axes[0].axhline(y=level, color=color, linestyle='--', alpha=0.4, linewidth=0.8)
        axes[0].text(df_long['time_tag'].iloc[0], level,
                     f' {label}', color=color, fontsize=9, va='bottom')

    # Hard X-ray
    axes[1].plot(df_short['time_tag'], df_short['flux'],
                 color='#4A9EFF', linewidth=1, label='Hard X-ray (0.05-0.4nm)')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Flux (W/m²)', color='#888')
    axes[1].set_xlabel('Date (UTC)', color='#888')
    axes[1].legend(facecolor='#1A1A2E', labelcolor='#CCC')

    plt.tight_layout()
    plt.savefig('goes_xray_7day.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A14')
    print("\nPlot saved as goes_xray_7day.png")
    plt.show()

if __name__ == "__main__":
    df = fetch_goes_data()
    plot_xray(df)