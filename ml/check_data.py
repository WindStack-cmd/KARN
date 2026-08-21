import os
import pandas as pd

f_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\aditya_synced_timeline.csv"
cat_path = r"C:\Users\DELL\OneDrive\Desktop\KARN\data\processed\full_catalog.csv"

print("--- Telemetry Workspace Audit ---")
if os.path.exists(f_path):
    df = pd.read_csv(f_path)
    print(f"• Total Synchronized Data Rows: {len(df)}")
    print(f"• Core Start Timestamp: {df['timestamp'].min()}")
    print(f"• Core End Timestamp:   {df['timestamp'].max()}")
else:
    print(f"❌ Cannot find synced timeline file at: {f_path}")

print("\n--- Catalog Status ---")
print(f"• Catalog File Present: {os.path.exists(cat_path)}")
