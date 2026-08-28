"""Causal feature derivation; scaling is fitted later on training rows only."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"processed"

def compute_temporal_features():
    master=pd.read_csv(DATA/"aditya_synced_timeline.csv"); master.timestamp=pd.to_datetime(master.timestamp,errors="raise"); pieces=[]
    for _, part in master.groupby("partition_id",sort=False):
        part=part.sort_values("timestamp").reset_index(drop=True); b=part.timestamp.diff().dt.total_seconds().ne(1); b.iloc[0]=True; starts=np.flatnonzero(b); ends=np.r_[starts[1:],len(part)]
        for start,end in zip(starts,ends):
            df=part.iloc[start:end].reset_index(drop=True).copy(); delta=df.timestamp.diff().dt.total_seconds().fillna(1.0)
            df["solexs_rate"]=df.solexs_flux.diff().fillna(0.)/delta; df["hel1os_rate"]=df.hel1os_flux.diff().fillna(0.)/delta; df["soft_hard_ratio"]=df.solexs_flux/(df.hel1os_flux+1e-5)
            for label,width in {"5m":300,"30m":1800,"1h":3600,"6h":21600}.items():
                for sensor in ("solexs","hel1os"):
                    raw=df[f"{sensor}_flux"]; mean=raw.rolling(width,closed="left",min_periods=1).mean(); std=raw.rolling(width,closed="left",min_periods=1).std(ddof=0)
                    # At the first observed second there is no prior history;
                    # use the contemporaneous measurement / zero dispersion, not a future value.
                    df[f"{sensor}_mean_{label}"]=mean.fillna(raw); df[f"{sensor}_std_{label}"]=std.fillna(0.0)
            med=df.hel1os_flux.median(); mad=np.median(np.abs(df.hel1os_flux-med)) or 1.; df["anomaly_flag"]=((df.hel1os_flux-med)/mad>15.).astype(int); df["quality_flag"]=0; pieces.append(df)
    result=pd.concat(pieces,ignore_index=True); cols=[c for c in result if c not in {"timestamp","time_seconds","partition_id","anomaly_flag","quality_flag"}]
    if len(cols)!=21 or not np.isfinite(result[cols].to_numpy(float)).all(): raise ValueError("Causal scientific feature contract failed")
    result.drop(columns=["partition_id"]).to_csv(DATA/"aditya_features.csv",index=False); print(f"Generated causal features: {len(result)} rows, {len(cols)} features")

if __name__=="__main__": compute_temporal_features()
