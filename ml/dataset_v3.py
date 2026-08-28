"""Build real, causal KARN windows using a complete final-period holdout."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data" / "processed"; OUT = DATA / "training_windows"
SEQ_LEN, FEATURE_COUNT = 30, 21
NAMES = {0: "Quiet", 1: "C", 2: "M", 3: "X"}

def dist(y): return {str(i): int(np.bincount(y.astype(int), minlength=4)[i]) for i in range(4)}

def compile_split_windows():
    feat, lab = pd.read_csv(DATA / "aditya_features.csv"), pd.read_csv(DATA / "aditya_labels.csv")
    timeline = pd.read_csv(DATA / "aditya_synced_timeline.csv", usecols=["timestamp", "partition_id"])
    if not (len(feat) == len(lab) == len(timeline)) or not feat.timestamp.equals(lab.timestamp) or not feat.timestamp.equals(timeline.timestamp): raise ValueError("Feature, label, and timeline rows/timestamps must align")
    cols = [c for c in feat if c not in {"timestamp", "time_seconds", "anomaly_flag", "quality_flag"}]
    if len(cols) != FEATURE_COUNT: raise ValueError(f"Expected {FEATURE_COUNT} scientifically defined features, found {len(cols)}")
    values = feat[cols].to_numpy(np.float32); y_all = lab.flare_class_token.to_numpy(np.int64); r_all = lab.time_to_peak_min.to_numpy(np.float32)
    if not np.isfinite(values).all() or not np.isfinite(r_all).all() or not np.isin(y_all, [0,1,2,3]).all(): raise ValueError("Source feature/label finite or class audit failed")
    groups = [g.index.to_numpy() for _, g in timeline.groupby("partition_id", sort=False)]
    if len(groups) < 2: raise ValueError("Complete-period chronological holdout requires at least two observation partitions")
    def contiguous(group):
        t = pd.to_datetime(timeline.timestamp.iloc[group]); b = t.diff().dt.total_seconds().ne(1); b.iloc[0] = True
        starts = np.flatnonzero(b.to_numpy()); ends = np.r_[starts[1:], len(group)]
        return [group[a:b] for a,b in zip(starts,ends) if b-a >= SEQ_LEN]
    # Policy: final complete observation period is held out for testing. Earlier
    # periods are split chronologically 85/15, with a 29-row embargo at the boundary.
    endpoints = np.concatenate([segment[SEQ_LEN-1:] for group in groups[:-1] for segment in contiguous(group)])
    test_endpoints = np.concatenate([segment[SEQ_LEN-1:] for segment in contiguous(groups[-1])])
    cut = int(.85 * len(endpoints)); guard = SEQ_LEN-1
    rows = {"train": endpoints[:cut], "val": endpoints[cut+guard:], "test": test_endpoints}
    raw_train_rows = np.unique(np.concatenate([np.arange(i-SEQ_LEN+1, i+1) for i in rows["train"]]))
    scaler = RobustScaler().fit(values[raw_train_rows])
    scaled = scaler.transform(values).astype(np.float32)
    arrays = {}; used = {}; report = {"feature_columns": cols, "feature_count": FEATURE_COUNT, "sequence_length": SEQ_LEN, "split_policy": "final complete observation period held out for test; prior periods chronological 85/15 train/validation with 29-row purge", "splits": {}}
    for split, end in rows.items():
        x = np.stack([scaled[i-SEQ_LEN+1:i+1] for i in end]); y, r = y_all[end], r_all[end,None]
        if not np.isfinite(x).all() or not np.isfinite(r).all(): raise ValueError(f"{split} NaN/Inf audit failed")
        arrays[split] = (x,y,r); used[split] = set(np.concatenate([np.arange(i-SEQ_LEN+1,i+1) for i in end]))
        report["splits"][split] = {"X_shape": list(x.shape), "y_cls_shape":list(y.shape), "y_reg_shape":list(r.shape), "class_distribution":dist(y), "nan":int(np.isnan(x).sum()+np.isnan(r).sum()), "inf":int(np.isinf(x).sum()+np.isinf(r).sum()), "timestamp_start":feat.timestamp.iloc[end[0]], "timestamp_end":feat.timestamp.iloc[end[-1]]}
    overlap={f"{a}_{b}":len(used[a]&used[b]) for a,b in (("train","val"),("train","test"),("val","test"))}
    if any(overlap.values()): raise ValueError(f"Leakage Audit failed: {overlap}")
    report["raw_window_overlap"] = overlap; report["leakage_audit"]="PASS"; report["test_partition"] = str(timeline.partition_id.iloc[groups[-1][0]])
    OUT.mkdir(parents=True, exist_ok=True)
    for split,(x,y,r) in arrays.items(): torch.save(torch.from_numpy(x),OUT/f"X_{split}.pt"); torch.save(torch.from_numpy(y),OUT/f"y_{split}_cls.pt"); torch.save(torch.from_numpy(r),OUT/f"y_{split}_reg.pt")
    np.savez_compressed(OUT/"scaler.npz", center=scaler.center_, scale=scaler.scale_); (OUT/"validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    for split,d in report["splits"].items(): print(f"{split}: X={tuple(d['X_shape'])}, y_cls={tuple(d['y_cls_shape'])}, y_reg={tuple(d['y_reg_shape'])}; classes={d['class_distribution']}; NaN={d['nan']}; Inf={d['inf']}; {d['timestamp_start']} -> {d['timestamp_end']}")
    print("Leakage Audit: PASS"); return report

if __name__ == "__main__": compile_split_windows()
