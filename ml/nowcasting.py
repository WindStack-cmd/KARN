"""Deterministic, artifact-backed Phase A X-ray event catalogue generation.

The detectors operate independently on the real synchronized telemetry.  Their
relative-intensity categories are detector categories, not GOES C/M/X classes.
Historical M/X labels are joined only after detection as reference context.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "nowcasting"


@dataclass(frozen=True)
class DetectorParameters:
    version: str = "phase-a-1.0"
    smoothing_seconds: int = 15
    baseline_seconds: int = 900
    baseline_quantile: float = 0.20
    ratio_threshold: float = 3.0
    minimum_excess: float = 8.0
    minimum_duration_seconds: int = 30
    minimum_separation_seconds: int = 120
    recovery_ratio: float = 1.25
    recovery_seconds: int = 30
    association_seconds: int = 300


SOLEXS_PARAMETERS = DetectorParameters(ratio_threshold=3.0, minimum_excess=12.0)
HEL1OS_PARAMETERS = DetectorParameters(ratio_threshold=4.0, minimum_excess=15.0)


def load_telemetry() -> pd.DataFrame:
    timeline = pd.read_csv(PROCESSED / "aditya_synced_timeline.csv")
    features = pd.read_csv(PROCESSED / "aditya_features.csv", usecols=["timestamp", "anomaly_flag", "quality_flag"])
    labels = pd.read_csv(PROCESSED / "aditya_labels.csv", usecols=["timestamp", "flare_class_token"])
    for frame in (timeline, features, labels):
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = timeline.merge(features, on="timestamp", how="left").merge(labels, on="timestamp", how="left")
    return frame.sort_values(["partition_id", "timestamp"]).reset_index(drop=True)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive true runs in a boolean vector."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def _join_close_runs(runs: list[tuple[int, int]], separation: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    joined = [list(runs[0])]
    for start, end in runs[1:]:
        if start - joined[-1][1] - 1 <= separation:
            joined[-1][1] = end
        else:
            joined.append([start, end])
    return [(int(start), int(end)) for start, end in joined]


def _reference_label(part: pd.DataFrame) -> str | None:
    # Labels are existing reference context and are never detector input.
    tokens = pd.to_numeric(part["flare_class_token"], errors="coerce").dropna().astype(int)
    if (tokens == 3).any():
        return "X"
    if (tokens == 2).any():
        return "M"
    return None


def detect_instrument(frame: pd.DataFrame, source: str, flux_column: str, params: DetectorParameters) -> pd.DataFrame:
    records: list[dict] = []
    prefix = "SOL" if source == "SoLEXS" else "HEL"
    event_number = 0
    for partition, part in frame.groupby("partition_id", sort=True):
        part = part.sort_values("timestamp").reset_index(drop=True).copy()
        raw = pd.to_numeric(part[flux_column], errors="coerce")
        valid = raw.notna() & np.isfinite(raw) & (part["quality_flag"].fillna(1) == 0)

        # Time-indexed views so rolling windows span real wall-clock seconds
        # rather than row counts — robust to sampling gaps/dropouts. min_periods
        # is still a row-count requirement, so a window is only used once it
        # contains enough real samples, not just enough elapsed time.
        ts_index = pd.DatetimeIndex(part["timestamp"])
        raw_ts = raw.copy(); raw_ts.index = ts_index
        masked_ts = raw_ts.where(valid.to_numpy())

        # Causal median smoothing and low-quantile baseline avoid future data use.
        smooth_ts = masked_ts.rolling(f"{params.smoothing_seconds}s", min_periods=params.smoothing_seconds).median()
        baseline_ts = smooth_ts.rolling(f"{params.baseline_seconds}s", min_periods=params.baseline_seconds).quantile(params.baseline_quantile)

        # Expanding (past-and-present-only) floor, replacing the old whole-
        # partition quantile which used future samples to set a value applied
        # to earlier rows.
        floor_ts = masked_ts.expanding(min_periods=1).quantile(0.10).clip(lower=1.0)
        baseline_ts = baseline_ts.clip(lower=floor_ts)

        smooth = smooth_ts.reset_index(drop=True)
        baseline = baseline_ts.reset_index(drop=True)
        excess = smooth - baseline
        ratio = smooth / baseline
        candidate = (valid & (ratio >= params.ratio_threshold) & (excess >= params.minimum_excess)).fillna(False).to_numpy(bool)
        candidate_runs = _join_close_runs(_runs(candidate), params.minimum_separation_seconds)
        for start, end in candidate_runs:
            if end - start + 1 < params.minimum_duration_seconds:
                continue
            # Require sustained recovery before closing; otherwise extend to the
            # observed candidate end. This is deterministic and causal in use.
            recovery = (smooth.iloc[end + 1:] <= baseline.iloc[end + 1:] * params.recovery_ratio).fillna(False).to_numpy()
            recovery_runs = _runs(recovery)
            for recovery_start, recovery_end in recovery_runs:
                if recovery_end - recovery_start + 1 >= params.recovery_seconds:
                    end = end + 1 + recovery_start
                    break
            segment = part.iloc[start:end + 1]
            seg_smooth, seg_baseline, seg_excess, seg_ratio = smooth.iloc[start:end + 1], baseline.iloc[start:end + 1], excess.iloc[start:end + 1], ratio.iloc[start:end + 1]
            if seg_smooth.notna().sum() == 0:
                continue
            peak_index = int(seg_smooth.idxmax())
            peak_value = float(smooth.iloc[peak_index])
            baseline_value = float(baseline.iloc[peak_index])
            peak_excess, peak_ratio = peak_value - baseline_value, peak_value / baseline_value
            score = min(1.0, (peak_ratio / params.ratio_threshold) * min(1.0, (end - start + 1) / params.minimum_duration_seconds))
            category = "high-relative-intensity" if peak_ratio >= 10 else "medium-relative-intensity" if peak_ratio >= 5 else "low-relative-intensity"
            event_number += 1
            records.append({
                "event_id": f"{prefix}-{part.timestamp.iloc[start]:%Y%m%dT%H%M%S}-{event_number:03d}",
                "source": source, "partition": partition,
                "start_time": part.timestamp.iloc[start], "peak_time": part.timestamp.iloc[peak_index], "end_time": part.timestamp.iloc[end],
                "duration_seconds": int((part.timestamp.iloc[end] - part.timestamp.iloc[start]).total_seconds()) + 1,
                "peak_value": peak_value, "baseline_value": baseline_value, "peak_excess": peak_excess, "peak_ratio": peak_ratio,
                "detection_score": score, "intensity_category": category,
                "historical_reference_label": _reference_label(segment),
                "anomaly_samples": int(segment.anomaly_flag.fillna(0).sum()), "quality_flagged_samples": int((segment.quality_flag.fillna(1) != 0).sum()),
                "parameters_version": params.version,
            })
    columns = ["event_id", "source", "partition", "start_time", "peak_time", "end_time", "duration_seconds", "peak_value", "baseline_value", "peak_excess", "peak_ratio", "detection_score", "intensity_category", "historical_reference_label", "anomaly_samples", "quality_flagged_samples", "parameters_version"]
    return pd.DataFrame(records, columns=columns)


def build_master(solexs: pd.DataFrame, helios: pd.DataFrame, params: DetectorParameters) -> pd.DataFrame:
    used_helios: set[int] = set()
    records: list[dict] = []
    master_number = 0
    for _, soft in solexs.sort_values("peak_time").iterrows():
        candidates = helios.loc[(helios.partition == soft.partition) & ~helios.index.isin(used_helios)].copy()
        if not candidates.empty:
            delta = (pd.to_datetime(candidates.peak_time) - pd.to_datetime(soft.peak_time)).abs().dt.total_seconds()
            candidates = candidates.loc[delta <= params.association_seconds]
        hard = None if candidates.empty else candidates.loc[(pd.to_datetime(candidates.peak_time) - pd.to_datetime(soft.peak_time)).abs().idxmin()]
        if hard is not None:
            used_helios.add(hard.name)
        records.append(_master_record(soft, hard, master_number)); master_number += 1
    for index, hard in helios.sort_values("peak_time").iterrows():
        if index not in used_helios:
            records.append(_master_record(None, hard, master_number)); master_number += 1
    columns = ["master_event_id", "start_time", "peak_time", "end_time", "duration_seconds", "detection_source", "solexs_event_id", "helios_event_id", "solexs_peak", "helios_peak", "solexs_baseline", "helios_baseline", "solexs_peak_excess", "helios_peak_excess", "detection_score", "intensity_category", "historical_reference_label", "matching_time_difference_seconds", "anomaly_samples", "quality_flagged_samples", "parameters_version"]
    return pd.DataFrame(records, columns=columns).sort_values("peak_time").reset_index(drop=True)


def _master_record(soft: pd.Series | None, hard: pd.Series | None, number: int) -> dict:
    sources = [item for item in (soft, hard) if item is not None]
    start, end = min(pd.Timestamp(item.start_time) for item in sources), max(pd.Timestamp(item.end_time) for item in sources)
    peak_source = max(sources, key=lambda item: float(item.detection_score))
    label = "X" if any(item.historical_reference_label == "X" for item in sources) else "M" if any(item.historical_reference_label == "M" for item in sources) else None
    both = soft is not None and hard is not None
    return {
        "master_event_id": f"KARN-NOW-{pd.Timestamp(peak_source.peak_time):%Y%m%dT%H%M%S}-{number:03d}",
        "start_time": start, "peak_time": peak_source.peak_time, "end_time": end, "duration_seconds": int((end - start).total_seconds()) + 1,
        "detection_source": "BOTH" if both else peak_source.source,
        "solexs_event_id": None if soft is None else soft.event_id, "helios_event_id": None if hard is None else hard.event_id,
        "solexs_peak": None if soft is None else soft.peak_value, "helios_peak": None if hard is None else hard.peak_value,
        "solexs_baseline": None if soft is None else soft.baseline_value, "helios_baseline": None if hard is None else hard.baseline_value,
        "solexs_peak_excess": None if soft is None else soft.peak_excess, "helios_peak_excess": None if hard is None else hard.peak_excess,
        "detection_score": float(max(float(item.detection_score) for item in sources)),
        "intensity_category": peak_source.intensity_category, "historical_reference_label": label,
        "matching_time_difference_seconds": None if not both else abs((pd.Timestamp(soft.peak_time) - pd.Timestamp(hard.peak_time)).total_seconds()),
        "anomaly_samples": int(sum(int(item.anomaly_samples) for item in sources)), "quality_flagged_samples": int(sum(int(item.quality_flagged_samples) for item in sources)), "parameters_version": peak_source.parameters_version,
    }


def validate_catalogue(catalogue: pd.DataFrame, id_column: str) -> None:
    if catalogue[id_column].duplicated().any(): raise ValueError(f"Duplicate {id_column} values")
    for column in ("start_time", "peak_time", "end_time"):
        if pd.to_datetime(catalogue[column], errors="coerce").isna().any(): raise ValueError(f"Invalid {column}")
    if (catalogue.duration_seconds < 1).any(): raise ValueError("Non-positive event duration")
    if ((pd.to_datetime(catalogue.peak_time) < pd.to_datetime(catalogue.start_time)) | (pd.to_datetime(catalogue.peak_time) > pd.to_datetime(catalogue.end_time))).any(): raise ValueError("Peak lies outside event interval")


def generate(output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    telemetry = load_telemetry()
    solexs = detect_instrument(telemetry, "SoLEXS", "solexs_flux", SOLEXS_PARAMETERS)
    helios = detect_instrument(telemetry, "HEL1OS", "hel1os_flux", HEL1OS_PARAMETERS)
    master = build_master(solexs, helios, SOLEXS_PARAMETERS)
    for catalogue, key in ((solexs, "event_id"), (helios, "event_id"), (master, "master_event_id")):
        validate_catalogue(catalogue, key)
    files = {"solexs": output / "solexs_nowcast_catalogue.csv", "helios": output / "helios_nowcast_catalogue.csv", "master": output / "master_flare_catalogue.csv"}
    solexs.to_csv(files["solexs"], index=False); helios.to_csv(files["helios"], index=False); master.to_csv(files["master"], index=False)
    summary = {
        "schema_version": "phase-a-1.0", "artifact_backed": True, "live_feed": False,
        "coverage": {"start": telemetry.timestamp.min().isoformat(), "end": telemetry.timestamp.max().isoformat(), "partitions": sorted(telemetry.partition_id.unique().tolist())},
        "counts": {"solexs": len(solexs), "helios": len(helios), "master": len(master), "solexs_only": int((master.detection_source == "SoLEXS").sum()), "helios_only": int((master.detection_source == "HEL1OS").sum()), "both": int((master.detection_source == "BOTH").sum())},
        "quality": {"input_anomaly_samples": int(telemetry.anomaly_flag.fillna(0).sum()), "input_quality_flagged_samples": int((telemetry.quality_flag.fillna(1) != 0).sum())},
        "parameters": {"solexs": asdict(SOLEXS_PARAMETERS), "helios": asdict(HEL1OS_PARAMETERS), "association_seconds": SOLEXS_PARAMETERS.association_seconds},
        "classification_note": "Relative intensity categories are detector categories, not GOES classes. Historical M/X labels are reference context only.",
        "evaluation_note": "No catalogue-level precision/recall is reported: available reference labels are not an independently verified per-instrument detection ground truth.",
    }
    (output / "nowcasting_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, catalogue in (("solexs", solexs), ("helios", helios), ("master", master)):
        (output / f"{name}_nowcast_catalogue.json").write_text(catalogue.to_json(orient="records", date_format="iso", indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic KARN Phase A nowcasting catalogues.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))
