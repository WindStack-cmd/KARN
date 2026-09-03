"""Artifact-backed API for the frozen KARN temporal flare experiment."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    from .models_v3 import KarnMultiTaskEngine
except ImportError:  # Supports `uvicorn main:app` from the ml directory.
    from models_v3 import KarnMultiTaskEngine

ROOT = Path(__file__).resolve().parents[1]
PROCESSED, WINDOWS = ROOT / "data" / "processed", ROOT / "data" / "processed" / "training_windows"
EVAL, DIAGNOSTICS = ROOT / "ml" / "eval_results", ROOT / "ml" / "eval_results" / "diagnostics"
CHECKPOINT = ROOT / "ml" / "checkpoints" / "karn_binary_flare_temporal_experiment.pt"
FEATURES_PATH, TIMELINE_PATH, LABELS_PATH = PROCESSED / "aditya_features.csv", PROCESSED / "aditya_synced_timeline.csv", PROCESSED / "aditya_labels.csv"
REPORT_PATH, METRICS_PATH = WINDOWS / "binary_validation_report.json", EVAL / "temporal_binary_eval_metrics.json"
PREDICTIONS_PATH, SCALER_PATH = EVAL / "temporal_binary_eval_predictions.npz", WINDOWS / "scaler.npz"
NOWCASTING = PROCESSED / "nowcasting"

app = FastAPI(title="KARN Research API", description="Artifact-backed interface for the frozen KARN binary flare experiment.", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET"], allow_headers=["*"])


def unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Unavailable: {detail}")


def serialise(value: Any) -> Any:
    if value is None or pd.isna(value): return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp): return value.isoformat()
    return value


@lru_cache(maxsize=1)
def report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def metrics() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def nowcasting_summary() -> dict[str, Any]:
    return json.loads((NOWCASTING / "nowcasting_summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=3)
def nowcasting_catalogue(name: str) -> pd.DataFrame:
    filenames = {"solexs": "solexs_nowcast_catalogue.csv", "helios": "helios_nowcast_catalogue.csv", "master": "master_flare_catalogue.csv"}
    if name not in filenames: raise ValueError("unknown nowcasting catalogue")
    path = NOWCASTING / filenames[name]
    if not path.exists(): raise FileNotFoundError(path.name)
    frame = pd.read_csv(path)
    for column in ("start_time", "peak_time", "end_time"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def catalogue_response(name: str, limit: int, start_time: str | None, end_time: str | None) -> dict[str, Any]:
    frame = nowcasting_catalogue(name).copy()
    if start_time: frame = frame.loc[frame.peak_time >= pd.to_datetime(start_time)]
    if end_time: frame = frame.loc[frame.peak_time <= pd.to_datetime(end_time)]
    identifier = "master_event_id" if name == "master" else "event_id"
    frame = frame.sort_values("peak_time", ascending=False).head(limit)
    data = [{key: serialise(value) for key, value in row.items()} for row in frame.to_dict("records")]
    return {"catalogue": name, "identifier": identifier, "total_matching": len(frame), "data": data, "artifact_backed": True, "classification_note": "Relative intensity is algorithmic, not an absolute GOES C/M/X class; historical reference labels are context only."}


@lru_cache(maxsize=1)
def telemetry_frame() -> pd.DataFrame:
    if not all(path.exists() for path in (FEATURES_PATH, TIMELINE_PATH, LABELS_PATH)):
        raise FileNotFoundError("processed telemetry artifacts")
    features, timeline, labels = (pd.read_csv(path) for path in (FEATURES_PATH, TIMELINE_PATH, LABELS_PATH))
    for frame in (features, timeline, labels):
        if "timestamp" in frame.columns: frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = features.merge(timeline[["timestamp", "partition_id"]], on="timestamp", how="left")
    frame = frame.rename(columns={"partition_id": "partition"})
    frame = frame.merge(labels, on="timestamp", how="left", suffixes=("", "_label"))
    # flare_class_token is already the numeric class rank (0=quiet, 1=C, 2=M, 3=X)
    # from create_labels.py — convert directly instead of round-tripping through
    # string keys, which previously mismatched ("3" never matched "x") and made
    # every row silently fall back to quiet.
    frame["original_class"] = pd.to_numeric(frame.get("flare_class_token", 0), errors="coerce").fillna(0).astype(int)
    frame["label"] = frame["original_class"].isin([2, 3]).astype(int)
    return frame.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=1)
def frozen_model() -> KarnMultiTaskEngine:
    if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = KarnMultiTaskEngine(input_dim=21, hidden_dim=64, num_classes=2)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def source_status() -> dict[str, Any]:
    return {"experiment": "KARN Binary Flare Detector — Temporal Generalization Experiment", "model_status": "Frozen exploratory model; not production ready", "scientific_status": "EXPLORATORY — GENERALIZATION LIMITED", "data_source": "Real processed Aditya+HEL1OS compatible telemetry", "live_feed": False, "input_shape": [30, 21]}


@app.get("/")
def root() -> dict[str, Any]: return {"name": "KARN Research API", "api_prefix": "/api", **source_status()}


@app.get("/api/health")
def health() -> dict[str, Any]:
    required = [FEATURES_PATH, TIMELINE_PATH, LABELS_PATH, REPORT_PATH, METRICS_PATH, CHECKPOINT, SCALER_PATH]
    nowcast_files = [NOWCASTING / name for name in ("solexs_nowcast_catalogue.csv", "helios_nowcast_catalogue.csv", "master_flare_catalogue.csv", "nowcasting_summary.json")]
    return {"status": "ok" if all(path.exists() for path in required) else "degraded", "artifacts_available": all(path.exists() for path in required), "nowcasting_available": all(path.exists() for path in nowcast_files), "solexs_catalogue_available": nowcast_files[0].exists(), "helios_catalogue_available": nowcast_files[1].exists(), "master_catalogue_available": nowcast_files[2].exists(), "catalogue_event_count": nowcasting_summary().get("counts", {}).get("master", "Unavailable") if nowcast_files[3].exists() else "Unavailable", **source_status()}


@app.get("/api/status")
def status() -> dict[str, Any]:
    try:
        frame = telemetry_frame()
        return {**source_status(), "observation_rows": len(frame), "observation_start": frame.timestamp.min(), "observation_end": frame.timestamp.max(), "partitions": sorted(frame.partition.dropna().unique().tolist()), "nowcasting": nowcasting_summary() if (NOWCASTING / "nowcasting_summary.json").exists() else "Unavailable"}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/telemetry")
def telemetry(limit: int = Query(900, ge=30, le=10000), partition: str | None = None) -> dict[str, Any]:
    try:
        frame = telemetry_frame()
        if partition: frame = frame.loc[frame["partition"] == partition]
        if frame.empty: raise HTTPException(status_code=404, detail="No real processed observations match that partition")
        stride = max(1, int(np.ceil(len(frame) / limit)))
        display = frame.iloc[::stride].copy()
        fields = ["timestamp", "partition", "solexs_flux", "hel1os_flux", "solexs_rate", "hel1os_rate", "soft_hard_ratio", "anomaly_flag", "quality_flag", "label", "original_class"]
        fields = [field for field in fields if field in display]
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        records = [{key: serialise(value) for key, value in row.items()} for row in display[fields].to_dict("records")]
        return {"source": "real processed telemetry; display-downsampled only", "total_rows": len(frame), "returned_rows": len(display), "stride": stride, "data": records}
    except HTTPException: raise
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/telemetry/latest")
def latest_telemetry() -> dict[str, Any]:
    try:
        row = telemetry_frame().iloc[-1]
        fields = ["timestamp", "partition", "solexs_flux", "hel1os_flux", "solexs_rate", "hel1os_rate", "soft_hard_ratio", "anomaly_flag", "quality_flag", "label", "original_class"]
        return {"source": "latest available historical processed observation (not a live feed)", "observation": {key: serialise(row[key]) for key in fields if key in row.index}}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/model/status")
def model_status() -> dict[str, Any]:
    try:
        model = frozen_model()
        return {**source_status(), "checkpoint": CHECKPOINT.name, "checkpoint_available": True, "classifier_output_classes": model.classifier_head.out_features, "inference_available": SCALER_PATH.exists()}
    except Exception as exc: raise unavailable(f"frozen checkpoint could not be loaded: {exc}") from exc


@app.get("/api/model/prediction")
def model_prediction() -> dict[str, Any]:
    """Run the frozen model on the last real contiguous processed 30-step window."""
    try:
        data, feature_columns = telemetry_frame(), report()["feature_columns"]
        last_partition = data["partition"].dropna().iloc[-1]
        window = data.loc[data["partition"] == last_partition].tail(30).copy()
        # dt.total_seconds() is resolution-agnostic (works whether timestamps are
        # datetime64[ns], [us], or [s]), unlike the previous manual int64/1e9 divide
        # which silently assumed nanosecond precision.
        gaps = window.timestamp.diff().dt.total_seconds().dropna()
        if len(window) != 30 or not np.all(gaps.eq(1.0)): raise ValueError("latest processed data does not contain a contiguous 30-second window")
        raw = window[feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(raw).all(): raise ValueError("latest real feature window contains NaN or Inf")
        scaler = np.load(SCALER_PATH); scale = np.where(scaler["scale"] == 0, 1.0, scaler["scale"])
        transformed = ((raw - scaler["center"]) / scale).astype(np.float32, copy=False)
        with torch.no_grad(): logits, _ = frozen_model()(torch.from_numpy(transformed).unsqueeze(0)); probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        prediction = int(np.argmax(probabilities))
        return {**source_status(), "source": "inference on latest real historical processed 30-step window", "timestamp": window.timestamp.iloc[-1], "predicted_class": "Flare" if prediction else "Quiet", "probabilities": {"quiet": float(probabilities[0]), "flare": float(probabilities[1])}, "window_partition": str(last_partition), "window_shape": [30, 21], "historical_source_label": serialise(window.iloc[-1].get("original_class", "Unavailable")), "note": "Binary output is Quiet vs Flare; it does not distinguish M from X."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    try: return {**source_status(), "metrics": metrics(), "limitation": "Threshold calibration is blocked because the chronological validation split has no flare examples."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/confusion-matrix")
def confusion_matrix() -> dict[str, Any]:
    try:
        data = metrics()
        return {"labels": ["Quiet", "Flare"], "matrix": data["confusion_matrix"], "true_negative": data["true_negative"], "false_positive": data["false_positive"], "false_negative": data["false_negative"], "true_positive": data["true_positive"], "source": METRICS_PATH.name}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/class-distribution")
def class_distribution() -> dict[str, Any]:
    try: return {"splits": {name: details["binary_distribution"] for name, details in report()["splits"].items()}, "source": REPORT_PATH.name}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/temporal")
def temporal_results() -> dict[str, Any]:
    try:
        data = metrics()["original_event_detection"]["X"]
        return {"train_flare_type": "M", "held_out_test_flare_type": "X", "x_detected": data["detected_as_flare"], "x_support": data["support"], "x_recall": data["flare_recall"], "verdict": "Learning confirmed, generalization limited.", "production_ready": False}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/false-positives")
def false_positives() -> dict[str, Any]:
    try:
        saved = np.load(PREDICTIONS_PATH); pred, truth = saved["pred_flare"].astype(bool), saved["y_flare"].astype(bool); fp = pred & ~truth
        times = telemetry_frame().loc[lambda d: d.partition == report()["test_partition"], "timestamp"].iloc[29:].reset_index(drop=True)
        groups, start = [], None
        for index, active in enumerate(fp):
            if active and start is None: start = index
            elif not active and start is not None: groups.append((start, index - 1)); start = None
        if start is not None: groups.append((start, len(fp) - 1))
        episodes = [{"start": times.iloc[a], "end": times.iloc[b], "windows": b - a + 1} for a, b in groups]
        episodes.sort(key=lambda item: item["windows"], reverse=True)
        return {"false_positive_windows": int(fp.sum()), "episode_count": len(episodes), "episodes_at_least_one_minute": sum(item["windows"] >= 60 for item in episodes), "largest_episode": episodes[0] if episodes else "Unavailable", "probability_summary": {"false_positive_mean_flare_probability": 0.835741, "true_x_mean_flare_probability": 0.888503, "true_negative_quiet_mean_flare_probability": 0.070153, "source": "temporal_false_positive_analysis.md"}, "top_episodes": episodes[:12], "interpretation": "High sensitivity with poor specificity in this frozen exploratory experiment."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/dataset")
def dataset() -> dict[str, Any]:
    try:
        data = report()
        return {"feature_count": data["feature_count"], "sequence_length": data["sequence_length"], "split_policy": data["split_policy"], "leakage_audit": data["leakage_audit"], "raw_window_overlap": data["raw_window_overlap"], "splits": data["splits"], "observation_periods": ["2026-07-30", "2026-08-01"], "nan": 0, "inf": 0}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/features")
def features() -> dict[str, Any]:
    try:
        data = report()
        return {"count": data["feature_count"], "columns": data["feature_columns"], "causal": True, "description": "Flux, causal rates, ratio, and causal rolling mean/std features across 5m, 30m, 1h, and 6h windows."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/solexs")
def nowcasting_solexs(limit: int = Query(100, ge=1, le=1000), start_time: str | None = None, end_time: str | None = None) -> dict[str, Any]:
    try: return catalogue_response("solexs", limit, start_time, end_time)
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/helios")
def nowcasting_helios(limit: int = Query(100, ge=1, le=1000), start_time: str | None = None, end_time: str | None = None) -> dict[str, Any]:
    try: return catalogue_response("helios", limit, start_time, end_time)
    except Exception as exc: raise unavailable(str(exc)) from exc


SOURCE_ALIASES = {"solexs": "SoLEXS", "helios": "HEL1OS", "both": "BOTH"}


@app.get("/api/nowcasting/master")
def nowcasting_master(limit: int = Query(100, ge=1, le=1000), start_time: str | None = None, end_time: str | None = None, source: str | None = None) -> dict[str, Any]:
    try:
        result = catalogue_response("master", limit, start_time, end_time)
        if source:
            # detection_source is stored as "SoLEXS" / "HEL1OS" / "BOTH" (mixed
            # case, and HEL1OS has a digit '1' not a letter 'i') — a plain
            # source.upper() comparison never matched "solexs" or "helios",
            # only "both". Normalize via an explicit alias map instead.
            normalized = SOURCE_ALIASES.get(source.strip().lower())
            if normalized is None:
                raise HTTPException(status_code=400, detail=f"Unknown source '{source}'; expected one of: {', '.join(SOURCE_ALIASES)}")
            result["data"] = [event for event in result["data"] if event["detection_source"] == normalized]
            result["total_matching"] = len(result["data"])
        return result
    except HTTPException: raise
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/summary")
def nowcasting_summary_endpoint() -> dict[str, Any]:
    try: return nowcasting_summary()
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/event/{event_id}")
def nowcasting_event(event_id: str) -> dict[str, Any]:
    try:
        for name, identifier in (("master", "master_event_id"), ("solexs", "event_id"), ("helios", "event_id")):
            frame = nowcasting_catalogue(name)
            found = frame.loc[frame[identifier] == event_id]
            if not found.empty:
                return {"catalogue": name, "event": {key: serialise(value) for key, value in found.iloc[0].to_dict().items()}, "classification_note": "Detector source/intensity is algorithmic. Historical reference label is context only."}
        raise HTTPException(status_code=404, detail="Nowcasting event not found")
    except HTTPException: raise
    except Exception as exc: raise unavailable(str(exc)) from exc
def serialise(value: Any) -> Any:
    if value is None or pd.isna(value): return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp): return value.isoformat()
    return value


@lru_cache(maxsize=1)
def report() -> dict[str, Any]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def metrics() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def nowcasting_summary() -> dict[str, Any]:
    return json.loads((NOWCASTING / "nowcasting_summary.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=3)
def nowcasting_catalogue(name: str) -> pd.DataFrame:
    filenames = {"solexs": "solexs_nowcast_catalogue.csv", "helios": "helios_nowcast_catalogue.csv", "master": "master_flare_catalogue.csv"}
    if name not in filenames: raise ValueError("unknown nowcasting catalogue")
    path = NOWCASTING / filenames[name]
    if not path.exists(): raise FileNotFoundError(path.name)
    frame = pd.read_csv(path)
    for column in ("start_time", "peak_time", "end_time"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def catalogue_response(name: str, limit: int, start_time: str | None, end_time: str | None) -> dict[str, Any]:
    frame = nowcasting_catalogue(name).copy()
    if start_time: frame = frame.loc[frame.peak_time >= pd.to_datetime(start_time)]
    if end_time: frame = frame.loc[frame.peak_time <= pd.to_datetime(end_time)]
    identifier = "master_event_id" if name == "master" else "event_id"
    frame = frame.sort_values("peak_time", ascending=False).head(limit)
    data = [{key: serialise(value) for key, value in row.items()} for row in frame.to_dict("records")]
    return {"catalogue": name, "identifier": identifier, "total_matching": len(frame), "data": data, "artifact_backed": True, "classification_note": "Relative intensity is algorithmic, not an absolute GOES C/M/X class; historical reference labels are context only."}


@lru_cache(maxsize=1)
def telemetry_frame() -> pd.DataFrame:
    if not all(path.exists() for path in (FEATURES_PATH, TIMELINE_PATH, LABELS_PATH)):
        raise FileNotFoundError("processed telemetry artifacts")
    features, timeline, labels = (pd.read_csv(path) for path in (FEATURES_PATH, TIMELINE_PATH, LABELS_PATH))
    for frame in (features, timeline, labels):
        if "timestamp" in frame.columns: frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = features.merge(timeline[["timestamp", "partition_id"]], on="timestamp", how="left")
    frame = frame.rename(columns={"partition_id": "partition"})
    frame = frame.merge(labels, on="timestamp", how="left", suffixes=("", "_label"))
    token_to_class = {"quiet": 0, "c": 1, "m": 2, "x": 3}
    frame["original_class"] = frame.get("flare_class_token", "quiet").fillna("quiet").astype(str).str.lower().map(token_to_class).fillna(0).astype(int)
    frame["label"] = frame["original_class"].isin([2, 3]).astype(int)
    return frame.sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=1)
def frozen_model() -> KarnMultiTaskEngine:
    if not CHECKPOINT.exists(): raise FileNotFoundError(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = KarnMultiTaskEngine(input_dim=21, hidden_dim=64, num_classes=2)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def source_status() -> dict[str, Any]:
    return {"experiment": "KARN Binary Flare Detector — Temporal Generalization Experiment", "model_status": "Frozen exploratory model; not production ready", "scientific_status": "EXPLORATORY — GENERALIZATION LIMITED", "data_source": "Real processed Aditya+HEL1OS compatible telemetry", "live_feed": False, "input_shape": [30, 21]}


@app.get("/")
def root() -> dict[str, Any]: return {"name": "KARN Research API", "api_prefix": "/api", **source_status()}


@app.get("/api/health")
def health() -> dict[str, Any]:
    required = [FEATURES_PATH, TIMELINE_PATH, LABELS_PATH, REPORT_PATH, METRICS_PATH, CHECKPOINT, SCALER_PATH]
    nowcast_files = [NOWCASTING / name for name in ("solexs_nowcast_catalogue.csv", "helios_nowcast_catalogue.csv", "master_flare_catalogue.csv", "nowcasting_summary.json")]
    return {"status": "ok" if all(path.exists() for path in required) else "degraded", "artifacts_available": all(path.exists() for path in required), "nowcasting_available": all(path.exists() for path in nowcast_files), "solexs_catalogue_available": nowcast_files[0].exists(), "helios_catalogue_available": nowcast_files[1].exists(), "master_catalogue_available": nowcast_files[2].exists(), "catalogue_event_count": nowcasting_summary().get("counts", {}).get("master", "Unavailable") if nowcast_files[3].exists() else "Unavailable", **source_status()}


@app.get("/api/status")
def status() -> dict[str, Any]:
    try:
        frame = telemetry_frame()
        return {**source_status(), "observation_rows": len(frame), "observation_start": frame.timestamp.min(), "observation_end": frame.timestamp.max(), "partitions": sorted(frame.partition.dropna().unique().tolist()), "nowcasting": nowcasting_summary() if (NOWCASTING / "nowcasting_summary.json").exists() else "Unavailable"}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/telemetry")
def telemetry(limit: int = Query(900, ge=30, le=10000), partition: str | None = None) -> dict[str, Any]:
    try:
        frame = telemetry_frame()
        if partition: frame = frame.loc[frame["partition"] == partition]
        if frame.empty: raise HTTPException(status_code=404, detail="No real processed observations match that partition")
        stride = max(1, int(np.ceil(len(frame) / limit)))
        display = frame.iloc[::stride].copy()
        fields = ["timestamp", "partition", "solexs_flux", "hel1os_flux", "solexs_rate", "hel1os_rate", "soft_hard_ratio", "anomaly_flag", "quality_flag", "label", "original_class"]
        fields = [field for field in fields if field in display]
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        records = [{key: serialise(value) for key, value in row.items()} for row in display[fields].to_dict("records")]
        return {"source": "real processed telemetry; display-downsampled only", "total_rows": len(frame), "returned_rows": len(display), "stride": stride, "data": records}
    except HTTPException: raise
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/telemetry/latest")
def latest_telemetry() -> dict[str, Any]:
    try:
        row = telemetry_frame().iloc[-1]
        fields = ["timestamp", "partition", "solexs_flux", "hel1os_flux", "solexs_rate", "hel1os_rate", "soft_hard_ratio", "anomaly_flag", "quality_flag", "label", "original_class"]
        return {"source": "latest available historical processed observation (not a live feed)", "observation": {key: serialise(row[key]) for key in fields if key in row.index}}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/model/status")
def model_status() -> dict[str, Any]:
    try:
        model = frozen_model()
        return {**source_status(), "checkpoint": CHECKPOINT.name, "checkpoint_available": True, "classifier_output_classes": model.classifier_head.out_features, "inference_available": SCALER_PATH.exists()}
    except Exception as exc: raise unavailable(f"frozen checkpoint could not be loaded: {exc}") from exc


@app.get("/api/model/prediction")
def model_prediction() -> dict[str, Any]:
    """Run the frozen model on the last real contiguous processed 30-step window."""
    try:
        data, feature_columns = telemetry_frame(), report()["feature_columns"]
        last_partition = data["partition"].dropna().iloc[-1]
        window = data.loc[data["partition"] == last_partition].tail(30).copy()
        if len(window) != 30 or not np.all(np.diff(window.timestamp.astype("int64") // 10**9) == 1): raise ValueError("latest processed data does not contain a contiguous 30-second window")
        raw = window[feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(raw).all(): raise ValueError("latest real feature window contains NaN or Inf")
        scaler = np.load(SCALER_PATH); scale = np.where(scaler["scale"] == 0, 1.0, scaler["scale"])
        transformed = ((raw - scaler["center"]) / scale).astype(np.float32, copy=False)
        with torch.no_grad(): logits, _ = frozen_model()(torch.from_numpy(transformed).unsqueeze(0)); probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        prediction = int(np.argmax(probabilities))
        return {**source_status(), "source": "inference on latest real historical processed 30-step window", "timestamp": window.timestamp.iloc[-1], "predicted_class": "Flare" if prediction else "Quiet", "probabilities": {"quiet": float(probabilities[0]), "flare": float(probabilities[1])}, "window_partition": str(last_partition), "window_shape": [30, 21], "historical_source_label": serialise(window.iloc[-1].get("original_class", "Unavailable")), "note": "Binary output is Quiet vs Flare; it does not distinguish M from X."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation")
def evaluation() -> dict[str, Any]:
    try: return {**source_status(), "metrics": metrics(), "limitation": "Threshold calibration is blocked because the chronological validation split has no flare examples."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/confusion-matrix")
def confusion_matrix() -> dict[str, Any]:
    try:
        data = metrics()
        return {"labels": ["Quiet", "Flare"], "matrix": data["confusion_matrix"], "true_negative": data["true_negative"], "false_positive": data["false_positive"], "false_negative": data["false_negative"], "true_positive": data["true_positive"], "source": METRICS_PATH.name}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/class-distribution")
def class_distribution() -> dict[str, Any]:
    try: return {"splits": {name: details["binary_distribution"] for name, details in report()["splits"].items()}, "source": REPORT_PATH.name}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/temporal")
def temporal_results() -> dict[str, Any]:
    try:
        data = metrics()["original_event_detection"]["X"]
        return {"train_flare_type": "M", "held_out_test_flare_type": "X", "x_detected": data["detected_as_flare"], "x_support": data["support"], "x_recall": data["flare_recall"], "verdict": "Learning confirmed, generalization limited.", "production_ready": False}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/evaluation/false-positives")
def false_positives() -> dict[str, Any]:
    try:
        saved = np.load(PREDICTIONS_PATH); pred, truth = saved["pred_flare"].astype(bool), saved["y_flare"].astype(bool); fp = pred & ~truth
        times = telemetry_frame().loc[lambda d: d.partition == report()["test_partition"], "timestamp"].iloc[29:].reset_index(drop=True)
        groups, start = [], None
        for index, active in enumerate(fp):
            if active and start is None: start = index
            elif not active and start is not None: groups.append((start, index - 1)); start = None
        if start is not None: groups.append((start, len(fp) - 1))
        episodes = [{"start": times.iloc[a], "end": times.iloc[b], "windows": b - a + 1} for a, b in groups]
        episodes.sort(key=lambda item: item["windows"], reverse=True)
        return {"false_positive_windows": int(fp.sum()), "episode_count": len(episodes), "episodes_at_least_one_minute": sum(item["windows"] >= 60 for item in episodes), "largest_episode": episodes[0] if episodes else "Unavailable", "probability_summary": {"false_positive_mean_flare_probability": 0.835741, "true_x_mean_flare_probability": 0.888503, "true_negative_quiet_mean_flare_probability": 0.070153, "source": "temporal_false_positive_analysis.md"}, "top_episodes": episodes[:12], "interpretation": "High sensitivity with poor specificity in this frozen exploratory experiment."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/dataset")
def dataset() -> dict[str, Any]:
    try:
        data = report()
        return {"feature_count": data["feature_count"], "sequence_length": data["sequence_length"], "split_policy": data["split_policy"], "leakage_audit": data["leakage_audit"], "raw_window_overlap": data["raw_window_overlap"], "splits": data["splits"], "observation_periods": ["2026-07-30", "2026-08-01"], "nan": 0, "inf": 0}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/features")
def features() -> dict[str, Any]:
    try:
        data = report()
        return {"count": data["feature_count"], "columns": data["feature_columns"], "causal": True, "description": "Flux, causal rates, ratio, and causal rolling mean/std features across 5m, 30m, 1h, and 6h windows."}
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/solexs")
def nowcasting_solexs(limit: int = Query(100, ge=1, le=1000), start_time: str | None = None, end_time: str | None = None) -> dict[str, Any]:
    try: return catalogue_response("solexs", limit, start_time, end_time)
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/helios")
def nowcasting_helios(limit: int = Query(100, ge=1, le=1000), start_time: str | None = None, end_time: str | None = None) -> dict[str, Any]:
    try: return catalogue_response("helios", limit, start_time, end_time)
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/master")
def nowcasting_master(limit: int = Query(100, ge=1, le=1000), start_time: str | None = None, end_time: str | None = None, source: str | None = None) -> dict[str, Any]:
    try:
        result = catalogue_response("master", limit, start_time, end_time)
        if source:
            result["data"] = [event for event in result["data"] if event["detection_source"] == source.upper()]
            result["total_matching"] = len(result["data"])
        return result
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/summary")
def nowcasting_summary_endpoint() -> dict[str, Any]:
    try: return nowcasting_summary()
    except Exception as exc: raise unavailable(str(exc)) from exc


@app.get("/api/nowcasting/event/{event_id}")
def nowcasting_event(event_id: str) -> dict[str, Any]:
    try:
        for name, identifier in (("master", "master_event_id"), ("solexs", "event_id"), ("helios", "event_id")):
            frame = nowcasting_catalogue(name)
            found = frame.loc[frame[identifier] == event_id]
            if not found.empty:
                return {"catalogue": name, "event": {key: serialise(value) for key, value in found.iloc[0].to_dict().items()}, "classification_note": "Detector source/intensity is algorithmic. Historical reference label is context only."}
        raise HTTPException(status_code=404, detail="Nowcasting event not found")
    except HTTPException: raise
    except Exception as exc: raise unavailable(str(exc)) from exc
