"""Causal historical-replay engine for KARN live-nowcast development.

No genuine streaming Aditya-L1 endpoint exists in this repository.  This engine
therefore replays *real saved synchronized observations* in timestamp order.
Each ``advance`` call processes one or more newly available samples; it never
reads a later observation to make the current decision.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import pandas as pd


def _safe_flag(value: Any) -> int:
    """Coerce a quality/anomaly flag to int, treating NaN/None as 0 (unflagged)
    instead of crashing. ``int(value or 0)`` looks safe but fails on NaN,
    since NaN is truthy in Python and int(NaN) raises ValueError."""
    return int(value) if pd.notna(value) else 0


try:
    from .nowcasting import HEL1OS_PARAMETERS, SOLEXS_PARAMETERS, DetectorParameters, load_telemetry
except ImportError:
    from nowcasting import HEL1OS_PARAMETERS, SOLEXS_PARAMETERS, DetectorParameters, load_telemetry


@dataclass
class LiveInstrument:
    source: str
    flux_column: str
    params: DetectorParameters
    raw: deque[float] = field(default_factory=deque)
    smoothed: deque[float] = field(default_factory=deque)
    candidate_count: int = 0
    recovery_count: int = 0
    state: str = "NORMAL"
    active_event: dict[str, Any] | None = None
    completed_events: list[dict[str, Any]] = field(default_factory=list)
    event_number: int = 0
    current: dict[str, Any] | None = None
    candidate_start_time: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self.raw = deque(maxlen=self.params.smoothing_seconds)
        self.smoothed = deque(maxlen=self.params.baseline_seconds)

    def observe(self, timestamp: pd.Timestamp, value: Any, quality_flag: Any, anomaly_flag: Any) -> None:
        valid = pd.notna(value) and np.isfinite(float(value)) and _safe_flag(quality_flag) == 0
        if not valid:
            self.current = {"timestamp": timestamp, "available": False, "message": "Invalid or quality-flagged observation"}
            return
        self.raw.append(float(value))
        if len(self.raw) < self.params.smoothing_seconds:
            self.current = {"timestamp": timestamp, "available": True, "raw_value": float(value), "message": "Waiting for causal smoothing window"}
            return
        smooth = float(np.median(np.asarray(self.raw)))
        # Baseline is calculated before appending the current smooth point.
        # Thus it only contains values available strictly before this decision.
        if len(self.smoothed) < self.params.baseline_seconds:
            self.smoothed.append(smooth)
            self.current = {"timestamp": timestamp, "available": True, "raw_value": float(value), "smoothed_value": smooth, "message": "Insufficient trailing background history"}
            return
        background = max(float(np.quantile(np.asarray(self.smoothed), self.params.baseline_quantile)), 1.0)
        ratio, excess = smooth / background, smooth - background
        threshold = max(background * self.params.ratio_threshold, background + self.params.minimum_excess)
        candidate = ratio >= self.params.ratio_threshold and excess >= self.params.minimum_excess
        self.smoothed.append(smooth)
        self.current = {"timestamp": timestamp, "available": True, "raw_value": float(value), "smoothed_value": smooth, "background": background, "threshold": threshold, "ratio": ratio, "excess": excess, "anomaly_flag": _safe_flag(anomaly_flag)}
        if self.active_event is None:
            if candidate:
                self.candidate_count += 1
                if self.candidate_count == 1:
                    # Record the real timestamp of the first candidate sample instead
                    # of assuming every intervening second was actually observed —
                    # back-calculating via subtraction silently mis-dates the event
                    # start whenever there's a sensor gap during the candidate run.
                    self.candidate_start_time = timestamp
            else:
                self.candidate_count = 0
                self.candidate_start_time = None
            self.state = "CANDIDATE" if self.candidate_count else "NORMAL"
            if self.candidate_count >= self.params.minimum_duration_seconds:
                self.event_number += 1
                start_time = self.candidate_start_time or (timestamp - pd.Timedelta(seconds=self.params.minimum_duration_seconds - 1))
                self.active_event = {"event_id": f"LIVE-{self.source[:3].upper()}-{start_time:%Y%m%dT%H%M%S}-{self.event_number:03d}", "source": self.source, "start_time": start_time, "trigger_time": timestamp, "peak_so_far": smooth, "peak_time_so_far": timestamp, "background_at_trigger": background, "trigger_ratio": ratio, "trigger_excess": excess, "anomaly_samples": _safe_flag(anomaly_flag), "state": "NOWCAST TRIGGERED"}
                self.state = "NOWCAST TRIGGERED"
        else:
            event = self.active_event
            event["anomaly_samples"] += _safe_flag(anomaly_flag)
            if smooth > event["peak_so_far"]:
                event["peak_so_far"], event["peak_time_so_far"] = smooth, timestamp
            if candidate:
                self.recovery_count = 0
                self.state = "DEVELOPING"
            else:
                self.recovery_count += 1
                self.state = "RECOVERING"
                if self.recovery_count >= self.params.recovery_seconds:
                    event.update({"end_time": timestamp, "duration_seconds": int((timestamp - event["start_time"]).total_seconds()) + 1, "state": "ENDED", "final_peak": event["peak_so_far"], "final_peak_time": event["peak_time_so_far"]})
                    self.completed_events.append(event)
                    self.active_event = None
                    self.candidate_count = self.recovery_count = 0
                    self.candidate_start_time = None
                    self.state = "ENDED"
        if self.active_event:
            event = self.active_event
            duration = int((timestamp - event["start_time"]).total_seconds()) + 1
            event["duration_so_far_seconds"] = duration
            event["state"] = self.state
            event["confidence"] = min(1.0, (ratio / self.params.ratio_threshold) * min(1.0, duration / self.params.minimum_duration_seconds))

    def response(self) -> dict[str, Any]:
        output = {
            "source": self.source,
            "state": self.state,
            "parameters": asdict(self.params),
            "observation": self.current,
            "active_event": self.active_event,
            "completed_event_count": len(self.completed_events),
            "candidate_duration_seconds": self.candidate_count,
            "recovery_duration_seconds": self.recovery_count,
        }
        return output


class TelemetrySource(Protocol):
    """A source that yields observations in arrival order."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        ...


class StreamingNowcastEngine:
    """The single detector implementation used by live sources and replay."""

    def __init__(self) -> None:
        self.solexs = LiveInstrument("SoLEXS", "solexs_flux", SOLEXS_PARAMETERS)
        self.helios = LiveInstrument("HEL1OS", "hel1os_flux", HEL1OS_PARAMETERS)
        self.history: deque[dict[str, Any]] = deque(maxlen=600)
        self.master_events: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.solexs = LiveInstrument("SoLEXS", "solexs_flux", SOLEXS_PARAMETERS)
        self.helios = LiveInstrument("HEL1OS", "hel1os_flux", HEL1OS_PARAMETERS)
        self.history, self.master_events = deque(maxlen=600), []

    def ingest(self, row: dict[str, Any]) -> dict[str, Any]:
        """Consume exactly one observation; no source may provide future rows."""
        timestamp = pd.Timestamp(row["timestamp"])
        self.solexs.observe(timestamp, row.get("solexs_flux"), row.get("quality_flag"), row.get("anomaly_flag"))
        self.helios.observe(timestamp, row.get("hel1os_flux"), row.get("quality_flag"), row.get("anomaly_flag"))
        self.history.append({
            "timestamp": timestamp,
            "solexs_flux": float(row["solexs_flux"]) if pd.notna(row.get("solexs_flux")) else None,
            "hel1os_flux": float(row["hel1os_flux"]) if pd.notna(row.get("hel1os_flux")) else None,
            "solexs_background": self.solexs.current.get("background") if self.solexs.current else None,
            "helios_background": self.helios.current.get("background") if self.helios.current else None,
            "solexs_threshold": self.solexs.current.get("threshold") if self.solexs.current else None,
            "helios_threshold": self.helios.current.get("threshold") if self.helios.current else None,
        })
        self._associate()
        return self.status(timestamp)

    def _associate(self) -> None:
        soft, hard = self.solexs.active_event, self.helios.active_event
        if not soft or not hard:
            return
        gap = abs((pd.Timestamp(soft["trigger_time"]) - pd.Timestamp(hard["trigger_time"])).total_seconds())
        paired_ids = {(event["solexs_event_id"], event["helios_event_id"]) for event in self.master_events}
        pair = (soft["event_id"], hard["event_id"])
        if gap <= SOLEXS_PARAMETERS.association_seconds and pair not in paired_ids:
            self.master_events.append({
                "master_event_id": f"LIVE-MASTER-{pd.Timestamp(max(soft['trigger_time'], hard['trigger_time'])):%Y%m%dT%H%M%S}",
                "source": "BOTH",
                "detection_source": "BOTH",
                "state": "DEVELOPING",
                "solexs_event_id": soft["event_id"],
                "helios_event_id": hard["event_id"],
                "solexs_contribution": soft,
                "helios_contribution": hard,
                "association_seconds": gap,
                "trigger_time": max(soft["trigger_time"], hard["trigger_time"]),
            })

    def status(self, timestamp: pd.Timestamp | None = None) -> dict[str, Any]:
        active_sources = [item for item in (self.solexs.active_event, self.helios.active_event) if item]
        current_pair = None if len(active_sources) != 2 else (self.solexs.active_event["event_id"], self.helios.active_event["event_id"])
        associated = next((event for event in reversed(self.master_events) if current_pair and (event["solexs_event_id"], event["helios_event_id"]) == current_pair), None)
        source = "BOTH" if associated else active_sources[0]["source"] if len(active_sources) == 1 else "UNASSOCIATED_MULTISOURCE" if active_sources else None
        if associated:
            master_active = associated
        elif len(active_sources) == 1:
            event = active_sources[0]
            master_active = {
                **event,
                "master_event_id": f"MASTER-{event['event_id']}",
                "detection_source": event["source"],
                "source": "single-source nowcast",
                "solexs_contribution": event if event["source"] == "SoLEXS" else None,
                "helios_contribution": event if event["source"] == "HEL1OS" else None,
            }
        else:
            master_active = None
        state = "NORMAL" if not active_sources else "FLARE RECOVERING" if any(item["state"] == "RECOVERING" for item in active_sources) else "FLARE DEVELOPING"
        return {
            "mode": "STREAMING_NOWCAST_ENGINE",
            "master": {"state": state, "source": source, "active_event": master_active},
            "solexs": self.solexs.response(),
            "helios": self.helios.response(),
            "recent_telemetry": list(self.history),
            "completed_events": {"solexs": self.solexs.completed_events[-10:], "helios": self.helios.completed_events[-10:]},
            "master_events": self.master_events[-10:],
            "latest_timestamp": timestamp,
        }


class HistoricalReplay:
    """Sequential source adapter; it feeds the shared streaming engine."""

    def __init__(self) -> None:
        self.frame = load_telemetry().sort_values(["partition_id", "timestamp"]).reset_index(drop=True)
        self.engine = StreamingNowcastEngine()
        self.cursor = 0

    @property
    def solexs(self) -> LiveInstrument:
        return self.engine.solexs

    @property
    def helios(self) -> LiveInstrument:
        return self.engine.helios

    @property
    def history(self) -> deque[dict[str, Any]]:
        return self.engine.history

    @property
    def master_events(self) -> list[dict[str, Any]]:
        return self.engine.master_events

    def reset(self, start_index: int = 0) -> None:
        if not 0 <= start_index < len(self.frame):
            raise ValueError("Replay start index is out of range")
        self.engine.reset()
        self.cursor = start_index

    def advance(self, steps: int = 1) -> dict[str, Any]:
        if not 1 <= steps <= 3600: raise ValueError("steps must be between 1 and 3600")
        for _ in range(steps):
            if self.cursor >= len(self.frame): break
            row = self.frame.iloc[self.cursor]
            # Partition discontinuities reset causal state; historical earlier
            # periods are never allowed to become background for later periods.
            if self.cursor and row.partition_id != self.frame.iloc[self.cursor - 1].partition_id:
                self.engine.reset()
            self.engine.ingest(row.to_dict())
            self.cursor += 1
        return self.status()

    def status(self) -> dict[str, Any]:
        timestamp = None if self.cursor == 0 else self.frame.iloc[min(self.cursor - 1, len(self.frame) - 1)].timestamp
        current = self.engine.status(timestamp)
        solexs_observation = current["solexs"].get("observation") or {}
        helios_observation = current["helios"].get("observation") or {}
        current.update({
            "mode": "HISTORICAL_REPLAY",
            "live_aditya_feed": False,
            "message": "Historical replay of real processed telemetry; not a live Aditya-L1 feed.",
            "source_mode": "REAL_ADITYA_L1_HISTORICAL_REPLAY",
            "source_status": "REPLAY_COMPLETE" if self.cursor >= len(self.frame) else "REAL_ARCHIVE_AVAILABLE",
            "connection_status": "LOCAL_REAL_TELEMETRY_AVAILABLE",
            "latest_timestamp": timestamp,
            "latest_solexs_reading": solexs_observation.get("raw_value"),
            "latest_hel1os_reading": helios_observation.get("raw_value"),
            "replay": {"cursor": self.cursor, "total_observations": len(self.frame), "complete": self.cursor >= len(self.frame), "timestamp": timestamp, "partition": None if self.cursor == 0 else self.frame.iloc[self.cursor - 1].partition_id},
        })
        return current


def validate_replay() -> dict[str, Any]:
    """Exercise no-event, single-source, both-source, and recovery states causally."""
    replay = HistoricalReplay()
    states, first_trigger = set(), {"SoLEXS": None, "HEL1OS": None}
    while not replay.status()["replay"]["complete"]:
        result = replay.advance(steps=300)
        states.update((result["solexs"]["state"], result["helios"]["state"], result["master"]["state"]))
        for source, payload in (("SoLEXS", result["solexs"]), ("HEL1OS", result["helios"])):
            if payload["active_event"] and first_trigger[source] is None: first_trigger[source] = payload["active_event"]["trigger_time"]
    return {"mode": "HISTORICAL_REPLAY", "states_observed": sorted(states), "first_triggers": first_trigger, "completed": {"solexs": len(replay.solexs.completed_events), "helios": len(replay.helios.completed_events)}, "future_data_used": False, "note": "Validation advances sequentially; at each sample background excludes the current and all future samples."}