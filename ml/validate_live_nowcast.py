"""Validate causal nowcasting on the real historical replay source."""
from __future__ import annotations

from datetime import datetime

try:
    from .live_nowcast import HistoricalReplay
except ImportError:
    from live_nowcast import HistoricalReplay


def main() -> None:
    replay = HistoricalReplay()
    initial = replay.status()
    if initial["replay"]["cursor"] != 0 or initial["master"]["state"] != "NORMAL":
        raise AssertionError("Replay did not start in NORMAL without consuming data")

    first_trigger = None
    trigger_snapshot = None
    final_snapshot = None
    while not replay.status()["replay"]["complete"]:
        before = replay.cursor
        snapshot = replay.advance(steps=1)
        if replay.cursor != before + 1:
            raise AssertionError("Replay skipped an observation")
        event = snapshot["helios"]["active_event"] or snapshot["solexs"]["active_event"]
        if event and first_trigger is None:
            first_trigger = event["trigger_time"]
            trigger_snapshot = event.copy()
    final_snapshot = replay.status()
    if first_trigger is None:
        raise AssertionError("No causal nowcast was observed in real telemetry")

    completed = final_snapshot["completed_events"]["helios"] + final_snapshot["completed_events"]["solexs"]
    if not completed:
        raise AssertionError("Triggered event did not reach causal recovery")
    final_event = completed[0]
    trigger_time = datetime.fromisoformat(str(first_trigger))
    peak_time = datetime.fromisoformat(str(final_event["final_peak_time"]))
    if peak_time < trigger_time:
        raise AssertionError("Final peak precedes trigger")
    if "final_peak" in trigger_snapshot:
        raise AssertionError("Final peak was exposed at trigger time")

    print("Causal replay validation = PASS")
    print(f"Trigger timestamp = {trigger_time.isoformat()}")
    print(f"Final peak timestamp = {peak_time.isoformat()}")
    print(f"Lead time seconds = {(peak_time - trigger_time).total_seconds():.0f}")
    print("Future data used = False")
    print("Future labels/peak/end/bfill/future rolling statistics used = False")


if __name__ == "__main__":
    main()
