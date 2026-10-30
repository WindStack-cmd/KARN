"""Audit official NOAA C/M/X events against existing Aditya-L1 coverage."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "official" / "noaa_ncei_goes_flare_report_2026.csv"
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "results" / "real_data_availability_audit.csv"


def _fits_coverage(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        header = hdul[1].header
        instrument = str(header.get("INSTRUME", "")).upper()
        if instrument == "SOLEXS":
            start = pd.to_datetime(header["DATE-OBS"], utc=True)
            end = pd.to_datetime(header["DATE-END"], utc=True)
        else:
            start = pd.Timestamp("1858-11-17", tz="UTC") + pd.to_timedelta(float(header["TSTART"]), unit="D")
            end = pd.Timestamp("1858-11-17", tz="UTC") + pd.to_timedelta(float(header["TSTOP"]), unit="D")
        return start, end


def _coverage() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(RAW.glob("day_*/*.fits*")):
        start, end = _fits_coverage(path)
        name = path.name.lower()
        instrument = "SoLEXS" if "solexs" in name else "HEL1OS"
        records.append({"source": str(path.relative_to(ROOT)), "instrument": instrument, "start": start, "end": end})
    return records


def main() -> None:
    catalog = pd.read_csv(CATALOG)
    catalog["start"] = pd.to_datetime(catalog["start_time"], utc=True, errors="coerce")
    catalog["peak"] = pd.to_datetime(catalog["time"], utc=True, errors="coerce")
    catalog["end"] = pd.to_datetime(catalog["end_time"], utc=True, errors="coerce")
    catalog["class_letter"] = catalog["flare_class"].astype(str).str.extract(r"([CMX])", expand=False)
    catalog = catalog[catalog["class_letter"].isin(["C", "M", "X"])].copy()
    coverage = _coverage()
    rows: list[dict[str, object]] = []
    for _, event in catalog.iterrows():
        event_end = event["end"]
        instruments: dict[str, list[dict[str, object]]] = {"SoLEXS": [], "HEL1OS": []}
        for item in coverage:
            if pd.notna(event_end) and item["start"] <= event_end and item["end"] >= event["start"]:
                instruments[str(item["instrument"])].append(item)
        simultaneous = []
        if pd.notna(event_end):
            for soft in instruments["SoLEXS"]:
                for hard in instruments["HEL1OS"]:
                    start = max(event["start"], soft["start"], hard["start"])
                    end = min(event_end, soft["end"], hard["end"])
                    if start <= end:
                        simultaneous.append((start, end))
        overlap_minutes = max(((end - start).total_seconds() / 60 for start, end in simultaneous), default=0.0)
        rows.append(
            {
                "noaa_event_id": str(event["flare_id"]),
                "class": event["flare_class"],
                "start_utc": event["start"].isoformat(),
                "peak_utc": event["peak"].isoformat() if pd.notna(event["peak"]) else "",
                "end_utc": event["end"].isoformat() if pd.notna(event["end"]) else "",
                "solexs_available": "YES" if instruments["SoLEXS"] else "NO",
                "hel1os_available": "YES" if instruments["HEL1OS"] else "NO",
                "simultaneous_overlap_minutes": overlap_minutes,
                "usable_for_karn": "YES" if overlap_minutes > 0 else "NO",
                "reason": "both instruments overlap event" if overlap_minutes > 0 else "no simultaneous complete SoLEXS/HEL1OS coverage",
            }
        )
    result = pd.DataFrame(rows).sort_values("start_utc")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Official NOAA C/M/X events: {len(result)}")
    print(f"Usable events: {int((result['usable_for_karn'] == 'YES').sum())}")
    print(result[result["usable_for_karn"] == "YES"].to_string(index=False))
    print(f"Audit written to {OUTPUT}")


if __name__ == "__main__":
    main()
