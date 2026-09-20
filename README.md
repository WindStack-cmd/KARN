# KARN

KARN is a research prototype for analysing compatible Aditya/HEL1OS and
SoLEXS telemetry with a binary solar-flare detector. The project presents the
complete scientific path: real FITS data, synchronization, causal features,
chronological windows, frozen model inference, and honest temporal evaluation.

## Scientific status


**Exploratory — generalization limited. Not a production-ready flare
classifier.** The frozen temporal experiment trained on real M-flare windows
from the earlier observation period and evaluated once on a later, untouched
X-flare period. It detected 901/901 X windows (recall 1.0), but has a very high
false-positive burden: flare precision 0.039177, F1 0.075401, and 22,097 false
positive Quiet windows. No threshold was tuned on the test data.

Only two compatible paired observation periods are available in this
repository: `2026-07-30` (Quiet + M) and `2026-08-01` (Quiet + X). The
chronological validation period has no positive flare examples, so independent
paired observations are required for scientifically valid calibration.

## Architecture

```
Raw FITS telemetry → parsing / timestamp correction → synchronization
→ 21 causal scientific features → 30-step causal windows
→ chronological split + purge → KarnMultiTaskEngine binary experiment
→ frozen evaluation artifacts → FastAPI → React dashboard
```

The frozen model consumes `(batch, 30, 21)` inputs and produces a binary
`Quiet`/`Flare` output. It does not distinguish M from X.

## Dataset and experiment

| Split | Windows | Quiet | Flare |
| --- | ---: | ---: | ---: |
| Train | 36,663 | 35,791 | 872 M |
| Validation | 6,441 | 6,441 | 0 |
| Untouched test | 43,158 | 42,257 | 901 X |

The feature set has 21 flux, causal-rate, ratio, and causal rolling-statistics
features. The feature/window audit reports zero NaN, zero Inf, zero raw-window
overlap, and leakage audit **PASS**.

The frozen temporal test confusion matrix is:

```
[[20160, 22097],
 [    0,   901]]
```

The false-positive diagnostic identifies 225 episodes (38 lasting at least one
minute); the largest spans `2026-08-01 12:00:43` to `13:29:53` with 5,351
windows. See `ml/eval_results/diagnostics/` for the preserved scientific audits.

## Run the demo

The app intentionally uses only local processed telemetry and frozen artifacts;
it does not fetch GOES data, create synthetic telemetry, or persist predictions.

```powershell
# from the repository root
python -m uvicorn ml.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open the Vite URL (normally `http://localhost:5173`). Set `VITE_API_URL` only
when the artifact-backed API is hosted elsewhere; its default is
`http://127.0.0.1:8000/api`.

## API

- `GET /api/health`, `/api/status`
- `GET /api/telemetry`, `/api/telemetry/latest`
- `GET /api/model/status`, `/api/model/prediction`
- `GET /api/evaluation`, `/api/evaluation/confusion-matrix`
- `GET /api/evaluation/class-distribution`, `/api/evaluation/temporal`
- `GET /api/evaluation/false-positives`
- `GET /api/dataset`, `/api/features`
- `GET /api/nowcasting/summary`
- `GET /api/nowcasting/solexs`, `/api/nowcasting/helios`
- `GET /api/nowcasting/master`, `/api/nowcasting/event/{event_id}`

Every dashboard metric comes from real processed data or saved evaluation
artifacts. If an artifact cannot be read, the API responds with `Unavailable`
rather than manufacturing a value.

## Phase A: independent nowcasting catalogues

Phase A adds deterministic, artifact-backed event detection beside the frozen
ML baseline; it does not retrain or modify that baseline.

```text
real synchronized telemetry -> independent SoLEXS detector
                           -> independent HEL1OS detector
                           -> temporal association (300 seconds)
                           -> master event catalogue
```

Run catalogue generation explicitly from the repository root:

```powershell
python ml/nowcasting.py
```

It writes new derived artifacts only under `data/processed/nowcasting/`:
SoLEXS, HEL1OS, and master catalogues in CSV/JSON plus a summary JSON. Each
detector uses causal 15-second median smoothing, a 15-minute trailing
low-quantile baseline, sustained ratio and excess thresholds, a 30-second
minimum duration, 120-second event separation, and a sustained recovery rule.
All detector parameters are recorded in the generated summary.

The categories `low-relative-intensity`, `medium-relative-intensity`, and
`high-relative-intensity` are detector-relative categories only. They are not
absolute GOES C/M/X classes. Existing M/X labels, when temporally present, are
historical reference context and are not detector input or output. No
catalogue-level precision/recall is claimed because the labels are not
independently verified per-instrument event-detection ground truth.

**KARN nowcasting is an artifact-backed research implementation and not a
production operational flare warning system.**

## Phase B: causal live-nowcast replay

The repository contains no genuine streaming Aditya-L1 telemetry endpoint.
KARN therefore does **not** claim a live instrument feed. Instead, Phase B
provides an interactive **Historical Replay** mode that sends the real saved
synchronized observations through independent SoLEXS and HEL1OS state machines
one timestamp at a time.

```text
one real replay observation -> causal median window -> trailing background
-> persistence check -> NOWCAST TRIGGERED -> DEVELOPING / RECOVERING -> ENDED
```

At time `T`, smoothing uses the current and earlier observations only; the
background excludes the current observation and contains no later samples.
There is no backward filling, future-window smoothing, future label, future
peak, or future end-time access. A peak is reported only as `peak_so_far` while
an event remains active; the final peak/end are set after causal recovery.

Replay API routes:

- `GET /api/nowcast/live`, `/api/nowcast/status`, `/api/nowcast/events`
- `POST /api/nowcast/replay/reset?start_index=0`
- `POST /api/nowcast/replay/advance?steps=60`

The dashboard exposes Reset, manual advance, and automatic replay controls.
It always labels this source as **HISTORICAL REPLAY**, not a live Aditya-L1
stream. A `BOTH` live-master association is created only when independently
active SoLEXS and HEL1OS events trigger within the configured 300 seconds.

There is no configured PRADAN/ISSDC Aditya streaming endpoint or credential in
this repository. The replay source is `data/processed/aditya_synced_timeline.csv`,
which is derived from the paired saved FITS observations. Legacy NOAA/GOES
download utilities are not part of this nowcasting path.

## Project layout

- `data/raw/` — paired real FITS source observations
- `data/processed/` — synchronized telemetry, labels, features, and windows
- `ml/` — data pipeline, model, frozen checkpoint, evaluation, and API
- `ml/eval_results/diagnostics/` — preserved scientific diagnostic reports
- `frontend/` — React/Vite research dashboard

## Future work

Acquire more independent paired Aditya+HEL1OS observation periods, especially
chronological validation periods containing real flares. Only then evaluate a
pre-registered calibration/training experiment using training and validation
data; the held-out X test must remain untouched.
