import torch
import torch.nn as nn
import numpy as np
import requests
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from datetime import datetime

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KARN — Solar Flare Forecasting API",
    description="Real-time solar flare nowcasting and forecasting using GOES-16 X-ray data",
    version="1.0.0"
)

# CORS — allows React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Architecture ────────────────────────────────────────────────────────
class FlareRecurrentLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])

# ── Load Model Once at Startup ────────────────────────────────────────────────
print("Loading KARN LSTM model...")
model = FlareRecurrentLSTM()
model.load_state_dict(torch.load("karn_lstm_v2.pth",
                                  map_location="cpu",
                                  weights_only=True))
model.eval()
print("Model loaded successfully")

CLASS_NAMES = ["Quiet", "C-class", "M-class", "X-class"]

# ── Helper Functions ──────────────────────────────────────────────────────────
def log_normalize(arr):
    arr     = np.clip(arr, 1e-10, None)
    log     = np.log10(arr)
    log_min = np.log10(1e-9)
    log_max = np.log10(1e-3)
    norm    = (log - log_min) / (log_max - log_min)
    return np.clip(norm, 0, 1)

def compute_severity(probs):
    return (1 * probs[1]) + (10 * probs[2]) + (100 * probs[3])

def compute_alert(probs, score):
    if probs[3] >= 0.3 or score >= 30.0:
        return "CRITICAL", "X-class flare imminent — immediate action required"
    elif probs[2] >= 0.5 or score >= 5.0:
        return "WARNING", "M-class flare expected — elevated solar activity"
    elif probs[1] >= 0.5 or score >= 1.0:
        return "ADVISORY", "C-class flare activity detected"
    else:
        return "GREEN", "Quiet Sun — normal solar weather"

def run_inference(window_array):
    """
    Takes a (60, 2) numpy array, runs KARN model, returns prediction dict.
    """
    # Validate shape
    if window_array.shape != (60, 2):
        raise ValueError(f"Expected shape (60, 2), got {window_array.shape}")

    # Check for NaN or inf
    if np.any(np.isnan(window_array)) or np.any(np.isinf(window_array)):
        raise ValueError("Input contains NaN or infinite values")

    # Log normalize
    soft_norm = log_normalize(window_array[:, 0])
    hard_norm = log_normalize(window_array[:, 1])
    normalized = np.stack([soft_norm, hard_norm], axis=1).astype(np.float32)

    # Run model
    tensor = torch.tensor(normalized).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_class = int(np.argmax(probs))
    score      = compute_severity(probs)
    level, msg = compute_alert(probs, score)

    return {
        "predicted_class":  CLASS_NAMES[pred_class],
        "probabilities": {
            "quiet":   round(probs[0], 4),
            "c_class": round(probs[1], 4),
            "m_class": round(probs[2], 4),
            "x_class": round(probs[3], 4),
        },
        "severity_score": round(score, 4),
        "alert_level":    level,
        "alert_message":  msg,
        "timestamp":      datetime.utcnow().isoformat() + "Z"
    }

def fetch_live_goes():
    """
    Fetch latest GOES-16 X-ray data and build a 60-minute window.
    """
    url      = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    response = requests.get(url, timeout=30)
    raw      = response.text.strip().replace('][', ',')
    data     = json.loads(raw)
    df       = pd.DataFrame(data)

    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['flux']     = pd.to_numeric(df['flux'], errors='coerce')
    df             = df.dropna(subset=['flux'])
    df             = df.sort_values('time_tag')

    # Get last 60 minutes of each channel
    soft = df[df['energy'] == '0.1-0.8nm'].tail(60)
    hard = df[df['energy'] == '0.05-0.4nm'].tail(60)

    min_len = min(len(soft), len(hard), 60)
    if min_len < 10:
        raise ValueError("Not enough live data available")

    soft_vals = soft['flux'].values[-min_len:]
    hard_vals = hard['flux'].values[-min_len:]

    # Interpolate to exactly 60 points
    soft_60 = np.interp(np.linspace(0, 1, 60),
                        np.linspace(0, 1, len(soft_vals)), soft_vals)
    hard_60 = np.interp(np.linspace(0, 1, 60),
                        np.linspace(0, 1, len(hard_vals)), hard_vals)

    window = np.stack([soft_60, hard_60], axis=1)

    # Current flux values for response
    current_soft = float(soft['flux'].iloc[-1])
    current_hard = float(hard['flux'].iloc[-1])
    latest_time  = str(soft['time_tag'].iloc[-1])

    return window, current_soft, current_hard, latest_time

# ── Input Schema ──────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    window: List[List[float]]  # Shape: (60, 2)

    class Config:
        json_schema_extra = {
            "example": {
                "window": [[8.5e-7, 1.2e-8]] * 60
            }
        }

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":        "KARN Solar Flare Forecasting API",
        "version":     "1.0.0",
        "description": "Real-time solar flare prediction using Aditya-L1 inspired ML pipeline",
        "endpoints": {
            "health":  "/health",
            "predict": "/predict  [POST]",
            "live":    "/live"
        }
    }

@app.get("/health")
def health():
    return {
        "status":     "ok",
        "model":      "karn_lstm_v2",
        "classes":    CLASS_NAMES,
        "input_shape": "(60, 2)",
        "timestamp":  datetime.utcnow().isoformat() + "Z"
    }

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Takes a 60x2 window of solar X-ray flux values.
    Returns flare class prediction, probabilities, and alert status.
    """
    try:
        window_array = np.array(request.window, dtype=np.float32)
        
        if window_array.shape != (60, 2):
            raise HTTPException(
                status_code=422,
                detail=f"Window must be shape (60, 2), got {window_array.shape}"
            )

        result = run_inference(window_array)
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live")
def live_prediction():
    """
    Fetches real-time GOES-16 X-ray data and returns current solar weather prediction.
    No input needed — KARN fetches live data automatically.
    """
    try:
        window, soft, hard, latest_time = fetch_live_goes()
        result = run_inference(window)

        # Add live data info to response
        result["live_data"] = {
            "latest_observation": latest_time,
            "soft_xray_flux":     f"{soft:.2e} W/m²",
            "hard_xray_flux":     f"{hard:.2e} W/m²",
            "data_source":        "NOAA GOES-16"
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))