import torch
import torch.nn as nn
import numpy as np
import requests
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KARN — Solar Flare Forecasting API",
    description="Real-time solar flare nowcasting and forecasting using GOES-16 X-ray data",
    version="1.0.0"
)

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

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading KARN LSTM model...")
model = FlareRecurrentLSTM()
model.load_state_dict(torch.load(
    "karn_lstm_v2.pth",
    map_location="cpu",
    weights_only=True
))
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
    if window_array.shape != (60, 2):
        raise ValueError(f"Expected shape (60, 2), got {window_array.shape}")

    if np.any(np.isnan(window_array)) or np.any(np.isinf(window_array)):
        raise ValueError("Input contains NaN or infinite values")

    soft_norm  = log_normalize(window_array[:, 0])
    hard_norm  = log_normalize(window_array[:, 1])
    normalized = np.stack([soft_norm, hard_norm], axis=1).astype(np.float32)

    tensor = torch.tensor(normalized).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_class = int(np.argmax(probs))
    score      = compute_severity(probs)
    level, msg = compute_alert(probs, score)

    return {
        "predicted_class": CLASS_NAMES[pred_class],
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
    url      = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"
    response = requests.get(url, timeout=30)
    raw      = response.text.strip().replace('][', ',')
    data     = json.loads(raw)
    df       = pd.DataFrame(data)

    df['time_tag'] = pd.to_datetime(df['time_tag'])
    df['flux']     = pd.to_numeric(df['flux'], errors='coerce')
    df             = df.dropna(subset=['flux'])
    df             = df.sort_values('time_tag')

    soft = df[df['energy'] == '0.1-0.8nm'].tail(60)
    hard = df[df['energy'] == '0.05-0.4nm'].tail(60)

    min_len = min(len(soft), len(hard), 60)
    if min_len < 10:
        raise ValueError("Not enough live data available")

    soft_vals = soft['flux'].values[-min_len:]
    hard_vals = hard['flux'].values[-min_len:]

    soft_60 = np.interp(
        np.linspace(0, 1, 60),
        np.linspace(0, 1, len(soft_vals)),
        soft_vals
    )
    hard_60 = np.interp(
        np.linspace(0, 1, 60),
        np.linspace(0, 1, len(hard_vals)),
        hard_vals
    )

    window       = np.stack([soft_60, hard_60], axis=1)
    current_soft = float(soft['flux'].iloc[-1])
    current_hard = float(hard['flux'].iloc[-1])
    latest_time  = str(soft['time_tag'].iloc[-1])

    return window, current_soft, current_hard, latest_time

# ── THIS MUST BE AT TOP LEVEL — NOT INSIDE fetch_live_goes ───────────────────
def save_prediction_to_db(result: dict, soft_flux: float, hard_flux: float):
    """Save prediction to Supabase — runs in background."""
    try:
        row = {
            "timestamp":       result["timestamp"],
            "predicted_class": result["predicted_class"],
            "quiet_prob":      result["probabilities"]["quiet"],
            "c_prob":          result["probabilities"]["c_class"],
            "m_prob":          result["probabilities"]["m_class"],
            "x_prob":          result["probabilities"]["x_class"],
            "severity_score":  result["severity_score"],
            "alert_level":     result["alert_level"],
            "alert_message":   result["alert_message"],
            "soft_xray_flux":  soft_flux,
            "hard_xray_flux":  hard_flux,
            "data_source":     "NOAA GOES-16"
        }
        supabase.table("predictions").insert(row).execute()
        print(f"Saved to Supabase: {result['alert_level']} — {result['predicted_class']}")

        # If flare — also save to flare_events
        if result["alert_level"] in ["WARNING", "CRITICAL", "ADVISORY"]:
            pred_id = supabase.table("predictions")\
                .select("id")\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()

            if pred_id.data:
                flare_row = {
                    "prediction_id": pred_id.data[0]["id"],
                    "detected_at":   result["timestamp"],
                    "flare_class":   result["predicted_class"],
                    "peak_prob":     max(
                        result["probabilities"]["c_class"],
                        result["probabilities"]["m_class"],
                        result["probabilities"]["x_class"]
                    ),
                    "severity_score": result["severity_score"],
                    "alert_level":    result["alert_level"],
                }
                supabase.table("flare_events").insert(flare_row).execute()
                print(f"Flare event saved to flare_events table")

    except Exception as e:
        print(f"Supabase save failed: {e}")

# ── Input Schema ──────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    window: List[List[float]]

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
            "predict": "/predict [POST]",
            "live":    "/live"
        }
    }

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model":       "karn_lstm_v2",
        "classes":     CLASS_NAMES,
        "input_shape": "(60, 2)",
        "timestamp":   datetime.utcnow().isoformat() + "Z"
    }

@app.post("/predict")
def predict(request: PredictRequest):
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
def live_prediction(background_tasks: BackgroundTasks):
    try:
        window, soft, hard, latest_time = fetch_live_goes()
        result = run_inference(window)

        result["live_data"] = {
            "latest_observation": latest_time,
            "soft_xray_flux":     f"{soft:.2e} W/m²",
            "hard_xray_flux":     f"{hard:.2e} W/m²",
            "data_source":        "NOAA GOES-16"
        }

        # Save in background — API responds instantly
        background_tasks.add_task(
            save_prediction_to_db, result, soft, hard
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))