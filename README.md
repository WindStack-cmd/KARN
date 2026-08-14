# 🔱 Project KARN (v2.4)

### Deep Learning Telemetry Control Panel & Aditya-L1 Solar Flare Simulator

Project KARN is an end-to-end, production-grade space weather monitoring system designed to parse live solar X-ray telemetry streams, predict impending solar flares using a deep memory architecture, and persist state arrays dynamically to a synchronized cloud telemetry dashboard. 

Optimized to process multi-channel inputs resembling the payload specs of India's **Aditya-L1 (SoLEXS & HEL1OS)** instruments, KARN accomplishes real-time streaming inference, probability calibration, and sub-20ms system telemetry visualization.

---

## 📊 Live Operational Dashboard Preview
*   **Mission Control Center Dashboard:** A responsive, deep space-themed command grid tracking rolling data light curves seamlessly.
*   **Performance Metrics:** Implements localized data window guards, multi-class distribution matrix meters, and continuous WebSockets synchronization hooks.

---

## ⚡ Key Architectural Features

### 🌌 1. Algorithmic Core & Deep Learning Engine
*   **Context-Aware Weighted Gate (Pre-Stage 1 Anomaly Filter):** Programmatic rule-based intercept layer designed to catch instrumentation dropouts and drop high-frequency cosmic-ray particle spikes before sequence evaluation.
*   **Deep Multi-Layer Recurrent LSTM Engine:** 2-layer sequential Long Short-Term Memory network utilizing 128 hidden dimensions to parse deep temporal patterns and slow, hours-long gradual solar storm developments.
*   **Asymmetric Operational Thresholds Matrix:** Customized decision bounds enforced to minimize catastrophic false negatives where missing highly destructive events is heavily penalized:
    *   **X-Class Flares (Severe Space Weather Threat):** Sensitive trigger alert at $\le 0.3$ probability.
    *   **M-Class Flares (Moderate Structural Risk):** Standard trigger alert calibrated at $\le 0.5$ probability.
*   **Consequence-Weighted Threat Scoring Formula:** Translates continuous raw probability arrays directly into a single, high-fidelity threat index:
    $$\text{Severity Risk Score} = 1 \times P(C) + 10 \times P(M) + 100 \times P(X)$$

### 🛠️ 2. High-Performance Backend Architecture
*   **FastAPI Ingestion Engine:** Microservice architecture loading the model binary once at boot time, managing endpoints efficiently (`/health`, `/predict`, `/live`).
*   **Automated Background Ingestion Worker:** Hits the public NOAA GOES-16 satellite data stream every 60s, executes rolling multi-channel log-rescaling, maps arrays, and processes inference asynchronously to preserve low latency.
*   **Supabase PostgreSQL Persistent Storage:** Relational database setup utilizing standard data validation rules and indexed timestamp structures to log pipeline tracking parameters cleanly.

### 💻 3. Real-Time Telemetry Frontend Dashboard
*   **Bidirectional WebSockets Integration:** Subscribes to database hooks to push updates to the UI immediately upon entry creation, bypassing standard HTTP page refreshes.
*   **Responsive Logarithmic Charting Visuals:** Renders interactive, multi-line graphs via Recharts, mapping SoLEXS soft tracks and HEL1OS hard tracks against standard solar class parameters securely.
*   **Framer-Motion Hazard Systems:** Triggers pulsing, animated warning banners across the layout header during active critical event changes.

---

## 🛠️ System Data Schema & Pipeline Flow

> **End-to-End Data Lifecycle Topology**
> 
> ```
> [ NOAA GOES-16 Satellite API ] ──► (Real-Time Ingestion Loop)
>                                              │
>                                              ▼
> [ FastAPI Microservice Engine ] ──► [ /live Async Telemetry Endpoint ]
>                                              │
>                                              ▼
> [ Pre-Stage 1 Telemetry Gate ] ──► (Filters Noise Spikes & NaNs)
>                                              │
>                                              ▼
> [ Core PyTorch LSTM Architecture ] ──► (Inference Execution & Logit Softmax)
>                                              │
>                                              ▼
> [ Cloud Supabase DB Cluster ] ──► [ Automated PostgreSQL Table Insert ]
>                                              │
>                                              ▼
> [ React Dashboard WebSockets ] ──► (Zero-Delay Instant UI State Re-Render)
> ```

### 📊 Step-by-Step Production Pipeline Execution
*   **Ingestion:** The background worker requests live flux readings from the GOES-16 satellite endpoint every 60 seconds, handling network failures gracefully.
*   **Preprocessing:** The incoming stream passes through your custom `gate.py` pipeline, isolating solar energetic events from cosmic ray noise hits.
*   **Sequential Inference:** Valid arrays are structured into `(1, 60, 2)` tracking matrices and evaluated via your deep LSTM cell memory layers.
*   **Probability Calibration:** Uncalibrated model logits are smoothed across your custom multi-class distribution matrix to calculate the risk score metrics.
*   **Persistence & Synchronization:** The backend writes the float output metrics to your cloud instance. Supabase intercepts this insert block and pushes a WebSocket data package to your active React application, instantly painting the telemetry lines on screen.

---

## 📊 Model Performance Matrix

Following extensive hyperparameter testing, model overfitting was successfully resolved by expanding training data profiles to **1,884 explicit historical records pulled via NASA's DONKI API**, resulting in top-tier classification safety bounds:

| Target Solar Class | Labeled Historical Events Caught | System Validation Recall Benchmark |
| :--- | :--- | :--- |
| **C-Class (Minor)** | 96 / 96 | **100.0% Recall** |
| **M-Class (Moderate)** | 318 / 320 | **99.4% Recall** |
| **X-Class (Severe)** | 21 / 21 | **100.0% Recall** |
| **Overall Model Accuracy** | — | **99.72%** |

---

## 📁 Repository Directory Structure

```text
KARN/
├── data/
│   ├── raw/                 # Contains raw downloaded GOES flux arrays
│   └── processed/           # Houses full_catalog_v2.csv training logs
├── ml/
│   ├── venv/                # Isolated Python library environment configuration
│   ├── main.py              # FastAPI Engine & Background Tasks Queue
│   ├── gate.py              # Pre-inference context anomaly filter
│   ├── dataset_v2.py        # 4-Class tensor processing pipeline
│   ├── train_v2.py          # Deep LSTM model mapping definition layers
│   ├── evaluate.py          # Metrics generation and verification script
│   ├── karn_lstm_v2.pth     # Saved production network checkpoint binary
│   └── .env                 # Secured local machine credential wrapper
└── frontend/
    ├── src/
    │   ├── components/      # Responsive Tailwind UI dashboard cards
    │   │   ├── AlertBanner.jsx
    │   │   ├── ForecastBars.jsx
    │   │   ├── MetricCards.jsx
    │   │   ├── PredictionTable.jsx
    │   │   └── XRayChart.jsx
    │   ├── lib/             # API Axios and Supabase connection wrappers
    │   │   ├── api.js
    │   │   └── supabase.js
    │   ├── App.jsx          # Front-End State Management & Websockets client Hub
    │   └── index.css        # Tailwind core directives and layout animations
    ├── tailwind.config.js   # Dark Space Palette Design Configuration
    ├── vite.config.js       # Vite 5 compilation environment parameters
    └── .env                 # Frontend cloud variable environment isolation
```

---

## 🚀 Local Installation & Deployment Guide

### Prerequisites
*   Python 3.10+
*   Node.js v18+ & npm
*   Active Supabase instance with configured predictions table schemas

### 1. Backend Ingestion Core Setup
```bash
cd KARN/ml
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\activate

pip install fastapi uvicorn pydantic requests supabase python-dotenv torch scikit-learn pandas numpy
```
Create an `ml/.env` file:
```text
SUPABASE_URL=https://supabase.co
SUPABASE_KEY=your-service-role-or-anon-key
```
Launch the backend server:
```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend Dashboard Setup
```bash
cd ../frontend
npm install
```
Create a `frontend/.env` file:
```text
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://supabase.co
VITE_SUPABASE_ANON_KEY=your-public-anon-key
```
Launch the development server:
```bash
npm run dev
```
Open your browser to `http://localhost:5173/` and explore the live mission tracking environment.

---


## 🔱 Development Roadmap Team Acknowledgement
Project KARN was built over an intense engineering timeline, scaling from basic raw API calls to fully decoupled full-stack platforms:
*   **ML & Neural Architecture:** Deep LSTM modeling, asymmetric tracking matrices, and threat calculations engineered by your name.
*   **Infrastructure Strategy Audit:** Database structure proofing, security audits, and environment parameters refined with technical review support.

---
*Developed as an engineering project tracking advanced solar physics analytics and time-series deep learning implementations.