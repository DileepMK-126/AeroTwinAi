# AEROTWIN — AI-Enabled Real-Time Digital Twin for Aero Piston Engines

[![CI](https://github.com/aerotwin/aerotwin/actions/workflows/ci.yml/badge.svg)](https://github.com/aerotwin/aerotwin/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **⚠️ Safety Statement:** AEROTWIN is a research prototype and decision-support tool. It is **not** flight-certified software and has **not** been validated to aviation standards (DO-178C, DO-160). All fault detection, prognosis, and mission recommendations are **advisory only**. Do not use for real flight decisions. Every prediction shown in the UI is clearly labelled with its provenance (simulated / real / replay).

---

## What is AEROTWIN?

AEROTWIN is an AI-enabled, real-time digital twin for health monitoring, fault prediction, and mission-reliability enhancement of aero piston engines used in MALE (Medium Altitude Long Endurance) UAVs. Built for SIH problem statement **SIH26054**.

**Key features:**
- 🔬 Physics-based mean-value engine model (Rotax 914-class)
- 🔄 Real-time state estimation (EKF) fusing noisy sensor data with physics
- ⚠️ Layered fault detection (residuals → statistical → ML)
- 🩺 Explainable fault diagnosis with natural-language alerts
- ⏱️ Remaining Useful Life (RUL) prediction with uncertainty
- 🎯 Mission reliability engine with actionable go/no-go decisions
- 📡 MQTT, MAVLink, serial, UDP, CSV/Parquet replay adapters
- 🖥️ Striking real-time ops-centre dashboard (Next.js)
- 🏁 Fully functional without trained ML models; upgrades seamlessly when ONNX models are added

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/aerotwin/aerotwin.git
cd aerotwin
make setup

# 2. Start all services
docker compose up -d

# 3. Run tests
make test

# 4. Run a simulated mission
make sim

# 5. Evaluate
make eval

# 6. Open dashboard
open http://localhost:3000
```

**Requirements:** Python 3.11, Docker + Docker Compose, Node.js 20+.

---

## Architecture

```
Telemetry (MQTT / MAVLink / Serial / CSV Replay)
    │
    ▼
Ingest Layer  ──►  Unified Schema (Pydantic)
    │
    ▼
Physics Twin  ──►  State Estimation (EKF)  ──►  Health Parameters
    │
    ▼
Residual Engine  ──►  Detection (Level A: thresholds/EWMA/CUSUM)
                  ──►  Detection (Level B: IsolationForest / Autoencoder)
    │
    ▼
Diagnosis  ──►  Explainability (SHAP / natural-language)
    │
    ▼
Prognosis  ──►  RUL with uncertainty
    │
    ▼
Mission Reliability  ──►  GO / CONTINUE_DEGRADED / DIVERT / RTB / EMERGENCY
    │
    ▼
WebSocket Hub  ──►  Live Dashboard (Next.js)
```

---

## Repository Layout

```
aerotwin/
  README.md            ← this file
  Makefile             ← make setup | dev | test | sim | eval
  docker-compose.yml
  .env.example
  configs/             ← all engine, sensor, fault, mission, model configs
  core/
    physics/           ← engine model, atmosphere, turbocharger, thermal
    faults/            ← fault injection library
    sensors/           ← sensor noise, drift, dropout, quantisation
    simulator/         ← closed-loop simulator + flight-profile generator
    twin/              ← state estimation (EKF/UKF) + health-param ID
    detection/         ← residual, EWMA/CUSUM, isolation forest, autoencoder
    diagnosis/         ← fault classifier + explainability
    prognosis/         ← RUL with uncertainty
    mission/           ← reliability + decision engine
    registry/          ← model registry + ONNX loader + fallbacks
  data/
    adapters/          ← cmapss, alfa, generic_csv, mavlink_log
    schemas.py         ← unified telemetry schema (Pydantic)
  ingest/              ← MQTT, MAVLink, UDP, serial, replay, HTTP
  api/                 ← FastAPI app, WebSocket hub, REST routes
  edge/                ← edge runtime, ONNX inference, latency bench
  training/
    notebooks/         ← Colab-ready training notebooks
    scripts/
  eval/                ← metrics, benchmarks, reports
  web/                 ← Next.js dashboard
  tests/               ← unit + integration tests
  docs/                ← architecture, physics, api, runbook
```

---

## Data Provenance

Every telemetry record carries a `source` field: `simulated` | `real_public` | `real_user` | `replay`. The dashboard permanently shows this badge. No result is ever hardcoded — all metrics come from re-runnable scripts.

---

## Adding Real Data

1. Create a channel-mapping YAML (see `configs/channel_mapping_example.yaml`).
2. Point the replay adapter at your CSV/Parquet file.
3. Start the backend — real data flows through automatically with no code changes.

See `docs/runbook.md` for full instructions.

---

## Training Models (later)

See `training/README_TRAINING.md`. The system runs fully on physics + statistical detectors today. Trained ONNX models are optional upgrades — drop them into the registry folder and restart.

---

## Limitations

- All default data is **simulated** using a Rotax 914-like physics model.
- No real MALE-UAV piston-engine telemetry dataset exists publicly; real data must be provided by the user.
- Engine parameters are from public datasheets and assumptions; HAL/DRDO test-bench calibration is required for operational use.
- Models are small and CPU-optimised for a 16 GB laptop; production deployment may need retraining on more data.

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built for Smart India Hackathon 2026 — Problem Statement SIH26054.*
