# AEROTWIN — Assumptions Log
<!-- Every engineering assumption is recorded here with a rationale. -->

## Process
- When requirements are ambiguous, choose the most defensible engineering default.
- Document here, continue implementation, do not stall.
- During HAL/DRDO calibration, replace assumed values with measured values.

---

## Engine Model

| # | Parameter | Assumed Value | Rationale | Replace With |
|---|-----------|--------------|-----------|-------------|
| 1 | Compression ratio | 9.0 | Typical for SI aero engines | HAL/DRDO datasheet |
| 2 | Idle RPM | 1400 RPM | Rotax 914 datasheet vicinity | Test-bench measurement |
| 3 | FMEP coefficients | a=0.8 bar, b=0.005 bar/RPM | Heywood engine modeling text | Motoring test |
| 4 | CHT time constant | 30 s | Lumped thermal RC, typical value | IR measurement |
| 5 | EGT time constant | 5 s | Fast-response thermocouple assumption | Thermocouple calibration |
| 6 | Oil time constant | 60 s | Slow thermal mass | Oil cooler test |
| 7 | VE at peak | 0.85 | Naturally aspirated reference | Flow bench |
| 8 | VE RPM peak | 3500 RPM | Mid-range for 4-cyl SI | Dyno sweep |
| 9 | Propeller diameter | 1.73 m | Typical ~100 hp UAV prop | Manufacturer spec |
| 10 | Propeller Cp ref | 0.045 | Standard NACA estimate | Wind tunnel |
| 11 | Stoich AFR | 15.0 | Gasoline (close to AVGAS) | Emissions test |
| 12 | Lambda cruise | 1.0 | Stoichiometric cruise | EGT lambda meter |
| 13 | Glide ratio | 8.0 (aircraft) | Conservative MALE estimate | Flight test |

---

## Sensor Models

| # | Parameter | Assumed Value | Rationale |
|---|-----------|--------------|-----------|
| 1 | RPM noise | ±20 RPM 1-sigma | Encoder encoder resolution estimate |
| 2 | EGT noise | ±5 °C 1-sigma | K-type thermocouple specification |
| 3 | CHT noise | ±2 °C 1-sigma | J-type thermocouple |
| 4 | Dropout probability | 0.001–0.003/sample | Typical UAV CANBUS fault rate |

---

## Fault Model

| # | Assumption | Rationale |
|---|-----------|-----------|
| 1 | Faults are independent | Simplification; real faults can cascade |
| 2 | Severity ramps are linear unless specified | Conservative default |
| 3 | Bearing wear leads to vibration increase ∝ severity^2 | Empirical literature |
| 4 | Thermostat stuck-open → coolant temp drops by 20 °C | Physics approximation |

---

## Mission Reliability

| # | Assumption | Rationale |
|---|-----------|-----------|
| 1 | Monte Carlo with 1000 samples | Sufficient for demo; increase for production |
| 2 | RUL uncertainty modelled as Gaussian | Simplification; real is often lognormal |
| 3 | Mission completion probability threshold 0.85 | Industry convention for non-critical UAV ops |

---

## Dataset Generation

| # | Assumption | Rationale |
|---|-----------|-----------|
| 1 | 500 simulated missions default | Large enough for ML demo, fast enough on laptop |
| 2 | Split is 70/15/15 train/val/test by mission | Standard; never by row to avoid data leakage |
| 3 | Fault onset time is uniformly random across mission | Realistic variation |

---

*Last updated: Phase 0 initialisation*
