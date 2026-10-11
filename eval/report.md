# AEROTWIN — System Evaluation & Benchmark Report

**Generated:** 2026-10-02T16:17:46.113810+00:00  
**Evaluation Status:** ✅ PASSED (5/5 Scenarios Passed)  
**Edge Computing Compliance (< 10 ms/step):** ✅ COMPLIANT

---

## 1. Executive Summary

This report evaluates the **AEROTWIN Digital Twin Pipeline** across all operational phases for a Rotax 914-class turbocharged UAV piston engine. Benchmarks cover nominal operations, abrupt thermal and lubrication failures, incipient mechanical wear, and sensor anomaly isolation.

| Scenario | Objective | Key Metric | Target | Measured | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Healthy Baseline** | False Alarm Rate & Latency | FAR & Latency | FAR ≤ 5.0%, < 10 ms | FAR = 0.1%, 2.01 ms | ✅ PASS |
| **Coolant Loss** | Abrupt Overheating Detection | Time-to-Detect (TTD) | TTD ≤ 25.0s | 12.0s (Diagnosed: True) | ✅ PASS |
| **Oil Starvation** | Rapid Pressure Loss & Safety | TTD & Safe Decision | TTD ≤ 15.0s, RTB/Land | 1.0s (Action: Safe) | ✅ PASS |
| **Bearing Wear** | Incipient Mechanical Degradation | Early Warning Detection | Early Warning = True | Detected = True | ✅ PASS |
| **Sensor Fault** | Sensor vs Engine Isolation | Avoid False Shutdown | False Shutdown = False | Avoided = True | ✅ PASS |

---

## 2. Benchmark Breakdown

### 2.1 Healthy Baseline & Latency Benchmark
- **Total Flight Steps:** 3420 seconds
- **False Alarms:** 3
- **False Alarm Rate (FAR):** 0.09% (Target: ≤ 5.0%)
- **Mean Pipeline Latency:** 2.010 ms / step
- **95th Percentile Latency:** 3.279 ms / step
- **Throughput:** 497.6 steps/second (Real-time capability: > 500x speedup)

### 2.2 Abrupt Thermal Failure (Coolant Loss)
- **Fault Onset:** t = 30.0s
- **Detection Triggered:** t = 42.0s
- **Time-to-Detect (TTD):** 12.0 seconds (Target: ≤ 25.0s)
- **Primary Diagnosis:** Coolant Loss with high confidence
- **Physics Effect:** CHT and coolant temperature residuals exceed warning threshold, triggering power reduction recommendation.

### 2.3 Catastrophic Lubrication Failure (Oil Starvation)
- **Fault Onset:** t = 40.0s
- **Detection Triggered:** t = 41.0s
- **Time-to-Detect (TTD):** 1.0 seconds (Target: ≤ 15.0s)
- **Mission Action:** Triggered emergency/RTB action before thermodynamic seizure limit.

### 2.4 Incipient Wear (Bearing Wear)
- **Profile:** Gradual friction increase and vibration RMS rise over 200s.
- **Result:** EKF tracked latent friction factor ($hp_{friction}$), triggering EWMA/CUSUM warnings prior to severe vibration excursion.

### 2.5 Sensor Anomaly Isolation
- **Profile:** Simulated abrupt bias on oil temperature thermocouple (+50 °C).
- **Isolation Result:** Single-channel anomaly successfully quarantined. Unrelated engine systems continued nominal flight without false emergency abort.

---

## 3. Operational Safety Notice

> **⚠️ SAFETY NOTICE:** AEROTWIN is an advisory decision-support system and digital twin prototype developed for research and evaluation purposes. All recommendations (CONTINUE / DIVERT / RTB / EMERGENCY_LAND) must be corroborated with flight manuals and operational procedures before executing in real flight.
