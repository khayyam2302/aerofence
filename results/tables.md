# AeroFence: tables regenerated from results/*.json

## Table 3. Learned logistic model

| Term | beta | 95% bootstrap interval | Weight w |
|---|---|---|---|
| Intercept | -10.22 | [-12.03, -8.58] | -- |
| S_div | 1.75 | [0.41, 3.01] | 0.125 |
| S_con | 0.39 | [-0.33, 1.02] | 0.028 |
| S_mot | 5.33 | [4.30, 6.80] | 0.382 |
| S_geo | 6.49 | [5.84, 7.17] | 0.465 |

P_on = 0.185 -> C_on = 0.626, C_off = 0.526; training epochs = 295781, positive share = 0.748; equal-weight threshold at 5% = 0.703

## Model diagnostics (Section VI)

Held-out calibration: Brier 0.133 (climatology 0.193), ECE 0.049, 202923 epochs
Label 10 m: w = 0.134, 0.031, 0.508, 0.327
Label 20 m: w = 0.120, 0.029, 0.458, 0.394
Label 50 m: w = 0.125, 0.028, 0.382, 0.465
Label 100 m: w = 0.124, 0.021, 0.374, 0.481
boot_drone: beta 95% = [-12.03, -8.58], [0.41, 3.01], [-0.33, 1.02], [4.30, 6.80], [5.84, 7.17]; share with all coefficients >= 0: 0.884
boot_scene: beta 95% = [-11.99, -8.63], [0.60, 2.93], [-0.22, 0.97], [4.24, 6.68], [5.80, 7.24]; share with all coefficients >= 0: 0.903

## Table 7. Operational indicators (learned weights)

| sigma | Nr | ttff | latency | continuity | verified | escalation | spoof_detect | genuine_investigate | silent_delay |
|---|---|---|---|---|---|---|---|---|---|
| 3 m | 1 | 8.01 ± 3.89 | 5.27 ± 1.32 | 0.66 ± 0.07 | 0.45 ± 0.09 | 0.965 ± 0.052 | 0.86 ± 0.24 | 0.583 ± 0.156 | 2.39 ± 0.74 |
| 3 m | 4 | 0.83 ± 0.22 | 4.02 ± 0.24 | 0.98 ± 0.01 | 0.95 ± 0.04 | 0.439 ± 0.120 | 0.77 ± 0.24 | 0.024 ± 0.044 | 2.78 ± 0.32 |
| 3 m | 8 | 0.57 ± 0.10 | 4.05 ± 0.32 | 1.00 ± 0.01 | 0.96 ± 0.03 | 0.026 ± 0.038 | 0.77 ± 0.22 | 0.013 ± 0.035 | 2.82 ± 0.27 |
| 3 m | 12 | 0.54 ± 0.08 | 3.97 ± 0.23 | 1.00 ± 0.01 | 0.97 ± 0.03 | 0.026 ± 0.045 | 0.81 ± 0.25 | 0.012 ± 0.035 | 2.78 ± 0.30 |
| 10 m | 1 | 8.01 ± 3.89 | 5.29 ± 1.28 | 0.66 ± 0.07 | 0.45 ± 0.09 | 0.966 ± 0.052 | 0.86 ± 0.24 | 0.585 ± 0.157 | 2.44 ± 0.71 |
| 10 m | 4 | 0.83 ± 0.22 | 4.01 ± 0.29 | 0.98 ± 0.01 | 0.95 ± 0.03 | 0.443 ± 0.126 | 0.76 ± 0.24 | 0.022 ± 0.042 | 2.81 ± 0.39 |
| 10 m | 8 | 0.57 ± 0.10 | 4.06 ± 0.42 | 1.00 ± 0.01 | 0.96 ± 0.03 | 0.028 ± 0.039 | 0.77 ± 0.22 | 0.011 ± 0.031 | 2.81 ± 0.27 |
| 10 m | 12 | 0.54 ± 0.08 | 3.94 ± 0.31 | 1.00 ± 0.01 | 0.97 ± 0.03 | 0.029 ± 0.047 | 0.83 ± 0.24 | 0.013 ± 0.035 | 2.80 ± 0.32 |

## Table 8. Comparison of decision methods (Nr = 8)

| Method | latency | flips s3 | flips s10 | escalation | spoof all | smooth | jump | AUC | TPR@5% | cost us/msg |
|---|---|---|---|---|---|---|---|---|---|---|
| (a) Raw | 0.71 | 0.066 | 0.161 | 0.000 | 0.00 | 0.00 | 0.00 | -- | -- | 1.7 |
| (b) KF | 4.03 | 0.051 | 0.063 | 0.000 | 0.00 | 0.00 | 0.00 | -- | -- | 57.6 |
| (c) Innovation | 4.03 | 0.051 | 0.063 | 0.054 | 0.47 | 0.02 | 0.99 | 0.65 | 0.53 | 57.9 |
| (d) RSS check | 4.03 | 0.051 | 0.063 | 0.022 | 0.60 | 1.00 | 0.16 | 0.83 | 0.59 | 58.3 |
| (e) Equal w | 4.03 | 0.051 | 0.063 | 0.001 | 0.23 | 0.23 | 0.20 | 0.69 | 0.54 | 58.4 |
| (f) Learned w | 4.03 | 0.051 | 0.063 | 0.030 | 0.77 | 0.88 | 0.64 | 0.96 | 0.92 | 58.8 |
| (g) (f) OR (c) | 4.03 | -- | -- | 0.030 | 0.93 | 0.88 | 1.00 | -- | -- | 58.9 |

Held-out operating points (thresholds fixed on training scenes): aerofence_L: FPR 0.040, detection 0.764 (smooth 0.894, jump 0.640); innov: FPR 0.057, detection 0.529 (smooth 0.035, jump 1.000); aeroinnov: FPR 0.044, detection 0.948 (smooth 0.894, jump 1.000)
Joint calibration chosen: C_on = 0.626, innovation threshold = 112.5

## Table 10. Robustness to radio assumptions

| Condition | (f) esc | (f) det | (g) esc | (g) det | (d) esc | (d) det |
|---|---|---|---|---|---|---|
| Nominal | 0.030 | 0.77 | 0.030 | 0.93 | 0.022 | 0.60 |
| Rx gain offsets 2 dB | 0.029 | 0.80 | 0.029 | 0.93 | 0.048 | 0.62 |
| Rx gain offsets 4 dB | 0.043 | 0.84 | 0.043 | 0.95 | 0.197 | 0.69 |
| Rx gain offsets 6 dB | 0.093 | 0.89 | 0.092 | 0.95 | 0.440 | 0.80 |
| Tx pattern 3 dB | 0.030 | 0.79 | 0.031 | 0.93 | 0.033 | 0.60 |
| Tx pattern 6 dB | 0.031 | 0.82 | 0.032 | 0.94 | 0.059 | 0.63 |
| Spoofer power +/-6 dB | 0.030 | 0.77 | 0.030 | 0.93 | 0.022 | 0.60 |

## Table 11. Zone flips for boundary-skimming drones

| Nr | raw: flips / spurious, latency | KF no hysteresis | KF hysteresis + dwell |
|---|---|---|---|
| 1 | 2.16 / 1.352, 6.2 s | 1.04 / 0.276, 5.5 s | 0.60 / 0.013, 23.2 s |
| 4 | 2.34 / 1.537, 2.0 s | 1.03 / 0.267, 3.6 s | 0.59 / 0.014, 21.3 s |
| 8 | 2.47 / 1.643, 1.6 s | 1.05 / 0.257, 3.7 s | 0.59 / 0.014, 21.8 s |
| 12 | 2.49 / 1.650, 1.5 s | 1.05 / 0.250, 4.2 s | 0.59 / 0.014, 22.3 s |

## Table 13. Global sensitivity (SRRC)

| parameter | latency | escalation | spoof_detect | spurious_flip | verified | silent_delay |
|---|---|---|---|---|---|---|
| H | -0.02 | -0.29 | -0.22 | -0.07 | 0.09 | 0.00 |
| T_ver | 0.06 | 0.19 | 0.09 | -0.02 | -0.97 | 0.00 |
| T_low | -0.06 | -0.28 | -0.11 | 0.16 | 0.11 | -0.00 |
| T_silent | -0.00 | 0.04 | -0.00 | -0.03 | 0.03 | 1.00 |
| tau_p | -0.00 | -0.03 | 0.01 | 0.18 | 0.01 | -0.00 |
| tau_v | -0.07 | -0.37 | -0.08 | 0.32 | 0.19 | 0.00 |
| m_h | 0.72 | -0.02 | 0.02 | -0.57 | 0.04 | 0.01 |
| T_dwell | 0.65 | -0.06 | -0.04 | -0.37 | 0.04 | 0.01 |
| C_on | 0.06 | 0.80 | 0.99 | -0.01 | -0.46 | -0.00 |
| R^2 | 0.96 | 0.81 | 0.97 | 0.58 | 0.96 | 1.00 |
