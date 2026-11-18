# Channel calibration from an external flight dataset

`propagation_fit.py` fits the log-distance model

    P(d) = P(d0) - 10 n log10(d/d0) + X,  X ~ N(0, sigma^2)

to signal levels that one fixed sensor measured during drone flights with GPS logs. It estimates the three channel parameters the AeroFence simulator uses:

- the path-loss exponent `n_pl`;
- the slow shadowing spread `shadow_db`;
- its decorrelation distance `shadow_dcorr`.

All three come with block-bootstrap 95% intervals. The absolute level P(d0) depends on the transmitter, band, antennas and SNR reference, so it is reported but not transferred.

## Steps

1. Check the dataset first. You need:
   - the sensor's coordinates;
   - SNR or RSS that is consistent over time;
   - a way to keep only the aircraft's own signal (not the controller's);
   - a licence that allows academic analysis.
2. Run the fit, giving the column names of your files:

       python3 propagation_fit.py --sensor sensor.csv --sensor-time utc --sensor-value snr \
           --sensor-filter "source=='aircraft'" --sensor-track target \
           --flight ua1.csv ua2.csv ua3.csv --flight-track-name UA1 UA2 UA3 \
           --flight-time time_ms --flight-lat lat --flight-lon lon --flight-alt alt \
           --sensor-pos LAT,LON,ALT --out ../results_channel

   Times can be epoch seconds, epoch milliseconds or date-time strings in UTC. Use `--time-offset` if the two clocks differ.
3. Rerun the simulation with the fitted channel. Results go to a separate folder.
   - (a) Nominal weights and thresholds under the measured channel:

         mkdir -p results_measured_fixed
         cp results/learn.json results/combo.json results_measured_fixed/
         AF_CHANNEL=results_channel/channel_measured.json AF_TAG=measured_fixed \
             python3 experiments.py kpi compare roc

   - (b) Refit weights and thresholds on the measured channel:

         AF_CHANNEL=results_channel/channel_measured.json AF_TAG=measured_refit \
             python3 experiments.py learn roc kpi compare combo

## Outputs

| File | Contents |
|---|---|
| `channel_fit.json` | Fit, intervals and the decorrelation fit |
| `channel_measured.json` | Simulator parameters |
| `channel_fit.pdf` | Plot of the data and the fit |
| `synchronized_samples.csv` | Every matched sample with its distance |

## Method notes

- **Weak signals.** Signals below the sensor floor are never logged, so ordinary least squares underestimates n. The script uses a truncated-normal maximum-likelihood fit.
- **Decorrelation distance.** The residual correlation is fitted against 3-D distance travelled, only on the range where truncation is negligible. In the bootstrap, pairs are formed only within one drawn block, so independent blocks are never treated as neighbours. Correlation is observed only up to 400 m, so the estimate is capped at 800 m. `d_corr_upper_bound_identified` is false when the data cannot bound it from above.
- **Intervals.** They resample 120-s blocks within each flight.
- **Sensitivity.** `channel_fit.json` also reports the fit with the floor raised by 3 and 6 dB, with minimum distances of 50 and 100 m, and with three margins for the decorrelation range. If these move n by more than its interval, the fitted model is not adequate for the data, and that has to be reported.

## Self-test on synthetic data

Run it with:

    python3 propagation_fit.py --selftest

It uses 20 synthetic data sets, each with three 15-min flights. The true values are n = 2.5, 5 dB shadowing and an 80 m decorrelation distance.

| Quantity | Mean estimate | Spread (sd) |
|---|---|---|
| n | 2.48 | 0.19 |
| Shadowing | 4.5 dB | |
| Decorrelation distance | 75 m | 25 m |

In a separate check on 15 data sets, the nominal 95% intervals covered the true value in 80% of cases for n and 87% for the decorrelation distance. Several decorrelation intervals had no usable upper bound. With only a few flights, report the intervals as approximate. The decorrelation distance is the least certain of the three parameters.

The self-test numbers are SYNTHETIC. Never report them as measurements.
