# Response table check: live brain vs cached table

100 random inputs were run once each on the live full brain (Intel64 Family 6 Model 186 Stepping 3, GenuineIntel) and compared with the intact table's interpolation. One live run is noisy, so the error should be read against the trial-to-trial spread stored in the table.

| Output | Live mean (Hz) | Table mean (Hz) | Mean abs. error (Hz) | Bias (Hz) | Trial-to-trial SD (Hz) | Correlation |
|---|---|---|---|---|---|---|
| MN9 (feed) | 4.47 | 4.98 | 1.18 | -0.51 | 1.76 | 0.978 |
| Giant Fibre (escape) | 59.65 | 59.32 | 2.63 | +0.33 | 2.34 | 0.999 |
| MDN (retreat) | 2.34 | 2.27 | 0.66 | +0.06 | 0.64 | 0.923 |
| DNp09 (advance) | 0.03 | 0.12 | 0.14 | -0.09 | 0.15 | 0.069 |

Giant Fibre spike: the live brain spiked in 69.0% of runs; the table predicted 65.9%. The table's call (probability at least 0.5) matched the live brain in 94.0% of runs. Brier score 0.042.
