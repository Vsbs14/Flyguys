# Benchmark: one batch of decisions

Device: Tesla T4 (cuda). Each brain runs 200 ms from rest with random 0-200 Hz sugar, bitter and looming input.

| Brains | Wall time (s) | ms per step | Decisions per second | Peak GPU memory (GB) |
|---|---|---|---|---|
| 64 | 6.03 | 3.02 | 10.6 | 0.19 |
| 256 | 15.68 | 7.84 | 16.3 | 0.42 |
| 1024 | 56.04 | 28.02 | 18.3 | 1.32 |
