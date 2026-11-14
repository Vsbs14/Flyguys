# Benchmark: one batch of decisions

Device: Tesla T4 (cuda). Each brain runs 200 ms from rest with random 0-200 Hz sugar, bitter and looming input.

| Brains | Wall time (s) | ms per step | Decisions per second | Peak GPU memory (GB) |
|---|---|---|---|---|
| 64 | 5.85 | 2.92 | 10.9 | 0.19 |
| 256 | 15.77 | 7.88 | 16.2 | 0.42 |
| 1024 | 55.75 | 27.88 | 18.4 | 1.32 |
