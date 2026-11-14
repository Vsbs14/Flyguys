# Benchmark: one batch of decisions

Device: NVIDIA GeForce RTX 5060 Laptop GPU (cuda). Each brain runs 200 ms from rest with random 0-200 Hz input on every sense.

| Brains | Wall time (s) | ms per step | Decisions per second | Peak GPU memory (GB) |
|---|---|---|---|---|
| 64 | 4.17 | 2.08 | 15.4 | 0.19 |
| 256 | 10.84 | 5.42 | 23.6 | 0.42 |
| 1024 | 37.83 | 18.91 | 27.1 | 1.32 |
