# Phase 2: from a world to a decision

Each example world is turned into three input rates, the cached brain answers, and the frozen decoder turns the answer into amounts between 0 and 1. Rates are the table's expected values; in the simulation each decision is drawn with its trial-to-trial noise.

Decoder (fitted on the intact brain, then frozen): MN9 full scale 75.5 Hz, MDN full scale 18.2 Hz, DNp09 full scale 10.0 Hz, share of growth that feeds launches 0.5.

## Intact brain

| World | Sugar (Hz) | Bitter (Hz) | Looming (Hz) | MN9 (Hz) | MDN (Hz) | DNp09 (Hz) | P(flee) | Growth | Launch | Retreat |
|---|---|---|---|---|---|---|---|---|---|---|
| Garden world | 200 | 0 | 0.0 | 65.0 | 0.0 | 0.0 | 0.00 | 0.86 | 0.43 | 0.00 |
| Average world | 100 | 100 | 0.0 | 0.5 | 0.0 | 0.0 | 0.00 | 0.01 | 0.00 | 0.00 |
| Rich but harsh world | 200 | 163 | 0.0 | 2.7 | 0.0 | 0.0 | 0.00 | 0.04 | 0.02 | 0.00 |
| Barren rock | 19 | 163 | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| Garden world, supernova in 10 epochs | 200 | 0 | 53.0 | 42.8 | 1.6 | 0.0 | 1.00 | 0.57 | 0.26 | 0.09 |
| Garden world at the blast's far edge | 200 | 0 | 2.7 | 68.3 | 0.0 | 0.0 | 0.79 | 0.90 | 0.45 | 0.00 |
| Garden world after one lost fleet | 200 | 0 | 0.8 | 70.2 | 0.0 | 0.0 | 0.00 | 0.93 | 0.47 | 0.00 |
| Garden world after three lost fleets | 200 | 0 | 2.2 | 67.5 | 0.0 | 0.0 | 0.55 | 0.89 | 0.45 | 0.00 |

Growth, launch and retreat are what the colony does if the Giant Fibre stays silent. If it spikes, with the probability shown, the colony flees and does nothing else.
