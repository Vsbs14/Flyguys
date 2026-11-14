# Sanity checks: intact brain

Device: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel (cpu). Seed 0, 6 replicate brains per condition, stimulus on for 200 ms.

| # | Check | Pass criterion | Measured | Result |
|---|-------|----------------|----------|--------|
| 1 | No input -> silence | 0 spikes in the whole brain | 0 spikes | PASS |
| 2 | Sugar (200 Hz) -> MN9 fires at about 50 Hz | MN9 rate within 25-125 Hz | MN9 = 66.7 Hz | PASS |
| 3 | Sugar + bitter -> MN9 drops a lot | MN9 below 50% of its sugar-only rate | MN9 = 0.4 Hz (1% of sugar-only) | PASS |
| 4 | Looming (150 Hz) -> Giant Fibre fires | Giant Fibre spikes in every replicate | spiked in 6/6, first spike at 4.8 ms | PASS |
| 5 | Giant Fibre silenced -> no escape | 0 Giant Fibre spikes under the same looming input | spiked in 0/6 | PASS |
| 6 | Input stopped -> activity dies out | 0 spikes in the whole brain 200-300 ms after input ends | 0 spikes | PASS |

Output firing rates (Hz, mean over the neurons of each type and over replicates):

| Condition | feed | escape | retreat | advance | Giant Fibre spiked |
|---|---|---|---|---|---|
| none | 0.0 | 0.0 | 0.0 | 0.0 | 0/6 |
| sugar | 66.7 | 0.0 | 0.0 | 0.0 | 0/6 |
| sugar+bitter | 0.4 | 0.0 | 0.0 | 0.0 | 0/6 |
| looming | 0.0 | 183.8 | 11.2 | 0.0 | 6/6 |
| looming, GF off | 0.0 | 0.0 | 10.2 | 0.0 | 0/6 |
