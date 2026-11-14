# Sanity checks: hungry brain

Device:  (cpu). Seed 0, 3 replicate brains per condition, stimulus on for 200 ms.

| # | Check | Pass criterion | Measured | Result |
|---|-------|----------------|----------|--------|
| 1 | No input -> silence | 0 spikes in the whole brain | 0 spikes | PASS |
| 2 | Sugar (200 Hz) -> MN9 fires at about 50 Hz | MN9 rate within 25-125 Hz | MN9 = 69.2 Hz | PASS |
| 3 | Sugar + bitter -> MN9 drops a lot | MN9 below 50% of its sugar-only rate | MN9 = 68.3 Hz (99% of sugar-only) | FAIL |
| 4 | Looming (150 Hz) -> Giant Fibre fires | Giant Fibre spikes in every replicate | spiked in 3/3, first spike at 4.7 ms | PASS |
| 5 | Giant Fibre silenced -> no escape | 0 Giant Fibre spikes under the same looming input | spiked in 0/3 | PASS |
| 6 | Input stopped -> activity dies out | 0 spikes in the whole brain 200-300 ms after input ends | 0 spikes | PASS |

Output firing rates (Hz, mean over the neurons of each type and over replicates):

| Condition | feed | escape | retreat | advance | Giant Fibre spiked |
|---|---|---|---|---|---|
| none | 0.0 | 0.0 | 0.0 | 0.0 | 0/3 |
| sugar | 69.2 | 0.0 | 0.0 | 0.0 | 0/3 |
| sugar+bitter | 68.3 | 0.0 | 0.0 | 0.0 | 0/3 |
| looming | 0.0 | 185.0 | 6.7 | 0.0 | 3/3 |
| looming, GF off | 0.0 | 0.0 | 8.3 | 0.0 | 0/3 |
