# Decision rule: calibration and checks

Device: NVIDIA GeForce RTX 5060 Laptop GPU (cuda). Seed 0, 16 replicate brains per row, 200 ms decision window from rest. Intensity 1 = 200 Hz Poisson input on every neuron of that sense.

## Reference rates

Each output's own sense at full intensity, everything else off. An output's score is its rate divided by this reference; the action is the highest score, or `idle` if no score reaches 0.25.

| Output | Neurons | Driven by | Reference rate (Hz) |
|---|---|---|---|
| feed | MN9 | sugar | 69.1 ± 8.2 |
| escape | Giant Fibre (DNp01) | looming | 201.2 ± 3.4 |
| retreat | MDN | obstacle | 35.9 ± 3.7 |
| advance | DNp09 | target | 47.8 ± 2.8 |

## Scenario checks

Pass = the expected action wins in at least 75% of the 16 brains. Rows without an expectation are recorded only.

| Scenario | Expected | Majority | Action split | Result |
|---|---|---|---|---|
| nothing | idle | idle | idle 100% | PASS |
| sugar | feed | feed | feed 100% | PASS |
| bitter | idle | idle | idle 100% | PASS |
| looming | escape | escape | escape 100% | PASS |
| obstacle | retreat | retreat | retreat 100% | PASS |
| target | advance | advance | advance 100% | PASS |
| weak sugar (0.1) | idle | idle | idle 100% | PASS |
| sugar + bitter | idle | idle | idle 100% | PASS |
| sugar + looming | escape | escape | escape 100% | PASS |
| target + looming | escape | escape | escape 100% | PASS |
| obstacle + looming | escape | escape | escape 100% | PASS |
| sugar + target | - | advance | advance 100% | - |
| sugar + obstacle | - | feed | feed 62%, retreat 38% | - |
| target + obstacle | - | advance | advance 100% | - |
| half of everything | - | escape | escape 100% | - |

## Dose-response

One sense alone. Cells: mean score of the output that sense drives / share of brains that chose its action.

| Dose | sugar | bitter | looming | obstacle | target |
|---|---|---|---|---|---|
| 0.0 | 0.00 / 0% | majority idle | 0.00 / 0% | 0.00 / 0% | 0.00 / 0% |
| 0.1 | 0.00 / 0% | majority idle | 0.36 / 100% | 0.00 / 0% | 0.01 / 0% |
| 0.2 | 0.04 / 6% | majority idle | 0.55 / 100% | 0.00 / 0% | 0.09 / 0% |
| 0.3 | 0.16 / 38% | majority idle | 0.66 / 100% | 0.00 / 0% | 0.27 / 62% |
| 0.4 | 0.53 / 100% | majority idle | 0.74 / 100% | 0.01 / 0% | 0.41 / 100% |
| 0.5 | 0.66 / 100% | majority idle | 0.80 / 100% | 0.22 / 31% | 0.58 / 100% |
| 0.6 | 0.74 / 100% | majority idle | 0.85 / 100% | 0.51 / 100% | 0.74 / 100% |
| 0.7 | 0.84 / 100% | majority idle | 0.90 / 100% | 0.74 / 100% | 0.80 / 100% |
| 0.8 | 0.87 / 100% | majority idle | 0.94 / 100% | 0.84 / 100% | 0.89 / 100% |
| 0.9 | 0.91 / 100% | majority idle | 0.96 / 100% | 0.93 / 100% | 0.96 / 100% |
| 1.0 | 1.02 / 100% | majority idle | 1.00 / 100% | 0.97 / 100% | 1.01 / 100% |
