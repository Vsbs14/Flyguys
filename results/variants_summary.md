# Brain variants

## Sanity checks

PASS* and FAIL* mark results the lesion is designed to produce. For the size and shuffled variants there is no prediction; the result is just recorded.

| Variant | No input: silence | Sugar: MN9 fires | Bitter: MN9 drops | Looming: GF fires | GF off: no escape | Input off: dies out |
|---|---|---|---|---|---|---|
| Intact brain (reference) | PASS | PASS | PASS | PASS | PASS | PASS |
| Giant Fibre silenced (no panic) | PASS | PASS | PASS | FAIL* | PASS | PASS |
| MDN silenced (no caution) | PASS | PASS | PASS | PASS | PASS | PASS |
| Bitter neurons silenced (always hungry) | PASS | PASS | FAIL* | PASS | PASS | PASS |
| 50% of interneurons kept | PASS | FAIL | FAIL | PASS | PASS | PASS |
| 20% of interneurons kept | PASS | FAIL | FAIL | PASS | PASS | PASS |
| 5% of interneurons kept | PASS | FAIL | FAIL | PASS | PASS | PASS |
| Interneuron wiring shuffled (control) | PASS | FAIL | FAIL | PASS | PASS | PASS |

## Response tables

Largest mean response anywhere on the 10 x 10 x 10 input grid. Tables are built on a reduced network of the neurons that matter. The two check columns compare it with the full brain on 64 fresh inputs: the share of the full brain's spikes that came from neurons the reduction dropped, and the number of brains whose output neurons fired exactly the same number of spikes.

| Variant | Neurons silenced | Neurons simulated | Spikes missed | Brains with identical outputs | Max MN9 (Hz) | Max P(Giant Fibre spike) | Max MDN (Hz) | Max DNp09 (Hz) |
|---|---|---|---|---|---|---|---|---|
| Intact brain (reference) | 0 | 6,377 | 0.0022% | 63/64 | 76 | 1.00 | 18.2 | 4.0 |
| Giant Fibre silenced (no panic) | 2 | 6,335 | 0.0022% | 63/64 | 74 | 0.00 | 17.5 | 3.0 |
| MDN silenced (no caution) | 4 | 6,489 | 0.0022% | 63/64 | 76 | 1.00 | 0.0 | 4.0 |
| Bitter neurons silenced (always hungry) | 42 | 6,169 | 0.0024% | 63/64 | 78 | 1.00 | 17.2 | 3.5 |
| 50% of interneurons kept | 69,121 | 2,355 | 0.0000% | 64/64 | 0 | 1.00 | 0.0 | 15.0 |
| 20% of interneurons kept | 110,593 | 939 | 0.0000% | 64/64 | 0 | 1.00 | 0.0 | 21.5 |
| 5% of interneurons kept | 131,329 | 488 | 0.0000% | 64/64 | 0 | 1.00 | 0.0 | 49.0 |
| Interneuron wiring shuffled (control) | 0 | 11,013 | 0.0005% | 63/64 | 0 | 1.00 | 1.0 | 82.0 |
