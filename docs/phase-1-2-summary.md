# Phases 1 and 2: what was built and what it means

This branch (`phase-1-2`) finishes Phase 1 (proving the fly-brain simulator works) and Phase 2
(connecting the brain to a world: senses in, a decision out). No training is involved anywhere. The
brain's wiring is the real FlyWire connectome and stays fixed. "Running" the project means simulating
that wiring and reading off what it does.

## The big picture

```
 world (galaxy, Phase 5)          fly brain (139,000 neurons)                 action
 ───────────────────────          ──────────────────────────                 ──────
 five numbers per colony   ──►    sense neurons fire  ──►  wiring  ──►       idle / feed / escape /
 each between 0 and 1             (0–200 Hz)               (fixed)            retreat / advance
```

Each colony gives the brain five numbers. The brain runs for 0.2 seconds of simulated time, four
output neurons respond, and the strongest response becomes the colony's action.

## Phase 1: is the simulator trustworthy?

The team's simulator (`brain/sim.py`) is a fast GPU rewrite of the model from Shiu et al. 2024 (the
paper whose model is the starting point). Before building on it, four things were checked:

| Check | Result | File |
|---|---|---|
| Exact match with the original model on a small 400-neuron network | All 15,201 spikes identical | `brain/test_equivalence.py` |
| Six basic behaviours (silence with no input, sugar → eat, bitter stops eating, looming → escape, no escape with the escape neuron silenced, activity stops when input stops) | 6 of 6 pass on CPU and GPU | `results/sanity_intact_cuda.md` |
| Whole brain compared with the original model (Brian2) | Within about 4% on eating, escape and overall activity | `results/port_validation.md` |
| Speed | About 27 brain decisions per second, 1.3 GB of GPU memory | `results/benchmark_cuda.md` |

One known gap is backing up (the MDN neuron) under a looming threat, which comes out about 22% lower
than in the original model (9.6 vs 12.4 Hz). It is small, but keep it in mind for Phase 4.

The GPU numbers above came from a laptop GPU. The Colab notebook (`fly_brain_gpu.ipynb`) runs the
same checks on Colab's free T4 GPU, which is the project's normal setup. Speed there will differ
somewhat.

## Phase 2: turning senses into decisions

### The problem found

The original three senses (sugar, bitter, looming) could only ever produce two actions: eat and
escape. The "walk forward" neuron (DNp09) never fired above 3 Hz, and the "walk backward" neuron (MDN)
only fired a little as a side effect of looming. A game where two of four actions can never happen
would not work.

### The fix: two new senses

Every sensory and visual cell type in the fly brain (537 of them) was switched on one at a time to see
what each drives. The two cleanest drivers became new senses:

| Sense | Neurons | Drives | Why trust it |
|---|---|---|---|
| sugar | LB3c taste cells | feed (MN9) | from the original project |
| bitter | bitter taste cells | suppresses feeding | from the original project |
| looming | LC4 + LPLC2 | escape (Giant Fibre) | from the original project |
| **obstacle** (new) | LC16 | retreat (MDN), ~35 Hz | Matches a published experiment: switching on LC16 makes real flies walk backwards (Wu et al. 2016) |
| **target** (new) | LC31b | advance (DNp09), ~50 Hz | Found by the wiring screen only, so it is weaker evidence; worth saying so in the pitch |

### How a decision is made

1. The world gives five numbers from 0 to 1. A value of 1 means that sense's neurons fire at 200 Hz.
2. The brain runs for 200 ms and the four output neurons' firing rates are measured.
3. Each rate is divided by that output's *reference rate*, which is how fast it fires when its own sense
   is fully on. This puts all four outputs on the same scale. Without it the escape neuron (up to
   200 Hz) would always beat the retreat neuron (about 35 Hz).
4. The highest score wins. If no score reaches 0.25, the fly does nothing (`idle`).

The reference rates are measured by `python -m brain.calibrate` and saved in `data/decision_rule.json`.

### Did it work?

Eleven situations were tested, each with an action decided in advance. All 11 gave the expected
action in 16 out of 16 simulated brains (`results/decision_calibration.md`). Several of these
behaviours were not programmed in; they come from the real wiring:

- Sugar and bitter together → the fly does nothing (bitter overrides sugar).
- Sugar while something looms → escape (danger overrides eating).
- With the escape neuron silenced, a looming threat → the fly backs away instead. This is a preview of
  what the Phase 4 lesion screen will explore.

The same file has a dose-response table showing how strong each sense has to be before the fly acts.
For example, feeding starts at about 0.3–0.4 sugar.

## Files added or changed

| File | What it is |
|---|---|
| `brain/interface.py` | **New.** The senses-to-decision code. `FlyBrain.decide(...)` is the one function the galaxy will call |
| `brain/calibrate.py` | **New.** Measures the reference rates and runs the 11 scenario checks |
| `brain/test_interface.py` | **New.** Quick checks of the decision logic. Needs no GPU and no data |
| `data/decision_rule.json` | **New.** The calibrated reference rates |
| `data/neuron_groups.json`, `scripts/build_neuron_groups.py` | Adds the obstacle and target senses. The original groups are unchanged |
| `brain/bench.py` | Benchmark now feeds all five senses instead of three |
| `fly_brain_gpu.ipynb` | New section 5b (Phase 2) and section 8 (example decisions). It clones `phase-1-2`; change `BRANCH` back to `"main"` after merging |
| `results/*` | Every result above, saved with the settings and seed that produced it |
| `README.md` | Status table and Phase 2 section updated |

## How to run it yourself (no local GPU needed)

Open `fly_brain_gpu.ipynb` in Colab from this branch, choose Runtime → Change runtime type → T4 GPU,
and run all cells. It downloads everything it needs and saves its results as a zip at the end.

## What comes next

- **Decide what the galaxy is (Phase 5 design).** The brain now expects five numbers per colony. Someone
  has to decide what in the world sets each one (for example, a nearby resource → sugar, a hostile
  neighbour → looming). This is the biggest open question.
- **Phase 3, response tables.** Pre-compute decisions over a grid of sense values so the galaxy can look
  them up instead of simulating live. Six levels per sense with 8 repeats is about 62,000 decisions,
  roughly 40 minutes per brain variant at the measured speed.
- **Decisions are a little random,** because the sense input is random spikes. Average several repeat
  brains rather than trusting one, especially for responses near the 0.25 cut-off.
