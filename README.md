# Flyguys

Real fruit fly connectomes, coupled across a simulated galaxy. Each colony decides what to do with a fly brain: its world tastes good, tastes bad, or something looms, and the fly's wiring answers eat, back off, or flee.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data, neuron IDs, repo layout | Done |
| 1 | Batched, maskable fly-brain engine with sanity checks | Done: CPU and GPU sanity checks, benchmark, whole-brain check against Brian2 |
| 2 | Galaxy → senses, neurons → decisions | Done on the brain side (`brain/interface.py`); the galaxy side plugs in at Phase 5 |
| 3 | Brain variants and response tables | Not started |
| 4 | Lesion screen | Not started |
| 5 | Galaxy simulation | Not started |
| 6 | Experiments and baselines | Not started |
| 7 | Visuals and pitch | Not started |

## Layout

| Folder | Contents |
|---|---|
| `brain/` | The simulator: PyTorch port of the Shiu et al. LIF model, sanity checks, benchmark |
| `data/` | Small committed inputs: neuron groups, the Brian2 equivalence case |
| `scripts/` | Data download, neuron-group builder, Brian2 reference runs |
| `results/` | Result tables, each saved with the configuration and seed that produced it |
| `tables/` | Cached brain response tables (Phase 3) |
| `galaxy/` | Galaxy simulation (Phase 5) |
| `viz/` | three.js playback (Phase 7) |
| `analysis/` | Figures |
| `docs/` | Plain-language write-ups, e.g. [what Phases 1 and 2 built](docs/phase-1-2-summary.md) |
| `external/` | Downloaded third-party data. Not committed |

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # Shiu's connectome (about 100 MB), no token needed
python -m brain.test_equivalence      # the port matches Brian2 spike for spike
python -m brain.sanity                # the six sanity checks, on CPU (about 2 minutes)
python -m brain.test_interface        # decision-rule unit checks (instant, no connectome)
python -m brain.calibrate             # Phase 2: calibrate the decision rule (slow on CPU; section 5b of the Colab notebook runs it on a GPU)
```

On a GPU, open [`fly_brain_gpu.ipynb`](fly_brain_gpu.ipynb) in Colab
([link](https://colab.research.google.com/github/Vsbs14/Flyguys/blob/main/fly_brain_gpu.ipynb)),
pick the T4 runtime and run all cells. `brain/modal_app.py` does the same on Modal if Colab is not enough.

## The engine

`brain/sim.py` is a PyTorch port of the leaky integrate-and-fire model of
[Shiu et al. 2024](https://www.nature.com/articles/s41586-024-07763-9), using their settings unchanged:

| Setting | Value |
|---|---|
| Resting and reset potential | −52 mV |
| Spike threshold | −45 mV |
| Membrane time constant | 20 ms |
| Synapse time constant | 5 ms |
| Refractory period | 2.2 ms |
| Synaptic delay | 1.8 ms |
| Weight per synapse | 0.275 mV |
| Time step | 0.1 ms |

Connection weights are `synapse count × (+1 or −1)`, read from Shiu's `Connectivity_783.parquet`. The transmitter signs are theirs.

State has shape (neurons × brains). Each column is one independent brain with its own inputs, random stream and mask of silenced neurons. Inputs are Poisson spike trains at 0–200 Hz per sense. Every decision starts from rest.

```python
from brain import Connectome, NeuronGroups, BrainSim

conn = Connectome.load()
groups = NeuronGroups(conn)
sim = BrainSim(conn, device="cuda")

state = sim.new_state(n_brains=64, seed=0)
res = sim.run(state, duration_ms=200,
              inputs=[(groups.sense("sugar"), 200.0)],
              monitor=groups.output("feed"))
res.rates_hz        # (monitored neurons, brains)
```

### Neurons used

Selected from the public FlyWire annotation table with the same filters as the hackathon starter notebook. The resolved v783 IDs are in `data/neuron_groups.json`.

| Role | Neurons | Count |
|---|---|---|
| Sugar (input) | LB3c taste neurons | 32 |
| Bitter (input) | bitter taste neurons | 42 |
| Looming (input) | LC4 and LPLC2 | 314 |
| Obstacle (input) | LC16 | 151 |
| Target (input) | LC31b | 10 |
| Feed (output) | MN9 | 2 |
| Escape (output) | Giant Fibre (DNp01) | 2 |
| Retreat (output) | MDN | 4 |
| Advance (output) | DNp09 | 2 |

### Validation

* **Against Brian2, exact.** `scripts/brian2_equivalence.py` runs a 400-neuron random network in Shiu's original Brian2 model with no random input. `python -m brain.test_equivalence` replays it in the port: all 15,201 spikes match in neuron and time step.
* **Against Brian2, whole brain.** `scripts/brian2_reference.py` runs Shiu's model on the full connectome and `python -m brain.validate_port` compares output firing rates ([results/port_validation.md](results/port_validation.md)). Feed and escape rates, whole-brain spike counts and the number of active neurons agree within about 4%. MDN under looming is 22% lower in the port (9.6 vs 12.4 Hz).
* **Sanity checks.** `python -m brain.sanity` runs the six checks and writes a pass/fail table to `results/`. All six pass on CPU and GPU ([results/sanity_intact_cuda.md](results/sanity_intact_cuda.md)).
* **Speed.** About 27 decisions (200 ms each) per second at 1,024 brains, using 1.3 GB of GPU memory ([results/benchmark_cuda.md](results/benchmark_cuda.md)). That run used a laptop GPU; a Colab T4 run of `fly_brain_gpu.ipynb` will give the numbers for the project's usual setup.

## Decisions (Phase 2)

`brain/interface.py` is the contract between the world and the brain. The world gives each colony five
sense intensities in [0, 1]. Intensity 1 means 200 Hz Poisson input on every neuron of that sense. The
brain runs 200 ms from rest and returns one action: `idle`, `feed`, `escape`, `retreat` or `advance`.

```python
from brain.interface import FlyBrain

brain = FlyBrain.load(device="cuda")
#                  sugar bitter looming obstacle target
d = brain.decide([[1,    0,     0,      0,       0],     # food            -> feed
                  [1,    1,     0,      0,       0],     # toxic food      -> idle
                  [1,    0,     1,      0,       0]])    # food + a threat -> escape
d.actions, d.rates_hz, d.scores
```

Each output's rate is divided by its reference rate, which is what it fires at when its own sense is fully
on (sugar → MN9, looming → Giant Fibre, obstacle → MDN, target → DNp09). The highest score wins, or
`idle` if no score reaches 0.25. Normalising matters because the Giant Fibre reaches 200 Hz and MDN only
about 35 Hz. `python -m brain.calibrate` measures the references, saves them to `data/decision_rule.json`,
and checks 11 scenarios with fixed expected actions ([results/decision_calibration.md](results/decision_calibration.md)).
All 11 pass in 16 of 16 brains, and the table also shows the dose at which each action starts.

**Why obstacle and target were added.** With only sugar, bitter and looming, DNp09 (advance) never fires
above about 3 Hz and MDN (retreat) only appears as a side effect of looming, so two of the four actions
could not happen. Stimulating every sensory and visual projection cell type on its own picked the types
that drive one output most cleanly. LC16 drives MDN at about 35 Hz, consistent with LC16 activation
causing backward walking (Wu et al. 2016). LC31b drives DNp09 at about 50 Hz; this one comes from the
connectome screen only, not from a published experiment.

Decisions are stochastic because the input is Poisson. Phase 3 should average replicates rather than
trust a single brain, especially near the threshold.

One detail differs from the starter notebook's simulator: in Brian2, synaptic input that reaches a neuron during its refractory period is discarded, and the port does the same. The starter notebook lets that input accumulate, so its firing rates are somewhat higher.

## Secrets

No file in this repo needs a token. If you use `fafbseg` (Phase 4), keep the FlyWire token in your home directory or an environment variable, never in the repo.
