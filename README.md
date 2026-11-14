# Flyguys

Real fruit fly connectomes, coupled across a simulated galaxy. Each colony decides what to do with a fly brain: its world tastes good, tastes bad, or something looms, and the fly's wiring answers eat, back off, or flee.

## Run the demo

From this directory, install `requirements.txt` in a Python virtual environment, then start the server.

```bash
python -m viz.server
```

Open http://localhost:8000. **Live Demo** plays `galaxy/frames.json`: the same 1,500-star galaxy run three times, by intact, fearless (Giant Fibre silenced) and always-hungry (bitter silenced) flies. **The Fly Brain** shows the escape response map from the intact response table. **Baselines** compares the brains across the sense-mapping gains of the sensitivity sweep. The live console also needs the connectome download.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data, neuron IDs, repo layout | Done |
| 1 | Batched, maskable fly-brain engine | Done. Matches Brian2 spike for spike; six sanity checks pass on CPU and on a T4 (`results/`) |
| 2 | Galaxy → senses, neurons → decisions | Done (`galaxy/mapping.py`, `results/mapping_examples.md`) |
| 3 | Brain variants and response tables | Done except the no-MDN variants and the shuffled control in `tables/`. Table vs live brain: `results/table_validation_intact.md` |
| 4 | Lesion screen | Partial: 913 candidate cell types found, 20 most active screened (`results/lesion_ranking.md`). 3D renders need a FlyWire token (`analysis/render_top10.py`) |
| 5 | Galaxy simulation | Done (`galaxy/world.py`, `galaxy/sim.py`) |
| 6 | Experiments and baselines | Done at 20 seeds, not 100. Sensitivity sweep at 8 seeds (`results/experiments/`) |
| 7 | Visuals and pitch | Viewer in `viz/`, figures in `analysis/figures/`, pitch in `PITCH.md` |

## Results so far (20 galaxy seeds, mean ± standard error)

| Brain | Final colonies | Living civilizations (of 8) | Flights |
|---|---|---|---|
| Intact fly | 70.7 ± 5.3 | 7.7 | 214 |
| Giant Fibre silenced (fearless) | 129.2 ± 5.8 | 7.7 | 0 |
| Bitter silenced (always hungry) | 506.6 ± 7.0 | 6.6 | 11,975 |
| 50%, 20% or 5% of interneurons | 0 | 0 | 6 |
| Hand-written rule | 67.6 ± 4.8 | 7.3 | 226 |
| Small neural net fitted to the fly | 75.4 ± 4.0 | 7.7 | 252 |

* **The fly matches a hand rule.** At the base mapping the intact fly, the hand rule and the neural net are within about one standard error. They separate only when the mapping gains move away from 1 (sensitivity sweep).
* **Dark forest needs a hungry brain.** In the three-galaxy run, first contacts between intact flies were all peaceful. Always-hungry flies produced 650 mutual flights and 101 escalations.
* **Fear costs territory.** Silencing the Giant Fibre stops all flight and nearly doubles final colonies.
* **Shrinking the brain kills feeding.** With half the interneurons removed, MN9 never fires, so colonies never grow and every civilization dies.

## Reproduce

```bash
python scripts/fetch_data.py
python -m brain.test_equivalence                # port vs Brian2, spike for spike
python -m brain.sanity                          # six checks; --variant for a lesioned brain
python -m brain.build_tables                    # response tables for every variant (tables/)
python -m galaxy.run three --seed 42 --every 3  # three-galaxy recording -> galaxy/frames.json
python -m galaxy.run ladder --seeds 20          # fly vs hand rule vs neural net
python -m galaxy.run variants --seeds 20
python -m galaxy.run sensitivity --seeds 8
python -m analysis.export_viewer                # results -> results/galaxy_experiment.json for the viewer
python -m analysis.figures
```

On a GPU, `fly_brain_gpu.ipynb` runs the sanity checks and benchmark, and its last section rebuilds the tables and the lesion screen on the full brain.

### The earlier quick model

`galaxy/responses.py` and `galaxy/simulate.py` are a smaller first version: one civilization, 64 stars, 48 epochs, and a 6 x 6 x 6 response table on evenly spaced inputs (`tables/responses.npz`). They still run (`python -m galaxy.responses`, then `python -m galaxy.simulate --results results/galaxy_experiment_quick.json --output galaxy/frames_quick.json`). The main model replaced them for two reasons. The Giant Fibre goes from never firing to always firing between about 1 and 3 Hz of looming input, so a looming axis in 40 Hz steps cannot show a graded response; the main table puts most of its looming levels below 10 Hz. And the main model maps the galaxy to the senses with the 2 SD clip the plan calls for.

## Layout

| Folder | Contents |
|---|---|
| `brain/` | The simulator: PyTorch port of the Shiu et al. LIF model, sanity checks, benchmark |
| `data/` | Small committed inputs: neuron groups, the Brian2 equivalence case |
| `scripts/` | Data download, neuron-group builder, Brian2 reference runs |
| `results/` | Result tables, each saved with the configuration and seed that produced it |
| `tables/` | Cached whole-brain responses |
| `galaxy/` | Response-table generator, paired galaxy experiments and recorded playback |
| `viz/` | Browser playback and local API |
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

Two decision paths exist. The galaxy, response tables, variants and lesion screen use the three core senses of the handbook rule (sugar, bitter, looming) and read output rates directly, with any Giant Fibre spike meaning flee (`galaxy/mapping.py`). The 5-sense interface below, from the `phase-1-2` branch, adds obstacle (LC16) and target (LC31b) so that MDN and DNp09 can be driven too, and picks one action per decision. It is calibrated and tested (`results/decision_calibration.md`) but not yet wired into the galaxy; that is the natural next step.

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
