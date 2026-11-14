# Flyguys

Real fruit fly connectomes, coupled across a simulated galaxy. Each colony decides what to do with a fly brain: its world tastes good, tastes bad, or something looms, and the fly's wiring answers eat, back off, or flee.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Data, neuron IDs, repo layout | Done |
| 1 | Batched, maskable fly-brain engine | Done. Matches Brian2 spike for spike; six sanity checks pass on CPU and on a T4 (`results/`) |
| 2 | Galaxy → senses, neurons → decisions | Done (`galaxy/mapping.py`, `results/mapping_examples.md`) |
| 3 | Brain variants and response tables | Done except the no-MDN variants and the shuffled control (still building). Table vs live brain: `results/table_validation_intact.md` |
| 4 | Lesion screen | Partial: 913 candidate cell types found, 20 most active screened (`results/lesion_ranking.md`). 3D renders need a FlyWire token (`analysis/render_top10.py`) |
| 5 | Galaxy simulation | Done (`galaxy/`); three-galaxy recording in `galaxy/frames.json` |
| 6 | Experiments and baselines | Done at 20 seeds, not 100. Sensitivity sweep at 8 seeds (`results/experiments/`) |
| 7 | Visuals and pitch | Figures in `analysis/figures/`, pitch in `PITCH.md`; viewer is on the `UI` branch |

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
| `external/` | Downloaded third-party data. Not committed |

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # Shiu's connectome (about 100 MB), no token needed
python -m brain.test_equivalence      # the port matches Brian2 spike for spike
python -m brain.sanity                # the six sanity checks, on CPU (about 2 minutes)
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
| Feed (output) | MN9 | 2 |
| Escape (output) | Giant Fibre (DNp01) | 2 |
| Retreat (output) | MDN | 4 |
| Advance (output) | DNp09 | 2 |

### Validation

* **Against Brian2, exact.** `scripts/brian2_equivalence.py` runs a 400-neuron random network in Shiu's original Brian2 model with no random input. `python -m brain.test_equivalence` replays it in the port: all 15,201 spikes match in neuron and time step.
* **Against Brian2, whole brain.** `scripts/brian2_reference.py` runs Shiu's model on the full connectome and `python -m brain.validate_port` compares output firing rates and writes `results/port_validation.md`. That reference run takes about half an hour and its table is not committed yet.
* **Sanity checks.** `python -m brain.sanity` runs the six checks and writes a pass/fail table to `results/`.

One detail differs from the starter notebook's simulator: in Brian2, synaptic input that reaches a neuron during its refractory period is discarded, and the port does the same. The starter notebook lets that input accumulate, so its firing rates are somewhat higher.

## Secrets

No file in this repo needs a token. If you use `fafbseg` (Phase 4), keep the FlyWire token in your home directory or an environment variable, never in the repo.
