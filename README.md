# Flyguys

Real fruit fly connectomes, coupled across a simulated galaxy. Each colony decides what to do with a fly brain: its world tastes good, tastes bad, or something looms, and the fly's wiring answers eat, back off, or flee.

## Run the demo

From this directory, install `requirements.txt` in a Python virtual environment, then start the server.

```bash
python -m viz.server
```

Open http://localhost:8000. **Live Demo** plays a recorded run from the cached fly responses.
**The Fly Brain** shows the escape response map. **Baselines** compares survival and
detectability at three signal gains. The playback and cached lookup use the supplied
artifacts. The live console also needs the connectome download.

## Reproduce the experiments

```bash
python scripts/fetch_data.py
python -m brain.test_equivalence
python -m brain.sanity
python -m galaxy.responses
python -m galaxy.simulate
python -m viz.frames
```

### Run the response table on Colab

Upload a copy of this checkout or clone a branch that contains the galaxy code.
Select a GPU runtime, install `requirements.txt`, then run these commands from
the repo root (prefix shell commands with `!` in a notebook cell).

```bash
python scripts/fetch_data.py
python -m galaxy.responses --device cuda --batch 64
```

Download `tables/responses.npz` and put it in the same directory of your local
checkout. Run `python -m galaxy.simulate` locally to refresh the results and
playback. The browser demo itself runs locally and needs no GPU.

Use `python -m galaxy.responses --device cuda` on a GPU. The default cache runs three
independent replicates at each of 216 input combinations for each of five brain
variants. Each decision starts from rest. The galaxy samples the neighboring grid
points and a stored replicate. It uses the same fixed rate normalization from the
intact trials for every brain variant. The threshold policy sees the same scaled
inputs and uses the same downstream colony rules.

The experiment compares the intact brain, Giant Fibre silenced, Giant Fibre and
MDN silenced, bitter inputs silenced, a degree-preserving shuffle, and a threshold
policy. Run the same six checks on a variant with `python -m brain.sanity --variant VARIANT`.
The shuffled control preserves incoming and outgoing edge counts by permuting
interneuron targets with equal incoming degree. It loses the sugar-to-MN9 response
in the recorded sanity run, which limits how broadly to interpret its survival result.
The intact GPU sanity checks and benchmark are in `results/`.

Galaxy assumptions are synthetic. Each run has 64 stars and 48 epochs. Scheduled
supernovae damage stars within 0.16 units of the source in a unit-radius disk.
The observer at (0.5, 0) detects colonies within 0.45 units that have reached at
least half their population capacity. We compare 50 matched maps per gain at
0.8×, 1.0× and 1.2×. The report saves means, standard errors, and paired survival
differences relative to the intact brain. Its uncertainty reflects galaxy seeds
with a fixed response table and one shuffled wiring realization.

The first-contact storyboard and hardware panel are previews. The wider lesion
screen, network thinning, small-network baseline and hardware integration are
follow-up work.

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
