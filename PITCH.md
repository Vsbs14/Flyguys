# Flyguys: pitch outline

**One line:** We coupled real fly connectomes across a galaxy and found which neurons decide whether it goes quiet.

Placeholders are marked **TO FILL** with the file the number comes from. Everything else is measured and in `results/`.

## 1. The question: where is everybody?

- Fermi's question: the galaxy is old and big, so why do we see no one?
- Most answers assume how a civilization decides: expand, hide, or run. We did not assume. We gave every colony a real brain and watched.
- The brain is the fruit fly's: 138,639 neurons and 15 million connections, wired as in the real animal (FlyWire v783).

## 2. The method: the real wiring and one input-to-output rule

- **The rule is the fly's own.** The world tastes good, tastes bad, or something looms. The brain answers eat, back off, or flee.
  - Sugar neurons = how habitable a world is. Bitter neurons = how hostile. Looming neurons = a threat approaching (supernova, black hole, asteroid swarm, an advancing neighbour).
  - MN9 firing rate = growth. Any Giant Fibre spike = abandon the system and flee. MDN = pull back.
- **The model is Shiu et al. 2024, unchanged:** −52 mV rest, −45 mV threshold, 20 ms membrane, 5 ms synapse, 2.2 ms refractory, 1.8 ms delay, 0.275 mV per synapse. Transmitter signs come from their file.
- **It is faithful.** A 400-neuron test network gives the same 15,201 spikes, neuron for neuron and step for step, in our PyTorch engine and in Shiu's original Brian2 code. On the whole brain, MN9 fires at 73 Hz in ours and 75 Hz in Brian2; the Giant Fibre at 185 Hz in both.
- **It passes all six sanity checks on a T4 GPU** (figure: `analysis/figures/sanity_table.png`): silence with no input, MN9 at 69 Hz under sugar, 3 Hz when bitter is added, the Giant Fibre firing in 8 of 8 brains under looming and 0 of 8 when silenced, and activity dying out when input stops.
- **It is fast enough to run thousands of brains.** 1,024 brains side by side on a T4: 18 decisions per second in 1.3 GB.
- **Decisions are cached, not faked.** Each brain variant is run on a 10 × 10 × 10 grid of inputs, 5 times per point, and the galaxy looks the answers up. **TO FILL** from `results/table_validation_intact.md`: error of the cache against the live brain on 200 random inputs.

## 3. The three-galaxy demo

- Same stars, same supernovae, same starting civilizations. Three brains: normal, fearless (Giant Fibre silenced), always hungry (bitter neurons silenced).
- Show them side by side in the viewer (`viz/`, playing `galaxy/frames.json`).
- **TO FILL** from `results/experiments/three_galaxies.json`: final colonies, flights and encounter outcomes for the three galaxies on the demo seed.

## 4. The dark forest result

- **The Giant Fibre is a hair-trigger.** On a 0–200 Hz input scale it fires in every brain once looming reaches 5 Hz, and in most at 2 Hz. A sweet world next to a dying star is not a dilemma for a fly: it leaves (figure: `analysis/figures/flee_probability.png`).
- **Looming also switches feeding off.** With sugar at maximum, MN9 falls from about 68 Hz to 0 Hz under strong looming. A threatened colony stops growing even if it stays.
- When two civilizations meet, each one's expansion looms for the other. **TO FILL** from `results/experiments/variants.json` (figure: `analysis/figures/encounter_outcomes.png`): share of first contacts ending in coexistence, one side retreating, mutual flight, or escalation, for intact and fearless brains.
- **TO FILL** from `results/experiments/variants.json` (figure: `analysis/figures/survival_curves.png`): living colonies over time per variant, and how many civilizations an observer at the Sun's position could detect.
- **TO FILL** (figure: `analysis/figures/effective_p.png`): effective P, the share of colonies that go on to colonise, against Landis's percolation threshold of 0.311.

## 5. The neurons that matter

- Lesion lab: silence one cell type at a time and rerun the galaxy. Only cell types that actually spike can matter, because neurons act on each other only through spikes; about 2,400 of the 138,639 neurons ever spike under our three senses.
- **TO FILL** from `results/lesion_ranking.json` (figure: `analysis/figures/lesion_ranking.png`): the cell types whose removal changes the final number of colonies most, and the brain-size sweep (100 / 50 / 20 / 5% of interneurons) on the same scale.
- **TO FILL**: 3D renders of the top 10 (needs a FlyWire token; script `analysis/render_top10.py`, run in Colab).

## 6. The honest baselines

- **The fly is a fixed filter, not an optimiser.** We say where it simply matches something simpler.
- The ladder: the intact fly, a hand-written rule using thresholds read off the fly's own table, a small neural network fitted to that table, and a control brain with shuffled wiring.
- **TO FILL** from `results/experiments/ladder.json` (figure: `analysis/figures/baseline_ladder.png`): final colonies and civilizations for the four, with error bars.
- **TO FILL** from `results/experiments/sensitivity.json`: which conclusions survive moving the galaxy-to-sense gains by ±50%.
- Things the wiring told us that we did not want to hear:
  - **DNp09, our planned "expand" neuron, stays almost silent** under all three senses (about 1 Hz at most). Ship-launching is therefore a galaxy rule fed by MN9-driven growth, and we say so.
  - **MDN only responds to strong looming** (6–14 Hz at 100 Hz and above), never to bitter. In the intact brain the Giant Fibre has always fired first.
  - **The three senses run on separate lines.** Sugar drives feeding motor neurons, bitter its own small set of descending neurons, looming the escape set. The interesting interactions are the inhibitory ones: bitter and looming both shut feeding down.
  - **Tables are built on a reduced network** of the roughly 6,400 neurons that matter. Against the full brain on fresh inputs it missed 11 of 500,201 spikes and gave identical output counts in 63 of 64 brains.

## Closing line

The galaxy's silence does not need a grand strategy. In a fly it comes down to a handful of neurons, and we can name them.
