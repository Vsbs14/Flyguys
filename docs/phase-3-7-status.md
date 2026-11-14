# Phases 3–7: work in progress

This branch (`phase-3-5`, built on `phase-1-2`) holds the code for Phases 3 to 7. **It is not finished.**
The code is written and parts of it are tested, but the long GPU jobs that produce the real results were
stopped before they completed, so no response tables, lesion-screen results or experiment results are
committed yet. This page says what exists, what was checked, and what is left to run.

## What each phase is for

| Phase | Question it answers | Code |
|---|---|---|
| 3. Response tables | What does each brain variant decide for every combination of senses? Precomputed once so the galaxy never has to simulate a brain live | `brain/tables.py` (build, GPU), `galaxy/brains.py` (read, NumPy only) |
| 4. Lesion screen | Which groups of neurons, when silenced, change the fly's decisions? The strongest, most different ones become brain variants | `brain/lesion_screen.py` |
| 5. Galaxy | A world where colonies sense, decide and live or die by those decisions | `galaxy/world.py`, `galaxy/run.py` |
| 6. Experiments and baselines | Does the real fly brain keep colonies alive better than a lesioned brain, a scrambled brain, a hand-written rule or random choice? | `galaxy/experiments.py`, `galaxy/policies.py` |
| 7. Visuals and backend | Playback files for the browser UI, an HTTP API, figures | `galaxy/playback.py`, `api/server.py`, `analysis/figures.py` |

## How the pieces fit

```
brain (GPU) ──► lesion screen ──► data/variants.json
     │                                   │
     └──────► response tables ◄──────────┘     tables/<variant>.npz
                     │
                     ▼
     galaxy (CPU) ──► experiments ──► results/experiments.*
           │
           └──► playback ──► galaxy/frames.json ──► api/server.py ──► browser UI
```

## The galaxy (Phase 5), in plain words

About 60 star systems on spiral arms, joined by jump lanes. Each system has resources that regrow, and a
quarter of them are toxic. Raider fleets wander between systems, drawn to populated ones. Every tick, each
colony senses its system with the five senses from Phase 2:

| Sense | Set by |
|---|---|
| sugar | how full the system's resources are |
| bitter | how toxic they are |
| looming | a raider in the system |
| obstacle | how crowded the system is |
| target | how much richer the best neighbouring system is |

Its brain picks an action: **feed** (harvest; toxic food poisons), **escape** (jump to a safe neighbour),
**retreat** (move to a less crowded neighbour), **advance** (send settlers to a richer neighbour) or
**idle**. Unlike the galaxy on the VSTemp branch, escaping moves the colony instead of deleting it, so a
fly that escapes has a real advantage. All numbers live in `GalaxyConfig` and are meant to be retuned.

## What was checked

* `python -m galaxy.test_galaxy` passes. It checks that table lookups are exact at grid points, that the
  same seed gives the same run, that the map is connected, that feeding grows colonies and not feeding
  shrinks them, that merging colonies conserves population, and that escaping colonies avoid raids
  while colonies that stay are hit.
* A small trial run (a coarse 3-level intact table plus a fake brain that never escapes) behaved as
  intended: the brain that cannot escape went extinct in 2 of 3 galaxies, the intact brain survived all 3.
* The scrambled-brain control keeps every neuron's number of inputs and outputs exactly. About 35% of
  connections stay in place, because many neurons have no partner with the same counts to swap with.
* The lesion screen and table builder both ran correctly at small scale on the GPU.

## What is not done

| Item | Status | To finish it |
|---|---|---|
| Lesion screen over all 1,337 active cell types | Stopped partway (about 50 min on a GPU). No results committed | `python -m brain.lesion_screen --device cuda --replicates 4` (or Colab section 5c) |
| Response tables for intact, the picked lesions and the scrambled brain | Not built | `python -m brain.tables --device cuda` (Colab section 5d), about 10 min per variant |
| Experiments | Code written, never run on real tables | `python -m galaxy.experiments` (CPU, a few minutes) |
| Playback file `galaxy/frames.json` | Code written, never run on real tables | `python -m galaxy.playback` |
| API end-to-end test | Written, not yet run (needs the tables) | `python -m api.test_api` |
| Figures | Only the dose-response figure exists; the others are drawn once their results exist | `python -m analysis.figures` |
| Connecting the UI on the VSTemp branch | Not started. `galaxy/frames.json` uses the UI's frame format, and `api/server.py` uses the same endpoint names as its `viz/server.py`, so the page should be able to point at this backend | Team decision on which galaxy to keep |

Run the unfinished items in this order: lesion screen, tables, experiments, playback, figures, API test.
