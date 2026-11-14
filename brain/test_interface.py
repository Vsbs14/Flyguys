"""Unit checks for the Phase 2 interface. Runs on the CPU in a second, no connectome needed.

  python -m brain.test_interface
"""
from __future__ import annotations

import torch

from .interface import ACTIONS, MAX_HZ, OUTPUTS, SENSES, DecisionRule, intensity_to_hz


def main() -> None:
    # intensities -> Hz: linear, clipped, transposed to (senses, brains)
    hz = intensity_to_hz([[0, 0.5, 1, 2, -1], [1, 1, 1, 1, 1]])
    assert hz.shape == (len(SENSES), 2)
    assert torch.allclose(hz[:, 0], torch.tensor([0, 0.5 * MAX_HZ, MAX_HZ, MAX_HZ, 0]))
    assert intensity_to_hz([0.2] * len(SENSES)).shape == (len(SENSES), 1)
    for bad in ([[0, 0]], [[float("nan")] * len(SENSES)]):
        try:
            intensity_to_hz(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad input {bad}")

    rule = DecisionRule(ref_hz={"feed": 70, "escape": 200, "retreat": 35, "advance": 48}, threshold=0.25)
    pick = lambda *r: ACTIONS[int(rule.choose(torch.tensor([r], dtype=torch.float32))[0])]
    assert pick(0, 0, 0, 0) == "idle"
    assert pick(10, 0, 0, 0) == "idle"                 # 10/70 = 0.14, below threshold
    assert pick(70, 0, 0, 0) == "feed"
    assert pick(0, 0, 20, 0) == "retreat"              # 20/35 = 0.57
    assert pick(0, 100, 30, 0) == "retreat"            # normalised: 0.86 beats 0.5 although 100 Hz > 30 Hz
    assert pick(0, 200, 12, 0) == "escape"             # looming's MDN side effect loses to the Giant Fibre
    assert pick(35, 100, 0, 0) == "feed"               # exact tie (0.5 vs 0.5) goes to the earlier output
    assert pick(0, 0, 0, 12) == "advance"              # 12/48 = 0.25, exactly at threshold
    batch = rule.choose(torch.tensor([[70, 0, 0, 0], [0, 0, 0, 0], [0, 200, 0, 0]], dtype=torch.float32))
    assert [ACTIONS[i] for i in batch.tolist()] == ["feed", "idle", "escape"]

    try:
        DecisionRule(ref_hz={o: 0.0 for o in OUTPUTS})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a zero reference rate")
    print("PASS: interface checks")


if __name__ == "__main__":
    main()
