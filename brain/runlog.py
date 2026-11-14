"""Record what produced a result, so every number can be reproduced."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from .connectome import ROOT
from .params import SHIU, LIFParams


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def run_record(config: dict, device: str | torch.device = "cpu", params: LIFParams = SHIU) -> dict:
    """Metadata block stored next to every result. `config` must include the seed."""
    device = torch.device(device)
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "lif_params": params.as_dict(),
        "connectome": "Shiu et al. Connectivity_783.parquet (FlyWire v783)",
        "config": config,
    }


def save_json(path: str | Path, doc: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf8")
    return path
