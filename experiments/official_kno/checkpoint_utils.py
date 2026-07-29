from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint_last(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    epoch: int,
    args: dict[str, Any],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.kernel.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "config": args,
        "seed": seed,
    }
    if hasattr(model, "scheduler"):
        checkpoint["scheduler_state_dict"] = model.scheduler.state_dict()
    torch.save(checkpoint, path)
