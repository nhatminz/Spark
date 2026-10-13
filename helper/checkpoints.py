"""Atomic checkpoints and reproducible random state."""

from pathlib import Path

import torch


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(output, step, payload):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"step_{step}.pt"
    temporary = output / ".checkpoint.tmp"
    torch.save(payload, temporary)
    temporary.replace(path)
    link = output / ".latest.tmp"
    link.unlink(missing_ok=True)
    link.symlink_to(path.name)
    link.replace(output / "latest.pt")


def rng_state():
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
