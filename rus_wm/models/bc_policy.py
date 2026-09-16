"""Behaviour cloning: a reactive image-to-action policy, SonoGym's imitation baseline.

SonoGym trains its imitation policies on demonstrations from a proportional controller
on the true probe-to-goal pose (`sonogym/lerobot.py`). Their released demonstrations come
from one patient and a GAN renderer, so they cannot support a cross-patient claim; this is
the same learning problem rebuilt on our renderer and our patient split. The expert is
`planner.OracleGeometryPolicy` -- the controller SonoGym's expert is, clipped to the task's
action limits -- and the policy sees what every other method sees: one frame, through the
same frozen encoder and the same ``grid4x4`` pooling the atlas uses. No planner, no
memory, no goal image: a trained reflex.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class BCHead(nn.Module):
    """``grid4x4`` feature -> raw action in [-1, 1]^3, with its input standardisation inside."""

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 3), nn.Tanh())
        # Buffers, not attributes, so saving the head saves its standardisation
        # (the lesson `atlas_probe.PositionHead` records).
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def save_bc_head(head: BCHead, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "in_dim": head.net[0].in_features,
                "hidden": head.net[0].out_features, "meta": meta}, path)


def load_bc_head(path: Path, device: str = "cpu") -> tuple[BCHead, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    head = BCHead(blob["in_dim"], blob["hidden"]).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head, blob["meta"]
