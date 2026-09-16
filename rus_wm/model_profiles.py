"""Resolve named inference profiles without importing ROS, Qt or PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_CHOICES = ("dino", "vjepa", "atlas", "dino_atlas", "vjepa_atlas", "custom")


@dataclass(frozen=True)
class ModelProfile:
    """Concrete model components selected for one inference process."""

    name: str
    checkpoint_path: Path | None
    atlas_enabled: bool

    @property
    def world_model_enabled(self) -> bool:
        """Return whether this profile contains an action-conditioned world model."""
        return self.checkpoint_path is not None


def resolve_model_profile(
    profile_name: str,
    deepusnav_root: str | Path,
    checkpoint_override: str | Path = "",
    custom_atlas_enabled: bool = False,
) -> ModelProfile:
    """Turn a stable profile name into checkpoint and Atlas choices.

    A checkpoint override replaces the preset checkpoint for world-model profiles.
    The ``custom`` profile requires an override and preserves ``atlas_enabled`` for
    backwards-compatible, fully manual configurations.
    """
    name = str(profile_name).strip().lower().replace("-", "_")
    if name == "vjeap":  # Accept the common transposition, but report the canonical name.
        name = "vjepa"
    if name not in MODEL_CHOICES:
        choices = ", ".join(MODEL_CHOICES)
        raise ValueError(f"unknown model_profile {profile_name!r}; choose one of: {choices}")

    root = Path(deepusnav_root).expanduser().resolve()
    checkpoint_dir = root / "training_setup_and_weights" / "checkpoints"
    checkpoint_names = {
        "dino": "dinov2_dino_wm_main.pt",
        "vjepa": "vjepa2_vjepa2_ac_main.pt",
        "dino_atlas": "dinov2_dino_wm_main.pt",
        "vjepa_atlas": "vjepa2_vjepa2_ac_main.pt",
    }
    atlas_enabled = name in {"atlas", "dino_atlas", "vjepa_atlas"}

    override = str(checkpoint_override).strip()
    if name == "atlas":
        if override:
            raise ValueError(
                "checkpoint_path cannot be combined with atlas-only mode; use "
                "dino_atlas, vjepa_atlas or custom"
            )
        checkpoint = None
    elif name == "custom":
        if not override:
            raise ValueError("custom model_profile requires checkpoint_path")
        checkpoint = Path(override).expanduser().resolve()
        atlas_enabled = bool(custom_atlas_enabled)
    else:
        checkpoint = (
            Path(override).expanduser().resolve()
            if override
            else checkpoint_dir / checkpoint_names[name]
        )

    return ModelProfile(
        name=name,
        checkpoint_path=checkpoint,
        atlas_enabled=atlas_enabled,
    )
