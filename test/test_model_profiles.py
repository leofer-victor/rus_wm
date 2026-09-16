"""Tests for runtime-selectable inference model profiles."""

from pathlib import Path

import pytest

from rus_wm.model_profiles import resolve_model_profile


@pytest.mark.parametrize(
    ("name", "checkpoint_name", "atlas_enabled"),
    [
        ("dino", "dinov2_dino_wm_main.pt", False),
        ("vjepa", "vjepa2_vjepa2_ac_main.pt", False),
        ("atlas", None, True),
        ("dino_atlas", "dinov2_dino_wm_main.pt", True),
        ("vjepa_atlas", "vjepa2_vjepa2_ac_main.pt", True),
    ],
)
def test_named_profile_components(
    tmp_path: Path,
    name: str,
    checkpoint_name: str | None,
    atlas_enabled: bool,
) -> None:
    profile = resolve_model_profile(name, tmp_path)
    expected = (
        None
        if checkpoint_name is None
        else tmp_path / "training_setup_and_weights" / "checkpoints" / checkpoint_name
    )
    assert profile.checkpoint_path == expected
    assert profile.atlas_enabled is atlas_enabled
    assert profile.world_model_enabled is (checkpoint_name is not None)


def test_checkpoint_override_does_not_require_code_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "experiment.pt"
    profile = resolve_model_profile("dino", tmp_path, checkpoint)
    assert profile.checkpoint_path == checkpoint


def test_custom_profile_can_combine_checkpoint_and_atlas(tmp_path: Path) -> None:
    checkpoint = tmp_path / "experiment.pt"
    profile = resolve_model_profile("custom", tmp_path, checkpoint, True)
    assert profile.checkpoint_path == checkpoint
    assert profile.atlas_enabled is True


def test_atlas_only_rejects_an_unused_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="atlas-only"):
        resolve_model_profile("atlas", tmp_path, tmp_path / "unused.pt")


def test_vjeap_typo_is_normalised(tmp_path: Path) -> None:
    profile = resolve_model_profile("vjeap", tmp_path)
    assert profile.name == "vjepa"
