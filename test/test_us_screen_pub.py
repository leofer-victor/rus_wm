"""Tests for the ROS 2 ultrasound capture configuration contract."""

from pathlib import Path

import pytest
import yaml

from rus_wm.us_screen_pub import load_capture_profile


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("probe_type", "crop_size", "flipped"),
    [
        ("linear", (600, 600), True),
        ("convex", (880, 660), False),
    ],
)
def test_capture_profiles_preserve_legacy_settings(
    probe_type: str,
    crop_size: tuple[int, int],
    flipped: bool,
) -> None:
    profile = load_capture_profile(
        ROOT / "config" / f"screen_cap_config_{probe_type}.yaml"
    )
    assert profile.video_index >= 0
    assert (profile.x1 - profile.x0, profile.y1 - profile.y0) == crop_size
    assert profile.flip_horizontal is flipped


def test_ultrasound_topics_are_consistent() -> None:
    config = yaml.safe_load((ROOT / "config/deepusnav_console.yaml").read_text())
    publisher = config["us_screen_pub"]["ros__parameters"]
    console = config["deepusnav_console"]["ros__parameters"]
    inference = config["deepusnav_inference"]["ros__parameters"]

    assert publisher["output_topic"] == console["ultrasound_topic"]
    assert publisher["output_topic"] == inference["ultrasound_topic"]
    assert publisher["output_topic"] in console["record_topics"]


def test_invalid_crop_is_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "video_index: 0\n"
        "hz: 15\n"
        "frame_size: {width: 100, height: 100}\n"
        "frame_cropped_coordinates: {x0: 0, x1: 101, y0: 0, y1: 50}\n"
    )
    with pytest.raises(ValueError, match="crop x"):
        load_capture_profile(invalid)
