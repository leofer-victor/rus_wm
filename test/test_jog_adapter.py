"""Contract tests for guarded relative-jog conversion."""

import math
from pathlib import Path

import pytest
import yaml

from rus_wm.jog_adapter import target_from_jog


ROOT = Path(__file__).resolve().parents[1]


def test_base_frame_jog_is_added_without_rotation() -> None:
    position, orientation = target_from_jog(
        (0.2, 0.1, 0.6),
        (0.0, 0.0, 0.0, 2.0),
        (0.005, -0.002, 0.001),
        "fr3_link0",
        "fr3_link0",
        "fr3_hand_tcp",
    )
    assert position == pytest.approx((0.205, 0.098, 0.601))
    assert orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_tool_frame_jog_uses_current_orientation() -> None:
    half_turn = math.sqrt(0.5)
    position, _ = target_from_jog(
        (0.2, 0.1, 0.6),
        (0.0, 0.0, half_turn, half_turn),
        (0.005, 0.0, 0.0),
        "fr3_hand_tcp",
        "fr3_link0",
        "fr3_hand_tcp",
    )
    assert position == pytest.approx((0.2, 0.105, 0.6), abs=1e-12)


def test_unknown_jog_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported jog frame"):
        target_from_jog(
            (0.2, 0.1, 0.6),
            (0.0, 0.0, 0.0, 1.0),
            (0.001, 0.0, 0.0),
            "camera",
            "fr3_link0",
            "fr3_hand_tcp",
        )


def test_jog_topics_are_consistent_and_recorded() -> None:
    config = yaml.safe_load((ROOT / "config/deepusnav_console.yaml").read_text())
    console = config["deepusnav_console"]["ros__parameters"]
    adapter = config["deepusnav_jog_adapter"]["ros__parameters"]
    assert console["robot_pose_topic"] == adapter["robot_pose_topic"]
    assert console["jog_command_topic"] == adapter["jog_command_topic"]
    assert console["jog_status_topic"] == adapter["status_topic"]
    assert adapter["target_pose_topic"] == "/topic_joint_impedance_controller/target_pose"
    assert adapter["target_pose_topic"] in console["record_topics"]
    assert adapter["status_topic"] in console["record_topics"]
