"""Static integration checks that do not require a GPU or a running ROS graph."""

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _parameters(node: str) -> dict:
    config = yaml.safe_load((ROOT / "config/deepusnav_console.yaml").read_text())
    return config[node]["ros__parameters"]


def test_atlas_topic_is_shared_and_recorded() -> None:
    console = _parameters("deepusnav_console")
    inference = _parameters("deepusnav_inference")
    topic = "/deepusnav/atlas/localisation"
    assert console["atlas_result_topic"] == topic
    assert inference["atlas_result_topic"] == topic
    assert topic in console["record_topics"]


def test_atlas_artifact_contract_is_configured() -> None:
    inference = _parameters("deepusnav_inference")
    assert inference["atlas_enabled"] is True
    assert inference["atlas_k"] == 8
    assert inference["atlas_target_tolerance_mm"] > 0
    assert inference["atlas_index_path"].endswith("__metric.pt")
    assert inference["atlas_target_path"].endswith("target_spine_cpr.json")


def test_ui_contains_complete_atlas_panel() -> None:
    tree = ET.parse(ROOT / "ui/deepusnav.ui")
    names = {element.attrib["name"] for element in tree.iter() if "name" in element.attrib}
    assert {
        "atlas_group",
        "atlas_status",
        "atlas_coordinate",
        "atlas_offset",
        "atlas_goal",
        "atlas_belief",
        "atlas_model",
        "atlas_contract",
    } <= names
