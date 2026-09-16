"""Publish cropped ultrasound frames from an OpenCV capture device in ROS 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import yaml


DEFAULT_ULTRASOUND_TOPIC = "/frame_grabber/us_img"


@dataclass(frozen=True)
class CaptureProfile:
    """Validated camera and crop settings loaded from a probe profile."""

    video_index: int
    width: int
    height: int
    rate_hz: float
    x0: int
    x1: int
    y0: int
    y1: int
    flip_horizontal: bool


def load_capture_profile(config_path: Path) -> CaptureProfile:
    """Load one legacy-compatible capture profile and validate its bounds."""
    with config_path.open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream) or {}

    try:
        size = raw["frame_size"]
        crop = raw["frame_cropped_coordinates"]
        profile = CaptureProfile(
            video_index=int(raw["video_index"]),
            width=int(size["width"]),
            height=int(size["height"]),
            rate_hz=float(raw["hz"]),
            x0=int(crop["x0"]),
            x1=int(crop["x1"]),
            y0=int(crop["y0"]),
            y1=int(crop["y1"]),
            flip_horizontal=bool(raw.get("flip_horizontal", False)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid capture profile {config_path}: {exc}") from exc

    if profile.video_index < 0:
        raise ValueError("video_index must be non-negative")
    if profile.width <= 0 or profile.height <= 0 or profile.rate_hz <= 0.0:
        raise ValueError("frame size and hz must be positive")
    if not (0 <= profile.x0 < profile.x1 <= profile.width):
        raise ValueError("crop x coordinates are outside the configured frame width")
    if not (0 <= profile.y0 < profile.y1 <= profile.height):
        raise ValueError("crop y coordinates are outside the configured frame height")
    return profile


class UltrasoundScreenPublisher(Node):
    """Capture, crop and publish ultrasound images with sensor-data QoS."""

    def __init__(self) -> None:
        super().__init__("us_screen_pub")
        self.declare_parameter("probe_type", "linear")
        self.declare_parameter("capture_profile", "")
        self.declare_parameter("output_topic", DEFAULT_ULTRASOUND_TOPIC)
        self.declare_parameter("frame_id", "ultrasound_frame")

        probe_type = str(self.get_parameter("probe_type").value).strip().lower()
        if probe_type not in {"linear", "convex"}:
            raise ValueError("probe_type must be 'linear' or 'convex'")

        configured_path = str(self.get_parameter("capture_profile").value).strip()
        if configured_path:
            profile_path = Path(configured_path).expanduser()
        else:
            package_share = Path(get_package_share_directory("rus_wm"))
            profile_path = package_share / "config" / f"screen_cap_config_{probe_type}.yaml"
        self.profile = load_capture_profile(profile_path)

        output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publisher = self.create_publisher(Image, output_topic, qos_profile_sensor_data)
        self.bridge = CvBridge()
        self.capture = cv2.VideoCapture(self.profile.video_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.profile.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.profile.height)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                f"cannot open video device index {self.profile.video_index}; "
                "check /dev/video* and device permissions"
            )

        self.timer = self.create_timer(1.0 / self.profile.rate_hz, self._publish_frame)
        self.get_logger().info(
            f"Publishing {probe_type} ultrasound from video{self.profile.video_index} "
            f"to {output_topic} at {self.profile.rate_hz:g} Hz"
        )

    def _publish_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.get_logger().warning("failed to read ultrasound frame", throttle_duration_sec=2.0)
            return

        height, width = frame.shape[:2]
        if self.profile.x1 > width or self.profile.y1 > height:
            self.get_logger().error(
                f"captured frame is {width}x{height}, smaller than crop ending at "
                f"({self.profile.x1}, {self.profile.y1})",
                throttle_duration_sec=2.0,
            )
            return

        cropped = frame[
            self.profile.y0:self.profile.y1,
            self.profile.x0:self.profile.x1,
        ]
        if self.profile.flip_horizontal:
            cropped = cv2.flip(cropped, 1)

        message = self.bridge.cv2_to_imgmsg(cropped, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        self.publisher.publish(message)

    def destroy_node(self) -> None:
        """Release the camera before shutting down the ROS node."""
        self.capture.release()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    """Run the ultrasound screen publisher."""
    rclpy.init(args=args)
    node: UltrasoundScreenPublisher | None = None
    try:
        node = UltrasoundScreenPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
