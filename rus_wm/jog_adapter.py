"""Guarded adapter from relative operator jog requests to FR3 pose targets."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String


def _finite_triplet(values: Sequence[float], label: str) -> tuple[float, float, float]:
    """Return three finite floats or raise a descriptive error."""
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain three finite numbers")
    return result


def normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate and normalize an XYZW quaternion."""
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ValueError("orientation must contain four finite numbers")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-9:
        raise ValueError("orientation quaternion has zero norm")
    return tuple(value / norm for value in quaternion)


def rotate_vector(
    quaternion_xyzw: Sequence[float], vector_xyz: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a vector by an XYZW quaternion without an external math dependency."""
    qx, qy, qz, qw = normalize_quaternion(quaternion_xyzw)
    vx, vy, vz = _finite_triplet(vector_xyz, "jog vector")
    # v' = v + 2*w*(q_xyz x v) + 2*(q_xyz x (q_xyz x v)).
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def target_from_jog(
    current_position: Sequence[float],
    current_orientation: Sequence[float],
    jog_delta: Sequence[float],
    command_frame: str,
    base_frame: str,
    tool_frame: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a base/tool-relative jog into an absolute base-frame pose target."""
    position = _finite_triplet(current_position, "current position")
    orientation = normalize_quaternion(current_orientation)
    delta = _finite_triplet(jog_delta, "jog vector")
    if command_frame == base_frame:
        base_delta = delta
    elif command_frame == tool_frame:
        base_delta = rotate_vector(orientation, delta)
    else:
        raise ValueError(
            f"unsupported jog frame '{command_frame}'; expected '{base_frame}' or "
            f"'{tool_frame}'"
        )
    return tuple(a + b for a, b in zip(position, base_delta)), orientation


class JogAdapter(Node):
    """Validate operator jogs and publish absolute targets for the impedance controller."""

    def __init__(self) -> None:
        super().__init__("deepusnav_jog_adapter")
        defaults = {
            "robot_pose_topic": "/fr3/current_pose",
            "jog_command_topic": "/deepusnav/operator/jog_command",
            "target_pose_topic": "/topic_joint_velocity_controller/target_pose",
            "status_topic": "/deepusnav/operator/jog_status",
            "base_frame": "fr3_link0",
            "tool_frame": "fr3_hand_tcp",
            "pose_timeout_sec": 0.5,
            "command_timeout_sec": 0.5,
            "max_jog_step_mm": 50.0,
            "min_command_interval_sec": 0.15,
            "workspace_min_xyz_m": [-0.8, -0.8, 0.05],
            "workspace_max_xyz_m": [0.8, 0.8, 1.0],
            "require_controller_subscriber": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.params = {name: self.get_parameter(name).value for name in defaults}
        self.workspace_min = _finite_triplet(
            self.params["workspace_min_xyz_m"], "workspace_min_xyz_m"
        )
        self.workspace_max = _finite_triplet(
            self.params["workspace_max_xyz_m"], "workspace_max_xyz_m"
        )
        if any(low >= high for low, high in zip(self.workspace_min, self.workspace_max)):
            raise ValueError("each workspace minimum must be below its maximum")

        self.current_pose: PoseStamped | None = None
        self.pose_received_at = -math.inf
        self.last_accepted_at = -math.inf

        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            PoseStamped,
            self.params["robot_pose_topic"],
            self._on_pose,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Vector3Stamped,
            self.params["jog_command_topic"],
            self._on_jog,
            command_qos,
        )
        self.target_publisher = self.create_publisher(
            PoseStamped, self.params["target_pose_topic"], qos_profile_sensor_data
        )
        self.status_publisher = self.create_publisher(
            String, self.params["status_topic"], command_qos
        )
        self.get_logger().info(
            f"jog adapter ready: {self.params['jog_command_topic']} -> "
            f"{self.params['target_pose_topic']}"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        self.current_pose = msg
        self.pose_received_at = time.monotonic()

    def _status(self, accepted: bool, reason: str, **details: object) -> None:
        payload = {"accepted": accepted, "reason": reason, **details}
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self.status_publisher.publish(message)
        if accepted:
            self.get_logger().info(reason)
        else:
            self.get_logger().warning(reason)

    def _reject(self, reason: str) -> None:
        self._status(False, reason)

    def _on_jog(self, msg: Vector3Stamped) -> None:
        now = time.monotonic()
        if self.current_pose is None:
            self._reject("Jog rejected: no robot pose received")
            return
        if now - self.pose_received_at > float(self.params["pose_timeout_sec"]):
            self._reject("Jog rejected: robot pose is stale")
            return
        if now - self.last_accepted_at < float(self.params["min_command_interval_sec"]):
            self._reject("Jog rejected: command interval is too short")
            return
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            self._reject("Jog rejected: command has no timestamp")
            return
        command_stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        command_age = (self.get_clock().now().nanoseconds - command_stamp_ns) / 1e9
        if command_age < -0.1 or command_age > float(self.params["command_timeout_sec"]):
            self._reject(f"Jog rejected: command timestamp age is {command_age:.3f} s")
            return
        if bool(self.params["require_controller_subscriber"]) and (
            self.target_publisher.get_subscription_count() == 0
        ):
            self._reject("Jog rejected: impedance controller is not subscribed")
            return

        delta = (msg.vector.x, msg.vector.y, msg.vector.z)
        try:
            delta = _finite_triplet(delta, "jog vector")
        except ValueError as exc:
            self._reject(f"Jog rejected: {exc}")
            return
        step_m = math.sqrt(sum(value * value for value in delta))
        max_step_m = float(self.params["max_jog_step_mm"]) / 1000.0
        if step_m <= 1e-9 or step_m > max_step_m + 1e-12:
            self._reject(
                f"Jog rejected: step {step_m * 1000.0:.3f} mm is outside (0, "
                f"{max_step_m * 1000.0:.3f}] mm"
            )
            return

        pose = self.current_pose.pose
        current_position = (pose.position.x, pose.position.y, pose.position.z)
        current_orientation = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        try:
            target_position, target_orientation = target_from_jog(
                current_position,
                current_orientation,
                delta,
                msg.header.frame_id,
                str(self.params["base_frame"]),
                str(self.params["tool_frame"]),
            )
        except ValueError as exc:
            self._reject(f"Jog rejected: {exc}")
            return
        if any(
            value < low or value > high
            for value, low, high in zip(
                target_position, self.workspace_min, self.workspace_max
            )
        ):
            self._reject(
                "Jog rejected: target is outside configured workspace "
                f"{tuple(round(value, 4) for value in target_position)}"
            )
            return

        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = str(self.params["base_frame"])
        target.pose.position.x, target.pose.position.y, target.pose.position.z = target_position
        (
            target.pose.orientation.x,
            target.pose.orientation.y,
            target.pose.orientation.z,
            target.pose.orientation.w,
        ) = target_orientation
        self.target_publisher.publish(target)
        self.last_accepted_at = now
        self._status(
            True,
            "Jog accepted and target pose published",
            command_frame=msg.header.frame_id,
            step_mm=round(step_m * 1000.0, 6),
            target_xyz_m=[round(value, 6) for value in target_position],
        )


def main(args=None) -> None:
    """Run the jog adapter until ROS shuts down."""
    rclpy.init(args=args)
    node = JogAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
