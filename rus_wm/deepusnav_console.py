"""
ROS 2 / Qt operator console for DeepUSNav FR3 experiments.

This node is deliberately a low-rate supervisor.  A jog message represents one bounded
relative displacement request; interpolation, force/contact control, watchdogs and hard
safety limits belong to the real-time controller computer.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, Vector3Stamped, WrenchStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QMainWindow

import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkOpenGLGPUVolumeRayCastMapper

try:
    from ui.Ui_deepusnav import Ui_DeepUSNavConsole
except ImportError as exc:  # generated from ui/deepusnav.ui by the developer
    raise ImportError(
        "Generate the Qt binding first: pyside6-uic ui/deepusnav.ui "
        "-o ui/Ui_deepusnav.py"
    ) from exc


def _sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class DeepUSNavRosNode(Node):
    """ROS-facing state cache and low-rate command endpoint."""

    def __init__(self) -> None:
        super().__init__("deepusnav_console")
        defaults = {
            "ultrasound_topic": "/deepusnav/ultrasound/image",
            "robot_pose_topic": "/fr3/current_pose",
            "robot_joint_state_topic": "/fr3/joint_states",
            "robot_wrench_topic": "/fr3/state/external_wrench",
            "robot_mode_topic": "/fr3/state/mode",
            "inference_status_topic": "/deepusnav/inference/status",
            "atlas_result_topic": "/deepusnav/atlas/localisation",
            "inference_enable_service": "/deepusnav/inference/set_enabled",
            "jog_command_topic": "/deepusnav/operator/jog_command",
            "stop_service": "/fr3/operator/stop",
            "reset_service": "/fr3/operator/reset",
            "base_frame": "fr3_link0",
            "tool_frame": "fr3_hand_tcp",
            "state_timeout_sec": 0.5,
            "image_timeout_sec": 1.0,
            "default_jog_step_mm": 1.0,
            "max_jog_step_mm": 5.0,
            "command_cooldown_sec": 0.15,
            "cbct_directory": "",
            "bag_directory": "~/deepusnav_bags",
            "record_topics": [
                "/deepusnav/ultrasound/image",
                "/fr3/state/current_pose",
                "/fr3/state/joint_states",
                "/fr3/state/external_wrench",
                "/fr3/state/mode",
                "/deepusnav/inference/status",
                "/deepusnav/operator/jog_command",
            ],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.params = {name: self.get_parameter(name).value for name in defaults}

        self.bridge = CvBridge()
        self.ultrasound_rgb: np.ndarray | None = None
        self.robot_pose: PoseStamped | None = None
        self.joint_state: JointState | None = None
        self.robot_wrench: WrenchStamped | None = None
        self.robot_mode = "unknown"
        self.inference_status = "not connected"
        self.atlas_result: dict | None = None
        self.received_at: dict[str, float] = {}
        self.image_times: list[float] = []

        qos = _sensor_qos()
        self.create_subscription(Image, self.params["ultrasound_topic"], self._on_image, qos)
        self.create_subscription(PoseStamped, self.params["robot_pose_topic"], self._on_pose, qos)
        self.create_subscription(
            JointState, self.params["robot_joint_state_topic"], self._on_joints, qos
        )
        self.create_subscription(
            WrenchStamped, self.params["robot_wrench_topic"], self._on_wrench, qos
        )
        self.create_subscription(String, self.params["robot_mode_topic"], self._on_mode, qos)
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, self.params["inference_status_topic"], self._on_inference, status_qos
        )
        self.create_subscription(
            String, self.params["atlas_result_topic"], self._on_atlas, status_qos
        )

        self.jog_publisher = self.create_publisher(
            Vector3Stamped, self.params["jog_command_topic"], 10
        )
        self.stop_client = self.create_client(Trigger, self.params["stop_service"])
        self.reset_client = self.create_client(Trigger, self.params["reset_service"])
        self.inference_client = self.create_client(
            SetBool, self.params["inference_enable_service"]
        )
        self.last_command_at = 0.0

    def _stamp(self, key: str) -> None:
        self.received_at[key] = time.monotonic()

    def age(self, key: str) -> float:
        received = self.received_at.get(key)
        return math.inf if received is None else time.monotonic() - received

    def _on_image(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif msg.encoding.lower().startswith("rgb"):
                rgb = image
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ultrasound_rgb = np.ascontiguousarray(rgb)
            now = time.monotonic()
            self.image_times = [t for t in self.image_times if now - t <= 2.0]
            self.image_times.append(now)
            self._stamp("image")
        except (CvBridgeError, cv2.error, ValueError) as exc:
            self.get_logger().error(f"ultrasound conversion failed: {exc}")

    def _on_pose(self, msg: PoseStamped) -> None:
        self.robot_pose = msg
        self._stamp("pose")

    def _on_joints(self, msg: JointState) -> None:
        self.joint_state = msg
        self._stamp("joints")

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self.robot_wrench = msg
        self._stamp("wrench")

    def _on_mode(self, msg: String) -> None:
        self.robot_mode = msg.data or "unknown"
        self._stamp("mode")

    def _on_inference(self, msg: String) -> None:
        self.inference_status = msg.data or "unknown"
        self._stamp("inference")

    def _on_atlas(self, msg: String) -> None:
        try:
            result = json.loads(msg.data)
            if not isinstance(result, dict) or "status" not in result:
                raise ValueError("expected a JSON object with a status field")
            self.atlas_result = result
            self._stamp("atlas")
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f"invalid Atlas result: {exc}")

    @property
    def image_rate_hz(self) -> float:
        if len(self.image_times) < 2:
            return 0.0
        elapsed = self.image_times[-1] - self.image_times[0]
        return (len(self.image_times) - 1) / elapsed if elapsed > 0.0 else 0.0

    def publish_jog(self, axis: int, sign: int, step_mm: float, frame_id: str) -> tuple[bool, str]:
        now = time.monotonic()
        if self.age("pose") > float(self.params["state_timeout_sec"]):
            return False, "Jog rejected: robot pose is stale"
        if step_mm <= 0.0 or step_mm > float(self.params["max_jog_step_mm"]):
            return False, "Jog rejected: step is outside configured limits"
        if now - self.last_command_at < float(self.params["command_cooldown_sec"]):
            return False, "Jog rejected: command cooldown"

        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        delta = [0.0, 0.0, 0.0]
        delta[axis] = sign * step_mm / 1000.0
        msg.vector.x, msg.vector.y, msg.vector.z = delta
        self.jog_publisher.publish(msg)
        self.last_command_at = now
        return True, f"Jog sent: {frame_id} {['X', 'Y', 'Z'][axis]} {sign * step_mm:+.1f} mm"

    def call_trigger(self, which: str, done) -> tuple[bool, str]:
        client = self.stop_client if which == "stop" else self.reset_client
        if not client.service_is_ready():
            return False, f"{which} service is unavailable"
        future = client.call_async(Trigger.Request())
        future.add_done_callback(lambda result: done(which, result))
        return True, f"{which} requested"

    @staticmethod
    def _pose_is_valid(msg: PoseStamped | None) -> bool:
        if msg is None:
            return False
        p, q = msg.pose.position, msg.pose.orientation
        values = np.asarray([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=float)
        q_norm = float(np.linalg.norm(values[3:]))
        return bool(np.isfinite(values).all() and 0.5 <= q_norm <= 1.5)

    def inference_readiness(self) -> tuple[bool, str]:
        if self.ultrasound_rgb is None or self.age("image") > float(
            self.params["image_timeout_sec"]
        ):
            return False, "Inference requires a fresh ultrasound image"
        if self.age("pose") > float(self.params["state_timeout_sec"]):
            return False, "Inference requires a fresh robot pose"
        if not self._pose_is_valid(self.robot_pose):
            return False, "Inference rejected: robot pose contains invalid values"
        return True, "Inference inputs are ready"

    def set_inference_enabled(self, enabled: bool, done) -> tuple[bool, str]:
        if enabled:
            ready, message = self.inference_readiness()
            if not ready:
                return False, message
        if not self.inference_client.service_is_ready():
            return False, "Inference node is not ready"
        request = SetBool.Request()
        request.data = enabled
        future = self.inference_client.call_async(request)
        future.add_done_callback(done)
        return True, "Starting inference..." if enabled else "Stopping inference..."


class DeepUSNavConsole(QMainWindow):
    def __init__(self, ros_node: DeepUSNavRosNode) -> None:
        super().__init__()
        self.ros = ros_node
        self.ui = Ui_DeepUSNavConsole()
        self.ui.setupUi(self)

        self.ui.jog_step_mm.setMaximum(float(self.ros.params["max_jog_step_mm"]))
        self.ui.jog_step_mm.setValue(float(self.ros.params["default_jog_step_mm"]))
        self.ui.jog_frame.addItem("Base", self.ros.params["base_frame"])
        self.ui.jog_frame.addItem("Probe / tool", self.ros.params["tool_frame"])

        self.image_scene = QGraphicsScene(self)
        self.ui.us_view.setScene(self.image_scene)
        self.image_item = self.image_scene.addPixmap(QPixmap())

        self._init_volume_view()
        self.bag_process: subprocess.Popen | None = None

        self.ui.load_cbct.clicked.connect(self.load_cbct)
        self.ui.record.clicked.connect(self.start_recording)
        self.ui.stop_record.clicked.connect(self.stop_recording)
        self.ui.start_inference.clicked.connect(lambda: self.set_inference(True))
        self.ui.stop_inference.clicked.connect(lambda: self.set_inference(False))
        self.ui.stop_robot.clicked.connect(lambda: self.call_service("stop"))
        self.ui.reset_robot.clicked.connect(lambda: self.call_service("reset"))
        self.ui.quit.clicked.connect(self.close)
        self.ui.jog_x_pos.clicked.connect(lambda: self.jog(0, 1))
        self.ui.jog_x_neg.clicked.connect(lambda: self.jog(0, -1))
        self.ui.jog_y_pos.clicked.connect(lambda: self.jog(1, 1))
        self.ui.jog_y_neg.clicked.connect(lambda: self.jog(1, -1))
        self.ui.jog_z_pos.clicked.connect(lambda: self.jog(2, 1))
        self.ui.jog_z_neg.clicked.connect(lambda: self.jog(2, -1))

        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(lambda: rclpy.spin_once(self.ros, timeout_sec=0.0))
        self.ros_timer.start(10)
        self.display_timer = QTimer(self)
        self.display_timer.timeout.connect(self.refresh)
        self.display_timer.start(50)

        initial_cbct = str(self.ros.params["cbct_directory"])
        if initial_cbct and Path(initial_cbct).expanduser().is_dir():
            self.display_cbct(Path(initial_cbct).expanduser())

    def _init_volume_view(self) -> None:
        self.volume_widget = QVTKRenderWindowInteractor(self)
        self.ui.vtk_volume.addWidget(self.volume_widget)
        self.volume_renderer = vtkRenderer()
        self.volume_renderer.SetBackground(0.08, 0.08, 0.08)
        self.volume_widget.GetRenderWindow().AddRenderer(self.volume_renderer)
        interactor = self.volume_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.volume_widget.Initialize()

    @staticmethod
    def _rpy_degrees(q) -> tuple[float, float, float]:
        x, y, z, w = q.x, q.y, q.z, q.w
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return tuple(math.degrees(v) for v in (roll, pitch, yaw))

    def refresh(self) -> None:
        image_timeout = float(self.ros.params["image_timeout_sec"])
        image_age = self.ros.age("image")
        if self.ros.ultrasound_rgb is not None:
            image = self.ros.ultrasound_rgb
            height, width = image.shape[:2]
            qimage = QImage(
                image.data, width, height, image.strides[0], QImage.Format.Format_RGB888
            )
            self.image_item.setPixmap(QPixmap.fromImage(qimage.copy()))
            self.ui.us_view.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.ui.us_status.setText(
            f"US: {self.ros.image_rate_hz:.1f} Hz | age {image_age:.2f} s"
            if math.isfinite(image_age) else "US: waiting"
        )
        self.ui.us_status.setStyleSheet(
            "color: #55dd88" if image_age <= image_timeout else "color: #ff6666"
        )

        pose = self.ros.robot_pose
        if pose is None:
            self.ui.robot_pose.setText("waiting for robot pose")
        else:
            p, q = pose.pose.position, pose.pose.orientation
            rpy = self._rpy_degrees(q)
            self.ui.robot_pose.setText(
                f"xyz [mm]  {p.x * 1000:8.2f}  {p.y * 1000:8.2f}  {p.z * 1000:8.2f}\n"
                f"rpy [deg] {rpy[0]:8.2f}  {rpy[1]:8.2f}  {rpy[2]:8.2f}"
            )

        joints = self.ros.joint_state
        self.ui.joint_state.setText(
            "q [rad] " + "  ".join(f"{q:+.3f}" for q in joints.position[:7])
            if joints is not None else "q: waiting"
        )
        wrench = self.ros.robot_wrench
        if wrench is None:
            self.ui.wrench_state.setText("External wrench: waiting")
        else:
            f, t = wrench.wrench.force, wrench.wrench.torque
            self.ui.wrench_state.setText(
                f"F [N] {f.x:+.2f} {f.y:+.2f} {f.z:+.2f}   "
                f"T [Nm] {t.x:+.2f} {t.y:+.2f} {t.z:+.2f}"
            )

        pose_age = self.ros.age("pose")
        self.ui.robot_status.setText(
            f"Robot: {self.ros.robot_mode} | pose age {pose_age:.2f} s"
            if math.isfinite(pose_age) else "Robot: disconnected"
        )
        self.ui.robot_status.setStyleSheet(
            "color: #55dd88"
            if pose_age <= float(self.ros.params["state_timeout_sec"])
            else "color: #ff6666"
        )
        self.ui.inference_status.setText(f"Inference: {self.ros.inference_status}")
        self._refresh_atlas()

    def _refresh_atlas(self) -> None:
        result = self.ros.atlas_result
        age = self.ros.age("atlas")
        if result is None:
            self.ui.atlas_status.setText("Atlas: waiting for inference node")
            self.ui.atlas_status.setStyleSheet("color: #ff6666; font-weight: bold;")
            self.ui.atlas_coordinate.setText("u-hat: --")
            self.ui.atlas_offset.setText("offset to L4 [mm]: --")
            self.ui.atlas_goal.setText("L4 goal cost: --")
            self.ui.atlas_belief.setText("retrieval belief: --")
            self.ui.atlas_model.setText("model: --")
            return

        model = (
            f"{result.get('encoder', '?')} / {result.get('representation', '?')} | "
            f"{result.get('canonical_frame', '?')} | bank {result.get('bank_size', 0):,}"
        )
        self.ui.atlas_model.setText(f"model: {model}")
        if result.get("status") != "ok":
            self.ui.atlas_status.setText(f"Atlas: {result.get('status', 'unknown')}")
            self.ui.atlas_status.setStyleSheet("color: #d6b66b; font-weight: bold;")
            return

        stale = age > float(self.ros.params["image_timeout_sec"]) * 2.0
        latency = float(result.get("latency_ms", float("nan")))
        self.ui.atlas_status.setText(
            f"Atlas: {'stale' if stale else 'live'} | age {age:.2f} s | {latency:.0f} ms"
        )
        self.ui.atlas_status.setStyleSheet(
            "color: #ff6666; font-weight: bold;"
            if stale else "color: #55dd88; font-weight: bold;"
        )

        coordinate = result.get("coordinate_u", [float("nan")] * 3)
        offset = result.get("offset_to_target_mm", [float("nan")] * 3)
        self.ui.atlas_coordinate.setText(
            "u-hat [canonical]  " + "  ".join(f"{float(v):+.4f}" for v in coordinate)
        )
        self.ui.atlas_offset.setText(
            "offset to L4 [mm]  " + "  ".join(f"{float(v):+.1f}" for v in offset)
        )

        expected = float(result.get("expected_goal_distance_mm", float("nan")))
        probability = 100.0 * float(result.get("target_probability", float("nan")))
        tolerance = float(result.get("target_tolerance_mm", float("nan")))
        inside = bool(result.get("within_target", False))
        self.ui.atlas_goal.setText(
            f"L4 expected cost {expected:.1f} mm | posterior within "
            f"{tolerance:g} mm: {probability:.0f}%"
        )
        self.ui.atlas_goal.setStyleSheet(
            "color: #55dd88; font-weight: bold;"
            if inside else "color: #d6b66b; font-weight: bold;"
        )

        spread = float(result.get("neighbour_spread_mm", float("nan")))
        entropy = float(result.get("entropy_bits", float("nan")))
        entropy_max = float(result.get("entropy_bits_max", float("nan")))
        patients = result.get("distinct_patients")
        patient_text = "--" if patients is None else str(patients)
        self.ui.atlas_belief.setText(
            f"neighbour spread {spread:.1f} mm | entropy {entropy:.2f}/{entropy_max:.2f} bit "
            f"| patients {patient_text}/{result.get('k', '?')}"
        )

    def set_note(self, text: str, ok: bool | None = None) -> None:
        self.ui.notes.setText(text)
        if ok is not None:
            self.ui.notes.setStyleSheet("color: #55dd88" if ok else "color: #ff6666")

    def jog(self, axis: int, sign: int) -> None:
        if not self.ui.enable_jog.isChecked():
            self.set_note("Jog is locked. Enable manual jog first.", False)
            return
        frame_id = str(self.ui.jog_frame.currentData())
        ok, message = self.ros.publish_jog(axis, sign, self.ui.jog_step_mm.value(), frame_id)
        self.set_note(message, ok)

    def call_service(self, which: str) -> None:
        if which == "reset" and not self.ui.enable_jog.isChecked():
            self.set_note("Reset rejected: enable manual jog first.", False)
            return
        ok, message = self.ros.call_trigger(which, self._service_done)
        self.set_note(message, ok)
        if which == "stop":
            self.ui.enable_jog.setChecked(False)

    def set_inference(self, enabled: bool) -> None:
        ok, message = self.ros.set_inference_enabled(enabled, self._inference_done)
        self.set_note(message, ok)
        if ok:
            self.ui.start_inference.setEnabled(False)
            self.ui.stop_inference.setEnabled(False)

    def _inference_done(self, future) -> None:
        try:
            response = future.result()
            self.set_note(response.message, bool(response.success))
            running = bool(response.success and "enabled" in response.message)
            self.ui.start_inference.setEnabled(not running)
            self.ui.stop_inference.setEnabled(running)
        except Exception as exc:
            self.set_note(f"Inference service failed: {exc}", False)
            self.ui.start_inference.setEnabled(True)
            self.ui.stop_inference.setEnabled(False)

    def _service_done(self, which: str, future) -> None:
        try:
            response = future.result()
            self.set_note(f"{which}: {response.message}", bool(response.success))
        except Exception as exc:  # ROS future reports transport errors here
            self.set_note(f"{which} failed: {exc}", False)

    def load_cbct(self) -> None:
        start = str(self.ros.params["cbct_directory"] or Path.home())
        selected = QFileDialog.getExistingDirectory(self, "Select CBCT DICOM directory", start)
        if selected:
            self.display_cbct(Path(selected))

    def display_cbct(self, directory: Path) -> None:
        self.set_note("Loading CBCT...", None)
        QApplication.processEvents()
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(str(directory))
        reader.Update()
        data = reader.GetOutput()
        if data is None or data.GetNumberOfPoints() == 0:
            self.set_note("CBCT load failed: no readable DICOM voxels", False)
            return

        mapper = vtkOpenGLGPUVolumeRayCastMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        color = vtkColorTransferFunction()
        color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
        color.AddRGBPoint(-200, 0.08, 0.08, 0.08)
        color.AddRGBPoint(80, 0.55, 0.35, 0.28)
        color.AddRGBPoint(300, 0.92, 0.86, 0.72)
        color.AddRGBPoint(1200, 1.0, 1.0, 1.0)
        opacity = vtkPiecewiseFunction()
        opacity.AddPoint(-1000, 0.0)
        opacity.AddPoint(-200, 0.0)
        opacity.AddPoint(80, 0.015)
        opacity.AddPoint(300, 0.08)
        opacity.AddPoint(1200, 0.55)
        properties = vtkVolumeProperty()
        properties.SetColor(color)
        properties.SetScalarOpacity(opacity)
        properties.SetInterpolationTypeToLinear()
        properties.ShadeOn()
        volume = vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(properties)

        self.volume_renderer.RemoveAllViewProps()
        self.volume_renderer.AddVolume(volume)
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(100.0, 100.0, 100.0)
        self.volume_renderer.AddActor(axes)
        self.volume_renderer.ResetCamera()
        self.volume_widget.GetRenderWindow().Render()
        self._cbct_objects = (reader, mapper, properties, volume, axes)
        self.ui.cbct_path.setText(str(directory))
        self.set_note("CBCT loaded", True)

    def start_recording(self) -> None:
        if self.bag_process is not None and self.bag_process.poll() is None:
            self.set_note("rosbag2 is already recording", False)
            return
        root = Path(os.path.expanduser(str(self.ros.params["bag_directory"])))
        root.mkdir(parents=True, exist_ok=True)
        destination = root / time.strftime("deepusnav_%Y%m%d_%H%M%S")
        topics = [str(topic) for topic in self.ros.params["record_topics"]]
        command = ["ros2", "bag", "record", "-o", str(destination), *topics]
        try:
            self.bag_process = subprocess.Popen(command)
            self.ui.record.setEnabled(False)
            self.ui.stop_record.setEnabled(True)
            self.set_note(f"Recording: {destination}", True)
        except OSError as exc:
            self.set_note(f"Could not start rosbag2: {exc}", False)

    def stop_recording(self) -> None:
        if self.bag_process is None or self.bag_process.poll() is not None:
            self.set_note("rosbag2 is not running", False)
            return
        self.bag_process.send_signal(signal.SIGINT)
        try:
            self.bag_process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            self.bag_process.terminate()
        self.bag_process = None
        self.ui.record.setEnabled(True)
        self.ui.stop_record.setEnabled(False)
        self.set_note("Recording stopped and metadata finalized", True)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.bag_process is not None and self.bag_process.poll() is None:
            self.stop_recording()
        self.ui.enable_jog.setChecked(False)
        self.ros.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        event.accept()


def main(args=None) -> None:
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    window = DeepUSNavConsole(DeepUSNavRosNode())
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
