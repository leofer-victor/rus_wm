"""
ROS 2 inference worker for DeepUSNav world models and population-atlas localisation.

The worker deliberately publishes policy-space proposals on its own topic.  It never
publishes Cartesian jog commands: converting the SonoGym surface action ``(dx, dz,
dalpha)`` to an FR3 motion requires a calibrated probe/surface adapter and safety checks
on the controller computer.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool

from .atlas_runtime import AtlasRuntime
from .model_profiles import resolve_model_profile


def _sensor_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class DeepUSNavInference(Node):
    """Load selected model components and run guarded inference on the latest frame."""

    def __init__(self) -> None:
        super().__init__("deepusnav_inference")
        default_root = Path.home() / "projects" / "python_projects" / "deepusnav"
        defaults = {
            "deepusnav_root": str(default_root),
            "model_profile": "dino",
            # Empty selects the checkpoint belonging to model_profile.
            "checkpoint_path": "",
            "goal_image_path": "",
            # Used only by the fully manual custom profile.
            "atlas_enabled": False,
            "atlas_index_path": str(default_root.joinpath(
                "training_setup_and_weights/atlas/heads/",
                "vjepa2_vitl__grid4x4__metric.pt",
            )),
            "atlas_target_path": str(
                default_root / "training_setup_and_weights/atlas/target_spine_cpr.json"
            ),
            "atlas_result_topic": "/deepusnav/atlas/localisation",
            "atlas_k": 8,
            "atlas_target_tolerance_mm": 20.0,
            "device": "cuda",
            "ultrasound_topic": "/frame_grabber/us_img",
            "robot_pose_topic": "/fr3/current_pose",
            "status_topic": "/deepusnav/inference/status",
            "proposal_topic": "/deepusnav/inference/action_proposal",
            "enable_service": "/deepusnav/inference/set_enabled",
            "state_timeout_sec": 0.5,
            "image_timeout_sec": 1.0,
            "inference_rate_hz": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.params = {name: self.get_parameter(name).value for name in defaults}
        self.model_profile = resolve_model_profile(
            self.params["model_profile"],
            self.params["deepusnav_root"],
            self.params["checkpoint_path"],
            bool(self.params["atlas_enabled"]),
        )

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._image: np.ndarray | None = None
        self._pose: PoseStamped | None = None
        self._received_at: dict[str, float] = {}
        self._enabled = False
        self._busy = False
        self._last_wait_reason = "disabled"

        callbacks = ReentrantCallbackGroup()
        qos = _sensor_qos()
        self.create_subscription(
            Image, self.params["ultrasound_topic"], self._on_image, qos,
            callback_group=callbacks,
        )
        self.create_subscription(
            PoseStamped, self.params["robot_pose_topic"], self._on_pose, qos,
            callback_group=callbacks,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_publisher = self.create_publisher(
            String, self.params["status_topic"], status_qos
        )
        self.proposal_publisher = self.create_publisher(
            Float32MultiArray, self.params["proposal_topic"], 10
        )
        self.atlas_publisher = self.create_publisher(
            String, self.params["atlas_result_topic"], status_qos
        )
        self.create_service(
            SetBool, self.params["enable_service"], self._set_enabled,
            callback_group=callbacks,
        )

        self.world_model = None
        self.goal_latent = None
        self.device = str(self.params["device"])
        if self.model_profile.world_model_enabled:
            self._publish_status(f"loading {self.model_profile.name} world model")
            self._load_model(self.model_profile.checkpoint_path)
        self.atlas = None
        if self.model_profile.atlas_enabled:
            self._publish_status("loading population atlas")
            encoder_cache = {}
            if self.model_profile.world_model_enabled:
                encoder_cache[self.encoder_name] = (
                    self.encoder,
                    self.input_size,
                    self.patch,
                    self.feature_key,
                )
            self.atlas = AtlasRuntime(
                deepusnav_root=self.params["deepusnav_root"],
                index_path=self.params["atlas_index_path"],
                target_path=self.params["atlas_target_path"],
                device=self.device,
                k=int(self.params["atlas_k"]),
                target_tolerance_mm=float(self.params["atlas_target_tolerance_mm"]),
                encoder_cache=encoder_cache,
            )
            self.device = self.atlas.device
            info = String()
            info.data = json.dumps({"status": "ready", **self.atlas.model_info()})
            self.atlas_publisher.publish(info)
        period = 1.0 / max(0.1, float(self.params["inference_rate_hz"]))
        self.create_timer(period, self._tick, callback_group=callbacks)
        if self.model_profile.world_model_enabled:
            mode = "goal-directed" if self.goal_latent is not None else "shadow"
        else:
            mode = "atlas-only"
        self._publish_status(
            f"ready (profile={self.model_profile.name}, {mode}, {self.device})"
        )

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_publisher.publish(msg)
        self.get_logger().info(text)

    def _load_model(self, checkpoint_path: Path | None) -> None:
        root = Path(str(self.params["deepusnav_root"])).expanduser().resolve()
        if checkpoint_path is None:
            raise ValueError("world-model profile did not resolve a checkpoint")
        checkpoint = checkpoint_path
        if not (root / "src" / "deepusnav").is_dir():
            raise FileNotFoundError(f"DeepUSNav source package not found under {root}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

        os.environ.setdefault("TORCH_HOME", str(root / "checkpoints" / "torch"))
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(root))
        os.chdir(root)

        import torch
        from models.world_model import LatentWorldModel, load_weights
        from models.world_model_data import ActionStats

        requested = str(self.params["device"])
        if requested.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warning("CUDA is unavailable; falling back to CPU")
            requested = "cpu"
        self.torch = torch
        self.device = requested

        blob = torch.load(checkpoint, map_location=requested, weights_only=False)
        self.world_model = LatentWorldModel(**blob["config"]).to(requested)
        load_weights(self.world_model, blob["state_dict"])
        self.world_model.eval()
        self.action_stats = ActionStats.from_dict(blob["action_stats"])
        self.encoder_name = str(blob["encoder"])

        loader_path = root / "scripts" / "data" / "cache_latents.py"
        spec = importlib.util.spec_from_file_location("_deepusnav_encoder_loader", loader_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import encoder loader: {loader_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.encoder, self.input_size, self.patch, self.feature_key = module.load_encoder(
            self.encoder_name, requested
        )

        # Seven deterministic proposals: hold, two surface translations in both signs,
        # and probe rotation in both signs. Values are in the policy's [-1, 1] space.
        self.candidates = np.asarray(
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
             [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float32,
        )
        command_actions = self.candidates * np.asarray([2.0, 2.0, 0.1], np.float32)
        self.normalized_actions = self.action_stats.normalize(command_actions)

        goal_path = str(self.params["goal_image_path"]).strip()
        self.goal_latent = None
        if goal_path:
            goal = cv2.imread(str(Path(goal_path).expanduser()), cv2.IMREAD_GRAYSCALE)
            if goal is None:
                raise FileNotFoundError(f"goal image is unreadable: {goal_path}")
            self.goal_latent = self._encode(goal)

    def _on_image(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            image = np.ascontiguousarray(image)
        except (CvBridgeError, cv2.error, ValueError) as exc:
            self.get_logger().error(f"ultrasound conversion failed: {exc}")
            return
        with self._lock:
            self._image = image
            self._received_at["image"] = time.monotonic()

    def _on_pose(self, msg: PoseStamped) -> None:
        with self._lock:
            self._pose = msg
            self._received_at["pose"] = time.monotonic()

    @staticmethod
    def _pose_valid(msg: PoseStamped | None) -> bool:
        if msg is None:
            return False
        p, q = msg.pose.position, msg.pose.orientation
        values = np.asarray([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=float)
        q_norm = float(np.linalg.norm(values[3:]))
        return bool(np.isfinite(values).all() and 0.5 <= q_norm <= 1.5)

    def _snapshot(self) -> tuple[np.ndarray | None, str | None]:
        now = time.monotonic()
        with self._lock:
            image = None if self._image is None else self._image.copy()
            pose = self._pose
            image_age = now - self._received_at.get("image", -math.inf)
            pose_age = now - self._received_at.get("pose", -math.inf)
        if image is None or image_age > float(self.params["image_timeout_sec"]):
            return None, "waiting for a fresh ultrasound image"
        if pose_age > float(self.params["state_timeout_sec"]):
            return None, "waiting for a fresh robot pose"
        if not self._pose_valid(pose):
            return None, "robot pose contains invalid values"
        return image, None

    def _set_enabled(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            _, reason = self._snapshot()
            if reason is not None:
                response.success = False
                response.message = f"inference not started: {reason}"
                return response
            self._enabled = True
            response.success = True
            response.message = "inference enabled"
            self._publish_status("running")
        else:
            self._enabled = False
            response.success = True
            response.message = "inference disabled"
            self._publish_status("ready (disabled)")
        return response

    def _encode(self, image: np.ndarray):
        from models.world_model_data import as_unit_interval, letterbox

        unit = as_unit_interval(image)
        x = letterbox(self.torch.from_numpy(unit[None]), self.input_size, self.patch)
        with self.torch.inference_mode():
            return self.encoder.forward_features(x.to(self.device))[self.feature_key][0].float()

    def _infer(self, image: np.ndarray) -> tuple[np.ndarray | None, float]:
        started = time.perf_counter()
        z = self._encode(image)
        count = len(self.candidates)
        actions = self.torch.from_numpy(self.normalized_actions).to(self.device)
        actions = actions[:, None].expand(-1, 2, -1)
        window = z[None, None].expand(count, 2, -1, -1)
        with self.torch.inference_mode():
            # Use the predictor path only. Calling ``world_model.forward`` would also
            # construct a training target and, for V-JEPA2-AC recipes, its rollout loss.
            # Neither belongs in online one-step inference.
            assembled = self.world_model.assemble(window, actions)
            predicted = self.world_model.predictor(assembled[:, :-1])
            predicted = self.world_model._norm(
                predicted[..., :self.world_model.feature_dim]
            )[:, -1]
            proposal = None
            if self.goal_latent is not None:
                goal = self.world_model._norm(self.goal_latent).unsqueeze(0)
                if self.world_model.loss_kind == "l1":
                    scores = (predicted - goal).abs().mean(dim=(1, 2))
                else:
                    scores = ((predicted - goal) ** 2).mean(dim=(1, 2))
                proposal = self.candidates[int(scores.argmin().item())]
        return proposal, (time.perf_counter() - started) * 1000.0

    def _tick(self) -> None:
        if not self._enabled or self._busy:
            return
        image, reason = self._snapshot()
        if reason is not None:
            if reason != self._last_wait_reason:
                self._publish_status(f"paused: {reason}")
                self._last_wait_reason = reason
            return
        self._last_wait_reason = ""
        self._busy = True
        try:
            proposal = None
            world_model_ms = None
            if self.model_profile.world_model_enabled:
                proposal, world_model_ms = self._infer(image)
            atlas_ms = None
            if self.atlas is not None:
                atlas_result = self.atlas.localise(image)
                atlas_result["stamp"] = self.get_clock().now().nanoseconds / 1e9
                atlas_msg = String()
                atlas_msg.data = json.dumps(atlas_result, separators=(",", ":"))
                self.atlas_publisher.publish(atlas_msg)
                atlas_ms = float(atlas_result["latency_ms"])
            if proposal is None:
                timing_parts = []
                if world_model_ms is not None:
                    timing_parts.append(f"WM {world_model_ms:.0f} ms")
                if atlas_ms is not None:
                    timing_parts.append(f"Atlas {atlas_ms:.0f} ms")
                timing = " | ".join(timing_parts)
                run_mode = (
                    "atlas" if not self.model_profile.world_model_enabled else "shadow"
                )
                self._publish_status(
                    f"running {run_mode} | profile={self.model_profile.name} | {timing}"
                )
                return
            msg = Float32MultiArray()
            msg.data = [float(v) for v in proposal]
            self.proposal_publisher.publish(msg)
            action = ", ".join(f"{v:+.2f}" for v in proposal)
            timing = f"WM {world_model_ms:.0f} ms"
            if atlas_ms is not None:
                timing += f" | Atlas {atlas_ms:.0f} ms"
            self._publish_status(f"running | action [{action}] | {timing}")
        except Exception as exc:
            self._enabled = False
            self._publish_status(f"error: {type(exc).__name__}: {exc}")
            self.get_logger().error(f"inference failed: {exc}")
        finally:
            self._busy = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeepUSNavInference()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
