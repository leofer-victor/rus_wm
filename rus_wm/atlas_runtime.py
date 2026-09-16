"""Runtime population-atlas localisation for one ultrasound frame.

This module is ROS-independent so the model contract can be tested without importing
``rclpy`` or Qt.  It deliberately exposes anatomical estimates only: canonical atlas
coordinates are not robot Cartesian coordinates and must never be sent to a controller.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class AtlasRuntime:
    """Frozen V-JEPA2 encoder + learned metric head + population retrieval bank."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        deepusnav_root: str | Path,
        index_path: str | Path,
        target_path: str | Path,
        device: str = "cuda",
        k: int = 8,
        target_tolerance_mm: float = 20.0,
        encoder_cache: dict[str, tuple[Any, int, int, Any]] | None = None,
    ) -> None:
        self.root = Path(deepusnav_root).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.target_path = Path(target_path).expanduser().resolve()
        self.device = device
        self.k = int(k)
        self.target_tolerance_mm = float(target_tolerance_mm)

        if not (self.root / "src" / "deepusnav").is_dir():
            raise FileNotFoundError(f"DeepUSNav source package not found under {self.root}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Atlas metric index not found: {self.index_path}")
        if not self.target_path.is_file():
            raise FileNotFoundError(f"Atlas target not found: {self.target_path}")
        if self.k < 1:
            raise ValueError("atlas k must be at least 1")
        if self.target_tolerance_mm <= 0:
            raise ValueError("atlas target tolerance must be positive")

        os.environ.setdefault("TORCH_HOME", str(self.root / "checkpoints" / "torch"))
        for path in (self.root / "src", self.root):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        import torch
        from models.atlas_probe import load_metric_index

        self.torch = torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        target = json.loads(self.target_path.read_text())
        head, bank_z, bank_u, meta = load_metric_index(self.index_path, self.device)
        required_meta = ("encoder", "representation", "frame", "scale_mm_per_unit")
        missing = [key for key in required_meta if key not in meta]
        if missing:
            raise ValueError(f"Atlas metric index is missing metadata: {missing}")
        if target.get("frame") != meta["frame"]:
            raise ValueError(
                f"Atlas target uses {target.get('frame')!r}, metric index uses "
                f"{meta['frame']!r}"
            )
        if meta["representation"] != "grid4x4":
            raise ValueError(
                "Online Atlas runtime currently requires a grid4x4 metric index; got "
                f"{meta['representation']!r}"
            )
        if len(bank_z) != len(bank_u) or not len(bank_z):
            raise ValueError(
                f"Atlas bank has {len(bank_z)} embeddings and {len(bank_u)} coordinates"
            )

        self.head = head
        self.meta = meta
        self.encoder_name = str(meta["encoder"])
        self.representation = str(meta["representation"])
        self.frame = str(meta["frame"])
        self.target_name = str(target.get("target", "unknown"))
        self.target_u = torch.as_tensor(
            target["mu"], dtype=torch.float32, device=self.device
        )
        self.scale = torch.as_tensor(
            target["scale_mm_per_unit"], dtype=torch.float32, device=self.device
        )
        self.bank_z = torch.as_tensor(bank_z, dtype=torch.float32, device=self.device)
        self.bank_u = torch.as_tensor(bank_u, dtype=torch.float32, device=self.device)
        self.k = min(self.k, len(self.bank_z))

        self.bank_owner = np.asarray(meta.get("bank_owner", []), dtype=np.int64)
        self.owner_pids = [str(pid) for pid in meta.get("owner_pids", [])]
        if len(self.bank_owner) not in (0, len(self.bank_z)):
            raise ValueError("Atlas bank_owner length does not match the retrieval bank")

        cached = (encoder_cache or {}).get(self.encoder_name)
        if cached is not None:
            self.encoder, self.input_size, self.patch, self.feature_key = cached
        else:
            loader_path = self.root / "scripts" / "data" / "cache_latents.py"
            spec = importlib.util.spec_from_file_location(
                "_deepusnav_atlas_encoder", loader_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not import encoder loader: {loader_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            previous = Path.cwd()
            try:
                # The shared loader resolves V-JEPA2 source and weights relative to the repo.
                os.chdir(self.root)
                spec.loader.exec_module(module)
                bundle = module.load_encoder(self.encoder_name, self.device)
                self.encoder, self.input_size, self.patch, self.feature_key = bundle
            finally:
                os.chdir(previous)

    def model_info(self) -> dict[str, Any]:
        """Return static provenance for a status panel or diagnostic command."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "encoder": self.encoder_name,
            "representation": self.representation,
            "canonical_frame": self.frame,
            "target": self.target_name,
            "target_u": self.target_u.detach().cpu().tolist(),
            "scale_mm_per_unit": self.scale.detach().cpu().tolist(),
            "bank_size": int(len(self.bank_z)),
            "embedding_dim": int(self.bank_z.shape[1]),
            "k": self.k,
            "target_tolerance_mm": self.target_tolerance_mm,
            "heldout_median_mm": self.meta.get("heldout_median_mm"),
            "heldout_mean_mm": self.meta.get("heldout_mean_mm"),
            "device": self.device,
            "index_path": str(self.index_path),
            "target_path": str(self.target_path),
        }

    def _encode(self, image: np.ndarray):
        from models.world_model_data import as_unit_interval, letterbox

        gray = np.asarray(image)
        if gray.ndim == 3:
            if gray.shape[2] not in (3, 4):
                raise ValueError(f"Unsupported ultrasound shape: {gray.shape}")
            rgb = gray[..., :3]
            gray = rgb.astype(np.float32).mean(axis=2)
            if image.dtype == np.uint8:
                gray = np.rint(gray).astype(np.uint8)
        if gray.ndim != 2:
            raise ValueError(f"Expected one grayscale ultrasound frame, got {gray.shape}")

        unit = as_unit_interval(gray)
        x = letterbox(self.torch.from_numpy(unit[None]), self.input_size, self.patch)
        with self.torch.inference_mode():
            return self.encoder.forward_features(x.to(self.device))[self.feature_key][0].float()

    def _grid4x4(self, tokens):
        if tokens.ndim != 2:
            raise ValueError(f"Expected (patches, features), got {tuple(tokens.shape)}")
        patches, features = tokens.shape
        side = math.isqrt(patches)
        if side * side != patches or side % 4:
            raise ValueError(f"Cannot pool {patches} patch tokens into a 4x4 grid")
        return (
            tokens.reshape(1, side, side, features)
            .reshape(1, 4, side // 4, 4, side // 4, features)
            .mean(dim=(2, 4))
            .reshape(1, -1)
        )

    def localise(self, image: np.ndarray) -> dict[str, Any]:
        """Return a JSON-serialisable single-frame anatomical belief."""
        started = time.perf_counter()
        with self.torch.inference_mode():
            tokens = self._encode(image)
            feature = self._grid4x4(tokens)
            query = self.head(feature)
            distances = self.torch.cdist(query, self.bank_z)[0]
            nearest_distance, nearest_index = distances.topk(self.k, largest=False)
            weights = 1.0 / (nearest_distance + 1e-6)
            weights = weights / weights.sum()
            neighbour_u = self.bank_u[nearest_index]
            estimate_u = (weights[:, None] * neighbour_u).sum(dim=0)

            neighbour_goal_mm = self.torch.linalg.vector_norm(
                (neighbour_u - self.target_u) * self.scale, dim=1
            )
            expected_goal_mm = (weights * neighbour_goal_mm).sum()
            point_goal_mm = self.torch.linalg.vector_norm(
                (estimate_u - self.target_u) * self.scale
            )
            offset_mm = (estimate_u - self.target_u) * self.scale
            neighbour_offset_norm = self.torch.linalg.vector_norm(
                (neighbour_u - estimate_u) * self.scale, dim=1
            )
            neighbour_spread_mm = self.torch.sqrt(
                (weights * neighbour_offset_norm.square()).sum()
            )
            entropy_bits = -(weights * self.torch.log2(weights.clamp_min(1e-12))).sum()
            target_probability = weights[
                neighbour_goal_mm <= self.target_tolerance_mm
            ].sum()

        idx = nearest_index.detach().cpu().numpy()
        owners: list[str] = []
        if len(self.bank_owner):
            for row in idx:
                owner = int(self.bank_owner[int(row)])
                owners.append(
                    self.owner_pids[owner] if 0 <= owner < len(self.owner_pids) else str(owner)
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "schema_version": self.SCHEMA_VERSION,
            "status": "ok",
            "encoder": self.encoder_name,
            "representation": self.representation,
            "canonical_frame": self.frame,
            "target": self.target_name,
            "coordinate_u": estimate_u.detach().cpu().tolist(),
            "target_u": self.target_u.detach().cpu().tolist(),
            "offset_to_target_mm": offset_mm.detach().cpu().tolist(),
            # The expected cost preserves multimodality; point distance is diagnostic only.
            "expected_goal_distance_mm": float(expected_goal_mm.item()),
            "point_goal_distance_mm": float(point_goal_mm.item()),
            "target_probability": float(target_probability.item()),
            "target_tolerance_mm": self.target_tolerance_mm,
            "within_target": bool(expected_goal_mm.item() <= self.target_tolerance_mm),
            "neighbour_spread_mm": float(neighbour_spread_mm.item()),
            "nearest_metric_distance": float(nearest_distance[0].item()),
            "entropy_bits": float(entropy_bits.item()),
            "entropy_bits_max": float(math.log2(self.k)),
            "effective_neighbours": float(2.0 ** entropy_bits.item()),
            "distinct_patients": len(set(owners)) if owners else None,
            "neighbour_patients": owners,
            "k": self.k,
            "bank_size": int(len(self.bank_z)),
            "latency_ms": elapsed_ms,
        }
        return result
