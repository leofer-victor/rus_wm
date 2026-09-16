"""One shard, three batch shapes: the loaders both baselines and the ranker read.

The point of putting them in one module is that they must not drift. A comparison between
DINO-WM and V-JEPA 2-AC is only about the models if everything before the encoder is
identical -- the same patients, the same frames, the same action normalisation, the same
letterbox. Two loaders written a week apart would differ in one of those and nobody would
notice, because both would train.

Three deliberate choices, each of which the obvious default gets wrong here:

**Letterbox, never crop.** The frame is 200 px of depth by 150 of width. A centre crop to
a square would throw away a quarter of the depth axis, which is the axis the vertebra
shadow lives on. Aspect-preserving resize puts depth across the full 224 (or 256) and pads
the sides with black -- what a scanner shows where there is no signal.

**No flips, no random crops.** A horizontal flip mirrors the image but not the action that
produced it, so it teaches the predictor that ``+dx`` and ``-dx`` look the same. Standard
augmentation is wrong for an action-conditioned model unless the action is transformed
with the pixels, and for a spatial action on a curved surface that is not a sign flip.

**No proprioception.** ``command`` is the probe's pose in the *patient's own* surface
frame, so feeding it assumes the registration this project exists to avoid needing. The
proprio slot is a constant zero, which keeps upstream's tensor shapes without giving the
model an answer. See the manifest's ``policy_inputs``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..envs.generate import TransitionBatch

#: What the released encoders were normalised with. Both DINOv2 and the V-JEPA 2 training
#: configs use ImageNet statistics, so a grayscale frame repeated to RGB goes through the
#: same transform its encoder saw in pretraining.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def as_unit_interval(images: np.ndarray) -> np.ndarray:
    """Frames as float32 in [0, 1], whatever they arrived as.

    Two conventions meet at the encoder: `TransitionBatch` stores frames as uint8 (the
    renderer's [0, 1] output times 255), while the environment hands a policy the
    renderer's float output directly. Dividing the second by 255 does not raise an error
    -- it produces an image whose brightest pixel is 0.0006, and an encoder fed that
    returns the latent of a black frame. Every model-based closed-loop policy up to
    2026-09-10 (`run_closed_loop.py::make_encode`) planned on exactly that, which is why
    each of them converged cleanly onto a wrong minimum. So the scaling is decided by the
    dtype here, in one place, and a float frame outside [0, 1] is refused rather than
    guessed at.
    """
    x = np.asarray(images)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    if x.size and (x.min() < 0.0 or x.max() > 1.0 + 1e-6):
        raise ValueError(f"float frames must already be in [0, 1]; got [{x.min():.3g}, "
                         f"{x.max():.3g}] -- integer frames are scaled by dtype, not by range")
    return x


def letterbox(images: torch.Tensor, size: int, patch: int = 1) -> torch.Tensor:
    """``(N, H, W)`` in [0, 1] -> ``(N, 3, size, size)``, aspect preserved, sides padded.

    ``patch`` rounds the resized extent down to a multiple of the encoder's patch size, so
    the anatomy occupies whole patches rather than straddling a boundary with the padding.
    """
    if images.ndim != 3:
        raise ValueError(f"expected (N, H, W) grayscale frames, got {tuple(images.shape)}")
    n, h, w = images.shape
    scale = min(size / h, size / w)
    new_h, new_w = (max(patch, round(d * scale) // patch * patch) for d in (h, w))

    x = images[:, None].repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    canvas = x.new_zeros((n, 3, size, size))
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[..., top:top + new_h, left:left + new_w] = x

    mean = torch.tensor(IMAGENET_MEAN, dtype=canvas.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=canvas.dtype).view(3, 1, 1)
    return (canvas - mean) / std


@dataclass(frozen=True)
class ActionStats:
    """Per-dimension action normalisation, fitted on **train patients only**.

    The three action dimensions are not commensurable -- two are voxel indices and one is
    radians, and their boxes differ by a factor of twenty -- so an unnormalised action
    vector hands the predictor a rotation that looks like rounding error next to a slide.
    Fitting on train alone is not pedantry: statistics fitted over val or test are a
    population quantity leaking across the split the whole claim rests on.
    """

    mean: np.ndarray  # (3,)
    std: np.ndarray  # (3,)
    patients: tuple[str, ...]

    @classmethod
    def fit(cls, shards: list[Path]) -> ActionStats:
        actions, patients = [], []
        for path in sorted(shards):
            # Only two columns are read here. Decompressing the pixels beside them is
            # 0.7 GB and two seconds a patient -- ten minutes of a 292-patient job spent
            # before the first optimiser step.
            b = TransitionBatch.load(path, with_images=False)
            actions.append(b.actions)
            patients.extend(b.patients)
        if not actions:
            raise ValueError("cannot fit action statistics on no shards")
        stacked = np.concatenate(actions)
        std = stacked.std(axis=0)
        # A dimension that never moved would divide by zero; leave it alone instead.
        std = np.where(std > 1e-8, std, 1.0)
        return cls(mean=stacked.mean(axis=0).astype(np.float32),
                   std=std.astype(np.float32), patients=tuple(sorted(set(patients))))

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return ((actions - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "fitted_on": list(self.patients)}

    @classmethod
    def from_dict(cls, d: dict) -> ActionStats:
        return cls(mean=np.asarray(d["mean"], np.float32),
                   std=np.asarray(d["std"], np.float32),
                   patients=tuple(d.get("fitted_on", ())))


def shard_paths(root, pids: list[str] | None = None) -> list[Path]:
    """The shards under one split directory, optionally restricted to some patients."""
    files = sorted(Path(root).glob("*.npz"))
    if pids is not None:
        keep = set(pids)
        files = [f for f in files if f.stem in keep]
    if not files:
        raise FileNotFoundError(f"no shards under {root} for pids={pids}")
    return files


class _ShardBacked(Dataset):
    """Shared plumbing: hold shards open, map a global index onto (shard, row)."""

    def __init__(self, root, *, size: int, patch: int, stats: ActionStats,
                 pids: list[str] | None = None, with_images: bool = True):
        self.paths = shard_paths(root, pids)
        self.size, self.patch, self.stats = int(size), int(patch), stats
        self.with_images = bool(with_images)
        # A split's pixels are ~0.7 GB a patient, so holding 292 of them open is ~200 GB
        # -- more than the node has, and pure waste when the caller collates from the
        # latent cache and throws every decoded frame away. See `TransitionBatch.load`.
        self.batches = [TransitionBatch.load(p, with_images=with_images)
                        for p in self.paths]
        self.pids = [p.stem for p in self.paths]
        self.index: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self.index)

    def _frames(self, shard: int, frame_index: np.ndarray) -> torch.Tensor:
        """The window's pixels, or an empty stand-in when the shard was opened without.

        A caller that collates from the latent cache never looks at ``visual``, so
        decoding and letterboxing it is the dominant per-item cost spent on an array that
        is discarded a moment later. The slot is kept -- shape ``(n, 0)`` rather than
        absent -- so an item's keys do not depend on how the shard was opened.
        """
        if not self.with_images:
            return torch.empty(len(frame_index), 0)
        images = self.batches[shard].images[frame_index].astype(np.float32) / 255.0
        return letterbox(torch.from_numpy(images), self.size, self.patch)


class USTrajDataset(_ShardBacked):
    """Contiguous windows: what DINO-WM and V-JEPA 2-AC train their predictors on.

    One item is ``{"visual", "action", "proprio", "patient", "frame_index"}``. ``visual``
    is ``(n_frames, 3, size, size)`` and ``action`` is ``(n_steps, 3 * frameskip)`` with
    the skipped steps concatenated, which is upstream's ``process_actions="concat"``.

    ``exit_action`` follows DINO-WM, whose model packs an action with every frame; leave
    it off for V-JEPA 2-AC, which conditions on the ``n_frames - 1`` transitions.
    """

    def __init__(self, root, *, size: int = 224, patch: int = 14, n_frames: int = 4,
                 frameskip: int = 1, exit_action: bool = False,
                 stats: ActionStats | None = None, pids: list[str] | None = None,
                 with_images: bool = True, min_target_fraction: float | None = None):
        super().__init__(root, size=size, patch=patch, pids=pids,
                         with_images=with_images,
                         stats=stats or ActionStats.fit(shard_paths(root, pids)))
        self.n_frames, self.frameskip = int(n_frames), int(frameskip)
        self.exit_action = bool(exit_action)
        self.windows = []
        for shard, b in enumerate(self.batches):
            frames, actions = b.windows(self.n_frames, self.frameskip, self.exit_action)
            if min_target_fraction is not None and len(frames):
                # Keep a window only if the target is visible somewhere along it. Off by
                # default: a window with nothing in view is still a true transition, and
                # dropping it changes the state distribution the predictor is fitted to.
                visible = b.target_visible(min_target_fraction)[frames].any(axis=1)
                frames, actions = frames[visible], actions[visible]
            for row in range(len(frames)):
                self.index.append((shard, len(self.windows)))
                self.windows.append((shard, frames[row], actions[row]))

    def __getitem__(self, i: int) -> dict:
        shard, frame_index, actions = self.windows[self.index[i][1]]
        action = self.stats.normalize(actions).reshape(len(actions), -1)
        return {
            "visual": self._frames(shard, frame_index),
            "action": torch.from_numpy(action),
            # Upstream's tensor slot, deliberately carrying nothing. See the module docs.
            "proprio": torch.zeros(len(frame_index), 1),
            "patient": self.pids[shard],
            "frame_index": torch.from_numpy(frame_index.astype(np.int64)),
        }


class BranchWindowDataset(_ShardBacked):
    """Branch edges as two-frame training windows: ``[anchor, successor]``.

    Rollout windows alone cannot teach action-conditioning, and the reason is structural
    rather than a matter of scale: in a rollout each visual history occurs exactly once,
    so the history already determines the next frame and the action carries no gradient.
    Measured on the debug split, a predictor trained on rollouts alone scores 24.4 % on
    K-way action matching against 25 % chance -- it ignores the action completely.

    A branch fan is the opposite case by construction: one anchor, K actions, K different
    successors, so the action is the *only* thing that separates the targets. Training on
    them makes ignoring the action impossible rather than merely unrewarding.

    Two frames, not a padded four. A branch anchor is a fresh reset with no past, and
    repeating it to fill a window would either assert that the action changed nothing (if
    the padding steps carry the real action) or invent null steps the renderer never
    produced. The predictor's position embedding is indexed to the window length, so a
    shorter window costs nothing and fabricates nothing.
    """

    def __init__(self, root, *, size: int = 224, patch: int = 14,
                 stats: ActionStats | None = None, pids: list[str] | None = None,
                 with_images: bool = True):
        super().__init__(root, size=size, patch=patch, pids=pids,
                         with_images=with_images,
                         stats=stats or ActionStats.fit(shard_paths(root, pids)))
        self.edges = []
        for shard, b in enumerate(self.batches):
            for edge in np.flatnonzero(b.is_branch):
                self.index.append((shard, len(self.edges)))
                self.edges.append((shard, int(edge)))

    def __getitem__(self, i: int) -> dict:
        shard, edge = self.edges[self.index[i][1]]
        b = self.batches[shard]
        frame_index = np.array([b.anchor_index[edge], b.next_index[edge]], dtype=np.int64)
        # One action per frame, as DINO-WM's model expects. The second is the step leaving
        # the window; the loss excludes the action dimensions, so its value is inert, and
        # repeating the real one keeps it from looking like a distinct token.
        action = np.repeat(self.stats.normalize(b.actions[edge])[None], 2, axis=0)
        return {
            "visual": self._frames(shard, frame_index),
            "action": torch.from_numpy(action),
            "proprio": torch.zeros(2, 1),
            "patient": self.pids[shard],
            "frame_index": torch.from_numpy(frame_index),
        }


class BranchDataset(_ShardBacked):
    """One anchor and its K counterfactual actions: the one-step ranking candidate set.

    Not a training set for the predictor -- it is what the predictor is *scored* on, and
    the only place a model is asked about actions nobody took. ``improved`` is the label;
    ``next_pos_error_mm`` and ``next_angle_error_rad`` are kept so regret can be computed
    against the best action in the fan rather than against a threshold.
    """

    def __init__(self, root, *, size: int = 224, patch: int = 14,
                 stats: ActionStats | None = None, pids: list[str] | None = None,
                 with_images: bool = True):
        super().__init__(root, size=size, patch=patch, pids=pids,
                         with_images=with_images,
                         stats=stats or ActionStats.fit(shard_paths(root, pids)))
        self.fans = []
        for shard, b in enumerate(self.batches):
            for edges in b.groups():
                self.index.append((shard, len(self.fans)))
                self.fans.append((shard, edges))

    def __getitem__(self, i: int) -> dict:
        shard, edges = self.fans[self.index[i][1]]
        b = self.batches[shard]
        frames = np.concatenate([b.anchor_index[edges[:1]], b.next_index[edges]])
        visual = self._frames(shard, frames)
        return {
            "anchor": visual[0],
            "candidates": visual[1:],                       # (K, 3, size, size)
            "action": torch.from_numpy(self.stats.normalize(b.actions[edges])),
            "improved": torch.from_numpy(b.improved[edges].astype(np.float32)),
            "next_pos_error_mm": torch.from_numpy(b.next_pos_error_mm[edges]),
            "next_angle_error_rad": torch.from_numpy(b.next_angle_error_rad[edges]),
            "pos_error_mm": torch.tensor(float(b.pos_error_mm[edges[0]])),
            "patient": self.pids[shard],
        }


class FrameCoordinates:
    """Anatomy coordinates keyed by ``(patient, shard frame index)``, as
    `cache_coordinates.py` writes.

    The read-only twin of `CachedLatents`, deliberately in the same module and with the
    same ``gather`` signature: a training batch asks both for the *same* frames, and the
    one way to be certain a latent and a coordinate describe the same frame is for the two
    lookups to be written next to each other and indexed identically.

    Unlike a latent cache this one is never partial -- coordinates are 277 KB a patient
    against 4.5 GB of features, so `cache_coordinates.py` writes every frame and a row
    number *is* a frame number. A missing patient raises rather than being skipped: a
    training run that quietly dropped the auxiliary loss for some patients would report a
    lambda it did not apply.
    """

    def __init__(self, root):
        self.root = Path(root)
        meta = self.root / ".." / "coords.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.frame = self.meta.get("frame")
        self.by_patient = {p.stem: np.load(p, mmap_mode="r")
                           for p in sorted(self.root.glob("*.npy"))
                           if not p.name.endswith(".world.npy")}
        if not self.by_patient:
            raise FileNotFoundError(
                f"no coordinates under {root}; run scripts/data/cache_coordinates.py")

    def __contains__(self, patient: str) -> bool:
        return patient in self.by_patient

    @property
    def patients(self) -> list[str]:
        return sorted(self.by_patient)

    def gather(self, patient: str, frame_index: np.ndarray) -> torch.Tensor:
        rows = self.by_patient.get(patient)
        if rows is None:
            raise KeyError(f"{patient} has no cached coordinates under {self.root}")
        return torch.from_numpy(
            np.asarray(rows[np.asarray(frame_index)], dtype=np.float32))

    def statistics(self, patients: list[str] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-axis mean and standard deviation over whole patients.

        These standardise the auxiliary loss's *target*, which is what makes its weight a
        dimensionless number rather than one that depends on how the canonical frame
        happens to be scaled -- `spine_cpr` compresses the long axis onto L1..L5 and
        `body_box` spreads it over the torso, so an unstandardised coordinate loss would
        mean something different under each and lambda would not transfer.
        """
        chosen = [self.by_patient[p] for p in (patients or self.patients)
                  if p in self.by_patient]
        stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a in chosen])
        return (torch.tensor(stacked.mean(0), dtype=torch.float32),
                torch.tensor(stacked.std(0), dtype=torch.float32).clamp_min(1e-6))


class CachedLatents:
    """Frame latents keyed by ``(patient, shard frame index)``, as `cache_latents.py` writes.

    **A cache need not hold every frame, and this is where that stays honest.** Ranking
    reads only the ~2 000 branch anchors out of a patient's 23 100 frames, so caching the
    rest costs 27x the disk for rows nothing ever asks for. A partial cache therefore
    ships ``<pid>.index.npy``, the shard frame each cached row came from, and lookup goes
    through it.

    Skipping that remap does not make a partial cache slower, it makes it **wrong**: row
    200 would silently answer for frame 200. So a missing frame raises here rather than
    returning a neighbour, and lives in the same module as the datasets that index it for
    the reason given at the top of this file -- the two must not drift.
    """

    def __init__(self, root):
        self.root = Path(root)
        meta = self.root / "cache.json"
        if not meta.exists():
            raise FileNotFoundError(
                f"no cache under {root}; run scripts/cache_latents.py first")
        self.meta = json.loads(meta.read_text())
        self.by_patient, self.index_of = {}, {}
        for path in sorted(self.root.glob("*.npy")):
            if path.name.endswith(".index.npy"):
                continue
            self.by_patient[path.stem] = np.load(path, mmap_mode="r")
            side = path.with_suffix(".index.npy")
            if side.exists():
                index = np.load(side)
                if len(index) != len(self.by_patient[path.stem]):
                    # This is how job 4129133's OOM was found: it re-encoded one patient
                    # at every frame without rewriting the 200-row index beside it, and
                    # the pair then described two different frame sets. Silently trusting
                    # the index would have fed the ranker the wrong frames for exactly one
                    # patient -- the kind of error a metric absorbs without complaint.
                    raise ValueError(
                        f"{path.name} has {len(self.by_patient[path.stem])} rows but "
                        f"{side.name} names {len(index)} frames. The cache and its index "
                        "were written by different runs; re-encode this patient.")
                self.index_of[path.stem] = index
        if not self.by_patient:
            raise FileNotFoundError(f"{root}: cache.json but no .npy shards")
        self.shape = tuple(self.meta["shape"])

    def __contains__(self, patient: str) -> bool:
        return patient in self.by_patient

    @property
    def patients(self) -> list[str]:
        return sorted(self.by_patient)

    def rows(self, patient: str, frame_index: np.ndarray) -> np.ndarray:
        """Shard frame numbers -> row numbers in this patient's cached array."""
        index = self.index_of.get(patient)
        if index is None:                       # a complete cache: row *is* the frame
            return np.asarray(frame_index)
        frame_index = np.asarray(frame_index)
        row = np.searchsorted(index, frame_index)
        row = np.clip(row, 0, len(index) - 1)
        missing = index[row] != frame_index
        if missing.any():
            raise KeyError(
                f"{patient}: frames {np.unique(frame_index[missing])[:8].tolist()} are "
                f"not in this cache ({len(index)} of them cached). Either cache them, or "
                "restrict the dataset to what the cache holds.")
        return row

    def gather(self, patient: str, frame_index: np.ndarray) -> torch.Tensor:
        rows = self.rows(patient, frame_index)
        return torch.from_numpy(
            np.asarray(self.by_patient[patient][rows], dtype=np.float32))
