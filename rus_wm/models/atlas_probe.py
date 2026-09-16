"""What both position probes need: frame coordinates, prototype features, error in mm.

`probe_retrieval` asks whether nearest-neighbour in a frozen latent space finds the same
anatomy on a new patient, and `probe_position` asks whether a learned head can. They are
different questions with the same three pieces of machinery underneath, and those pieces
each have a way to be silently wrong -- a subsampled cache paired with the wrong frames, a
canonical spread quoted in units that differ between frames -- so they live here once
rather than twice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..anatomy.atlas import patient_frame
from ..envs.generate import TransitionBatch
from ..probe.surface_cmd import SurfaceCommand

#: How an atlas could afford to store one prototype. `patch` is an upper bound no atlas
#: can hold -- 256 x 1024 per bucket is ~17 TB over a 16k-bucket grid -- and is measured
#: only to show what the affordable ones give up.
REPRESENTATIONS = ("pooled", "grid4x4", "patch", "tokens")


def load_cached_latents(root, pid: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Load latent rows and the transition-frame indices they represent.

    A cache created with ``--frames anchors`` or ``--frames random`` stores fewer rows
    than the transition shard. Its sidecar is therefore part of the data, not optional
    metadata: ignoring it silently pairs each latent with the wrong anatomical position.
    Full caches have no sidecar and map one-to-one onto every frame.
    """
    root = Path(root)
    z = np.load(root / f"{pid}.npy").astype(np.float32)
    index_path = root / f"{pid}.index.npy"
    index = (np.load(index_path).astype(np.int64) if index_path.exists()
             else np.arange(n_frames, dtype=np.int64))
    if len(z) != len(index):
        raise ValueError(f"{pid}: {len(z)} latent rows but {len(index)} frame indices")
    if len(index) and (index.min() < 0 or index.max() >= n_frames):
        raise ValueError(f"{pid}: cache index out of range for {n_frames} frames")
    return z, index


def representation(latents: np.ndarray, kind: str) -> np.ndarray:
    """``(N, P, D)`` patch tokens -> one flat feature per frame."""
    n, p, d = latents.shape
    if kind == "pooled":
        return latents.mean(axis=1)
    if kind == "patch":
        return latents.reshape(n, -1)
    if kind == "tokens":
        # The grid untouched. Every other representation pools it -- `grid4x4` by 16x --
        # and a head that reads position out of fine spatial structure cannot see any of
        # it afterwards. Which is a confound in every read-out measured so far, since all
        # of them were fitted on `grid4x4`.
        return latents
    if kind != "grid4x4":
        raise ValueError(f"unknown representation {kind!r}; use one of {REPRESENTATIONS}")
    side = round(p ** 0.5)
    grid = latents.reshape(n, side, side, d)
    return grid.reshape(n, 4, side // 4, 4, side // 4, d).mean(axis=(2, 4)).reshape(n, -1)


def frame_coordinates(batch: TransitionBatch, patient, frame_kind: str,
                      keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Every frame's probe origin, in patient millimetres and in canonical coordinates.

    Reconstructed from the stored commands rather than read from a column: every frame is
    an endpoint of at least one edge, so this is exact, and it leaves the canonical frame
    a choice made at analysis time -- which is the whole reason `u` is not stored.

    ``keep`` is the index a subsampled latent cache wrote alongside itself. Ignoring it
    pairs features with the wrong positions and produces a plausible number measuring
    nothing.
    """
    command = np.zeros((len(batch.images), 3))
    command[batch.anchor_index] = batch.command
    command[batch.next_index] = batch.next_command
    if keep is None:
        keep = np.arange(len(command))
    world = np.stack([SurfaceCommand.from_array(command[i]).pose(patient.heightmap)[:3, 3]
                      for i in keep])
    return world, np.asarray(patient_frame(patient, frame_kind).to_canonical(world))


def canonical_error_mm(predicted_u: np.ndarray, true_world_mm: np.ndarray, patient,
                       frame_kind: str) -> np.ndarray:
    """A canonical coordinate's error, in the millimetres of the frame's own patient.

    Canonical units are not comparable between frames -- `spine_affine` compresses the
    long axis onto L1..L5 while `body_box` spreads it over the whole torso -- so a spread
    quoted in them says nothing across frames. Millimetres of the *query* patient's
    anatomy do, and are what "34 mm between adjacent vertebrae" can be read against.
    """
    return np.linalg.norm(canonical_offset_mm(predicted_u, true_world_mm, patient, frame_kind),
                          axis=1)


def canonical_offset_mm(predicted_u: np.ndarray, true_world_mm: np.ndarray, patient,
                        frame_kind: str) -> np.ndarray:
    """``(N, 3)`` signed ``guess - truth`` in patient mm along the CT axes: 0 lateral,
    1 antero-posterior (depth), 2 cranio-caudal. Its norm is `canonical_error_mm`."""
    guess = np.asarray(patient_frame(patient, frame_kind).from_canonical(predicted_u))
    return guess - true_world_mm


class PositionHead(nn.Module):
    """A fitted frozen-latent -> canonical-coordinate map, with its normalisation inside it.

    The probe used to build a bare ``nn.Sequential`` and hang ``input_mean`` / ``input_std``
    on it as plain attributes. That works for the length of one process and silently loses
    the standardisation the moment the weights are saved, because attributes are not in a
    ``state_dict`` -- and a head evaluated without its own training statistics does not
    fail, it produces coordinates that are wrong by a fixed affine map. Registering them
    as buffers is what makes "save the head" mean "save the head".

    The module stack is built in the same order and with the same calls as the loose
    version it replaces, so a given seed still initialises the same weights and the
    published 32.4 mm stays reproducible.
    """

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        self.register_buffer("input_mean", torch.zeros(1, in_dim))
        self.register_buffer("input_std", torch.ones(1, in_dim))

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean.reshape(1, -1))
        self.input_std.copy_(std.reshape(1, -1).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.input_mean) / self.input_std)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        device = self.input_mean.device
        return self(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def save_head(path, head: PositionHead, **meta) -> Path:
    """Persist a head together with everything needed to interpret its output.

    A canonical coordinate is meaningless without the frame that defines it and the
    representation that produced it: the same three numbers name a different place under
    `body_box` than under `spine_cpr`, and a head fitted on `grid4x4` reads noise from
    `pooled` features of the same encoder. Those are recorded here rather than left to a
    caller to remember, along with the subjects the head was fitted on -- which is what
    lets a later script assert it is not evaluating on a training patient.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = ("encoder", "representation", "frame", "fitted_on")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"save_head needs {missing}: a head without them cannot be read back")
    spatial = isinstance(head, SpatialTokenHead)
    torch.save({"state_dict": head.state_dict(),
                "arch": "spatial" if spatial else "mlp",
                "in_dim": int(head.input_mean.shape[-1]),
                "hidden": int(head.token[0].out_features if spatial
                              else head.net[0].out_features),
                "n_tokens": int(head.position.shape[1]) if spatial else 0,
                "out_dim": int((head.out[-1] if spatial else head.net[-1]).out_features),
                **meta}, path)
    return path


def load_head(path, device: str = "cpu") -> tuple[PositionHead, dict]:
    """``(head, metadata)``. The head comes back with its normalisation attached."""
    blob = torch.load(path, map_location=device, weights_only=False)
    # Read the shape back from the weights rather than trusting a recorded field. A head
    # that predicts a cost has one output and one that predicts a coordinate has three;
    # rebuilding with the wrong number fails loudly here, but only because the geometry is
    # taken from the tensors that are actually in the file.
    if blob.get("arch") == "spatial":
        head = SpatialTokenHead(blob["in_dim"], blob["n_tokens"], blob["hidden"],
                                blob["out_dim"]).to(device)
    else:
        last = max(k for k in blob["state_dict"] if k.endswith(".weight"))
        out_dim = int(blob["state_dict"][last].shape[0])
        head = PositionHead(blob["in_dim"], blob["hidden"], out_dim).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    return head, {k: v for k, v in blob.items() if k not in ("state_dict",)}


class SpatialTokenHead(nn.Module):
    """A read-out that sees the patch grid instead of a 16x-pooled summary of it.

    Every localiser measured so far was an MLP over `grid4x4`: 256 patch tokens averaged
    down to 16 cells, then flattened to 16 384 numbers. That is a 16-fold spatial
    compression before the head sees anything, and it is a confound in the conclusion that
    the frame does not carry fan-scale position -- a vertebra shifting by a few millimetres
    moves within a `grid4x4` cell, not between cells.

    This keeps the grid. A shared MLP over tokens, a learned position embedding so the head
    knows *where* each token sat, and attention pooling with one query, which lets it
    weight the tokens that see bone rather than averaging them with the ones that do not.
    It is also far smaller than the head it replaces -- about 0.4 M parameters against
    8.4 M -- because it shares weights across tokens instead of learning one row per cell,
    and the flattened version's near-zero training error was always a sign that the
    capacity was going into memorising subjects.
    """

    def __init__(self, feature_dim: int = 1024, n_tokens: int = 256, hidden: int = 256,
                 out_dim: int = 3):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden))
        self.position = nn.Parameter(torch.zeros(1, n_tokens, hidden))
        nn.init.normal_(self.position, std=0.02)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.attend = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        self.register_buffer("input_mean", torch.zeros(1, 1, feature_dim))
        self.register_buffer("input_std", torch.ones(1, 1, feature_dim))

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean.reshape(1, 1, -1))
        self.input_std.copy_(std.reshape(1, 1, -1).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token((x - self.input_mean) / self.input_std) + self.position
        pooled, _ = self.attend(self.query.expand(len(h), -1, -1), h, h, need_weights=False)
        return self.out(pooled[:, 0])

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        device = self.input_mean.device
        return self(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


class MetricHead(nn.Module):
    """A frozen latent -> a space whose Euclidean distance is anatomical millimetres.

    Route B's learned half. `PositionHead` regresses the coordinate; this one never names a
    position at all. It only arranges frames so that neighbours in the embedding are
    neighbours in the patient, and position then comes from *whose* neighbours they are --
    the coordinates of the indexed training frames. That is what makes retrieval a
    different integration rather than a differently-shaped regression: it can only return
    positions the training set already visited, and it degrades by returning the wrong
    patient's anatomy rather than by drifting to a coordinate nobody occupies.

    **Deliberately the same stack, width and input standardisation as `PositionHead`**, for
    the reason `CoordinateHead` is: a difference between the two rows should be the
    read-out mechanism and not capacity.

    **Provenance.** AdLocUI (Yeung, Aliasi, Haak, INTERGROWTH-21st Consortium, Xie,
    Namburete, MICCAI 2022, arXiv:2209.05477) localises 2D slices into a 3D volume by
    embedding-space matching; Chen, Schmidt, Prisman, Salcudean (arXiv:2412.07741) put
    probe location into a contrastive objective for ultrasound retrieval. Neither loss is
    reused verbatim -- theirs are contrastive, with a temperature and a positive/negative
    split to choose. The objective here regresses the *distance itself*, which has no such
    knobs and makes the embedding's units the millimetres the rest of the comparison is
    already in.
    """

    def __init__(self, in_dim: int, hidden: int = 512, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, embed_dim))
        self.register_buffer("input_mean", torch.zeros(1, in_dim))
        self.register_buffer("input_std", torch.ones(1, in_dim))

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.input_mean.copy_(mean.reshape(1, -1))
        self.input_std.copy_(std.reshape(1, -1).clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.input_mean) / self.input_std)

    @torch.no_grad()
    def embed(self, x) -> np.ndarray:
        device = self.input_mean.device
        return self(torch.as_tensor(np.asarray(x), dtype=torch.float32,
                                    device=device)).cpu().numpy()


def retrieve_neighbours(bank_z: np.ndarray, query_z: np.ndarray, *, k: int = 8,
                        device: str = "cpu", chunk: int = 512,
                        metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray]:
    """``(distance, index)`` of each query's ``k`` nearest bank rows, ``(Q, k)`` each.

    The lookup and nothing else: what is done with the neighbours -- a coordinate, a
    level vote, a count of how many patients they came from -- is the caller's, so the
    localisation benchmark and the ranking score cannot disagree about *which* rows they
    were looking at. ``metric`` is Euclidean for the metric head, whose embedding is in
    millimetres by construction, and cosine for a raw encoder latent, where the norm
    carries nothing anatomical and `probe_retrieval.py` established cosine as the
    reference raw-encoder baseline.

    ``chunk`` bounds the distance matrix, not the answer: `cdist` materialises
    ``chunk x len(bank)`` scalars, so a hundred-thousand-row bank at the default is about
    200 MB and a four-thousand-row chunk would be 1.6 GB on a card that is also holding a
    world model.
    """
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"metric must be euclidean or cosine, not {metric!r}")
    b = torch.as_tensor(np.asarray(bank_z), dtype=torch.float32, device=device)
    if metric == "cosine":
        b = torch.nn.functional.normalize(b, dim=1)
    k = min(int(k), len(b))
    dist, index = [], []
    for start in range(0, len(query_z), chunk):
        q = torch.as_tensor(np.asarray(query_z[start:start + chunk]),
                            dtype=torch.float32, device=device)
        if metric == "cosine":
            # 1 - cos, so that "smaller is nearer" holds for both metrics and the same
            # inverse-distance weights apply downstream.
            d = 1.0 - torch.nn.functional.normalize(q, dim=1) @ b.T
        else:
            d = torch.cdist(q, b)
        d, idx = d.topk(k, dim=1, largest=False)
        dist.append(d.cpu().numpy())
        index.append(idx.cpu().numpy())
    if not dist:
        return np.zeros((0, k)), np.zeros((0, k), dtype=np.int64)
    return np.concatenate(dist), np.concatenate(index)


def neighbour_weights(dist: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """``(Q, k)`` normalised inverse-distance weights: the single-frame posterior over rows.

    Shepard's inverse-distance weighting (Shepard, "A two-dimensional interpolation
    function for irregularly-spaced data", Proc. ACM National Conference, 1968,
    doi:10.1145/800186.810616), with exponent 1. Weighted rather than plain-mean because
    the bank is not uniform over the anatomy: a query near a densely visited region gets
    neighbours that are genuinely close, and one in a sparse region gets whatever is
    nearest, and averaging those equally would let the sparse case pull an answer towards
    wherever the training set happened to go. ``eps`` is one micron in embedding units and
    is only there so an exact hit does not divide by zero. No temperature: the embedding
    is in millimetres by construction, and a knob here would be a second metric.

    Kept as a distribution over the k rows rather than collapsed, because anything that is
    multimodal -- "L3 or L5" -- has a mean that lands on L4 and a cost that has to be an
    expectation over the modes, not the cost of the mean (`method_costs('retrieval')`).
    """
    w = 1.0 / (np.asarray(dist, dtype=np.float64) + eps)
    return w / w.sum(axis=1, keepdims=True)


def weighted_coordinates(bank_u: np.ndarray, dist: np.ndarray, index: np.ndarray,
                         eps: float = 1e-6) -> np.ndarray:
    """``u`` for each query as the posterior mean of its neighbours' coordinates.

    The point estimate for an *absolute localisation error*, where a single coordinate is
    what the millimetres are measured against. It is not what a cost should be computed
    on -- see `neighbour_weights`.
    """
    w = neighbour_weights(dist, eps)
    return (w[:, :, None] * np.asarray(bank_u, dtype=np.float64)[index]).sum(axis=1)


def retrieve_coordinates(bank_z: np.ndarray, bank_u: np.ndarray, query_z: np.ndarray, *,
                         k: int = 8, device: str = "cpu",
                         chunk: int = 512) -> np.ndarray:
    """``u`` for each query, as the inverse-distance mean of its ``k`` nearest bank rows.

    `retrieve_neighbours` followed by `weighted_coordinates`; kept as one call because it
    is the read-out `retrieval` scores with and the one the index's held-out check
    reports, and the two must be the same arithmetic.
    """
    if len(query_z) == 0:
        return np.zeros((0, np.asarray(bank_u).shape[1]))
    dist, index = retrieve_neighbours(bank_z, query_z, k=k, device=device, chunk=chunk)
    return weighted_coordinates(bank_u, dist, index).astype(np.float32)


def save_metric_index(path, head: MetricHead, bank_z: np.ndarray, bank_u: np.ndarray,
                      **meta) -> Path:
    """The head and the indexed training set, in one file.

    They are two of `retrieval`'s `needs` and they are useless apart: an embedding is
    defined only relative to the bank it is matched against, and a bank embedded by a
    different head names nothing. Saving them separately would make a mismatched pair a
    silent wrong answer rather than a missing file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = ("encoder", "representation", "frame", "fitted_on")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"save_metric_index needs {missing}")
    if len(bank_z) != len(bank_u):
        raise ValueError(f"{len(bank_z)} embeddings against {len(bank_u)} coordinates")
    torch.save({"state_dict": head.state_dict(),
                "in_dim": int(head.input_mean.shape[-1]),
                "hidden": int(head.net[0].out_features),
                "embed_dim": int(head.net[-1].out_features),
                "bank_z": np.asarray(bank_z, dtype=np.float32),
                "bank_u": np.asarray(bank_u, dtype=np.float32),
                **meta}, path)
    return path


def load_metric_index(path, device: str = "cpu"):
    """``(head, bank_z, bank_u, metadata)``, the head already in eval mode."""
    blob = torch.load(path, map_location=device, weights_only=False)
    head = MetricHead(blob["in_dim"], blob["hidden"], blob["embed_dim"]).to(device)
    head.load_state_dict(blob["state_dict"])
    head.eval()
    meta = {k: v for k, v in blob.items()
            if k not in ("state_dict", "bank_z", "bank_u")}
    return head, blob["bank_z"], blob["bank_u"], meta
