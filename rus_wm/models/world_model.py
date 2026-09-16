"""The action-conditioned latent predictor, and the frozen encoder it sits on.

This is DINO-WM's architecture, reimplemented rather than imported: upstream's model is
bound to hydra, to a repo-relative import path and to a CUDA-only attention mask built at
construction time. The pieces that matter are small and are reproduced faithfully, with
the upstream file and line noted where a choice is theirs rather than ours.

What is trained here is the **predictor and the action encoder, nothing else**. The visual
encoder is frozen: 22 M parameters trained on LVD-142M are not going to be improved by 400
ultrasound windows, and letting them move would make the target of the latent loss drift
under the thing being fitted to it. Upstream freezes it the same way (`train.py:216`).

The loss is a latent MSE with teacher forcing, computed on the visual dimensions only --
the action dimensions are concatenated into every patch and then excluded from the loss
(`visual_world_model.py:229`), because asking the predictor to reproduce the action it was
given is free and would flatter the number.

Two things a reader should not expect to find:

* **no decoder gradient.** Upstream decodes `z_pred.detach()`, so pixel reconstruction
  trains a decoder for looking at and never touches the dynamics. Not implemented here.
* **no proprioception by default.** The slot exists, sized and concatenated exactly as
  upstream, and is fed zeros -- `command` is the probe's pose in the *patient's own*
  surface frame, and feeding it would assume the registration this project exists to avoid.
  See `deepusnav.models.world_model_data`. The one exception is the atlas-as-input arm
  (`proprio_dim=3`, `train_world_model.py --input-coord`), which fills the slot with the
  frame's coordinate in the **population** canonical frame. That is a different object:
  it is what a population-trained localiser produces from the image alone, not a
  patient-specific registration, so the objection above does not apply to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def block_causal_mask(n_frames: int, n_patches: int,
                      device=None) -> torch.Tensor:
    """Frame ``i`` may attend to every patch of frames ``0..i``, and no later one.

    Causal at the granularity of a *frame*, not a token: all patches of one frame are
    simultaneous observations, so ordering them would be inventing a raster scan the
    ultrasound does not have. This is upstream's `generate_mask_matrix` (`vit.py:13`),
    built on demand instead of at import time so it can live on any device.
    """
    frames = torch.arange(n_frames, device=device)
    allowed = frames[:, None] >= frames[None, :]              # (T, T) lower triangular
    return allowed.repeat_interleave(n_patches, 0).repeat_interleave(n_patches, 1)


class ActionEncoder(nn.Module):
    """``(B, T, A) -> (B, T, emb)``, one linear map applied at every timestep.

    A 1-D convolution of kernel 1 over time, which is what upstream's
    `ProprioceptiveEmbedding` is. Trained jointly with the predictor under the same loss
    and the same optimizer step -- it is the predictor's first layer, not a pretrained
    component, which is why nothing is loaded into it. Our action is 3-D; upstream's was
    2-D on PushT and 7-D on DROID, so there is nothing to transfer even in principle.
    """

    def __init__(self, action_dim: int, emb_dim: int = 10):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Conv1d(action_dim, emb_dim, kernel_size=1)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.proj(actions.transpose(1, 2)).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        inner = heads * dim_head
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:n, :n],
            dropout_p=self.dropout if self.training else 0.0)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class _FeedForward(nn.Sequential):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))


class ViTPredictor(nn.Module):
    """``(B, T, P, D) -> (B, T, P, D)``: the token at frame ``t`` predicts frame ``t+1``.

    Upstream's defaults (`conf/predictor/vit.yaml`): depth 6, 16 heads, MLP 2048,
    dropout 0.1, and a learned position embedding over all ``T * P`` tokens -- so position
    carries both which patch and which frame, and the predictor is not free to permute
    frames.
    """

    def __init__(self, dim: int, n_frames: int, n_patches: int, depth: int = 6,
                 heads: int = 16, dim_head: int = 64, mlp_dim: int = 2048,
                 dropout: float = 0.1, checkpointing: bool = False):
        super().__init__()
        self.n_frames, self.n_patches = n_frames, n_patches
        # V-JEPA 2-AC's `use_activation_checkpointing: true`. Their predictor is 24 layers
        # over 8 x 256 = 2048 tokens, and upstream runs it on 32 GPUs with 220 GB each;
        # on one A40 the activations alone exhaust 48 GB. Recomputing them in the backward
        # pass trades roughly a third more time for the memory, and changes no result.
        self.checkpointing = bool(checkpointing)
        self.pos = nn.Parameter(torch.randn(1, n_frames * n_patches, dim) * 0.02)
        self.layers = nn.ModuleList(
            nn.ModuleList([_Attention(dim, heads, dim_head, dropout),
                           _FeedForward(dim, mlp_dim, dropout)]) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("mask", block_causal_mask(n_frames, n_patches),
                             persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, t, p, d = z.shape
        x = z.reshape(b, t * p, d) + self.pos[:, :t * p]
        for attention, feed_forward in self.layers:
            if self.checkpointing and self.training:
                x = checkpoint(lambda y, a=attention: a(y, self.mask), x,
                               use_reentrant=False) + x
                x = checkpoint(feed_forward, x, use_reentrant=False) + x
            else:
                x = attention(x, self.mask) + x
                x = feed_forward(x) + x
        return self.norm(x).reshape(b, t, p, d)


@dataclass
class WorldModelOutput:
    predicted: torch.Tensor  # (B, T-1, P, D) the visual dims only
    target: torch.Tensor
    loss: torch.Tensor
    rollout_loss: torch.Tensor | None = None
    coord_loss: torch.Tensor | None = None
    coord: torch.Tensor | None = None   # (B, T-1, C) canonical, un-standardised


class CoordinateHead(nn.Module):
    """Predicted latent -> anatomy coordinate. The read-out half of M3.

    **Deliberately the same three-layer shape as `atlas_probe.PositionHead`.** M3 is only
    interesting if it differs from the frozen-representation probes in *one* thing --
    whether the coordinate gradient reached the predictor -- and giving it a stronger head
    would confound "the representation changed" with "the head got bigger". R5 through R8
    already established what four different read-out heads on a frozen representation
    score, and all four landed in 0.53-0.55; this one is architecturally the first of them.

    ``coord_mean`` / ``coord_std`` standardise the *target*, and live here as buffers so
    they are saved with the weights. A head restored without them predicts coordinates
    that are wrong by a fixed affine map and fails nowhere -- the same trap `PositionHead`
    registers its input statistics to avoid.

    Patch tokens are mean-pooled before the MLP, which is the ``pooled`` representation the
    probes used. Not an oversight: the spatial-token variant was measured (R6) and scored
    0.538 against pooled's 0.546, so the pooling is not what is losing the position.
    """

    def __init__(self, feature_dim: int, hidden: int = 512, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))
        self.register_buffer("coord_mean", torch.zeros(out_dim))
        self.register_buffer("coord_std", torch.ones(out_dim))

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.coord_mean.copy_(mean.reshape(-1))
        self.coord_std.copy_(std.reshape(-1).clamp_min(1e-6))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, D) -> (B, T, C)``, in standardised units."""
        return self.net(z.mean(dim=-2))

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """The same, in the canonical units the atlas target is expressed in.

        Under `no_grad` like `atlas_probe.PositionHead.predict`, and for the same reason:
        every caller is inference. Without it the head's parameters put `requires_grad` on
        the output, and a planner that cannot call `.numpy()` on its own cost is a planner
        that cannot run -- which is how `atlas_aux_coord` in the closed loop died on its
        first step. Training uses `forward`, never this, so the gradient it needs is
        untouched.
        """
        return self(z) * self.coord_std + self.coord_mean


class LatentWorldModel(nn.Module):
    """Frozen visual features in, next-frame visual features out.

    ``visual`` is what the encoder produced, already detached; this module never holds the
    encoder, so a cached latent and a live one are the same input. Actions are
    concatenated into the feature dimension of every patch (upstream's ``concat_dim=1``),
    which is why the predictor's width is ``feature_dim + action_emb + proprio_emb``.
    """

    def __init__(self, feature_dim: int, n_patches: int, n_frames: int, *,
                 action_dim: int = 3, action_emb_dim: int = 10, proprio_emb_dim: int = 10,
                 depth: int = 6, heads: int = 16, mlp_dim: int = 2048,
                 dropout: float = 0.1, loss_kind: str = "mse", auto_steps: int = 0,
                 normalize_reps: bool = False, checkpointing: bool = False,
                 coord_dim: int = 0, coord_hidden: int = 512, coord_weight: float = 0.0,
                 coord_detach: bool = False, proprio_dim: int = 1):
        super().__init__()
        if loss_kind not in ("mse", "l1"):
            raise ValueError(f"loss_kind must be 'mse' or 'l1', got {loss_kind!r}")
        self.feature_dim, self.n_frames = feature_dim, n_frames
        self.loss_kind, self.auto_steps = loss_kind, int(auto_steps)
        self.normalize_reps = bool(normalize_reps)
        self.action_emb_dim, self.proprio_emb_dim = action_emb_dim, proprio_emb_dim
        self.action_encoder = ActionEncoder(action_dim, action_emb_dim)
        # 1 and fed zeros for every released checkpoint; 3 for the atlas-as-input arm, where
        # the slot carries a canonical coordinate. The default keeps old weights loading
        # bit-for-bit, and the standardisation buffers below default to the identity so a
        # zero in is a zero out.
        self.proprio_dim = int(proprio_dim)
        self.proprio_encoder = ActionEncoder(self.proprio_dim, proprio_emb_dim)
        self.register_buffer("proprio_mean", torch.zeros(self.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(self.proprio_dim))
        self.predictor = ViTPredictor(
            feature_dim + action_emb_dim + proprio_emb_dim, n_frames, n_patches,
            depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
            checkpointing=checkpointing)
        # M3, the anatomy-supervised predictor. Off unless a coordinate dimension is
        # asked for, so every checkpoint trained before this existed still loads and
        # every recipe still means what its paper says. `coord_weight` is lambda; it is
        # kept on the module rather than passed per call so that a checkpoint records
        # the weight it was actually trained under, which is the whole comparison.
        self.coord_weight = float(coord_weight)
        # The control arm. With `coord_detach` the head is built, trained and scored
        # exactly as in M3, on the same windows for the same number of steps -- only the
        # gradient path back into the predictor is cut. That is the one difference the
        # M3 claim is about, so it is the one difference the control should have.
        # Setting lambda to zero instead would be a worse control: the head would then get
        # no gradient either, and the arm would end with an untrained read-out, which
        # confounds "the representation did not change" with "nothing was fitted".
        self.coord_detach = bool(coord_detach)
        self.coord_head = (CoordinateHead(feature_dim, coord_hidden, coord_dim)
                           if coord_dim else None)

    def assemble(self, visual: torch.Tensor, action: torch.Tensor,
                 proprio: torch.Tensor | None = None) -> torch.Tensor:
        """``(B,T,P,D)`` + ``(B,T,A)`` -> ``(B,T,P,D + emb)``, action tiled onto each patch."""
        b, t, p, _ = visual.shape
        if action.shape[:2] != (b, t):
            raise ValueError(
                f"need one action per frame, got {tuple(action.shape[:2])} for {t} frames "
                "-- DINO-WM packs an action with every frame; see `windows(exit_action=True)`"
            )
        if proprio is None:
            if self.proprio_dim != 1:
                raise ValueError(
                    f"this model expects a {self.proprio_dim}-d proprio input (the "
                    "atlas-as-input arm) and none was given; zero-filling it would score "
                    "the arm with its atlas silently removed")
            proprio = visual.new_zeros((b, t, 1))
        else:
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"proprio is {proprio.shape[-1]}-d, model expects {self.proprio_dim}-d")
            proprio = (proprio.to(visual.dtype) - self.proprio_mean) / self.proprio_std
        parts = [visual,
                 self.proprio_encoder(proprio).unsqueeze(2).expand(-1, -1, p, -1),
                 self.action_encoder(action).unsqueeze(2).expand(-1, -1, p, -1)]
        return torch.cat(parts, dim=-1)

    def set_proprio_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Buffers, so the standardisation is saved with the weights (see `CoordinateHead`)."""
        self.proprio_mean.copy_(mean.reshape(-1))
        self.proprio_std.copy_(std.reshape(-1).clamp_min(1e-6))

    def _norm(self, z: torch.Tensor) -> torch.Tensor:
        """V-JEPA 2-AC's ``normalize_reps``: layer-norm both sides before comparing.

        Without it an L1 loss is dominated by whichever feature dimensions happen to have
        the largest scale, and DINOv2's and V-JEPA's differ by more than a factor of two.
        Off for the DINO-WM recipe, which compares raw features.
        """
        return F.layer_norm(z, (z.size(-1),)) if self.normalize_reps else z

    def _distance(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "l1":
            return (predicted - target).abs().mean()
        return F.mse_loss(predicted, target)

    def forward(self, visual: torch.Tensor, action: torch.Tensor,
                proprio: torch.Tensor | None = None,
                coord: torch.Tensor | None = None) -> WorldModelOutput:
        """``coord`` is ``(B, T, C)``, the canonical coordinate of *every* frame in the
        window, un-standardised. Only frames ``1..T-1`` are used -- those are the ones the
        predictor is asked for. Passing it without a coordinate head, or a head without
        it, is an error rather than a silent no-op: both are ways to run the lambda-zero
        control while believing you ran M3.
        """
        z = self.assemble(visual, action, proprio)
        predicted = self._norm(self.predictor(z[:, :-1])[..., :self.feature_dim])
        # The target is the *next* frame's features, detached: the encoder is frozen, and
        # a gradient into it would let the model make the target easier instead of the
        # prediction better.
        target = self._norm(z[:, 1:, :, :self.feature_dim].detach())
        loss = self._distance(predicted, target)

        rollout_loss = None
        if self.auto_steps:
            # V-JEPA 2-AC's second term (`auto_steps: 2`): after the teacher-forced step,
            # keep going on the model's *own* output. Teacher forcing never asks the
            # predictor to survive its own error, and a planner does nothing else.
            imagined = self.rollout(visual[:, :1], action[:, :self.auto_steps],
                                    proprio=proprio)
            steps = imagined.shape[1]
            rollout_loss = self._distance(self._norm(imagined),
                                          target[:, :steps])
            loss = loss + rollout_loss

        # M3: the anatomy coordinate is supervised on the *prediction*, not on the
        # observed frame. That is the whole point. A head fitted on the encoder's own
        # features -- which is every probe from R5 to R8 -- can only read what DINOv2
        # already encodes; a head fitted on `z-hat` sends its gradient back through the
        # predictor, so the representation the planner actually ranks with is the one
        # being shaped. Precedent for adding an auxiliary spatial prediction to a
        # navigation objective: Mirowski et al., "Learning to Navigate in Complex
        # Environments" (ICLR 2017, arXiv:1611.03673), whose agent gains an auxiliary
        # depth-prediction and loop-closure loss on the shared representation, and
        # Jaderberg et al., "Reinforcement Learning with Unsupervised Auxiliary Tasks"
        # (ICLR 2017, arXiv:1611.05397). DINO-WM (arXiv:2411.04983) deliberately does
        # *not* do this -- its claim is that task-agnostic pre-trained features suffice --
        # so M3 is a departure from the baseline recipe and is reported as one.
        coord_loss, coord_out = None, None
        if coord is not None and self.coord_head is None:
            raise ValueError(
                "coordinate targets were passed to a model with no coordinate head, so "
                "they would be silently dropped and the run would report a lambda it "
                "never applied. Build with coord_dim=3 (train_world_model.py "
                "--aux-coord), or stop passing coord.")
        # The other direction -- a head and no targets -- is *not* an error, because that
        # is what inference looks like: `rank_actions.py` and the planner load an M3
        # checkpoint and ask it to predict, with no ground truth anywhere in sight. The
        # training script asserts the targets are present on its side, where the absence
        # would actually mean something went wrong.
        if self.coord_head is not None and coord is not None:
            head = self.coord_head
            standardised = (coord[:, 1:] - head.coord_mean) / head.coord_std
            guess = head(predicted.detach() if self.coord_detach else predicted)
            coord_loss = F.mse_loss(guess, standardised.to(guess.dtype))
            coord_out = guess * head.coord_std + head.coord_mean
            loss = loss + self.coord_weight * coord_loss

        return WorldModelOutput(predicted=predicted, target=target, loss=loss,
                                rollout_loss=rollout_loss, coord_loss=coord_loss,
                                coord=coord_out)

    def rollout(self, visual: torch.Tensor, actions: torch.Tensor,
                proprio: torch.Tensor | None = None) -> torch.Tensor:
        """Imagine forward from a history, with no further observations.

        This is what a planner does and what teacher-forced training never exercises: the
        prediction at step two is fed the model's own step-one output. ``visual`` is the
        history ``(B, H, P, D)``; ``actions`` is ``(B, H + n_steps - 1, A)``.
        """
        history = visual
        out = []
        for step in range(actions.shape[1] - history.shape[1] + 1):
            window = history[:, -self.n_frames:]
            start = max(0, history.shape[1] - self.n_frames)
            a = actions[:, start:start + window.shape[1]]
            # An imagined frame has no measured coordinate, so the atlas-as-input arm
            # holds the one it started planning from across the whole window -- the
            # same convention training used (`collate`), so the two match exactly.
            held = (proprio[:, :1].expand(-1, window.shape[1], -1)
                    if proprio is not None else None)
            z = self.assemble(window, a, held)
            nxt = self.predictor(z)[:, -1:, :, :self.feature_dim]
            out.append(nxt)
            history = torch.cat([history, nxt], dim=1)
        return torch.cat(out, dim=1)


#: Buffers that arrived after checkpoints had already been written, and whose constructor
#: default *is* the behaviour the older checkpoint was trained under. They are the only
#: keys a load is allowed to supply for itself.
BACKFILLED_BUFFERS = ("proprio_mean", "proprio_std")


def load_weights(model: LatentWorldModel, state_dict: dict) -> None:
    """`load_state_dict`, strict except for buffers whose default is the old behaviour.

    `proprio_mean` and `proprio_std` arrived with the atlas-as-input arm. Every checkpoint
    written before it fed the proprio slot a constant zero, which is exactly what the
    zero-mean/unit-std defaults reproduce -- so supplying them is a restatement of how the
    weights were trained, not a guess. Everything else missing or unexpected means the
    weights and the class have genuinely diverged, and that has to stop the run: a strict
    load is what caught this in the first place, and the fix must not turn into
    `strict=False`, which would let a partly initialised predictor go and plan.

    A checkpoint that *uses* the slot has the buffers, because training wrote them. If one
    with `proprio_dim != 1` arrives without them, the standardisation is genuinely unknown
    and identity would silently rescale a coordinate, so that case is refused too.
    """
    report = model.load_state_dict(state_dict, strict=False)
    backfilled = [k for k in report.missing_keys if k in BACKFILLED_BUFFERS]
    missing = [k for k in report.missing_keys if k not in BACKFILLED_BUFFERS]
    if missing or report.unexpected_keys:
        raise RuntimeError(
            "this checkpoint does not match this LatentWorldModel: "
            f"missing {sorted(missing)}, unexpected {sorted(report.unexpected_keys)}")
    if backfilled and model.proprio_dim != 1:
        raise RuntimeError(
            f"the checkpoint has no {sorted(backfilled)} but the model takes a "
            f"{model.proprio_dim}-d proprio input; the standardisation it was trained "
            "with is unknown, and the identity would rescale the coordinate silently")


@torch.no_grad()
def k_way_action_matching(model: LatentWorldModel, transitions, latents,
                          stats, device: str = "cpu", max_fans: int = 0) -> dict:
    """Does the predictor use its action? Scored on the counterfactual fans.

    **Provenance.** The evaluation *construction* is not ours: Shi et al., "Overcoming
    Statistical Bias in Action-Controllable World Models" (arXiv:2608.04653, 2026) build
    Mini-SSMB by applying "multiple actions to the same state and recording their
    corresponding next states", which is exactly a counterfactual fan. Their *metrics* are
    not the ones here -- ARC compares an action against its inverse and against a reference
    action, and Drift Energy penalises motion under a zero action -- because their action
    space is three discrete choices while ours is eight continuous candidates, where
    "which of these did the model predict" is the question that has an answer. So the
    K-way identification below is ours, and `spread` is the closest analogue to their
    Drift Energy: theirs asks whether a model moves when it should not, ours whether it
    moves enough when it should.

    Neither baseline evaluates this at all. DINO-WM reports LPIPS, PSNR, SSIM and MSE --
    reconstruction quality -- plus planning success rate, none of which separates a
    predictor that uses its action from one that has learned the conditional mean.

    One anchor, K actions, K true successors: predict under each action from the *same*
    two-frame window, assign each prediction to its nearest true successor, compare with
    `1/K`. A predictor that ignores the action emits K identical predictions and scores
    at chance by construction -- which is what rollout-only training produces, measured at
    25.2 % on the data it was trained on.

    Two scores, because they answer different questions. ``raw`` is nearest-neighbour on
    the predictions themselves, which is what a planner does and which is dominated by
    whatever the K predictions share. ``centred`` subtracts each set's mean first and so
    asks only about the part that varies with the action. A model can be firmly
    action-conditioned and still score near chance raw, if the error it makes on an unseen
    patient's appearance is larger than the difference an action makes.

    ``stderr`` is the standard error of a proportion under the null over ``assignments``
    independent draws. Debug's val split has forty fans, so its standard error is 3.4
    points and a four-point lead is not a result. Reported so that it cannot be read as
    one.
    """
    from ..envs.generate import TransitionBatch
    from .world_model_data import CachedLatents

    model.eval()
    transitions, latents = Path(transitions), Path(latents)
    raw_hits = centred_hits = total = 0
    spreads, truth_spreads, chances = [], [], []
    # Unlike the ranker, this gate needs each fan's K *successors*, not only its anchor,
    # so a split cached for ranking alone cannot answer it. Such patients are skipped and
    # counted rather than crashing the run or, worse, quietly shrinking the sample.
    cache, skipped = CachedLatents(latents), []
    # `main` has 2 000 fans a patient, so a whole split is ~170 000 forward passes and
    # hours of them for the 24-layer recipe. `max_fans` caps the sample; the standard
    # error it buys is reported alongside, so a capped run states its own resolution
    # instead of implying the full split's.
    shards = sorted(transitions.glob("*.npz"))
    # Evenly across patients, not the first N fans: a prefix would score the split on
    # whichever patients sort first, and patients are the axis this generalises over.
    per_patient = max(1, max_fans // max(1, len(shards))) if max_fans else 0
    for path in shards:
        if path.stem not in cache:
            skipped.append(path.stem)
            continue
        b = TransitionBatch.load(path, with_images=False)
        fans = b.groups()[:per_patient] if per_patient else b.groups()
        try:
            # Ask once, for every frame *these* fans will touch. Validating the whole
            # split's fans instead would reject a cache built for the capped sample --
            # `--frames fans` stores exactly the prefix that will be scored, and checking
            # beyond it skipped every patient and returned an empty result.
            # Gathering them all *at* once would be 24 GB of float32 a patient, so this
            # only validates; the read stays per fan below.
            cache.rows(path.stem, np.unique(np.concatenate(
                [np.concatenate([[b.anchor_index[e[0]]], b.next_index[e]]) for e in fans])))
        except KeyError:
            skipped.append(path.stem)
            continue
        for edges in fans:
            k = len(edges)
            if k < 2:
                continue
            anchor = int(b.anchor_index[edges[0]])
            z = cache.gather(
                path.stem, np.concatenate([[anchor], b.next_index[edges]])).to(device)
            history, truth = z[0], z[1:]
            # A branch anchor is a fresh reset with no past, so the window is two frames
            # and nothing is padded -- the shape `BranchWindowDataset` trains in.
            visual = torch.stack([torch.stack([history, t]) for t in truth])
            action = torch.from_numpy(stats.normalize(b.actions[edges]))
            action = action[:, None].expand(-1, 2, -1).to(device)
            predicted = model(visual, action).predicted[:, -1].flatten(1)

            flat_truth = truth.flatten(1)
            target = torch.arange(k, device=device)
            raw_hits += int((torch.cdist(predicted, flat_truth).argmin(1) == target).sum())
            centred_hits += int((torch.cdist(predicted - predicted.mean(0),
                                             flat_truth - flat_truth.mean(0)
                                             ).argmin(1) == target).sum())
            total += k
            chances.append(1.0 / k)
            spreads.append(float(torch.pdist(predicted).mean()))
            truth_spreads.append(float(torch.pdist(flat_truth).mean()))

    if not chances:
        # Every patient was skipped, which means the cache cannot answer this question at
        # all. Saying so here is the difference between a readable message and a KeyError
        # in whatever formats the result.
        return {"fans": 0, "assignments": 0, "chance": float("nan"),
                "patients_skipped": skipped, "raw": float("nan"), "centred": float("nan"),
                "stderr": float("nan"), "raw_sigma": float("nan"),
                "centred_sigma": float("nan"), "spread": float("nan"),
                "truth_spread": float("nan")}
    chance = float(np.mean(chances))
    stderr = float(np.sqrt(chance * (1.0 - chance) / total))
    raw, centred = raw_hits / total, centred_hits / total
    spread, truth_spread = float(np.mean(spreads)), float(np.mean(truth_spreads))
    return {"fans": len(chances), "assignments": total, "chance": chance,
            "patients_skipped": skipped,
            "raw": raw, "centred": centred, "stderr": stderr,
            "raw_sigma": (raw - chance) / stderr,
            "centred_sigma": (centred - chance) / stderr,
            "spread": spread, "truth_spread": truth_spread,
            "spread_fraction": spread / truth_spread if truth_spread else float("nan")}
