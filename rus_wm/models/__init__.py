"""Models for the current latent-dynamics and anatomical-atlas pipeline.

The active model path is intentionally small:

- ``world_model.py`` implements the action-conditioned latent predictor;
- ``world_model_data.py`` provides rollout windows, branch fans, action statistics, and
  safe access to complete or partial latent caches;
- ``atlas_probe.py`` converts cached representations and frame positions for Atlas
  retrieval and supervised coordinate probes.

Frozen DINOv2 and V-JEPA encoders are external pretrained models. Their weights are
prepared by ``scripts/fetch_encoders.py`` and their outputs are written by
``scripts/cache_latents.py``; this package does not maintain a second trainable image
encoder or the retired single-frame localizer. Import concrete classes from their
submodules so dependencies remain explicit.
"""

__all__: list[str] = []
