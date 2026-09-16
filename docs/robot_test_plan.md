# DeepUSNav FR3 test plan

## Responsibility boundary

The Ubuntu 24 / Jetson computer owns image acquisition, UI, logging, inference and
low-rate decisions.  The Ubuntu 22 real-time computer owns FCI, the 1 kHz loop,
trajectory interpolation, contact control, collision thresholds, workspace limits and
the final authority to accept or reject every request.

Never route the UI jog topic directly into an FCI velocity or torque interface.

## Interfaces still required on the real-time computer

1. A state bridge publishing pose, joints, external wrench and controller mode with the
   exact message types in `README.md`.
2. A jog adapter consuming one relative `Vector3Stamped`, transforming base/tool-frame
   requests, checking freshness and limits, and producing a smooth bounded trajectory.
3. Idempotent stop and reset services.  Stop cancels motion and holds safely; reset must
   not move unless the controller's own preconditions pass.
4. A watchdog that stops on stale commands, stale perception, DDS loss, controller
   exceptions or invalid numeric values.
5. A controller acknowledgement/status topic containing accepted/rejected, reason,
   request identifier and final achieved displacement.  This should be added before
   autonomous model commands are enabled.

## Current inference scope and remaining work

The downloaded world-model checkpoints predict a successor latent; they are not a robot
policy. The supplied inference node now provides the matching grayscale/letterbox/ImageNet
preprocessing, frozen visual encoder, one-step World Model proposals, and single-frame
population-Atlas retrieval. Before motion, the deployment still needs:

- a four-frame history aligned with the actions that were actually achieved;
- candidate action generation within the training support;
- a calibrated mapping from the SonoGym surface action to bounded robot/tool motion;
- validated confidence/OOD thresholds and a no-motion result when confidence is
  insufficient;
- inference latency, selected action and health diagnostics for the UI and rosbag2.

The Atlas panel is diagnostic. `u_hat`, the L4 offset, neighbour spread and target
probability are estimates in a population canonical anatomy; none is an FR3 pose. The
current single-frame Atlas has a recorded held-out median localisation error of about
26.3 mm, so `within_target` is a planning signal to validate on the phantom, not an
arrival certificate.

Before every Atlas session:

1. Run `sha256sum -c SHA256SUMS.runtime` inside
   `training_setup_and_weights/atlas`.
2. Confirm that the status panel names `vjepa2_vitl`, `grid4x4`, `spine_cpr`, and a
   90,800-row bank.
3. Replay a recorded ultrasound bag with robot output disconnected and confirm that
   `/deepusnav/atlas/localisation` is recorded at the configured inference rate.
4. Check a labelled phantom sweep: the reported cranio-caudal trend must follow the
   sweep direction, and repeated stationary frames must remain stable.
5. Treat stale results, non-finite values, high neighbour spread or an inference exception
   as no-motion conditions downstream.

CBCT may be displayed and used to establish phantom evaluation ground truth.  It must not
be passed to a policy evaluated under the ultrasound-only protocol.

## Calibration and acquisition

- Calibrate `T_flange_probe`, including probe acoustic origin and image-plane axes.
- Confirm base/tool X/Y/Z signs with the probe out of contact.
- Configure tool mass, centre of mass and inertia in the FR3 stack.
- Fix ultrasound depth, gain, TGC, dynamic range, orientation and frame-grabber crop.
- Measure end-to-end image latency and synchronize both hosts with PTP or chrony.
- Record receive time as well as source time; never use a stale frame for a new action.
- Estimate the achieved probe-frame displacement from robot state and feed that value to
  the world-model history rather than assuming requested equals executed.

## Phantom progression

1. Replay-only: play rosbag2 data through UI and inference with no controller connected.
2. Free-space jog: verify all signs, frames, limits, stop, reset and network-loss behavior.
3. Contact characterization: establish conservative normal-force and tangential-motion
   limits with the model disabled.
4. Shadow mode: model ranks actions while an operator controls motion; compare its choice
   with the measured successor images.
5. Branch test: repeatedly return to one anchor and execute candidate actions to measure
   real-image K-way matching and ranking against random/copy-last baselines.
6. Confirmed one-step mode: the operator approves each model-selected action.
7. Short closed loop: fixed small workspace, small step, strict timeout and independent
   stop operator.

Advance only if logs are complete, communication failure always yields no new motion,
contact remains within validated limits, and real-ultrasound action ranking is reliably
above chance.

## Before any human study

- Institutional ethics approval, consent and a written clinical/research protocol.
- A task-specific risk assessment and validated independent safety chain.
- Defined force/pressure, speed, workspace, duration and skin-shear limits.
- Probe hygiene, electrical safety and medical-device responsibilities.
- Trained robot and ultrasound operators with immediate physical stop access.
- Exclusion criteria and automatic/manual termination criteria.
- Privacy controls for ultrasound, CBCT, robot logs and identifiers.
- A frozen model/configuration and a documented rollback procedure.

Good phantom performance is necessary evidence, not sufficient authorization for a human
experiment.
