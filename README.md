# rus_wm

ROS 2 Jazzy operator console for DeepUSNav experiments with a Franka Research 3.
This computer performs visualization and learned-model inference.  It does not run the
1 kHz robot loop: the Ubuntu 22 real-time computer owns interpolation, Cartesian/contact
control, robot limits and watchdogs.

## Active files

- `rus_wm/deepusnav_console.py`: ROS 2 node and Qt application.
- `rus_wm/deepusnav_inference.py`: guarded checkpoint inference worker.
- `rus_wm/atlas_runtime.py`: ROS-independent population-Atlas runtime.
- `ui/deepusnav.ui`: editable Qt Designer source.
- `config/deepusnav_console.yaml`: topic, frame, timeout and jog limits.
- `launch/deepusnav_console.launch.py`: ROS 2 launch file.

The former ROS 1 dual-arm/NDI/camera panel is intentionally removed from the active
package.  The console keeps only live ultrasound, FR3 state, CBCT visualization,
rosbag2 recording and guarded Cartesian jog requests.

## Generate the Qt Python binding

Run this from the package directory whenever `ui/deepusnav.ui` changes:

```bash
pyside6-uic ui/deepusnav.ui -o ui/Ui_deepusnav.py
```

The generated file is not the source of truth; edit `deepusnav.ui`, then regenerate it.

## ROS interface contract

Subscriptions are configurable in `config/deepusnav_console.yaml`:

| Data | Default topic | Type |
|---|---|---|
| Ultrasound | `/deepusnav/ultrasound/image` | `sensor_msgs/msg/Image` |
| End-effector pose | `/fr3/state/current_pose` | `geometry_msgs/msg/PoseStamped` |
| Joints | `/fr3/state/joint_states` | `sensor_msgs/msg/JointState` |
| External wrench | `/fr3/state/external_wrench` | `geometry_msgs/msg/WrenchStamped` |
| Robot mode | `/fr3/state/mode` | `std_msgs/msg/String` |
| Inference health | `/deepusnav/inference/status` | `std_msgs/msg/String` |
| Atlas localisation | `/deepusnav/atlas/localisation` | `std_msgs/msg/String` (JSON) |

The **Start inference** button calls `/deepusnav/inference/set_enabled`
(`std_srvs/srv/SetBool`). It is accepted only while both the ultrasound image and robot
pose are fresh and the pose contains finite values with a valid quaternion. The inference
worker repeats the same checks before every model call and pauses if either stream becomes
stale.

When Atlas is enabled, the same guarded inference cycle also passes the current frame
through the frozen V-JEPA2 ViT-L encoder, the trained metric head, and the 90,800-row
retrieval bank. The UI reports:

- the estimated coordinate `u_hat` in the `spine_cpr` canonical frame;
- signed canonical-axis offset to the population L4 target, converted to millimetres;
- expected distance to L4 over the complete k-neighbour posterior;
- posterior mass within the configured L4 tolerance;
- neighbour spread, entropy, patient diversity and inference latency.

`/deepusnav/atlas/localisation` contains a versioned JSON object so it can be recorded in a
rosbag and consumed without a custom ROS message build. Important fields are
`coordinate_u`, `offset_to_target_mm`, `expected_goal_distance_mm`, `target_probability`,
`neighbour_spread_mm`, `within_target`, `k`, `bank_size`, and `latency_ms`.

The expected goal cost is computed over all retrieved neighbour coordinates. It is not
the distance of the posterior mean alone, because neighbours split between L3 and L5 can
otherwise average to a false L4 solution. Atlas coordinates and offsets are anatomical
estimates; they are not coordinates in `fr3_link0` or the tool frame and must not be sent
directly to a robot controller.

The default checkpoint is `dinov2_dino_wm_main.pt`. Its predictor is a latent world model,
not a direct Cartesian controller. With `goal_image_path` empty it therefore runs in
**shadow mode** and reports model health and latency only. If a target ultrasound image is
configured, the node scores seven one-step surface-action candidates against its encoded
goal and publishes the best raw policy action `(dx, dz, dalpha)` on
`/deepusnav/inference/action_proposal` as `std_msgs/msg/Float32MultiArray`.

That proposal topic must not be remapped to `/deepusnav/operator/jog_command`. A separate,
calibrated adapter must convert the SonoGym surface action to the robot/tool frame and
enforce workspace, velocity, force and watchdog limits. Begin with shadow mode.

Each click on a jog button publishes one `geometry_msgs/msg/Vector3Stamped` on
`/deepusnav/operator/jog_command`.  The vector is a **relative translation in metres**;
`header.frame_id` selects base or tool axes.  Using a vector message prevents this
request from being mistaken for a velocity command.  The real-time controller adapter must reject stale,
out-of-workspace or unsafe requests, interpolate accepted requests and publish the
resulting robot state.  Do not connect this topic directly to a velocity controller.

`/fr3/operator/stop` and `/fr3/operator/reset` are `std_srvs/srv/Trigger`.  Stop must be
implemented as an idempotent hold/cancel operation on the controller computer.  It is an
operational stop, not a replacement for certified emergency-stop hardware.

## Build and run

```bash
cd ~/projects/ros_projects/rus_wm
source ~/miniconda3/bin/activate ruswm
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch rus_wm deepusnav_console.launch.py
```

PySide6 and a Qt-compatible VTK Python build must be visible to the Python interpreter
used by ROS 2.  The UI is expected to run in an environment that can also import
`rclpy`, `cv_bridge`, OpenCV and NumPy.

The inference process additionally needs the `deepusnav` environment (PyTorch and the
editable `deepusnav` package). The DINOv2 encoder weights must already exist at
`checkpoints/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`; inference never downloads
weights during an experiment. Configure all paths in `config/deepusnav_console.yaml`.

Atlas additionally requires these local files under the configured `deepusnav_root`:

```text
checkpoints/vjepa2/vitl.pt
references/repos/vjepa2/
training_setup_and_weights/atlas/target_spine_cpr.json
training_setup_and_weights/atlas/heads/vjepa2_vitl__grid4x4__metric.pt
```

The 3.3 GB `features/*.npz.x.npy` cache is a training artifact and is intentionally not
loaded online. The metric checkpoint already contains the trained head, bank embeddings,
bank coordinates and provenance needed at runtime.

Inspect a live result with:

```bash
ros2 topic echo /deepusnav/atlas/localisation std_msgs/msg/String --once
```

## Network boundary

Use the same ROS domain on both computers and constrain DDS discovery to the experiment
network.  Sensor streams use best-effort depth-one QoS to avoid displaying old frames;
jog commands and services use reliable ROS defaults.  Downsample the state bridge to at
most 100 Hz, but keep the 1 kHz FCI loop and its watchdog entirely on the real-time PC.

CBCT is visualization-only.  It must not become an input to a DeepUSNav policy whose
declared deployment observation is ultrasound alone.

## Before phantom motion

- Define and test the sign of base/tool X, Y and Z with the probe out of contact.
- Calibrate flange-to-probe TCP and ultrasound image orientation.
- Confirm tool mass, centre of mass and inertia on the FR3 controller.
- Add workspace, speed, acceleration, contact-force and command-age checks downstream.
- Require a fresh image and robot state before inference or motion.
- Record image, state, wrench, commands, inference output and controller acknowledgements.
- Test loss of DDS, delayed messages, NaNs, controller faults and application exit.
- Begin in shadow mode, then one-step operator-confirmed motion, then short phantom loops.

Human-subject work needs a separate risk analysis, ethical approval, trained supervision,
validated force/pressure limits, hygienic probe handling and an independent safety chain.
