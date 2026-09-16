# rus_wm

ROS 2 Jazzy operator console for DeepUSNav experiments with a Franka Research 3.
This computer performs visualization and learned-model inference.  It does not run the
1 kHz robot loop: the Ubuntu 22 real-time computer owns interpolation, Cartesian/contact
control, robot limits and watchdogs.

## Active files

- `rus_wm/deepusnav_console.py`: ROS 2 node and Qt application.
- `rus_wm/deepusnav_inference.py`: guarded checkpoint inference worker.
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

The **Start inference** button calls `/deepusnav/inference/set_enabled`
(`std_srvs/srv/SetBool`). It is accepted only while both the ultrasound image and robot
pose are fresh and the pose contains finite values with a valid quaternion. The inference
worker repeats the same checks before every model call and pauses if either stream becomes
stale.

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
cd /home/robus/projects/ros_projects/rus_wm
source /home/robus/miniconda3/bin/activate ruswm
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
