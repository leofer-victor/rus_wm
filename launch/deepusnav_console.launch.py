"""Launch the DeepUSNav console and its ROS 2 worker nodes."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create a launch description with runtime-selectable inference models."""
    package_share = Path(get_package_share_directory("rus_wm"))
    default_config = str(package_share / "config" / "deepusnav_console.yaml")

    config = LaunchConfiguration("config")
    inference = Node(
        package="rus_wm",
        executable="deepusnav_inference",
        name="deepusnav_inference",
        output="screen",
        parameters=[
            config,
            {
                "model_profile": LaunchConfiguration("model"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "device": LaunchConfiguration("device"),
            },
        ],
    )
    console = Node(
        package="rus_wm",
        executable="deepusnav_console",
        name="deepusnav_console",
        output="screen",
        parameters=[config],
    )
    jog_adapter = Node(
        package="rus_wm",
        executable="deepusnav_jog_adapter",
        name="deepusnav_jog_adapter",
        output="screen",
        parameters=[config],
    )
    us_screen_pub = Node(
        package="rus_wm",
        executable="us_screen_pub",
        name="us_screen_pub",
        output="screen",
        parameters=[
            config,
            {
                "probe_type": LaunchConfiguration("probe_type"),
                "output_topic": LaunchConfiguration("ultrasound_topic"),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("probe_type", default_value="linear"),
        DeclareLaunchArgument(
            "ultrasound_topic",
            default_value="/frame_grabber/us_img",
        ),
        DeclareLaunchArgument(
            "model",
            default_value="dino",
            choices=[
                "dino",
                "vjepa",
                "vjeap",
                "atlas",
                "dino_atlas",
                "vjepa_atlas",
                "custom",
            ],
            description="Inference model profile selected at process startup",
        ),
        DeclareLaunchArgument(
            "checkpoint_path",
            default_value="",
            description="Optional world-model checkpoint override",
        ),
        DeclareLaunchArgument("device", default_value="cuda"),
        inference,
        jog_adapter,
        console,
        us_screen_pub,
        RegisterEventHandler(
            OnProcessExit(target_action=console, on_exit=[Shutdown()])
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=jog_adapter, on_exit=[Shutdown()])
        ),
    ])
