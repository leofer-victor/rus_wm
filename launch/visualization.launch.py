"""Launch the ROS 2 ultrasound capture publisher."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the ultrasound publisher launch description."""
    package_share = Path(get_package_share_directory("rus_wm"))
    default_config = str(package_share / "config" / "deepusnav_console.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("probe_type", default_value="linear"),
        DeclareLaunchArgument(
            "ultrasound_topic",
            default_value="/frame_grabber/us_img",
        ),
        Node(
            package="rus_wm",
            executable="us_screen_pub",
            name="us_screen_pub",
            output="screen",
            parameters=[
                LaunchConfiguration("config"),
                {
                    "probe_type": LaunchConfiguration("probe_type"),
                    "output_topic": LaunchConfiguration("ultrasound_topic"),
                },
            ],
        ),
    ])
