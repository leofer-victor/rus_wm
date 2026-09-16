from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("rus_wm"))
    default_config = str(package_share / "config" / "deepusnav_console.yaml")
    config = LaunchConfiguration("config")
    inference = Node(
        package="rus_wm",
        executable="deepusnav_inference",
        name="deepusnav_inference",
        output="screen",
        parameters=[config],
    )
    console = Node(
        package="rus_wm",
        executable="deepusnav_console",
        name="deepusnav_console",
        output="screen",
        parameters=[config],
    )

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        inference,
        console,
        RegisterEventHandler(
            OnProcessExit(target_action=console, on_exit=[Shutdown()])
        ),
    ])
