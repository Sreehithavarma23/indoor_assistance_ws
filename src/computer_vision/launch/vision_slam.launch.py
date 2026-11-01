import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='computer_vision',
            executable='object_detector',
            name='object_detector',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
    ])
