#!/usr/bin/env python3
"""
display.launch.py — RViz2 preview of minibot URDF (no Gazebo).
Loads the xacro, starts joint_state_publisher_gui for manual joint
control, robot_state_publisher for TF, and RViz2.

Usage:
  ros2 launch minibot display.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('minibot')
    xacro_file  = os.path.join(pkg, 'urdf',   'minibot_urdf.xacro')
    rviz_config = os.path.join(pkg, 'launch',  'urdf.rviz')

    robot_description = Command(['xacro ', xacro_file])

    return LaunchDescription([

        DeclareLaunchArgument(
            'use_gui',
            default_value='true',
            description='Launch joint_state_publisher_gui for manual joint sliders'
        ),

        # Publishes /robot_description and TF from URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),

        # Interactive joint-angle sliders (replaces ROS 1 joint_state_publisher_gui)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_gui')),
        ),

        # RViz2 — NOTE: urdf.rviz was saved by RViz 1 and may need
        # re-saving in RViz2. Run without -d to start fresh if needed.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
