#!/usr/bin/env python3
"""
bringup.launch.py — Full simulation: Gazebo + ros2_control controllers.
Replaces the ROS 1 controller.launch + gazebo.launch combination.

Launch sequence:
  1. Gazebo (empty world, paused)
  2. robot_state_publisher + robot spawn  (via gazebo.launch.py)
  3. [3 s delay] joint_state_broadcaster + 4x wheel velocity controllers

Usage:
  ros2 launch minibot bringup.launch.py

Test a wheel (after unpausing Gazebo):
  ros2 topic pub --once /wheel_front_velocity_controller/commands \\
    std_msgs/msg/Float64MultiArray "data: [10.0]"

View wheel velocities:
  ros2 topic echo /joint_states

View sonar readings:
  ros2 topic echo /sonar/front
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = get_package_share_directory('minibot')

    # ── 1. Gazebo + robot model ───────────────────────────────────────
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'gazebo.launch.py')
        )
    )

    # ── 2. Controller spawners ────────────────────────────────────────
    # Delay 3 s to allow libgazebo_ros2_control.so to register
    # controller_manager with ROS 2 before we try to spawn controllers.
    def make_spawner(name):
        """Return a spawner Node for the given controller name."""
        return Node(
            package='controller_manager',
            executable='spawner',
            # Replace spaces in controller names for the node name
            name='spawner_' + name.replace(' ', '_').replace('/', '_'),
            arguments=[name],
            output='screen',
            parameters=[{'use_sim_time': True}],
        )

    controllers = TimerAction(
        period=3.0,
        actions=[
            make_spawner('joint_state_broadcaster'),
            make_spawner('wheel_front_velocity_controller'),
            make_spawner('wheel_right_velocity_controller'),
            make_spawner('wheel_back_velocity_controller'),
            make_spawner('wheel_left_velocity_controller'),
        ]
    )

    # ── 3. Additional Custom Scripts (delayed 7 s) ────────────────────
    custom_scripts = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='minibot',
                executable='omni_teleop.py',
                name='omni_teleop_node',
                output='screen',
                parameters=[{'use_sim_time': True}],
            ),
            Node(
                package='minibot',
                executable='scan_merger.py',
                name='scan_merger_node',
                output='screen',
                parameters=[{'use_sim_time': True}],
            )
        ]
    )

    # ── 4. SLAM and RViz (delayed 5 s) ────────────────────────────────
    slam_and_rviz = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    FindPackageShare('slam_toolbox'),
                    '/launch/online_async_launch.py'
                ]),
                launch_arguments={
                    'use_sim_time': 'true',
                    'slam_params_file': os.path.join(pkg, 'config', 'mapper_params_online_async.yaml')
                }.items()
            ),
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', os.path.join(pkg, 'launch', 'urdf.rviz')],
                parameters=[{'use_sim_time': True}],
            )
        ]
    )

    return LaunchDescription([
        gazebo_launch,
        controllers,
        custom_scripts,
        slam_and_rviz,
    ])
