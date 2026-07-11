#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math

from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class OmniTeleopNode(Node):
    def __init__(self):
        super().__init__('omni_teleop_node')
        
        # Subscribing to cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Subscriber for Ground Truth Odometry to broadcast TF
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # ros2_control JointGroupVelocityController subscribes with BEST_EFFORT.
        # A RELIABLE publisher is QoS-incompatible → messages silently dropped.
        _qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Publishers to the 4 wheel controllers
        self.pub_front = self.create_publisher(Float64MultiArray, '/wheel_front_velocity_controller/commands', _qos)
        self.pub_back  = self.create_publisher(Float64MultiArray, '/wheel_back_velocity_controller/commands', _qos)
        self.pub_right = self.create_publisher(Float64MultiArray, '/wheel_right_velocity_controller/commands', _qos)
        self.pub_left  = self.create_publisher(Float64MultiArray, '/wheel_left_velocity_controller/commands', _qos)
        
        # Robot physical parameters
        self.R = 0.024  # wheel radius in meters (48mm dia)
        self.L = 0.045  # distance from center to wheel axis in meters

        # Target velocities
        self.target_v_x = 0.0
        self.target_v_y = 0.0
        self.target_omega_z = 0.0

        # Velocity smoothing state
        self.current_v_x = 0.0
        self.current_v_y = 0.0
        self.current_omega_z = 0.0
        self.alpha = 0.15  # Smoothing factor (lower = smoother/slower acceleration)
        
        # Control loop timer (50 Hz)
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('Omni teleop node started! Listening to /cmd_vel and /odom...')

    def odom_callback(self, msg: Odometry):
        # Broadcast the odom -> base_link transform for RViz visualization
        t = TransformStamped()
        
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        
        t.transform.rotation = msg.pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)

    def cmd_vel_callback(self, msg: Twist):
        self.target_v_x = msg.linear.x
        self.target_v_y = msg.linear.y
        self.target_omega_z = msg.angular.z

    def control_loop(self):
        # Apply low-pass filter to prevent instant high-acceleration wheelies
        self.current_v_x = (self.alpha * self.target_v_x) + ((1.0 - self.alpha) * self.current_v_x)
        self.current_v_y = (self.alpha * self.target_v_y) + ((1.0 - self.alpha) * self.current_v_y)
        self.current_omega_z = (self.alpha * self.target_omega_z) + ((1.0 - self.alpha) * self.current_omega_z)

        # In standard ROS, linear.x is FORWARD, linear.y is LEFT.
        # But in this specific URDF, the robot's physical front (where sensor_front_1 is) is along the +Y axis.
        # Therefore, we must map FORWARD (linear.x) to +Y motion, and LEFT (linear.y) to -X motion.
        v_y = self.current_v_x  
        v_x = -self.current_v_y 
        omega_z = self.current_omega_z

        # Correct holonomic kinematics derived from the no-slip constraint:
        #   omega = (ay * vwx - ax * vwy) / R
        # where (ax, ay) is the joint axis unit vector and (vwx, vwy) is the
        # wheel-centre velocity: vwx = vx - wz*yw,  vwy = vy + wz*xw
        #
        # Joint axes in this URDF:
        #   front_joint : axis = (0, +1, 0)  pos = ( 0, +L)
        #   back_joint  : axis = (0, -1, 0)  pos = ( 0, -L)
        #   right_joint : axis = (+1,  0, 0) pos = (+L,  0)
        #   left_joint  : axis = (-1,  0, 0) pos = (-L,  0)

        # Front  (ax=0, ay=+1, xw=0, yw=+L)
        omega_front = (v_x - omega_z * self.L) / self.R

        # Back   (ax=0, ay=-1, xw=0, yw=-L)
        #   vwx = vx - wz*(-L) = vx + wz*L
        #   omega = (-1)*vwx / R = -(vx + wz*L) / R
        omega_back = -(v_x + omega_z * self.L) / self.R

        # Right  (ax=+1, ay=0, xw=+L, yw=0)
        #   vwy = vy + wz*L
        #   omega = -ax*vwy / R = -(vy + wz*L) / R
        omega_right = -(v_y + omega_z * self.L) / self.R

        # Left   (ax=-1, ay=0, xw=-L, yw=0)
        #   vwy = vy + wz*(-L) = vy - wz*L
        #   omega = -ax*vwy / R = -(-1)*(vy - wz*L) / R = (vy - wz*L) / R
        omega_left = (v_y - omega_z * self.L) / self.R
        
        # Publish commands
        msg_front = Float64MultiArray(data=[omega_front])
        msg_back = Float64MultiArray(data=[omega_back])
        msg_right = Float64MultiArray(data=[omega_right])
        msg_left = Float64MultiArray(data=[omega_left])
        
        self.pub_front.publish(msg_front)
        self.pub_back.publish(msg_back)
        self.pub_right.publish(msg_right)
        self.pub_left.publish(msg_left)

def main(args=None):
    rclpy.init(args=args)
    node = OmniTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
