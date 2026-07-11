#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class SonarToLaserscan(Node):
    def __init__(self):
        super().__init__('sonar_to_laserscan')
        
        self.pub = self.create_publisher(LaserScan, '/scan', 10)
        
        self.scans = {}
        
        # Configuration for mapping each sonar's local ray frame to base_link
        self.sensor_configs = {
            '/scan/front': {'offset_x': 0.0, 'offset_y': 0.035, 'yaw_offset': math.pi / 2},
            '/scan/back':  {'offset_x': 0.0, 'offset_y': -0.035, 'yaw_offset': -math.pi / 2},
            '/scan/left':  {'offset_x': -0.035, 'offset_y': 0.0, 'yaw_offset': math.pi},
            '/scan/right': {'offset_x': 0.035, 'offset_y': 0.0, 'yaw_offset': 0.0}
        }
        
        self.sub_f = self.create_subscription(LaserScan, '/scan/front', lambda msg: self.scan_cb('/scan/front', msg), 10)
        self.sub_b = self.create_subscription(LaserScan, '/scan/back', lambda msg: self.scan_cb('/scan/back', msg), 10)
        self.sub_l = self.create_subscription(LaserScan, '/scan/left', lambda msg: self.scan_cb('/scan/left', msg), 10)
        self.sub_r = self.create_subscription(LaserScan, '/scan/right', lambda msg: self.scan_cb('/scan/right', msg), 10)
        
        # Publish at 10Hz
        self.timer = self.create_timer(0.1, self.publish_scan)
        self.get_logger().info("Sonar to LaserScan Node Started. Merging 4 sonars to /scan")

    def scan_cb(self, topic, msg):
        self.scans[topic] = msg

    def publish_scan(self):
        # 360 degree laser scan array initialized to infinity
        combined_ranges = [float('inf')] * 360
        
        # Merge all recent scans
        for topic, msg in self.scans.items():
            config = self.sensor_configs[topic]
            ox = config['offset_x']
            oy = config['offset_y']
            yaw = config['yaw_offset']
            
            for i, r in enumerate(msg.ranges):
                # Ignore invalid ranges
                if r < msg.range_min or r > msg.range_max or math.isinf(r) or math.isnan(r):
                    continue
                    
                angle_local = msg.angle_min + i * msg.angle_increment
                angle_base = angle_local + yaw
                
                # Project the hit point into base_link frame
                hit_x = ox + r * math.cos(angle_base)
                hit_y = oy + r * math.sin(angle_base)
                
                # Convert back to polar coordinates from base_link center
                R_base = math.sqrt(hit_x**2 + hit_y**2)
                Phi_base = math.atan2(hit_y, hit_x)
                
                # Map angle to our 360-degree array index
                Phi_base_norm = Phi_base % (2 * math.pi)
                idx = int(round(Phi_base_norm * 180.0 / math.pi)) % 360
                
                combined_ranges[idx] = min(combined_ranges[idx], R_base)
                
        # Publish the combined scan
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'base_link'
        
        scan.angle_min = 0.0
        scan.angle_max = 2 * math.pi - (math.pi / 180.0)
        scan.angle_increment = math.pi / 180.0
        
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        
        scan.range_min = 0.02
        scan.range_max = 4.5
        
        scan.ranges = combined_ranges
        self.pub.publish(scan)

def main():
    rclpy.init()
    node = SonarToLaserscan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
