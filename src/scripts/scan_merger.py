#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class ScanMerger(Node):
    def __init__(self):
        super().__init__('scan_merger')
        
        self.num_samples = 360
        self.angle_increment = 2.0 * math.pi / self.num_samples
        self.min_angle = -math.pi
        self.max_angle = math.pi - self.angle_increment
        
        # Buffer to hold ranges. Initialize to inf.
        self.ranges = [float('inf')] * self.num_samples
        
        # We need a frame_id and some reasonable min/max range limits
        self.frame_id = 'base_footprint'
        self.range_min = 0.02
        self.range_max = 4.5
        
        # Subscriptions
        # Topics and their respective yaw offsets (in radians) relative to base_footprint
        self.sensors = {
            '/scan/right': 0.0,
            '/scan/front': math.pi / 2.0,
            '/scan/left': math.pi,
            '/scan/back': -math.pi / 2.0
        }
        
        self.subs = []
        for topic, offset in self.sensors.items():
            sub = self.create_subscription(
                LaserScan,
                topic,
                lambda msg, off=offset: self.scan_callback(msg, off),
                10
            )
            self.subs.append(sub)
            
        # Publisher
        self.publisher = self.create_publisher(LaserScan, '/scan', 10)
        
        # Timer to publish merged scan at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_scan)
        self.get_logger().info('Scan Merger Node started. Merging sonar scans into /scan.')

    def scan_callback(self, msg, yaw_offset):
        # Iterate over rays in the incoming scan
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and r >= msg.range_min and r <= msg.range_max:
                global_angle = angle + yaw_offset
                
                # Wrap to [-pi, pi)
                while global_angle >= math.pi:
                    global_angle -= 2.0 * math.pi
                while global_angle < -math.pi:
                    global_angle += 2.0 * math.pi
                    
                # Find array index
                idx = int(round((global_angle - self.min_angle) / self.angle_increment))
                idx = idx % self.num_samples
                
                # We want the newest data to overwrite old data, but what if there's an obstacle?
                # Actually, to prevent ghost obstacles from lingering, we should clear the sector 
                # before applying the new scan. But a simpler approach is just to overwrite the 
                # current ranges in the buffer with the new ones.
                # Since scans overlap, taking the minimum range is safer for obstacle avoidance.
                # However, to clear out moved obstacles, we can slowly decay ranges to inf, or 
                # clear the buffer entirely in the publish loop. 
                # Let's just overwrite for this sensor's coverage area.
                
                # Actually, just storing the minimum observed range isn't great for dynamic obstacles.
                # Instead, let's just clear the buffer on publish.
                # Wait, if we clear on publish, we might publish a scan missing some sensors if they 
                # haven't arrived yet in that 0.1s window. 
                # Let's keep the buffer, but only update it. If an obstacle moves, the new ray will 
                # report a larger distance, and we should update to the new larger distance!
                # To do this correctly, we can keep 4 separate buffers and combine them on publish.
                pass
            angle += msg.angle_increment
            
        # Better approach: store the latest msg from each sensor
        if not hasattr(self, 'latest_scans'):
            self.latest_scans = {}
        # Find which topic this belongs to based on yaw_offset (hacky but works)
        for topic, offset in self.sensors.items():
            if abs(offset - yaw_offset) < 0.01:
                self.latest_scans[topic] = msg
                break

    def publish_scan(self):
        if not hasattr(self, 'latest_scans') or not self.latest_scans:
            return
            
        merged_ranges = [float('inf')] * self.num_samples
        
        # For each stored scan, project it into the merged_ranges
        for topic, msg in self.latest_scans.items():
            yaw_offset = self.sensors[topic]
            angle = msg.angle_min
            for r in msg.ranges:
                if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                    global_angle = angle + yaw_offset
                    
                    while global_angle >= math.pi:
                        global_angle -= 2.0 * math.pi
                    while global_angle < -math.pi:
                        global_angle += 2.0 * math.pi
                        
                    idx = int(round((global_angle - self.min_angle) / self.angle_increment))
                    idx = idx % self.num_samples
                    
                    # If multiple sensors hit the same angle (overlap), take the min distance
                    if r < merged_ranges[idx]:
                        merged_ranges[idx] = r
                
                angle += msg.angle_increment
                
        # Create and publish the merged scan
        out_msg = LaserScan()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = self.frame_id
        out_msg.angle_min = self.min_angle
        out_msg.angle_max = self.max_angle
        out_msg.angle_increment = self.angle_increment
        out_msg.time_increment = 0.0
        out_msg.scan_time = 0.1
        out_msg.range_min = self.range_min
        out_msg.range_max = self.range_max
        out_msg.ranges = merged_ranges
        # Intensities could be added, but not strictly necessary for SLAM
        out_msg.intensities = []
        
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
