#!/usr/bin/env python3
"""Detect a target object with YOLOv8n and report where it is in the world.

Three outputs, because each is useful to a different consumer:

  * TF ``target_bottle`` under the camera's optical frame -- the live, raw
    measurement, for anything that wants the object relative to the sensor.
  * ``~/pose`` (PoseStamped in the map frame) -- the measurement anchored to
    the world. This is the one a navigation client must use. The TF frame
    hangs off ``camera_link_optical``, so the instant the robot drives on, the
    composed map->target_bottle transform drags the object along with the
    camera and points at empty floor; a map-frame pose stays where the bottle
    actually is.
  * ``~/image_annotated`` -- the frame with the box drawn on it, so the
    detection can be watched in RViz instead of inferred from logs.
"""

from cv_bridge import CvBridge, CvBridgeError
import cv2
from geometry_msgs.msg import PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
# Imported for the side effect: it registers the PoseStamped <-> tf2 converter
# that Buffer.transform() dispatches on. Without it the call raises TypeError.
import tf2_geometry_msgs  # noqa: F401
from tf2_ros import TransformBroadcaster
from ultralytics import YOLO

# COCO class index for 'bottle' in the stock YOLOv8 weights.
BOTTLE_CLASS_ID = 39


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('camera_frame', 'camera_link_optical')
        self.declare_parameter('target_frame', 'target_bottle')
        # Half-width (pixels) of the window used to sample depth at the centroid.
        self.declare_parameter('depth_window', 2)
        # Ultralytics resolves a bare filename against its weights cache and
        # downloads yolov8n.pt on first use.
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('target_class_id', BOTTLE_CLASS_ID)
        # Measured on this world: true-positive confidence for the water_bottle
        # model ranges 0.19 to 0.56 and is driven as much by what is behind the
        # bottle as by range -- the same bottle scores 0.48 at 0.9 m against a
        # brick wall, nothing at 1.1 m, and 0.55 at 4 m against a plain one. No
        # class-39 detection above 0.07 was ever seen in a bottle-free frame, so
        # 0.25 buys a great deal of recall for very little risk. The node's
        # consumers are built for an intermittent detector anyway: the router
        # remembers the last map-frame sighting rather than needing a live one.
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('imgsz', 640)
        # 'cpu', or 'cuda:0' when a GPU is available.
        self.declare_parameter('device', 'cpu')
        # Frame the published pose is expressed in. 'map' needs a SLAM or
        # localisation stack running; 'odom' works with the robot alone.
        self.declare_parameter('world_frame', 'map')
        # The frame that bridges the two clocks below. odom is published by the
        # drive plugin at 30 Hz; map->odom comes from RTAB-Map at ~1 Hz.
        self.declare_parameter('fixed_frame', 'odom')
        # How long to wait for the odom leg of the lookup. Gazebo stamps a
        # rendered frame with the sim time it was captured, but under load the
        # transform for that instant can land in the buffer a few hundred
        # milliseconds later; too short a wait silently drops the detection.
        self.declare_parameter('tf_timeout', 1.0)
        # Publishing the annotated frame costs an encode per detection.
        self.declare_parameter('publish_annotated', True)

        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.depth_window = self.get_parameter('depth_window').value
        self.target_class_id = self.get_parameter('target_class_id').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.imgsz = self.get_parameter('imgsz').value
        self.device = self.get_parameter('device').value
        self.world_frame = self.get_parameter('world_frame').value
        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.tf_timeout = self.get_parameter('tf_timeout').value
        self.publish_annotated = self.get_parameter('publish_annotated').value

        model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'Loading YOLO weights: {model_path}')
        self.model = YOLO(model_path)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        # spin_thread=True gives the listener its own executor. Without it the
        # buffer can only fill between callbacks, so any blocking lookup made
        # inside this node's callback deadlocks against itself and times out.
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)

        self.pose_pub = self.create_publisher(PoseStamped, '~/pose', 10)
        self.annotated_pub = self.create_publisher(Image, '~/image_annotated', 1)

        # CPU inference is slower than the 10 Hz camera, so frames that arrive
        # mid-inference are dropped rather than queued behind a stale backlog.
        self.busy = False

        info_sub = Subscriber(
            self, CameraInfo, '/camera/depth/camera_info',
            qos_profile=qos_profile_sensor_data)
        rgb_sub = Subscriber(
            self, Image, '/camera/image_raw',
            qos_profile=qos_profile_sensor_data)
        depth_sub = Subscriber(
            self, Image, '/camera/depth/image_raw',
            qos_profile=qos_profile_sensor_data)

        self.sync = ApproximateTimeSynchronizer(
            [info_sub, rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info('Object detector started')

    def synced_callback(self, info_msg, rgb_msg, depth_msg):
        if self.busy:
            return
        self.busy = True
        try:
            self.process(info_msg, rgb_msg, depth_msg)
        finally:
            self.busy = False

    def process(self, info_msg, rgb_msg, depth_msg):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        bbox = self.detect_object(rgb_image)
        if bbox is None:
            if self.publish_annotated:
                self.publish_overlay(rgb_msg.header, rgb_image, None, None)
            return

        x_min, y_min, x_max, y_max = bbox
        u = (x_min + x_max) // 2
        v = (y_min + y_max) // 2

        depth = self.get_depth(depth_image, depth_msg.encoding, u, v)
        if depth is None:
            self.get_logger().warn(
                f'No valid depth at pixel ({u}, {v})', throttle_duration_sec=2.0)
            return

        # Pinhole deprojection using the intrinsic matrix K = [fx 0 cx; 0 fy cy; 0 0 1].
        fx, fy = info_msg.k[0], info_msg.k[4]
        cx, cy = info_msg.k[2], info_msg.k[5]
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        self.get_logger().info(
            f'{self.target_frame} at pixel ({u}, {v}) -> '
            f'X={x:.3f} Y={y:.3f} Z={z:.3f} m',
            throttle_duration_sec=1.0)

        self.broadcast_target(depth_msg.header.stamp, x, y, z)
        self.publish_world_pose(depth_msg.header.stamp, x, y, z)
        if self.publish_annotated:
            self.publish_overlay(rgb_msg.header, rgb_image, bbox, z)

    def detect_object(self, rgb_image):
        """
        Return a bounding box (x_min, y_min, x_max, y_max) for the target, or None.

        Runs YOLOv8n and keeps the highest-confidence detection of the target class.
        """
        results = self.model.predict(
            rgb_image,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            classes=[self.target_class_id],
            device=self.device,
            verbose=False)
        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        best = int(np.argmax(boxes.conf.cpu().numpy()))
        x_min, y_min, x_max, y_max = boxes.xyxy[best].cpu().numpy()
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def get_depth(self, depth_image, encoding, u, v):
        """Return the median valid depth (metres) around pixel (u, v), or None."""
        height, width = depth_image.shape[:2]
        if not (0 <= u < width and 0 <= v < height):
            return None

        w = self.depth_window
        patch = depth_image[max(v - w, 0):v + w + 1, max(u - w, 0):u + w + 1]
        patch = patch.astype(np.float32)
        if encoding == '16UC1':
            patch /= 1000.0  # millimetres -> metres

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def publish_world_pose(self, stamp, x, y, z):
        """Re-express the camera-frame measurement in the world frame."""
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.camera_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0

        # Two clocks have to be reconciled here. The camera stamp is "now",
        # but map->odom only updates when RTAB-Map finishes a frame, roughly
        # once a second -- so asking for map->camera at the image stamp fails
        # with "extrapolation into the future" almost every time. Asking for
        # it at the latest available time instead would pair the measurement
        # with the wrong robot pose and smear the object across the map while
        # the robot turns.
        #
        # lookup_transform_full splits the chain at odom: the slow map->odom
        # correction is taken at its latest value (it drifts slowly, so this is
        # harmless) while the fast odom->camera leg is taken at the image stamp,
        # where the 30 Hz odometry has real data.
        try:
            transform = self.tf_buffer.lookup_transform_full(
                target_frame=self.world_frame,
                target_time=rclpy.time.Time(),
                source_frame=self.camera_frame,
                source_time=rclpy.time.Time.from_msg(stamp),
                fixed_frame=self.fixed_frame,
                timeout=Duration(seconds=self.tf_timeout))
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f'Cannot express the target in {self.world_frame}: {e}',
                throttle_duration_sec=5.0)
            return

        world_pose = tf2_geometry_msgs.do_transform_pose_stamped(
            pose, transform)
        world_pose.header.frame_id = self.world_frame
        self.pose_pub.publish(world_pose)

    def publish_overlay(self, header, rgb_image, bbox, depth):
        """Republish the frame with the detection drawn on it."""
        image = rgb_image.copy()
        if bbox is None:
            cv2.putText(image, 'no target', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 64, 200), 2)
        else:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                          (0, 220, 0), 2)
            cv2.putText(image, f'{self.target_frame} {depth:.2f} m',
                        (x_min, max(y_min - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        try:
            msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Cannot encode the overlay: {e}')
            return
        msg.header = header
        self.annotated_pub.publish(msg)

    def broadcast_target(self, stamp, x, y, z):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.target_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
