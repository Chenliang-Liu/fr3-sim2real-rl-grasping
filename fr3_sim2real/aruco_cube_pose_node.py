#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

from tf2_ros import TransformBroadcaster


class ArucoCubePoseNode(Node):
    def __init__(self):
        super().__init__("aruco_cube_pose_node")

        # =========================
        # 你主要改这里
        # =========================
        self.image_topic = "/camera/camera/color/image_raw"
        self.camera_info_topic = "/camera/camera/color/camera_info"

        self.marker_id = 0              # 你生成的是 id=0
        self.marker_length = 0.04       # ArUco 实际边长，单位 m
        self.cube_size = 0.05           # 立方体边长，单位 m

        self.pose_topic = "/cube_pose_in_camera"
        self.cube_frame = "cube"
        # =========================

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None

        self.pose_pub = self.create_publisher(
            PoseStamped,
            self.pose_topic,
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.aruco_params = cv2.aruco.DetectorParameters()
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.aruco_params
            )
        else:
            self.detector = None

        self.last_print_time = self.get_clock().now()

        self.get_logger().info("Aruco cube pose node started.")
        self.get_logger().info(f"Subscribing image: {self.image_topic}")
        self.get_logger().info(f"Subscribing camera_info: {self.camera_info_topic}")
        self.get_logger().info(f"Publishing pose: {self.pose_topic}")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.camera_frame = msg.header.frame_id

    def detect_markers(self, gray):
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params
            )
        return corners, ids

    def image_callback(self, msg: Image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        corners, ids = self.detect_markers(gray)

        if ids is None:
            return

        ids = ids.flatten()

        # 找指定 ID 的 marker
        target_index = None
        for i, detected_id in enumerate(ids):
            if detected_id == self.marker_id:
                target_index = i
                break

        if target_index is None:
            return

        marker_corners = corners[target_index].reshape(4, 2).astype(np.float32)

        # OpenCV 检测到的角点顺序通常是：
        # top-left, top-right, bottom-right, bottom-left
        s = self.marker_length / 2.0

        object_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

        image_points = marker_corners

        # 优先使用适合正方形平面 marker 的 IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"):
            pnp_flag = cv2.SOLVEPNP_IPPE_SQUARE
        else:
            pnp_flag = cv2.SOLVEPNP_ITERATIVE

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=pnp_flag
        )

        if not success:
            return

        # marker 相对于 camera 的旋转和平移
        R_cam_marker, _ = cv2.Rodrigues(rvec)
        p_cam_marker = tvec.reshape(3, 1)

        # marker 贴在立方体上表面中心时：
        # 立方体中心在 marker 坐标系下沿 -Z 方向偏移 cube_size / 2
        p_marker_cube = np.array([
            [0.0],
            [0.0],
            [0.0]
        ])

        # cube center 在 camera 坐标系下的位置
        p_cam_cube = p_cam_marker + R_cam_marker @ p_marker_cube

        # 这里定义 cube 坐标系和 marker 坐标系方向平行，只是原点移动到 cube center
        quat_cam_cube = R.from_matrix(R_cam_marker).as_quat()
        # scipy 输出顺序：[qx, qy, qz, qw]

        frame_id = "camera_calib_optical_frame"

        # 发布 PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = frame_id

        pose_msg.pose.position.x = float(p_cam_cube[0, 0])
        pose_msg.pose.position.y = float(p_cam_cube[1, 0])
        pose_msg.pose.position.z = float(p_cam_cube[2, 0])

        pose_msg.pose.orientation.x = float(quat_cam_cube[0])
        pose_msg.pose.orientation.y = float(quat_cam_cube[1])
        pose_msg.pose.orientation.z = float(quat_cam_cube[2])
        pose_msg.pose.orientation.w = float(quat_cam_cube[3])

        self.pose_pub.publish(pose_msg)

        # 同时发布 TF：camera_color_optical_frame -> cube
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = frame_id
        tf_msg.child_frame_id = self.cube_frame

        tf_msg.transform.translation.x = pose_msg.pose.position.x
        tf_msg.transform.translation.y = pose_msg.pose.position.y
        tf_msg.transform.translation.z = pose_msg.pose.position.z

        tf_msg.transform.rotation = pose_msg.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)

        # 控制打印频率，避免刷屏
        now = self.get_clock().now()
        if (now - self.last_print_time).nanoseconds > 1_000_000_000:
            self.last_print_time = now
            self.get_logger().info(
                f"cube in {frame_id}: "
                f"p=({pose_msg.pose.position.x:.3f}, "
                f"{pose_msg.pose.position.y:.3f}, "
                f"{pose_msg.pose.position.z:.3f}), "
                f"q=({pose_msg.pose.orientation.x:.3f}, "
                f"{pose_msg.pose.orientation.y:.3f}, "
                f"{pose_msg.pose.orientation.z:.3f}, "
                f"{pose_msg.pose.orientation.w:.3f})"
            )


def main():
    rclpy.init()
    node = ArucoCubePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

        