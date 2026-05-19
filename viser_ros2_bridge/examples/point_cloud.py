"""PointCloud2 examples."""

from __future__ import annotations

import numpy as np
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from viser_ros2_bridge import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display NumPy and ROS PointCloud2 data."""
    cloud = np.array(
        [
            [-0.2, -0.2, 0.0, 255, 0, 0],
            [0.2, -0.2, 0.0, 0, 255, 0],
            [0.2, 0.2, 0.0, 0, 0, 255],
            [-0.2, 0.2, 0.0, 255, 255, 0],
        ],
        dtype=np.float32,
    )
    node.show_point_cloud("numpy_cloud", cloud, frame_id=frame, point_size=0.03)

    header = Header(frame_id=frame)
    cloud_msg = point_cloud2.create_cloud_xyz32(
        header,
        [(-0.45, -0.2, 0.05), (-0.35, 0.0, 0.15), (-0.45, 0.2, 0.05)],
    )
    node.show_point_cloud("point_cloud2_msg", cloud_msg, point_size=0.035)
