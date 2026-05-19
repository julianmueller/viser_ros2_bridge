"""Image examples."""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import Image

from viser_ros2_bridge import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display NumPy and ROS Image data."""
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[:, :60] = (255, 80, 80)
    image[:, 60:] = (80, 160, 255)
    node.show_image("numpy_image", image, frame_id=frame, render_width=0.5, position=(0.0, -0.6, 0.4))

    image_msg = Image()
    image_msg.header.frame_id = frame
    image_msg.height = 2
    image_msg.width = 2
    image_msg.encoding = "rgb8"
    image_msg.step = 6
    image_msg.data = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0])
    node.show_image("image_msg", image_msg, render_width=0.25, position=(0.6, -0.6, 0.4))
