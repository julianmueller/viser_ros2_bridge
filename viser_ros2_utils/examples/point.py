"""Point example."""

from __future__ import annotations

from geometry_msgs.msg import Point, PointStamped

from viser_ros2_utils import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display Point and PointStamped values."""
    node.show_point("point", Point(x=0.0, y=0.3, z=0.2), frame_id=frame, color=(255, 220, 80))

    stamped = PointStamped()
    stamped.header.frame_id = frame
    stamped.point = Point(x=0.2, y=0.3, z=0.35)
    node.show_point("point_stamped", stamped, color=(80, 180, 255))
