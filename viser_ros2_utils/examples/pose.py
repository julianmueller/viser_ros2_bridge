"""Pose example."""

from __future__ import annotations

from geometry_msgs.msg import Point, Pose

from viser_ros2_utils import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display a Pose."""
    pose = Pose()
    pose.position = Point(x=0.4, y=0.0, z=0.2)
    pose.orientation.w = 1.0
    node.show_pose("pose", pose, frame_id=frame)
