"""Marker example."""

from __future__ import annotations

from geometry_msgs.msg import Point, Pose
from rviz2_utils import markers as rviz_markers
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from viser_ros2_bridge import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display individual visualization markers."""
    pose = Pose()
    pose.position = Point(x=0.4, y=0.0, z=0.2)
    pose.orientation.w = 1.0

    marker = rviz_markers.make_sphere(
        pose,
        frame=frame,
        ns="manual_marker",
        id=0,
        scale=0.18,
        color=ColorRGBA(r=1.0, g=0.35, b=0.1, a=1.0),
    )
    node.show_marker("marker", marker)

    text_marker = Marker()
    text_marker.header.frame_id = frame
    text_marker.ns = "manual_text"
    text_marker.id = 3
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    text_marker.pose = pose
    text_marker.scale.z = 0.08
    text_marker.text = "viser marker text"
    node.show_marker("text_marker", text_marker)
