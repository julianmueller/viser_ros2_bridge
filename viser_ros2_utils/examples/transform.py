"""TransformStamped example."""

from __future__ import annotations

from geometry_msgs.msg import TransformStamped

from viser_ros2_utils import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display a TransformStamped."""
    transform = TransformStamped()
    transform.header.frame_id = frame
    transform.child_frame_id = "tool0"
    transform.transform.translation.x = 0.25
    transform.transform.translation.y = -0.35
    transform.transform.rotation.w = 1.0
    node.show_transform(transform)
