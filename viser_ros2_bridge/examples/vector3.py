"""Vector3 example."""

from __future__ import annotations

from geometry_msgs.msg import Vector3

from viser_ros2_bridge import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display a Vector3 as an arrow."""
    node.show_vector3(
        "vector3",
        Vector3(x=0.3, y=0.2, z=0.1),
        frame_id=frame,
        color=(80, 180, 255),
    )
