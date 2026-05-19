"""MarkerArray example."""

from __future__ import annotations

from geometry_msgs.msg import Point, Pose, Vector3
from rviz2_utils import markers as rviz_markers
from std_msgs.msg import ColorRGBA

from viser_ros2_bridge import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display a visualization MarkerArray."""
    pose = Pose()
    pose.position = Point(x=0.4, y=0.0, z=0.2)
    pose.orientation.w = 1.0

    marker_array = rviz_markers.make_marker_array(
        [
            rviz_markers.make_sphere(
                pose,
                frame=frame,
                ns="manual_marker_array",
                id=0,
                scale=0.18,
                color=ColorRGBA(r=1.0, g=0.35, b=0.1, a=1.0),
            ),
            rviz_markers.make_cuboid(
                pose,
                frame=frame,
                ns="manual_marker_array",
                id=1,
                scale=(0.12, 0.18, 0.08),
                color=ColorRGBA(r=0.2, g=0.8, b=1.0, a=1.0),
            ),
            rviz_markers.make_wireframe(
                pose,
                Vector3(x=0.4, y=0.25, z=0.2),
                frame=frame,
                ns="manual_marker_array",
                id=2,
                line_width=0.01,
            ),
        ]
    )
    node.show_marker_array("marker_array", marker_array)
