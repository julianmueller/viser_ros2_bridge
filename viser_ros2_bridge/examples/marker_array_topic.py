"""MarkerArray topic example."""

from __future__ import annotations

from visualization_msgs.msg import MarkerArray

from viser_ros2_bridge import ViserROS2Bridge

MARKER_ARRAY_TOPIC = "/rviz2_utils/markers"


def create(node: ViserROS2Bridge, frame: str, topic: str = MARKER_ARRAY_TOPIC) -> None:
    """Subscribe to a MarkerArray topic."""
    node.subscribe_topic(topic, MarkerArray, name=topic.strip("/") or "marker_array")
