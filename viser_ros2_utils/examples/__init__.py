"""Command line examples for viser_ros2_utils."""

from __future__ import annotations

import argparse
import time

import rclpy

from viser_ros2_utils import ViserROS2Bridge
from viser_ros2_utils.examples import (
    image,
    marker,
    marker_array,
    marker_array_topic,
    mesh,
    point,
    point_cloud,
    pose,
    transform,
    vector3,
)


MANUAL_EXAMPLES = {
    "image": image.create,
    "marker": marker.create,
    "marker_array": marker_array.create,
    "mesh": mesh.create,
    "point": point.create,
    "point_cloud": point_cloud.create,
    "pose": pose.create,
    "transform": transform.create,
    "vector3": vector3.create,
}
TOPIC_EXAMPLES = {
    "marker_array_topic": marker_array_topic.create,
}
EXAMPLES = {**MANUAL_EXAMPLES, **TOPIC_EXAMPLES}


def create_all(node: ViserROS2Bridge, frame: str) -> None:
    """Create all manual datatype examples."""
    for create in MANUAL_EXAMPLES.values():
        create(node, frame)


def main() -> None:
    """Run one viser_ros2_utils example."""
    parser = argparse.ArgumentParser(description="Display viser_ros2_utils examples.")
    parser.add_argument("example", nargs="?", choices=sorted(EXAMPLES), help="Example to display.")
    parser.add_argument("--all", action="store_true", help="Display all manual datatype examples.")
    parser.add_argument("--host", default="0.0.0.0", help="viser server host address.")
    parser.add_argument("--port", type=int, default=8080, help="viser server port.")
    parser.add_argument("--frame", default="world", help="Frame ID used by the examples.")
    parser.add_argument(
        "--topic", default=marker_array_topic.MARKER_ARRAY_TOPIC, help="MarkerArray topic for topic examples."
    )
    parser.add_argument("--duration", type=float, default=None, help="Seconds to keep spinning; omit to spin forever.")
    parser.add_argument("--tf", action="store_true", help="Subscribe to /tf and /tf_static while running.")
    args = parser.parse_args()

    if args.all and args.example is not None:
        parser.error("pass either an example name or --all, not both")
    if not args.all and args.example is None:
        parser.error("choose an example name or pass --all")

    rclpy.init()
    node = ViserROS2Bridge(
        host=args.host,
        port=args.port,
        start_tf_subscribers=args.tf,
    )
    try:
        if args.all:
            create_all(node, args.frame)
        elif args.example in TOPIC_EXAMPLES:
            TOPIC_EXAMPLES[args.example](node, args.frame, topic=args.topic)
        else:
            MANUAL_EXAMPLES[args.example](node, args.frame)

        if args.duration is None:
            rclpy.spin(node)
        else:
            end_time = time.monotonic() + max(args.duration, 0.0)
            while rclpy.ok() and time.monotonic() < end_time:
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
