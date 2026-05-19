"""Mesh examples."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point, Pose
from shape_msgs.msg import Mesh, MeshTriangle

from viser_ros2_bridge import ViserROS2Bridge


def create(node: ViserROS2Bridge, frame: str) -> None:
    """Display NumPy and shape_msgs Mesh data."""
    pose = Pose()
    pose.position = Point(x=0.4, y=0.0, z=0.2)
    pose.orientation.w = 1.0

    vertices = np.array(
        [
            [0.0, 0.0, 0.25],
            [-0.2, -0.2, 0.0],
            [0.2, -0.2, 0.0],
            [0.2, 0.2, 0.0],
            [-0.2, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 4, 3], [1, 3, 2]],
        dtype=np.uint32,
    )
    node.show_mesh("numpy_mesh", (vertices, faces), frame_id=frame, color=(180, 220, 255), pose=pose)

    mesh_msg = Mesh()
    mesh_msg.vertices = [Point(x=-0.15, y=-0.15, z=0.0), Point(x=0.15, y=-0.15, z=0.0), Point(x=0.0, y=0.15, z=0.18)]
    mesh_msg.triangles = [MeshTriangle(vertex_indices=[0, 1, 2])]
    node.show_mesh("shape_mesh", mesh_msg, frame_id=frame, color=(255, 180, 120))
