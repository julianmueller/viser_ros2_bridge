"""Mesh conversions for viser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shape_msgs.msg import Mesh


def mesh_msg_to_numpy(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``shape_msgs/Mesh`` to vertex and face arrays."""
    vertices = np.array([[vertex.x, vertex.y, vertex.z] for vertex in mesh.vertices], dtype=np.float32)
    faces = np.array([triangle.vertex_indices for triangle in mesh.triangles], dtype=np.uint32)
    if faces.size == 0:
        faces = faces.reshape(0, 3)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mesh faces must have shape (N, 3), got {faces.shape}")
    return vertices, faces


def mesh_like_to_numpy(
    mesh: Mesh | tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert supported mesh inputs to vertex and face arrays."""
    if isinstance(mesh, Mesh):
        return mesh_msg_to_numpy(mesh)
    if isinstance(mesh, dict):
        return np.asarray(mesh["vertices"], dtype=np.float32), np.asarray(mesh["faces"], dtype=np.uint32)
    vertices, faces = mesh
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def mesh_file_to_trimesh(filename: str | Path):
    """Load a mesh file with trimesh."""
    import trimesh

    loaded = trimesh.load(str(filename))
    if isinstance(loaded, trimesh.Scene):
        return loaded.dump(concatenate=True)
    return loaded
