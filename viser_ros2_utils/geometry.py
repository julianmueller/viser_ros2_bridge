"""Lightweight geometry conversions for viser."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Quaternion, Transform, TransformStamped, Vector3
from std_msgs.msg import ColorRGBA
from tf2_utils import conversions as tf2_conv

DEFAULT_COLOR = (255, 180, 60)
DEFAULT_POSITION = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def color_to_rgb(color: ColorRGBA | Sequence[float] | np.ndarray) -> tuple[int, int, int]:
    """Convert a ROS or numeric color to a viser RGB tuple."""
    if isinstance(color, ColorRGBA):
        values = np.array([color.r, color.g, color.b], dtype=float)
    else:
        values = np.asarray(color, dtype=float).reshape(-1)[:3]

    if values.shape != (3,):
        raise ValueError(f"color must contain three RGB values, got shape {values.shape}")
    if np.nanmax(values) <= 1.0:
        values = values * 255.0
    return tuple(np.clip(values, 0, 255).astype(np.uint8).tolist())


def xyzw_to_wxyz(quaternion: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert ROS xyzw quaternion order to viser wxyz order."""
    values = np.asarray(quaternion, dtype=float).reshape(-1)
    if values.shape != (4,):
        raise ValueError(f"quaternion must contain four values, got shape {values.shape}")
    return values[[3, 0, 1, 2]]


def quaternion_to_wxyz(quaternion: Quaternion) -> np.ndarray:
    """Convert a ROS quaternion message to viser wxyz order."""
    return xyzw_to_wxyz(tf2_conv.quat_to_np(quaternion))


def matrix_to_viser_pose(matrix: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transform matrix to viser ``position`` and ``wxyz``."""
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), got {array.shape}")
    quaternion = tf2_conv.quat_to_np(tf2_conv.rot_matrix_to_quat(array[:3, :3]))
    return array[:3, 3].astype(float), xyzw_to_wxyz(quaternion)


def pose_to_viser_pose(pose: Pose | PoseStamped) -> tuple[np.ndarray, np.ndarray]:
    """Convert a ROS pose message to viser ``position`` and ``wxyz``."""
    matrix = tf2_conv.pose_stamped_to_np(pose) if isinstance(pose, PoseStamped) else tf2_conv.pose_to_np(pose)
    return matrix_to_viser_pose(matrix)


def transform_to_viser_pose(transform: Transform | TransformStamped) -> tuple[np.ndarray, np.ndarray]:
    """Convert a ROS transform message to viser ``position`` and ``wxyz``."""
    matrix = (
        tf2_conv.transform_stamped_to_np(transform)
        if isinstance(transform, TransformStamped)
        else tf2_conv.transform_to_np(transform)
    )
    return matrix_to_viser_pose(matrix)


def pose_like_to_viser_pose(
    pose: Pose | PoseStamped | Transform | TransformStamped | Sequence[Sequence[float]] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert common pose-like inputs to viser ``position`` and ``wxyz``."""
    if pose is None:
        return DEFAULT_POSITION.copy(), DEFAULT_WXYZ.copy()
    if isinstance(pose, (Pose, PoseStamped)):
        return pose_to_viser_pose(pose)
    if isinstance(pose, (Transform, TransformStamped)):
        return transform_to_viser_pose(pose)
    return matrix_to_viser_pose(pose)


def point_to_viser_position(point: Point | PointStamped | Vector3 | Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert a point-like value to a viser xyz position."""
    if isinstance(point, PointStamped):
        return tf2_conv.point_to_np(point.point)
    if isinstance(point, Point):
        return tf2_conv.point_to_np(point)
    if isinstance(point, Vector3):
        return tf2_conv.vector3_to_np(point)
    values = np.asarray(point, dtype=float).reshape(-1)
    if values.shape != (3,):
        raise ValueError(f"point must contain three values, got shape {values.shape}")
    return values


def vector_to_arrow_points(
    vector: Vector3 | Sequence[float] | np.ndarray,
    start: Point | Vector3 | Sequence[float] | np.ndarray = DEFAULT_POSITION,
) -> np.ndarray:
    """Convert a vector into viser arrow start/end point data."""
    start_xyz = point_to_viser_position(start)
    vector_xyz = tf2_conv.vector3_to_np(vector) if isinstance(vector, Vector3) else np.asarray(vector, dtype=float)
    vector_xyz = vector_xyz.reshape(-1)
    if vector_xyz.shape != (3,):
        raise ValueError(f"vector must contain three values, got shape {vector_xyz.shape}")
    return np.array([[start_xyz, start_xyz + vector_xyz]], dtype=np.float32)
