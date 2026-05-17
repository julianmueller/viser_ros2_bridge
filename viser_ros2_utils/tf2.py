"""TF2 conversions for viser."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

from .geometry import transform_to_viser_pose


def transform_to_frame_data(transform: TransformStamped) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Convert a stamped transform to parent, child, position, and wxyz."""
    parent_frame = transform.header.frame_id or "world"
    child_frame = transform.child_frame_id or "child"
    position, wxyz = transform_to_viser_pose(transform)
    return parent_frame, child_frame, position, wxyz


def tf_message_to_frame_data(message: TFMessage) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Convert all transforms in a TFMessage to viser frame data."""
    return [transform_to_frame_data(transform) for transform in message.transforms]
