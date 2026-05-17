"""Sensor message conversions for viser."""

from __future__ import annotations

from collections.abc import Sequence
import sys

import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField

from .geometry import color_to_rgb

DEFAULT_POINT_COLOR = (180, 220, 255)

_IMAGE_ENCODINGS: dict[str, tuple[np.dtype, int, str]] = {
    "rgb8": (np.dtype(np.uint8), 3, "rgb"),
    "rgba8": (np.dtype(np.uint8), 4, "rgba"),
    "bgr8": (np.dtype(np.uint8), 3, "bgr"),
    "bgra8": (np.dtype(np.uint8), 4, "bgra"),
    "mono8": (np.dtype(np.uint8), 1, "mono"),
    "8uc1": (np.dtype(np.uint8), 1, "mono"),
    "16uc1": (np.dtype(np.uint16), 1, "mono"),
    "mono16": (np.dtype(np.uint16), 1, "mono"),
    "32fc1": (np.dtype(np.float32), 1, "mono"),
}

_POINT_FIELD_DTYPES = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}


def image_msg_to_numpy(message: Image) -> np.ndarray:
    """Convert ``sensor_msgs/Image`` to an RGB/RGBA image array."""
    dtype, channels, order = _image_encoding_info(message)
    data = np.frombuffer(message.data, dtype=dtype)
    if dtype.itemsize > 1 and bool(message.is_bigendian) != (sys.byteorder == "big"):
        data = data.byteswap()

    row_items = message.step // dtype.itemsize
    rows = data.reshape(message.height, row_items)
    pixels = rows[:, : message.width * channels]
    if channels > 1:
        image = pixels.reshape(message.height, message.width, channels)
    else:
        image = pixels[:, : message.width]
    image = _normalise_to_uint8(image)

    if order == "mono":
        return np.ascontiguousarray(np.repeat(image[:, :, None], 3, axis=2))
    if order in ("bgr", "bgra"):
        image = image[..., [2, 1, 0, 3]] if channels == 4 else image[..., ::-1]
    return np.ascontiguousarray(image)


def point_cloud2_msg_to_numpy(
    message: PointCloud2,
    default_color: Sequence[float] | np.ndarray = DEFAULT_POINT_COLOR,
    skip_nans: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``sensor_msgs/PointCloud2`` to point and color arrays."""
    cloud = _structured_cloud(message)
    names = set(cloud.dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PointCloud2 message must contain x, y, and z fields")

    points = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1).astype(np.float32)
    if "rgb" in names:
        colors = _rgb_field_to_colors(cloud["rgb"])
    elif "rgba" in names:
        colors = _rgb_field_to_colors(cloud["rgba"])
    elif {"r", "g", "b"}.issubset(names):
        colors = np.stack([cloud["r"], cloud["g"], cloud["b"]], axis=1).astype(np.uint8)
    else:
        colors = np.tile(np.asarray(color_to_rgb(default_color), dtype=np.uint8), (points.shape[0], 1))

    if skip_nans:
        mask = np.isfinite(points).all(axis=1)
        points = points[mask]
        colors = colors[mask]
    return np.ascontiguousarray(points), np.ascontiguousarray(colors)


def point_cloud_array_to_numpy(
    cloud: np.ndarray,
    colors: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an ``(N, 3+)`` cloud array to viser point and color arrays."""
    points = np.asarray(cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"cloud must have shape (N, 3+) but got {points.shape}")
    point_colors = _point_colors(points[:, 3:6] if colors is None and points.shape[1] >= 6 else colors, points.shape[0])
    return np.ascontiguousarray(points[:, :3]), np.ascontiguousarray(point_colors)


def downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    stride: int = 1,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply simple stride and max-count downsampling."""
    stride = max(1, int(stride))
    if stride > 1:
        points = points[::stride]
        colors = colors[::stride]
    if max_points is not None and points.shape[0] > max_points:
        step = int(np.ceil(points.shape[0] / max_points))
        points = points[::step]
        colors = colors[::step]
    return points, colors


def _image_encoding_info(message: Image) -> tuple[np.dtype, int, str]:
    encoding = message.encoding.lower()
    if encoding in ("", "passthrough"):
        if message.step == message.width:
            return np.dtype(np.uint8), 1, "mono"
        if message.step == message.width * 3:
            return np.dtype(np.uint8), 3, "rgb"
        if message.step == message.width * 4:
            return np.dtype(np.uint8), 4, "rgba"
    try:
        return _IMAGE_ENCODINGS[encoding]
    except KeyError as exc:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}") from exc


def _normalise_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    low = float(np.nanmin(image[finite]))
    high = float(np.nanmax(image[finite]))
    if np.isclose(low, high):
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image.astype(float) - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _point_cloud_dtype(message: PointCloud2) -> np.dtype:
    byte_order = ">" if message.is_bigendian else "<"
    names = []
    formats = []
    offsets = []
    for field in message.fields:
        if field.datatype not in _POINT_FIELD_DTYPES:
            continue
        dtype = _POINT_FIELD_DTYPES[field.datatype].newbyteorder(byte_order)
        names.append(field.name)
        formats.append(dtype if field.count == 1 else (dtype, (field.count,)))
        offsets.append(field.offset)
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": message.point_step})


def _structured_cloud(message: PointCloud2) -> np.ndarray:
    dtype = _point_cloud_dtype(message)
    dense_row_step = message.point_step * message.width
    if message.row_step == dense_row_step:
        return np.frombuffer(message.data, dtype=dtype, count=message.width * message.height)
    rows = [
        np.frombuffer(message.data, dtype=dtype, count=message.width, offset=row * message.row_step)
        for row in range(message.height)
    ]
    return np.concatenate(rows) if rows else np.array([], dtype=dtype)


def _rgb_field_to_colors(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "f":
        raw = values.astype(np.float32, copy=False).view(np.uint32)
    else:
        raw = values.astype(np.uint32)
    red = ((raw >> 16) & 255).astype(np.uint8)
    green = ((raw >> 8) & 255).astype(np.uint8)
    blue = (raw & 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=1)


def _point_colors(colors: Sequence[float] | np.ndarray | None, count: int) -> np.ndarray:
    if colors is None:
        return np.tile(np.asarray(DEFAULT_POINT_COLOR, dtype=np.uint8), (count, 1))
    array = np.asarray(colors, dtype=float)
    if array.shape == (3,):
        return np.tile(np.asarray(color_to_rgb(array), dtype=np.uint8), (count, 1))
    if array.shape != (count, 3):
        raise ValueError(f"colors must have shape (3,) or ({count}, 3), got {array.shape}")
    if np.nanmax(array) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)
