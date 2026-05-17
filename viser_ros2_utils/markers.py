"""Marker conversions for viser."""

from __future__ import annotations

from functools import cache
from math import cos, pi, sin
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray
from .geometry import color_to_rgb, point_to_viser_position, pose_like_to_viser_pose
from .meshes import mesh_file_to_trimesh

MIN_LINE_WIDTH = 1.0
LINE_WIDTH_PIXELS_PER_METER = 200.0


def marker_name(marker: Marker) -> str:
    """Return a stable marker name from namespace and ID."""
    namespace = marker.ns or "marker"
    return f"{namespace}_{marker.id}"


def marker_array_to_specs(marker_array: MarkerArray) -> list[dict]:
    """Convert a MarkerArray to scene specs."""
    specs = []
    for marker in marker_array.markers:
        specs.extend(marker_to_specs(marker))
    return specs


def marker_to_specs(marker: Marker) -> list[dict]:
    """Convert one visualization marker to scene specs."""
    if marker.action == Marker.DELETEALL:
        return [{"kind": "delete_all_markers"}]
    if marker.action == Marker.DELETE:
        return [{"kind": "delete_marker", "marker_name": marker_name(marker), "frame_id": marker.header.frame_id}]

    name = marker_name(marker)
    position, wxyz = pose_like_to_viser_pose(marker.pose)
    color = _marker_color(marker)
    scale = _marker_scale(marker)

    if marker.type == Marker.ARROW:
        specs = [_arrow_spec(marker, name, position, wxyz, color, scale)]
    elif marker.type == Marker.CUBE:
        specs = [_box_spec(marker, name, position, wxyz, color, scale)]
    elif marker.type == Marker.SPHERE:
        specs = [_sphere_spec(marker, name, position, wxyz, color, scale)]
    elif marker.type == Marker.CYLINDER:
        specs = [_cylinder_spec(marker, name, position, wxyz, color, scale)]
    elif marker.type == Marker.LINE_STRIP:
        specs = [_line_strip_spec(marker, name, position, wxyz, color)]
    elif marker.type == Marker.LINE_LIST:
        specs = [_line_list_spec(marker, name, position, wxyz, color)]
    elif marker.type == Marker.POINTS:
        specs = [_points_spec(marker, name, position, wxyz, color)]
    elif marker.type == Marker.CUBE_LIST:
        specs = [_box_list_spec(marker, name, position, wxyz, scale, color)]
    elif marker.type == Marker.SPHERE_LIST:
        specs = [_sphere_list_spec(marker, name, position, wxyz, scale, color)]
    elif marker.type == Marker.TRIANGLE_LIST:
        specs = [_triangle_list_spec(marker, name, position, wxyz, color)]
    elif marker.type == Marker.TEXT_VIEW_FACING:
        specs = [_label_spec(marker, name, position, wxyz)]
    elif marker.type == Marker.MESH_RESOURCE:
        specs = [_mesh_resource_spec(marker, name, position, wxyz, color, scale)]
    else:
        raise ValueError(f"unsupported marker type: {marker.type}")

    for spec in specs:
        spec["marker_name"] = name
    return specs


def _arrow_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color, scale: np.ndarray) -> dict:
    points = _points_array(marker)
    if points.shape[0] >= 2:
        arrow_points = points[:2].reshape(1, 2, 3)
        shaft_radius = max(scale[0] * 0.5, 1e-6)
        head_radius = max(scale[1] * 0.5, shaft_radius)
        head_length = max(scale[2], 1e-6)
    else:
        arrow_points = np.array([[[0.0, 0.0, 0.0], [scale[0], 0.0, 0.0]]], dtype=np.float32)
        shaft_radius = max(scale[1] * 0.5, 1e-6)
        head_radius = max(scale[2] * 0.5, shaft_radius)
        head_length = max(scale[0] * 0.2, 1e-6)
    return {
        "kind": "arrows",
        "name": name,
        "frame_id": marker.header.frame_id,
        "points": arrow_points,
        "color": color,
        "shaft_radius": shaft_radius,
        "head_radius": head_radius,
        "head_length": head_length,
        "position": position,
        "wxyz": wxyz,
    }


def _box_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color, scale: np.ndarray) -> dict:
    return {
        "kind": "box",
        "name": name,
        "frame_id": marker.header.frame_id,
        "dimensions": scale,
        "color": color,
        "position": position,
        "wxyz": wxyz,
    }


def _sphere_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color, scale: np.ndarray) -> dict:
    return {
        "kind": "sphere",
        "name": name,
        "frame_id": marker.header.frame_id,
        "radius": 0.5,
        "scale": scale,
        "color": color,
        "position": position,
        "wxyz": wxyz,
    }


def _cylinder_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color, scale: np.ndarray) -> dict:
    radius = max(scale[0], scale[1]) * 0.5
    cylinder_scale = np.array([scale[0] / max(radius * 2.0, 1e-9), scale[1] / max(radius * 2.0, 1e-9), 1.0])
    return {
        "kind": "cylinder",
        "name": name,
        "frame_id": marker.header.frame_id,
        "radius": radius,
        "height": scale[2],
        "scale": cylinder_scale,
        "color": color,
        "position": position,
        "wxyz": wxyz,
    }


def _line_strip_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color) -> dict:
    marker_points = _points_array(marker)
    if marker_points.shape[0] < 2:
        points = np.zeros((0, 2, 3), dtype=np.float32)
    else:
        points = np.stack([marker_points[:-1], marker_points[1:]], axis=1)
    colors = _line_colors(marker, points, color, line_strip=True)
    return _line_spec(marker, name, points, colors, position, wxyz)


def _line_list_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color) -> dict:
    marker_points = _points_array(marker)
    marker_points = marker_points[: marker_points.shape[0] - marker_points.shape[0] % 2]
    points = marker_points.reshape(-1, 2, 3)
    colors = _line_colors(marker, points, color, line_strip=False)
    return _line_spec(marker, name, points, colors, position, wxyz)


def _line_spec(
    marker: Marker,
    name: str,
    points: np.ndarray,
    colors,
    position: np.ndarray,
    wxyz: np.ndarray,
) -> dict:
    return {
        "kind": "line_segments",
        "name": name,
        "frame_id": marker.header.frame_id,
        "points": points,
        "colors": colors,
        "line_width": _line_width(marker),
        "position": position,
        "wxyz": wxyz,
    }


def _line_width(marker: Marker) -> float:
    if marker.scale.x <= 0.0:
        return MIN_LINE_WIDTH
    return max(marker.scale.x * LINE_WIDTH_PIXELS_PER_METER, MIN_LINE_WIDTH)


def _line_colors(marker: Marker, points: np.ndarray, fallback, line_strip: bool):
    if not marker.colors:
        return fallback

    colors = np.asarray([color_to_rgb(color) for color in marker.colors], dtype=np.uint8)
    if line_strip and colors.shape[0] == points.shape[0] + 1:
        return np.stack([colors[:-1], colors[1:]], axis=1)
    if not line_strip and colors.shape[0] == points.shape[0] * 2:
        return colors.reshape(-1, 2, 3)
    return fallback


def _points_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color) -> dict:
    points = _points_array(marker)
    return {
        "kind": "point_cloud",
        "name": name,
        "frame_id": marker.header.frame_id,
        "points": points,
        "colors": _marker_colors(marker, len(points), color),
        "point_size": marker.scale.x if marker.scale.x > 0.0 else 0.01,
        "position": position,
        "wxyz": wxyz,
    }


def _box_list_spec(
    marker: Marker,
    name: str,
    position: np.ndarray,
    wxyz: np.ndarray,
    scale: np.ndarray,
    color,
) -> dict:
    vertices, faces = _unit_box_mesh()
    return _batched_mesh_spec(marker, name, vertices, faces, position, wxyz, scale, color, flat_shading=True)


def _sphere_list_spec(
    marker: Marker,
    name: str,
    position: np.ndarray,
    wxyz: np.ndarray,
    scale: np.ndarray,
    color,
) -> dict:
    vertices, faces = _unit_sphere_mesh()
    return _batched_mesh_spec(marker, name, vertices, faces, position, wxyz, scale, color, flat_shading=False)


def _batched_mesh_spec(
    marker: Marker,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    position: np.ndarray,
    wxyz: np.ndarray,
    scale: np.ndarray,
    color,
    flat_shading: bool,
) -> dict:
    points = _points_array(marker)
    count = points.shape[0]
    batched_wxyzs = np.zeros((count, 4), dtype=np.float32)
    batched_wxyzs[:, 0] = 1.0
    batched_scales = np.tile(scale.astype(np.float32), (count, 1))
    return {
        "kind": "batched_mesh",
        "name": name,
        "frame_id": marker.header.frame_id,
        "vertices": vertices,
        "faces": faces,
        "batched_positions": points,
        "batched_wxyzs": batched_wxyzs,
        "batched_scales": batched_scales,
        "batched_colors": _marker_colors(marker, count, color),
        "flat_shading": flat_shading,
        "position": position,
        "wxyz": wxyz,
    }


def _triangle_list_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray, color) -> dict:
    vertices = _points_array(marker)
    vertices = vertices[: vertices.shape[0] - vertices.shape[0] % 3]
    faces = np.arange(vertices.shape[0], dtype=np.uint32).reshape(-1, 3)
    return {
        "kind": "mesh",
        "name": name,
        "frame_id": marker.header.frame_id,
        "vertices": vertices,
        "faces": faces,
        "color": color,
        "position": position,
        "wxyz": wxyz,
    }


def _label_spec(marker: Marker, name: str, position: np.ndarray, wxyz: np.ndarray) -> dict:
    return {
        "kind": "label",
        "name": name,
        "frame_id": marker.header.frame_id,
        "text": marker.text,
        "font_scene_height": marker.scale.z if marker.scale.z > 0.0 else 0.075,
        "position": position,
        "wxyz": wxyz,
    }


def _mesh_resource_spec(
    marker: Marker,
    name: str,
    position: np.ndarray,
    wxyz: np.ndarray,
    color,
    scale: np.ndarray,
) -> dict:
    return {
        "kind": "mesh_trimesh",
        "name": name,
        "frame_id": marker.header.frame_id,
        "mesh": mesh_file_to_trimesh(_resolve_mesh_resource(marker.mesh_resource)),
        "scale": tuple(scale.tolist()),
        "color": color,
        "position": position,
        "wxyz": wxyz,
    }


@cache
def _unit_box_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


@cache
def _unit_sphere_mesh() -> tuple[np.ndarray, np.ndarray]:
    radius = 0.5
    latitudes = 8
    longitudes = 16
    vertices = []
    for lat in range(latitudes + 1):
        theta = pi * lat / latitudes
        z = radius * cos(theta)
        ring_radius = radius * sin(theta)
        for lon in range(longitudes):
            phi = 2.0 * pi * lon / longitudes
            vertices.append([ring_radius * cos(phi), ring_radius * sin(phi), z])

    faces = []
    for lat in range(latitudes):
        for lon in range(longitudes):
            a = lat * longitudes + lon
            b = lat * longitudes + (lon + 1) % longitudes
            c = (lat + 1) * longitudes + lon
            d = (lat + 1) * longitudes + (lon + 1) % longitudes
            if lat != 0:
                faces.append([a, c, b])
            if lat != latitudes - 1:
                faces.append([b, c, d])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _points_array(marker: Marker) -> np.ndarray:
    return np.array([point_to_viser_position(point) for point in marker.points], dtype=np.float32)


def _marker_scale(marker: Marker) -> np.ndarray:
    return np.array([marker.scale.x, marker.scale.y, marker.scale.z], dtype=float)


def _marker_color(marker: Marker, index: int | None = None) -> tuple[int, int, int]:
    if index is not None and index < len(marker.colors):
        return color_to_rgb(marker.colors[index])
    return color_to_rgb(marker.color)


def _marker_colors(marker: Marker, count: int, fallback) -> np.ndarray:
    if len(marker.colors) == count:
        return np.asarray([color_to_rgb(color) for color in marker.colors], dtype=np.uint8)
    return np.tile(np.asarray(fallback, dtype=np.uint8), (count, 1))


def _resolve_mesh_resource(resource: str) -> Path:
    if resource.startswith("package://"):
        package_and_path = resource.removeprefix("package://")
        package, _, path = package_and_path.partition("/")
        return Path(get_package_share_directory(package)) / path
    if resource.startswith("file://"):
        return Path(resource.removeprefix("file://"))
    return Path(resource)
