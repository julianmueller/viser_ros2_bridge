"""Small viser scene manipulation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import DEFAULT_COLOR, DEFAULT_POSITION, DEFAULT_WXYZ, color_to_rgb


def clean_scene_segment(value: str) -> str:
    """Return one scene-tree-safe path segment."""
    segment = str(value).strip().strip("/")
    return segment.replace(" ", "_") if segment else "unnamed"


def join_scene_name(*parts: str) -> str:
    """Join scene path parts into an absolute viser scene name."""
    segments = []
    for part in parts:
        segments.extend(clean_scene_segment(piece) for piece in str(part).split("/") if piece)
    return "/" + "/".join(segments or ["unnamed"])


def frame_path(frame_id: str, frame_paths: dict[str, str], root: str = "/tf") -> str:
    """Return the scene path associated with a ROS frame."""
    key = clean_scene_segment(frame_id or "world")
    if key not in frame_paths:
        frame_paths[key] = join_scene_name(root, key)
    return frame_paths[key]


def remove(handles: dict[str, Any], name: str) -> None:
    """Remove one registered scene node."""
    scene_name = join_scene_name(name)
    handle = handles.pop(scene_name, None)
    if handle is not None:
        handle.remove()


def clear(handles: dict[str, Any]) -> None:
    """Remove all registered scene nodes."""
    for handle in list(handles.values()):
        handle.remove()
    handles.clear()


def remove_prefix(handles: dict[str, Any], name_prefix: str) -> None:
    """Remove all scene nodes below a prefix."""
    prefix = join_scene_name(name_prefix)
    for name in list(handles):
        if name == prefix or name.startswith(prefix + "/"):
            remove(handles, name)


def remove_category(handles: dict[str, Any], category: str) -> None:
    """Remove all scene nodes in a named category."""
    needle = "/" + clean_scene_segment(category) + "/"
    for name in list(handles):
        if needle in name or name.endswith("/" + clean_scene_segment(category)):
            remove(handles, name)


def apply_spec(scene: Any, handles: dict[str, Any], name: str, spec: dict) -> Any:
    """Apply one converted scene spec."""
    kind = spec["kind"]
    if kind == "arrows":
        return add_or_update_arrows(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "box":
        return add_or_update_box(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "sphere":
        return add_or_update_sphere(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "cylinder":
        return add_or_update_cylinder(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "line_segments":
        return add_or_update_line_segments(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "point_cloud":
        return add_or_update_point_cloud(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "mesh":
        return add_or_update_mesh(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "batched_mesh":
        return add_or_update_batched_mesh(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    if kind == "mesh_trimesh":
        return add_mesh_trimesh(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id", "color"))
    if kind == "label":
        return add_or_update_label(scene, handles, name, **_kwargs(spec, "kind", "name", "frame_id"))
    raise ValueError(f"unsupported scene spec kind: {kind}")


def add_or_update_frame(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    axes_length: float = 0.2,
    axes_radius: float = 0.01,
    origin_radius: float | None = None,
    origin_color=DEFAULT_COLOR,
    visible: bool = True,
) -> Any:
    """Add or update a viser frame."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "axes_length")
    if handle is None:
        handle = scene.add_frame(
            scene_name,
            axes_length=axes_length,
            axes_radius=axes_radius,
            origin_radius=origin_radius,
            origin_color=color_to_rgb(origin_color),
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.axes_length = axes_length
    handle.axes_radius = axes_radius
    if origin_radius is not None:
        handle.origin_radius = origin_radius
    handle.origin_color = color_to_rgb(origin_color)
    handle.visible = visible
    return handle


def add_or_update_point(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    position: np.ndarray,
    radius: float = 0.025,
    color=DEFAULT_COLOR,
    visible: bool = True,
) -> Any:
    """Add or update a point as an icosphere."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "radius")
    if handle is None:
        handle = scene.add_icosphere(
            scene_name,
            radius=radius,
            color=color_to_rgb(color),
            position=np.asarray(position, dtype=float),
            wxyz=DEFAULT_WXYZ,
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.position = np.asarray(position, dtype=float)
    handle.radius = radius
    handle.color = color_to_rgb(color)
    handle.visible = visible
    return handle


def add_or_update_arrows(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    points: np.ndarray,
    color=DEFAULT_COLOR,
    shaft_radius: float = 0.01,
    head_radius: float = 0.025,
    head_length: float = 0.06,
    line_width: float = 1.0,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update viser arrows."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "shaft_radius")
    if handle is None:
        handle = scene.add_arrows(
            scene_name,
            points=np.asarray(points, dtype=np.float32),
            colors=color_to_rgb(color),
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            head_length=head_length,
            line_width=line_width,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.points = np.asarray(points, dtype=np.float32)
    handle.colors = color_to_rgb(color)
    handle.shaft_radius = shaft_radius
    handle.head_radius = head_radius
    handle.head_length = head_length
    handle.line_width = line_width
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_line_segments(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    points: np.ndarray,
    color=DEFAULT_COLOR,
    colors=None,
    line_width: float = 1.0,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update viser line segments."""
    scene_name = join_scene_name(name)
    line_colors = _line_colors(colors, color)
    handle = _typed_handle(handles, scene_name, "line_width")
    if handle is None:
        handle = scene.add_line_segments(
            scene_name,
            points=np.asarray(points, dtype=np.float32),
            colors=line_colors,
            line_width=line_width,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.points = np.asarray(points, dtype=np.float32)
    handle.colors = line_colors
    handle.line_width = line_width
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_box(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    dimensions: np.ndarray,
    color=DEFAULT_COLOR,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update a viser box."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "dimensions")
    if handle is None:
        handle = scene.add_box(
            scene_name,
            dimensions=np.asarray(dimensions, dtype=float),
            color=color_to_rgb(color),
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.dimensions = np.asarray(dimensions, dtype=float)
    handle.color = color_to_rgb(color)
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_sphere(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    radius: float = 0.5,
    scale: np.ndarray = np.ones(3),
    color=DEFAULT_COLOR,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update a viser sphere."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "radius")
    if handle is None:
        handle = scene.add_icosphere(
            scene_name,
            radius=radius,
            color=color_to_rgb(color),
            scale=np.asarray(scale, dtype=float),
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.radius = radius
    handle.scale = np.asarray(scale, dtype=float)
    handle.color = color_to_rgb(color)
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_cylinder(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    radius: float,
    height: float,
    scale: np.ndarray = np.ones(3),
    color=DEFAULT_COLOR,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update a viser cylinder."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "height")
    if handle is None:
        handle = scene.add_cylinder(
            scene_name,
            radius=radius,
            height=height,
            color=color_to_rgb(color),
            scale=np.asarray(scale, dtype=float),
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.radius = radius
    handle.height = height
    handle.scale = np.asarray(scale, dtype=float)
    handle.color = color_to_rgb(color)
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_point_cloud(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    points: np.ndarray,
    colors: np.ndarray,
    point_size: float = 0.01,
    point_shape: str = "rounded",
    precision: str = "float32",
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update a viser point cloud."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "point_size")
    if handle is None:
        handle = scene.add_point_cloud(
            scene_name,
            points=np.asarray(points, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.uint8),
            point_size=point_size,
            point_shape=point_shape,
            precision=precision,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.precision = precision
    handle.points = np.asarray(points, dtype=np.float32)
    handle.colors = np.asarray(colors, dtype=np.uint8)
    handle.point_size = point_size
    handle.point_shape = point_shape
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_image(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    image: np.ndarray,
    render_width: float | None = None,
    render_height: float | None = None,
    image_format: str = "auto",
    jpeg_quality: int | None = None,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update a viser image plane."""
    scene_name = join_scene_name(name)
    image = np.asarray(image)
    height, width = image.shape[:2]
    render_width = float(render_width if render_width is not None else 1.0)
    render_height = float(render_height if render_height is not None else render_width * height / max(width, 1))
    handle = _typed_handle(handles, scene_name, "image")
    if handle is None:
        handle = scene.add_image(
            scene_name,
            image=np.ascontiguousarray(image),
            render_width=render_width,
            render_height=render_height,
            format=image_format,
            jpeg_quality=jpeg_quality,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.image = np.ascontiguousarray(image)
    handle.render_width = render_width
    handle.render_height = render_height
    handle.format = image_format
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_mesh(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color=DEFAULT_COLOR,
    wireframe: bool = False,
    opacity: float | None = None,
    flat_shading: bool = False,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update a simple viser mesh."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "vertices")
    if handle is None:
        handle = scene.add_mesh_simple(
            scene_name,
            vertices=np.asarray(vertices, dtype=np.float32),
            faces=np.asarray(faces, dtype=np.uint32),
            color=color_to_rgb(color),
            wireframe=wireframe,
            opacity=opacity,
            flat_shading=flat_shading,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.vertices = np.asarray(vertices, dtype=np.float32)
    handle.faces = np.asarray(faces, dtype=np.uint32)
    handle.color = color_to_rgb(color)
    handle.wireframe = wireframe
    handle.opacity = opacity
    handle.flat_shading = flat_shading
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_or_update_batched_mesh(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    batched_positions: np.ndarray,
    batched_wxyzs: np.ndarray,
    batched_scales: np.ndarray | None = None,
    batched_colors=DEFAULT_COLOR,
    lod: str = "auto",
    wireframe: bool = False,
    opacity: float | None = None,
    flat_shading: bool = False,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update batched viser meshes."""
    scene_name = join_scene_name(name)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    batched_positions = np.asarray(batched_positions, dtype=np.float32)
    batched_wxyzs = np.asarray(batched_wxyzs, dtype=np.float32)
    if batched_scales is not None:
        batched_scales = np.asarray(batched_scales, dtype=np.float32)
    batched_colors = np.asarray(batched_colors, dtype=np.uint8)

    handle = _typed_handle(handles, scene_name, "batched_positions")
    if handle is None:
        handle = scene.add_batched_meshes_simple(
            scene_name,
            vertices=vertices,
            faces=faces,
            batched_positions=batched_positions,
            batched_wxyzs=batched_wxyzs,
            batched_scales=batched_scales,
            batched_colors=batched_colors,
            lod=lod,
            wireframe=wireframe,
            opacity=opacity,
            flat_shading=flat_shading,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.vertices = vertices
    handle.faces = faces
    handle.batched_positions = batched_positions
    handle.batched_wxyzs = batched_wxyzs
    handle.batched_scales = batched_scales
    handle.batched_colors = batched_colors
    handle.lod = lod
    handle.wireframe = wireframe
    handle.opacity = opacity
    handle.flat_shading = flat_shading
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def add_mesh_trimesh(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    mesh: Any,
    scale: float | tuple[float, float, float] = 1.0,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Replace a scene node with a trimesh-backed mesh."""
    scene_name = join_scene_name(name)
    remove(handles, scene_name)
    handle = scene.add_mesh_trimesh(
        scene_name,
        mesh,
        scale=scale,
        position=np.asarray(position, dtype=float),
        wxyz=np.asarray(wxyz, dtype=float),
        visible=visible,
    )
    handles[scene_name] = handle
    return handle


def add_or_update_label(
    scene: Any,
    handles: dict[str, Any],
    name: str,
    text: str,
    font_scene_height: float = 0.075,
    position: np.ndarray = DEFAULT_POSITION,
    wxyz: np.ndarray = DEFAULT_WXYZ,
    visible: bool = True,
) -> Any:
    """Add or update a viser label."""
    scene_name = join_scene_name(name)
    handle = _typed_handle(handles, scene_name, "text")
    if handle is None:
        handle = scene.add_label(
            scene_name,
            text=text,
            font_size_mode="scene",
            font_scene_height=font_scene_height,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )
        handles[scene_name] = handle
        return handle

    handle.text = text
    handle.font_scene_height = font_scene_height
    handle.position = np.asarray(position, dtype=float)
    handle.wxyz = np.asarray(wxyz, dtype=float)
    handle.visible = visible
    return handle


def _line_colors(colors, fallback):
    if colors is None:
        return color_to_rgb(fallback)
    if isinstance(colors, np.ndarray):
        return colors.astype(np.uint8, copy=False)
    return color_to_rgb(colors)


def _kwargs(values: dict, *skip: str) -> dict:
    skipped = set(skip) | {'marker_name'}
    return {key: value for key, value in values.items() if key not in skipped}


def _typed_handle(handles: dict[str, Any], scene_name: str, attribute: str) -> Any | None:
    handle = handles.get(scene_name)
    if handle is None:
        return None
    if hasattr(handle, attribute):
        return handle
    remove(handles, scene_name)
    return None
