"""ROS2 node that owns a viser server and displays common ROS data types."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Transform, TransformStamped, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image, PointCloud2
from shape_msgs.msg import Mesh
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from . import geometry
from . import markers
from . import meshes
from . import scene as scene_tools
from . import sensors
from . import tf2

MESSAGE_TYPES = {
    "Image": Image,
    "sensor_msgs/Image": Image,
    "sensor_msgs/msg/Image": Image,
    "PointCloud2": PointCloud2,
    "sensor_msgs/PointCloud2": PointCloud2,
    "sensor_msgs/msg/PointCloud2": PointCloud2,
    "TFMessage": TFMessage,
    "tf2_msgs/TFMessage": TFMessage,
    "tf2_msgs/msg/TFMessage": TFMessage,
    "TransformStamped": TransformStamped,
    "geometry_msgs/TransformStamped": TransformStamped,
    "geometry_msgs/msg/TransformStamped": TransformStamped,
    "PoseStamped": PoseStamped,
    "geometry_msgs/PoseStamped": PoseStamped,
    "geometry_msgs/msg/PoseStamped": PoseStamped,
    "Pose": Pose,
    "geometry_msgs/Pose": Pose,
    "geometry_msgs/msg/Pose": Pose,
    "PointStamped": PointStamped,
    "geometry_msgs/PointStamped": PointStamped,
    "geometry_msgs/msg/PointStamped": PointStamped,
    "Point": Point,
    "geometry_msgs/Point": Point,
    "geometry_msgs/msg/Point": Point,
    "Vector3": Vector3,
    "geometry_msgs/Vector3": Vector3,
    "geometry_msgs/msg/Vector3": Vector3,
    "Mesh": Mesh,
    "shape_msgs/Mesh": Mesh,
    "shape_msgs/msg/Mesh": Mesh,
    "Marker": Marker,
    "visualization_msgs/Marker": Marker,
    "visualization_msgs/msg/Marker": Marker,
    "MarkerArray": MarkerArray,
    "visualization_msgs/MarkerArray": MarkerArray,
    "visualization_msgs/msg/MarkerArray": MarkerArray,
}


class ViserROS2Bridge(Node):
    """ROS2 node that hosts a viser server and visualizes ROS messages."""

    def __init__(
        self,
        node_name: str = "viser_ros2_bridge",
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        label: str | None = "viser ROS2 bridge",
        scene_root: str = "/ros",
        tf_root: str = "/tf",
        start_tf_subscribers: bool = True,
        tf_topic: str = "/tf",
        tf_static_topic: str = "/tf_static",
        **kwargs: Any,
    ) -> None:
        """Start ROS and viser state for the bridge."""
        super().__init__(node_name, **kwargs)
        import viser

        self.server = viser.ViserServer(host=host, port=port, label=label)
        self.scene = self.server.scene
        self.gui = self.server.gui
        self.scene_root = scene_root
        self.tf_root = tf_root
        self._handles: dict[str, Any] = {}
        self._tf_frame_paths: dict[str, str] = {}
        self._topic_subscriptions: list[Any] = []

        self._set_default_scene_options()
        self.create_service(Trigger, "~/clear", self._on_clear)
        self.create_service(Trigger, "~/republish", self._on_republish)

        self.tf_subscription = None
        self.tf_static_subscription = None
        if start_tf_subscribers:
            self.start_tf_subscribers(tf_topic=tf_topic, tf_static_topic=tf_static_topic)

        self.get_logger().info(f"viser ROS2 bridge ready at http://{host}:{port}")

    @property
    def handles(self) -> dict[str, Any]:
        """Return tracked viser scene handles."""
        return self._handles

    @property
    def tf_frame_paths(self) -> dict[str, str]:
        """Return the ROS frame to viser scene path map."""
        return self._tf_frame_paths

    def start_tf_subscribers(self, tf_topic: str = "/tf", tf_static_topic: str = "/tf_static") -> None:
        """Subscribe to TF topics and visualize incoming transforms."""
        tf_qos = QoSProfile(depth=100)
        tf_static_qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_subscription = self.create_subscription(TFMessage, tf_topic, self._on_tf_message, tf_qos)
        self.tf_static_subscription = self.create_subscription(
            TFMessage,
            tf_static_topic,
            self._on_tf_message,
            tf_static_qos,
        )

    def subscribe_topic(
        self,
        topic: str,
        message_type: type | str,
        *,
        name: str | None = None,
        qos_profile: QoSProfile | int = 10,
        **visualization_kwargs: Any,
    ) -> Any:
        """Subscribe to a ROS topic and visualize each received message."""
        resolved_type = resolve_message_type(message_type)
        display_name = name or topic.strip("/") or resolved_type.__name__
        subscription = self.create_subscription(
            resolved_type,
            topic,
            lambda message: self.visualize_message(display_name, message, **visualization_kwargs),
            qos_profile,
        )
        self._topic_subscriptions.append(subscription)
        return subscription

    def visualize_message(self, name: str, message: Any, **kwargs: Any) -> Any:
        """Visualize one supported ROS message."""
        if isinstance(message, Image):
            return self.show_image(name, message, **kwargs)
        if isinstance(message, PointCloud2):
            return self.show_point_cloud(name, message, **kwargs)
        if isinstance(message, TFMessage):
            return self.show_tf_message(message, **kwargs)
        if isinstance(message, TransformStamped):
            return self.show_transform(message, **kwargs)
        if isinstance(message, (Pose, PoseStamped, Transform)):
            return self.show_frame(name, message, **kwargs)
        if isinstance(message, (Point, PointStamped)):
            return self.show_point(name, message, **kwargs)
        if isinstance(message, Vector3):
            return self.show_vector3(name, message, **kwargs)
        if isinstance(message, MarkerArray):
            return self.show_marker_array(name, message, **kwargs)
        if isinstance(message, Marker):
            return self.show_marker(name, message, **kwargs)
        if isinstance(message, Mesh):
            return self.show_mesh(name, message, **kwargs)
        raise TypeError(f"unsupported message type: {type(message).__name__}")

    def show_frame(
        self,
        name: str,
        pose: Pose | PoseStamped | Transform | TransformStamped | Sequence[Sequence[float]] | np.ndarray | None = None,
        *,
        frame_id: str | None = None,
        attach_to_frame: bool = True,
        axes_length: float = 0.2,
        axes_radius: float = 0.01,
        origin_radius: float | None = None,
        origin_color: ColorRGBA | Sequence[float] | np.ndarray = geometry.DEFAULT_COLOR,
        visible: bool = True,
    ) -> Any:
        """Display a pose-like object as coordinate axes."""
        position, wxyz = geometry.pose_like_to_viser_pose(pose)
        path = self._path_for_message("frames", name, frame_id or self._frame_id(pose), attach_to_frame)
        return scene_tools.add_or_update_frame(
            self.scene,
            self._handles,
            path,
            position=position,
            wxyz=wxyz,
            axes_length=axes_length,
            axes_radius=axes_radius,
            origin_radius=origin_radius,
            origin_color=origin_color,
            visible=visible,
        )

    def show_pose(
        self,
        name: str,
        pose: Pose | PoseStamped | Sequence[Sequence[float]] | np.ndarray,
        **kwargs: Any,
    ) -> Any:
        """Display a pose as coordinate axes."""
        return self.show_frame(name, pose, **kwargs)

    def show_point(
        self,
        name: str,
        point: Point | PointStamped | Vector3 | Sequence[float] | np.ndarray,
        *,
        frame_id: str | None = None,
        attach_to_frame: bool = True,
        radius: float = 0.025,
        color: ColorRGBA | Sequence[float] | np.ndarray = geometry.DEFAULT_COLOR,
        visible: bool = True,
    ) -> Any:
        """Display a point as a small sphere."""
        position = geometry.point_to_viser_position(point)
        path = self._path_for_message("points", name, frame_id or self._frame_id(point), attach_to_frame)
        return scene_tools.add_or_update_point(
            self.scene,
            self._handles,
            path,
            position=position,
            radius=radius,
            color=color,
            visible=visible,
        )

    def show_vector3(
        self,
        name: str,
        vector: Vector3 | Sequence[float] | np.ndarray,
        *,
        frame_id: str | None = None,
        attach_to_frame: bool = True,
        start: Point | Vector3 | Sequence[float] | np.ndarray = geometry.DEFAULT_POSITION,
        color: ColorRGBA | Sequence[float] | np.ndarray = geometry.DEFAULT_COLOR,
        shaft_radius: float = 0.01,
        head_radius: float = 0.025,
        head_length: float = 0.06,
        visible: bool = True,
    ) -> Any:
        """Display a vector as an arrow."""
        points = geometry.vector_to_arrow_points(vector, start=start)
        path = self._path_for_message("vectors", name, frame_id, attach_to_frame)
        return scene_tools.add_or_update_arrows(
            self.scene,
            self._handles,
            path,
            points=points,
            color=color,
            shaft_radius=shaft_radius,
            head_radius=head_radius,
            head_length=head_length,
            visible=visible,
        )

    def show_transform(
        self,
        transform: TransformStamped,
        *,
        axes_length: float = 0.2,
        axes_radius: float = 0.01,
        show_parent_frame: bool = True,
        visible: bool = True,
    ) -> Any:
        """Display one TF transform in the scene tree."""
        parent_frame, child_frame, position, wxyz = tf2.transform_to_frame_data(transform)
        parent_path = scene_tools.frame_path(parent_frame, self._tf_frame_paths, root=self.tf_root)
        child_key = scene_tools.clean_scene_segment(child_frame)
        child_path = scene_tools.join_scene_name(parent_path, child_key)
        previous_path = self._tf_frame_paths.get(child_key)
        if previous_path is not None and previous_path != child_path:
            scene_tools.remove(self._handles, previous_path)
        self._tf_frame_paths[child_key] = child_path

        if show_parent_frame and scene_tools.join_scene_name(parent_path) not in self._handles:
            scene_tools.add_or_update_frame(
                self.scene,
                self._handles,
                parent_path,
                axes_length=axes_length,
                axes_radius=axes_radius,
                visible=visible,
            )
        return scene_tools.add_or_update_frame(
            self.scene,
            self._handles,
            child_path,
            position=position,
            wxyz=wxyz,
            axes_length=axes_length,
            axes_radius=axes_radius,
            visible=visible,
        )

    def show_tf_message(self, message: TFMessage, **kwargs: Any) -> list[Any]:
        """Display all transforms in a TFMessage."""
        return [self.show_transform(transform, **kwargs) for transform in message.transforms]

    def show_image(
        self,
        name: str,
        image: Image | np.ndarray,
        *,
        frame_id: str | None = None,
        attach_to_frame: bool = True,
        render_width: float | None = None,
        render_height: float | None = None,
        image_format: str = "auto",
        jpeg_quality: int | None = None,
        position: Sequence[float] | np.ndarray = geometry.DEFAULT_POSITION,
        wxyz: Sequence[float] | np.ndarray = geometry.DEFAULT_WXYZ,
        visible: bool = True,
    ) -> Any:
        """Display an image message or image array."""
        image_array = sensors.image_msg_to_numpy(image) if isinstance(image, Image) else np.asarray(image)
        path = self._path_for_message("images", name, frame_id or self._frame_id(image), attach_to_frame)
        return scene_tools.add_or_update_image(
            self.scene,
            self._handles,
            path,
            image=image_array,
            render_width=render_width,
            render_height=render_height,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )

    def show_point_cloud(
        self,
        name: str,
        cloud: PointCloud2 | np.ndarray,
        *,
        frame_id: str | None = None,
        attach_to_frame: bool = True,
        colors: Sequence[float] | np.ndarray | None = None,
        default_color: Sequence[float] | np.ndarray = sensors.DEFAULT_POINT_COLOR,
        point_size: float = 0.01,
        point_shape: str = "rounded",
        precision: str = "float32",
        stride: int = 1,
        max_points: int | None = None,
        position: Sequence[float] | np.ndarray = geometry.DEFAULT_POSITION,
        wxyz: Sequence[float] | np.ndarray = geometry.DEFAULT_WXYZ,
        visible: bool = True,
    ) -> Any:
        """Display a PointCloud2 message or point array."""
        if isinstance(cloud, PointCloud2):
            points, point_colors = sensors.point_cloud2_msg_to_numpy(cloud, default_color=default_color)
        else:
            points, point_colors = sensors.point_cloud_array_to_numpy(cloud, colors=colors)
        points, point_colors = sensors.downsample_point_cloud(
            points,
            point_colors,
            stride=stride,
            max_points=max_points,
        )
        path = self._path_for_message("point_clouds", name, frame_id or self._frame_id(cloud), attach_to_frame)
        return scene_tools.add_or_update_point_cloud(
            self.scene,
            self._handles,
            path,
            points=points,
            colors=point_colors,
            point_size=point_size,
            point_shape=point_shape,
            precision=precision,
            position=np.asarray(position, dtype=float),
            wxyz=np.asarray(wxyz, dtype=float),
            visible=visible,
        )

    def show_mesh(
        self,
        name: str,
        mesh: Mesh | tuple[np.ndarray, np.ndarray] | dict[str, np.ndarray],
        *,
        frame_id: str | None = None,
        attach_to_frame: bool = True,
        pose: Pose | PoseStamped | Transform | TransformStamped | Sequence[Sequence[float]] | np.ndarray | None = None,
        color: ColorRGBA | Sequence[float] | np.ndarray = geometry.DEFAULT_COLOR,
        wireframe: bool = False,
        opacity: float | None = None,
        flat_shading: bool = False,
        visible: bool = True,
    ) -> Any:
        """Display a Mesh message or vertex/face arrays."""
        vertices, faces = meshes.mesh_like_to_numpy(mesh)
        position, wxyz = geometry.pose_like_to_viser_pose(pose)
        path = self._path_for_message("meshes", name, frame_id, attach_to_frame)
        return scene_tools.add_or_update_mesh(
            self.scene,
            self._handles,
            path,
            vertices=vertices,
            faces=faces,
            color=color,
            wireframe=wireframe,
            opacity=opacity,
            flat_shading=flat_shading,
            position=position,
            wxyz=wxyz,
            visible=visible,
        )

    def show_mesh_file(
        self,
        name: str,
        filename: str | Path,
        *,
        frame_id: str | None = None,
        attach_to_frame: bool = True,
        pose: Pose | PoseStamped | Transform | TransformStamped | Sequence[Sequence[float]] | np.ndarray | None = None,
        scale: float | tuple[float, float, float] = 1.0,
        visible: bool = True,
    ) -> Any:
        """Display a mesh file supported by trimesh."""
        mesh = meshes.mesh_file_to_trimesh(filename)
        position, wxyz = geometry.pose_like_to_viser_pose(pose)
        path = self._path_for_message("meshes", name, frame_id, attach_to_frame)
        return scene_tools.add_mesh_trimesh(
            self.scene,
            self._handles,
            path,
            mesh=mesh,
            scale=scale,
            position=position,
            wxyz=wxyz,
            visible=visible,
        )

    def show_marker(
        self,
        name: str,
        marker: Marker,
        *,
        attach_to_frame: bool = True,
        visible: bool = True,
    ) -> list[Any]:
        """Display one visualization marker."""
        specs = markers.marker_to_specs(marker)
        return self._show_marker_specs(name, specs, attach_to_frame=attach_to_frame, visible=visible)

    def show_marker_array(
        self,
        name: str,
        marker_array: MarkerArray,
        *,
        attach_to_frame: bool = True,
        visible: bool = True,
    ) -> list[Any]:
        """Display a MarkerArray."""
        specs = markers.marker_array_to_specs(marker_array)
        return self._show_marker_specs(name, specs, attach_to_frame=attach_to_frame, visible=visible)

    def _show_marker_specs(
        self,
        name: str,
        specs: list[dict],
        *,
        attach_to_frame: bool,
        visible: bool,
    ) -> list[Any]:
        handles = []
        prepared_markers = set()
        for spec in specs:
            kind = spec["kind"]
            if kind == "delete_all_markers":
                scene_tools.remove_category(self._handles, "markers")
                continue
            if kind == "delete_marker":
                base_path = self._marker_base_path(name, spec["marker_name"], spec.get("frame_id"), attach_to_frame)
                scene_tools.remove_prefix(self._handles, base_path)
                continue
            frame_id = spec.get("frame_id")
            marker_name = spec["name"]
            base_marker_name = spec.get("marker_name", marker_name)
            if base_marker_name not in prepared_markers:
                self._prepare_marker_update(name, base_marker_name, marker_name, frame_id, attach_to_frame)
                prepared_markers.add(base_marker_name)
            spec = dict(spec)
            spec["visible"] = visible
            scene_name = self._marker_base_path(name, marker_name, frame_id, attach_to_frame)
            handles.append(scene_tools.apply_spec(self.scene, self._handles, scene_name, spec))
        return handles

    def _prepare_marker_update(
        self,
        collection_name: str,
        base_marker_name: str,
        spec_name: str,
        frame_id: str | None,
        attach_to_frame: bool,
    ) -> None:
        base_path = self._marker_base_path(collection_name, base_marker_name, frame_id, attach_to_frame)
        normalized_base = scene_tools.join_scene_name(base_path)
        has_exact = normalized_base in self._handles
        has_children = any(path.startswith(normalized_base + "/") for path in self._handles)
        if has_children or (spec_name != base_marker_name and has_exact):
            scene_tools.remove_prefix(self._handles, normalized_base)

    def _marker_base_path(
        self,
        collection_name: str,
        marker_name: str,
        frame_id: str | None,
        attach_to_frame: bool,
    ) -> str:
        return self._path_for_message("markers", f"{collection_name}/{marker_name}", frame_id, attach_to_frame)

    def remove(self, name: str) -> None:
        """Remove one tracked scene object."""
        scene_tools.remove(self._handles, name)

    def clear(self) -> None:
        """Remove tracked scene objects."""
        scene_tools.clear(self._handles)
        self._tf_frame_paths.clear()
        self._set_default_scene_options()

    def republish(self) -> None:
        """Report current scene state; viser handles remain live."""
        self.get_logger().info(f"viser scene contains {len(self._handles)} tracked handles")

    def destroy_node(self) -> bool:
        """Stop the viser server before destroying the ROS node."""
        stop = getattr(self.server, "stop", None)
        if stop is not None:
            stop()
        return super().destroy_node()

    def _on_tf_message(self, message: TFMessage) -> None:
        try:
            self.show_tf_message(message)
        except Exception as exc:
            self.get_logger().warning(f"failed to display TF message in viser: {exc}")

    def _on_clear(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.clear()
        response.success = True
        response.message = "Cleared tracked viser scene objects."
        return response

    def _on_republish(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.republish()
        response.success = True
        response.message = f"viser scene contains {len(self._handles)} tracked handles."
        return response

    def _set_default_scene_options(self) -> None:
        set_up_direction = getattr(self.scene, "set_up_direction", None)
        if set_up_direction is not None:
            set_up_direction("+z")

    def _path_for_message(self, category: str, name: str, frame_id: str | None, attach_to_frame: bool) -> str:
        if attach_to_frame and frame_id:
            parent_path = scene_tools.frame_path(frame_id, self._tf_frame_paths, root=self.tf_root)
            return scene_tools.join_scene_name(parent_path, category, name)
        return scene_tools.join_scene_name(self.scene_root, category, name)

    @staticmethod
    def _frame_id(message: Any) -> str | None:
        header = getattr(message, "header", None)
        frame_id = getattr(header, "frame_id", None)
        return frame_id or None


def resolve_message_type(message_type: type | str) -> type:
    """Resolve a supported message type class or string."""
    if isinstance(message_type, type):
        return message_type
    try:
        return MESSAGE_TYPES[message_type]
    except KeyError as exc:
        raise ValueError(f"unsupported message type: {message_type!r}") from exc


def main(args: Sequence[str] | None = None) -> None:
    """Run the bridge node."""
    rclpy.init(args=args)
    node = ViserROS2Bridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
