# Viser ROS2 Utils

[Viser](https://viser.studio/main) is a 3D visualization library for computer vision and robotics in Python. This package adds a small ROS2 bridge node plus lightweight conversion helpers for displaying common ROS messages in a viser scene.

## File structure

- `viser_ros2_utils/node.py`: `ViserROS2Bridge`, the ROS2 node that owns the `viser.ViserServer`, subscriptions, services, and manual visualization methods.
- `viser_ros2_utils/geometry.py`: lightweight geometry conversions that return viser-ready `position`, `wxyz`, color, and arrow data.
- `viser_ros2_utils/tf2.py`: TF message conversions to parent frame, child frame, `position`, and `wxyz` data.
- `viser_ros2_utils/sensors.py`: `Image` and `PointCloud2` conversions to NumPy image, point, and color arrays.
- `viser_ros2_utils/meshes.py`: `shape_msgs/Mesh` and mesh file conversions to vertex/face or trimesh data.
- `viser_ros2_utils/markers.py`: `visualization_msgs/Marker` and `MarkerArray` conversions to viser scene specs.
- `viser_ros2_utils/scene.py`: viser scene and handle manipulation helpers used by the node.

## Quick start

```bash
ros2 run viser_ros2_utils viser_bridge
```

The bridge starts a viser server on `http://0.0.0.0:8080`, listens to `/tf` and `/tf_static`, and exposes `~/clear` and `~/republish` services.

```python
from viser_ros2_utils import ViserROS2Bridge

node = ViserROS2Bridge(host='0.0.0.0', port=8080)
node.subscribe_topic('/camera/color/image_raw', 'sensor_msgs/msg/Image', render_width=0.8)
node.subscribe_topic('/camera/points', 'sensor_msgs/msg/PointCloud2', point_size=0.003, max_points=200_000)
node.subscribe_topic('/viser/markers', 'visualization_msgs/msg/MarkerArray')

# Manual visualization is also available.
node.show_pose('target_pose', pose_stamped)
node.show_point_cloud('debug_cloud', xyz_rgb_array)
```

Conversions use NumPy as the intermediate representation before the node updates viser handles. Repeated calls with the same scene name update the existing object instead of creating duplicates.

## Examples

The examples package mirrors the `rviz2_utils` structure: every datatype module has a `create(...)` function, while `viser_ros2_utils/examples/__init__.py` owns the CLI.

```bash
ros2 run viser_ros2_utils examples --all
ros2 run viser_ros2_utils examples marker_array_topic --topic /viser/markers
```

Manual examples are split into `pose`, `point`, `vector3`, `transform`, `image`, `point_cloud`, `mesh`, `marker`, and `marker_array`. The `marker_array_topic` example only subscribes to the topic; publish the `visualization_msgs/msg/MarkerArray` content separately.

## Install

Create a virtual environment:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install python dependencies from `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

Expose the venv packages to ros2 before starting the bridge:

```bash
source install/setup.bash
export PYTHONPATH=.venv/lib/python3.12/site-packages:$PYTHONPATH
```

Run the Bridge:

```bash
ros2 run viser_ros2_utils viser_bridge
```