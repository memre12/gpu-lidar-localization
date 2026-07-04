# cuda_icp_localizer

GPU-accelerated LiDAR localization for ROS 2. The node matches incoming LiDAR
scans against a prebuilt point cloud map (PCD) using a CUDA implementation of
the Iterative Closest Point (ICP) algorithm, and publishes the estimated pose
in real time.

See the [repository README](../README.md) for an overview, benchmarks and
credits.

## Node: `icp_localizer`

### Subscribed topics

| Topic | Type | Description |
|---|---|---|
| `/lidar_front/points_in` (param: `source_topic`) | `sensor_msgs/PointCloud2` | Live LiDAR scan (QoS: best effort, keep last 1 — stale scans are dropped) |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Initial pose estimate, e.g. from RViz *2D Pose Estimate* tool |

### Published topics

| Topic | Type | Description |
|---|---|---|
| `/icp_localizer/estimated_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Estimated pose in the map frame |
| `/icp_localizer/aligned_cloud` | `sensor_msgs/PointCloud2` | The scan transformed into the map frame by the ICP result |
| `/icp_localizer/target_cloud` | `sensor_msgs/PointCloud2` | Downsampled map (transient local QoS, published once) |
| `/tf` | `map` → `lidar_front` | Broadcast when `publish_tf` is true |

### Parameters (`config/icp_localizer_params.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `source_topic` | `/lidar_front/points_in` | Input scan topic |
| `target_pcd_file` | *(path)* | Map PCD file (see the Drive link in the repo README). Empty = use the first received scan as target |
| `max_iterations` | 60 | Upper bound on ICP iterations (stops early on convergence) |
| `convergence_threshold` | 0.005 | Stop when per-iteration motion (m + rad) drops below this |
| `max_points` | 150000 | Cap on source scan points sent to the GPU |
| `max_target_points` | 50000 | Cap on map points kept on the GPU |
| `downsample_size` | 0.1 | Voxel size (m) used to downsample the map evenly |
| `local_map_radius` | 60.0 | Radius (m) of the map crop kept on the GPU; refreshed as the vehicle moves |
| `max_scan_range` / `min_scan_range` | 45.0 / 1.5 | Scan points outside this range band are dropped |
| `correspondence_max_dist` | 5.0 | ICP pairs farther apart than this (m) are rejected as outliers |
| `use_initial_pose` | true | `true`: start localizing immediately from `initial_x/y/z/roll/pitch/yaw`. `false`: wait for RViz **2D Pose Estimate** (`/initialpose`) |
| `initial_x/y/z`, `initial_roll/pitch/yaw` | — | Auto-start pose used when `use_initial_pose` is true |
| `publish_tf` | true | Broadcast `map` → `source_frame` TF |
| `target_frame` / `source_frame` | `map` / `lidar_front` | Frame names |

## Usage

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch cuda_icp_localizer icp_localizer.launch.py          # opens RViz too
ros2 launch cuda_icp_localizer icp_localizer.launch.py rviz:=false
```

Set the vehicle's rough position at any time with RViz's **2D Pose Estimate**
tool; ICP snaps it onto the map and tracks from there (initial guess for each
scan is the previous estimate).

## How it works

Each scan is processed on the GPU (see `../src/scan_matching.cu`):

1. Brute-force nearest-neighbour correspondence search — one CUDA thread per
   source point.
2. Mean-centering and the cross-covariance matrix via `thrust` parallel
   reductions.
3. Rotation from SVD (Kabsch / orthogonal Procrustes), translation from the
   means.
4. Source points transformed in place; steps repeat until the incremental
   motion falls below `convergence_threshold`.

The accumulated transform is composed with the initial guess and published.
Wild jumps (>5 m in one frame) are rejected as divergence and the previous
pose is kept.
