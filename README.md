# GPU-Accelerated LiDAR Localization (CUDA ICP + ROS 2)

Real-time, map-based LiDAR localization for autonomous vehicles. Incoming
LiDAR scans are aligned against a prebuilt point cloud map with a CUDA
implementation of the Iterative Closest Point (ICP) algorithm, giving a
6-DoF pose estimate in a few milliseconds per scan.

Tested end-to-end in a Gazebo simulation of an autonomous vehicle (Ouster
LiDAR, ~5,700 points/scan, 50,000-point map on the GPU):

- **5–30 ms per scan** while tracking (2–5 ICP iterations) on an RTX 3060
  Laptop GPU — comfortably real-time at 10 Hz scan rate.
- Converges from a **0.5 m + 5° off** initial guess to **~1 cm** accuracy.
- Compared against a CPU-based localizer on the same route, the GPU ICP
  localizer stayed noticeably more stable, especially through sharp turns.

<!-- TODO: add a demo GIF/screenshot here (RViz: white map, orange aligned scan, green pose arrow) -->

## Repository layout

| Path | Description |
|---|---|
| [`cuda_icp_localizer/`](cuda_icp_localizer/) | **ROS 2 (Humble) package** — the localization node, launch file, RViz config and parameters |
| [`src/`](src/) | CUDA ICP core: correspondence kernels, thrust reductions, SVD (Kabsch) solve |
| [`docker/`](docker/) | Docker image (CUDA 12 + ROS 2 Humble) and Compose setup, driven by the top-level `Makefile` |

## Requirements

- ROS 2 Humble
- CUDA Toolkit 11+ (tested with 12.x), NVIDIA GPU (`sm_86` by default — adjust
  `CUDA_NVCC_FLAGS` in `cuda_icp_localizer/CMakeLists.txt` for your GPU)
- PCL, GLM (`libglm-dev`), GCC 10 (`gcc-10 g++-10`)

## Map & sample data

The point cloud map (`gazebo_map.pcd`, ~1.5M points) and a sample LiDAR rosbag
recorded in the simulation are hosted on Google Drive:

- **[Download map + sample bag (Google Drive)](https://drive.google.com/drive/folders/1_-O-QYGeOHzxYxyVDFM6r3RSKeE7fwxj?usp=sharing)**

After downloading, set `target_pcd_file` in
`cuda_icp_localizer/config/icp_localizer_params.yaml` to the PCD path.

## Build & run

### Docker (recommended)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The container runs with the host network (ROS 2 nodes inside and outside the
container see each other directly), GPU access and X11 forwarding for RViz.

```bash
make build    # build the image (CUDA 12 + ROS 2 Humble)
make up       # start the container in the background
make colcon   # colcon build inside the container
make shell    # open a shell inside the container
make down     # stop the container
```

The repository is mounted at `/ws` inside the container; in-container build
artifacts live in named Docker volumes, so they never mix with host builds.
Note that paths in `icp_localizer_params.yaml` (e.g. `target_pcd_file`) must
use container paths — the map at the repository root is `/ws/gazebo_map.pcd`.
Inside the shell the workspace is already sourced — launch directly:

```bash
ros2 launch cuda_icp_localizer icp_localizer.launch.py
```

A [CI workflow](.github/workflows/docker.yml) builds the image and compiles
the package on every push.

### Native

```bash
cd cuda_icp_localizer
colcon build --symlink-install
source install/setup.bash

ros2 launch cuda_icp_localizer icp_localizer.launch.py

# in another terminal, play the sample bag (publishes /lidar_front/points_in):
ros2 bag play <path-to-downloaded-bag>
```

By default (`use_initial_pose: true`) localization starts automatically from
the pose configured in the YAML. Set it to `false` to have the node wait until
you provide a pose with RViz's **2D Pose Estimate** tool instead. In both
modes you can re-initialize at any time from RViz.

RViz opens preconfigured: the map in white, the ICP-aligned scan in orange and
the estimated pose as a green arrow.

Full topic/parameter reference: [`cuda_icp_localizer/README.md`](cuda_icp_localizer/README.md).

## How it works

Point-to-point ICP, fully parallelized on the GPU:

1. **Correspondences** — one CUDA thread per source point does a
   nearest-neighbour search over the map points.
2. **Alignment** — means and the cross-covariance matrix are computed with
   `thrust` parallel reductions; the rotation comes from a 3×3 SVD
   ([Kabsch / orthogonal Procrustes](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)).
3. **Iterate** — source points are transformed in place until the incremental
   motion drops below a convergence threshold.

The ROS 2 node feeds each scan to this core with the previous pose as the
initial guess (tracking), composes the accumulated ICP transform with it,
rejects implausible jumps, and publishes the pose, the aligned cloud and the
`map → lidar` TF. The map is voxel-downsampled once at startup; live scans are
consumed with a keep-last-1, best-effort QoS so the localizer never lags
behind the sensor even if an occasional scan takes longer to converge.

## Roadmap

- [x] Docker image (CUDA + ROS 2 Humble) with in-container `colcon build`
- [x] CI pipeline that builds the image on every push
- [ ] Demo GIF/screenshot in the README

## Credits

This project builds on
[**thegyro/Project4-Scan-Matching**](https://github.com/thegyro/Project4-Scan-Matching)
by [Srinath Rajagopalan](https://github.com/thegyro) — a CUDA scan matching
project from the University of Pennsylvania course
*CIS 565: GPU Programming and Architecture* (Project 4). The CUDA ICP core in
`src/` originates from that work; see its README for the theory write-up and
benchmarks on the Stanford bunny/dragon models.

Additions in this repository:

- ROS 2 Humble integration (`cuda_icp_localizer` package): live LiDAR input,
  map loading from PCD, pose/TF publishing, RViz setup
- Accumulated-transform API on the CUDA core (`getTransform`), convergence
  early-exit, per-frame GPU memory cleanup
- Hash-grid voxel downsampling for large maps, initial-pose handling
  (parameters + `/initialpose`), pose tracking with outlier rejection
- CUDA 12 / GCC 10 build fixes

The 3×3 SVD implementation is [`svd3`](https://github.com/ericjang/svd3) by
Eric Jang.
