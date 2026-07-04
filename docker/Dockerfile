# syntax=docker/dockerfile:1
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# base: CUDA 12 + ROS 2 Humble + build/runtime deps (dev image, source is
# bind-mounted at /ws by docker/compose.yaml)
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    ROS_DISTRO=humble

RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl ca-certificates lsb-release \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 apt repository (official ros2-apt-source package)
RUN ROS_APT_SOURCE_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}') \
    && curl -fsSL -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(lsb_release -cs)_all.deb" \
    && apt-get install -y /tmp/ros2-apt-source.deb \
    && rm /tmp/ros2-apt-source.deb

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-dev-tools \
        python3-colcon-common-extensions \
        ros-humble-pcl-ros \
        ros-humble-pcl-conversions \
        ros-humble-tf2-geometry-msgs \
        libpcl-dev \
        libglm-dev \
        gcc-10 g++-10 \
        gdb git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws

COPY docker/entrypoint.sh /entrypoint.sh
# The container runs as root while host-side nodes run as a regular user;
# Fast DDS shared-memory segments are not accessible across users, so force
# UDP-only transport (see docker/fastdds_no_shm.xml).
COPY docker/fastdds_no_shm.xml /etc/fastdds/fastdds_no_shm.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/etc/fastdds/fastdds_no_shm.xml
RUN chmod +x /entrypoint.sh \
    && echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc \
    && echo '[ -f /ws/cuda_icp_localizer/install/setup.bash ] && source /ws/cuda_icp_localizer/install/setup.bash' >> /root/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# ---------------------------------------------------------------------------
# build: copies the source in and runs colcon build — used by CI to verify
# the package compiles, and usable as a standalone runnable image
# ---------------------------------------------------------------------------
FROM base AS build

COPY . /ws

RUN cd /ws/cuda_icp_localizer \
    && . /opt/ros/humble/setup.sh \
    && colcon build --symlink-install
