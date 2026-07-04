#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
if [ -f /ws/cuda_icp_localizer/install/setup.bash ]; then
    source /ws/cuda_icp_localizer/install/setup.bash
fi

exec "$@"
