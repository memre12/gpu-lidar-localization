cmake_minimum_required(VERSION 3.8)
project(cuda_icp_localizer)

# Force GCC-10 for CUDA compatibility BEFORE any other checks
set(CMAKE_C_COMPILER "/usr/bin/gcc-10")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native -mtune=native)
endif()

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Find PCL
find_package(PCL 1.10 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 10 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Force CUDA to use GCC-10
set(CUDA_HOST_COMPILER "/usr/bin/gcc-10")

# Set CUDA flags for GCC-10 compatibility and MAX PERFORMANCE
set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -std=c++14;
  --expt-relaxed-constexpr;
  --expt-extended-lambda;
  -Xcompiler=-fpermissive;
  -ccbin=/usr/bin/gcc-10;
  -O3;
  -use_fast_math;
  --use_fast_math;
  -arch=sm_86;
  -gencode=arch=compute_86,code=sm_86;
  --ptxas-options=-v;
  --maxrregcount=128;
  -lineinfo;
  --default-stream per-thread;
)

# Find GLM
find_package(glm QUIET)
if(NOT glm_FOUND)
  # Try to find GLM headers directly
  find_path(GLM_INCLUDE_DIR glm/glm.hpp
    HINTS
      /usr/include
      /usr/local/include
  )
  if(GLM_INCLUDE_DIR)
    message(STATUS "Found GLM: ${GLM_INCLUDE_DIR}")
    include_directories(${GLM_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "GLM not found. Please install libglm-dev")
  endif()
else()
  include_directories(${GLM_INCLUDE_DIRS})
endif()

# Include directories
include_directories(
  include
  ../src
)

# Scan matching CUDA library sources
set(SCAN_MATCHING_SOURCES
  ../src/scan_matching.cu
  ../src/kdtree.cpp
  ../src/utilityCore.cpp
)

# Create CUDA library for scan matching
cuda_add_library(scan_matching_cuda SHARED ${SCAN_MATCHING_SOURCES})

# ROS2 node executable
add_executable(icp_localizer
  src/icp_localizer.cpp
)

# Specify dependencies for the node
ament_target_dependencies(icp_localizer
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Link libraries
target_link_libraries(icp_localizer
  scan_matching_cuda
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
)

# Install executables
install(TARGETS
  icp_localizer
  DESTINATION lib/${PROJECT_NAME}
)

# Install CUDA library
install(TARGETS
  scan_matching_cuda
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  maps
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install header files
install(DIRECTORY
  include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
