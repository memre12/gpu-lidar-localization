#include "cuda_icp_localizer/icp_localizer.hpp"

#include <memory>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

// Include scan matching header - it's in a C++ namespace, not extern "C"
#include "../src/scan_matching.h"

using namespace std::chrono_literals;

namespace cuda_icp_localizer
{

ScanMatchingNode::ScanMatchingNode() : Node("icp_localizer")
{
  // Declare parameters
  this->declare_parameter("source_topic", "/lidar/points");
  this->declare_parameter("target_pcd_file", "");
  this->declare_parameter("max_iterations", 30);  // Reduced from 50 for speed
  this->declare_parameter("convergence_threshold", 0.001);
  this->declare_parameter("downsample_size", 0.05);
  this->declare_parameter("max_points", 10000);  // Reduced from 50000 for 10Hz
  this->declare_parameter("max_target_points", 50000);
  this->declare_parameter("local_map_radius", 60.0);
  this->declare_parameter("max_scan_range", 45.0);
  this->declare_parameter("min_scan_range", 1.5);
  this->declare_parameter("correspondence_max_dist", 5.0);
  this->declare_parameter("publish_tf", true);
  this->declare_parameter("target_frame", "map");
  this->declare_parameter("source_frame", "base_link");
  this->declare_parameter("use_initial_pose", false);
  this->declare_parameter("initial_x", 0.0);
  this->declare_parameter("initial_y", 0.0);
  this->declare_parameter("initial_z", 0.0);
  this->declare_parameter("initial_roll", 0.0);
  this->declare_parameter("initial_pitch", 0.0);
  this->declare_parameter("initial_yaw", 0.0);

  // Get parameters
  source_topic_ = this->get_parameter("source_topic").as_string();
  target_pcd_file_ = this->get_parameter("target_pcd_file").as_string();
  max_iterations_ = this->get_parameter("max_iterations").as_int();
  convergence_threshold_ = this->get_parameter("convergence_threshold").as_double();
  max_points_ = this->get_parameter("max_points").as_int();
  max_target_points_ = this->get_parameter("max_target_points").as_int();
  local_map_radius_ = this->get_parameter("local_map_radius").as_double();
  max_scan_range_ = this->get_parameter("max_scan_range").as_double();
  min_scan_range_ = this->get_parameter("min_scan_range").as_double();
  correspondence_max_dist_ = this->get_parameter("correspondence_max_dist").as_double();

  // Scan points must stay inside the GPU-resident local map crop, otherwise
  // edge points get matched to crop-boundary points and drag the solution
  if (max_scan_range_ > local_map_radius_ - 10.0) {
    max_scan_range_ = local_map_radius_ - 10.0;
    RCLCPP_WARN(this->get_logger(),
                "max_scan_range clamped to %.1fm (local_map_radius - 10m)", max_scan_range_);
  }
  downsample_size_ = this->get_parameter("downsample_size").as_double();
  publish_tf_ = this->get_parameter("publish_tf").as_bool();
  target_frame_ = this->get_parameter("target_frame").as_string();
  source_frame_ = this->get_parameter("source_frame").as_string();
  use_initial_pose_ = this->get_parameter("use_initial_pose").as_bool();
  initial_x_ = this->get_parameter("initial_x").as_double();
  initial_y_ = this->get_parameter("initial_y").as_double();
  initial_z_ = this->get_parameter("initial_z").as_double();
  initial_roll_ = this->get_parameter("initial_roll").as_double();
  initial_pitch_ = this->get_parameter("initial_pitch").as_double();
  initial_yaw_ = this->get_parameter("initial_yaw").as_double();

  // Create subscriber for source point cloud.
  // keep_last(1) + best_effort: never queue stale scans — if ICP runs longer
  // than the scan period, old scans are dropped and the newest one is used.
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    source_topic_, rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&ScanMatchingNode::pointcloud_callback, this, std::placeholders::_1));

  // Create subscriber for initial pose from RViz
  initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/initialpose", 10,
    std::bind(&ScanMatchingNode::initial_pose_callback, this, std::placeholders::_1));

  // Create publisher for aligned point cloud
  aligned_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/icp_localizer/aligned_cloud", 10);

  // Create publisher for target point cloud (map) with transient local QoS
  rclcpp::QoS target_qos(10);
  target_qos.transient_local();
  target_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/icp_localizer/target_cloud", target_qos);

  // Create publisher for estimated pose
  pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
    "/icp_localizer/estimated_pose", 10);

  // TF broadcaster
  if (publish_tf_) {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  }

  target_loaded_ = false;
  first_cloud_received_ = false;
  initial_pose_received_ = false;
  latest_initial_pose_ = nullptr;
  current_initial_transform_ = Eigen::Matrix4f::Identity();
  last_estimated_pose_ = Eigen::Matrix4f::Identity();
  use_last_pose_as_initial_ = false;
  have_crop_ = false;
  crop_center_ = Eigen::Vector3f::Zero();
  num_target_ = 0;
  num_source_ = 0;

  RCLCPP_INFO(this->get_logger(), "===========================================");
  RCLCPP_INFO(this->get_logger(), "CUDA ICP Localizer Initialized (ROS2 Humble)");
  RCLCPP_INFO(this->get_logger(), "===========================================");
  RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", source_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Max iterations: %d", max_iterations_);
  RCLCPP_INFO(this->get_logger(), "Max points: %d", max_points_);
  
  if (use_initial_pose_) {
    RCLCPP_INFO(this->get_logger(), "Initial pose: x=%.2f, y=%.2f, z=%.2f, yaw=%.2f",
                initial_x_, initial_y_, initial_z_, initial_yaw_);
  }
  
  if (!target_pcd_file_.empty()) {
    RCLCPP_INFO(this->get_logger(), "Loading target from PCD: %s", target_pcd_file_.c_str());
    if (load_target_from_pcd(target_pcd_file_)) {
      RCLCPP_INFO(this->get_logger(), "Target PCD loaded successfully!");
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to load PCD, will use first cloud as target");
    }
  } else {
    RCLCPP_INFO(this->get_logger(), "Will use first received cloud as target");
  }
  RCLCPP_INFO(this->get_logger(), "===========================================");
}

ScanMatchingNode::~ScanMatchingNode()
{
  try {
    ScanMatching::endSimulation();
  } catch (...) {
    // Ignore cleanup errors
  }
  
  RCLCPP_INFO(this->get_logger(), "ICP localizer node shutdown");
}

void ScanMatchingNode::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  
  // Convert ROS PointCloud2 to PCL
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *cloud);

  if (cloud->empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Received empty point cloud");
    return;
  }

  // Use first cloud as target if no target PCD file provided
  if (!target_loaded_) {
    RCLCPP_INFO(this->get_logger(), "Using first cloud as target");
    load_target_from_cloud(cloud);
    first_cloud_received_ = true;
    return;  // Skip first cloud processing
  }

  // use_initial_pose=false: don't localize until a pose arrives on /initialpose
  if (!use_initial_pose_ && !initial_pose_received_) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Waiting for initial pose (use RViz '2D Pose Estimate' -> /initialpose)");
    return;
  }

  // Process source cloud
  process_source_cloud(cloud, msg->header);
}

bool ScanMatchingNode::load_target_from_pcd(const std::string& pcd_file)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file: %s", pcd_file.c_str());
    return false;
  }
  
  if (cloud->empty()) {
    RCLCPP_ERROR(this->get_logger(), "PCD file is empty: %s", pcd_file.c_str());
    return false;
  }
  
  RCLCPP_INFO(this->get_logger(), "Loaded PCD with %zu points", cloud->size());
  load_target_from_cloud(cloud);
  return true;
}

void ScanMatchingNode::load_target_from_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  // Voxel-grid downsample for even spatial coverage (one point per voxel,
  // hash-grid based: handles arbitrarily large maps without index overflow)
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  if (downsample_size_ > 0.0) {
    const float inv_leaf = 1.0f / static_cast<float>(downsample_size_);
    std::unordered_set<uint64_t> voxels;
    voxels.reserve(cloud->size());
    filtered->points.reserve(cloud->size() / 4);

    for (const auto& p : cloud->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }
      int64_t ix = static_cast<int64_t>(std::floor(p.x * inv_leaf));
      int64_t iy = static_cast<int64_t>(std::floor(p.y * inv_leaf));
      int64_t iz = static_cast<int64_t>(std::floor(p.z * inv_leaf));
      uint64_t key = (static_cast<uint64_t>(ix & 0x1FFFFF) << 42) |
                     (static_cast<uint64_t>(iy & 0x1FFFFF) << 21) |
                     (static_cast<uint64_t>(iz & 0x1FFFFF));
      if (voxels.insert(key).second) {
        filtered->points.push_back(p);
      }
    }
    filtered->width = filtered->points.size();
    filtered->height = 1;
    RCLCPP_INFO(this->get_logger(), "Voxel filter (%.2fm): %zu -> %zu points",
                downsample_size_, cloud->size(), filtered->size());
  } else {
    filtered = cloud;
  }

  // Keep the full downsampled map on the host; a local crop around the
  // current pose is uploaded to the GPU on demand (update_local_map)
  map_points_.clear();
  map_points_.reserve(filtered->size());
  for (const auto& p : filtered->points) {
    map_points_.emplace_back(p.x, p.y, p.z);
  }

  have_crop_ = false;
  target_loaded_ = true;
  RCLCPP_INFO(this->get_logger(), "Map loaded with %zu points (local crop radius %.1fm, max %d on GPU)",
              map_points_.size(), local_map_radius_, max_target_points_);

  // Publish (a capped version of) the map for visualization
  size_t display_points = std::min(map_points_.size(), (size_t)200000);
  size_t step = map_points_.size() / display_points;
  if (step == 0) step = 1;

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_pcl(new pcl::PointCloud<pcl::PointXYZ>);
  target_pcl->points.reserve(display_points);
  for (size_t i = 0; i < map_points_.size(); i += step) {
    target_pcl->points.emplace_back(map_points_[i].x, map_points_[i].y, map_points_[i].z);
  }
  target_pcl->width = target_pcl->points.size();
  target_pcl->height = 1;

  sensor_msgs::msg::PointCloud2 target_msg;
  pcl::toROSMsg(*target_pcl, target_msg);
  target_msg.header.frame_id = target_frame_;
  target_msg.header.stamp = this->now();
  target_cloud_pub_->publish(target_msg);
  RCLCPP_INFO(this->get_logger(), "Published target cloud to /icp_localizer/target_cloud");
}

void ScanMatchingNode::update_local_map(const Eigen::Vector3f& center)
{
  // Crop the map to a radius around the pose (2D: vehicle moves in the plane)
  std::vector<glm::vec3> crop;
  crop.reserve(max_target_points_);
  const float r2 = static_cast<float>(local_map_radius_ * local_map_radius_);

  std::vector<glm::vec3> candidates;
  candidates.reserve(map_points_.size() / 8);
  for (const auto& p : map_points_) {
    float dx = p.x - center.x();
    float dy = p.y - center.y();
    if (dx * dx + dy * dy < r2) {
      candidates.push_back(p);
    }
  }

  if (candidates.size() < 100) {
    // Pose is off-map: fall back to a stride-sampled full map
    RCLCPP_WARN(this->get_logger(),
                "Local map crop has only %zu points — pose may be off-map, using full map",
                candidates.size());
    candidates = map_points_;
  }

  size_t step = candidates.size() / (size_t)max_target_points_ + 1;
  for (size_t i = 0; i < candidates.size(); i += step) {
    crop.push_back(candidates[i]);
  }

  ScanMatching::setTarget(crop.size(), crop.data());
  num_target_ = crop.size();
  crop_center_ = center;
  have_crop_ = true;

  RCLCPP_INFO(this->get_logger(), "Local map updated: %d points within %.0fm of [%.1f, %.1f]",
              num_target_, local_map_radius_, center.x(), center.y());
}

void ScanMatchingNode::process_source_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                       const std_msgs::msg::Header& header)
{
  // Get initial transformation
  current_initial_transform_ = get_initial_transform();

  // Range-filter the scan in the lidar frame: drop ego-vehicle hits and
  // points beyond the local map crop (their true match isn't on the GPU)
  pcl::PointCloud<pcl::PointXYZ>::Ptr ranged(new pcl::PointCloud<pcl::PointXYZ>);
  ranged->points.reserve(cloud->size());
  const float min_r2 = static_cast<float>(min_scan_range_ * min_scan_range_);
  const float max_r2 = static_cast<float>(max_scan_range_ * max_scan_range_);
  for (const auto& p : cloud->points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    if (r2 > min_r2 && r2 < max_r2) {
      ranged->points.push_back(p);
    }
  }
  if (ranged->points.empty()) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "All scan points filtered out by range limits");
    return;
  }
  ranged->width = ranged->points.size();
  ranged->height = 1;
  ranged->is_dense = true;

  // Apply initial transform to source cloud BEFORE ICP
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*ranged, *transformed_cloud, current_initial_transform_);
  
  // Use more points for better GPU utilization
  size_t num_points = std::min(transformed_cloud->size(), (size_t)max_points_);

  source_points_.clear();
  source_points_.reserve(num_points);
  num_source_ = num_points;

  size_t step = transformed_cloud->size() / num_points;
  if (step == 0) step = 1;

  // Convert transformed cloud to glm format
  for (size_t i = 0; i < num_points; ++i) {
    size_t idx = i * step;
    if (idx >= transformed_cloud->size()) idx = transformed_cloud->size() - 1;

    source_points_.emplace_back(
      transformed_cloud->points[idx].x,
      transformed_cloud->points[idx].y,
      transformed_cloud->points[idx].z
    );
  }

  // Refresh the GPU-side local map when the vehicle strays from the crop center
  Eigen::Vector3f pos = current_initial_transform_.block<3,1>(0,3);
  if (!have_crop_ || (pos - crop_center_).head<2>().norm() > 0.4f * local_map_radius_) {
    update_local_map(pos);
  }

  auto callback_start = std::chrono::high_resolution_clock::now();

  float rotation[9];
  float translation[3];
  int iterations_run = 0;

  try {
    // Upload scan; the target (local map) is already resident on the GPU
    ScanMatching::setSource(num_source_, source_points_.data());

    // Run ICP iterations until convergence
    const float corr_gate = static_cast<float>(correspondence_max_dist_);
    for (int i = 0; i < max_iterations_; ++i) {
      float delta = ScanMatching::transformGPU(0.1f, corr_gate);
      iterations_run = i + 1;

      if (delta < convergence_threshold_) {
        break;
      }
    }

    // Retrieve accumulated ICP transform (in map frame, on top of initial guess)
    ScanMatching::getTransform(rotation, translation);

  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Scan matching error: %s", e.what());
    return;
  }

  // Publish results; returns how far ICP moved the pose from the initial guess
  float correction = publish_results(header, rotation, translation);

  auto callback_end = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(callback_end - callback_start).count();

  Eigen::Vector3f p = last_estimated_pose_.block<3,1>(0,3);
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                       "pose [%.2f, %.2f, %.2f] | accuracy %.3f m | %ld ms (%d iters) | points %d src / %d map",
                       p.x(), p.y(), p.z(), correction, total_ms, iterations_run,
                       num_source_, num_target_);
}

float ScanMatchingNode::publish_results(const std_msgs::msg::Header& header, float* rotation, float* translation)
{
  // ICP gives us refinement on top of initial guess
  Eigen::Matrix3f rot_matrix;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rot_matrix(i, j) = rotation[i * 3 + j];
    }
  }
  
  Eigen::Matrix4f icp_refinement = Eigen::Matrix4f::Identity();
  icp_refinement.block<3,3>(0,0) = rot_matrix;
  icp_refinement(0,3) = translation[0];
  icp_refinement(1,3) = translation[1];
  icp_refinement(2,3) = translation[2];
  
  // Final pose is: ICP_refinement * Initial_guess
  Eigen::Matrix4f final_tf = icp_refinement * current_initial_transform_;

  // Reject wild ICP jumps (likely divergence) and fall back to the initial guess
  if (!isValidPose(final_tf, current_initial_transform_)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "ICP result jumped too far, rejecting correction this frame");
    final_tf = current_initial_transform_;
    icp_refinement = Eigen::Matrix4f::Identity();
  }

  // Save this as the last estimated pose for next iteration (for tracking)
  last_estimated_pose_ = final_tf;

  // Extract position and orientation from final transform
  Eigen::Vector3f position = final_tf.block<3,1>(0,3);
  Eigen::Matrix3f final_rot = final_tf.block<3,3>(0,0);
  Eigen::Quaternionf quat(final_rot);

  // How far ICP actually moved the pose relative to the initial guess
  float correction = (position - current_initial_transform_.block<3,1>(0,3)).norm();

  // Create aligned cloud
  // source_points_ are already in the coordinate frame after applying initial_transform
  // ICP further refined them, so we just need to apply ICP refinement
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  aligned_cloud->width = num_source_;
  aligned_cloud->height = 1;
  aligned_cloud->is_dense = false;
  aligned_cloud->points.resize(num_source_);
  
  for (int i = 0; i < num_source_; ++i) {
    // source_points_[i] is already transformed by initial_tf
    // Apply ICP refinement to get final aligned position
    Eigen::Vector4f pt(source_points_[i].x, source_points_[i].y, source_points_[i].z, 1.0f);
    Eigen::Vector4f transformed = icp_refinement * pt;
    aligned_cloud->points[i].x = transformed.x();
    aligned_cloud->points[i].y = transformed.y();
    aligned_cloud->points[i].z = transformed.z();
  }
  
  // Publish aligned cloud
  sensor_msgs::msg::PointCloud2 output_msg;
  pcl::toROSMsg(*aligned_cloud, output_msg);
  output_msg.header = header;
  output_msg.header.frame_id = target_frame_;
  aligned_cloud_pub_->publish(output_msg);

  // Publish pose
  geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
  pose_msg.header = header;
  pose_msg.header.frame_id = target_frame_;
  pose_msg.pose.pose.position.x = position.x();
  pose_msg.pose.pose.position.y = position.y();
  pose_msg.pose.pose.position.z = position.z();
  pose_msg.pose.pose.orientation.w = quat.w();
  pose_msg.pose.pose.orientation.x = quat.x();
  pose_msg.pose.pose.orientation.y = quat.y();
  pose_msg.pose.pose.orientation.z = quat.z();
  // Diagonal covariance: [x, y, z, roll, pitch, yaw]
  pose_msg.pose.covariance[0] = 0.01;   // x
  pose_msg.pose.covariance[7] = 0.01;   // y
  pose_msg.pose.covariance[14] = 0.01;  // z
  pose_msg.pose.covariance[21] = 0.0025; // roll
  pose_msg.pose.covariance[28] = 0.0025; // pitch
  pose_msg.pose.covariance[35] = 0.0025; // yaw
  pose_pub_->publish(pose_msg);

  // Publish TF
  if (publish_tf_ && tf_broadcaster_) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header = header;
    tf_msg.header.frame_id = target_frame_;
    tf_msg.child_frame_id = source_frame_;
    tf_msg.transform.translation.x = position.x();
    tf_msg.transform.translation.y = position.y();
    tf_msg.transform.translation.z = position.z();
    tf_msg.transform.rotation.w = quat.w();
    tf_msg.transform.rotation.x = quat.x();
    tf_msg.transform.rotation.y = quat.y();
    tf_msg.transform.rotation.z = quat.z();
    tf_broadcaster_->sendTransform(tf_msg);
  }

  return correction;
}

void ScanMatchingNode::initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  RCLCPP_WARN(this->get_logger(), "=== RECEIVED INITIAL POSE FROM RVIZ ===");
  latest_initial_pose_ = msg;
  initial_pose_received_ = true;
  use_last_pose_as_initial_ = false;  // Reset to use the new initial pose
  
  RCLCPP_WARN(this->get_logger(), "Initial pose: x=%.2f, y=%.2f, z=%.2f",
              msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
}

Eigen::Matrix4f ScanMatchingNode::get_initial_transform()
{
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  
  RCLCPP_DEBUG(this->get_logger(), "get_initial_transform: initial_pose_received_=%d, use_last_pose_as_initial_=%d", 
               initial_pose_received_, use_last_pose_as_initial_);
  
  // Use RViz initial pose if available (takes priority when newly set)
  if (initial_pose_received_ && latest_initial_pose_ && !use_last_pose_as_initial_) {
    auto& pos = latest_initial_pose_->pose.pose.position;
    auto& ori = latest_initial_pose_->pose.pose.orientation;
    
    // Convert quaternion to rotation matrix
    Eigen::Quaternionf q(ori.w, ori.x, ori.y, ori.z);
    Eigen::Matrix3f rot = q.toRotationMatrix();
    
    transform.block<3,3>(0,0) = rot;
    transform(0,3) = pos.x;
    transform(1,3) = pos.y;
    transform(2,3) = pos.z;
    
    RCLCPP_WARN(this->get_logger(), ">>> USING RVIZ INITIAL POSE: x=%.2f, y=%.2f, z=%.2f", 
                pos.x, pos.y, pos.z);
    
    // After first use, enable tracking from last pose
    use_last_pose_as_initial_ = true;
  }
  // Use last estimated pose if available (for tracking)
  else if (use_last_pose_as_initial_) {
    transform = last_estimated_pose_;
    RCLCPP_DEBUG(this->get_logger(), "Using last estimated pose as initial (tracking mode)");
  }
  // Use parameter-based initial pose
  else if (use_initial_pose_) {
    // Create rotation from RPY
    Eigen::AngleAxisf rollAngle(initial_roll_, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(initial_pitch_, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(initial_yaw_, Eigen::Vector3f::UnitZ());
    
    Eigen::Matrix3f rot = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();
    
    transform.block<3,3>(0,0) = rot;
    transform(0,3) = initial_x_;
    transform(1,3) = initial_y_;
    transform(2,3) = initial_z_;
    
    RCLCPP_INFO(this->get_logger(), "Using parameter initial pose");
    
    // Enable tracking from last pose after first use
    use_last_pose_as_initial_ = true;
  }
  else {
    RCLCPP_WARN(this->get_logger(), "No initial pose set - using identity transform");
  }
  
  return transform;
}

bool ScanMatchingNode::isValidPose(const Eigen::Matrix4f& pose, const Eigen::Matrix4f& reference)
{
  // Check if pose is too far from reference (outlier rejection)
  Eigen::Vector3f pos = pose.block<3,1>(0,3);
  Eigen::Vector3f ref_pos = reference.block<3,1>(0,3);
  
  float distance = (pos - ref_pos).norm();
  
  // Reject if moved more than 5 meters in one frame (likely ICP failure)
  if (distance > 5.0f) {
    return false;
  }
  
  return true;
}

Eigen::Matrix4f ScanMatchingNode::smoothPose(const Eigen::Matrix4f& new_pose)
{
  // Add to history
  pose_history_.push_back(new_pose);
  
  // Keep only recent poses
  if (pose_history_.size() > pose_history_size_) {
    pose_history_.erase(pose_history_.begin());
  }
  
  // If we don't have enough history, return as-is
  if (pose_history_.size() < 2) {
    return new_pose;
  }
  
  // Weighted average: recent poses have more weight
  Eigen::Matrix4f smoothed = Eigen::Matrix4f::Zero();
  float total_weight = 0.0f;
  
  for (size_t i = 0; i < pose_history_.size(); ++i) {
    // Exponential weight: more recent = higher weight
    float weight = std::exp(static_cast<float>(i) / pose_history_.size());
    smoothed += weight * pose_history_[i];
    total_weight += weight;
  }
  
  smoothed /= total_weight;
  
  // Re-orthogonalize rotation matrix
  Eigen::Matrix3f rot = smoothed.block<3,3>(0,0);
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(rot, Eigen::ComputeFullU | Eigen::ComputeFullV);
  rot = svd.matrixU() * svd.matrixV().transpose();
  
  smoothed.block<3,3>(0,0) = rot;
  
  return smoothed;
}

} // namespace cuda_icp_localizer

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<cuda_icp_localizer::ScanMatchingNode>();
  
  RCLCPP_INFO(node->get_logger(), "Spinning ICP localizer node...");
  
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  return 0;
}
