#ifndef CUDA_ICP_LOCALIZER_HPP_
#define CUDA_ICP_LOCALIZER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/transform_broadcaster.h>

#include <glm/glm.hpp>

namespace cuda_icp_localizer
{

class ScanMatchingNode : public rclcpp::Node
{
public:
  ScanMatchingNode();
  ~ScanMatchingNode();

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void load_target_from_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
  bool load_target_from_pcd(const std::string& pcd_file);
  void process_source_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                           const std_msgs::msg::Header& header);
  // Returns how far ICP moved the pose from the initial guess (meters)
  float publish_results(const std_msgs::msg::Header& header, float* rotation, float* translation);
  Eigen::Matrix4f get_initial_transform();
  void update_local_map(const Eigen::Vector3f& center);

  // ROS2 interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr target_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // Parameters
  std::string source_topic_;
  std::string target_pcd_file_;
  int max_iterations_;
  double convergence_threshold_;
  int max_points_;
  int max_target_points_;
  double downsample_size_;
  double local_map_radius_;
  double max_scan_range_;
  double min_scan_range_;
  double correspondence_max_dist_;
  bool publish_tf_;
  std::string target_frame_;
  std::string source_frame_;
  bool use_initial_pose_;
  double initial_x_;
  double initial_y_;
  double initial_z_;
  double initial_roll_;
  double initial_pitch_;
  double initial_yaw_;

  // Internal state
  bool target_loaded_;
  bool first_cloud_received_;
  bool initial_pose_received_;
  bool cuda_initialized_;  // Track if CUDA is initialized
  geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr latest_initial_pose_;
  Eigen::Matrix4f current_initial_transform_;
  Eigen::Matrix4f last_estimated_pose_;
  bool use_last_pose_as_initial_;
  
  // Pose smoothing
  std::vector<Eigen::Matrix4f> pose_history_;
  int pose_history_size_ = 5;
  
  // Full downsampled map (host); a local crop of it lives on the GPU
  std::vector<glm::vec3> map_points_;
  Eigen::Vector3f crop_center_;
  bool have_crop_;
  int num_target_;

  std::vector<glm::vec3> source_points_;
  int num_source_;
  
  // Helper functions for smoothing
  Eigen::Matrix4f smoothPose(const Eigen::Matrix4f& new_pose);
  bool isValidPose(const Eigen::Matrix4f& pose, const Eigen::Matrix4f& reference);
};

} // namespace cuda_icp_localizer

#endif // CUDA_ICP_LOCALIZER_HPP_
