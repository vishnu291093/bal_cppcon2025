#include <visualize_dataset.h>

#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace bal_cppcon {

DatasetVisualizer::DatasetVisualizer(rclcpp::Node::SharedPtr node, std::string camera_topic, std::string point_topic)
{
    point_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(point_topic, 10);
    camera_pub_ = node->create_publisher<visualization_msgs::msg::Marker>(camera_topic, 10);
}
} // namespace bal_cppcon