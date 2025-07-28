#ifndef BAL_CPPCON_VISUALIZE_DATASET_H
#define BAL_CPPCON_VISUALIZE_DATASET_H

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <vector>
#include <array>
#include <concepts>
#include <type_traits>
#include <numeric>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <parser/data_model.h>

// Define Vec3 type
using Vec3 = Eigen::Vector3d;


namespace bal_cppcon {

class DatasetVisualizer
{
public:
    explicit DatasetVisualizer(rclcpp::Node::SharedPtr node, std::string camera_topic, std::string point_topic);

    // Templated method to handle any 3D point container type
    template<Point3DContainer Container>
    void publishPoints(const Container& points, double r_val, double g_val, double b_val);
    
    // Templated method for camera parameters to handle different container types
    template<typename CameraContainer>
    void publishCameras(const CameraContainer& camera_params, double r_val, double g_val, double b_val);

private:
    // Helper to extract coordinates from different point types
    template<typename PointType>
    geometry_msgs::msg::Point extractCoordinates(const PointType& point) const;
    
    // Debug helper to print coordinate distributions
    template<typename CameraContainer>
    void printCoordinateDistributions(const CameraContainer& camera_params) const;
    
    template<Point3DContainer Container>
    void printLandmarkDistributions(const Container& points) const;
    
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr point_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr camera_pub_;
};

// Template method implementations
template<typename PointType>
geometry_msgs::msg::Point DatasetVisualizer::extractCoordinates(const PointType& point) const
{
    geometry_msgs::msg::Point p;
    
    if constexpr (EigenPoint3D<PointType>) {
        // Handle Eigen-like points (Vec3, etc.)
        p.x = point.x();
        p.y = point.y();
        p.z = point.z();
    } else if constexpr (ArrayLikePoint3D<PointType>) {
        // Handle array-like points (std::array, std::vector)
        if constexpr (std::is_same_v<PointType, std::vector<double>>) {
            // Additional size check for std::vector
            if (point.size() >= 3) {
                p.x = point[0];
                p.y = point[1];
                p.z = point[2];
            }
        } else {
            // For std::array<double, 3>
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];
        }
    }
    
    return p;
}

template<Point3DContainer Container>
void DatasetVisualizer::publishPoints(const Container& points, double r_val, double g_val, double b_val)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = rclcpp::Clock().now();
    marker.ns = "bal_dataset_points";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.scale.x = 0.01;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;
    marker.color.r = r_val;
    marker.color.g = g_val;
    marker.color.b = b_val;
    marker.color.a = 1.0f;


    // Transform points to geometry_msgs::msg::Point using the template helper
    for (const auto& point : points) {
        // For std::vector<double>, check size first
        if constexpr (std::is_same_v<typename Container::value_type, std::vector<double>>) {
            if (point.size() >= 3) {
                marker.points.push_back(extractCoordinates(point));
            }
        } else {
            marker.points.push_back(extractCoordinates(point));
        }
    }

    point_pub_->publish(marker);
}

template<typename CameraContainer>
void DatasetVisualizer::publishCameras(const CameraContainer& camera_params, double r_val, double g_val, double b_val)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = rclcpp::Clock().now();
    marker.ns = "bal_dataset_cameras";
    marker.id = 1;
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.scale.x = 0.1;  // Point size for cameras
    marker.scale.y = 0.1;
    marker.color.r = r_val;
    marker.color.g = g_val;
    marker.color.b = b_val;
    marker.color.a = 1.0f;

    for (const auto& cam : camera_params)
    {
        // Extract parameters based on container type
        double qw, qx, qy, qz, tx, ty, tz;
        if constexpr (std::is_same_v<typename CameraContainer::value_type, std::vector<double>>) {
            if (cam.size() < 7) {
                RCLCPP_WARN(rclcpp::get_logger("DatasetVisualizer"), "Camera parameters do not have at least 7 values.");
                continue;
            }
            qw = cam[0]; qx = cam[1]; qy = cam[2]; qz = cam[3];
            tx = cam[4]; ty = cam[5]; tz = cam[6];
        } else {
            // For std::array or other array-like types
            qw = cam[0]; qx = cam[1]; qy = cam[2]; qz = cam[3];
            tx = cam[4]; ty = cam[5]; tz = cam[6];
        }

        // Extract rotation and translation
        Eigen::Quaterniond quat(qw, qx, qy, qz);
        Eigen::Vector3d t(tx, ty, tz);

        // Extract camera center position
        Eigen::Matrix3d rot = quat.normalized().toRotationMatrix();
        Eigen::Vector3d camera_center = -rot.transpose() * t;  // Extract camera center from world-to-camera params

        // Add camera position as a point
        geometry_msgs::msg::Point p;
        p.x = camera_center.x();
        p.y = camera_center.y();
        p.z = camera_center.z();
        marker.points.push_back(p);
    }

    camera_pub_->publish(marker);
}

template<typename CameraContainer>
void DatasetVisualizer::printCoordinateDistributions(const CameraContainer& camera_params) const
{
    if (camera_params.empty()) {
        RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), "No camera data to analyze");
        return;
    }

    // Extract camera positions (translation components)
    std::vector<double> cam_x, cam_y, cam_z;
    
    for (const auto& cam : camera_params) {
        double tx, ty, tz;
        if constexpr (std::is_same_v<typename CameraContainer::value_type, std::vector<double>>) {
            if (cam.size() >= 7) {
                tx = cam[4]; ty = cam[5]; tz = cam[6];
            } else {
                continue;
            }
        } else {
            // For std::array or other array-like types
            tx = cam[4]; ty = cam[5]; tz = cam[6];
        }
        cam_x.push_back(tx);
        cam_y.push_back(ty);
        cam_z.push_back(tz);
    }

    // Compute statistics
    auto compute_stats = [](const std::vector<double>& vals) {
        if (vals.empty()) return std::make_tuple(0.0, 0.0, 0.0);
        auto [min_it, max_it] = std::minmax_element(vals.begin(), vals.end());
        double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        return std::make_tuple(*min_it, *max_it, mean);
    };

    auto [cam_x_min, cam_x_max, cam_x_mean] = compute_stats(cam_x);
    auto [cam_y_min, cam_y_max, cam_y_mean] = compute_stats(cam_y);
    auto [cam_z_min, cam_z_max, cam_z_mean] = compute_stats(cam_z);

    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "=== COORDINATE DISTRIBUTION ANALYSIS ===");
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "CAMERAS (%zu total):", camera_params.size());
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "  X: min=%.3f, max=%.3f, mean=%.3f, range=%.3f", 
                cam_x_min, cam_x_max, cam_x_mean, cam_x_max - cam_x_min);
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "  Y: min=%.3f, max=%.3f, mean=%.3f, range=%.3f", 
                cam_y_min, cam_y_max, cam_y_mean, cam_y_max - cam_y_min);
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "  Z: min=%.3f, max=%.3f, mean=%.3f, range=%.3f", 
                cam_z_min, cam_z_max, cam_z_mean, cam_z_max - cam_z_min);
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "========================================");
}

template<Point3DContainer Container>
void DatasetVisualizer::printLandmarkDistributions(const Container& points) const
{
    if (points.empty()) {
        RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), "No landmark data to analyze");
        return;
    }

    // Extract landmark positions
    std::vector<double> pt_x, pt_y, pt_z;
    
    for (const auto& point : points) {
        if constexpr (EigenPoint3D<typename Container::value_type>) {
            // Handle Eigen-like points (Vec3, etc.)
            pt_x.push_back(point.x());
            pt_y.push_back(point.y());
            pt_z.push_back(point.z());
        } else if constexpr (ArrayLikePoint3D<typename Container::value_type>) {
            // Handle array-like points (std::array, std::vector)
            if constexpr (std::is_same_v<typename Container::value_type, std::vector<double>>) {
                if (point.size() >= 3) {
                    pt_x.push_back(point[0]);
                    pt_y.push_back(point[1]);
                    pt_z.push_back(point[2]);
                }
            } else {
                // For std::array<double, 3>
                pt_x.push_back(point[0]);
                pt_y.push_back(point[1]);
                pt_z.push_back(point[2]);
            }
        }
    }

    // Compute statistics
    auto compute_stats = [](const std::vector<double>& vals) {
        if (vals.empty()) return std::make_tuple(0.0, 0.0, 0.0);
        auto [min_it, max_it] = std::minmax_element(vals.begin(), vals.end());
        double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        return std::make_tuple(*min_it, *max_it, mean);
    };

    auto [pt_x_min, pt_x_max, pt_x_mean] = compute_stats(pt_x);
    auto [pt_y_min, pt_y_max, pt_y_mean] = compute_stats(pt_y);
    auto [pt_z_min, pt_z_max, pt_z_mean] = compute_stats(pt_z);

    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "LANDMARKS (%zu total):", points.size());
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "  X: min=%.3f, max=%.3f, mean=%.3f, range=%.3f", 
                pt_x_min, pt_x_max, pt_x_mean, pt_x_max - pt_x_min);
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "  Y: min=%.3f, max=%.3f, mean=%.3f, range=%.3f", 
                pt_y_min, pt_y_max, pt_y_mean, pt_y_max - pt_y_min);
    RCLCPP_INFO(rclcpp::get_logger("DatasetVisualizer"), 
                "  Z: min=%.3f, max=%.3f, mean=%.3f, range=%.3f", 
                pt_z_min, pt_z_max, pt_z_mean, pt_z_max - pt_z_min);
}

} // namespace bal_cppcon

#endif // BAL_CPPCON_VISUALIZE_DATASET_H

