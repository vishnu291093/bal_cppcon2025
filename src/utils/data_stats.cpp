#include <utils/data_stats.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <iostream>  // For std::cout
#include <sophus/se3.hpp>

namespace bal_cppcon {

// Constructor
DataStats::DataStats(std::shared_ptr<rclcpp::Node> node, 
                    double outlier_threshold, 
                    bool verbose_output) noexcept
    : node_(node), outlier_threshold_(outlier_threshold), verbose_output_(verbose_output) {
    

}

// Main analysis method
DatasetQualityMetrics DataStats::analyzeDataset(const BALData& data, 
                                               std::string_view label) const {
    DatasetQualityMetrics metrics;
    
    std::string analysis_label = label.empty() ? "Dataset Analysis" : std::string(label);
    logMessage("INFO", "Starting " + analysis_label);
    
    // Compute reprojection errors
    auto errors = computeReprojectionErrors(data);
    metrics.overall_reprojection_errors = computeStatistics(errors);
    
    // Compute observations per camera
    std::vector<double> obs_per_camera;
    for (size_t cam_idx = 0; cam_idx < data.num_cameras(); ++cam_idx) {
        size_t count = std::count_if(data.observations.begin(), data.observations.end(),
            [cam_idx](const auto& obs) { return obs.camera_index == static_cast<int>(cam_idx); });
        obs_per_camera.push_back(static_cast<double>(count));
        metrics.observations_per_camera_histogram[static_cast<int>(count)]++;
    }
    metrics.observations_per_camera = computeStatistics(obs_per_camera);
    
    // Compute observations per point
    std::vector<double> obs_per_point;
    for (size_t point_idx = 0; point_idx < data.num_points(); ++point_idx) {
        size_t count = std::count_if(data.observations.begin(), data.observations.end(),
            [point_idx](const auto& obs) { return obs.point_index == static_cast<int>(point_idx); });
        obs_per_point.push_back(static_cast<double>(count));
        metrics.observations_per_point_histogram[static_cast<int>(count)]++;
    }
    metrics.observations_per_point = computeStatistics(obs_per_point);
    
   
    // Compute coverage metrics
    metrics.point_visibility_ratio = data.num_observations() > 0 ? 
        static_cast<double>(data.num_points()) / data.num_observations() : 0.0;
    metrics.camera_visibility_ratio = data.num_observations() > 0 ?
        static_cast<double>(data.num_cameras()) / data.num_observations() : 0.0;
    
    if (verbose_output_) {
        printAnalysisReport(metrics, label);
    }
    
    return metrics;
}

// Compute reprojection errors
std::vector<double> DataStats::computeReprojectionErrors(const BALData& data) const {
    std::vector<double> errors;
    errors.reserve(data.observations.size());
    
    for (const auto& obs : data.observations) {
        if (obs.is_valid() && 
            obs.camera_index < static_cast<int>(data.camera_params.size()) &&
            obs.point_index < static_cast<int>(data.points.size())) {
            
            const auto& camera_params = data.camera_params[obs.camera_index];
            const auto& point_3d = data.points[obs.point_index];
            
            // Project 3D point to 2D
            auto projected = projectPoint(
                std::span<const double, CAMERA_PARAM_SIZE>(camera_params),
                std::span<const double, POINT_3D_SIZE>(point_3d)
            );
            
            // Compute reprojection error
            double dx = projected.x() - obs.x;
            double dy = projected.y() - obs.y;
            double error = std::sqrt(dx * dx + dy * dy);
            
            errors.push_back(error);
        }
    }
    
    return errors;
}

// Analyze camera-specific errors
std::vector<CameraErrorAnalysis> DataStats::analyzeCameraErrors(const BALData& data) const {
    std::vector<CameraErrorAnalysis> analysis_vec;
    analysis_vec.reserve(data.num_cameras());
    
    auto all_errors = computeReprojectionErrors(data);
    
    for (size_t cam_idx = 0; cam_idx < data.num_cameras(); ++cam_idx) {
        CameraErrorAnalysis analysis;
        analysis.camera_id = static_cast<int>(cam_idx);
        
        // Extract errors for this camera
        std::vector<double> camera_errors;
        size_t error_idx = 0;
        
        for (const auto& obs : data.observations) {
            if (obs.is_valid()) {  // Only process valid observations
                if (obs.camera_index == static_cast<int>(cam_idx) && 
                    error_idx < all_errors.size()) {
                    camera_errors.push_back(all_errors[error_idx]);
                    analysis.individual_errors.push_back(all_errors[error_idx]);
                }
                ++error_idx;  // Only increment for valid observations
            }
        }
        
        analysis.num_observations = camera_errors.size();
        if (!camera_errors.empty()) {
            analysis.error_stats = computeStatistics(camera_errors);
        }
        
        analysis_vec.push_back(std::move(analysis));
    }
    
    return analysis_vec;
}

// Analyze point-specific errors
std::vector<PointErrorAnalysis> DataStats::analyzePointErrors(const BALData& data) const {
    std::vector<PointErrorAnalysis> analysis_vec;
    analysis_vec.reserve(data.num_points());
    
    auto all_errors = computeReprojectionErrors(data);
    
    for (size_t point_idx = 0; point_idx < data.num_points(); ++point_idx) {
        PointErrorAnalysis analysis;
        analysis.point_id = static_cast<int>(point_idx);
        
        // Extract errors for this point
        std::vector<double> point_errors;
        std::vector<Eigen::Vector3d> camera_positions;
        size_t error_idx = 0;
        
        for (const auto& obs : data.observations) {
            if (obs.is_valid()) {  // Only process valid observations
                if (obs.point_index == static_cast<int>(point_idx) && 
                    error_idx < all_errors.size()) {
                    point_errors.push_back(all_errors[error_idx]);
                    analysis.individual_errors.push_back(all_errors[error_idx]);
                    
                    // Extract camera position for triangulation angle calculation
                    if (obs.camera_index >= 0 && obs.camera_index < static_cast<int>(data.camera_params.size())) {
                        const auto& cam_params = data.camera_params[obs.camera_index];
                        // Camera translation is stored in parameters 4-6 (after quaternion)
                        camera_positions.emplace_back(cam_params[4], cam_params[5], cam_params[6]);
                    }
                }
                ++error_idx;  // Only increment for valid observations
            }
        }
        
        analysis.num_observations = point_errors.size();
        if (!point_errors.empty()) {
            analysis.error_stats = computeStatistics(point_errors);
        }
        
        // Compute triangulation angle (simplified - average pairwise angle)
        if (camera_positions.size() >= 2) {
            const auto& point_pos = Eigen::Vector3d(
                data.points[point_idx][0], 
                data.points[point_idx][1], 
                data.points[point_idx][2]
            );
            
            double total_angle = 0.0;
            size_t angle_count = 0;
            
            for (size_t i = 0; i < camera_positions.size(); ++i) {
                for (size_t j = i + 1; j < camera_positions.size(); ++j) {
                    auto ray1 = (point_pos - camera_positions[i]).normalized();
                    auto ray2 = (point_pos - camera_positions[j]).normalized();
                    double dot_product = std::clamp(ray1.dot(ray2), -1.0, 1.0);
                    double angle = std::acos(std::abs(dot_product)) * 180.0 / M_PI;
                    total_angle += angle;
                    ++angle_count;
                }
            }
            
            analysis.triangulation_angle = angle_count > 0 ? total_angle / angle_count : 0.0;
        }
        
        analysis_vec.push_back(std::move(analysis));
    }
    
    return analysis_vec;
}

// Compare optimization results
OptimizationComparison DataStats::compareOptimization(
    const BALData& data_before, 
    const BALData& data_after,
    bool print_report) const {
    
    OptimizationComparison comparison;
    
    auto errors_before = computeReprojectionErrors(data_before);
    auto errors_after = computeReprojectionErrors(data_after);
    
    if (errors_before.size() != errors_after.size()) {
        logMessage("WARN", "Dataset sizes don't match for comparison");
        return comparison;
    }
    
    auto stats_before = computeStatistics(errors_before);
    auto stats_after = computeStatistics(errors_after);
    
    comparison.mean_error_before = stats_before.mean;
    comparison.mean_error_after = stats_after.mean;
    
    // Avoid division by zero
    if (stats_before.mean > 1e-10) {
        comparison.error_reduction_percentage = 
            ((stats_before.mean - stats_after.mean) / stats_before.mean) * 100.0;
    } else {
        comparison.error_reduction_percentage = 0.0;
    }
    
    // Count improved/degraded observations
    for (size_t i = 0; i < errors_before.size(); ++i) {
        double diff = errors_after[i] - errors_before[i];
        if (std::abs(diff) < 1e-6) {
            ++comparison.unchanged_observations;
        } else if (diff < 0) {
            ++comparison.improved_observations;
        } else {
            ++comparison.degraded_observations;
        }
    }
    
    if (print_report && verbose_output_) {
        logMessage("INFO", "=== OPTIMIZATION COMPARISON ===");
        logMessage("INFO", comparison.to_string());
    }
    
    return comparison;
}

// Print analysis report
void DataStats::printAnalysisReport(const DatasetQualityMetrics& metrics, 
                                   std::string_view label) const {
    std::string title = label.empty() ? "Dataset Quality Report" : 
                       "Dataset Quality Report: " + std::string(label);
    
    logMessage("INFO", "=== " + title + " ===");
    logMessage("INFO", "Observations: " + formatNumber(metrics.overall_reprojection_errors.count));
    logMessage("INFO", "Reprojection Errors: " + metrics.overall_reprojection_errors.to_string());
    logMessage("INFO", "Observations per Camera: " + metrics.observations_per_camera.to_string());
    logMessage("INFO", "Observations per Point: " + metrics.observations_per_point.to_string());

    logMessage("INFO", "Point Visibility Ratio: " + 
               std::to_string(metrics.point_visibility_ratio));
    logMessage("INFO", "Camera Visibility Ratio: " + 
               std::to_string(metrics.camera_visibility_ratio));
}

// Print camera error report
void DataStats::printCameraErrorReport(const std::vector<CameraErrorAnalysis>& camera_analyses, 
                                      size_t top_n) const {
    logMessage("INFO", "=== Camera Error Analysis (Top " + std::to_string(top_n) + ") ===");
    
    // Sort by mean error (descending)
    auto sorted_analyses = camera_analyses;
    std::sort(sorted_analyses.begin(), sorted_analyses.end(),
        [](const auto& a, const auto& b) {
            return a.error_stats.mean > b.error_stats.mean;
        });
    
    size_t count = std::min(top_n, sorted_analyses.size());
    for (size_t i = 0; i < count; ++i) {
        logMessage("INFO", sorted_analyses[i].to_string());
    }
}

// Print point error report
void DataStats::printPointErrorReport(const std::vector<PointErrorAnalysis>& point_analyses, 
                                     size_t top_n) const {
    logMessage("INFO", "=== Point Error Analysis (Top " + std::to_string(top_n) + ") ===");
    
    // Sort by mean error (descending)
    auto sorted_analyses = point_analyses;
    std::sort(sorted_analyses.begin(), sorted_analyses.end(),
        [](const auto& a, const auto& b) {
            return a.error_stats.mean > b.error_stats.mean;
        });
    
    size_t count = std::min(top_n, sorted_analyses.size());
    for (size_t i = 0; i < count; ++i) {
        logMessage("INFO", sorted_analyses[i].to_string());
    }
}

// Private: Project 3D point to 2D
Eigen::Vector2d DataStats::projectPoint(
    std::span<const double, CAMERA_PARAM_SIZE> camera_params,
    std::span<const double, POINT_3D_SIZE> point_3d) const noexcept {
    
    // Processed BAL camera parameters: [qw, qx, qy, qz, tx, ty, tz, focal, k1, k2]
    // Extract quaternion rotation
    Eigen::Quaterniond quat(camera_params[0], camera_params[1], camera_params[2], camera_params[3]);
    quat.normalize(); // Ensure unit quaternion
    
    // Extract translation
    Eigen::Vector3d translation(camera_params[4], camera_params[5], camera_params[6]);
    
    // Create SE3 transformation from quaternion and translation
    Sophus::SE3d T_c_w(quat, translation);
    
    // Transform 3D point to camera frame
    Eigen::Vector3d point_world(point_3d[0], point_3d[1], point_3d[2]);
    Eigen::Vector3d point_camera = T_c_w * point_world;
    
    // Project to image plane
    if (point_camera.z() <= 0) {
        return Eigen::Vector2d(0, 0); // Point behind camera
    }
    
    double x_normalized = point_camera.x() / point_camera.z();
    double y_normalized = point_camera.y() / point_camera.z();
    
    // Apply radial distortion (BAL format)
    double r2 = x_normalized * x_normalized + y_normalized * y_normalized;
    double focal_length = camera_params[7];  // Index 7 for focal length
    double k1 = camera_params[8];            // Index 8 for k1
    double k2 = camera_params[9];            // Index 9 for k2
    
    // Radial distortion correction: L(r) = 1 + k1*r^2 + k2*r^4
    double distortion_factor = 1.0 + k1 * r2 + k2 * r2 * r2;
    
    // Apply focal length and distortion
    // BAL assumes principal point at origin (0, 0)
    double u = focal_length * distortion_factor * x_normalized;
    double v = focal_length * distortion_factor * y_normalized;
    
    return Eigen::Vector2d(u, v);
}

// Private: Compute statistics
StatisticalSummary DataStats::computeStatistics(const std::vector<double>& values) const {
    if (values.empty()) {
        return StatisticalSummary{};
    }
    
    auto sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    // Mean
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    
    // Standard deviation
    double sum_sq_diff = 0.0;
    for (double val : values) {
        double diff = val - mean;
        sum_sq_diff += diff * diff;
    }
    double std_dev = std::sqrt(sum_sq_diff / values.size());
    
    // Percentiles with proper boundary handling
    size_t n = sorted_values.size();
    double median = (n % 2 == 0) ? 
        (sorted_values[n/2 - 1] + sorted_values[n/2]) / 2.0 :
        sorted_values[n/2];
    
    // Ensure indices are within bounds
    size_t q25_idx = std::max(static_cast<size_t>(0), static_cast<size_t>(0.25 * (n - 1)));
    size_t q75_idx = std::min(n - 1, static_cast<size_t>(0.75 * (n - 1)));
    
    double q25 = sorted_values[q25_idx];
    double q75 = sorted_values[q75_idx];
    
    return StatisticalSummary{
        mean, std_dev, median, 
        sorted_values.front(), sorted_values.back(),
        q25, q75, values.size()
    };
}

// Private: Log message
void DataStats::logMessage(std::string_view level, std::string_view message) const {
    if (level == "INFO") {
        RCLCPP_INFO(node_->get_logger(), "%s", message.data());
    } else if (level == "WARN") {
        RCLCPP_WARN(node_->get_logger(), "%s", message.data());
    } else if (level == "ERROR") {
        RCLCPP_ERROR(node_->get_logger(), "%s", message.data());
    }
    
    if (verbose_output_) {
        std::cout << "[" << level << "] " << message << std::endl;
    }
}

// Private: Format number
std::string DataStats::formatNumber(double value) const noexcept {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1);
    
    if (value >= 1e6) {
        ss << (value / 1e6) << "M";
    } else if (value >= 1e3) {
        ss << (value / 1e3) << "K";
    } else {
        ss << std::setprecision(0) << value;
    }
    
    return ss.str();
}

// Private: Create histogram
std::unordered_map<double, size_t> DataStats::createHistogram(
    const std::vector<double>& values, size_t bins) const {
    std::unordered_map<double, size_t> histogram;
    
    if (values.empty() || bins == 0) {
        return histogram;
    }
    
    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    double min_val = *min_it;
    double max_val = *max_it;
    double bin_width = (max_val - min_val) / bins;
    
    if (bin_width == 0) {
        histogram[min_val] = values.size();
        return histogram;
    }
    
    for (double value : values) {
        size_t bin_idx = static_cast<size_t>((value - min_val) / bin_width);
        if (bin_idx >= bins) bin_idx = bins - 1;
        
        double bin_center = min_val + (bin_idx + 0.5) * bin_width;
        histogram[bin_center]++;
    }
    
    return histogram;
}

// Struct to_string() implementations
std::string CameraErrorAnalysis::to_string() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "Camera " << camera_id << ": " << error_stats.to_string() 
       << " (obs=" << num_observations << ")";
    return ss.str();
}

std::string PointErrorAnalysis::to_string() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "Point " << point_id << ": " << error_stats.to_string() 
       << " (obs=" << num_observations << ", angle=" << triangulation_angle << "°)";
    return ss.str();
}

std::string DatasetQualityMetrics::to_string() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "Quality Metrics:\n";
    ss << "  Reprojection Errors: " << overall_reprojection_errors.to_string() << "\n";
    ss << "  Point Visibility: " << point_visibility_ratio << "\n";
    ss << "  Camera Visibility: " << camera_visibility_ratio;
    return ss.str();
}

std::string OptimizationComparison::to_string() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "Optimization Results:\n";
    ss << "  Error Reduction: " << error_reduction_percentage << "%\n";
    ss << "  Before: " << mean_error_before << " pixels\n";
    ss << "  After: " << mean_error_after << " pixels\n";
    ss << "  Improved: " << improved_observations << "\n";
    ss << "  Degraded: " << degraded_observations << "\n";
    ss << "  Unchanged: " << unchanged_observations << "\n";
    ss << "  Improvement Ratio: " << improvement_ratio() * 100.0 << "%";
    return ss.str();
}

} // namespace bal_cppcon
