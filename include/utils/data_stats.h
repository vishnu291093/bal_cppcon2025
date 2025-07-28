#ifndef BAL_CPPCON_DATA_STATS_H
#define BAL_CPPCON_DATA_STATS_H

#include <rclcpp/rclcpp.hpp>
#include <parser/data_model.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <string_view>  // C++20: efficient string parameters
#include <memory>
#include <span>         // C++20: safe array access
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <Eigen/Core>

namespace bal_cppcon {

// Statistical summary structure
struct StatisticalSummary {
    double mean{0.0};
    double std_dev{0.0};
    double median{0.0};
    double min{0.0};
    double max{0.0};
    double q25{0.0};          // 25th percentile
    double q75{0.0};          // 75th percentile
    size_t count{0};
    
    // C++20: designated initializer constructor
    constexpr StatisticalSummary() = default;
    constexpr StatisticalSummary(double m, double s, double med, double mn, double mx, 
                                double q_25, double q_75, size_t c) noexcept
        : mean{m}, std_dev{s}, median{med}, min{mn}, max{mx}, q25{q_25}, q75{q_75}, count{c} {}
    
    [[nodiscard]] std::string to_string() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "Stats(μ=" << mean << ", σ=" << std_dev << ", med=" << median 
           << ", range=[" << min << ", " << max << "], n=" << count << ")";
        return ss.str();
    }
};

// Reprojection error analysis per camera
struct CameraErrorAnalysis {
    int camera_id{-1};
    StatisticalSummary error_stats{};
    size_t num_observations{0};
    std::vector<double> individual_errors{};
    
    [[nodiscard]] std::string to_string() const;
};

// Reprojection error analysis per 3D point
struct PointErrorAnalysis {
    int point_id{-1};
    StatisticalSummary error_stats{};
    size_t num_observations{0};
    double triangulation_angle{0.0};    // Average viewing angle
    std::vector<double> individual_errors{};
    
    [[nodiscard]] std::string to_string() const;
};

// Overall dataset quality metrics
struct DatasetQualityMetrics {
    StatisticalSummary overall_reprojection_errors{};
    StatisticalSummary observations_per_camera{};
    StatisticalSummary observations_per_point{};

    // Coverage metrics
    double point_visibility_ratio{0.0};
    double camera_visibility_ratio{0.0};
    
    // Data distribution
    std::unordered_map<int, size_t> observations_per_camera_histogram{};
    std::unordered_map<int, size_t> observations_per_point_histogram{};
    
    [[nodiscard]] std::string to_string() const;
};

// Comparison metrics between two datasets
struct OptimizationComparison {
    double error_reduction_percentage{0.0};
    double mean_error_before{0.0};
    double mean_error_after{0.0};
    size_t improved_observations{0};
    size_t degraded_observations{0};
    size_t unchanged_observations{0};
    
    // C++20: spaceship operator (useful for comparison)
    auto operator<=>(const OptimizationComparison&) const = default;
    
    // Computed properties
    [[nodiscard]] constexpr size_t total_observations() const noexcept {
        return improved_observations + degraded_observations + unchanged_observations;
    }
    
    [[nodiscard]] constexpr double improvement_ratio() const noexcept {
        auto total = total_observations();
        return total > 0 ? static_cast<double>(improved_observations) / total : 0.0;
    }
    
    [[nodiscard]] std::string to_string() const;
};

class DataStats {
public:
    /**
     * @brief Constructor taking ROS node for parameter access
     * @param node ROS node shared pointer for accessing parameters
     * @param outlier_threshold Standard deviations for outlier detection
     * @param verbose_output Whether to print detailed output
     */
    explicit DataStats(std::shared_ptr<rclcpp::Node> node, 
                      double outlier_threshold = 3.0, 
                      bool verbose_output = true) noexcept;
    
    ~DataStats() = default;
    
    // C++20: explicit move semantics
    DataStats(const DataStats&) = delete;
    DataStats& operator=(const DataStats&) = delete;
    DataStats(DataStats&&) = default;
    DataStats& operator=(DataStats&&) = default;
    
    /**
     * @brief Analyze overall dataset quality
     * @param data The BAL dataset to analyze
     * @param label Optional label for the analysis
     * @return DatasetQualityMetrics containing all computed statistics
     */
    [[nodiscard]] DatasetQualityMetrics analyzeDataset(const BALData& data, 
                                                       std::string_view label = "") const;
    
    /**
     * @brief Compute reprojection errors for all observations
     * @param data The BAL dataset
     * @return Vector of reprojection errors (in pixels)
     */
    [[nodiscard]] std::vector<double> computeReprojectionErrors(const BALData& data) const;
    
    /**
     * @brief Analyze errors per camera
     * @param data The BAL dataset
     * @return Vector of camera-specific error analyses
     */
    [[nodiscard]] std::vector<CameraErrorAnalysis> analyzeCameraErrors(const BALData& data) const;
    
    /**
     * @brief Analyze errors per 3D point
     * @param data The BAL dataset
     * @return Vector of point-specific error analyses
     */
    [[nodiscard]] std::vector<PointErrorAnalysis> analyzePointErrors(const BALData& data) const;
    
    /**
     * @brief Compare two datasets (before and after optimization)
     * @param data_before Dataset before optimization
     * @param data_after Dataset after optimization
     * @param print_report Whether to print detailed comparison report
     * @return OptimizationComparison metrics
     */
    [[nodiscard]] OptimizationComparison compareOptimization(
        const BALData& data_before, 
        const BALData& data_after,
        bool print_report = true) const;
    
    /**
     * @brief Print detailed analysis report
     * @param metrics The computed dataset quality metrics
     * @param label Optional label for the report
     */
    void printAnalysisReport(const DatasetQualityMetrics& metrics, 
                           std::string_view label = "") const;
    
    /**
     * @brief Print camera-specific error analysis
     * @param camera_analyses Vector of camera error analyses
     * @param top_n Number of worst cameras to highlight
     */
    void printCameraErrorReport(const std::vector<CameraErrorAnalysis>& camera_analyses, 
                              size_t top_n = 5) const;
    
    /**
     * @brief Print point-specific error analysis
     * @param point_analyses Vector of point error analyses
     * @param top_n Number of worst points to highlight
     */
    void printPointErrorReport(const std::vector<PointErrorAnalysis>& point_analyses, 
                             size_t top_n = 5) const;

private:
    std::shared_ptr<rclcpp::Node> node_;
    double outlier_threshold_{3.0};
    bool verbose_output_{true};
    
    /**
     * @brief Project 3D point to 2D using camera parameters
     * @param camera_params Camera parameters (span for safety)
     * @param point_3d 3D point coordinates (span for safety)
     * @return 2D pixel coordinates
     */
    [[nodiscard]] Eigen::Vector2d projectPoint(
        std::span<const double, CAMERA_PARAM_SIZE> camera_params,
        std::span<const double, POINT_3D_SIZE> point_3d) const noexcept;
    
    /**
     * @brief Compute statistical summary from vector of values
     * @param values Vector of double values
     * @return StatisticalSummary with computed statistics
     */
    [[nodiscard]] StatisticalSummary computeStatistics(const std::vector<double>& values) const;
    
    /**
     * @brief Log message to ROS logger and console
     * @param level Log level
     * @param message Message to log
     */
    void logMessage(std::string_view level, std::string_view message) const;
    
    /**
     * @brief Format numbers with appropriate units (K, M, etc.)
     * @param value Numeric value to format
     * @return Formatted string
     */
    [[nodiscard]] std::string formatNumber(double value) const noexcept;
    
    /**
     * @brief Create histogram from vector of values
     * @param values Vector of values
     * @param bins Number of histogram bins
     * @return Histogram as unordered_map
     */
    [[nodiscard]] std::unordered_map<double, size_t> createHistogram(
        const std::vector<double>& values, size_t bins = 20) const;
};

} // namespace bal_cppcon

#endif // BAL_CPPCON_DATA_STATS_H
