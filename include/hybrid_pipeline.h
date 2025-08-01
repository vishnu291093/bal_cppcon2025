#ifndef BAL_CPPCON_HYBRID_PIPELINE_H
#define BAL_CPPCON_HYBRID_PIPELINE_H

#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>
#include <utils/data_stats.h>
#include <parser/data_model.h>
#include <memory>
#include <string_view>  // C++20: efficient string parameters
#include <span>         // C++20: safe container access
#include <random>
#include <algorithm>
#include <ranges>       // C++20: ranges and views
#include <execution>  // C++17: parallel execution policies
#include <optional>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>

namespace bal_cppcon {

struct HybridPipelineConfig {
    // C++20: explicit constructor with better initialization
    explicit HybridPipelineConfig(std::shared_ptr<rclcpp::Node> node) noexcept;
    
    // Quality assessment thresholds
    double good_error_threshold{1.0};        // pixels - below this is "good"
    double moderate_error_threshold{3.0};    // pixels - below this is "moderate"
    
    // Default analysis parameters
    size_t worst_cameras_stats{10};          // Default number of worst cameras to show
    size_t worst_points_stats{10};           // Default number of worst points to show
    
    // Output formatting precision
    int error_precision{2};                  // Decimal places for error reduction %
    
    // DataStats configuration
    bool verbose_logging{true};               // Verbose output for DataStats
    
    // Camera window size for the moving optimization
    size_t camera_window_size{10};

    bool cyclic_window{false};

    size_t window_step_size{4};
    
    // RANSAC parameters
    int max_ransac_iterations{1000};
    double reprojection_threshold{2.0};      // pixels
    int min_sample_size{8};
    
    // C++20: constexpr validation method
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return good_error_threshold > 0.0 && 
               moderate_error_threshold > good_error_threshold &&
               worst_cameras_stats > 0 && 
               worst_points_stats > 0 &&
               error_precision >= 0 &&
               camera_window_size > 0 &&
               max_ransac_iterations > 0 &&
               reprojection_threshold > 0.0 &&
               min_sample_size > 0;
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
    
    // Helper method to read parameters from ROS
    void loadParameters();
};



    class HybridPipeline {
        public:

            using Point3D = boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;
            using Value = std::pair<Point3D, int>;

            // C++20: explicit constructor with modern shared_ptr
            explicit HybridPipeline(std::shared_ptr<rclcpp::Node> node, BALData data) noexcept;
            ~HybridPipeline() = default;
            
            // C++20: explicit move semantics
            HybridPipeline(const HybridPipeline&) = delete;
            HybridPipeline& operator=(const HybridPipeline&) = delete;
            HybridPipeline(HybridPipeline&&) = default;
            HybridPipeline& operator=(HybridPipeline&&) = default;
            
            /**
             * @brief Analyze and print initial dataset metrics
             * @param label Optional label for the analysis report (string_view for efficiency)
             */
            void analyzeInitialDataset(std::string_view label = "Initial Dataset") const;
            
            /**
             * @brief Analyze and print detailed camera error metrics
             */
            void analyzeCameraErrors() const;
            
            /**
             * @brief Analyze and print detailed point error metrics  
             */
            void analyzePointErrors() const;
            
            /**
             * @brief Compare optimization results before and after
             * @param optimized_data Dataset after optimization
             * @param print_detailed_report Whether to print detailed comparison
             */
            void compareOptimizationResults(const BALData& optimized_data, 
                                          bool print_detailed_report = true) const;
            
            /**
             * @brief Print comprehensive dataset quality report
             * @param label Optional label for the report (string_view for efficiency)
             */
            void printDatasetQualityReport(std::string_view label = "Dataset Quality") const;
            
            /**
             * @brief Get the current dataset
             * @return Reference to the stored BALData
             */
            [[nodiscard]] const BALData& getData() const noexcept { return data_; }
            
            /**
             * @brief Get configuration reference
             * @return Reference to the configuration
             */
            [[nodiscard]] const HybridPipelineConfig& getConfig() const noexcept { return config_; }
            
            /**
             * @brief Check if pipeline is properly initialized
             * @return true if all components are valid
             */
            [[nodiscard]] constexpr bool isValid() const noexcept {
                return node_ != nullptr && data_stats_ != nullptr && config_.is_valid();
            }

            void runWindowedOptimization();

            std::unordered_map<int, std::vector<int>> runSinglePassOutlierRejection(int window_start_node);



        private:
            std::shared_ptr<rclcpp::Node> node_;
            std::unique_ptr<DataStats> data_stats_;
            BALData data_;
            HybridPipelineConfig config_;
            boost::geometry::index::rtree<Value, boost::geometry::index::linear<16>> rtree_;
            std::unordered_map<int, std::set<int>> camera_to_valid_observations_map{};

            // Random number generator for RANSAC
            mutable std::mt19937 rng_;
            
            // Mutex for thread-safe operations
            mutable std::mutex single_pass_mutex_;
            
            // C++20: private helper with string_view
            void logMessage(std::string_view level, std::string_view message) const noexcept;
            
            std::vector<Observation> randomSample(std::span<const Observation> observations, int sample_size) const;
            Vec2 projectPoint(const Vec3& point_3d, const Camera& camera) const;

            void setupRTree();
            void performSpatialFiltering(const int window_start, const std::unordered_map<int, std::vector<int>>& filter_observations);
    };
}

#endif // BAL_CPPCON_HYBRID_PIPELINE_H
