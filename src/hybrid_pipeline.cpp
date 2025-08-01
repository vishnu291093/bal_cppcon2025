#include <hybrid_pipeline.h>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <ranges>
#include <optional>
#include <execution>  // C++17: parallel execution policies


namespace bal_cppcon {

    HybridPipelineConfig::HybridPipelineConfig(std::shared_ptr<rclcpp::Node> node) noexcept
        : node_(node)
    {
        loadParameters();
    }
    
    void HybridPipelineConfig::loadParameters() {
        // Quality assessment thresholds
        good_error_threshold = node_->declare_parameter("hybrid_pipeline.good_error_threshold", 1.0);
        moderate_error_threshold = node_->declare_parameter("hybrid_pipeline.moderate_error_threshold", 3.0);
        
        // Default analysis parameters
        worst_cameras_stats = static_cast<size_t>(node_->declare_parameter("hybrid_pipeline.worst_cameras_stats", 10));
        worst_points_stats = static_cast<size_t>(node_->declare_parameter("hybrid_pipeline.worst_points_stats", 10));
        
        // Output formatting precision
        error_precision = node_->declare_parameter("hybrid_pipeline.error_precision", 2);
        
        // DataStats configuration
        verbose_logging = node_->declare_parameter("hybrid_pipeline.verbose_logging", true);

        // camera window size for the moving optimization
        camera_window_size = static_cast<size_t>(node_->declare_parameter("hybrid_pipeline.camera_window_size", 10));
        cyclic_window = node_->declare_parameter("hybrid_pipeline.cyclic_window", false);
        window_step_size = static_cast<size_t>(node_->declare_parameter("hybrid_pipeline.window_step_size", 4));
        
        // RANSAC parameters
        max_ransac_iterations = node_->declare_parameter("hybrid_pipeline.max_ransac_iterations", 1000);
        reprojection_threshold = node_->declare_parameter("hybrid_pipeline.reprojection_threshold", 200.0);
        min_sample_size = node_->declare_parameter("hybrid_pipeline.min_sample_size", 8);
    }

    HybridPipeline::HybridPipeline(std::shared_ptr<rclcpp::Node> node, BALData data) noexcept
        : node_(node),
        config_(node),
        data_stats_(std::make_unique<DataStats>(node, 3.0, config_.verbose_logging)),
        data_(std::move(data)),  // C++20: move semantics for large data
        rng_(std::random_device{}())
    {
        if (config_.verbose_logging) {
            logMessage("INFO", "HybridPipeline initialized with " + 
                      std::to_string(data_.num_cameras()) + " cameras, " +
                      std::to_string(data_.num_points()) + " points, " +
                      std::to_string(data_.num_observations()) + " observations");
        }
    }
    
    void HybridPipeline::analyzeInitialDataset(std::string_view label) const {
        if (config_.verbose_logging) {
            logMessage("INFO", "=== Analyzing Initial Dataset: " + std::string(label) + " ===");
        }
        
        // Perform comprehensive dataset analysis
        auto metrics = data_stats_->analyzeDataset(data_, std::string(label));
        
        // Print additional summary information based on config
        if (config_.verbose_logging) {
            logMessage("INFO", "Dataset loaded successfully:");
            logMessage("INFO", "  - Cameras: " + std::to_string(data_.num_cameras()));
            logMessage("INFO", "  - 3D Points: " + std::to_string(data_.num_points()));
            logMessage("INFO", "  - Observations: " + std::to_string(data_.num_observations()));
            
            // C++20 span: Analyze camera parameter ranges using spans
            if (!data_.camera_params.empty()) {
                auto first_camera_span = std::span<const double, CAMERA_PARAM_SIZE>{data_.camera_params[0]};
                logMessage("INFO", "  - First camera focal length: " + std::to_string(first_camera_span[7]));
            }
            
            // C++20 span: Analyze point coordinate ranges using spans  
            if (!data_.points.empty()) {
                auto first_point_span = std::span<const double, POINT_3D_SIZE>{data_.points[0]};
                logMessage("INFO", "  - First point coordinates: [" + 
                          std::to_string(first_point_span[0]) + ", " +
                          std::to_string(first_point_span[1]) + ", " +
                          std::to_string(first_point_span[2]) + "]");
            }
        }
    }
    
    void HybridPipeline::analyzeCameraErrors() const {
        if (config_.verbose_logging) {
            logMessage("INFO", "=== Analyzing Camera-Specific Errors ===");
        }
        
        // Get camera error analysis
        auto camera_analyses = data_stats_->analyzeCameraErrors(data_);
        
        // Print the report
        data_stats_->printCameraErrorReport(camera_analyses, config_.worst_cameras_stats);
        
        if (config_.verbose_logging) {
            logMessage("INFO", "Camera error analysis completed for " + 
                      std::to_string(camera_analyses.size()) + " cameras (showing top " +
                      std::to_string(config_.worst_cameras_stats) + ")");
        }
    }
    
    void HybridPipeline::analyzePointErrors() const {
        if (config_.verbose_logging) {
            logMessage("INFO", "=== Analyzing Point-Specific Errors ===");
        }
        
        
        // Get point error analysis
        auto point_analyses = data_stats_->analyzePointErrors(data_);
        
        // Print the report
        data_stats_->printPointErrorReport(point_analyses, config_.worst_points_stats);
        
        if (config_.verbose_logging) {
            logMessage("INFO", "Point error analysis completed for " + 
                      std::to_string(point_analyses.size()) + " points (showing top " +
                      std::to_string(config_.worst_points_stats) + ")");
        }
    }
    
    void HybridPipeline::compareOptimizationResults(const BALData& optimized_data, 
                                                   bool print_detailed_report) const {
        if (config_.verbose_logging) {
            logMessage("INFO", "=== Comparing Optimization Results ===");
        }
        
       
        // Perform before/after comparison
        auto comparison = data_stats_->compareOptimization(data_, optimized_data,  config_.verbose_logging);
        
        // Print summary results with configurable precision
        logMessage("INFO", "Optimization Summary:");
        
        // C++20: Create helper for formatting numeric results with precision
        auto format_with_precision = [precision = config_.error_precision](double value, std::string_view label) -> std::string {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(precision) << value;
            return std::string(label) + oss.str();
        };
        
        logMessage("INFO", format_with_precision(comparison.error_reduction_percentage, "  - Error Reduction: ") + "%");
        logMessage("INFO", format_with_precision(comparison.mean_error_before, "  - Mean Error Before: ") + " pixels");
        logMessage("INFO", format_with_precision(comparison.mean_error_after, "  - Mean Error After: ") + " pixels");
        logMessage("INFO", "  - Improved Observations: " + std::to_string(comparison.improved_observations));
        logMessage("INFO", "  - Degraded Observations: " + std::to_string(comparison.degraded_observations));
        logMessage("INFO", "  - Unchanged Observations: " + std::to_string(comparison.unchanged_observations));
        logMessage("INFO", format_with_precision(comparison.improvement_ratio() * 100.0, "  - Improvement Ratio: ") + "%");
    }
    

    void HybridPipeline::printDatasetQualityReport(std::string_view label) const {
        if (config_.verbose_logging) {
            logMessage("INFO", "=== Comprehensive Dataset Quality Report ===");
        }
        
        // Get comprehensive metrics
        auto metrics = data_stats_->analyzeDataset(data_, std::string(label));
        
        // Print the detailed report if configured
        if (config_.verbose_logging) {
            data_stats_->printAnalysisReport(metrics, std::string(label));
        }
        
        // Print additional insights with quality indicators
        if (config_.verbose_logging) {
            logMessage("INFO", "Quality Assessment:");
            
            // C++20: Create error message formatter with precision
            auto create_error_msg = [precision = config_.error_precision](std::string_view status, std::string_view quality, double error_value) -> std::string {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(precision) << error_value;
                return std::string(status) + std::string(quality) + " reprojection errors (" + oss.str() + " pixels)";
            };
            
            // Assess reprojection error quality using config thresholds
            if (metrics.overall_reprojection_errors.mean < config_.good_error_threshold) {
                std::string good_msg = create_error_msg("  ✓ Good: ", "Low", metrics.overall_reprojection_errors.mean);
                logMessage("INFO", good_msg);
            } else if (metrics.overall_reprojection_errors.mean < config_.moderate_error_threshold) {
                std::string moderate_msg = create_error_msg("  ⚠ Moderate: ", "Medium", metrics.overall_reprojection_errors.mean);
                logMessage("WARN", moderate_msg);
            } else {
                std::string poor_msg = create_error_msg("  ✗ Poor: ", "High", metrics.overall_reprojection_errors.mean);
                logMessage("ERROR", poor_msg);
            }
        }
    }

    std::unordered_map<int, std::vector<int>> HybridPipeline::runSinglePassOutlierRejection(int window_start_node) {
        if (config_.verbose_logging) {
            logMessage("INFO", "Running RANSAC for window starting at camera " + std::to_string(window_start_node));
        }

        // C++20 ranges: Create camera window using views
        auto camera_indices = std::views::iota(window_start_node, 
                                             std::min(window_start_node + static_cast<int>(config_.camera_window_size), 
                                                     static_cast<int>(data_.num_cameras())));
        
        // C++20 ranges: Create camera window using direct camera access
        auto window_cameras = camera_indices 
                            | std::views::transform([this](int idx) { 
                                return data_.cameras[idx]; 
                              });
        
        // Store filtered observations using existing structures
        std::unordered_map<int, std::vector<int>> filtered_observation_indices;
        filtered_observation_indices.reserve(window_cameras.size());
        
        // Convert range to vector for parallel processing
        std::vector<Camera> cameras_vec{window_cameras.begin(), window_cameras.end()};

        // Run outlier rejection for each camera in parallel using std::execution
        std::for_each(std::execution::par, cameras_vec.begin(), cameras_vec.end(),
            [this, &filtered_observation_indices](const Camera& camera) {
                // Use camera.id directly instead of calculating from indices
                int camera_id = camera.id;
                
                // C++20 ranges: Use camera_to_observations_map for direct access
                auto obs_it = data_.camera_to_observations_map.find(camera_id);
                if (obs_it == data_.camera_to_observations_map.end()) {
                    if (config_.verbose_logging) {
                        logMessage("WARN", "No observations found for camera " + std::to_string(camera_id));
                    }
                    return; // Continue to next camera
                }

                // Work directly with the set of indices - no intermediate vector needed
                const auto& obs_indices_set = obs_it->second;

                // Use the set directly in the outlier rejection
                if (obs_indices_set.empty()) {
                    if (config_.verbose_logging) {
                        logMessage("WARN", "No observations found for camera " + std::to_string(camera_id));
                    }
                    return; // Continue to next camera
                }

                // Single-pass outlier rejection (no need for multiple iterations)
                // C++20 ranges: Filter inliers using views and functional approach
                auto compute_error = [this, &camera](int obs_idx) -> std::optional<std::pair<int, double>> {
                    const auto& obs = data_.observations[obs_idx];
                    
                    // Validate point index
                    if (obs.point_index >= static_cast<int>(data_.num_points())) {
                        return std::nullopt;
                    }
                    
                    // Use span for point coordinates
                    auto point_coords = data_.point_coordinates(obs.point_index);
                    if (point_coords.empty()) {
                        return std::nullopt;
                    }
                    
                    // Create span for safer access to point coordinates
                    std::span<const double> coords_span{point_coords};
                    Vec3 point_3d(coords_span[0], coords_span[1], coords_span[2]);
                    
                    // Project 3D point using the known camera pose and intrinsics
                    Vec2 projected = projectPoint(point_3d, camera);
                    Vec2 observed = obs.pixel_coordinates();

                    // Compute reprojection error
                    double error = (projected - observed).norm();
                    
                    return std::make_pair(obs_idx, error);
                };

                // C++20 ranges: Transform observation indices to error pairs, filter valid ones, then filter inliers
                auto inliers = obs_indices_set 
                             | std::views::transform(compute_error)
                             | std::views::filter([](const auto& opt) { return opt.has_value(); })
                             | std::views::transform([](const auto& opt) { return opt.value(); })
                             | std::views::filter([this](const auto& pair) { 
                                 return pair.second < config_.reprojection_threshold; 
                               })
                             | std::views::transform([](const auto& pair) { return pair.first; });

                // Convert inliers view to vector
                std::vector<int> best_consensus_set{inliers.begin(), inliers.end()};

                // Thread-safe insertion into results map
                {
                    std::lock_guard<std::mutex> lock(single_pass_mutex_);
                    filtered_observation_indices[camera_id] = best_consensus_set;
                }
                
                if (config_.verbose_logging) {
                    double inlier_ratio = static_cast<double>(best_consensus_set.size()) / obs_indices_set.size();
                    logMessage("INFO", "Camera " + std::to_string(camera_id) + 
                              ": " + std::to_string(best_consensus_set.size()) + "/" + 
                              std::to_string(obs_indices_set.size()) + " inliers (" + 
                              std::to_string(static_cast<int>(inlier_ratio * 100)) + "%)");
                }
            });
        
        if (config_.verbose_logging) {
            logMessage("INFO", "RANSAC completed for window starting at camera " + std::to_string(window_start_node));
        }

        return filtered_observation_indices;
    }

    void HybridPipeline::setupRTree() {
        // Create R-tree for spatial filtering
        int count = 0;
        for (const auto& point : data_.points) {
            rtree_.insert(std::make_pair(Point3D(point[0], point[1], point[2]), count++));
        }
    }

    void HybridPipeline::performSpatialFiltering(const int window_start, const std::unordered_map<int, std::vector<int>>& filter_observations) {
    }

    void HybridPipeline::runWindowedOptimization() {
        if (config_.verbose_logging) {
            logMessage("INFO", "=== Running Moving Optimization ===");
            logMessage("INFO", "Camera window size: " + std::to_string(config_.camera_window_size));
        }

        setupRTree();

        // C++20 ranges: Generate window start positions using views
        auto window_starts = std::views::iota(0, static_cast<int>(data_.num_cameras())) 
                           | std::views::stride(static_cast<int>(config_.window_step_size));
        
        // Process each window using ranges
        for (int window_start : window_starts) {
           auto filter_observations = runSinglePassOutlierRejection(window_start);

            performSpatialFiltering(window_start, filter_observations);
        }
        
        // TODO: Implement moving window optimization logic
        // For now, just log that it's a placeholder
        logMessage("INFO", "Moving optimization implementation pending...");
    }
    
    std::vector<Observation> HybridPipeline::randomSample(
        std::span<const Observation> observations, int sample_size) const {
        
        if (sample_size >= static_cast<int>(observations.size())) {
            return std::vector<Observation>{observations.begin(), observations.end()};
        }
        
        // C++20 ranges: Create indices view and shuffle
        std::vector<int> indices(observations.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);
        
        // C++20 ranges: Take first sample_size indices and transform to observations using span
        auto sampled_indices = indices | std::views::take(sample_size);
        auto sampled_observations = sampled_indices 
                                  | std::views::transform([observations](int idx) { 
                                      return observations[idx]; 
                                    });
        
        return std::vector<Observation>{sampled_observations.begin(), sampled_observations.end()};
    }
    
    Vec2 HybridPipeline::projectPoint(const Vec3& point_3d, const Camera& camera) const {
        // Transform point to camera frame using the Camera's SE3 pose
        Vec3 point_cam = camera.T_c_w * point_3d;
        
        // Check if point is in front of camera
        if (point_cam.z() <= 0.0) {
            return Vec2::Zero(); // Return zero for points behind camera
        }
        
        // Project to normalized image coordinates
        double x_norm = point_cam.x() / point_cam.z();
        double y_norm = point_cam.y() / point_cam.z();
        
        // Apply radial distortion using the Camera's intrinsics
        double r_squared = x_norm * x_norm + y_norm * y_norm;
        double distortion = camera.intrinsics.distortion_factor(r_squared);
        
        double x_distorted = x_norm * distortion;
        double y_distorted = y_norm * distortion;
        
        // Apply focal length to get pixel coordinates
        Vec2 projected;
        projected.x() = camera.intrinsics.focal_length * x_distorted;
        projected.y() = camera.intrinsics.focal_length * y_distorted;
        
        return projected;
    }
    
    // C++20: Private helper method with string_view for efficiency
    void HybridPipeline::logMessage(std::string_view level, std::string_view message) const noexcept {
        if (level == "INFO") {
            RCLCPP_INFO(node_->get_logger(), "%s", message.data());
        } else if (level == "WARN") {
            RCLCPP_WARN(node_->get_logger(), "%s", message.data());
        } else if (level == "ERROR") {
            RCLCPP_ERROR(node_->get_logger(), "%s", message.data());
        }
    }
}
