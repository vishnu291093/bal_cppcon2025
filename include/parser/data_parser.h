#ifndef BAL_CPPCON_DATA_PARSER_H
#define BAL_CPPCON_DATA_PARSER_H

#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <unordered_map>
#include <random>
#include <filesystem>
#include <span>
#include <array>
#include <concepts>
#include <optional>
#include <ranges>
#include <expected>
#include <fstream>
#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <parser/data_model.h>

namespace bal_cppcon {

// C++20 concepts for type safety
template<typename T>
concept DatasetPath = std::convertible_to<T, std::filesystem::path>;

// Error handling with C++23 expected
enum class ParseError {
    FileNotFound,
    InvalidFormat,
    CorruptedData,
    InsufficientData,
    InvalidParameters
};

[[nodiscard]] constexpr std::string_view parse_error_message(ParseError error) noexcept {
    switch (error) {
        case ParseError::FileNotFound: return "Dataset file not found";
        case ParseError::InvalidFormat: return "Invalid dataset format";
        case ParseError::CorruptedData: return "Corrupted dataset data";
        case ParseError::InsufficientData: return "Insufficient data for processing";
        case ParseError::InvalidParameters: return "Invalid parsing parameters";
        default: return "Unknown error";
    }
}

// Parser configuration with C++20 designated initializers
struct ParserConfig {
    bool perform_filtering{false};
    double depth_threshold{0.0};
    double scale{100.0};
    bool normalize_data{true};
    size_t min_observations_per_landmark{2};
    
    // C++20 designated initializer constructor
    constexpr ParserConfig() = default;
    
    // Constructor for brace initialization
    constexpr ParserConfig(bool filter, double depth_thresh, double sc, bool normalize, size_t min_obs) noexcept
        : perform_filtering{filter}, depth_threshold{depth_thresh}, scale{sc}, 
          normalize_data{normalize}, min_observations_per_landmark{min_obs} {}
    
    // Validation method
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return scale > 0.0 && min_observations_per_landmark > 0 && depth_threshold >= 0.0;
    }
};
// Main parser class with C++20/C++23 improvements
class DataParser {
public:
    // C++20 concepts in constructor
    template<DatasetPath PathType>
    explicit DataParser(PathType&& dataset_path, ParserConfig config = {}) 
        : dataset_path_{std::forward<PathType>(dataset_path)}, config_{std::move(config)} {
        static_assert(std::is_constructible_v<std::filesystem::path, PathType>);
    }
    
    // Deleted copy/move operations for now (can be implemented if needed)
    DataParser(const DataParser&) = delete;
    DataParser& operator=(const DataParser&) = delete;
    DataParser(DataParser&&) = default;
    DataParser& operator=(DataParser&&) = default;
    
    ~DataParser() = default;

    // C++23 expected for error handling
    void load_dataset();
    void normalize() noexcept;
    void filter_observations() noexcept;

    // Modern const-correct getters with C++20 features
    [[nodiscard]] constexpr std::span<const Camera> cameras() const noexcept { 
        return std::span{cameras_}; 
    }
    
    [[nodiscard]] constexpr std::span<const Landmark> landmarks() const noexcept { 
        return std::span{landmarks_}; 
    }
    
    [[nodiscard]] constexpr size_t num_cameras() const noexcept { 
        return cameras_.size(); 
    }
    
    [[nodiscard]] constexpr size_t num_landmarks() const noexcept { 
        return landmarks_.size(); 
    }
    
    
    // C++20 ranges-friendly access
    [[nodiscard]] auto valid_cameras() const noexcept {
        return cameras_ | std::views::filter([](const auto& cam) { 
            return cam.intrinsics.is_valid(); 
        });
    }
    
    [[nodiscard]] auto valid_landmarks() const noexcept {
        return landmarks_ | std::views::filter([](const auto& landmark) { 
            return landmark.is_valid(); 
        });
    }

    [[nodiscard]] BALData extract_data() const noexcept;
    
    // Configuration access
    [[nodiscard]] constexpr const ParserConfig& config() const noexcept { return config_; }
    [[nodiscard]] constexpr const std::filesystem::path& dataset_path() const noexcept { return dataset_path_; }

private:
    // Internal validation methods
    [[nodiscard]] constexpr bool is_valid_observation(const Vec3& point_camera, double threshold) const noexcept {
        return point_camera.z() > threshold;
    }
    
    void load_bal_format() noexcept;
    [[nodiscard]] Scalar compute_median(std::span<Scalar> data) const noexcept;
    
    // C++20 constexpr math operations
    [[nodiscard]] constexpr Eigen::Quaterniond angle_axis_to_quaternion(const Eigen::Vector3d& angle_axis) const noexcept;
    [[nodiscard]] constexpr Eigen::Quaterniond convert_camera_rotation(const Eigen::Vector3d& rotation) const noexcept;

private:
    std::filesystem::path dataset_path_;
    ParserConfig config_{};
    std::vector<Camera> cameras_{};
    std::vector<Landmark> landmarks_{};
};


// Factory function with C++20 concepts
template<DatasetPath PathType>
[[nodiscard]] std::expected<std::unique_ptr<DataParser>, ParseError> 
create_parser(PathType&& path, ParserConfig config = {}) {
    if (!std::filesystem::exists(path)) {
        return std::unexpected{ParseError::FileNotFound};
    }
    
    if (!config.is_valid()) {
        return std::unexpected{ParseError::InvalidParameters};
    }
    return std::make_unique<DataParser>(std::forward<PathType>(path), std::move(config));
}

} // namespace bal_cppcon

#endif // BAL_CPPCON_DATA_PARSER_H