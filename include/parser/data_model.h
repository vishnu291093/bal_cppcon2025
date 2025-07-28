#ifndef BAL_CPPCON_DATA_MODEL_H
#define BAL_CPPCON_DATA_MODEL_H

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <span>
#include <array>
#include <sstream>
#include <iomanip>
#include <concepts>
#include <type_traits>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace bal_cppcon {

// Type aliases
using Scalar = double;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec9 = Eigen::Matrix<Scalar, 9, 1>;
using SO3 = Sophus::SO3d;
using SE3 = Sophus::SE3d;

// Camera parameter constants
inline constexpr size_t CAMERA_PARAM_SIZE = 10;
inline constexpr size_t POINT_3D_SIZE = 3;

// C++20 concepts for type safety and generic programming
template<typename T>
concept EigenPoint3D = requires(T t) {
    { t.x() } -> std::convertible_to<double>;
    { t.y() } -> std::convertible_to<double>;
    { t.z() } -> std::convertible_to<double>;
    requires std::is_same_v<T, Vec3> || 
             std::is_same_v<T, Eigen::Vector3f> ||
             std::is_same_v<T, Eigen::Vector3d>;
};

template<typename T>
concept ArrayLikePoint3D = requires(T t) {
    { t[0] } -> std::convertible_to<double>;
    { t[1] } -> std::convertible_to<double>;
    { t[2] } -> std::convertible_to<double>;
    requires std::is_same_v<T, std::array<double, 3>> || 
             std::is_same_v<T, std::vector<double>>;
};

template<typename T>
concept Point3D = EigenPoint3D<T> || ArrayLikePoint3D<T>;

template<typename Container>
concept Point3DContainer = requires(Container c) {
    std::begin(c);
    std::end(c);
    requires Point3D<typename Container::value_type>;
};

template<typename T>
concept CameraParameter = requires(T t) {
    requires std::is_arithmetic_v<T>;
    requires std::convertible_to<T, double>;
};

template<typename T>
concept ObservationData = requires(T t) {
    { t.camera_index } -> std::convertible_to<int>;
    { t.point_index } -> std::convertible_to<int>;
    { t.x } -> std::convertible_to<double>;
    { t.y } -> std::convertible_to<double>;
};

// Camera intrinsics model for BAL dataset format
struct CameraModel {
    double focal_length{0.0};
    double k1{0.0};  // First radial distortion parameter
    double k2{0.0};  // Second radial distortion parameter
    
    constexpr CameraModel() = default;
    constexpr CameraModel(double f, double k1_param, double k2_param) noexcept 
        : focal_length{f}, k1{k1_param}, k2{k2_param} {}
    
    constexpr explicit CameraModel(const Vec3& parameters) noexcept 
        : focal_length{parameters[0]}, k1{parameters[1]}, k2{parameters[2]} {}
    
    // C++20: spaceship operator for default comparisons
    auto operator<=>(const CameraModel&) const = default;
    
    [[nodiscard]] constexpr Vec3 as_vec3() const noexcept { 
        return Vec3{focal_length, k1, k2}; 
    }
    
    [[nodiscard]] constexpr bool is_valid() const noexcept {
        return focal_length > 0.0;
    }
    
    [[nodiscard]] constexpr double distortion_factor(double r_squared) const noexcept {
        return 1.0 + k1 * r_squared + k2 * r_squared * r_squared;
    }
};

// Camera structure
struct Camera {
    SE3 T_c_w{};
    CameraModel intrinsics{};
    
    constexpr Camera() = default;
    constexpr Camera(const SE3& transform, const CameraModel& camera_intrinsics) noexcept 
        : T_c_w{transform}, intrinsics{camera_intrinsics} {}
    
    // C++20: spaceship operator for default comparisons
    auto operator<=>(const Camera&) const = default;
    
    [[nodiscard]] constexpr Vec3 translation() const noexcept { return T_c_w.translation(); }
    [[nodiscard]] constexpr SO3 rotation() const noexcept { return T_c_w.so3(); }
};

// Landmark with observations
struct Landmark {
    Vec3 p_w{Vec3::Zero()};
    int id{-1};
    std::unordered_map<int, Vec2> obs{};
    
    constexpr Landmark() = default;
    constexpr Landmark(const Vec3& position, int landmark_id) noexcept 
        : p_w{position}, id{landmark_id} {}
    
    // C++20: spaceship operator for default comparisons
    auto operator<=>(const Landmark&) const = default;
    
    [[nodiscard]] constexpr bool is_valid() const noexcept { return id >= 0; }
    [[nodiscard]] constexpr size_t observation_count() const noexcept { return obs.size(); }
};

// Single 2D observation
struct Observation {
    int camera_index{-1};
    int point_index{-1};
    double x{0.0};
    double y{0.0};
    
    constexpr Observation() = default;
    constexpr Observation(int cam_idx, int pt_idx, double x_coord, double y_coord) noexcept
        : camera_index{cam_idx}, point_index{pt_idx}, x{x_coord}, y{y_coord} {}
    
    // C++20: spaceship operator for default comparisons
    auto operator<=>(const Observation&) const = default;
    
    [[nodiscard]] constexpr bool is_valid() const noexcept { 
        return camera_index >= 0 && point_index >= 0; 
    }
    
    [[nodiscard]] constexpr Vec2 pixel_coordinates() const noexcept { 
        return Vec2{x, y}; 
    }
};

// Final extracted dataset
struct BALData {
    std::vector<Observation> observations{};
    std::vector<std::array<double, CAMERA_PARAM_SIZE>> camera_params{};
    std::vector<std::array<double, POINT_3D_SIZE>> points{};
    std::unordered_map<int, std::set<int>> camera_to_observations_map{};
    
    constexpr BALData() = default;
    
    // C++20: spaceship operator for default comparisons
    auto operator<=>(const BALData&) const = default;
    
    // Basic accessors
    [[nodiscard]] constexpr size_t num_observations() const noexcept { return observations.size(); }
    [[nodiscard]] constexpr size_t num_cameras() const noexcept { return camera_params.size(); }
    [[nodiscard]] constexpr size_t num_points() const noexcept { return points.size(); }
    [[nodiscard]] constexpr bool empty() const noexcept { 
        return observations.empty() && camera_params.empty() && points.empty(); 
    }
    
    // C++20: Safe array access with span
    [[nodiscard]] std::span<const double> camera_parameters(size_t camera_idx) const {
        if (camera_idx >= camera_params.size()) {
            return {};
        }
        return std::span{camera_params[camera_idx]};
    }
    
    [[nodiscard]] std::span<const double> point_coordinates(size_t point_idx) const {
        if (point_idx >= points.size()) {
            return {};
        }
        return std::span{points[point_idx]};
    }
};

} // namespace bal_cppcon

#endif // BAL_CPPCON_DATA_MODEL_H