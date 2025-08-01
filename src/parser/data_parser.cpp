#include <parser/data_parser.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>

namespace bal_cppcon {

// DataParser implementation

BALData DataParser::extract_data() const noexcept {
    BALData data{};
    
    // Extract observations using ranges
    for (const auto& landmark : landmarks_) {
        for (const auto& [cam_idx, obs] : landmark.obs) {
            data.observations.emplace_back(cam_idx, landmark.id, obs.x(), obs.y());
            data.camera_to_observations_map[cam_idx].insert(landmark.id);
        }
    }
    
    // Extract 3D points using ranges
    auto point_transformer = [](const auto& landmark) {
        return std::array<double, 3>{landmark.p_w.x(), landmark.p_w.y(), landmark.p_w.z()};
    };
    
    std::ranges::transform(landmarks_, std::back_inserter(data.points), point_transformer);
    
    // Extract camera parameters using ranges
    auto camera_transformer = [this](const auto& camera) {
        auto rotation = camera.T_c_w.so3().log();
        auto quat = convert_camera_rotation(rotation);
        auto translation = camera.T_c_w.translation();
        auto intrinsics = camera.intrinsics.as_vec3();
        
        return std::array<double, 10>{
            quat.w(), quat.x(), quat.y(), quat.z(),
            translation(0), translation(1), translation(2),
            intrinsics(0), intrinsics(1), intrinsics(2)
        };
    };
    data.cameras = cameras_;
    std::ranges::transform(cameras_, std::back_inserter(data.camera_params), camera_transformer);
    
    return data;
}

void DataParser::load_dataset() {
    if (!std::filesystem::exists(dataset_path_)) {
        throw std::runtime_error("Dataset file not found: " + dataset_path_.string() + " - " +
                                std::string(parse_error_message(ParseError::FileNotFound)));
    }
    
    // Validate file format
    auto extension = dataset_path_.extension();
    if (extension != ".txt" && extension != ".bal") {
        throw std::runtime_error("Unsupported dataset format: " + extension.string() + " - " +
                                std::string(parse_error_message(ParseError::InvalidFormat)));
    }
    
    try {
        // Load the actual data
        load_bal_format();
        
        
        if (config_.perform_filtering) {
            filter_observations();
        }
        
        // Apply post-processing based on configuration
        std::cout << "Normalize data: " << config_.normalize_data << std::endl;
        if (config_.normalize_data) {
            std::cout << "Normalizing data" << std::endl;
            normalize();
        }
        
        // Validate that we have sufficient data after processing
        if (cameras_.empty() || landmarks_.empty()) {
            throw std::runtime_error("Insufficient data after processing: " +
                                    std::string(parse_error_message(ParseError::InsufficientData)));
        }
        
    } catch (const std::exception& e) {
        // Clear any partially loaded data on error
        cameras_.clear();
        landmarks_.clear();
        throw; // Re-throw the exception
    }
}

void DataParser::normalize() noexcept {
    if (landmarks_.empty()) return;
    
    // Compute median using C++20 ranges
    Vec3 median{Vec3::Zero()};
    std::vector<Scalar> coords(landmarks_.size());
    
    for (int dim = 0; dim < 3; ++dim) {
        auto coordinate_extractor = [dim](const auto& landmark) { 
            return landmark.p_w(dim); 
        };
        
        std::ranges::transform(landmarks_, coords.begin(), coordinate_extractor);
        median(dim) = compute_median(std::span{coords});
    }
    
    // Compute scale factor using ranges
    auto deviation_calculator = [&median](const auto& landmark) {
        return (landmark.p_w - median).template lpNorm<1>();
    };
    
    std::ranges::transform(landmarks_, coords.begin(), deviation_calculator);
    Scalar median_dev = compute_median(std::span{coords});
    Scalar scale_factor = config_.scale / median_dev;
    
    // Apply normalization using ranges
    auto landmark_normalizer = [scale_factor, &median](auto& landmark) {
        landmark.p_w = scale_factor * (landmark.p_w - median);
    };
    
    std::ranges::for_each(landmarks_, landmark_normalizer);
    
    // Normalize camera poses
    auto camera_normalizer = [scale_factor, &median](auto& camera) {
        SE3 T_w_c = camera.T_c_w.inverse();
        T_w_c.translation() = scale_factor * (T_w_c.translation() - median);
        camera.T_c_w = T_w_c.inverse();
    };
    
    std::ranges::for_each(cameras_, camera_normalizer);
}

void DataParser::filter_observations() noexcept {
    if (config_.depth_threshold <= 0.0) return;
    
    // Filter observations for each landmark
    auto observation_filter = [this](auto& landmark) {
        std::unordered_map<int, Vec2> valid_obs;
        
        for (const auto& [cam_idx, obs] : landmark.obs) {
            if (cam_idx < 0 || cam_idx >= static_cast<int>(cameras_.size())) {
                continue; // Skip invalid camera index
            }
            
            const auto& camera = cameras_[cam_idx];
            Vec3 p_cam = camera.T_c_w * landmark.p_w;
            
            if (is_valid_observation(p_cam, config_.depth_threshold)) {
                valid_obs.emplace(cam_idx, obs);
            }
        }
        
        landmark.obs = std::move(valid_obs);
    };
    
    std::ranges::for_each(landmarks_, observation_filter);
    
    // Remove landmarks with insufficient observations using ranges
    auto insufficient_observations = [min_obs = config_.min_observations_per_landmark](const auto& landmark) {
        return landmark.observation_count() < min_obs;
    };
    
    landmarks_.erase(
        std::ranges::remove_if(landmarks_, insufficient_observations).begin(),
        landmarks_.end()
    );
}

void DataParser::load_bal_format() noexcept {
    std::ifstream file(dataset_path_);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + dataset_path_.string());
    }

    try {
        const SO3 axis_inversion = SO3(Vec3(1, -1, -1).asDiagonal());

        int num_cams = 0, num_lms = 0, num_obs = 0;
        if (!(file >> num_cams >> num_lms >> num_obs)) {
            throw std::runtime_error("Failed to read header");
        }

        cameras_.resize(num_cams);
        landmarks_.resize(num_lms);

        // Initialize landmark IDs
        for (int i = 0; i < num_lms; ++i) {
            landmarks_[i].id = i;
        }

        // Read observations
        for (int i = 0; i < num_obs; ++i) {
            int cam_idx = 0, lm_idx = 0;
            Vec2 pos;
            if (!(file >> cam_idx >> lm_idx >> pos.x() >> pos.y())) {
                throw std::runtime_error("Failed to read observation " + std::to_string(i));
            }

            pos.y() = -pos.y(); // Flip y-axis

            if (lm_idx >= 0 && lm_idx < num_lms) {
                landmarks_[lm_idx].obs.emplace(cam_idx, pos);
            }
        }

        // Read camera parameters
        for (int i = 0; i < num_cams; ++i) {
            Vec9 params;
            for (int j = 0; j < 9; ++j) {
                if (!(file >> params(j))) {
                    throw std::runtime_error("Failed to read camera parameters for camera " + std::to_string(i));
                }
            }
            cameras_[i].id = i;
            cameras_[i].T_c_w.so3() = axis_inversion * SO3::exp(params.head<3>());
            cameras_[i].T_c_w.translation() = axis_inversion * params.segment<3>(3);
            cameras_[i].intrinsics = CameraModel{params(6), params(7), params(8)};
        }

        // Read landmark positions
        for (int i = 0; i < num_lms; ++i) {
            if (!(file >> landmarks_[i].p_w.x() >> landmarks_[i].p_w.y() >> landmarks_[i].p_w.z())) {
                throw std::runtime_error("Failed to read landmark position for landmark " + std::to_string(i));
            }
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse BAL file: " + std::string(e.what()));
    }
}


Scalar DataParser::compute_median(std::span<Scalar> data) const noexcept {
    if (data.empty()) return 0.0;
    
    std::vector<Scalar> temp_data(data.begin(), data.end());
    size_t n = temp_data.size() / 2;
    std::nth_element(temp_data.begin(), temp_data.begin() + n, temp_data.end());
    return temp_data[n];
}

constexpr Eigen::Quaterniond DataParser::angle_axis_to_quaternion(const Eigen::Vector3d& angle_axis) const noexcept {
    double angle = angle_axis.norm();
    
    // For very small angles, avoid numerical instability
    if (angle < 1e-10) {
        return Eigen::Quaterniond::Identity();
    }
    
    Eigen::Vector3d axis = angle_axis / angle;
    return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
}

constexpr Eigen::Quaterniond DataParser::convert_camera_rotation(const Eigen::Vector3d& rotation) const noexcept {
    return angle_axis_to_quaternion(rotation);
}

/* 
Example usage of the create_parser factory function:

// Method 1: Using create_parser factory function (recommended for external use)
auto parser_result = bal_23::create_parser("dataset.bal", bal_23::ParserConfig{
    .perform_filtering = true,
    .depth_threshold = 10.0,
    .scale = 50.0,
    .normalize_data = true,
    .min_observations_per_landmark = 3
});

if (parser_result) {
    auto parser = std::move(*parser_result);
    parser->load_dataset();  // This will use the improved validation
    
    auto data = parser->extract_data();
    std::cout << "Loaded " << data.camera_params.size() << " cameras and " 
              << data.points.size() << " points" << std::endl;
} else {
    std::cout << "Failed to create parser: " 
              << parse_error_message(parser_result.error()) << std::endl;
}

// Method 2: Direct construction (when you know the path is valid)
bal_23::DataParser parser{"dataset.bal", bal_23::ParserConfig{
    .perform_filtering = true,
    .depth_threshold = 5.0
}};
parser.load_dataset();  // Uses improved validation with create_parser-style error messages

// Method 3: Using filesystem::path
std::filesystem::path dataset_path = "path/to/dataset.bal";
auto parser_result2 = bal_23::create_parser(dataset_path, bal_23::ParserConfig{
    .normalize_data = false
});
*/

} // namespace bal_cppcon       