#ifndef BAL_CPPCON_PARAM_READER_H
#define BAL_CPPCON_PARAM_READER_H

#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

namespace bal_cppcon {
    class ParamReader {
        public:
            ParamReader(rclcpp::Node::SharedPtr node)
            {
                node_ = node;
                node_->declare_parameter<std::string>("dataset_path", "");
                node_->declare_parameter<bool>("use_filter", false);
                node_->declare_parameter<double>("depth_threshold", 0.0);
                node_->declare_parameter<double>("scale", 100.0);
                node_->declare_parameter<bool>("normalize_data", true);
                node_->declare_parameter<int>("min_observations_per_landmark", 2);
                
                ReadParams();
            }

            void ReadParams()
            {
                use_filter = node_->get_parameter("use_filter").as_bool();
                dataset_path = node_->get_parameter("dataset_path").as_string();
                depth_threshold = node_->get_parameter("depth_threshold").as_double();
                scale = node_->get_parameter("scale").as_double();
                normalize_data = node_->get_parameter("normalize_data").as_bool();
                min_observations_per_landmark = node_->get_parameter("min_observations_per_landmark").as_int();
            }
            ~ParamReader() = default;

            rclcpp::Node::SharedPtr node_;
            bool use_filter = false;
            std::string dataset_path = "";
            double depth_threshold = 0.0;
            double scale = 100.0;
            bool normalize_data = true;
            int min_observations_per_landmark = 2;

    };
}

#endif // BAL_CPPCON_PARAM_READER_H
