#include <rclcpp/rclcpp.hpp>
#include <visualize_dataset.h>
#include <param_reader.h>
#include <parser/data_parser.h>

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("bal_visualizer_node");

  bal_cppcon::ParamReader param_reader(node);

  if (param_reader.dataset_path.empty()) {
    RCLCPP_ERROR(node->get_logger(), "Parameter 'dataset_path' is not set.");
    return 1;
  }


  bal_cppcon::DataParser dataset_processor(param_reader.dataset_path, bal_cppcon::ParserConfig{
    param_reader.use_filter,
    param_reader.depth_threshold,
    param_reader.scale,
    param_reader.normalize_data,
    param_reader.min_observations_per_landmark
  }); 
  dataset_processor.load_dataset();
  bal_cppcon::BALData data = dataset_processor.extract_data();


  RCLCPP_INFO(node->get_logger(), "Parsed dataset with %d cameras, %d points, and %d observations.",
              int(data.camera_params.size()), int(data.points.size()), int(data.observations.size()));

  // Create publishers for points and cameras
  bal_cppcon::DatasetVisualizer visualizer(node, "original_camera_markers", "original_point_markers");


  // Publish the dataset markers
  rclcpp::Rate rate(1.0);
  while (rclcpp::ok()) {
    visualizer.publishPoints(data.points, 1.0, 0.0, 0.0);
    visualizer.publishCameras(data.camera_params, 0.0, 1.0, 0.0);
    rclcpp::spin_some(node);
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
