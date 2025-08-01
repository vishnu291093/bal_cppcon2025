cmake_minimum_required(VERSION 3.14)
project(bundle_adjustment_at_large_cppcon)

# -----------------------------------------------------------------------------
# C++ Standard
# -----------------------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 23)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
include(FetchContent)

# Fetch Sophus (without ceres + tests)
set(SOPHUS_ENABLE_CERES OFF CACHE BOOL "Disable Sophus Ceres Manifold support" FORCE)
set(SOPHUS_ENABLE_TESTS OFF CACHE BOOL "Disable Sophus tests" FORCE)
FetchContent_Declare(
  Sophus
  GIT_REPOSITORY https://github.com/strasdat/Sophus.git
  GIT_TAG main
)
FetchContent_MakeAvailable(Sophus)

# ROS 2 and external dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(TBB REQUIRED)
find_package(Boost REQUIRED)

pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# -----------------------------------------------------------------------------
# Dataset Parser Library
# -----------------------------------------------------------------------------
add_library(data_parser
  src/parser/data_parser.cpp
)
target_include_directories(data_parser PUBLIC
  include
  ${sophus_SOURCE_DIR}
)
target_link_libraries(data_parser
  Sophus::Sophus
  Eigen3::Eigen
  glog
)
ament_target_dependencies(data_parser
  rclcpp
  Eigen3
)

# -----------------------------------------------------------------------------
# Dataset Visualizer Library
# -----------------------------------------------------------------------------
add_library(dataset_visualizer
  src/visualize_dataset.cpp
)
target_include_directories(dataset_visualizer PUBLIC
  include
  ${sophus_SOURCE_DIR}
)
target_link_libraries(dataset_visualizer
  data_parser
  Sophus::Sophus
)
ament_target_dependencies(dataset_visualizer
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  Ceres
)

# -----------------------------------------------------------------------------
# Hybrid Pipeline Library
# -----------------------------------------------------------------------------
add_library(hybrid_pipeline
  src/hybrid_pipeline.cpp
  src/utils/data_stats.cpp
)
target_include_directories(hybrid_pipeline PUBLIC
  include
  ${sophus_SOURCE_DIR}
  ${YAML_CPP_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(hybrid_pipeline
  data_parser
  Sophus::Sophus
  ${YAML_CPP_LIBRARIES}
  ${catkin_LIBRARIES}
  TBB::tbb
)


ament_target_dependencies(hybrid_pipeline
  rclcpp
  Eigen3
)

# -----------------------------------------------------------------------------
# Main Executable
# -----------------------------------------------------------------------------
add_executable(main_visualize_dataset
  src/main_visualize_dataset.cpp
)
target_link_libraries(main_visualize_dataset
  data_parser
  dataset_visualizer
)
ament_target_dependencies(main_visualize_dataset
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
)

# -----------------------------------------------------------------------------
# BAL Optimization Node Executable
# -----------------------------------------------------------------------------
add_executable(main_bal_optimization_node
  src/main_bal_optimization_node.cpp
)
target_link_libraries(main_bal_optimization_node
  data_parser
  dataset_visualizer
  hybrid_pipeline
)
ament_target_dependencies(main_bal_optimization_node
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
)

# -----------------------------------------------------------------------------
# Install Targets
# -----------------------------------------------------------------------------
install(TARGETS
  main_visualize_dataset
  main_bal_optimization_node
  data_parser
  dataset_visualizer
  hybrid_pipeline
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

# -----------------------------------------------------------------------------
ament_package()
