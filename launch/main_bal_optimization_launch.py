# launch/bal_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('bundle_adjustment_at_large_cppcon'),
        'config',
        'bal_optimization_params.yaml')

    return LaunchDescription([
        Node(
            package='bundle_adjustment_at_large_cppcon',
            executable='main_bal_optimization_node',
            name='bal_optimization_node',
            parameters=[config_path],
            output='screen'
        )
    ])
