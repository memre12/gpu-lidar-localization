from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare('cuda_icp_localizer').find('cuda_icp_localizer')

    rviz_config_file = PathJoinSubstitution([
        pkg_share, 'config', 'icp_localizer.rviz'
    ])

    # Parameters: config YAML first, CLI overrides on top
    parameters = [
        LaunchConfiguration('config_file'),
        {'use_sim_time': LaunchConfiguration('use_sim_time')},
    ]

    # Only override the map path when the argument is actually set,
    # otherwise keep the value from the config YAML
    target_pcd_file = LaunchConfiguration('target_pcd_file').perform(context)
    if target_pcd_file:
        parameters.append({'target_pcd_file': target_pcd_file})

    return [
        Node(
            package='cuda_icp_localizer',
            executable='icp_localizer',
            name='icp_localizer',
            output='screen',
            parameters=parameters,
            emulate_tty=True,
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ]


def generate_launch_description():
    pkg_share = FindPackageShare('cuda_icp_localizer').find('cuda_icp_localizer')

    config_file = PathJoinSubstitution([
        pkg_share, 'config', 'icp_localizer_params.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=config_file,
            description='Path to config YAML file'
        ),
        DeclareLaunchArgument(
            'target_pcd_file',
            default_value='',
            description='Override the map PCD path from the config file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Launch RViz automatically'
        ),
        OpaqueFunction(function=launch_setup),
    ])
