import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    # You can set your API key as an environment variable and the launch file will use it
    # export GOOGLE_API_KEY="your_secret_key_here"
    google_api_key_arg = DeclareLaunchArgument(
        'google_api_key',
        default_value=os.environ.get('GOOGLE_API_KEY', ''),
        description='Your Google API key for the Gemini model'
    )

    return LaunchDescription([
        google_api_key_arg,
        
        Node(
            package='object_detection_pkg',
            executable='gemini_node',
            name='gemini_detector_node',
            output='screen',
            emulate_tty=True, # To see logger messages
            parameters=[{
                'google_api_key': LaunchConfiguration('google_api_key')
            }]
        ),
    ])
