from distutils.core import setup
ROS_ENABLED = False # how can we define this globally
if ROS_ENABLED:
    from catkin_pkg.python_setup import generate_distutils_setup

    setup_args = generate_distutils_setup(
        packages=["elevation_mapping_cupy", "elevation_mapping_cupy.plugins",], package_dir={"": "script"},
    )

    setup(**setup_args)

else:
    setup(
        name='elevation_mapping_cupy',
        version='1.0',
        packages=['elevation_mapping_cupy', 'elevation_mapping_cupy.plugins'],
        package_dir={'': 'script'},
    )
