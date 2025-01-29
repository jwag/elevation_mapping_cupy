from distutils.core import setup
ROS_ENABLED = False # how can we define this globally
SOIL_MAP_ENABLED = False # how can we define this globally
if ROS_ENABLED:
    from catkin_pkg.python_setup import generate_distutils_setup

    setup_args = generate_distutils_setup(
        packages=["elevation_mapping_cupy", "elevation_mapping_cupy.plugins",], package_dir={"": "script"},
    )

    setup(**setup_args)

else:
    install_requires = ['numpy', 'cupy-cuda12x', 'scipy', 'matplotlib', 'ruamel.yaml',
                        'shapely==1.7.1', 'simple_parsing', 'trimesh[easy]', 'embreex', 'pandas', 'gitpython']
    if SOIL_MAP_ENABLED:
        install_requires.append('torch @ https://download.pytorch.org/whl/cu124')
        install_requires.append('lightning')
        install_requires.append('tensorboard')
    setup(
        name='elevation_mapping_cupy',
        version='1.0',
        packages=['elevation_mapping_cupy', 'elevation_mapping_cupy.plugins'],
        package_dir={'': 'script'},
        install_requires=install_requires,
    )
