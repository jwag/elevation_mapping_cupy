from setuptools import setup
ROS_ENABLED = False # how can we define this globally
SOIL_MAP_ENABLED = True # can we define this globally
if ROS_ENABLED:
    from catkin_pkg.python_setup import generate_distutils_setup

    setup_args = generate_distutils_setup(
        packages=["elevation_mapping_cupy", "elevation_mapping_cupy.plugins",], package_dir={"": "script"},
    )

    setup(**setup_args)

else:
    install_requires = ['numpy', 'cupy-cuda12x', 'scipy', 'matplotlib', 'ruamel.yaml', 'opencv-python',
                        'shapely', 'simple_parsing', 'trimesh[easy]', 'embreex', 'pandas', 'gitpython']
    if SOIL_MAP_ENABLED:
        # This was the latest version of pytorch that is compatible with cuda 12.4 and python 3.8 that i could find
        # Unlike with calling pip directly, specifying the --index-url https://download.pytorch.org/whl/cu124
        # doesn't work and a specific wheel has to be specified I think
        # Alternative pip install torch --index-url https://download.pytorch.org/whl/cu124
        # install_requires.append('torch @ https://download.pytorch.org/whl/cu124/torch/torch-2.4.1+cu124-cp38-cp38-win_amd64.whl')
        # install_requires.append('torch @ https://download.pytorch.org/whl/cu124/torch/torch-2.4.1%Bcu124-cp38-cp38-win_amd64.whl')
        # install_requires.append('lightning')
        # install_requires.append('tensorboard')
        debug=1
    setup(
        name='elevation_mapping_cupy',
        version='1.0',
        packages=['elevation_mapping_cupy', 'elevation_mapping_cupy.plugins'],
        package_dir={'': 'script'},
        install_requires=install_requires,
    )
