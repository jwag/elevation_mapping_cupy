from setuptools import setup, find_packages
from glob import glob
import os
ROS_ENABLED = False # how can we define this globally
SOIL_MAP_ENABLED = False # can we define this globally

package_name = 'elevation_mapping_cupy'
if ROS_ENABLED:
    setup(
        name=package_name,
        version='2.0.0',
        packages=find_packages(include=[package_name, f'{package_name}.*']),
        install_requires=['setuptools'],
        zip_safe=True,
        author='Your Name',
        author_email='your.email@example.com',
        maintainer='Your Name',
        maintainer_email='your.email@example.com',
        description='Elevation mapping on GPU',
        license='MIT',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'elevation_mapping_node.py = '+package_name+'.elevation_mapping_node:main',
            ],
        },
        data_files=[
            ('share/ament_index/resource_index/packages',['resource/' + package_name]),
            (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
            *[(os.path.join('share', package_name, os.path.dirname(yaml_file)), [yaml_file]) for yaml_file in glob('config/**/*.yaml', recursive=True)],
            # also the .*dat files
            *[(os.path.join('share', package_name, os.path.dirname(dat_file)), [dat_file]) for dat_file in glob('config/**/*.dat', recursive=True)],
            # add rviz files
            *[(os.path.join('share', package_name, os.path.dirname(rviz_file)), [rviz_file]) for rviz_file in glob('rviz/**/*.rviz', recursive=True)],
            (os.path.join('share', package_name), ['package.xml']),
        ],
    )

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
        name=package_name,
        version='1.0',
        packages=find_packages(include=[package_name, f'{package_name}.*']),
        # package_dir={'': package_name},
        install_requires=install_requires,
        data_files=[
            *[(os.path.join('build', package_name, os.path.dirname(yaml_file)), [yaml_file]) for yaml_file in glob('config/**/*.yaml', recursive=True)],
            # also the .*dat files
            *[(os.path.join('build', package_name, os.path.dirname(dat_file)), [dat_file]) for dat_file in glob('config/**/*.dat', recursive=True)],
        ]
    )
