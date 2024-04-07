# Jacob Installation Instructions 
TODO: Update instructions to use requirements_no_ros.txt or something similar
```
python -m venv em_cupy_venv
.\em_cupy_venv\Scripts\Activate.ps1
pip install cupy-cuda117
cd elevation_mapping_cupy
pip install -e .
pip install ruamel.yaml
pip install shapely==1.7.1
pip install simple_parsing
pip install scipy
pip install torch --index-url https://download.pytorch.org/whl/cu117
pip install matplotlib
```

## Terminology
See this [docs page](https://leggedrobotics.github.io/elevation_mapping_cupy/usage/semantics.html)
* Layer: A GridMap concept that refers to one of possibly many 2D arrays where each index corresponds to the same location in space in the horizontal plane. 
* Channel: A dual meaning term. First it refers to the data contained within an observation (currenlty image or pointcloud) such as rgb or a class label such as tree or grass. Second, refers to a semantic layer of a GridMap. 
* Fusion: An algorithm for taking observations of spatial and semantic data and combining that data with prior data contained within the map. The fusion algorithm can be specified per channel/layer (there is a 1:1 relationship between channels and layers that is currenlty assumed I think, but it might be easy to extend this to allow for multiple fusion algorithms to be applied to the same channel of input data if different layers existed for each fusion algorithm.)
* Plugin: A software modulde that can be loaded dynamically based on a configuration file to generate a new map layer from existing map layer data. (executed upon update of the map or periodically?) My current understanding is that it is not suitable for modifying the map itself and is just useful for generation of new layers.
* Layer Specs: A dictionary of the fusion algorithms that is used for each semantic channel. Indexed by channel name. get_fusion() assigns this based on looking through the parameters pointcloud_channel_fusions or image_channel_fusions for pointclouds and images respectively.
* The parameter pointcloud_channel_fusions is a dictionary with keys equal to channel names and values equal to the suffix of the fusion algorithm to be used to fuse the channel. For example, pointcloud_channel_fusions = {"rgb": "color"} indicates that for the channel name "rgb" in the pointcloud that the fusion algorithm "pointcloud_color" should be used.
* The parameter image_channel_fusions is used in the same way as pointcloud_channel_fusions, but for image data with the prefix "image_\<fusion algorithm\>"
* The parameter fusion_algorithms is a list of the names of all of the implemented fusion algorithms, e.g. ["pointcloud_color", "image_color"]. 
* The parameter additional_layers appears to be a convenience parameter that is only used outside of the mapping classes. It is a list of the layers in addition to x, y, and z that are added to the map (sematic map I think) outside of the plugins. These should correspond to the channel names in the image_channel_fusions and pointcloud_channel_fusions.
* Point Cloud vs. Image: Different input datatypes that use different fusion algorithms even though they produce changes on the same layers potentially e.g. rgb observations could come from a point cloud or an image but both modify the same rgb layer. (WARNING: I wonder if using multiple sensors on the same layer the system should enforce use of the same fusion algorithm. the fusion for Image and pointcloud may be different classes, but both should be bayesian if doing bayesian estiamtion for examle.)
* rgb vs. Color: There seems to have been a change in terminology at somepoint within the package. rgb now is used to refer to a layer name where the suffix _color is used to define fusion algorithms and kernels.
* PCL rgb: the rgb field in a PCL pointcloud is a float32, but contains 3 uint 8 values. See [PCL Docs](https://pointclouds.org/documentation/structpcl_1_1_point_x_y_z_r_g_b.html)
* Map Size: Set by resolution and map_length. Map is assumed to be square which is wasteful for my needs


## TODO
* Allow for disabling traversability filter layer
* Allow for rectangualr map instead of just square
* Add checks for invalid rotation matrix?
* Figure out why they flip the maps when accessing a layer
* Also what is the buffer of 1 cell around each edge for?
* Review Soil Mass Sensor dimensions in Vortex and placement wrt blade
* Finish Lidar sensor addition and figure out transforms

