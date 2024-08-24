# Overview
This is a working document that serves as guide to understanding the functionality of the `elevation_mapping_cupy` package. It includes detailed explanations of the package's operations, addresses any open questions, supplements missing documentation, and suggests potential improvements.

## Jacob Installation Instructions 
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

To get profiling working:
* In NVIDIA Control Panel Desktop->Enable Developer Settings->Manage GPU Performance Counters->Allow Access to the GPU performance counters to all users
* Open Nsight Systems as admin
* Select computer as target
* Use command line argument:
```c:/Users/RDCERWJW/Documents/REO/code/soil-property-estimation/soil_venv/Scripts/python.exe c:/Users/RDCERWJW/Documents/REO/code/soil-property-estimation/elevation_mapping_cupy/elevation_mapping_cupy/script/elevation_mapping_cupy/sensor_processor.py```
* Use working directory
```c:/Users/RDCERWJW/Documents/REO/code/soil-property-estimation/```
* Navigate down to green section in timeline (Threads (x)->PID->NVTX) to look at function logs or select Stats System View->NVTX GPU Projection Summary

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
* Height Drift Compensation: The parameters position_noise_thresh and orientation_noise_thresh are used to determine if the height drift compensation should be applied for a given measurement or not. If the noise from the measurement is larger than either of the thresholds then it is applied as long as enable_height_drift_compensation is true, at least min_height_drift_cnt points from the observation are used to compute the height drift error, and the computed mean_error is less than the specified max_drift parameter. The drift compensation is applied based on the formula h += mean_drift_error * drift_compensation_alpha
* The map_util transform_p() is used to transform a point represented in the sensor frame to a point in the world/map frame. 
* I'm not sure I understand why, but the cell_n includes a border of 1 cell around the entire boundary of the map that is never indexed from what I can tell. That means that real data starts at index 1 and ends at index cell_n-2. This is why you see the map[:,1:-1] syntax everywhere
* Map frame is not related to the map center location. That is the position of the elevation map within the Map frame.

## TODO
* Allow for disabling traversability filter layer. Would have to modify the height drift compensation as well as it uses travesability to mask out points used in the calculation.
* Allow for rectangualr map instead of just square
* Add checks for invalid rotation matrix?
* Review Soil Mass Sensor dimensions in Vortex and placement wrt blade
* Finish Lidar sensor addition and figure out transforms
* Add ENUMS or something to accessing layers in CUDA code to make more obvious what layers are being used at a given point.
* Modification of layers in ElevationMap class would require re-coding because the cuda models index the layers directly assuming a fixed ordering
* Consider modifying height drift compensation to modify the height of the observation instead of moving the map.
* In update_map_with_kernel() the height drift compensation is only applied to the elevation layer and not to the upper bound layer. This should likely be applied to both.
* Clean up indexing of points, e.g. see add_points_kernel() rx, ry, rz indexing
* In python ros node the variance due to change in time is only updated with a fixed value of time_variance whenever the timer based callback is called. The rate is set by another parameter update_variance_fps. A better way of parameterizing this would be to set the rate at which you want to have the variance evolve and then set a timer for how often this update should be performed. Then at each call the amount of temporal variance to add could be calcualted based on a timestamp differential.
* Deal with datatype on trimesh code
* Rethink the use of center for transforming coordinates. May not be necessary if our transforms are already wrt map frame not world frame...
* Rework code to make better use of **kwargs to clean up number of arguments and ordering issues

Next Steps:
* Clean up functions and add note about how intersections could be a problem for us if blade is rotated in place
* Propose method to deal with this by splitting model into separate surfaces that we extrude seprately.
* Does it matter if we solve this? If in generated mesh is still generating surfaces that are correct, but normals that are wrong do we care? Maybe still just be able to ray trace to it. Then maybe we can use the normal of that plane where it intersects to tell us which direction to move the soil?
* I think it does matter because you still have surfaces that are thin that are the result of improper
* If dealing with polygons could use plane mesh intersection and try to build a mesh mesh intersection out of it...
* Or post process existing mesh using something like [this](https://github.com/mikedh/trimesh/issues/895)
* Test out extruding a more complex blade

* Figure out this whole transform issue where initial mesh is translated and rotated...
* The side of the starting surface that a cell falling within the intersected volume is determines the side of the ending surface that the material should be moved to. if on the positive side of start surface and intersected then will end up on the positive side of the second surface (assuming normals are not flipped to show outward normal for display of mesh)


## Questions
* The z noise function that is used to determine the appropriate level of noise for a point seems wrong. Why would it have the error be related to the z axis coordinate of the point cloud. It should be based on the range if anything. used in error_counting_kernel and add_points_kernel
* Also don't understand is_valid function? What is dxy used for. Does this assume the map is moved to the current robot position?
* For wall detection and counting of points within a cell, is that tracked over time or only within a given scan?
* Figure out why they flip the maps when accessing a layer
* Also what is the buffer of 1 cell around each edge for?
* What is the is_valid() doing?


## Notes
* Layer names and indicies in ElevationMap class: `self.layer_names = ["elevation": 0, "variance": 1, "is_valid": 2, "traversability": 3, "time": 4, "upper_bound": 5, "is_upper_bound": 6]` 
* FYI the ROS wrapper does move the elevation map periodically to be at the location of the vehicle using the move_to function. I don't think not using that breaks anything but maybe the robot_centric_elevation_map plugin which uses the base_rotation variable. This could be provided via other means though.
* The map center is tracked seprately and is not necessarily coincident with the map frame. If move_to() or move() is called this shifts the entire map to that location while ensuring that the data remains at the same coordinates in the map frame. It just changes coordinates relative to the map center frame. This is performed with a roll function and unknown regions are intialized to the initial variance and an invalid elevation state. 
* When data is provided via a pointcloud with a transform from map frame to the sensor this translation (not rotation) is shifted so that it represents the translation of the sensor frame with respect to the map center. 
* The error_counting_kernel is applied next and calculates the error between the point cloud and the existing map. This is used to implement the height drift compensation as is described in section II.D of [1]. The main output of this kernel is the error, error_cnt and newmap which has traversability and time layers assigned some values.
* R and t provided to input_pointcloud() and subsequently update_map_with_kernel() are the rotation matrix and translation vector representing $T_{ws}(R,t)$ i.e. the transform from the world frame to the sensor frame. In update_map_with_kernel() t is modifed using the function shift_translation_to_map_center(). What this effectively does is make it so that the old R and the new t represent the transform $T_{ms}=T_{mw} T_{ws} = T_{wm}^{-1}T_{ws}$ where $T_{wm}$ is the transfrom from the world to map frame cosisting purely of a translation. The rotation component of $T_{ws}$ is equavlent to the rotation component of $T_{ms}$ because the map frame $\{m\}$ is aligned with the world frame $\{w\}$ by construction. The point coordinates in the map frame can therefore be provided by $p_m = T_{ms} p_s$ which is implemneted in the add_points_kernel().
* The time layer doesn't get used for time based variance but is instead only used in the visibility cleanup step
* In original elevation_mapping package is this [product of transforms](https://github.com/ANYbotics/elevation_mapping/blob/82aa8a566a62e9cb9c4013c13b6fa3591ce755ec/elevation_mapping/src/sensor_processors/StructuredLightSensorProcessor.cpp#L58) an error? I don't think so, just different syntax. look at how CBM is defined.
* The new_layers in the semantic_map is being used to store the uncertainty for bayesian fusion. This seems somewhat hacky. It would be better to have a separate layer in the semantic map for the uncertainty


## References

### Paper [1]
If you use the Elevation Mapping CuPy, please cite the following paper:
Elevation Mapping for Locomotion and Navigation using GPU

[Elevation Mapping for Locomotion and Navigation using GPU](https://arxiv.org/abs/2204.12876)

Takahiro Miki, Lorenz Wellhausen, Ruben Grandia, Fabian Jenelten, Timon Homberger, Marco Hutter  

```bibtex
@inproceedings{miki2022elevation,
  title={Elevation mapping for locomotion and navigation using gpu},
  author={Miki, Takahiro and Wellhausen, Lorenz and Grandia, Ruben and Jenelten, Fabian and Homberger, Timon and Hutter, Marco},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2273--2280},
  year={2022},
  organization={IEEE}
}
```
### Paper [2]
If you use the Multi-modal Elevation Mapping for color or semantic layers, please cite the following paper:

[MEM: Multi-Modal Elevation Mapping for Robotics and Learning](https://arxiv.org/abs/2309.16818v1)

Gian Erni, Jonas Frey, Takahiro Miki, Matias Mattamala, Marco Hutter

```bibtex
@inproceedings{erni2023mem,
  title={MEM: Multi-Modal Elevation Mapping for Robotics and Learning},
  author={Erni, Gian and Frey, Jonas and Miki, Takahiro and Mattamala, Matias and Hutter, Marco},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={11011--11018},
  year={2023},
  organization={IEEE}
}
```