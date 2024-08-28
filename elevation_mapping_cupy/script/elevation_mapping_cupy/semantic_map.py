#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from elevation_mapping_cupy.parameter import Parameter
import cupy as cp
import numpy as np
from typing import List, Dict
import re


from elevation_mapping_cupy.fusion.fusion_manager import FusionManager

xp = cp


class SemanticMap:
    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """

        self.param = param

        self.layer_specs_points = {}
        self.layer_specs_image = {}
        self.layer_specs_GET = {}
        self.layer_names = []
        self.unique_fusion = []
        self.unique_data = []
        self.elements_to_shift = {}

        self.unique_fusion = self.param.fusion_algorithms

        self.amount_layer_names = len(self.layer_names)

        self.semantic_map = xp.zeros(
            (self.amount_layer_names, self.param.cell_n, self.param.cell_n), dtype=param.data_type,
        )
        self.new_map = xp.zeros((self.amount_layer_names, self.param.cell_n, self.param.cell_n), param.data_type,)
        # which layers should be reset to zero at each update, per default everyone,
        # if a layer should not be reset, it is defined in compile_kernels function
        self.delete_new_layers = cp.ones(self.new_map.shape[0], cp.bool8)
        self.var_layer_names = [] # List of layer names for which self.new_map is the variance for that layer 
        self.fusion_manager = FusionManager(self.param)

    def clear(self):
        """Clear the semantic map."""
        self.semantic_map *= 0.0

    def initialize_fusion(self):
        """Initialize the fusion algorithms."""
        for fusion in self.unique_fusion:
            # if "pointcloud_class_bayesian" == fusion:
            #     pcl_ids = self.get_layer_indices("class_bayesian", self.layer_specs_points)
            #     self.delete_new_layers[pcl_ids] = 0
            # elif "pointcloud_class_max" == fusion:
            #     pcl_ids = self.get_layer_indices("class_max", self.layer_specs_points)
            #     self.delete_new_layers[pcl_ids] = 0
            #     layer_cnt = self.param.fusion_algorithms.count("class_max")
            #     id_max = cp.zeros((layer_cnt, self.param.cell_n, self.param.cell_n), dtype=cp.uint32,)
            #     self.elements_to_shift["id_max"] = id_max
            # elif "GET_latest" == fusion:
            #     GET_ids = self.get_layer_indices("latest", self.layer_specs_GET)
            #     # self.delete_new_layers[GET_ids] = 0 # for latest we want to reset the new_map array to 0 so keep commented
            # elif "GET_bayesian_inference" == fusion:
            #     GET_ids = self.get_layer_indices("bayesian_inference", self.layer_specs_GET)
            #     self.delete_new_layers[GET_ids] = 0
            self.fusion_manager.register_plugin(fusion)

    def update_fusion_setting(self, name):
        """
        Update the fusion settings.
        """
        for fusion in self.unique_fusion:
            if "pointcloud_class_bayesian" == fusion and name in self.layer_specs_points:
                pcl_ids = self.get_layer_indices("class_bayesian", self.layer_specs_points)
                if name in pcl_ids:
                    # The new_layers is not being deleted because it is being used to store the uncertainty
                    # This seems somewhat hacky. It would be better to have a separate layer in the semantic map for the uncertainty
                    self.delete_new_layers[pcl_ids[name]] = 0
                    self.var_layer_names.append(name+ "_var")
            elif "pointcloud_class_max" == fusion and name in self.layer_specs_points:
                pcl_ids = self.get_layer_indices("class_max", self.layer_specs_points)
                if name in pcl_ids:
                    self.delete_new_layers[pcl_ids[name]] = 0
                    # self.var_layer_names.append(name+ "_var") # Don't know how to handle this yet
                    layer_cnt = self.param.fusion_algorithms.count("class_max")
                    id_max = cp.zeros((layer_cnt, self.param.cell_n, self.param.cell_n), dtype=cp.uint32,)
                self.elements_to_shift["id_max"] = id_max
            elif "GET_latest" == fusion and name in self.layer_specs_GET:
                GET_ids = self.get_layer_indices("latest", self.layer_specs_GET)
                # if name in GET_ids:
                    # self.delete_new_layers[GET_ids[name]] = 0 # for latest we want to reset the new_map array to 0 so keep commented
            elif "GET_bayesian_inference" == fusion and name in self.layer_specs_GET:
                GET_ids = self.get_layer_indices("bayesian_inference", self.layer_specs_GET)
                if name in GET_ids:
                    self.delete_new_layers[GET_ids[name]] = 0
                    self.var_layer_names.append(name+ "_var")

    def add_layer(self, name):
        """
        Add a new layer to the semantic map.

        Args:
            name (str): The name of the new layer.
        """
        if name not in self.layer_names:
            self.layer_names.append(name)
            self.semantic_map = cp.append(
                self.semantic_map,
                cp.zeros((1, self.param.cell_n, self.param.cell_n), dtype=self.param.data_type),
                axis=0,
            )
            self.new_map = cp.append(
                self.new_map, cp.zeros((1, self.param.cell_n, self.param.cell_n), dtype=self.param.data_type), axis=0,
            )
            self.delete_new_layers = cp.append(self.delete_new_layers, cp.array([1], dtype=cp.bool8))
            self.update_fusion_setting(name)

    def pad_value(self, x, shift_value, idx=None, value=0.0):
        """Create a padding of the map along x,y-axis according to amount that has shifted.

        Args:
            x (cupy._core.core.ndarray):
            shift_value (cupy._core.core.ndarray):
            idx (Union[None, int, None, None]):
            value (float):
        """
        if idx is None:
            if shift_value[0] > 0:
                x[:, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[:, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[:, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[:, :, shift_value[1] :] = value
        else:
            if shift_value[0] > 0:
                x[idx, : shift_value[0], :] = value
            elif shift_value[0] < 0:
                x[idx, shift_value[0] :, :] = value
            if shift_value[1] > 0:
                x[idx, :, : shift_value[1]] = value
            elif shift_value[1] < 0:
                x[idx, :, shift_value[1] :] = value

    def shift_map_xy(self, shift_value):
        """Shift the map along x,y-axis according to shift values.

        Args:
            shift_value:
        """
        self.semantic_map = cp.roll(self.semantic_map, shift_value, axis=(1, 2))
        self.pad_value(self.semantic_map, shift_value, value=0.0)
        self.new_map = cp.roll(self.new_map, shift_value, axis=(1, 2))
        self.pad_value(self.new_map, shift_value, value=0.0)
        for el in self.elements_to_shift.values():
            el = cp.roll(el, shift_value, axis=(1, 2))
            self.pad_value(el, shift_value, value=0.0)

    def get_fusion(
        self, channels: List[str], channel_fusions: Dict[str, str], layer_specs: Dict[str, str]
    ) -> List[str]:
        """Get all fusion algorithms that need to be applied to a specific input.
            Modifies layer_specs in place to add the fusion algorith as a value for the channel key.

        Args:
            channels (List[str]):
        """
        fusion_list = []
        process_channels = []
        for channel in channels:
            if channel not in layer_specs:
                # If the channel is not in the layer_specs, we use the default fusion algorithm
                matched_fusion = self.get_matching_fusion(channel, channel_fusions)
                if matched_fusion is None:
                    if "default" in channel_fusions:
                        default_fusion = channel_fusions["default"]
                        print(
                            f"[WARNING] Layer {channel} not found in layer_specs. Using {default_fusion} algorithm as default."
                        )
                        layer_specs[channel] = default_fusion
                    # If there's no default fusion algorithm, we skip this channel
                    else:
                        print(
                            f"[WARNING] Layer {channel} not found in layer_specs ({layer_specs}) and no default fusion is configured. Skipping."
                        )
                        continue
                else:
                    layer_specs[channel] = matched_fusion
            x = layer_specs[channel]
            fusion_list.append(x)
            process_channels.append(channel)
        return process_channels, fusion_list

    def get_matching_fusion(self, channel: str, fusion_algs: Dict[str, str]):
        """ Use regular expression to check if the fusion algorithm matches the channel name."""
        for fusion_alg, alg_value in fusion_algs.items():
            if re.match(f"^{fusion_alg}$", channel):
                return alg_value
        return None

    def get_layer_indices(self, fusion_alg, layer_specs):
        """Get the indices of the layers that are used for a specific fusion algorithm.

        Args:
            fusion_alg(str): fusion algorithm name

        Returns:
            layer_indicies (dict of cp.array): indices of the layers
        """
        layer_indices = {}
        for it, (key, val) in enumerate(layer_specs.items()):
            if val == fusion_alg: #TODO: Double check that this is correct
                layer_indices[key] = cp.array([it], dtype=cp.int32)
        return layer_indices

    def get_indices_fusion(self, pcl_channels: List[str], fusion_alg: str, layer_specs: Dict[str, str]):
        """Computes the indices of the channels of the pointcloud and the layers of the semantic map of type fusion_alg.

        Args:
            pcl_channels (List[str]): list of all channel names
            fusion_alg (str): fusion algorithm type we want to use for channel selection

        Returns:
            Union[Tuple[List[int], List[int]], Tuple[cupy._core.core.ndarray, cupy._core.core.ndarray]]:


        """
        # this contains exactly the fusion alg type for each channel of the pcl
        pcl_val_list = [layer_specs[x] for x in pcl_channels]
        # this contains the indices of the point cloud where we have to perform a certain fusion
        pcl_indices = cp.array([idp + 3 for idp, x in enumerate(pcl_val_list) if x == fusion_alg], dtype=cp.int32,)
        # create a list of indices of the layers that will be updated by the point cloud with specific fusion alg
        layer_indices = cp.array([], dtype=cp.int32)
        for it, (key, val) in enumerate(layer_specs.items()):
            if key in pcl_channels and val == fusion_alg:
                layer_idx = self.layer_names.index(key)
                layer_indices = cp.append(layer_indices, layer_idx).astype(cp.int32)
        return pcl_indices, layer_indices

    def update_layers_pointcloud(self, points_all, channels, R, t, elevation_map):
        """Update the semantic map with the pointcloud.

        Args:
            points_all: semantic point cloud
            channels: list of channel names
            R: rotation matrix
            t: translation vector
            elevation_map: elevation map object
        """
        process_channels, additional_fusion = self.get_fusion(
            channels, self.param.pointcloud_channel_fusions, self.layer_specs_points
        )
        # If channels has a new layer that is not in the semantic map, add it
        for channel in process_channels:
            if channel not in self.layer_names:
                print(f"Layer {channel} not found, adding it to the semantic map")
                self.add_layer(channel)

        # Resetting new_map for the layers that are to be deleted
        self.new_map[self.delete_new_layers] = 0.0
        for fusion in list(set(additional_fusion)):
            # which layers need to be updated with this fusion algorithm
            pcl_ids, layer_ids = self.get_indices_fusion(process_channels, fusion, self.layer_specs_points)
            # update the layers with the fusion algorithm
            self.fusion_manager.execute_plugin(
                fusion,
                points_all,
                R,
                t,
                pcl_ids,
                layer_ids,
                elevation_map,
                self.semantic_map,
                self.new_map,
                self.elements_to_shift,
            )
    
    def update_layers_GET(self, FEE_params, FEE_params_var, channels, surf_points_dict):
        """Update the semantic map with the GET.

        Args:
            FEE_params: FEE parameters
            FEE_params_std: FEE parameters standard deviation
            channels: list of channel names (i.e. which FEE params to map, should be unique)
            surf_points_dict: surface points dictionary
        """
        process_channels, fusion_methods = self.get_fusion(
            channels, self.param.GET_channel_fusions, self.layer_specs_GET
        )

        # If channels has a new layer that is not in the semantic map, add it
        for channel in process_channels:
            if channel not in self.layer_names:
                print(f"Layer {channel} not found, adding it to the semantic map")
                self.add_layer(channel)

        # Resetting new_map for the layers that are to be deleted
        self.new_map[self.delete_new_layers] = 0.0

        # First compute the soil_wedge_inds and soil_wedge_weights as they are the same across properties
        # Compute the maximum acceptable distance from the soil surface for a cell to be considered part of the soil wedge
        x_t_max = FEE_params["d_prime_prime"] / (np.tan(FEE_params["alpha"] + FEE_params["beta"])- np.tan(FEE_params["alpha"]))
        # TODO: Speed this up by doing this in parallel
        # TODO: Switch this to xp for cupy compatibility after upgrading versions of cupy that will support unique axis argument
        soil_wedge_inds = np.zeros((0,2), dtype=np.uint32)
        soil_wedge_weights = np.zeros((0,), dtype=self.param.data_type)
        for p, mi in zip(surf_points_dict["points"], surf_points_dict["map_inds"]):
            valid_inds = p[:,0] < x_t_max
            # Make sure that first set of inds in the list is always valid as it is an intersected cell
            valid_inds[0] = True
            soil_wedge_inds = np.append(soil_wedge_inds, mi[valid_inds], axis=0)
            # Normalize weights by the maximum distance from the blade
            # Could also use depth instead of distance via d_w = d_prime_prime - x_t * (tan(alpha+beta) - tan(alpha))
            # If normalizing though, these should be equivalent
            weights = 1.0-p[valid_inds,0]/x_t_max
            # Ensure that the weights are between 0 and 1
            weights = np.clip(weights, 0.0, 1.0)
            # Make the first weight 1.0 as it is the intersected cell
            weights[0] = 1.0
            soil_wedge_weights = xp.append(soil_wedge_weights, weights, axis=0)
        if soil_wedge_inds.shape[0] == 0:
            print("No valid soil wedge points found")
            return
        # make sure that the index is unique and take the highest weight
        # TODO: Use cp.unique after upgrading to cupy
        # soil_wedge_inds_, idx, un_inv = np.unique(soil_wedge_inds, return_index=True, return_inverse=True, axis=0)
        # # Take the maximum weight for each unique index
        # # make array of true/false values for each unique index
        # # TODO: Speed this up by getting rid of for loop, but this should work
        # if len(idx) < soil_wedge_inds.shape[0]:
        #     soil_wedge_weights_ = np.zeros((len(idx),), dtype=self.param.data_type)
        #     print("Duplicate indices found in soil_wedge_inds")
        #     for i in range(len(idx)):
        #         # TODO: WARNING: THIS INDEXING IS WRONG, FIX IT
        #         soil_wedge_weights_[i] = cp.max(soil_wedge_weights[idx[i] == un_inv])
        #     soil_wedge_inds = soil_wedge_inds_
        # else: # No duplicates # TODO: Comment out
        #     debug=1
        
        # ensure that the soil_wedge_inds and weights are xp arrays
        soil_wedge_inds = xp.array(soil_wedge_inds, dtype=xp.uint32)
        soil_wedge_weights = xp.array(soil_wedge_weights, dtype=self.param.data_type)

        # TODO: Get rid of this For loop and modify fusion algorithm to handle multiple channels simultaneously
        for j, (fusion, channel) in enumerate(zip(fusion_methods, process_channels)):
            sem_map_idx = self.get_index(channel)
            if sem_map_idx == -1:
                print(f"Layer {channel} not found!")
                return

            # extract corresponding FEE parameter and conver to cp datatype....
            # param = xp.array(FEE_params[channel], dtype=self.param.data_type)
            param = self.param.data_type(FEE_params[channel].item())
            if channel in FEE_params_var:
                param_var = self.param.data_type(FEE_params_var[channel].item())
            else:
                param_var = 0.0
            # update the layers with the fusion algorithm
            self.fusion_manager.execute_GET_plugin(
                fusion,
                cp.uint64(sem_map_idx),
                param,
                param_var,
                soil_wedge_weights,
                soil_wedge_inds,
                self.semantic_map,
                self.new_map, # NOTE: In reality hold std deviation of the layer
            )
                # self.elements_to_shift, # Might need this if we want to shift the map, but I don't currently understand it
            debug=1

    def update_layers_image(
        self,
        # sub_key: str,
        image: cp._core.core.ndarray,
        channels: List[str],
        # fusion_methods: List[str],
        uv_correspondence: cp._core.core.ndarray,
        valid_correspondence: cp._core.core.ndarray,
        image_height: cp._core.core.ndarray,
        image_width: cp._core.core.ndarray,
    ):
        """Update the semantic map with the new image.

        Args:
            sub_key:
            image:
            uv_correspondence:
            valid_correspondence:
            image_height:
            image_width:
        """

        process_channels, fusion_methods = self.get_fusion(
            channels, self.param.image_channel_fusions, self.layer_specs_image
        )
        self.new_map[self.delete_new_layers] = 0.0
        for j, (fusion, channel) in enumerate(zip(fusion_methods, process_channels)):
            if channel not in self.layer_names:
                print(f"Layer {channel} not found, adding it to the semantic map")
                self.add_layer(channel)
            sem_map_idx = self.get_index(channel)

            if sem_map_idx == -1:
                print(f"Layer {channel} not found!")
                return

            # update the layers with the fusion algorithm
            self.fusion_manager.execute_image_plugin(
                fusion,
                cp.uint64(sem_map_idx),
                image,
                j,
                uv_correspondence,
                valid_correspondence,
                image_height,
                image_width,
                self.semantic_map,
                self.new_map,
            )

    def decode_max(self, mer):
        """Decode the float32 value into two 16 bit value containing the class probability and the class id.

        Args:
            mer:

        Returns:
            cp.array: probability
            cp.array: class id
        """
        mer = mer.astype(cp.float32)
        mer = mer.view(dtype=cp.uint32)
        ma = cp.bitwise_and(mer, 0xFFFF, dtype=np.uint16)
        ma = ma.view(np.float16)
        ma = ma.astype(np.float32)
        ind = cp.right_shift(mer, 16)
        return ma, ind

    def get_map_with_name(self, name):
        """Return the map with the given name.

        Args:
            name: layer name

        Returns:
            cp.array: map
        """
        # If the layer is a color layer, return the rgb map
        if name in self.layer_specs_points and self.layer_specs_points[name] == "color":
            m = self.get_rgb(name)
            return m
        elif name in self.layer_specs_image and self.layer_specs_image[name] == "color":
            m = self.get_rgb(name)
            return m
        else:
            m = self.get_semantic(name)
            return m
        
    def get_var_map_with_name(self, var_name):
        """Return the variance map with the given name.
        Args:
            var_name: name of layer for which we want the variance map + "_var"
                      e.g. "c_var" for the variance of the "c" layer

        Returns:
            cp.array: map
        """
        name = var_name[:-4]
        # If the layer is a color layer, return the rgb map
        if name in self.layer_specs_points and self.layer_specs_points[name] == "color":
            raise NotImplementedError("Variance for color layers not implemented")
        elif name in self.layer_specs_image and self.layer_specs_image[name] == "color":
            raise NotImplementedError("Variance for color layers not implemented")
        else:
            m = self.get_semantic(name, is_var=True)
            return m

    def get_rgb(self, name):
        """Return the rgb map with the given name.

        Args:
            name:

        Returns:
            cp.array: rgb map
        """
        idx = self.layer_names.index(name)
        c = self.process_map_for_publish(self.semantic_map[idx])
        c = c.astype(np.float32)
        return c

    def get_semantic(self, name, is_var=False):
        """Return the semantic map layer with the given name.

        Args:
            name(str): layer name
            is_var(bool): whether to return the variance map

        Returns:
            cp.array: semantic map layer
        """
        idx = self.layer_names.index(name)
        if is_var:
            m = self.new_map[idx]
        else:
            m = self.semantic_map[idx]
        c = self.process_map_for_publish(m)
        return c

    def process_map_for_publish(self, input_map):
        """Remove padding.

        Args:
            input_map(cp.array): map layer

        Returns:
            cp.array: map layer without padding
        """
        m = input_map.copy()
        return m[1:-1, 1:-1]

    def get_index(self, name):
        """Return the index of the layer with the given name.

        Args:
            name(str):

        Returns:
            int: index
        """
        if name not in self.layer_names:
            return -1
        else:
            return self.layer_names.index(name)
