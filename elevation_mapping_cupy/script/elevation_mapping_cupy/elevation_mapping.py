#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import os
from typing import List, Any, Tuple, Union

import numpy as np
import threading
import subprocess

# TODO: Move into GET movement
from shapely.geometry import Polygon
import shapely

from elevation_mapping_cupy.parameter import Parameter

from elevation_mapping_cupy.kernels import (
    add_points_kernel,
    add_color_kernel,
    color_average_kernel,
)
from elevation_mapping_cupy.kernels import sum_kernel
from elevation_mapping_cupy.kernels import error_counting_kernel
from elevation_mapping_cupy.kernels import average_map_kernel
from elevation_mapping_cupy.kernels import dilation_filter_kernel
from elevation_mapping_cupy.kernels import normal_filter_kernel
from elevation_mapping_cupy.kernels import polygon_mask_kernel
from elevation_mapping_cupy.kernels import image_to_map_correspondence_kernel
from elevation_mapping_cupy.kernels import soil_erosion_kernel

from elevation_mapping_cupy.sensor_processor import SensorProcessor, make_3x1vec
from elevation_mapping_cupy.GET_movement import GETMovement
from elevation_mapping_cupy.GET_movement import get_ext_euler_angles
from elevation_mapping_cupy.map_initializer import MapInitializer
from elevation_mapping_cupy.plugins.plugin_manager import PluginManager
from elevation_mapping_cupy.semantic_map import SemanticMap
from elevation_mapping_cupy.traversability_polygon import (
    get_masked_traversability,
    is_traversable,
    calculate_area,
    transform_to_map_position,
    transform_to_map_index,
)

import cupy as cp

xp = cp
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

class ElevationMap:
    """Core elevation mapping class."""

    def __init__(self, param: Parameter):
        """

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """
        self.param = param
        # For Soil Prop Estimation
        if param.use_soil_property_estimation:
            from train_model import DozerSoilPropEstModel
            import copy
            self.dz = DozerSoilPropEstModel.load_from_checkpoint(param.soil_prop_est_net_checkpoint_path, mode='deploy')
        
        self.data_type = self.param.data_type
        self.resolution = param.resolution
        self.center = xp.array([[0],[0],[0]], dtype=self.data_type)
        self.base_rotation = xp.eye(3, dtype=self.data_type)
        self.map_length = param.map_length
        self.cell_n = param.cell_n

        self.map_lock = threading.Lock()
        self.semantic_map = SemanticMap(self.param)
        self.layer_names = [
            "elevation",
            "variance",
            "is_valid",
            "traversability",
            "time",
            "upper_bound",
            "is_upper_bound",
            "elevation_loose"
        ]
        self.elevation_map = xp.zeros((len(self.layer_names), self.cell_n, self.cell_n), dtype=self.data_type)

        # buffers
        self.traversability_buffer = xp.full((self.cell_n, self.cell_n), xp.nan)
        self.normal_map = xp.zeros((3, self.cell_n, self.cell_n), dtype=self.data_type)
        # Initial variance
        self.initial_variance = param.initial_variance
        self.elevation_map[1] += self.initial_variance
        self.elevation_map[3] += 1.0

        # Configure sensors
        self.configure_sensors(param)

        # overlap clearance
        cell_range = int(self.param.overlap_clear_range_xy / self.resolution)
        cell_range = np.clip(cell_range, 0, self.cell_n)
        self.cell_min = self.cell_n // 2 - cell_range // 2
        self.cell_max = self.cell_n // 2 + cell_range // 2

        # Initial mean_error
        self.mean_error = 0.0
        self.additive_mean_error = 0.0

        self.compile_kernels()

        self.compile_image_kernels()

        self.semantic_map.initialize_fusion()

        # The below subprocess command does not work on Windows properly. It leaves a set of quotations around the path.
        # Bypassing this for now by assuming weight_file is already in the correct format.
        # weight_file = subprocess.getoutput('echo "' + param.weight_file + '"')
        weight_file = param.weight_file
        param.load_weights(weight_file)

        if param.use_chainer:
            from elevation_mapping_cupy.traversability_filter import get_filter_chainer
            self.traversability_filter = get_filter_chainer(param.w1, param.w2, param.w3, param.w_out)
        else:
            from elevation_mapping_cupy.traversability_filter import get_filter_torch
            self.traversability_filter = get_filter_torch(param.w1, param.w2, param.w3, param.w_out)
        self.untraversable_polygon = xp.zeros((1, 2))

        # Plugins
        self.plugin_manager = PluginManager(cell_n=self.cell_n)
        # The below subprocess command does not work on Windows properly. It leaves a set of quotations around the path.
        # Bypassing this for now by assuming plugin_config_file is already in the correct format.
        # plugin_config_file = subprocess.getoutput('echo "' + param.plugin_config_file + '"')
        plugin_config_file = param.plugin_config_file
        self.plugin_manager.load_plugin_settings(plugin_config_file)

        self.map_initializer = MapInitializer(self.initial_variance, param.initialized_variance, xp=cp, method="points")
    
    def configure_sensors(self, param):
        """Configure the sensor Processors for the elevation map.

        Args:
            param (elevation_mapping_cupy.parameter.Parameter):
        """
        self.sensor_processors = {}
        self.GETs = {}
        for sensor_ID, config in param.subscriber_cfg.items():
            if config["data_type"] == "pointcloud":
                nm_name = config["noise_model_name"]
                nm_param_name = nm_name + "_noise_model_params"
                nm_params = config.get(nm_param_name, {})
                sp = SensorProcessor(sensor_ID, nm_name, nm_params, xp)
                self.sensor_processors[sensor_ID] = sp
            elif config["data_type"] == "GET":
                GET_model_name = config["GET_model_name"]
                GET_param_name = GET_model_name + "_GET_params"
                GET_params = config.get(GET_param_name, {})
                self.GETs[sensor_ID] = GETMovement(sensor_ID, GET_model_name, param, GET_params, xp=xp)
                # Only load a single soil prop est model for now
                # if param.use_soil_property_estimation:
                #     if not hasattr(self, 'dz'):
                #         self.dz = DozerSoilPropEstModel.load_from_checkpoint(param.soil_prop_est_net_checkpoint_path, mode='deploy')
                #     else:
                #         print("Soil prop est model already loaded")
    def clear(self):
        """Reset all the layers of the elevation & the semantic map."""
        with self.map_lock:
            self.elevation_map *= 0.0
            # Initial variance
            self.elevation_map[1] += self.initial_variance
            self.semantic_map.clear()

        self.mean_error = 0.0
        self.additive_mean_error = 0.0

    def get_position(self, position):
        """Return the position of the map center.

        Args:
            position (numpy.ndarray):

        """
        position = xp.asnumpy(self.center)

    def move(self, delta_position):
        """Shift the map along all three axes according to the input.

        Args:
            delta_position (numpy.ndarray):
        """
        # Shift map using delta position.
        delta_position = xp.asarray(delta_position)
        delta_pixel = xp.round(delta_position[:2] / self.resolution)
        delta_position_xy = delta_pixel * self.resolution
        self.center[:2] += xp.asarray(delta_position_xy)
        self.center[2] += xp.asarray(delta_position[2])
        self.shift_map_xy(delta_pixel)
        self.shift_map_z(-delta_position[2])

    def move_to(self, position, R):
        """Shift the map to an absolute position and update the rotation of the robot.

        Args:
            position (numpy.ndarray):
            R (cupy._core.core.ndarray):
        """
        # Shift map to the center of robot.
        self.base_rotation = xp.asarray(R, dtype=self.data_type)
        position = xp.asarray(position)
        delta = position - self.center
        delta_pixel = xp.around(delta[:2] / self.resolution)
        delta_xy = delta_pixel * self.resolution
        self.center[:2] += delta_xy
        self.center[2] += delta[2]
        self.shift_map_xy(-delta_pixel)
        self.shift_map_z(-delta[2])

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

    def shift_map_xy(self, delta_pixel):
        """Shift the map along the horizontal axes according to the input.

        Args:
            delta_pixel (cupy._core.core.ndarray):

        """
        shift_value = delta_pixel.astype(cp.int32)
        if cp.abs(shift_value).sum() == 0:
            return
        with self.map_lock:
            self.elevation_map = cp.roll(self.elevation_map, shift_value, axis=(1, 2))
            self.pad_value(self.elevation_map, shift_value, value=0.0)
            self.pad_value(self.elevation_map, shift_value, idx=1, value=self.initial_variance)
            self.semantic_map.shift_map_xy(shift_value)

    def shift_map_z(self, delta_z):
        """Shift the relevant layers along the vertical axis.

        Args:
            delta_z (cupy._core.core.ndarray):
        """
        with self.map_lock:
            # elevation
            self.elevation_map[0] += delta_z
            # upper bound
            self.elevation_map[5] += delta_z

    def compile_kernels(self):
        """Compile all kernels belonging to the elevation map."""

        self.new_map = cp.zeros((self.elevation_map.shape[0], self.cell_n, self.cell_n), dtype=self.data_type,)
        self.traversability_input = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.traversability_mask_dummy = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.min_filtered = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.min_filtered_mask = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.mask = cp.zeros((self.cell_n, self.cell_n), dtype=self.data_type)
        self.add_points_kernel = add_points_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.mahalanobis_thresh,
            self.param.outlier_variance,
            self.param.wall_num_thresh,
            self.param.max_ray_length,
            self.param.cleanup_step,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.cleanup_cos_thresh,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
            self.param.enable_edge_sharpen,
            self.param.enable_visibility_cleanup,
            self.param.enable_visibility_height_update,
        )
        self.error_counting_kernel = error_counting_kernel(
            self.resolution,
            self.cell_n,
            self.cell_n,
            self.param.mahalanobis_thresh,
            self.param.drift_compensation_variance_inlier,
            self.param.traversability_inlier,
            self.param.min_valid_distance,
            self.param.max_height_range,
            self.param.ramped_height_range_a,
            self.param.ramped_height_range_b,
            self.param.ramped_height_range_c,
        )
        self.average_map_kernel = average_map_kernel(
            self.cell_n, self.cell_n, self.param.max_variance, self.initial_variance
        )

        self.dilation_filter_kernel = dilation_filter_kernel(self.cell_n, self.cell_n, self.param.dilation_size)
        self.dilation_filter_kernel_initializer = dilation_filter_kernel(
            self.cell_n, self.cell_n, self.param.dilation_size_initialize
        )
        self.polygon_mask_kernel = polygon_mask_kernel(self.cell_n, self.cell_n, self.resolution)
        self.normal_filter_kernel = normal_filter_kernel(self.cell_n, self.cell_n, self.resolution)

        # Compute the loose soil density based on the swell factor and the assumed compacted soil density
        loose_soil_gamma = self.param.compacted_soil_moist_unit_weight / self.param.swell_factor
        self.soil_erosion_kernel = soil_erosion_kernel(self.cell_n,
                                                       self.cell_n,
                                                       self.resolution,
                                                       self.param.loose_soil_cohesion,
                                                       self.param.loose_soil_phi,
                                                       loose_soil_gamma,
                                                       self.param.soil_erosion_alpha_min,)

    def compile_image_kernels(self):
        """Compile kernels related to processing image messages."""

        for config in self.param.subscriber_cfg.values():
            if config["data_type"] == "image":
                self.valid_correspondence = cp.asarray(
                    np.zeros((self.cell_n, self.cell_n), dtype=np.bool_), dtype=np.bool_
                )
                self.uv_correspondence = cp.asarray(
                    np.zeros((2, self.cell_n, self.cell_n), dtype=np.float32), dtype=np.float32,
                )
                # self.distance_correspondence = cp.asarray(
                #     np.zeros((self.cell_n, self.cell_n), dtype=np.float32), dtype=np.float32
                # )
                # TODO tolerance_z_collision add parameter
                self.image_to_map_correspondence_kernel = image_to_map_correspondence_kernel(
                    resolution=self.resolution, width=self.cell_n, height=self.cell_n, tolerance_z_collision=0.10,
                )
                break

    def shift_translation_to_map_center(self, t):
        """Deduct the map center to get the translation of a point w.r.t. the map center.

        Args:
            t (cupy._core.core.ndarray): Absolute point position
        
        Returns:
            cupy._core.core.ndarray: Translated point position
        """
        t_shift = t -  self.center
        return t_shift

    def update_map_with_kernel(self, sensor_ID, points_all, channels, C_MB, B_r_MB, Sigma_b_r_MB, Sigma_Theta_MB):
        """Update map with new measurement.

        Args:
            sensor_ID (str):                            Sensor ID
            points_all (cupy._core.core.ndarray):       Points in the sensor frame (i.e. S_r_SP)
            channels (List[str]):                       List of channels in the point cloud besides x, y, z
            C_MB (cupy._core.core.ndarray):             Rotation matrix from the map frame to the base frame
            B_r_MB (cupy._core.core.ndarray):           Translation vector from the map frame to the base frame
            Sigma_b_r_MB (cupy._core.core.ndarray):     Covariance of base position in map frame.
            Sigma_Theta_MB (cupy._core.core.ndarray):   Covariance of base orientation in map frame where orientation is defined 
                                                        as fixed-axis(extrinsic) xyz(roll, pitch, yaw) Euler angles.
        """
        self.new_map *= 0.0
        error = cp.array([0.0], dtype=cp.float32)
        error_cnt = cp.array([0], dtype=cp.float32)
        points = points_all[:, :3]
        # Obtain the transform from the map frame to the sensor frame
        C_BS = self.sensor_processors[sensor_ID].C_BS
        B_r_BS = self.sensor_processors[sensor_ID].B_r_BS
        C_MS = C_MB @ C_BS
        B_r_MS = C_MB @ B_r_BS + B_r_MB
        # additional_fusion = self.get_fusion_of_pcl(channels)
        with self.map_lock:
            # Now the translation is w.r.t. the map center frame O
            O_r_OS = self.shift_translation_to_map_center(B_r_MS)
            # C_MS is equivalent to C_OS because we assume the map center frame O is aligned with the map frame M
            C_OS = C_MS
            self.error_counting_kernel(
                self.elevation_map,
                points,
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                C_OS,
                O_r_OS,
                self.new_map,
                error,
                error_cnt,
                size=(points.shape[0]),
            )
            # Compute position and orientation noise of the base frame in the map frame for
            # triggering drift compensation
            position_noise = cp.sqrt(Sigma_b_r_MB[0, 0] + Sigma_b_r_MB[1, 1] + Sigma_b_r_MB[2, 2])
            orientation_noise = cp.sqrt(Sigma_Theta_MB[0, 0] + Sigma_Theta_MB[1, 1] + Sigma_Theta_MB[2, 2])
            if (
                self.param.enable_drift_compensation
                and error_cnt > self.param.min_height_drift_cnt
                and (
                    position_noise > self.param.position_noise_thresh
                    or orientation_noise > self.param.orientation_noise_thresh
                )
            ):
                self.mean_error = error / error_cnt
                self.additive_mean_error += self.mean_error
                if np.abs(self.mean_error) < self.param.max_drift:
                    self.elevation_map[0] += self.mean_error * self.param.drift_compensation_alpha
            # Compute the vertical variance for the sensor using uncertainty propagation for the sensor
            var_h = self.sensor_processors[sensor_ID].get_z_variance(points, C_MB, B_r_MB, Sigma_Theta_MB, Sigma_b_r_MB)
            self.add_points_kernel(
                cp.array([0.0], dtype=self.data_type),
                cp.array([0.0], dtype=self.data_type),
                C_OS,
                O_r_OS,
                var_h,
                self.normal_map,
                points,
                self.elevation_map,
                self.new_map,
                size=(points.shape[0]),
            )
            self.average_map_kernel(self.new_map, self.elevation_map, size=(self.cell_n * self.cell_n))

            self.semantic_map.update_layers_pointcloud(points_all, channels, C_OS, O_r_OS, self.new_map)

            if self.param.enable_overlap_clearance:
                self.clear_overlap_map(O_r_OS)
            # dilation before traversability_filter
            self.traversability_input *= 0.0
            self.dilation_filter_kernel(
                self.elevation_map[5],
                self.elevation_map[2] + self.elevation_map[6],
                self.traversability_input,
                self.traversability_mask_dummy,
                size=(self.cell_n * self.cell_n),
            )
            # calculate traversability
            traversability = self.traversability_filter(self.traversability_input)
            self.elevation_map[3][3:-3, 3:-3] = traversability.reshape(
                (traversability.shape[2], traversability.shape[3])
            )

        # calculate normal vectors
        self.update_normal(self.traversability_input)

    def clear_overlap_map(self, t):
        """Clear overlapping areas around the map center.

        Args:
            t (cupy._core.core.ndarray): Absolute point position
        """

        height_min = t[2] - self.param.overlap_clear_range_z
        height_max = t[2] + self.param.overlap_clear_range_z
        near_map = self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max]
        valid_idx = ~cp.logical_or(near_map[0] < height_min, near_map[0] > height_max)
        near_map[0] = cp.where(valid_idx, near_map[0], 0.0)
        near_map[1] = cp.where(valid_idx, near_map[1], self.initial_variance)
        near_map[2] = cp.where(valid_idx, near_map[2], 0.0)
        valid_idx = ~cp.logical_or(near_map[5] < height_min, near_map[5] > height_max)
        near_map[5] = cp.where(valid_idx, near_map[5], 0.0)
        near_map[6] = cp.where(valid_idx, near_map[6], 0.0)
        self.elevation_map[:, self.cell_min : self.cell_max, self.cell_min : self.cell_max] = near_map

    def get_additive_mean_error(self):
        """Returns the additive mean error.

        Returns:

        """
        return self.additive_mean_error

    def update_variance(self, dt):
        """Increase the variance of the valid cells at the rate specified"""
        with self.map_lock:
            # Ideally we would use the time layer to do this update, but that would require clearing the time layer after
            # increasing the variance. We don't want to do that because the time layer should be increasing until
            # a cell is updated with a new measurement. Therefore we instead require that the caller provides the time elapsed
            self.elevation_map[1] += self.param.variance_inflation_rate * dt * self.elevation_map[2]

    def update_time(self, dt):
        """adds the time elapsed to the time layer. Function is should be called at rate of 1/time_interval approximately"""
        with self.map_lock:
            self.elevation_map[4] += dt

    def update_upper_bound_with_valid_elevation(self):
        """Filters all invalid cell's upper_bound and is_upper_bound layers."""
        mask = self.elevation_map[2] > 0.5
        self.elevation_map[5] = cp.where(mask, self.elevation_map[0], self.elevation_map[5])
        self.elevation_map[6] = cp.where(mask, 0.0, self.elevation_map[6])
    
    def input_GET_movement(self, 
                           GET_ID: str,
                           T_MG0: cp._core.core.ndarray,
                           T_MG1: cp._core.core.ndarray,
                           M_r_MG: np.ndarray,
                           n_steps: np.int32,
                           var_h: float,
                           roll: float,
                           soil_nn_input: dict = None
    ):
        """Input the GET movement and update the elevation map.

        Args:
            GET_ID (str):                               GET ID
            T_MG0 (cupy._core.core.ndarray):            Transformation matrix from the GET frame to the map frame at time t0
            T_MG1 (cupy._core.core.ndarray):            Transformation matrix from the GET frame to the map frame at time t1
            M_r_MG (np.ndarray) (n_steps,3):            Position of the GET frame w.r.t. the map frame over time steps
            n_steps (np.int32):                         Number of time steps between T_MG0 and T_MG1 (inclusive) used for interpolation
            var_h (float):                              Variance of the height measurement
            roll (float):                               Roll angle of the GET frame w.r.t. the map frame
            soil_nn_input (dict):                       Dictionary containing the input data for the soil property estimation model
        Returns:
            FEE_em_params:                              FEE parameters obtained from map as dictionary containing:
                                                        d, alpha, rho, w, Q (None if no valid GET movement is detected)
            surf_points_dict:                           Dictionary containing the surface points for each slice/intersected cell
                                                        with keys points and inds where points includes x_t and z, the distance to 
                                                        the blade along the translation direction and the height of the cell, and
                                                        inds includes the indicies of the cells used for each line/surface fit for each slice.
        """
        T_MG0 = np.asarray(T_MG0, dtype=self.data_type)
        T_MG1 = np.asarray(T_MG1, dtype=self.data_type)
        
        with self.map_lock:
            position = np.array([0, 0, 0], dtype=self.data_type)
            self.get_position(position)
            # TODO: Make the varh derived from the pose uncertainty and use a sensor model
            FEE_em_params, surf_points_dict = self.GETs[GET_ID].update_map_with_GET_movement(
                self.elevation_map,
                position,
                self.cell_n,
                self.resolution,
                T_MG0,
                T_MG1,
                M_r_MG,
                n_steps,
                var_h,
                roll,
            )


            if surf_points_dict is None:
                intersected_inds = None
            else:
                # Pull out the intersected inds so that erosion is not allowed to erode these cells or into them
                # The first entry in map_inds is the map inds of the intersected cells
                # keeping as numpy as we will have to perform numpy operations with shapely later
                intersected_inds = np.array([map_ind[0] for map_ind in surf_points_dict['map_inds']])

            # Just using T_MG1 for now to perform soil erosion. This shouldn't matter as long as our ROI is large enough
            # The factor of 3 here is a bit of a hack to get the soil to erode more quickly since we aren't accounting for v_0
            dT = n_steps / 60.0 * 5.0 # TODO: This should probably be passed in to the function
            if self.param.use_soil_erosion:
                # Perform erosion as many times as necessary to cover the dT time interval given the maximum erosion time step
                while dT > 0:
                    dt = min(dT, self.param.soil_erosion_maximum_dt)
                    dT -= dt
                    self.perform_soil_erosion(GET_ID, T_MG1, intersected_inds, dt)

            if self.param.use_soil_property_estimation and surf_points_dict is not None:
                # Now predict the soil properties
                sample_len = self.dz.hparams['sample_len']
                assert len(soil_nn_input['position']) == len(soil_nn_input['velocity']) == len(soil_nn_input['action'] == sample_len), "Lengths of position, velocity, and action must be the same"
                # Add the FEE parameters
                em_metadata = copy.deepcopy(FEE_em_params) # TODO: Do we need to copy here?
                # Override Q for no particles. TODO: Remove later
                NO_PARTICLES = False
                if NO_PARTICLES:
                    em_metadata['Q'] *= 0.0
                # Offset the step to match the sample length (the params should correspond to the end of the sample)
                # These two should beare equivalent
                # sweep_start_step = sample_len - em_metadata['d_step'][-1] - 1
                sweep_start_step = sample_len - len(em_metadata['d_step'])
                em_metadata['step'] = sweep_start_step + em_metadata['d_step'] # Needs to be a list because of the way model process batches
                soil_nn_input['em_metadata'] = [em_metadata]
                # Pass empty PGT metadata
                soil_nn_input['metadata'] = {}
                dataset = self.dz.deploy_dataset([soil_nn_input])
                dataloader = self.dz.deploy_dataloader(dataset)
                FEE_params, FEE_params_var = self.dz.deploy_step(next(iter(dataloader)))
                # print(metadata)
                # Add d_prime_prime to metadata
                FEE_params['d_prime_prime'] = FEE_em_params['d_prime_prime']

                # Now add the estimated soil properties to the semantic map
                channels = ['c', 'phi', 'gamma', 'delta', 'c_a', 'd', 'MeanCuttingForceMag'] # TODO: Should this be d_prime?
                soil_wedge_inds = self.semantic_map.update_layers_GET(FEE_params, FEE_params_var, channels, surf_points_dict)

                # Trigger FEE Plugin processing
                if False: # Disable for now as this is time consuming
                    self.plugin_manager.update_with_name("FEE_index",
                                                        self.elevation_map,
                                                        self.layer_names,
                                                        semantic_map=self.semantic_map.semantic_map,
                                                        semantic_params=self.semantic_map.layer_names,
                                                        semantic_new_map=self.semantic_map.new_map,
                                                        semantic_var_params=self.semantic_map.var_layer_names,
                                                        updated_inds=soil_wedge_inds,
                                                        )
        # TODO: Possibly get rid of surf_points_dict and just return FEE_em_params once we get working with semantic map
        return FEE_em_params, surf_points_dict
    
    def perform_soil_erosion(self,
                            GET_ID: str,
                            T_MG: cp._core.core.ndarray,
                            GET_inds: cp._core.core.ndarray,
                            dt: float,
    ):
        """Perform soil erosion on the map around the current position of the blade. (different than erosion plugin)
        This will only erode the loose soil layer and is only an approximation of the actual erosion process
        as it assumes fixed nominal soil properties and does not account for the actual soil properties.
        Args:
            GET_ID (str):                   GET ID
            T_MG (np.ndarray)(4,4):         Transformation matrix from the GET frame to the map frame
            GET_inds (np.ndarray)(n,2):     Indicies of the GET blade
            dt (float):                     Time step for erosion
        Returns:
            None:
        """

        # Create mask of the GET blade where no erosion should occur
        GET_mask = cp.ones((self.cell_n, self.cell_n), dtype=cp.bool_)
        if GET_inds is not None:
            GET_mask[GET_inds[:,0], GET_inds[:,1]] = False

        # Account for the translation of the map origin (map_center) frame from the map frame
        # Operations are done in the map origin frame O
        M_r_MG = cp.array(T_MG[:3, 3:], dtype=self.data_type)
        O_r_OG = self.shift_translation_to_map_center(M_r_MG).get()
        T_OG = np.eye(4, dtype=self.data_type)
        # Extract yaw from the rotation matrix
        r, p, y = get_ext_euler_angles(T_MG[:3,:3], xp=np)
        # Only apply yaw rotation to ROI
        T_OG[0, 0] = np.cos(y)
        T_OG[0, 1] = np.sin(y)
        T_OG[1, 0] = -np.sin(y)
        T_OG[1, 1] = np.cos(y)
        T_OG[:3, 3:] = O_r_OG

        # Define a Rectanguar ROI in frame G around the blade to perform erosion
        # TODO: Consider moving these parameters to the parameters.yaml file as they should be the same for all blades
        dx = self.param.erosion_ROI_dx/2.0
        dy = (self.param.erosion_ROI_dy + self.GETs[GET_ID].GET_params['blade_width'])/2.0


        ROI_G = np.zeros((4,4), dtype=self.data_type)
        ROI_G[:,0] = np.array([dx, dy, 0, 1.0])
        ROI_G[:,1] = np.array([dx, -dy, 0, 1.0])
        ROI_G[:,2] = np.array([-dx, -dy, 0, 1.0])
        ROI_G[:,3] = np.array([-dx, dy, 0, 1.0])
        
        # Transform the ROI to the map origin frame
        ROI_O = T_OG @ ROI_G
        
        # Get the map indices of the ROI corners
        # TODO: Maybe this should be done in metric coordinates not indices to avoid rounding errors
        ROI_inds = transform_to_map_index(ROI_O[0:2].T, self.center[0:2].get(), self.cell_n, self.resolution, xp=np)
        # Define a polygon in the map index frame that will be usd to test if a cell is within the ROI
        ROI_poly = Polygon(ROI_inds)
        # Obtain an array of map indicies for a map aligned bounding box around the ROI
        bnds = np.array(ROI_poly.bounds, dtype=np.int32) #(minx, miny, maxx, maxy)
        # Loop through tiled bounding boxes to avoid paralell erosion on the same cell
        # Will go thorough 4 possible regions or smaller if the ROI is small or on the edge of the map
        for i in range(2):
            for j in range(2):
                ROI_bbox_inds = np.mgrid[bnds[0]+i:bnds[2]+1:2, bnds[1]+j:bnds[3]+1:2].reshape(2,-1).T
                # Get the indicies that are within the ROI
                valid_inds = shapely.contains_xy(ROI_poly, ROI_bbox_inds)
                if np.all(~valid_inds):
                    continue
                # Finally convert to cupy after interacting with shapely
                ROI_bbox_inds = cp.asarray(ROI_bbox_inds[valid_inds], dtype=cp.int32)
                # Perform erosion on the cells within the diagonal
                self.soil_erosion_kernel(ROI_bbox_inds, dt, GET_mask, self.elevation_map, size=(ROI_bbox_inds.shape[0]))
        
    
    def get_GET_depth(self,
                      GET_ID: str,
                      M_r_MG: np.ndarray,
                      vel_xy: np.ndarray,
    ):
        """
        Obtain blade depth given the current position of the blade using last best FEE parameter fit.
        This may be useful for control purposes.
        This function determines the swept mesh projection parameters to use based on the velocity direction.
        The default is to use the positive (swept volume along the blade normal), but if the velocity is in the opposite
        direction then the negative swept volume parameters are used.
        Args:
            GET_ID (str):                   GET ID
            M_r_MG (np.ndarray)(3,):        The origin of the GET in the map frame over the sweep
            vel_xy (np.ndarray)(2,):        The velocity of the blade in the xy plane in the map frame
        Returns:
            d_prime (np.ndarray)(1,):       The blade depth wrt the horizontal plane
            d (np.ndarray)(1,):             The blade depth wrt the terrain surface
        """
        d_prime, d = self.GETs[GET_ID].get_blade_depth(M_r_MG, vel_xy, xp.asnumpy(self.center.flatten()))
        return d_prime, d

    def input_pointcloud(
        self,
        sensor_ID: str,
        raw_points: cp._core.core.ndarray,
        channels: List[str],
        C_MB: cp._core.core.ndarray,
        B_r_MB: cp._core.core.ndarray,
        Sigma_b_r_MB: cp._core.core.ndarray,
        Sigma_Theta_MB: cp._core.core.ndarray,
    ):
        """Input the point cloud and fuse the new measurements to update the elevation map.

        Args:
            sensor_ID (str):                            Sensor ID
            raw_points (cupy._core.core.ndarray):       Points in the sensor frame (S_r_SP)
            channels (List[str]):                       List of channels in the point cloud including x, y, z as
                                                        the first three channels
            C_MB (cupy._core.core.ndarray):             Rotation matrix from the map frame to the base frame
            B_r_MB (cupy._core.core.ndarray):           Translation vector from the map frame to the base frame
            Sigma_b_r_MB (cupy._core.core.ndarray):     Covariance of base position in map frame.
            Sigma_Theta_MB (cupy._core.core.ndarray):   Covariance of base orientation in map frame where orientation is defined 
                                                        as fixed-axis(extrinsic) xyz(roll, pitch, yaw) Euler angles.
        Returns:
            None:
        """
        raw_points = cp.asarray(raw_points, dtype=self.data_type)
        additional_channels = channels[3:]
        raw_points = raw_points[~cp.isnan(raw_points).any(axis=1)]
        self.update_map_with_kernel(
            sensor_ID,
            raw_points,
            additional_channels,
            cp.asarray(C_MB, dtype=self.data_type),
            make_3x1vec(cp.asarray(B_r_MB, dtype=self.data_type)),
            cp.asarray(Sigma_b_r_MB, dtype=self.data_type),
            cp.asarray(Sigma_Theta_MB, dtype=self.data_type),
        )

    def input_image(
        self,
        image: List[cp._core.core.ndarray],
        channels: List[str],
        # fusion_methods: List[str],
        R: cp._core.core.ndarray,
        t: cp._core.core.ndarray,
        K: cp._core.core.ndarray,
        image_height: int,
        image_width: int,
    ):
        """Input image and fuse the new measurements to update the elevation map.

        Args:
            sub_key (str): Key used to identify the subscriber configuration
            image (List[cupy._core.core.ndarray]): List of array containing the individual image input channels
            R (cupy._core.core.ndarray): Camera optical center rotation
            t (cupy._core.core.ndarray): Camera optical center translation
            K (cupy._core.core.ndarray): Camera intrinsics
            image_height (int): Image height
            image_width (int): Image width

        Returns:
            None:
        """
        image = np.stack(image, axis=0)
        if len(image.shape) == 2:
            image = image[None]

        # Convert to cupy
        image = cp.asarray(image, dtype=self.data_type)
        K = cp.asarray(K, dtype=self.data_type)
        R = cp.asarray(R, dtype=self.data_type)
        t = cp.asarray(t, dtype=self.data_type)
        image_height = cp.float32(image_height)
        image_width = cp.float32(image_width)

        # Calculate transformation matrix
        P = cp.asarray(K @ cp.concatenate([R, t[:, None]], 1), dtype=np.float32)
        t_cam_map = -R.T @ t - self.center
        t_cam_map = t_cam_map.get()
        x1 = cp.uint32((self.cell_n / 2) + ((t_cam_map[0]) / self.resolution))
        y1 = cp.uint32((self.cell_n / 2) + ((t_cam_map[1]) / self.resolution))
        z1 = cp.float32(t_cam_map[2])

        self.uv_correspondence *= 0
        self.valid_correspondence[:, :] = False
        # self.distance_correspondence *= 0.0

        with self.map_lock:
            self.image_to_map_correspondence_kernel(
                self.elevation_map,
                x1,
                y1,
                z1,
                P.reshape(-1),
                image_height,
                image_width,
                self.center,
                self.uv_correspondence,
                self.valid_correspondence,
                size=int(self.cell_n * self.cell_n),
            )
            self.semantic_map.update_layers_image(
                image, channels, self.uv_correspondence, self.valid_correspondence, image_height, image_width,
            )

    def update_normal(self, dilated_map):
        """Clear the normal map and then apply the normal kernel with dilated map as input.

        Args:
            dilated_map (cupy._core.core.ndarray):
        """
        with self.map_lock:
            self.normal_map *= 0.0
            self.normal_filter_kernel(
                dilated_map, self.elevation_map[2], self.normal_map, size=(self.cell_n * self.cell_n),
            )
    
    def update_plugin(self, plugin_name=None, layer_name=None):
        """Update the plugin layers.
           Either by specifying the plugin name or a layer name to find the corresponding plugin.

        Args:
            plugin_name (str): Plugin name
            layer_name (str): Layer name
        """
        if plugin_name is None:
            p_idx = self.plugin_manager.get_plugin_index_with_layer_name(layer_name)
            plugin_name = self.plugin_manager.plugin_names[p_idx]
        self.plugin_manager.update_with_name(
                    plugin_name,
                    self.elevation_map,
                    self.layer_names,
                    semantic_map=self.semantic_map.semantic_map,
                    semantic_params=self.semantic_map.layer_names,
                    semantic_new_map=self.semantic_map.new_map,
                    semantic_var_params=self.semantic_map.var_layer_names,
                    rotation=self.base_rotation,
                    elements_to_shift=self.semantic_map.elements_to_shift,
                    updated_inds=None, # Use internal logic to determine which indices to update
                )

    def process_map_for_publish(self, input_map, fill_nan=False, add_z=False, xp=cp):
        """Process the input_map according to the fill_nan and add_z flags.

        Args:
            input_map (cupy._core.core.ndarray):
            fill_nan (bool):
            add_z (bool):
            xp (module):

        Returns:
            cupy._core.core.ndarray:
        """
        m = input_map.copy()
        if fill_nan:
            m = xp.where(self.elevation_map[2] > 0.5, m, xp.nan)
        if add_z:
            m = m + self.center[2]
        return m[1:-1, 1:-1]

    def get_elevation(self):
        """Get the elevation layer.

        Returns:
            elevation layer

        """
        return self.process_map_for_publish(self.elevation_map[0], fill_nan=True, add_z=True)
    
    def get_elevation_loose(self):
        """Get the elevation loose layer.

        Returns:
            elevation layer

        """
        return self.process_map_for_publish(self.elevation_map[7], fill_nan=True, add_z=False)

    def get_variance(self):
        """Get the variance layer.

        Returns:
            variance layer
        """
        return self.process_map_for_publish(self.elevation_map[1], fill_nan=False, add_z=False)

    def get_traversability(self):
        """Get the traversability layer.

        Returns:
            traversability layer
        """
        traversability = cp.where(
            (self.elevation_map[2] + self.elevation_map[6]) > 0.5, self.elevation_map[3].copy(), cp.nan,
        )
        self.traversability_buffer[3:-3, 3:-3] = traversability[3:-3, 3:-3]
        traversability = self.traversability_buffer[1:-1, 1:-1]
        return traversability

    def get_time(self):
        """Get the time layer.

        Returns:
            time layer
        """
        return self.process_map_for_publish(self.elevation_map[4], fill_nan=False, add_z=False)

    def get_upper_bound(self):
        """Get the upper bound layer.

        Returns:
            upper_bound: upper bound layer
        """
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5,
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        upper_bound = cp.where(valid, self.elevation_map[5].copy(), cp.nan)
        upper_bound = upper_bound[1:-1, 1:-1] + self.center[2]
        return upper_bound

    def get_is_upper_bound(self):
        """Get the is upper bound layer.

        Returns:
            is_upper_bound: layer
        """
        if self.param.use_only_above_for_upper_bound:
            valid = cp.logical_or(
                cp.logical_and(self.elevation_map[5] > 0.0, self.elevation_map[6] > 0.5), self.elevation_map[2] > 0.5,
            )
        else:
            valid = cp.logical_or(self.elevation_map[2] > 0.5, self.elevation_map[6] > 0.5)
        is_upper_bound = cp.where(valid, self.elevation_map[6].copy(), cp.nan)
        is_upper_bound = is_upper_bound[1:-1, 1:-1]
        return is_upper_bound

    def xp_of_array(self, array):
        """Indicate which library is used for xp.

        Args:
            array (cupy._core.core.ndarray):

        Returns:
            module: either np or cp
        """
        if type(array) == cp.ndarray:
            return cp
        elif type(array) == np.ndarray:
            return np

    def copy_to_cpu(self, array, data, stream=None):
        """Transforms the data to float32 and if on gpu loads it to cpu.

        Args:
            array (cupy._core.core.ndarray):
            data (numpy.ndarray):
            stream (Union[None, cupy.cuda.stream.Stream, None, None, None, None, None, None, None]):
        """
        if type(array) == np.ndarray:
            data[...] = array.astype(np.float32)
        elif type(array) == cp.ndarray:
            if stream is not None:
                data[...] = cp.asnumpy(array.astype(np.float32), stream=stream)
            else:
                data[...] = cp.asnumpy(array.astype(np.float32))

    def exists_layer(self, name):
        """Check if the layer exists in elevation map or in the semantic map.

        Args:
            name (str): Layer name

        Returns:
            bool: Indicates if layer exists.
        """
        if name in self.layer_names:
            return True
        elif name in self.semantic_map.layer_names:
            return True
        elif name in self.semantic_map.var_layer_names:
            return True
        elif name in self.plugin_manager.layer_names:
            return True
        else:
            return False

    def get_map_with_name_ref(self, name, data, update_plugin=False):
        """Load a layer according to the name input to the data input.

        Args:
            name (str): Name of the layer.
            data (numpy.ndarray): Data structure that contains layer.

        """
        use_stream = True
        xp = cp
        with self.map_lock:
            if name == "elevation":
                m = self.get_elevation()
                use_stream = False
            elif name == "variance":
                m = self.get_variance()
            elif name == "traversability":
                m = self.get_traversability()
            elif name == "time":
                m = self.get_time()
            elif name == "upper_bound":
                m = self.get_upper_bound()
            elif name == "is_upper_bound":
                m = self.get_is_upper_bound()
            elif name == "elevation_loose":
                m = self.get_elevation_loose()
            elif name == "normal_x":
                m = self.normal_map.copy()[0, 1:-1, 1:-1]
            elif name == "normal_y":
                m = self.normal_map.copy()[1, 1:-1, 1:-1]
            elif name == "normal_z":
                m = self.normal_map.copy()[2, 1:-1, 1:-1]
            elif name in self.semantic_map.layer_names:
                m = self.semantic_map.get_map_with_name(name)
            elif name in self.semantic_map.var_layer_names:
                m = self.semantic_map.get_var_map_with_name(name)
            elif name in self.plugin_manager.layer_names:
                if update_plugin:
                    self.update_plugin(layer_name=name)
                m = self.plugin_manager.get_map_with_name(name)
                p = self.plugin_manager.get_param_with_name(name)
                xp = self.xp_of_array(m)
                m = self.process_map_for_publish(m, fill_nan=p.fill_nan, add_z=p.is_height_layer, xp=xp)
            else:
                print("Layer {} is not in the map".format(name))
                return
        # TODO: Determine if flipping is necessary, pretty sure it isn't
        # m = xp.flip(m, 0)
        # m = xp.flip(m, 1)
        if use_stream:
            stream = cp.cuda.Stream(non_blocking=False)
        else:
            stream = None
        self.copy_to_cpu(m, data, stream=stream)

    def get_normal_maps(self):
        """Get the normal maps.

        Returns:
            maps: the three normal values for each cell
        """
        normal = self.normal_map.copy()
        normal_x = normal[0, 1:-1, 1:-1]
        normal_y = normal[1, 1:-1, 1:-1]
        normal_z = normal[2, 1:-1, 1:-1]
        maps = xp.stack([normal_x, normal_y, normal_z], axis=0)
        # TODO: Determine if flipping is necessary, pretty sure it isn't
        # maps = xp.flip(maps, 1)
        # maps = xp.flip(maps, 2)
        maps = xp.asnumpy(maps)
        return maps

    def get_normal_ref(self, normal_x_data, normal_y_data, normal_z_data):
        """Get the normal maps as reference.

        Args:
            normal_x_data:
            normal_y_data:
            normal_z_data:
        """
        maps = self.get_normal_maps()
        self.stream = cp.cuda.Stream(non_blocking=True)
        normal_x_data[...] = xp.asnumpy(maps[0], stream=self.stream)
        normal_y_data[...] = xp.asnumpy(maps[1], stream=self.stream)
        normal_z_data[...] = xp.asnumpy(maps[2], stream=self.stream)

    def get_layer(self, name, update_plugin=False):
        """Return the layer with the name input.

        Args:
            name: The layers name.

        Returns:
            return_map: The rqeuested layer.

        """
        if name in self.layer_names:
            idx = self.layer_names.index(name)
            return_map = self.elevation_map[idx]
        elif name in self.semantic_map.layer_names:
            idx = self.semantic_map.layer_names.index(name)
            return_map = self.semantic_map.semantic_map[idx]
        elif name in self.plugin_manager.layer_names:
            if update_plugin:
                self.update_plugin(name)
            return_map = self.plugin_manager.get_map_with_name(name)
        else:
            print("Layer {} is not in the map, returning traversabiltiy!".format(name))
            return
        return return_map

    def get_polygon_traversability(self, polygon, result):
        """Check if input polygons are traversable.

        Args:
            polygon (cupy._core.core.ndarray):
            result (numpy.ndarray):

        Returns:
            Union[None, int]:
        """
        polygon = xp.asarray(polygon)
        area = calculate_area(polygon)
        polygon = polygon.astype(self.data_type)
        pmin = self.center[:2] - self.map_length / 2 + self.resolution
        pmax = self.center[:2] + self.map_length / 2 - self.resolution
        polygon[:, 0] = polygon[:, 0].clip(pmin[0], pmax[0])
        polygon[:, 1] = polygon[:, 1].clip(pmin[1], pmax[1])
        polygon_min = polygon.min(axis=0)
        polygon_max = polygon.max(axis=0)
        polygon_bbox = cp.concatenate([polygon_min, polygon_max]).flatten()
        polygon_n = xp.array(polygon.shape[0], dtype=np.int16)
        clipped_area = calculate_area(polygon)
        self.polygon_mask_kernel(
            polygon,
            self.center[0],
            self.center[1],
            polygon_n,
            polygon_bbox,
            self.mask,
            size=(self.cell_n * self.cell_n),
        )
        tmp_map = self.get_layer(self.param.checker_layer)
        masked, masked_isvalid = get_masked_traversability(self.elevation_map, self.mask, tmp_map)
        if masked_isvalid.sum() > 0:
            t = masked.sum() / masked_isvalid.sum()
        else:
            t = cp.asarray(0.0, dtype=self.data_type)
        is_safe, un_polygon = is_traversable(
            masked, self.param.safe_thresh, self.param.safe_min_thresh, self.param.max_unsafe_n,
        )
        untraversable_polygon_num = 0
        if un_polygon is not None:
            un_polygon = transform_to_map_position(un_polygon, self.center[:2], self.cell_n, self.resolution)
            untraversable_polygon_num = un_polygon.shape[0]
        if clipped_area < 0.001:
            is_safe = False
            print("requested polygon is outside of the map")
        result[...] = np.array([is_safe, t.get(), area.get()])
        self.untraversable_polygon = un_polygon
        return untraversable_polygon_num

    def get_untraversable_polygon(self, untraversable_polygon):
        """Copy the untraversable polygons to input untraversable_polygons.

        Args:
            untraversable_polygon (numpy.ndarray):
        """
        untraversable_polygon[...] = xp.asnumpy(self.untraversable_polygon)

    def initialize_map(self, points, method="cubic"):
        """Initializes the map according to some points and using an approximation according to method.

        Args:
            points (numpy.ndarray):
            method (str): Interpolation method ['linear', 'cubic', 'nearest']
        """
        self.clear()
        with self.map_lock:
            points = cp.asarray(points, dtype=self.data_type)
            indices = transform_to_map_index(points[:, :2], self.center[:2], self.cell_n, self.resolution)
            points[:, :2] = indices.astype(points.dtype)
            points[:, 2] -= self.center[2]
            self.map_initializer(self.elevation_map, points, method)
            if self.param.dilation_size_initialize > 0:
                for i in range(2):
                    self.dilation_filter_kernel_initializer(
                        self.elevation_map[0],
                        self.elevation_map[2],
                        self.elevation_map[0],
                        self.elevation_map[2],
                        size=(self.cell_n * self.cell_n),
                    )
            self.update_upper_bound_with_valid_elevation()


if __name__ == "__main__":
    #  Test script for profiling.
    #  $ python -m cProfile -o profile.stats elevation_mapping.py
    #  $ snakeviz profile.stats

    # Get the directory of the script
    # Which should be located at: ws_dir/elevation_mapping_cupy/elevation_mapping_cupy/script/elevation_mapping.py
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Back up two directories to find the base directory of the package
    # located here: ws_dir/elevation_mapping_cupy/
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    # Navigate down to the core config directory
    # located here: ws_dir/elevation_mapping_cupy/elevation_mapping_cupy/config/core/plugin_config.yaml
    core_dir = os.path.join(base_dir, 'config', 'core')

    xp.random.seed(123)
    C_MB = xp.eye(3,3)
    b_r_MB = xp.zeros(3)
    b_r_MB[2] += 1.0
    print(C_MB, b_r_MB)

    Sigma_b_r_MB = xp.eye(3,3)*0.1
    Sigma_Theta_MB = xp.eye(3,3)*0.1
    param = Parameter(
        use_chainer=False, weight_file=os.path.join(core_dir, "weights.dat"), plugin_config_file=os.path.join(core_dir, "plugin_config.yaml"),
    )
    param.load_from_yaml(os.path.join(core_dir, "core_param.yaml"))
    param.load_from_yaml(os.path.join(core_dir, "example_setup.yaml"))
    # Override weights file location and plugin config file location that are set in the yaml file
    # Not removing there for ROS compatability
    param.weight_file = os.path.join(core_dir, "weights.dat")
    param.plugin_config_file = os.path.join(core_dir, "plugin_config.yaml")
    param.update()
    param.additional_layers = []
    # Should be front_cam
    sensor_ID = list(param.subscriber_cfg.keys())[0]
    elevation = ElevationMap(param)
    layers = [
        "elevation",
        "variance",
        # "traversability",
        "min_filter",
        "smooth",
        # "inpaint",
        # "rgb",
    ]
    # Points within a 1m.x1m.x1m. cube
    points = xp.random.rand(10000, 3 + len(param.additional_layers))
    # Set starting elevation to -1.0 meter
    points[:, 2] = -1.0

    channels = ["x", "y", "z"] + param.additional_layers
    print(channels)
    data = np.zeros((elevation.cell_n - 2, elevation.cell_n - 2), dtype=np.float32)
    for i in range(50):
        # Move points along xy direction and up to test mapping
        points[:, 0] += param.resolution*2
        points[:, 1] += param.resolution*2
        points[:, 2] += 0.1
        elevation.input_pointcloud(sensor_ID, points, channels, C_MB, b_r_MB, Sigma_b_r_MB, Sigma_Theta_MB)
        # elevation.update_normal(elevation.elevation_map[0])
        # pos = np.array([i * 0.01, i * 0.02, i * 0.01])
        # elevation.move_to(pos, R)
        for layer in layers:
            elevation.get_map_with_name_ref(layer, data)
        print(i)
        # polygon = cp.array([[0, 0], [2, 0], [0, 2]], dtype=param.data_type)
        # result = np.array([0, 0, 0])
        # elevation.get_polygon_traversability(polygon, result)
        # print(result)
    import matplotlib.pyplot as plt
    elevation.get_map_with_name_ref("elevation", data)
    # Set value of corners to 0.1 and -0.1 to make sure we can interpret the plot axes properly
    data[0:10,0:2] = 0.1
    data[0:2,0:10] = -0.1
    plt.imshow(data)
    plt.show()
