#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from dataclasses import dataclass, field
import pickle
import numpy as np
from simple_parsing.helpers import Serializable
from dataclasses import field
from typing import Tuple
import os
import yaml


@dataclass
class Parameter(Serializable):
    """
      This class holds the parameters for the elevation mapping algorithm.
    
    Attributes:
        resolution: The resolution in meters.
                    (Default: ``0.04``)
        subscriber_cfg: The configuration for the subscriber.
                        (Default: ``{ "front_cam": { "channels": ["rgb", "person"], "topic_name": "/elevation_mapping/pointcloud_semantic", "data_type": "pointcloud", } }``)
        additional_layers: The additional layers for the map.  
                           (Default: ``["rgb"]``)
        fusion_algorithms: The list of fusion algorithms.  
                           (Default: ``[ "image_color", "image_exponential", "pointcloud_average", "pointcloud_bayesian_inference", "pointcloud_class_average", "pointcloud_class_bayesian", "pointcloud_class_max", "pointcloud_color", ]``)
        pointcloud_channel_fusions: The fusion for pointcloud channels.  
                                   (Default: ``{"rgb": "color", "default": "class_average"}``)
        GET_channel_fusions: The fusion for GET channels.
                            (Default: ``{"default": "latest"}``)
        image_channel_fusions: The fusion for image channels.  
                               (Default: ``{"rgb": "color", "default": "exponential"}``)
        data_type: The data type for the map.  
                   (Default: ``np.float32``)
        average_weight: The weight for the average fusion.  
                        (Default: ``0.5``)
        map_length: The map's size in meters.  
                    (Default: ``8.0``)
        sensor_noise_factor: The point's noise is sensor_noise_factor*z^2 (z is distance from sensor).  
                            (Default: ``0.05``)
        mahalanobis_thresh: Points outside this number of standard deviations are considered outliers.
                            (Default: ``2.0``)
        outlier_variance: If point is outlier, add this value to the cell.  
                          (Default: ``0.01``)
        drift_compensation_variance_inlier: Cells under this value is used for drift compensation.  
                                           (Default: ``0.1``)
        variance_inflation_rate: Update the variance at this rate (m^2/s) to compensate for uncertainty due to drift.
                       (Default: ``0.01``)
        time_interval: Time layer is updated at this interval.  
                       (Default: ``0.1``)
        max_variance: The maximum variance for each cell.  
                       (Default: ``1.0``)
        dilation_size: The dilation filter size before traversability filter.  
                       (Default: ``2``)
        dilation_size_initialize: The dilation size after the init.  
                                  (Default: ``10``)
        drift_compensation_alpha: The drift compensation alpha for smoother update of drift compensation.  
                                  (Default: ``1.0``)
        traversability_inlier: Cells with higher traversability are used for drift compensation.  
                               (Default: ``0.1``)
        wall_num_thresh: If there are more points than this value, only higher points than the current height are used to make the wall more sharp.  
                         (Default: ``100``)
        min_height_drift_cnt: Drift compensation only happens if the valid cells are more than this number.  
                              (Default: ``100``)
        max_ray_length: The maximum length for ray tracing.  
                        (Default: ``2.0``)
        cleanup_step: Substitute this value from validity layer at visibility cleanup.  
                      (Default: ``0.01``)
        cleanup_cos_thresh: Substitute this value from validity layer at visibility cleanup.  
                            (Default: ``0.5``)
        min_valid_distance: Points with shorter distance will be filtered out.  
                            (Default: ``0.3``)
        max_height_range: Points higher than this value from sensor will be filtered out to disable ceiling.  
                           (Default: ``1.0``)
        ramped_height_range_a: If z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.  
                               (Default: ``0.3``)
        ramped_height_range_b: If z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.  
                               (Default: ``1.0``)
        ramped_height_range_c: If z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.  
                               (Default: ``0.2``)
        safe_thresh: If traversability is smaller, it is counted as unsafe cell.  
                     (Default: ``0.5``)
        safe_min_thresh: Polygon is unsafe if there exists lower traversability than this.  
                          (Default: ``0.5``)
        max_unsafe_n: If the number of cells under safe_thresh exceeds this value, polygon is unsafe.  
                      (Default: ``20``)
        checker_layer: Layer used for checking safety.  
                       (Default: ``"traversability"``)
        max_drift: The maximum drift for the compensation.  
                   (Default: ``0.10``)
        overlap_clear_range_xy: XY range [m] for clearing overlapped area. This defines the valid area for overlap clearance. (used for multi floor setting)  
                               (Default: ``4.0``)
        overlap_clear_range_z: Z range [m] for clearing overlapped area. Cells outside this range will be cleared. (used for multi floor setting)  
                              (Default: ``2.0``)
        enable_edge_sharpen: Enable edge sharpening.  
                             (Default: ``True``)
        enable_drift_compensation: Enable drift compensation.  
                                   (Default: ``True``)
        enable_visibility_cleanup: Enable visibility cleanup.  
                                   (Default: ``True``)
        enable_visibility_height_update: Enable updating elevation and loose soil heights using the the upper bound from the ray trace.
                                            (Default: ``False``)
        enable_overlap_clearance: Enable overlap clearance.  
                                  (Default: ``True``)
        use_only_above_for_upper_bound: Use only above for upper bound.  
                                        (Default: ``True``)
        use_chainer: Use chainer as a backend of traversability filter or pytorch. If false, it uses pytorch. Pytorch requires ~2GB more GPU memory compared to chainer but runs faster.  
                     (Default: ``True``)
        position_noise_thresh: If the position change is bigger than this value, the drift compensation happens.  
                              (Default: ``0.1``)
        orientation_noise_thresh: If the orientation change is bigger than this value, the drift compensation happens.  
                                  (Default: ``0.1``)
        plugin_config_file: Configuration file for the plugin.  
                            (Default: ``"config/plugin_config.yaml"``)
        weight_file: Weight file for traversability filter.  
                     (Default: ``"config/weights.dat"``)
        initial_variance: Initial variance for each cell.  
                          (Default: ``10.0``)
        initialized_variance: Initialized variance for each cell.  
                              (Default: ``10.0``)
        w1: Weights for the first layer.  
            (Default: ``np.zeros((4, 1, 3, 3))``)
        w2: Weights for the second layer.  
            (Default: ``np.zeros((4, 1, 3, 3))``)
        w3: Weights for the third layer.  
            (Default: ``np.zeros((4, 1, 3, 3))``)
        w_out: Weights for the output layer.  
               (Default: ``np.zeros((1, 12, 1, 1))``)
        true_map_length: True length of the map.  
                         (Default: ``None``)
        cell_n: Number of cells in the map.  
                (Default: ``None``)
        true_cell_n: True number of cells in the map.  
                     (Default: ``None``)
        move_dir_normal_weight: Weight for the normal direction of movement in GET update.
                                (Default: ``0.5``)
        swell_factor: Swell factor for the soil. Set to 1 for no swell of GET disturbed soil.
                      (Default: ``1.0``)
        spill_factor: (deprecated. Use erosion instead) Proportion of loose material moved during a sweep that is spilled.  
                      (Default: ``0.05``)
        compacted_soil_moist_unit_weight: Unit weight of compacted soil in N/m^3.  
                                          (Default: ``14635.91630769231`` 50% relative density of Vortex loam.)
        l_fit_max: Maximum distance from the blade to find surface points for surface plane fitting in FEE param calculation.
                   (Default: ``1.0``)
        surf_interp_coeff: Coefficient for exponential function weighting distance from blade for surface fitting in FEE param calculation.
                           (Default: ``3.0``)
        l_surcharge_max: Maximum distance from the blade to include loose soil in the FEE surcharge calculation.
                         (Default: ``1.0``)
        depth_weight_avg_coeff: Coefficient for depth-weighted average in FEE param calculation. Set to 0 for regular average.  
                                (Default: ``0.0``)
        max_surcharge_vol_per_unit_width: Maximum surcharge volume per unit width of GET. Set to -1 to disable limit.  
                                          (Default: ``0.3``)
        em_FEE_max_projection_dist: Maximum distance to extrapolate FEE parameters from last valid elevation mapping data.  
                                    (Default: ``0.5``)
        use_soil_erosion: Enable soil erosion calculation or not.
                            (Default: ``True``)
        soil_erosion_maximum_dt: Maximum time step for soil erosion calculation.
                                (Default: ``0.1``)
        erosion_ROI_dx: Length of region around GET center to consider for soil erosion.  
                        (Default: ``1.5``)
        erosion_ROI_dy: Width of region around GET center to consider for soil erosion.  
                        (Default: ``0.6``)
        loose_soil_cohesion: Cohesion of loose soil in Pa. Used in soil erosion calculation.
                             (Default: ``74``)
        loose_soil_phi: Internal friction angle of loose soil in radians.  Used in soil erosion calculation.
                        (Default: ``0.26``)
        soil_erosion_alpha_min: Minimum angle of slope to consider for soil erosion in radians.  
                                (Default: ``0.05``)
        use_soil_property_estimation: Use soil property estimation network.  
                                      (Default: ``False``)
        soil_prop_est_net_checkpoint_path: Path to the soil property estimation network checkpoint.  
                                           (Default: ``None``)
        bayesian_soil_wedge_exp_weight_coeff: The coefficient for the exponential weight in the Bayesian inference for the soil wedge.  
                                              (Default: ``1.0``)
        min_soil_prop_marking_dist: Minimum distance for marking soil properties.  
                                    (Default: ``0.3``)
    """
    resolution: float = 0.04  # resolution in m.
    subscriber_cfg: dict = field(
        default_factory=lambda: {
            "front_cam": {
                "channels": ["rgb", "person"],
                "topic_name": "/elevation_mapping/pointcloud_semantic",
                "data_type": "pointcloud",
                "noise_model_name": "SLS",
                "SLS_noise_model_params": {
                    "a": 6.8e-3,
                    "b": 28.0e-3,
                    "c": 38.0e-3,
                    "d": 22.0e-3,
                    "Sigma_Theta_BS_diag": [1.0e-1, 1.0e-1, 1.0e-1],
                    "Sigma_b_r_BS_diag": [1.0e-3, 1.0e-3, 1.0e-3]
                }
            }
        }
    )  # configuration for the subscriber
    additional_layers: list = field(default_factory=lambda: ["rgb"])  # additional layers for the map
    fusion_algorithms: list = field(
        default_factory=lambda: [
            "image_color",
            "image_exponential",
            "pointcloud_average",
            "pointcloud_bayesian_inference",
            "pointcloud_class_average",
            "pointcloud_class_bayesian",
            "pointcloud_class_max",
            "pointcloud_color",
            "GET_latest",
            "GET_bayesian_inference",
        ]
    )  # list of fusion algorithms
    pointcloud_channel_fusions: dict = field(default_factory=lambda: {"rgb": "color", "default": "class_average"})  # fusion for pointcloud channels
    image_channel_fusions: dict = field(default_factory=lambda: {"rgb": "color", "default": "exponential"})  # fusion for image channels
    GET_channel_fusions: dict = field(default_factory=lambda: {"default": "latest"})  # fusion for GET channels
    data_type: str = np.float32  # data type for the map
    average_weight: float = 0.5  # weight for the average fusion

    map_length: float = 8.0  # map's size in m.
    sensor_noise_factor: float = 0.05  # point's noise is sensor_noise_factor*z^2 (z is distance from sensor).
    mahalanobis_thresh: float = 2.0  # points outside this distance is outlier.
    outlier_variance: float = 0.01  # if point is outlier, add this value to the cell.
    drift_compensation_variance_inlier: float = 0.1  # cells under this value is used for drift compensation.
    variance_inflation_rate: float = 0.01  # update the variance at this rate (m^2/s) to compensate for uncertainty due to drift.
    time_interval: float = 0.1  # Time layer is updated at this interval.

    max_variance: float = 1.0  # maximum variance for each cell.
    dilation_size: int = 2  # dilation filter size before traversability filter.
    dilation_size_initialize: int = 10  # dilation size after the init.
    drift_compensation_alpha: float = 1.0  # drift compensation alpha for smoother update of drift compensation.

    traversability_inlier: float = 0.1  # cells with higher traversability are used for drift compensation.
    wall_num_thresh: int = 100  # if there are more points than this value, only higher points than the current height are used to make the wall more sharp.
    min_height_drift_cnt: int = 100  # drift compensation only happens if the valid cells are more than this number.

    max_ray_length: float = 2.0  # maximum length for ray tracing.
    cleanup_step: float = 0.01  # substitute this value from validity layer at visibility cleanup.
    cleanup_cos_thresh: float = 0.5  # substitute this value from validity layer at visibility cleanup.
    min_valid_distance: float = 0.3  # points with shorter distance will be filtered out.
    max_height_range: float = 1.0  # points higher than this value from sensor will be filtered out to disable ceiling.
    ramped_height_range_a: float = 0.3  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.
    ramped_height_range_b: float = 1.0  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.
    ramped_height_range_c: float = 0.2  # if z > max(d - ramped_height_range_b, 0) * ramped_height_range_a + ramped_height_range_c, reject.

    safe_thresh: float = 0.5  # if traversability is smaller, it is counted as unsafe cell.
    safe_min_thresh: float = 0.5  # polygon is unsafe if there exists lower traversability than this.
    max_unsafe_n: int = 20  # if the number of cells under safe_thresh exceeds this value, polygon is unsafe.
    checker_layer: str = "traversability"  # layer used for checking safety

    max_drift: float = 0.10  # maximum drift for the compensation

    overlap_clear_range_xy: float = 4.0  # xy range [m] for clearing overlapped area. this defines the valid area for overlap clearance. (used for multi floor setting)
    overlap_clear_range_z: float = 2.0  # z range [m] for clearing overlapped area. cells outside this range will be cleared. (used for multi floor setting)

    enable_edge_sharpen: bool = True  # enable edge sharpening
    enable_drift_compensation: bool = True  # enable drift compensation
    enable_visibility_cleanup: bool = True  # enable visibility cleanup
    enable_visibility_height_update: bool = False  # enable updating elevation and loose soil heights using the the upper bound from the ray trace.
    enable_overlap_clearance: bool = True  # enable overlap clearance
    use_only_above_for_upper_bound: bool = True  # use only above for upper bound
    use_chainer: bool = True  # use chainer as a backend of traversability filter or pytorch. If false, it uses pytorch. pytorch requires ~2GB more GPU memory compared to chainer but runs faster.
    position_noise_thresh: float = 0.1  # if the position change is bigger than this value, the drift compensation happens.
    orientation_noise_thresh: float = 0.1  # if the orientation change is bigger than this value, the drift compensation happens.

    plugin_config_file: str = "config/plugin_config.yaml"  # configuration file for the plugin
    weight_file: str = "config/weights.dat"  # weight file for traversability filter

    initial_variance: float = 10.0  # initial variance for each cell.
    initialized_variance: float = 10.0  # initialized variance for each cell.
    w1: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))  # weights for the first layer
    w2: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))  # weights for the second layer
    w3: np.ndarray = field(default_factory=lambda: np.zeros((4, 1, 3, 3)))  # weights for the third layer
    w_out: np.ndarray = field(default_factory=lambda: np.zeros((1, 12, 1, 1)))  # weights for the output layer

    # GET Movement parameters
    move_dir_normal_weight: float = 0.5  # Weight for the normal direction of movement in GET update.
    swell_factor: float = 1.0  # Swell factor for the soil. Set to 1 for no swell of GET disturbed soil.
    spill_factor: float = 0.05  # Proportion of loose material moved during a sweep that is spilled.
    compacted_soil_moist_unit_weight: float = 14635.91630769231  # Unit weight of compacted soil in N/m^3.
    l_fit_max: float = 1.0  # Maximum distance from the blade to find surface points for surface plane fitting in FEE param calculation.
    surf_interp_coeff: float = 3.0  # Coefficient for exponential function weighting distance from blade for surface fitting in FEE param calculation.
    l_surcharge_max: float = 1.0  # Maximum distance from the blade to include loose soil in the FEE surcharge calculation.
    depth_weight_avg_coeff: float = 0.0  # Coefficient for depth-weighted average in FEE param calculation. Set to 0 for regular average.
    max_surcharge_vol_per_unit_width: float = 0.3  # Maximum surcharge volume per unit width of GET. Set to -1 to disable limit.
    em_FEE_max_projection_dist: float = 0.5  # Maximum distance to extrapolate FEE parameters from last valid elevation mapping data.
    # Soil Erosion parameters
    use_soil_erosion: bool = True # Enable soil erosion calculation or not.
    soil_erosion_maximum_dt: float = 0.1  # Maximum time step for soil erosion calculation.
    erosion_ROI_dx: float = 1.5  # Length of region around GET center to consider for soil erosion.
    erosion_ROI_dy: float = 0.6 # Width of region around GET center to consider for soil erosion.
    # TODO: Tune these default values
    loose_soil_cohesion: float = 74.0  # Cohesion of loose soil in Pa. Used in soil erosion calculation.
    loose_soil_phi: float = 0.26  # Internal friction angle of loose soil in radians. Used in soil erosion calculation.
    soil_erosion_alpha_min: float = 0.05  # Minimum angle of slope to consider for soil erosion in radians.

    # Soil property estimation
    use_soil_property_estimation: bool = False  # use soil property estimation network
    soil_prop_est_net_checkpoint_path: str = None  # path to the soil property estimation network checkpoint
    bayesian_soil_wedge_exp_weight_coeff: float = 1.0  # the coefficient for the exponential weight in the Bayesian inference for the soil wedge
    min_soil_prop_marking_dist: float = 0.3  # minimum distance for marking soil properties

    # # not configurable params
    true_map_length: float = None  # true length of the map
    cell_n: int = None  # number of cells in the map
    true_cell_n: int = None  # true number of cells in the map

    def load_weights(self, filename: str):
        """
        Load weights from a file into the model's parameters.
        
        Args:
            filename (str): The path to the file containing the weights.
        """
        with open(filename, "rb") as file:
            weights = pickle.load(file)
            self.w1 = weights["conv1.weight"]
            self.w2 = weights["conv2.weight"]
            self.w3 = weights["conv3.weight"]
            self.w_out = weights["conv_final.weight"]

    def get_names(self):
        """
        Get the names of the parameters.
        
        Returns:
            list: A list of parameter names.
        """
        return list(self.__annotations__.keys())

    def get_types(self):
        """
        Get the types of the parameters.
        
        Returns:
            list: A list of parameter types.
        """
        return [v.__name__ for v in self.__annotations__.values()]

    def set_value(self, name, value):
        """
        Set the value of a parameter.
        
        Args:
            name (str): The name of the parameter.
            value (any): The new value for the parameter.
        """
        setattr(self, name, value)

    def get_value(self, name):
        """
        Get the value of a parameter.
        
        Args:
            name (str): The name of the parameter.
        
        Returns:
            any: The value of the parameter.
        """
        return getattr(self, name)

    def update(self):
        """
        Update the parameters related to the map size and resolution.
        """
        # +2 is a border for outside map
        self.cell_n = int(round(self.map_length / self.resolution)) + 2
        self.true_cell_n = round(self.map_length / self.resolution)
        self.true_map_length = self.true_cell_n * self.resolution
    
    def load_from_yaml(self, filename):

        with open(filename, 'r') as file:
            data = yaml.safe_load(file)

        for key, value in data.items():
            if key in self.__dict__:
                setattr(self, key, value)
            elif key == "subscribers":
                key = "subscriber_cfg"
                setattr(self, key, value)
            else:
                print(f"Key {key} not found in parameter class")


if __name__ == "__main__":
    param = Parameter()
    # Use the directory of the script to navigate to the core config directory
    # Which should be located at: ws_dir/elevation_mapping_cupy/elevation_mapping_cupy/script/elevation_mapping_cupy/parameter.py
    emcupy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    config_dir = os.path.join(emcupy_dir, 'config')
    # Navigate down to the core directory for testing
    core_dir = os.path.join(config_dir, 'core')
    filename = os.path.join(core_dir, "example_setup.yaml")
    print("Default parameter 'subscriber_cfg' ", param.subscriber_cfg)
    param.load_from_yaml(filename)
    print("Loaded parameter 'subscriber_cfg' ", param.subscriber_cfg)
    # print(param)
    # print(param.resolution)
    # param.set_value("resolution", 0.1)
    # print(param.resolution)

    # print("names ", param.get_names())
    # print("types ", param.get_types())
