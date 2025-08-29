#!/usr/bin/env python3
from functools import partial
from collections import deque
import threading
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSPresetProfiles
# from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
import ros2_numpy as rnp
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_matrix, quaternion_conjugate, quaternion_multiply, euler_from_quaternion, quaternion_from_euler
import tf2_ros
import message_filters
from cv_bridge import CvBridge
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD
from shape_msgs.msg import Plane
from geometry_msgs.msg import PolygonStamped, Point32
from elevation_mapping_cupy import ElevationMap, Parameter
import numpy as np

PDC_DATATYPE = {
    "1": np.int8,
    "2": np.uint8,
    "3": np.int16,
    "4": np.uint16,
    "5": np.int32,
    "6": np.uint32,
    "7": np.float32,
    "8": np.float64,
}

class ElevationMappingNode(Node):
    def __init__(self):
        super().__init__(
            'elevation_mapping_node',
            automatically_declare_parameters_from_overrides=True,
            allow_undeclared_parameters=True,
            # parameter_overrides=[
            #     rclpy.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            # ]
        )
        # self.root = get_package_share_directory("elevation_mapping_cupy")
        # weight_file = self.get_parameter('weight_filee').value
        # if weight_file is None:
        #     weight_file = os.path.join(self.root, "config/core/weights.dat")
        # plugin_config_file = self.get_parameter('plugin_config_file').value
        # if plugin_config_file is None:
        #     plugin_config_file = os.path.join(self.root, "config/core/plugin_config.yaml")

        # Initialize parameters with default values Override later
        self.param = Parameter()

        # Read ROS parameters (including YAML)
        self.initialize_ros()

        self.initialize_elevation_mapping()
        self.register_subscribers()
        self.register_publishers()
        self.register_timers()
        self._last_t = None
        

        # Queues for buffering messages
        self.pointcloud_queue = deque()
        self.pointcloud_sub_key_queue = deque()
        self.image_queue = deque()
        self.image_sub_key_queue = deque()
        self.camera_info_queue = deque()
        self.msg_filter_lock = threading.Lock()

    def initialize_elevation_mapping(self) -> None:
        self._pointcloud_process_counter = 0
        self._image_process_counter = 0
        self._map = ElevationMap(self.param)
        self._map_data = np.zeros(
            (self._map.cell_n - 2, self._map.cell_n - 2), dtype=self.param.data_type
        )
        self.get_logger().info(f"Initialized map with length: {self._map.map_length}, resolution: {self._map.resolution}, cells: {self._map.cell_n}")

        self.pose_initialized = False
        self._map_t = None
    

    def process_queues(self):
        # self.get_logger().info("Starting Processing queues")
        current_time = self.get_clock().now()

        # Take a quick snapshot
        with self.msg_filter_lock:
            pointcloud_queue = list(self.pointcloud_queue)
            pointcloud_sub_key_queue = list(self.pointcloud_sub_key_queue)
            self.pointcloud_queue.clear()
            self.pointcloud_sub_key_queue.clear()

        # Now work on the local copy without locks                          # Offset for tf lookup. This is used to avoid the extrapolation errors in the GET tf and for initialization of the map
        new_pointcloud_queue = []
        new_pointcloud_sub_key_queue = []

        for msg, sub_key in zip(pointcloud_queue, pointcloud_sub_key_queue):
            msg_time = Time.from_msg(msg.header.stamp)
            age = current_time - msg_time

            if age > self.max_pointcloud_process_queue_age:
                self.get_logger().warn(f"Pointcloud message is too old. Dropping oldest message. Age: {age.nanoseconds / 1e9} seconds")
                # self.get_logger().warn(f"Pointcloud queue size: {len(pointcloud_queue)}")
                continue

            if self.try_process_pointcloud(msg, sub_key):
                # self.get_logger().info("Successfully processed pointcloud message.")
                continue  # Message processed successfully
            else:
                # Couldn't process, put it back for next time
                new_pointcloud_queue.append(msg)
                new_pointcloud_sub_key_queue.append(sub_key)
                break  # Assume we should wait for the transform before going further

        # If the buffer size is exceeded, drop the oldest messages
        if len(new_pointcloud_queue) > self.max_pointcloud_process_queue_size:
            excess = len(new_pointcloud_queue) - self.max_pointcloud_process_queue_size
            self.get_logger().warn(f"Pointcloud queue exceeded max size. Dropping {excess} oldest messages.")
            new_pointcloud_queue = new_pointcloud_queue[excess:]
            new_pointcloud_sub_key_queue = new_pointcloud_sub_key_queue[excess:]

        # If any unprocessed messages, put them back quickly
        if new_pointcloud_queue:
            # self.get_logger().info(f"Re-adding {len(new_pointcloud_queue)} unprocessed pointcloud messages to the front of the queue.")
            with self.msg_filter_lock:
                # Prepend unprocessed ones to the front
                self.pointcloud_queue.extendleft(reversed(new_pointcloud_queue))
                self.pointcloud_sub_key_queue.extendleft(reversed(new_pointcloud_sub_key_queue))
            # # Process Image messages
            # while len(self.image_queue) > self.image_queue_max_size:
            #     self.image_queue.popleft()
            #     self.image_sub_key_queue.popleft()
            #     self.camera_info_queue.popleft()

            # while self.image_queue and (current_time - self.image_queue[0].header.stamp) > self.image_queue_max_age:
            #     self.image_queue.popleft()
            #     self.image_sub_key_queue.popleft()
            #     self.camera_info_queue.popleft()

            # for _ in range(len(self.image_queue)):
            #     msg = self.image_queue.popleft()
            #     camera_info_msg = self.camera_info_queue.popleft()
            #     sub_key = self.image_sub_key_queue.popleft()
            #     if self.handle_image(msg, camera_info_msg, sub_key):
            #         continue
            #     self.image_queue.appendleft(msg)
            #     self.camera_info_queue.appendleft(camera_info_msg)
            #     self.image_sub_key_queue.appendleft(sub_key)
            #     break  # Wait for transform to become available
        # self.get_logger().info("Finished Processing queues")

    def try_process_pointcloud(self, msg: PointCloud2, sub_key: str) -> bool:
        frame_sensor_id = msg.header.frame_id
        # First get the base to sensor transform if it hasn't been set yet
        # This currently assumes the transform is fixed
        if not self._map.sensor_processors[sub_key].BS_transform_set:
            transform_base_to_sensor, error_msg = self.try_transform(
                self.base_frame,
                frame_sensor_id,
                msg.header.stamp
            )
            if transform_base_to_sensor is None:
                self.get_logger().warn(f"PointCloud callback Base to Sensor: {error_msg}")
                return False
            t = transform_base_to_sensor.transform.translation
            q = transform_base_to_sensor.transform.rotation
            B_r_BS = np.array([t.x, t.y, t.z], dtype=np.float32)
            C_BS = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
            # Set the transform
            self._map.sensor_processors[sub_key].set_BS_transform(C_BS, B_r_BS)
        # Now get the transform from the map to the base
        transform_map_to_base, error_msg = self.try_transform(
            self.map_frame,
            self.base_frame,
            msg.header.stamp
        )
        if transform_map_to_base is None:
            self.get_logger().warn(f"PointCloud callback: {error_msg}")
            return False
        # Now we can actually process the pointcloud
        self.handle_pointcloud(msg, sub_key, transform_map_to_base)

        return True

    def try_transform(self, target_frame, source_frame, lookup_time):
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                lookup_time,
                timeout=Duration(seconds=0.01)
            )
            return transform, None
        except tf2_ros.ExtrapolationException as e:
            return None, e
    
    def initialize_map(self) -> None:
        """Initialize the map with a square grid of points of size initialize_tf_grid_size*2 around
        either the base/chassis position or the world/map frame. The points are initialized at the height of the
        chassis position plus the initialize_tf_offset. The initialize_tf_offset is a list of offsets for each
        """
        # Initialize the map if enabled and if we have initialized the starting map position
        if self.use_initializer_at_start and self.pose_initialized and self.initializer_method == "points":
            # TODO: Add initialization from pointcloud here too
            points = np.zeros((0,3))
            for i, frame_id in enumerate(self.initialize_frame_id):
                transform = self.safe_lookup_transform(
                    self.map_frame,
                    frame_id,
                    self.get_clock().now()-self.tf_lookup_offset)
                t = transform.transform.translation
                p = np.array([t.x, t.y, t.z])
                p[2] += self.initialize_tf_offset[i]
                points = np.vstack((points, p))
            if points.shape[0] != 0:
                # Initialize the map with a square grid of points of size initialize_tf_grid_size*2
                square_pts = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * self.initialize_tf_grid_size
                init_points = np.zeros((0, 3))
                for p in points:
                    init_points = np.vstack((init_points, square_pts + p))
                self._map.initialize_map(init_data=init_points,
                                         init_method="points",
                                         new_variance=self.initialized_variance,
                                         points_interp_method=self.initialize_method,
                                         dilation_size_initialize=self.dilation_size_initialize) # TODO: Add rosparams for dilation_size_initialize and initialized_variance
                # Only initialize once at startup
                self.use_initializer_at_start = False
            # self._map.elevation_map[7,:] = loose_depth
        elif self.use_initializer_at_start and self.pose_initialized and self.initializer_method == "heightmap":
            raise NotImplementedError("ROS Support for Heightmap initialization not implemented yet")
            # Initialize the map with the generated heightmap
            self._map.initialize_map(init_data=self.init_heightmap,
                                     init_method="heightmap",
                                     new_variance=self._map.param.mgmt_params['initialized_variance'])
            # Only initialize once at startup
            self.use_initializer_at_start = False

    def initialize_ros(self) -> None:
        # For rolling and beyond checkout bitbots_tf2 from bitbots_tf_buffer import Buffer
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self, spin_thread=True)
        self._last_update_time_t = None
        self._last_update_variance_t = None
        self.set_param_values_from_ros()
        self.get_ros_params()
    
    def get_ros_params(self) -> None:
        # TODO: get rid of this and shove into self._map.parm.mgmt_params instead
        ''' These are parameters not in the parameter class but are used in the node.
        Commented out parameters are not used in the current python node but are left here for future
        improvements as they are used in the C++ node.
        '''
        self.initialize_method = self.get_parameter('initialize_method').get_parameter_value().string_value
        self.initialize_frame_id = self.get_parameter('initialize_frame_id').get_parameter_value().string_array_value
        self.initialize_tf_offset = self.get_parameter('initialize_tf_offset').get_parameter_value().double_array_value
        self.initialize_tf_grid_size = self.get_parameter('initialize_tf_grid_size').get_parameter_value().double_value
        self.use_initializer_at_start = self.get_parameter('use_initializer_at_start').get_parameter_value().bool_value
        self.initializer_method = self.get_parameter('initializer_method').get_parameter_value().string_value
        self.dilation_size_initialize = self.get_parameter('dilation_size_initialize').get_parameter_value().integer_value
        self.initialized_variance = self.get_parameter('initialized_variance').get_parameter_value().double_value
        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        # self.corrected_map_frame = self.get_parameter('corrected_map_frame').get_parameter_value().string_value
        # self.position_lowpass_alpha = self.get_parameter('position_lowpass_alpha').get_parameter_value().double_value
        # self.orientation_lowpass_alpha = self.get_parameter('orientation_lowpass_alpha').get_parameter_value().double_value
        # self.recordable_fps = self.get_parameter('recordable_fps').get_parameter_value().double_value
        self.update_variance_fps = self.get_parameter('update_variance_fps').get_parameter_value().double_value
        self.time_interval = self.get_parameter('time_interval').get_parameter_value().double_value
        self.update_pose_fps = self.get_parameter('update_pose_fps').get_parameter_value().double_value
        self.GET_tf_fps = self.get_parameter('GET_tf_fps').get_parameter_value().double_value
        # TODO: THis timer rate should be a param
        tf_lookup_offset = self.get_parameter('tf_lookup_offset').get_parameter_value().double_value
        self.tf_lookup_offset = rclpy.time.Duration(seconds=tf_lookup_offset)
        self.process_queue_fps = self.get_parameter('process_queue_fps').get_parameter_value().double_value
        self.max_pointcloud_process_queue_size = self.get_parameter('max_pointcloud_process_queue_size').get_parameter_value().integer_value
        max_pointcloud_process_queue_age = self.get_parameter('max_pointcloud_process_queue_age').get_parameter_value().double_value
        self.max_pointcloud_process_queue_age = rclpy.time.Duration(seconds=max_pointcloud_process_queue_age)
        # # self.map_acquire_fps = self.get_parameter('map_acquire_fps').get_parameter_value().double_value
        # self.publish_statistics_fps = self.get_parameter('publish_statistics_fps').get_parameter_value().double_value
        # self.enable_pointcloud_publishing = self.get_parameter('enable_pointcloud_publishing').get_parameter_value().bool_value
        # self.enable_normal_arrow_publishing = self.get_parameter('enable_normal_arrow_publishing').get_parameter_value().bool_value
        # self.enable_drift_corrected_TF_publishing = self.get_parameter('enable_drift_corrected_TF_publishing').get_parameter_value().bool_value

    def get_dict_parameters(self, ros_param_name: str):
        # Get dictionary type parameters from ROS parameters
        # In ROS, these parameters are nested with a '.'
        # the ros_param_name is to allow for the ros parameter name to be different that the name of the attribute in the parameter class
        # e.g. the ros parameter name is "subscribers" but the attribute name is "subscriber_cfg"
        ros_dict_param = self.get_parameters_by_prefix(ros_param_name)
        if len(ros_dict_param) == 0:
            return None
        else:
            param_dict = {}
            for p_name, param_value in ros_dict_param.items():
                parts = p_name.split('.')
                current_level = param_dict
                for part in parts[:-1]:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                current_level[parts[-1]] = param_value.value
            return param_dict

    def set_param_values_from_ros(self):
        for name in self.param.get_names():
            p_type = type(self.param.get_value(name))
            if p_type == dict:
                # Subscribers is special case because the param name and the key in the dict are different
                if name == "subscriber_cfg":
                    param = self.get_dict_parameters("subscribers")
                else:
                    param = self.get_dict_parameters(name)
                
                if param is None:
                    self.get_logger().warn(f"Parameter dictionary {name} not set as ROS parameter. Using default value.")
                else:
                    self.param.set_value(name, param)
            else:
                param = self.get_parameter(name)
                if param.type_ == rclpy.parameter.Parameter.Type['NOT_SET']:
                    self.get_logger().warn(f"Parameter {name} not set as ROS parameter. Using default value.")
                else:
                    self.param.set_value(name, param.value)

        self.my_publishers = self.get_dict_parameters("publishers")
        # Update the computed parameters
        self.param.update()


    def register_subscribers(self) -> None:
        if any(config.get("data_type") == "image" for config in self.param.subscriber_cfg.values()):
            self.cv_bridge = CvBridge()

        self._pointcloud_subs = {}
        self._image_subs = {}
        self._GET_subs = {}
        self._GET_subs_history = {}

        for key, config in self.param.subscriber_cfg.items():
            data_type = config.get("data_type")
            if data_type == "image":
                topic_name_camera = config.get("topic_name_camera", "/camera/image")
                topic_name_camera_info = config.get("topic_name_camera_info", "/camera/camera_info")
                camera_sub = message_filters.Subscriber(
                    self,
                    Image,
                    topic_name_camera
                )
                camera_info_sub = message_filters.Subscriber(
                    self,
                    CameraInfo,
                    topic_name_camera_info
                )
                image_sync = message_filters.ApproximateTimeSynchronizer(
                    [camera_sub, camera_info_sub], queue_size=10, slop=0.5
                )
                image_sync.registerCallback(partial(self.image_callback, sub_key=key))
                self._image_subs[key] = image_sync
            elif data_type == "pointcloud":
                topic_name = config.get("topic_name", "/pointcloud")
                # qos_profile = rclpy.qos.QoSProfile(
                #     depth=10,
                #     reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                #     durability=rclpy.qos.DurabilityPolicy.VOLATILE,
                #     history=rclpy.qos.HistoryPolicy.KEEP_LAST
                # )
                qos_profile = QoSPresetProfiles.get_from_short_key("sensor_data")
                # qos_profile = rclpy.qos.QoSProfile(depth=10)
                # qos_profile = 10
                subscription = self.create_subscription(
                    PointCloud2,
                    topic_name,
                    partial(self.pointcloud_callback, sub_key=key),
                    qos_profile
                )
                self._pointcloud_subs[key] = subscription
            elif data_type == "GET":
                topic_name = config.get("topic_name", "/odom_blade")
                use_blade_odometry = config.get("use_blade_odometry", True)
                if use_blade_odometry:
                    subscription = self.create_subscription(
                    Odometry,
                    topic_name,
                    partial(self.GET_odometry_callback, sub_key=key),
                    10
                    )
                    self._GET_subs[key] = subscription
                else:
                    # Setup timer callback to get the transform from the tf2 buffer
                    # Probably bad to shove this in a subs array, but this isn't used so I think its ok for now
                    self._GET_subs[key] = self.create_timer(
                        1.0 / self.GET_tf_fps,
                        partial(self.GET_tf_callback, sub_key=key)
                    )

                min_max_translation_m = self.param.resolution * np.sqrt(2.0)
                if config['max_translation_m'] < min_max_translation_m:
                    # This also leads to issues with erosion slipping under the blade
                    self.get_logger().warn(f"Maximum translation distance for subscriber '{key}' is less than sqrt(2) times the resolution of the map. This could cause FEE width and surcharge calculation issues. Setting to {min_max_translation_m} m.")
                    config['max_translation_m'] = min_max_translation_m
                # if config['max_rotation_deg'] < 45.0:
                #     # This also leads to issues with erosion slipping under the blade
                #     self.get_logger().warn(f"Maximum rotation angle for subscriber '{key}' is less than 45 degrees. This causes erosion issues. Setting to 45 degrees.")
                #     config['max_rotation_deg'] = 45.0
                # For help in determining when to process the GET movement
                self._GET_subs_history[key] = self.GET_history(em_node=self, GET_config=config)

    def register_publishers(self) -> None:
        self._publishers_dict = {}
        self._publishers_timers = []
        sensor_qos = QoSPresetProfiles.get_from_short_key("sensor_data")

        for pub_key, pub_config in self.my_publishers.items():
            topic_name = f"/{self.get_name()}/{pub_key}"
            publisher = self.create_publisher(GridMap, topic_name, sensor_qos)
            self._publishers_dict[pub_key] = publisher

            fps = pub_config.get("fps", 1.0)
            timer = self.create_timer(
                1.0 / fps,
                partial(self.publish_map, key=pub_key)
            )
            self._publishers_timers.append(timer)
        
        # Also publish plane fit parameters
        self.plane_publisher = self.create_publisher(Plane, "smooth_plane_fit", sensor_qos)
        self.polygon_publisher = self.create_publisher(PolygonStamped, "smooth_plane_corners", sensor_qos)

    def register_timers(self) -> None:
        pose_fps = self.update_pose_fps
        if pose_fps <= 0.0:
            # If pose_fps is 0.0, then we will call the pose_update at a low frequency
            # but internally it will only update the pose once the map is initialized
            pose_fps = 1.0
        self.time_pose_update = self.create_timer(
            1.0 / pose_fps,
            self.pose_update
        )
        self.timer_variance = self.create_timer(
            1.0 / self.update_variance_fps,
            self.update_variance
        )
        self.timer_time = self.create_timer(
            self.time_interval,
            self.update_time
        )

        # Timer to process buffered messages
        self.process_queue_timer = self.create_timer(1.0/self.process_queue_fps, self.process_queues)

    def publish_map(self, key: str) -> None:
        if not self.pose_initialized:
            return
        gm = GridMap()
        gm.header.frame_id = self.map_frame
        gm.header.stamp = self.get_clock().now().to_msg()
        gm.info.resolution = self._map.resolution
        gm.info.length_x = self._map.map_length
        gm.info.length_y = self._map.map_length
        gm.info.pose.position.x = float(self._map_t[0,0])
        gm.info.pose.position.y = float(self._map_t[1,0])
        gm.info.pose.position.z = float(self._map_t[2,0])
        gm.info.pose.orientation.w = 1.0
        gm.info.pose.orientation.x = 0.0
        gm.info.pose.orientation.y = 0.0
        gm.info.pose.orientation.z = 0.0
        gm.layers = []
        gm.basic_layers = self.my_publishers[key]["basic_layers"]

        for layer in self.my_publishers[key].get("layers", []):
            gm.layers.append(layer)
            self._map.get_map_with_name_ref(layer, self._map_data)
            arr = Float32MultiArray()
            arr.layout = MAL()
            N = self._map_data.shape[0]
            M = self._map_data.shape[1]
            arr.layout.dim.append(MAD(label="column_index", size=N, stride=int(N * M)))
            arr.layout.dim.append(MAD(label="row_index", size=M, stride=1))
            arr.data = np.flip(self._map_data, axis=[0, 1]).flatten(order='F').tolist()
            gm.data.append(arr)

        gm.outer_start_index = 0
        gm.inner_start_index = 0
        self._publishers_dict[key].publish(gm)

    def safe_lookup_transform(self, target_frame, source_frame, time):
        try:
            # Use the provided time for the lookup
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time,
                # timeout=Duration(seconds=0.02), # TODO: Make this a parameter and decide where we would want to use it.
            )
        except tf2_ros.ExtrapolationException as e:
            # Fallback to the current time using the node's clock
            self.get_logger().warn(f"ExtrapolationException encountered: {str(e)}")
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time=rclpy.time.Time(), # latest available transform
                # self.get_clock().now(),
                # timeout=Duration(seconds=5.0)
            )

    def image_callback(self, camera_msg: Image, camera_info_msg: CameraInfo, sub_key: str) -> None:
        with self.msg_filter_lock:
            self.image_queue.append(camera_msg)
            self.camera_info_queue.append(camera_info_msg)
            self.image_sub_key_queue.append(sub_key)
    
    def handle_image(self, camera_msg: Image, camera_info_msg: CameraInfo, sub_key: str) -> None:
        self._last_t = camera_msg.header.stamp
        try:
            semantic_img = self.cv_bridge.imgmsg_to_cv2(camera_msg, desired_encoding="passthrough")
        except:
            return
        if len(semantic_img.shape) != 2:
            semantic_img = [semantic_img[:, :, k] for k in range(semantic_img.shape[2])]
        else:
            semantic_img = [semantic_img]

        K = np.array(camera_info_msg.k, dtype=np.float32).reshape(3, 3)
        D = np.array(camera_info_msg.d, dtype=np.float32).reshape(-1, 1)

        transform_camera_to_map = self.safe_lookup_transform(
            self.map_frame,
            camera_msg.header.frame_id,
            camera_msg.header.stamp
        )
        t = transform_camera_to_map.transform.translation
        q = transform_camera_to_map.transform.rotation
        t_np = np.array([t.x, t.y, t.z], dtype=np.float32)
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
        self._map.input_image(
            sub_key, semantic_img, R, t_np, K, D,
            camera_info_msg.height, camera_info_msg.width
        )
        self._image_process_counter += 1


    # Temporarily replacing the function in ros2_numpy
    # pts = rnp.point_cloud2.get_xyz_points(points)
    def get_xyz_points(self, cloud_array, remove_nans=True, dtype=float):
        '''Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
        '''
        # remove crap points
        mask = None
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & \
                np.isfinite(cloud_array['y']) & \
                np.isfinite(cloud_array['z'])
            # cloud_array = np.where(mask[..., None], cloud_array, np.nan)
            cloud_array = cloud_array[mask]

        # pull out x, y, and z values
        points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        points[...,0] = cloud_array['x']
        points[...,1] = cloud_array['y']
        points[...,2] = cloud_array['z']

        return points, mask


    def pointcloud_callback(self, msg: PointCloud2, sub_key: str) -> None:
        with self.msg_filter_lock:
            self.pointcloud_queue.append(msg)
            self.pointcloud_sub_key_queue.append(sub_key)
    
    def handle_pointcloud(self, msg: PointCloud2, sub_key: str, transform_map_to_base) -> None:
        self._last_t = msg.header.stamp
        # self.get_logger().info(f"Received pointcloud with {msg.width} points")
        additional_channels = self.param.subscriber_cfg[sub_key].get("channels", [])
        channels = ["x", "y", "z"] + additional_channels
        try:
            points = rnp.numpify(msg)
        except:
            return
        # if points['xyz'].size == 0:
        if points['x'].size == 0:
            return
        t = transform_map_to_base.transform.translation
        q = transform_map_to_base.transform.rotation
        B_r_MB = np.array([t.x, t.y, t.z], dtype=np.float32)
        C_MB = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)

        pts, nan_mask = self.get_xyz_points(points, remove_nans=True) # Old way the points, trying this below
        # pts = points['xyz']
        # TODO: This is probably expensive. Consider modifying rnp or input_pointcloud()
        # Append additional channels to pts
        for channel in additional_channels:
            # if channel in points.keys():
            if channel in points.dtype.names:
                data = points[channel][nan_mask]
                if data.ndim == 1:
                    data = data[:, np.newaxis]
                pts = np.hstack((pts, data))
        # Adding static noise value for now to trigger alignment
        Sigma_b_r_MB = np.zeros((3, 3), dtype=np.float32)
        Sigma_b_r_MB[2, 2] = 0.1
        self._map.input_pointcloud(sensor_ID=sub_key,
                                   raw_points=pts,
                                   channels=channels,
                                   C_MB=C_MB, B_r_MB=B_r_MB,
                                   Sigma_b_r_MB=Sigma_b_r_MB,
                                   Sigma_Theta_MB=None)
        self._pointcloud_process_counter += 1
    
    class GET_history():
        """Class to aid in monintoring the change in GET pose and twist over time to enable
        calling the input_GET_movement function only when the GET movement of the get meets
        certain criteria.
        """
        def __init__(self, em_node: 'ElevationMappingNode', GET_config: dict):
            self.em_node = em_node
            self.GET_config = GET_config
            self.stamp_float = None
            self.position = None
            self.orientation = None
            self.var_z = None
            # Indicates the direction of movment with repsect to the blade normal vector
            self.movement_dir = None

        @classmethod
        def GET_from_msg(cls, em_node: 'ElevationMappingNode', msg: Odometry, GET_config: dict) -> 'ElevationMappingNode.GET_history':
            # Make sure the frame id is in the map_frame
            if msg.header.frame_id != em_node.map_frame:
                em_node.get_logger().warn(f"Odometry frame id {msg.header.frame_id} does not match map frame id {em_node.map_frame}. Not processing GET Movement.")
                return
            if msg.child_frame_id != GET_config["blade_frame"]:
                em_node.get_logger().warn(f"Odometry child frame id {msg.child_frame_id} does not match blade frame id {GET_config['blade_frame']}. Not processing GET Movement.")
                return
            # Create an instance of GET_history and populate it with data from the message
            instance = cls(em_node=em_node, GET_config=GET_config)
            stamp = msg.header.stamp
            instance.stamp_float = stamp.sec + stamp.nanosec * 1e-9 
            pos = msg.pose.pose.position
            instance.position = np.array([pos.x, pos.y, pos.z])
            instance.var_z = msg.pose.covariance[14]
            orientation = msg.pose.pose.orientation
            # Using the trasnformations convention of [x, y, z, w]
            instance.orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
            # Sign of the X direction velocity in the child_frame_id=blade_frame will tell us if the GET is moving forward or backward
            # WARNING: This means that the velocity of this message should be relatively smooth as noise could cause the sign to change frequently
            instance.movement_dir = np.sign(msg.twist.twist.linear.x)
            # TODO: Only currently supporting a child frame id that is the same as the blade frame id. Could maybe support otherwise later
            # Using a tf lookup to get the normal vector of the blade and .. Keeping below code for later in case it comes in handy
            # Project the velocity into the map frame and pull out the x and y components
            # Only apply the rotation not the translation as this is a velocity
            # transform_child_to_map = em_node.safe_lookup_transform(msg.child_frame_id, em_node.map_frame, em_node.map_frame)
            # rot_q = transform_child_to_map.transform.rotation
            # rot = quaternion_matrix([rot_q.x, rot_q.y, rot_q.z, rot_q.w])[:3, :3]
            # linear = msg.twist.twist.linear
            # vel = np.array([linear.x, linear.y, linear.z])
            # xy_vel = np.dot(rot, vel)[:2]
            return instance
        
        def assign_from_instance(self, instance: 'ElevationMappingNode.GET_history'):
            # Assign the values from the instance to the current instance
            self.em_node = instance.em_node
            self.GET_config = instance.GET_config
            self.stamp_float = instance.stamp_float
            self.position = instance.position
            self.orientation = instance.orientation
            self.var_z = instance.var_z
            self.movement_dir = instance.movement_dir

        def check_movement(self, msg: Odometry):
            # Update is a flag that will be set to True if the GET has moved more than a specified distance,
            # the maximum amount of time between processings has passed, the orientation has changed more than
            # a specified amount, or the sign of the dot product between the xy velocity in the map frame 
            # with the GET normal vector changes sign indicating a change in direction. If any of these conditions
            # are met then this indicates that the movement should be used to update the map.
            update = False
            # Use the first message to set the initial values
            if self.stamp_float is None:
                GET_hist = self.GET_from_msg(self.em_node, msg, self.GET_config)
                self.assign_from_instance(GET_hist)
                GET_curr = None
            else:
                # Check if the GET has moved more than a specified distance
                GET_curr = self.GET_from_msg(self.em_node, msg, self.GET_config)
                dist = np.linalg.norm(GET_curr.position - self.position)
                if dist > self.GET_config["max_translation_m"]:
                    self.em_node.get_logger().info(f"GET has moved {dist} meters. Triggering update.")
                    update = True
                # Check if the orientation has changed more than a specified amount
                # See: https://www.mathworks.com/help/driving/ref/quaternion.dist.html
                q1 = self.orientation
                q2 = GET_curr.orientation
                q_rel = quaternion_multiply(quaternion_conjugate(q1), q2)
                # TODO: Redo this with pure quaternion math. project vectors onto plane of blade and then take the dot product
                #       see : https://math.stackexchange.com/a/357991
                # Only use pitch and yaw. Ignore roll because that is rotation about the blade plane
                roll, pitch, yaw = euler_from_quaternion(q_rel, axes='sxyz')
                q_rel_new = quaternion_from_euler(0.0, pitch, yaw, axes='sxyz')
                angle_deg = 2 * np.arccos(np.abs(q_rel_new[3])) * 180 / np.pi
                if angle_deg > self.GET_config["max_rotation_deg"]:
                    self.em_node.get_logger().info(f"GET has rotated {angle_deg} degrees. Triggering update.")
                    update = True
                # Check if we have changed direction
                # if GET_curr.movement_dir != self.movement_dir:
                #     self.em_node.get_logger().info(f"GET has changed direction. Triggering update.")
                #     update = True
                # Check if the maximum amount of time between processings has passed
                # This was originally to ensure maximum sweep length was not exceeded for feeding
                # to the soil property estimation network. However, when vehicle is stationary
                # this leads to issues so could disable or could modify it so that the original
                # transform is kept, but the timestep is reset so that when the system does move,
                # the sweep length is not exceeded.
                dt = GET_curr.stamp_float - self.stamp_float
                if not update and dt > self.GET_config["max_time_s"]:
                    # self.em_node.get_logger().info(f"GET map update has not been applied for {dt} seconds. Advancing timestep of GET history.")
                    self.stamp_float = GET_curr.stamp_float
            return update, GET_curr
        
        def get_transform(self):
            T_MG = quaternion_matrix(self.orientation).astype(np.float32)
            T_MG[:3, 3] = self.position
            return T_MG
    
    def GET_odometry_callback(self, msg: Odometry, sub_key: str) -> None:
        # self.get_logger().info(f"Received GET odometry message for {sub_key}")
        self._last_t = msg.header.stamp
        GET_hist = self._GET_subs_history[sub_key]
        update, GET_curr = GET_hist.check_movement(msg)
        if update and self.pose_initialized:
            # self.get_logger().info(f"Processing GET movement for {sub_key}")
            T_MG0 = GET_hist.get_transform()
            T_MG1 = GET_curr.get_transform()
            # Average the variance of the two messages for now
            var_h = (GET_hist.var_z + GET_curr.var_z) / 2
            # TODO: provide some sort of interpolation here using tf2
            # Pull out translation from T_MG0 and T_MG1 and append
            M_r_MG = np.array([T_MG0[:3, 3], T_MG1[:3, 3]]).astype(np.float32)
            n_steps = 2
            dt = GET_curr.stamp_float - GET_hist.stamp_float
            roll = 0.1 # TODO: could probably more efficiently obtain roll here than the internal implementation
            FEE_em_params, surf_points_dict, plane_fit_params = self._map.input_GET_movement(
                GET_ID=sub_key,
                T_MG0=T_MG0,
                T_MG1=T_MG1,
                M_r_MG=M_r_MG,
                n_steps=n_steps,
                var_h=var_h,
                roll=roll
            )
            # Update the history with the current message
            GET_hist.assign_from_instance(GET_curr)
            # Publish the plane fit
            self.publish_plane_fit(plane_fit_params)


    def odom_msg_from_tf(self, from_frame: str, to_frame: str, stamp: rclpy.time.Time) -> Odometry:
        # Get the transform from the tf2 buffer
        transform = self.safe_lookup_transform(from_frame, to_frame, stamp)
        odom_msg = Odometry()
        odom_msg.header.frame_id = from_frame
        odom_msg.child_frame_id = to_frame
        odom_msg.header.stamp = stamp.to_msg()
        odom_msg.pose.pose.position.x = transform.transform.translation.x
        odom_msg.pose.pose.position.y = transform.transform.translation.y
        odom_msg.pose.pose.position.z = transform.transform.translation.z
        odom_msg.pose.pose.orientation.x = transform.transform.rotation.x
        odom_msg.pose.pose.orientation.y = transform.transform.rotation.y
        odom_msg.pose.pose.orientation.z = transform.transform.rotation.z
        odom_msg.pose.pose.orientation.w = transform.transform.rotation.w
        # Don't fill out the twist for now as we aren't currently using it for the check movement
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        # TODO: Fill out the covariance here and use param to set vallues
        odom_msg.pose.covariance = [
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 100.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, (np.pi / 2) ** 2, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, (np.pi / 2) ** 2, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, (np.pi / 2) ** 2
        ]
        odom_msg.twist.covariance = [
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 100.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 100.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, (np.pi / 2) ** 2, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, (np.pi / 2) ** 2, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, (np.pi / 2) ** 2
        ]
        return odom_msg

    def GET_tf_callback(self, sub_key: str) -> None:
        # self.get_logger().info(f"Received GET odometry message for {sub_key}")
        # Adding a small delay to the last time to avoid tf2 lookup issues
        self._last_t = self.get_clock().now() - self.tf_lookup_offset
        GET_hist = self._GET_subs_history[sub_key]
        try:
            msg = self.odom_msg_from_tf(
                from_frame=self.map_frame,
                to_frame=GET_hist.GET_config["blade_frame"],
                stamp=self._last_t
            )
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"LookupException encountered: {str(e)}")
            return
        update, GET_curr = GET_hist.check_movement(msg)
        if update and self.pose_initialized:
            # self.get_logger().info(f"Processing GET movement for {sub_key}")
            T_MG0 = GET_hist.get_transform()
            T_MG1 = GET_curr.get_transform()
            # Average the variance of the two messages for now
            var_h = (GET_hist.var_z + GET_curr.var_z) / 2
            # TODO: provide some sort of interpolation here using tf2
            # Pull out translation from T_MG0 and T_MG1 and append
            M_r_MG = np.array([T_MG0[:3, 3], T_MG1[:3, 3]]).astype(np.float32)
            n_steps = 2
            dt = GET_curr.stamp_float - GET_hist.stamp_float
            roll = 0.1 # TODO: could probably more efficiently obtain roll here than the internal implementation
            FEE_em_params, surf_points_dict, plane_fit_params = self._map.input_GET_movement(
                GET_ID=sub_key,
                T_MG0=T_MG0,
                T_MG1=T_MG1,
                M_r_MG=M_r_MG,
                n_steps=n_steps,
                var_h=var_h,
                roll=roll
            )
            # Update the history with the current message
            GET_hist.assign_from_instance(GET_curr)
            # Publish the plane fit
            self.publish_plane_fit(plane_fit_params)
    
    def publish_plane_fit(self, plane_fit_params: dict) -> None:
        # Publish shape_msgs/Plane message
        # Note: These are in map frame
        plane_msg = Plane()
        coeffs = plane_fit_params["coeffs"] # a, b, c, d
        plane_msg.coef = coeffs.tolist()

        self.plane_publisher.publish(plane_msg)

        # Publish geometry_msgs/PolygonStamped message
        polygon_msg = PolygonStamped()
        polygon_msg.header.frame_id = self.map_frame
        polygon_msg.header.stamp = self.get_clock().now().to_msg()
        for point in plane_fit_params["points"]:
            pt = Point32()
            pt.x, pt.y, pt.z = point
            polygon_msg.polygon.points.append(pt)

        self.polygon_publisher.publish(polygon_msg)

    def pose_update(self) -> None:
        if not self.pose_initialized and self.update_pose_fps <= 0.0:
            # Obtain the discretized map position. Should be zero in this case
            self._map_t = self._map.get_position()
            self.pose_initialized = True
        # If pose_fps is 0.0, then we only want to use the pose update to initialize the map
        elif (not self.pose_initialized) or (self.update_pose_fps > 0.0):
            # self.get_logger().info("Updating pose")
            # Don't update the pose if we haven't received any data yet unless we are initializing the map
            if self._last_t is None and self.pose_initialized:
                pass
                return
            elif self._last_t is None:
                stamp = self.get_clock().now()
                # Wait until we have received a pointcloud or image to initialize the map
                return
            else:
                stamp = self._last_t
            transform = self.safe_lookup_transform(
                self.map_frame,
                self.base_frame,
                stamp
            )
            t = transform.transform.translation
            q = transform.transform.rotation
            trans = np.array([t.x, t.y, t.z], dtype=np.float32)
            rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3].astype(np.float32)
            self._map.move_to(trans, rot)
            # Obtain the discretized map position
            self._map_t = self._map.get_position()
            # TODO: Could remove self.time_pose_update after initialzation
            self.pose_initialized = True
            # Initialize the map if configured to do so
        self.initialize_map()

    def update_variance(self) -> None:
        t2 = self.get_clock().now()
        if self._last_update_variance_t is not None:
            dt = (t2 - self._last_update_variance_t).nanoseconds / 1e9
            self._map.update_variance(dt)
        self._last_update_variance_t = t2

    def update_time(self) -> None:
        t2 = self.get_clock().now()
        if self._last_update_time_t is not None:
            dt = (t2 - self._last_update_time_t).nanoseconds / 1e9
            self._map.update_time(dt)
        self._last_update_time_t = t2

    def destroy_node(self) -> None:
        super().destroy_node()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ElevationMappingNode()
    # executor = rclpy.executors.SingleThreadedExecutor()
    # Fixes tf extrapolation issues in sim right now
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
