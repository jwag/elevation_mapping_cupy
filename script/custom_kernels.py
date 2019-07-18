import cupy as cp
import string


def add_points_kernel(resolution, width, height, sensor_noise_factor,
                      mahalanobis_thresh, outlier_variance, wall_num_thresh,
                      enable_edge_shaped=True):

    add_points_kernel = cp.ElementwiseKernel(
            in_params='raw U p, U center_x, U center_y, raw U R, raw U t',
            out_params='raw U map, raw T newmap',
            preamble=\
            string.Template('''
            __device__ float16 clamp(float16 x, float16 min_x, float16 max_x) {

                return max(min(x, max_x), min_x);
            }
            __device__ float16 round(float16 x) {
                return (int)x + (int)(2 * (x - (int)x));
            }
            __device__ int get_xy_idx(float16 x, float16 center) {
                const float resolution = ${resolution};
                int i = round((x - center) / resolution);
                return i;
            }
            __device__ int get_idx(float16 x, float16 y, float16 center_x, float16 center_y) {
                int idx_x = clamp(get_xy_idx(x, center_x) + ${width} / 2, 0, ${width} - 1);
                int idx_y = clamp(get_xy_idx(y, center_y) + ${height} / 2, 0, ${height} - 1);
                return ${width} * idx_x + idx_y;
            }
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ float transform_p(float16 x, float16 y, float16 z,
                                         float16 r0, float16 r1, float16 r2, float16 t) {
                return r0 * x + r1 * y + r2 * z + t;
            }
            __device__ float z_noise(float16 z){
                return ${sensor_noise_factor} * z * z;
            }

            ''').substitute(resolution=resolution, width=width, height=height,
                            sensor_noise_factor=sensor_noise_factor),
            operation=\
            string.Template(
            '''
            U rx = p[i * 3];
            U ry = p[i * 3 + 1];
            U rz = p[i * 3 + 2];
            U x = transform_p(rx, ry, rz, R[0], R[1], R[2], t[0]);
            U y = transform_p(rx, ry, rz, R[3], R[4], R[5], t[1]);
            U z = transform_p(rx, ry, rz, R[6], R[7], R[8], t[2]);
            U v = z_noise(rz);
            int idx = get_idx(x, y, center_x, center_y);
            U map_h = map[get_map_idx(idx, 0)];
            U map_v = map[get_map_idx(idx, 1)];
            U num_points = newmap[get_map_idx(idx, 3)];
            if (abs(map_h - z) > (map_v * ${mahalanobis_thresh})) {
                atomicAdd(&map[get_map_idx(idx, 1)], ${outlier_variance});
            }
            else {
                if (${enable_edge_shaped} && num_points > ${wall_num_thresh} && z < map_h) { continue; }
                T new_h = (map_h * v + z * map_v) / (map_v + v);
                T new_v = (map_v * v) / (map_v + v);
                atomicAdd(&newmap[get_map_idx(idx, 0)], new_h);
                atomicAdd(&newmap[get_map_idx(idx, 1)], new_v);
                atomicAdd(&newmap[get_map_idx(idx, 2)], 1.0);
                map[get_map_idx(idx, 2)] = 1;
            }
            ''').substitute(mahalanobis_thresh=mahalanobis_thresh,
                            outlier_variance=outlier_variance,
                            wall_num_thresh=wall_num_thresh,
                            enable_edge_shaped=int(enable_edge_shaped)),
            name='add_points_kernel')
    return add_points_kernel


def error_counting_kernel(resolution, width, height, sensor_noise_factor,
                          mahalanobis_thresh, outlier_variance, traversability_inlier):

    error_counting_kernel = cp.ElementwiseKernel(
            in_params='raw U map, raw U p, U center_x, U center_y, raw U R, raw U t',
            out_params='raw U newmap, raw T error, raw T error_cnt',
            preamble=\
            string.Template('''
            __device__ float16 clamp(float16 x, float16 min_x, float16 max_x) {
                return max(min(x, max_x), min_x);
            }
            __device__ float16 round(float16 x) {
                return (int)x + (int)(2 * (x - (int)x));
            }
            __device__ int get_xy_idx(float16 x, float16 center) {
                const float resolution = ${resolution};
                int i = round((x - center) / resolution);
                return i;
            }
            __device__ int get_idx(float16 x, float16 y, float16 center_x, float16 center_y) {
                int idx_x = clamp(get_xy_idx(x, center_x) + ${width} / 2, 0, ${width} - 1);
                int idx_y = clamp(get_xy_idx(y, center_y) + ${height} / 2, 0, ${height} - 1);
                return ${width} * idx_x + idx_y;
            }
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            __device__ float transform_p(float16 x, float16 y, float16 z,
                                         float16 r0, float16 r1, float16 r2, float16 t) {
                return r0 * x + r1 * y + r2 * z + t;
            }
            __device__ float z_noise(float16 z){
                return ${sensor_noise_factor} * z * z;
            }

            ''').substitute(resolution=resolution, width=width, height=height,
                            sensor_noise_factor=sensor_noise_factor),
            operation=\
            string.Template(
            '''
            U rx = p[i * 3];
            U ry = p[i * 3 + 1];
            U rz = p[i * 3 + 2];
            U x = transform_p(rx, ry, rz, R[0], R[1], R[2], t[0]);
            U y = transform_p(rx, ry, rz, R[3], R[4], R[5], t[1]);
            U z = transform_p(rx, ry, rz, R[6], R[7], R[8], t[2]);
            U v = z_noise(rz);
            int idx = get_idx(x, y, center_x, center_y);
            U map_h = map[get_map_idx(idx, 0)];
            U map_v = map[get_map_idx(idx, 1)];
            U map_t = map[get_map_idx(idx, 3)];
            if (abs(map_h - z) < (map_v * ${mahalanobis_thresh})
                && map_v < ${outlier_variance} / 2.0
                && map_t < ${traversability_inlier}) {
                T e = z - map_h;
                atomicAdd(&error[0], e);
                atomicAdd(&error_cnt[0], 1);
                atomicAdd(&newmap[get_map_idx(idx, 3)], 1.0);
            }
            ''').substitute(mahalanobis_thresh=mahalanobis_thresh,
                            outlier_variance=outlier_variance,
                            traversability_inlier=traversability_inlier),
            name='error_counting_kernel')
    return error_counting_kernel


def average_map_kernel(width, height, max_variance, initial_variance):
    average_map_kernel = cp.ElementwiseKernel(
            in_params='raw U newmap',
            out_params='raw U map',
            preamble=\
            string.Template('''
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }
            ''').substitute(width=width, height=height),
            operation=\
            string.Template('''
            U h = map[get_map_idx(i, 0)];
            U v = map[get_map_idx(i, 1)];
            U new_h = newmap[get_map_idx(i, 0)];
            U new_v = newmap[get_map_idx(i, 1)];
            U new_cnt = newmap[get_map_idx(i, 2)];
            if (new_cnt > 0) {
                if (new_v / new_cnt > ${max_variance}) {
                    map[get_map_idx(i, 0)] = 0;
                    map[get_map_idx(i, 1)] = ${initial_variance};
                    map[get_map_idx(i, 2)] = 0;
                }
                else {
                    map[get_map_idx(i, 0)] = new_h / new_cnt;
                    map[get_map_idx(i, 1)] = new_v / new_cnt;
                    map[get_map_idx(i, 2)] = 1;
                }
            }
            ''').substitute(max_variance=max_variance,
                            initial_variance=initial_variance),
            name='average_map_kernel')
    return average_map_kernel


def dilation_filter_kernel(width, height, dilation_size):
    dilation_filter_kernel = cp.ElementwiseKernel(
            in_params='raw U map, raw U mask',
            out_params='raw U newmap',
            preamble=\
            string.Template('''
            __device__ int get_map_idx(int idx, int layer_n) {
                const int layer = ${width} * ${height};
                return layer * layer_n + idx;
            }

            __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n) {
                const int layer = ${width} * ${height};
                const int relative_idx = idx + ${width} * dy + dx;
                return layer * layer_n + relative_idx;
            }
            ''').substitute(width=width, height=height),
            operation=\
            string.Template('''
            U h = map[get_map_idx(i, 0)];
            U valid = mask[get_map_idx(i, 0)];
            newmap[get_map_idx(i, 0)] = h;
            if (valid < 0.5) {
                U distance = 100;
                U near_value = 0;
                for (int dy = -${dilation_size}; dy <= ${dilation_size}; dy++) {
                    for (int dx = -${dilation_size}; dx <= ${dilation_size}; dx++) {
                        U valid = mask[get_relative_map_idx(i, dx, dy, 0)];
                        if(valid > 0.5 && dx + dy < distance) {
                            distance = dx + dy;
                            near_value = map[get_relative_map_idx(i, dx, dy, 0)];
                        }
                    }
                }
                if(distance < 100) {
                    newmap[get_map_idx(i, 0)] = near_value;
                    // newmap[get_map_idx(i, 0)] = 10;
                }
            }
            ''').substitute(dilation_size=dilation_size),
            name='dilation_filter_kernel')
    return dilation_filter_kernel
