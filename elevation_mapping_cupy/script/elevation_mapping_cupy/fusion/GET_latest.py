import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def latest_kernel(width, height):
    latest_kernel = cp.ElementwiseKernel(
        in_params="raw T sem_map_idx, raw U FEE_param, raw T soil_wedge_inds_x, raw T soil_wedge_inds_y",
        out_params="raw U semantic_map",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int x_idx, int y_idx, int layer_n) {
                const int layer = ${width} * ${height};
                // Assuming column-major order here
                int idx = x_idx * ${width} + y_idx;
                return layer * layer_n + idx;
            }
            """
        ).substitute(width=width, height=height),
        operation=
            """
            // i here corresponds to the first index of the soil_wedge_inds array
            // Must extract index for soil_wedge_inds as cupy serializes the array
            int x_idx = soil_wedge_inds_x[i];
            int y_idx = soil_wedge_inds_y[i];
            int cell_idx = get_map_idx(x_idx, y_idx, sem_map_idx);
            // TODO: This should be atomic if there are overlapping indicies
            semantic_map[cell_idx] = FEE_param
            """
        ,
        name="latest_kernel",
    )
    return latest_kernel

class Latest(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "GET_latest"
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.latest_kernel = latest_kernel(width=self.cell_n, height=self.cell_n)

    # TODO: Resume here and compare to image fusion as it may line up better.
    def __call__(self, sem_map_idx, FEE_param, FEE_param_sigma, soil_wedge_weights, soil_wedge_inds, semantic_map, new_map):
        self.latest_kernel(
            sem_map_idx,
            FEE_param,
            soil_wedge_inds[:,0],
            soil_wedge_inds[:,1],
            semantic_map,
            size=int(soil_wedge_inds.shape[0]),
        )
