import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def exponential_correspondences_to_map_kernel():
    exponential_correspondences_to_map_kernel = cp.ElementwiseKernel(
        in_params="raw U sem_map, raw U map_idx, raw U image_mono, raw U uv_correspondence, raw B valid_correspondence, raw U image_height, raw U image_width",
        "raw U sem_map_idx, raw F FEE_param, raw U soil_wedge_inds, raw F soil_wedge_weights, semantic_map, new_map"
        out_params="raw U new_sem_map",
        preamble=string.Template(
            """
            """
        ),
        operation=string.Template(
            """
            // i here corresponds to the index of the soil_wedge_inds and soil_wedge_weights arrays
            if (valid_correspondence[cell_idx]){
                int cell_idx_2 = get_map_idx(i, 1);
                int idx = int(uv_correspondence[cell_idx]) + int(uv_correspondence[cell_idx_2]) * image_width; 
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)] * (1-${alpha}) +  ${alpha} * image_mono[idx];
            }else{
                new_sem_map[get_map_idx(i, map_idx)] = sem_map[get_map_idx(i, map_idx)];
            }

            """
        ),
        name="exponential_correspondences_to_map_kernel",
    )
    return exponential_correspondences_to_map_kernel

class Latest(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "GET_latest"
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.latest_kernel = latest_kernel(
            resolution=self.resolution, width=self.cell_n, height=self.cell_n,
        )

    # TODO: Resume here and compare to image fusion as it may line up better.
    def __call__(self, sem_map_idx, FEE_param, soil_wedge_inds, soil_wedge_weights, semantic_map, new_map):
        self.latest_kernel(
            sem_map_idx,
            FEE_param,
            soil_wedge_inds,
            soil_wedge_weights,
            semantic_map,
            new_map,
            size=(soil_wedge_inds.shape[0]),
        )
