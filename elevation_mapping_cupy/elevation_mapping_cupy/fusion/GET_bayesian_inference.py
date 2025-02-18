import cupy as cp
import numpy as np
import string

from .fusion_manager import FusionBase


def bayesian_inference_kernel(width, height, exp_weight_coeff):
    bayesian_inference_kernel = cp.ElementwiseKernel(
        in_params="raw T sem_map_idx, raw F FEE_param, raw F FEE_param_var,  raw F soil_wedge_weights, raw T soil_wedge_inds_x, raw T soil_wedge_inds_y",
        out_params="raw U semantic_map, raw U new_map",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int x_idx, int y_idx, int layer_n) {
                const int layer = ${width} * ${height};
                // Assuming column-major order here
                int idx = x_idx * ${width} + y_idx;
                return layer * layer_n + idx;
            }
            __device__ float exp_weight(float weight) {
                return exp(-${exp_weight_coeff} * (weight-1.0));
            }
            """
        ).substitute(width=width, height=height, exp_weight_coeff=exp_weight_coeff),
        operation=
            """
            // i here corresponds to the first index of the soil_wedge_inds and soil_wedge_weights arrays
            // Must extract index for soil_wedge_inds as cupy serializes the array
            int x_idx = soil_wedge_inds_x[i];
            int y_idx = soil_wedge_inds_y[i];
            int cell_idx = get_map_idx(x_idx, y_idx, sem_map_idx);
            U cnt = 1.0; // Assuming n=1 for now
            U feat_ml = FEE_param;
            U feat_old = semantic_map[cell_idx];
            U sigma_old = new_map[cell_idx];
            U sigma = FEE_param_var; // This is the variance of the FEE_param not the std deviation
            U sig_inflate = exp_weight(soil_wedge_weights[i]);
            // Apply the exponential weight to the sigma inflating it for low weights
            sigma = sigma * sig_inflate;

            // If the sigma_old is zero, then we have no prior information and should initialize the map
            // with the FEE_param and FEE_param_var
            // Or if the denominator for the Bayesian inference is zero, then we will also initialize the map
            // Otherwise, we have use Bayesian inference to update the map
            U feat_new = feat_ml;
            // TODO: Decide if we want to weight the sigma with the weight or not
            U sigma_new = sigma;
            if (sigma_old != 0.0 && (cnt*sigma_old+sigma) != 0.0) {
                feat_new = sigma*feat_old /(cnt*sigma_old + sigma) +cnt*sigma_old *feat_ml /(cnt*sigma_old+sigma);
                sigma_new = sigma*sigma_old /(cnt*sigma_old +sigma);
            }

            // TODO: The array assignment operations should be atomic if there are overlapping indicies also should account for n>1,
            // but assuming n=1 for now
            semantic_map[cell_idx] = feat_new;
            new_map[cell_idx] = sigma_new;
            """
        ,
        name="bayesian_inference_kernel",
    )
    return bayesian_inference_kernel

class GET_bayesian_inference(FusionBase):
    def __init__(self, params, *args, **kwargs):
        # super().__init__(fusion_params, *args, **kwargs)
        # print("Initialize fusion kernel")
        self.name = "GET_bayesian_inference"
        self.cell_n = params.cell_n
        self.resolution = params.resolution
        self.exp_weight_coeff = params.bayesian_soil_wedge_exp_weight_coeff
        self.bayesian_inference_kernel = bayesian_inference_kernel(width=self.cell_n, height=self.cell_n, exp_weight_coeff=self.exp_weight_coeff)

    # TODO: Resume here and compare to image fusion as it may line up better.
    def __call__(self, sem_map_idx, FEE_param, FEE_param_var, soil_wedge_weights, soil_wedge_inds, semantic_map, new_map):
        self.bayesian_inference_kernel(
            sem_map_idx,
            FEE_param,
            FEE_param_var,
            soil_wedge_weights,
            soil_wedge_inds[:,0],
            soil_wedge_inds[:,1],
            semantic_map,
            new_map, # NOTE: In reality hold std deviation of the layer
            size=int(soil_wedge_inds.shape[0]),
        )
