#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage

from .plugin_manager import PluginBase

from FEE import FEE
from Jacobians_xp import FEE_jacobian


class FEEIndex(PluginBase):
    """
    FEEIndex is a class that computes the Fundamental Equation of Earth (FEE) index from the semantic map.
    The FEE index is defined as the force required to shear the soil when the following parameters are fixed:
    d, rho, alpha, Q, vx, vz, s_q.
    The soil-properties are however not fixed and are determined by the semantic map and include the following:
    c, phi, gamma, delta, c_a, 
    to the elevation map. The filter is applied to the layer specified by the input_layer_name parameter.
    If the specified layer is not found, the filter is applied to the elevation layer.

    The parameter variance values are based on ranges of parameters in the network.
    WARNING: If parameter ranges change in the network normalization then these values should be updated.
             Also if the variance parameters that the network estimates is changed, these values should be updated

    Args:
        cell_n (int): The width and height of the elevation map. Default is 100.
        d (float): The depth of cut to evaluate the FEE. Default is 0.1.
        rho (float): The blade angle. Default is 80 degrees or 1.3962634 radians.
        alpha (float): The surface angle. Default is 0 degrees or 0 radians.
        Q (float): The surcharge force in KN. Default is 0.
        s_q (float): The surcharge contribution factor. Default is 1.0.
        vx (float): The horizontal velocity in m/s. Default is 1.
        vz (float): The vertical velocity in m/s. Default is 0.
    """
    def __init__(
        self,
        cell_n: int = 100,
        d: float = 0.1,
        d_var: float = 1.0e-6,
        rho: float = 1.3962634,
        rho_var: float = 1.3539e-6,
        alpha: float = 0.0,
        alpha_var: float = 1.3539e-6,
        Q: float = 0.0,
        Q_var: float = 2500.0,
        vx: float = 1.0,
        vx_var: float = 1.1111e-5,
        vz: float = 0.0,
        vz_var: float = 4.4444e-5,
        s_q: float = 1.0,
        w: float = 3.164,
        w_var: float = 1.0e-4,
        beta_var: float = 1.2219e-6,
        **kwargs
    ):
        super().__init__()
        # TODO: Ideally all of this would be closer tied to the network config and loaded from that automatically
        self.known_params_dict = {'d': d, 'rho': rho, 'alpha': alpha, 'Q': Q, 'vx': vx, 'vz': vz, 's_q': s_q, 'w': w}
        # self.known_params = ['d', 'rho', 'alpha', 'Q', 'vx', 'vz', 's_q', 'w']
        self.unknown_params = ['c', 'phi', 'gamma', 'delta', 'c_a']
        self.known_params_var_dict = {'d': d_var, 'rho': rho_var, 'alpha': alpha_var, 'Q': Q_var, 'vx': vx_var, 'vz': vz_var, 'w': w_var, 'beta': beta_var}
        # self.known_params_var = [d_var, rho_var, alpha_var, Q_var, vx_var, vz_var, w_var]
        self.unknown_params_var = ['c_var', 'phi_var', 'gamma_var', 'delta_var', 'c_a_var']
        self.metadata_FEE_param_names = ['phi', 'c', 'gamma', 'delta', 'c_a', 'alpha',
                            'rho', 'w', 'Q', 'd', 'beta', 'vx', 'vz', 's_q']
        self.metadata_FEE_param_names.sort() # Sort the list so that it is always in the same order and sort can be used elsewhere
        self.J_keys = self.metadata_FEE_param_names.copy()
        # Remove s_q as we don't currently have it in the Jacobian math because it wasn't used previously.
        # TODO: Consider adding s_q to the jacobian math to enable use of s_q_logvars. Right now can use Q_logvars instead and it should be fine.
        self.J_keys.remove('s_q')

    # TODO: Clean this up after specifying some sort of call type so that these extra arguments that aren't used are not passed
    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        semantic_new_map: cp.ndarray,
        semantic_var_params: List[str],
        updated_inds: cp.ndarray,
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (cupy._core.core.ndarray):
            semantic_layer_names (List[str]):
            rotation (float): NOT USED
            elements_to_shift (List[int]): NOT USED
            semantic_new_map (cupy._core.core.ndarray):
            semantic_var_params (List[str]):
            updated_inds (cupy._core.core.ndarray):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        # TODO: Make this do the error prop too!!!!
        for param in self.unknown_params:
            if param not in semantic_layer_names:
                raise ValueError(f"Parameter {param} is not in the semantic_var_params list.")

        # F, Fx, Fz = get_dig_difficulty_FEE_force(output['metadata'], in_metadata, self.metadata_FEE_param_names, fixed_params)
        # TODO: Make this a cuda accelerated function by implementing as a kernel
        FI_ind = plugin_layer_names.index('FEE_index')
        dtype = plugin_layers[FI_ind].dtype # TODO: Should be a member variable
        FI_var_ind = plugin_layer_names.index('FEE_index_var')
        # FI_var = cp.zeros_like(FI)
        if updated_inds is None:
            # If no indices are passed, use phi phi != 0 as a mask to indicate valid FEE params
            mask = semantic_map[semantic_layer_names.index('phi')] != 0
            xinds, yinds = cp.where(mask)
        else:
            xinds = updated_inds[:, 0]
            yinds = updated_inds[:, 1]

        # TODO: Could at least do a batch version of this instead of a for loop
        for xind, yind in zip(xinds, yinds):
            est_params = {}
            for param in self.unknown_params:
                est_params[param] = semantic_map[semantic_layer_names.index(param), xind, yind].get().item()
            est_params.update(**self.known_params_dict)
            # TODO: Replace all of this with a function that computes FEE and Jacobian error prop. at once
            # This whole approach is kinda loony because we define the FEE physics in like 3 different places in the code base
            # which is necessary because of the use of regular python types, numpy, cupy, and tensor math. They also differ slightly because
            # in training the network we enforce certain limits on the parameters and elsewhere not... This is a mess.
            # Ideally this should be consolidated into a single function that can be used for all purposes or at least a code generator that
            # handles this automatically so that mistakes don't happen and changes only need to be made manually in one location.
            plugin_layers[FI_ind, xind,yind], beta, Fx, Fz, components = FEE(**est_params, minimizer='beta_from_phi', print_ang_sum=False, clip_angsum=True, return_components=True)

            # Combine all arguments needed for FEE_jacobian into a single dictionary
            J_args = {**est_params, **components}
            # double check that delta_prime and delta are not the same because of calls to FEE
            # Convert all the values to cupy arrays
            J_args = {key: cp.array(val, dtype) for key, val in J_args.items()}
            
            # This computes the derivative of the magnitude of F with respect to the FEE parameters
            # But we want the 
            J_dict = FEE_jacobian(**J_args)
            J = cp.stack([J_dict[key] for key in self.J_keys]) # This will ensure the ordering of the keys is the same as the ordering of the params in the FEE_params_vars

            # make array of logvars for the FEE parameters and fill with the known variances and unknown variances
            # Initialize to zeros
            FEE_params_vars = cp.zeros(len(self.J_keys), dtype=dtype)
            # Fill in the known variances
            for param, var in self.known_params_var_dict.items():
                FEE_params_vars[self.J_keys.index(param)] = cp.array(var, dtype=dtype)
            # Fill in the unknown variances
            for param in self.unknown_params_var:
                var = semantic_new_map[semantic_var_params.index(param), xind, yind].get().item()
                FEE_params_vars[self.J_keys.index(param[0:-4])] = var
            plugin_layers[FI_var_ind, xind, yind] = pow(J, 2).dot(FEE_params_vars) # TODO: Just trying out to see what var looks like. Change this back after fixing plugin layers
        
        FI_inds = [FI_ind, FI_var_ind]
        # Sort the list so that the indicies are in increasing order.
        # This is necessary as the update function expects the returned array to be sorted.
        FI_inds.sort()
        return plugin_layers[FI_inds]
