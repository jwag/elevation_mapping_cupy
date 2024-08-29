#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
from typing import List
import cupyx.scipy.ndimage as ndimage

from .plugin_manager import PluginBase

from FEE import FEE


class FEEIndex(PluginBase):
    """
    FEEIndex is a class that computes the Fundamental Equation of Earth (FEE) index from the semantic map.
    The FEE index is defined as the force required to shear the soil when the following parameters are fixed:
    d, rho, alpha, Q, vx, vz, s_q.
    The soil-properties are however not fixed and are determined by the semantic map and include the following:
    c, phi, gamma, delta, c_a, 
    to the elevation map. The filter is applied to the layer specified by the input_layer_name parameter.
    If the specified layer is not found, the filter is applied to the elevation layer.

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

    def __init__(self, cell_n: int = 100, d: float = 0.1, rho: float = 1.3962634, alpha: float = 0.0, Q: float = 0.0, vx: float = 1.0, vz: float = 0.0, s_q: float = 1.0, w: float = 3.164, **kwargs):
        super().__init__()
        self.known_params_dict = {'d': d, 'rho': rho, 'alpha': alpha, 'Q': Q, 'vx': vx, 'vz': vz, 's_q': s_q, 'w': w}
        # self.known_params = ['d', 'rho', 'alpha', 'Q', 'vx', 'vz', 's_q', 'w']
        self.unknown_params = ['c', 'phi', 'gamma', 'delta', 'c_a']

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
        plugin_layer_index = plugin_layer_names.index('FEE_index')

        # F, Fx, Fz = get_dig_difficulty_FEE_force(output['metadata'], in_metadata, self.metadata_FEE_param_names, fixed_params)
        # TODO: Make this a cuda accelerated function by implementing as a kernel
        FI = plugin_layers[plugin_layer_index]
        if updated_inds is None:
            # If no indices are passed, use phi phi != 0 as a mask to indicate valid FEE params
            mask = semantic_map[semantic_layer_names.index('phi')] != 0
            xinds, yinds = cp.where(mask)
        else:
            xinds = updated_inds[:, 0]
            yinds = updated_inds[:, 1]

        for xind, yind in zip(xinds, yinds):
            est_params = {}
            for param in self.unknown_params:
                est_params[param] = semantic_map[semantic_layer_names.index(param), xind, yind].get().item()
            est_params.update(**self.known_params_dict)
            FI[xind,yind], beta, Fx, Fz = FEE(**est_params, minimizer='beta_from_phi', print_ang_sum=False, clip_angsum=True)
        return FI
