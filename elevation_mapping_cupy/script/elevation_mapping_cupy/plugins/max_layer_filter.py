#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import numpy as np
from typing import List

from elevation_mapping_cupy.plugins.plugin_manager import PluginBase


class MaxLayerFilter(PluginBase):
    """Applies a maximum filter to the input layers and updates the traversability map.
    This can be used to enhance navigation by identifying traversable areas.

    Args:
        cell_n (int): The width and height of the elevation map.
        layers (list): List of layers for semantic traversability. Default is ["traversability"].
        thresholds (list): List of thresholds for each layer. Default is [0.5].
        type (list): List of types for each layer. Default is ["traversability"].
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        cell_n: int = 100,
        layers: list = ["traversability"],
        reverse: list = [True],
        min_or_max: str = "max",
        thresholds: list = [False],
        default_value: float = 0.0,
        apply_sqrt: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.layers = layers
        self.reverse = reverse
        self.min_or_max = min_or_max
        self.thresholds = thresholds
        self.default_value = default_value
        self.apply_sqrt = apply_sqrt

    def get_layer_data(
        self,
        elevation_map,
        layer_names,
        plugin_layers,
        plugin_layer_names,
        semantic_map,
        semantic_layer_names,
        name,
    ):
        if name in layer_names:
            idx = layer_names.index(name)
            layer = elevation_map[idx].copy()
        elif name in plugin_layer_names:
            idx = plugin_layer_names.index(name)
            layer = plugin_layers[idx].copy()
        elif name in semantic_layer_names:
            idx = semantic_layer_names.index(name)
            layer = semantic_map[idx].copy()
        else:
            print(f"Could not find layer {name}!")
            layer = None
        return layer

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            semantic_map (elevation_mapping_cupy.semantic_map.SemanticMap):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        layers = []
        for it, name in enumerate(self.layers):
            layer = self.get_layer_data(
                elevation_map, layer_names, plugin_layers, plugin_layer_names, semantic_map, semantic_layer_names, name
            )
            if layer is None:
                continue
            if isinstance(self.default_value, float):
                layer = cp.where(layer == 0.0, float(self.default_value), layer)
            elif isinstance(self.default_value, str):
                default_layer = self.get_layer_data(
                    elevation_map,
                    layer_names,
                    plugin_layers,
                    plugin_layer_names,
                    semantic_map,
                    semantic_layer_names,
                    self.default_value,
                )
                layer = cp.where(layer == 0, default_layer, layer)
            if self.reverse[it]:
                layer = 1.0 - layer
            if isinstance(self.thresholds[it], float):
                layer = cp.where(layer > float(self.thresholds[it]), 1, 0)
            layers.append(layer)
        if len(layers) == 0:
            print("No layers are found, returning traversability!")
            idx = layer_names.index("traversability")
            return elevation_map[idx]
        result = cp.stack(layers, axis=0)
        if self.min_or_max == "min":
            result = cp.min(result, axis=0)
        else:
            result = cp.max(result, axis=0)
        if self.apply_sqrt:
            result = cp.square(result)
        return result
