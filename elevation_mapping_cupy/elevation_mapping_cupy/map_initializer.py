#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
from scipy.interpolate import griddata
import numpy as np
import cupy as cp


class MapInitializer(object):
    def __init__(self, initial_variance, xp=np, method="points"):
        self.methods = ["points", "heightmap"]
        assert method in self.methods, "method should be chosen from {}".format(self.methods)
        self.method = method
        self.xp = xp
        self.initial_variance = initial_variance

    def __call__(self, *args, **kwargs):
        if self.method == "points":
            self.points_initializer(*args, **kwargs)
        elif self.method == "heightmap":
            self.heightmap_initializer(*args, **kwargs)
        else:
            return

    def points_initializer(self, elevation_map, points, new_variance=None, method="linear"):
        """Initialize the map using interpolation between given points

        Args:
            elevation_map (cupy._core.core.ndarray): elevation_map data.
            points (cupy._core.core.ndarray): points used to interpolate.
            new_variance (float): variance for new points. If None, initial_variance is used.
            method (str): method for interpolation. (nearest, linear, cubic)

        """
        if new_variance is None:
            new_variance = self.initial_variance
        # points from existing map.
        points_idx = self.xp.where(elevation_map[2] > 0.5)
        values = elevation_map[0, points_idx[0], points_idx[1]]

        # Add external points for interpolation.
        points_idx = self.xp.stack(points_idx).T
        points_idx = self.xp.vstack([points_idx, points[:, :2]])
        values = self.xp.hstack([values, points[:, 2]])

        assert points_idx.shape[0] > 3, "Initialization points must be more than 3."

        # Interpolation using griddata function.
        w = elevation_map.shape[1]
        h = elevation_map.shape[2]
        grid_x, grid_y = np.mgrid[0:w, 0:h]
        if self.xp == cp:
            points_idx = cp.asnumpy(points_idx)
            values = cp.asnumpy(values)
        interpolated = griddata(points_idx, values, (grid_x, grid_y), method=method)
        if self.xp == cp:
            interpolated = cp.asarray(interpolated)

        # Update elevation map.
        elevation_map[0] = self.xp.nan_to_num(interpolated)
        # Make the elevation_reference the same as the initial elevation
        elevation_map[8] = self.xp.nan_to_num(interpolated)  # elevation_reference
        elevation_map[1] = self.xp.where(
            self.xp.invert(self.xp.isnan(interpolated)), new_variance, self.initial_variance
        )
        elevation_map[2] = self.xp.where(self.xp.invert(self.xp.isnan(interpolated)), 1.0, 0.0)
        return

    def heightmap_initializer(self, elevation_map, heightmap, new_variance=None):
        """Initialize the map using a heightmap

        Args:
            elevation_map (cupy._core.core.ndarray): elevation_map data.
            heightmap (cupy._core.core.ndarray): heightmap data to initialize the elevation map.
            new_variance (float): variance for new points. If None, initial_variance is used.

        """
        if new_variance is None:
            new_variance = self.initial_variance

        # Ensure heightmap dimensions match the inner dimensions of the elevation map (excluding borders)
        inner_shape = (elevation_map.shape[1] - 2, elevation_map.shape[2] - 2)
        assert heightmap.shape == inner_shape, "Heightmap dimensions must match the inner dimensions of the elevation map (excluding borders)."

        # Update elevation map
        elevation_map[0,1:-1,1:-1] = heightmap
        # Make the elevation_reference the same as the initial elevation
        # TODO: Make this configurable or add another initializer
        elevation_map[8,1:-1,1:-1] = heightmap
        # Set variance for the new heightmap area
        elevation_map[1,1:-1,1:-1] = self.xp.full_like(heightmap, new_variance)
        # Set entire map as valid
        elevation_map[2,1:-1,1:-1] = self.xp.ones_like(heightmap)
        return


if __name__ == "__main__":
    initializer = MapInitializer(100, method="heightmap", xp=cp)
    m = np.zeros((4, 10, 10))
    heightmap = cp.array([[0.1 * i for i in range(10)] for j in range(10)])
    m = cp.asarray(m)
    initializer(m, heightmap, new_variance=50)
    print(m[0])
    print(m[1])
    print(m[2])
