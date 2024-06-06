import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors, colormaps
from scipy.interpolate import LinearNDInterpolator

def surf_elev_map_plot_pretty(elev, layer_color, cell_n, resolution, offset, colormap='Spectral', scale_mode="z_fit", layer_color_min=None):
   """
   Plot a 3D surface plot of the elevation map using maptlotlib plot_surface. Colors are determined by the layer_color map
   which could be variance, or some other value. The surface plot is interpolated to make it look smoother
   This causes some artifacts at the edges of the map when nans exist at the edge, but it looks better than
   the alternatives. If you really want to know what the underlying data looks like then use the bar_elev_map_plot.
   
   Args:
      elev:             2D numpy array of elevation values
      layer_color:      2D numpy array of values to determine the color of the surface plot
      cell_n:           Number of cells in the x and y direction (true numbers not padded)
      resolution:       Resolution of the cells in the x and y direction
      offset:           Offset of the map in the x and y direction
      colormap:         Colormap to use for the surface plot
      scale_mode:       How to scale the z axis of the plot. "z_fit" scales size of the z axis box to match the
                        maximum x or y axis size. "equal" sets the size of the z axis such that the scale is the
                        same in all directions (i.e. a sphere would look like a sphere)
      layer_color_min:  Minimum value to consider for coloring the map with layer. If None, the minimum 
                        value of the layer_color is used
   """
   cmap = colormaps[colormap]
   cmap.set_bad(color='black', alpha=0.0)
    # Include padding of 1 cell on each side for plotting and interpolation purposes
   elev_x = np.arange(cell_n )*resolution + offset[0]
   elev_y = np.arange(cell_n )*resolution + offset[1]
   elev_x_pad = np.arange(cell_n + 2)*resolution + offset[0] - resolution
   elev_y_pad = np.arange(cell_n + 2 )*resolution + offset[1] - resolution
   elev_x_mg, elev_y_mg = np.meshgrid(elev_x_pad, elev_y_pad, indexing='ij')

   # Now generate interpolated xy and z coordinates for map to make surface plot look smooth
   # Pad Z now
   z_pad = np.pad(elev.copy(), (1,1), mode='edge')
   valid_pad = ~np.isnan(z_pad)

   x = np.linspace(elev_x[0] - resolution/3, elev_x[-1] + 2*resolution/3, 3*cell_n+1)
   y = np.linspace(elev_y[0] - resolution/3, elev_y[-1] + 2*resolution/3, 3*cell_n+1)
   x, y = np.meshgrid(x, y, indexing='ij')
   # Interpolate z values
   interp = LinearNDInterpolator(list(zip(elev_x_mg[valid_pad].ravel(), elev_y_mg[valid_pad].ravel())), z_pad[valid_pad].ravel())
   # interp = LinearNDInterpolator(list(zip(elev_x_mg.ravel(), elev_y_mg.ravel())), z_pad.ravel())
   z = interp((x.ravel(), y.ravel())).reshape(x.shape)
   # Shift x and y to be in the center of the cell
   x = x - resolution/(2*3)
   y = y - resolution/(2*3)
   # Remove values that were interpolated from nans by making those values nans again
   # elev_tmp = np.repeat(np.repeat(elev,3,axis=0),3, axis=1)
   # elev_tmp = np.pad(elev_tmp, (0,1), mode='edge')
   # valid_interp = ~np.isnan(elev_tmp)
   # z[~valid_interp] = np.nan
   # If there are nans in the interpolated values, replace them with the uninterpolated values
   # This helps correct some artifacts at the edges of the map
   valid = ~np.isnan(z[:-1,:-1])
   elev_repeated = np.repeat(np.repeat(elev, 3, axis=0), 3, axis=1)
   # Hacky indexing, but it works i supose
   z[:-1,:-1][~valid] = elev_repeated[~valid]
   layer_color = np.repeat(np.repeat(layer_color,3,axis=0),3, axis=1)
   layer_color = np.pad(layer_color, (0,1), mode='edge')
   layer_color_max = np.nanmax(layer_color)
   if layer_color_min is None:
      layer_color_min = np.nanmin(layer_color)
   # TODO: Deal with min value
   norm = colors.Normalize(vmin=layer_color_min, vmax=layer_color_max)
   rgba = cmap(norm(layer_color))
   # rgba[0:n_repeats*n_pad_front,:] = [0, 0, 0, 0]
   # rgba[:,0:n_repeats*n_pad_front] = [0, 0, 0, 0]
   # fig = plt.figure()
   # ax = fig.add_subplot(1, 1, 1, projection='3d')
   fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
   surf = ax.plot_surface(x, y, z, rstride=3, cstride=3, facecolors=rgba,
                        linewidth=0.1, antialiased=True, shade=False, edgecolor='white')
   
   # wirex = np.arange(cell_n +1)*resolution + offset_x - resolution/3
   # wirey = np.arange(cell_n +1)*resolution + offset_y - resolution/3
   # wirex, wirey = np.meshgrid(wirex, wirey, indexing='ij')
   # wirez = np.pad(elev, (0,1), mode='edge')
   # ax.plot_wireframe(wirex, wirey, wirez, color='black', linewidth=0.5)
   # ax.plot_wireframe(x, y, z, rstride=3, cstride=3,
   #                   color='black', linewidth=0.5)
   # add colorboar
   mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
   mappable.set_array(layer_color)
   fig.colorbar(mappable, ax=ax, shrink=0.5)
   ax.set_title('Elevation Map')
   ax.set_xlabel('x [m]')
   ax.set_ylabel('y [m]')
   ax.set_zlabel('z [m]')
   # Make plot aspect ratio equal
   # ax.set_box_aspect([1,1,1])
   # ax.axis('equal')
   # ax.axis('scaled')

   v = ~np.isnan(z)
   xmin = np.min(x[v])
   xmax = np.max(x[v])
   ymin = np.min(y[v])
   ymax = np.max(y[v])
   zmin = np.nanmin(z)
   zmax = np.nanmax(z)
   xrange = xmax - xmin
   yrange = ymax - ymin
   zrange = zmax - zmin
   ax.set_xlim(xmin - 0.1*xrange, xmax + 0.1*xrange)
   ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
   ax.set_zlim(zmin - 0.1*zrange, zmax + 0.1*zrange)
   if scale_mode == "z_fit":
      z_scale = np.max([xrange, yrange])/zrange
   elif scale_mode == "equal":
      z_scale = 1.0
   ax.set_box_aspect([xrange, yrange, z_scale*zrange])
   # ax.set_zlim(-3.0, 3.0)

def surf_elev_map_plot(elev, layer_color, cell_n, resolution, offset, colormap='Spectral', scale_mode="z_fit"):
   """
   Plot a 3D surface plot of the elevation map. Colors are determined by the layer_color map
   which could be variance, or some other value. The surfaces that are generated by plot_surface
   are generated by procedurally generating the faces of the surface from the [0,0] corner of the cell.
   This results in some artifacts at the edges of the map when nans exist at the edge.
   """
   
   valid = np.logical_not(np.isnan(elev))
   # Pad to enable visualization of +x and +y the edges of the map
   x = np.arange(cell_n + 1)*resolution + offset[0] - resolution/2
   y = np.arange(cell_n + 1 )*resolution + offset[1] - resolution/2
   x, y = np.meshgrid(x, y, indexing='ij')
   z = np.pad(elev, (0,1), mode='edge')
   fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
   cmap = colormaps[colormap]
   cmap.set_bad(color='black', alpha=0.0)
   layer_max = np.nanmax(layer_color[valid])
   # TODO: Deal with min value
   norm = colors.Normalize(vmin=0.0, vmax=layer_max)
   layer_color = np.pad(layer_color, (0,1), mode='edge')
   rgba = cmap(norm(layer_color))
   # Interpolate nan z values and plot as different color
   valid_pad = ~np.isnan(z)
   interp = LinearNDInterpolator(list(zip(x[valid_pad].ravel(), y[valid_pad].ravel())), z[valid_pad].ravel())
   z = interp((x.ravel(), y.ravel())).reshape(x.shape)
   surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgba,
                           linewidth=0.033, antialiased=False, shade=False, edgecolor='white')
   # Add lines to the surface plot
   # ax.plot_wireframe(x, y, z, color='black', linewidth=0.5)
   # add color bar which maps variances to colors
   mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
   mappable.set_array(layer_color)
   fig.colorbar(mappable, ax=ax, shrink=0.5)
   ax.set_title('Elevation Map')
   ax.set_xlabel('x [m]')
   ax.set_ylabel('y [m]')
   ax.set_zlabel('z [m]')
   # Make plot aspect ratio equal
   # ax.axis('scaled')
   v = ~np.isnan(z)
   xmin = np.min(x[v])
   xmax = np.max(x[v])
   ymin = np.min(y[v])
   ymax = np.max(y[v])
   zmin = np.nanmin(z)
   zmax = np.nanmax(z)
   xrange = xmax - xmin
   yrange = ymax - ymin
   zrange = zmax - zmin
   ax.set_xlim(xmin - 0.1*xrange, xmax + 0.1*xrange)
   ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
   ax.set_zlim(zmin - 0.1*zrange, zmax + 0.1*zrange)
   if scale_mode == "z_fit":
      z_scale = np.max([xrange, yrange])/zrange
   elif scale_mode == "equal":
      z_scale = 1.0
   ax.set_box_aspect([xrange, yrange, z_scale*zrange])
   # ax.set_box_aspect([1,1,1])
   # ax.set_zlim(-3.0, 3.0)


def bar_elev_map_plot(elev, layer_color, cell_n, resolution, offset, colormap='Spectral', scale_mode="z_fit"):
      """
      Plot a 3D bar plot of the elevation map. Colors are determined by the layer_color map
      which could be variance, or some other value. This plot is not interpolated and shows the
      actual data values. This may make for slower rendering due to each column being a separate bar with 6 faces.

      Args:
         elev: 2D numpy array of elevation values
         layer_color: 2D numpy array of values to determine the color of the surface plot
         cell_n: Number of cells in the x and y direction (true numbers not padded)
         resolution: Resolution of the cells in the x and y direction
         offset: Offset of the map in the x and y direction
         colormap: Colormap to use for the surface plot
      """
      cmap = colormaps[colormap]
      cmap.set_bad(color='black', alpha=0.0)
      fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
      bottom = np.zeros_like(elev)
      top = elev.copy()
      x = np.arange(cell_n)*resolution + offset[0] - resolution/2
      y = np.arange(cell_n )*resolution + offset[1] - resolution/2
      x, y = np.meshgrid(x, y, indexing='ij')
      valid = ~np.isnan(elev)
      layer_color_max = np.nanmax(elev)
      norm = colors.Normalize(vmin=0.0, vmax=layer_color_max)
      rgba = cmap(norm(layer_color[valid]))
      ax.bar3d(x[valid].ravel(), y[valid].ravel(), bottom[valid].ravel(), resolution, resolution, top[valid].ravel(), shade=False, color=rgba.reshape(-1, 4))
      # add colorboar
      mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
      mappable.set_array(layer_color[valid])
      fig.colorbar(mappable, ax=ax, shrink=0.5)
      ax.set_title('Elevation Map')
      ax.set_xlabel('x [m]')
      ax.set_ylabel('y [m]')
      ax.set_zlabel('z [m]')
      # Make plot aspect ratio equal
      # ax.set_box_aspect([1,1,1])
      # ax.axis('equal')
      # ax.axis('scaled')
      v = ~np.isnan(top)
      xmin = np.min(x[v])
      xmax = np.max(x[v]) + resolution
      ymin = np.min(y[v])
      ymax = np.max(y[v]) + resolution
      zmin = np.nanmin(top)
      zmax = np.nanmax(top)
      xrange = xmax - xmin
      yrange = ymax - ymin
      zrange = zmax - zmin
      ax.set_xlim(xmin - 0.1*xrange, xmax + 0.1*xrange)
      ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
      ax.set_zlim(zmin - 0.1*zrange, zmax + 0.1*zrange)
      if scale_mode == "z_fit":
         z_scale = np.max([xrange, yrange])/zrange
      elif scale_mode == "equal":
         z_scale = 1.0
      ax.set_box_aspect([xrange, yrange, z_scale*zrange])
      # ax.set_zlim(-3.0, 3.0)

if __name__ == '__main__':
   # Generate Test Map
   cell_n = 5
   resolution = .1
   offset = np.zeros(2)
   elev = np.zeros((cell_n, cell_n))
   elev[0,0] = 1.0
   elev[1,0] = 0.5
   elev[0,1] = 0.5
   elev[1,1] = np.nan
   elev[2,1] = 1.5
   elev[1,2] = 2.5
   elev[2,2] = 3.0
   elev[3:,:] = np.nan
   # elev[:,3] = np.nan
   # Original xy coordinates
   elev *= resolution
   # scale_mode = "z_fit"
   scale_mode = "equal"
   surf_elev_map_plot_pretty(elev, elev, cell_n, resolution, offset, scale_mode=scale_mode)
   bar_elev_map_plot(elev, elev, cell_n, resolution, offset, scale_mode=scale_mode)
   surf_elev_map_plot(elev, elev, cell_n, resolution, offset, scale_mode=scale_mode)
   plt.show()
   debug = 1

