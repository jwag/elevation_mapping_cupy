import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors, colormaps
from scipy.interpolate import LinearNDInterpolator

def surf_elev_map_plot_pretty(elev, layer_color, layer_color_name, cell_n, resolution, offset, colormap='Spectral', scale_mode="z_fit", layer_color_min=None):
   """
   Plot a 3D surface plot of the elevation map using maptlotlib plot_surface. Colors are determined by the layer_color map
   which could be variance, or some other value. The surface plot is interpolated to make it look smoother
   This causes some artifacts at the edges of the map when nans exist at the edge, but it looks better than
   the alternatives. If you really want to know what the underlying data looks like then use the bar_elev_map_plot.
   
   Args:
      elev:             2D numpy array of elevation values
      layer_color:      2D numpy array of values to determine the color of the surface plot
      layer_color_name: Name of the layer_color values for the colorbar
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
   # Get x and y coordinates of the map
   elev_x = np.arange(cell_n )*resolution + offset[0]
   elev_y = np.arange(cell_n )*resolution + offset[1]
   # Pad the edges of the map to enable visualization of the edge cells.
   elev_x_pad = np.arange(cell_n + 2)*resolution + offset[0] - resolution
   elev_y_pad = np.arange(cell_n + 2 )*resolution + offset[1] - resolution
   elev_x_mg, elev_y_mg = np.meshgrid(elev_x_pad, elev_y_pad, indexing='ij')
   z_pad = np.pad(elev.copy(), (1,1), mode='edge')
   # Only use non-nan values for interpolation
   valid_pad = ~np.isnan(z_pad)

   # Break the map into a grid of points to interpolate so that the surface plot looks smoother
   # and the plotted surface at the coordinate (elev_x[i,j], elev_y[i,j]) has an actual height of elev[i,j]
   # instead of being a sloped surface that is interpolated between the heights elev[i,j] and 
   # (elev[i+1,j] and elev[i,j+1]), which is what matplotlib plot_surface does by default
   # This grid is 3 times the resolution of the original map
   x = np.linspace(elev_x[0] - resolution/3, elev_x[-1] + 2*resolution/3, 3*cell_n+1)
   y = np.linspace(elev_y[0] - resolution/3, elev_y[-1] + 2*resolution/3, 3*cell_n+1)
   x, y = np.meshgrid(x, y, indexing='ij')
   # Interpolate z values
   interp = LinearNDInterpolator(list(zip(elev_x_mg[valid_pad].ravel(), elev_y_mg[valid_pad].ravel())), z_pad[valid_pad].ravel())
   # interp = LinearNDInterpolator(list(zip(elev_x_mg.ravel(), elev_y_mg.ravel())), z_pad.ravel())
   z = interp((x.ravel(), y.ravel())).reshape(x.shape)
   # Shift x and y so that the generated surface for the point elev[i,j] is centered on (elev_x[i,j], elev_y[i,j])
   # This is because of matplotlib plot_surface generates the surface from the [0,0] corner of the cell
   # Shift by 1/2 of the resolution of the split grid
   x = x - resolution/(2*3)
   y = y - resolution/(2*3)
   # If there are nans in the interpolated values, replace them with the uninterpolated values
   # This helps correct some artifacts at the edges of the map
   valid = ~np.isnan(z)
   elev_repeated = np.repeat(np.repeat(elev, 3, axis=0), 3, axis=1)
   elev_repeated = np.pad(elev_repeated, (0,1), mode='edge')
   z[~valid] = elev_repeated[~valid]

   # Repeat the layer_color values so that they match the shape of the interpolated z values
   layer_color = np.repeat(np.repeat(layer_color,3,axis=0),3, axis=1)
   # Pad the layer_color values to enable visualization of the far edge cells
   layer_color = np.pad(layer_color, (0,1), mode='edge')
   # Set values that are nan in the elevation map to be nan in the layer_color map
   valid2 = ~np.isnan(elev_repeated)
   layer_color[~valid2] = np.nan
   # Obtain the colormap and normalize the layer_color values
   layer_color_max = np.nanmax(layer_color)
   if layer_color_min is None:
      layer_color_min = np.nanmin(layer_color)
   norm = colors.Normalize(vmin=layer_color_min, vmax=layer_color_max)
   cmap = colormaps[colormap]
   cmap.set_bad(color='black', alpha=0.0)
   rgba = cmap(norm(layer_color))
   # # Set the color of nan values in the elevation map to be transparent
   # rgba[:-1,:-1, 3][~valid] = 0.0
   
   # Plot the surface
   fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
   surf = ax.plot_surface(x, y, z, rstride=3, cstride=3, facecolors=rgba,
                        linewidth=0.1, antialiased=True, shade=False, edgecolor='white')
   # add colorboar
   mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
   mappable.set_array(layer_color)
   fig.colorbar(mappable, ax=ax, shrink=0.5)
   ax.set_title('Elevation Map with ' + layer_color_name + ' Color')
   ax.set_xlabel('x [m]')
   ax.set_ylabel('y [m]')
   ax.set_zlabel('z [m]')

   # Control the aspect ratio and ranges of the plot
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

def surf_elev_map_plot(elev, layer_color, layer_color_name, cell_n, resolution, offset, colormap='Spectral', scale_mode="z_fit", layer_color_min=None):
   """
   Plot a 3D surface plot of the elevation map. Colors are determined by the layer_color map
   which could be variance, or some other value. The surfaces that are generated by plot_surface
   are generated by procedurally generating the faces of the surface from the [0,0] corner of the cell.
   This results in some artifacts at the edges of the map when nans exist at the edge,
   cells next to nans on the edge are not visualized. If you want a smoother plot use surf_elev_map_plot_pretty.

   Args:
      elev: 2D numpy array of elevation values
      layer_color: 2D numpy array of values to determine the color of the surface plot
      layer_color_name: Name of the layer_color values for the colorbar
      cell_n: Number of cells in the x and y direction (true numbers not padded)
      resolution: Resolution of the cells in the x and y direction
      offset: Offset of the map in the x and y direction
      colormap: Colormap to use for the surface plot
      scale_mode: How to scale the z axis of the plot. "z_fit" scales size of the z axis box to match the
                  maximum x or y axis size. "equal" sets the size of the z axis such that the scale is the
                  same in all directions (i.e. a sphere would look like a sphere)
      layer_color_min: Minimum value to consider for coloring the map with layer. If None, the minimum
                         value of the layer_color is used
   """
   
   # Pad to enable visualization of +x and +y the edges of the map
   x = np.arange(cell_n + 1)*resolution + offset[0] - resolution/2
   y = np.arange(cell_n + 1 )*resolution + offset[1] - resolution/2
   x, y = np.meshgrid(x, y, indexing='ij')
   z = np.pad(elev, (0,1), mode='edge')
   fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

   # Get the colormap and normalize the layer_color values
   valid = ~np.isnan(elev)
   # Set values that are nan in the elevation map to be nan in the layer_color map
   layer_color[~valid] = np.nan
   layer_max = np.nanmax(layer_color[valid])
   if layer_color_min is None:
      layer_color_min = np.nanmin(layer_color[valid])
   norm = colors.Normalize(vmin=layer_color_min, vmax=layer_max)
   layer_color = np.pad(layer_color, (0,1), mode='edge')
   cmap = colormaps[colormap]
   # For plotting nan values as transparent
   cmap.set_bad(color='black', alpha=0.0)
   rgba = cmap(norm(layer_color))

   # Interpolate nan z in the middle of the map (holes) to make surrounding cell plots look proper
   valid_pad = ~np.isnan(z)
   interp = LinearNDInterpolator(list(zip(x[valid_pad].ravel(), y[valid_pad].ravel())), z[valid_pad].ravel())
   z = interp((x.ravel(), y.ravel())).reshape(x.shape)
   surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgba,
                           antialiased=False, shade=False, edgecolor='white', linewidth=0.01)
   
   # add colorboar
   mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
   mappable.set_array(layer_color)
   fig.colorbar(mappable, ax=ax, shrink=0.5)
   ax.set_title('Elevation Map with ' + layer_color_name + ' Color')
   ax.set_xlabel('x [m]')
   ax.set_ylabel('y [m]')
   ax.set_zlabel('z [m]')
   
   # Control the aspect ratio and ranges of the plot
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

# TODO: Add plot to show loose soil on top of the elevation map
def bar_elev_map_plot(elev, title, layer_color, colorbar_label, crop_mask, true_cell_n, resolution, offset, colormap='Spectral', scale_mode="z_fit", layer_color_rng=[None,None], layer_truth_str=None):
      """
      Plot a 3D bar plot of the elevation map. Colors are determined by the layer_color map
      which could be variance, or some other value. This plot is not interpolated and shows the
      actual data values. This may make for slower rendering due to each column being a separate bar with 6 faces.

      Args:
         elev: 2D numpy array of elevation values
         elev_name: Name of the elevation values for the title of the plot
         layer_color: 2D numpy array of values to determine the color of the surface plot
         layer_color_name: Name of the layer_color values for the colorbar
         cell_n: Number of cells in the x and y direction (true numbers not padded)
         resolution: Resolution of the cells in the x and y direction
         offset: Offset of the map in the x and y direction
         colormap: Colormap to use for the surface plot
         scale_mode: How to scale the z axis of the plot. "z_fit" scales size of the z axis box to match the
                     maximum x or y axis size. "equal" sets the size of the z axis such that the scale is the
                     same in all directions (i.e. a sphere would look like a sphere)
         layer_color_rng: Minimum and maximium value to consider for coloring the map with layer. When None,
                          the range of the layer_color is used
      """
      # Set bottom as minimum elevation
      minz = np.nanmin(elev)
      bottom = np.ones_like(elev) * minz
      dz = elev.copy() - minz

      # This is the center index of the map accounting for rounding taking place in get_x_idx() and get_y_idx()
      # This is necessary because for odd numbers of cells the center of the map
      # is at the edge of a cell, but for even numbers of cells the center is in the middle of a cell
      # This is a result of the way the get_x_idx() and get_y_idx() functions are implemented
      center_ind_float = (true_cell_n)/2.0 # Not the map center but the center index of the map
      # Offset here is the actual xy coordinates of the map center
      x = (np.arange(true_cell_n)  - center_ind_float)*resolution + offset[0]
      y = (np.arange(true_cell_n) - center_ind_float)*resolution + offset[1]
      x, y = np.meshgrid(x, y, indexing='ij')

      # Set values that are nan in the elevation map to be nan in the layer_color map
      valid = ~np.isnan(dz)
      # Add in crop mask
      valid = valid & crop_mask
      layer_color[~valid] = np.nan
      # Get the colormap and normalize the layer_color values
      if layer_color_rng[0] is None:
         layer_color_rng[0] = np.nanmin(layer_color)
      if layer_color_rng[1] is None:
         layer_color_rng[1] = np.nanmax(layer_color)
      norm = colors.Normalize(vmin=layer_color_rng[0], vmax=layer_color_rng[1])
      cmap = colormaps[colormap]
      cmap.set_bad(color='black', alpha=0.0)
      rgba = cmap(norm(layer_color[valid]))

      # Plot the bars
      fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
      ax.bar3d(x[valid].ravel(), y[valid].ravel(), bottom[valid].ravel(), resolution, resolution, dz[valid].ravel(),
               shade=False, color=rgba.reshape(-1, 4), edgecolor='white', linewidth=0.1)
      # add colorbar
      mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
      mappable.set_array(layer_color[valid])
      fig.colorbar(mappable, ax=ax, shrink=0.5, label=colorbar_label)
      # title  = elev_name + ' Map with ' + layer_color_name + ' Color'
      if layer_truth_str is not None:
         title += ' with true value ' + layer_truth_str
      ax.set_title(title)
      ax.set_xlabel('x [m]')
      ax.set_ylabel('y [m]')
      # ax.set_zlabel('z [m]')

      # Remove grid in z direction
      # ax.zaxis._axinfo['grid'].update(color = (1,1,1,0))
      ax.zaxis.pane.fill = False
      ax.zaxis.pane.set_edgecolor('white')

      # Control the aspect ratio and ranges of the plot
      xmin = np.min(x[valid])
      xmax = np.max(x[valid]) + resolution
      ymin = np.min(y[valid])
      ymax = np.max(y[valid]) + resolution
      zmin = np.nanmin(elev)
      zmax = np.nanmax(elev)
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
      ax.grid(False)
      ax.set_axis_off()
      debug = 1

if __name__ == '__main__':
   # Generate Test Map
   cell_n = 4
   resolution = .1
   offset = np.zeros(2)
   elev = np.zeros((cell_n, cell_n))
   elev[0,0] = 1.0
   # elev[0,0] = 1.0
   # elev[1,0] = 0.5
   # elev[0,1] = 0.5
   # elev[1,1] = np.nan
   # elev[2,1] = 1.5
   # elev[1,2] = 2.5
   # elev[2,2] = 3.0
   # elev[3:,:] = np.nan
   # elev[:,3] = np.nan
   # Original xy coordinates
   elev *= resolution
   elev -= 3
   # scale_mode = "z_fit"
   scale_mode = "equal"
   layer_color_name = "Elevation"
   bar_elev_map_plot(elev, elev, layer_color_name, cell_n, resolution, offset, scale_mode=scale_mode)
   # surf_elev_map_plot(elev, elev, layer_color_name, cell_n, resolution, offset, scale_mode=scale_mode)
   # surf_elev_map_plot_pretty(elev, elev, layer_color_name, cell_n, resolution, offset, scale_mode=scale_mode)
   plt.show()
   debug = 1

