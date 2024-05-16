import numpy as np
import trimesh

from trimesh.base import Trimesh
from trimesh import grouping, util
from trimesh.typed import ArrayLike, Dict, Optional
from trimesh.constants import tol

# Define GET Geometry and Sequence of thin 4 point polygons
# using the trimesh library
# Could do this with a CAD file, but this is a simple test
def simble_blade_geometry(blade_width=3.0, blade_height=0.6, blade_angle_deg=-10, blade_origin=[1.634, 0.0, 0.060+0.265]):
    # Blade consists of a single plane with 4 points
    # Define the 4 points of the blade
    # The blade is a thin 4 point polygon
    # The blade is defined in the ZY plane
    # where the top of the blade is in the positive Z direction,
    # the left side of the blade is in the positive Y direction,
    # and front of the blade facing the positive X direction when the blade is at 0 degrees.
    # The origin of the blade is at the center of the rectangle.
    # The blade is rotated about the Y axis at the origin by blade_angle degrees ccw about the Y axis.

    # Define the 4 vertices of the blade
    vertices = np.array([[0, blade_width/2.0, blade_height/2.0],
                        [0, -blade_width/2.0, blade_height/2.0],
                        [0, -blade_width/2.0, -blade_height/2.0],
                        [0, blade_width/2.0, -blade_height/2.0]])
    
    curved_blade_vertices = np.array([[0.4, blade_width/2.0+0.4, blade_height/2.0],
                                      [0.4, blade_width/2.0+0.4, -blade_height/2.0],
                                      [0.4, -blade_width/2.0-0.4, -blade_height/2.0],
                                      [0.4, -blade_width/2.0-0.4, blade_height/2.0]])
    
    # vertices = np.concatenate((vertices, curved_blade_vertices), axis=0)
    
    print("Original Vertices: ", vertices)

    # Define the single face of the blade
    # The blade will be visible from the front since the normal is pointing in the positive X direction
    # of the untransformed blade
    faces_front = np.array([[0, 1, 2, 3]])

    faces_curved = np.array([[0, 3, 5, 4], [1, 7, 6, 2]])

    # faces_front = np.concatenate((faces_front, faces_curved), axis=0) 

    # Create the trimesh object
    blade = trimesh.Trimesh(vertices=vertices, faces=faces_front)

    # Rotate the blade about the Y axis at the origin by blade_angle degrees ccw
    rot = trimesh.transformations.rotation_matrix(np.radians(blade_angle_deg), [0, 1, 0])
    # Translate the blade so that the origin is at [0, 0, blade_origin_z]
    # The transform from world frame to blade frame
    T_WB = trimesh.transformations.translation_matrix(blade_origin)@rot
    blade.apply_transform(T_WB)

    print("Transformed Vertices: ", blade.vertices)

    return blade, T_WB, faces_front

def sweep_thin_poly_mesh(
    poly_mesh: Trimesh,
    transforms: ArrayLike,
    roll_dirs: ArrayLike,
    convex_interp: bool = True,
    cap: bool = True,
    connect: bool = True,
    kwargs: Optional[Dict] = None,
    **triangulation,
) -> Trimesh:
    """
    Modified from trimesh.creation sweep_polygon function
    Extrude a thin polygon mesh into a 3D mesh along a 3D path. Note that this
    does *not* handle the case where there is very sharp curvature leading
    the polygon to intersect the plane of a previous slice, and does *not*
    scale the polygon along the induced normal to result in a constant cross section.

    This function differs from the trimesh.creation.sweep_polygon function in that
    it enables sweeping along a path that is not normal to the polygon plane.
    This is accomplished by specifying the path as a list of transforms of the poly_mesh.

    # TODO Finish updating the docstring to reflect the changes made to the function and 
    # remove references to the path parameter

    You may want to resample your path with a B-spline, i.e:
      `trimesh.path.simplify.resample_spline(path, smooth=0.2, count=100)`

    Parameters
    ----------
    poly_mesh : trimesh.Trimesh
      Profile to sweep along sequence of transforms
    transforms : (n+1, 4, 4) float
      A sequence of transforms to apply to the poly_mesh. The first transform
      should be the initial pose of the poly_mesh.
    roll_dirs : (n,) bool
      The direction of roll between each slice. True for positive roll, False for negative roll.
    convex_interp : bool
      If True, interpolate convex hulls between each slice. If False, interpolate concave hulls.
    cap
      If an open path is passed apply a cap to both ends.
    connect
      If a closed path is passed connect the sweep into
      a single watertight mesh.
    kwargs : dict
      Passed to the mesh constructor.
    **triangulation
      Passed to `triangulate_polygon`, i.e. `engine='triangle'`

    Returns
    -------
    swept_mesh : trimesh.Trimesh
      Geometry of result
    """

    transforms = np.asanyarray(transforms, dtype=np.float64)
    if not (util.is_shape(transforms, (-1, 4, 4)) and len(transforms) > 1):
        raise ValueError("transforms must be (n+1, 4, 4)!")
    
    n_sweeps = len(transforms) - 1

    # check to see if path is closed i.e. first and last vertex are the same
    closed = np.linalg.norm(transforms[0] - transforms[-1]) < tol.merge
    # Extract 2D vertices and triangulation
    org_vertices = poly_mesh.vertices
    org_faces = poly_mesh.faces
    org_edges = poly_mesh.edges

    # edges which only occur once are on the boundary of the polygon
    # since the triangulation may have subdivided the boundary of the
    # polygon, we need to find it again
    edges_unique = grouping.group_rows(np.sort(org_edges, axis=1), require_count=1)
    # subset the vertices to only ones included in the boundary
    unique, inverse = np.unique(org_edges[edges_unique].reshape(-1), return_inverse=True)
    # take only the vertices in the boundary
    # and stack them with zeros and ones so we can use dot
    # products to transform them all over the place
    vertices_tf = np.column_stack(
        (org_vertices[unique], np.ones(len(unique)))
    )
    # the indices of vertices_tf
    boundary = inverse.reshape((-1, 2))

    # apply transforms to prebaked homogeneous coordinates
    # TODO: Speed this up with einsum or similar
    vertices_3D = np.concatenate(
        [np.dot(vertices_tf, matrix.T) for matrix in transforms], axis=0
    )[:, :3]

    # now construct the faces with one group of boundary faces per slice
    stride = len(unique)
    boundary_next = boundary + stride
    faces_slice_pos = np.column_stack(
        [boundary, boundary_next[:, :1], boundary_next[:, ::-1], boundary[:, 1:]]
        ).reshape((-1, 3))
    faces_slice_neg = np.column_stack(
      [boundary, boundary_next[:, -1:], boundary_next[:, ::-1], boundary[:, :-1]]
      ).reshape((-1, 3))
    if not convex_interp:
        # if we're interpolating concave hulls we need to flip the sign of roll_dirs
        roll_dirs = np.logical_not(roll_dirs)
    # Select appropriate face slice based on roll direction for each slice
    faces_slices = np.where(roll_dirs[:,None, None], faces_slice_pos[None,:,:], faces_slice_neg[None,:,:])
    # Now offset the faces for each slice
    offsets = (np.arange(n_sweeps) * stride)[:, None, None]
    faces = (faces_slices + offsets).reshape((-1, 3))

    # connect only applies to closed paths
    if closed and connect:
        # the last slice will not be required
        max_vertex = n_sweeps * stride
        # clip off the duplicated vertices
        vertices_3D = vertices_3D[:max_vertex]
        # apply the modulus in-place to a conservative subset
        faces[-1] %= max_vertex
        face_colors = None
    elif cap:
        # these are indices of `vertices_2D` that were not on the boundary
        # which can happen for triangulation algorithms that added vertices
        # we don't currently support that but you could append the unconsumed
        # vertices and then update the mapping below to reflect that
        unconsumed = set(unique).difference(org_faces.ravel())
        if len(unconsumed) > 0:
            raise NotImplementedError("triangulation added vertices: no logic to cap!")

        # map the 2D faces to the order we used
        mapped = np.zeros(unique.max() + 2, dtype=np.int64)
        mapped[unique] = np.arange(len(unique))

        # now should correspond to the first vertex block
        cap_zero = mapped[org_faces]
        # winding will be along +Z so flip for the bottom cap
        # faces.append(np.fliplr(cap_zero))
        faces = np.concatenate((faces, np.fliplr(cap_zero)), axis=0)
        # offset the end cap
        # faces.append(cap_zero + stride * n_sweeps)
        faces = np.concatenate((faces, cap_zero + stride * n_sweeps), axis=0)
        # Define face_colors for mesh where the original face is green,
        # the swept faces are grey, and the final face is red
        alpha = 125
        n_faces = len(faces)
        n_org_faces = len(org_faces)
        face_colors = np.ones((n_faces, 4), dtype=int) * 169 # Set color to grey
        face_colors[n_faces-n_org_faces*2:-n_org_faces,:3] = [0, 255, 0]
        face_colors[-n_org_faces:,:3] = [255, 0, 0]
        face_colors[:, 3] = alpha # Set transparency to alpha

    if kwargs is None:
        kwargs = {}

    if "process" not in kwargs:
        # we should be constructing clean meshes here
        # so we don't need to run an expensive verex merge
        kwargs["process"] = False

    # generate the mesh from the face data
    swept_mesh = Trimesh(vertices=vertices_3D, faces=faces, face_colors=face_colors, **kwargs)

    if tol.strict:
        # we should not have included any unused vertices
        assert len(np.unique(faces)) == len(vertices_3D)

        if cap:
            # mesh should always be a volume if cap is true
            assert swept_mesh.is_volume

        if closed and connect:
            assert swept_mesh.is_volume
            assert swept_mesh.body_count == 1

    return swept_mesh

if __name__ == "__main__":
    # Create the blade geometry
    blade_origin=[1.634, 0.0, 0.060+0.265]
    # blade_origin=[0.0, 0.0, 0.0]
    blade_mesh, T_WB, faces_front = simble_blade_geometry(blade_width=3.0, blade_height=0.6, blade_angle_deg=-10, blade_origin=blade_origin)

    # Define the path of the blade
    # T_dB = trimesh.transformations.translation_matrix([1.5, 1.5, 0.0])
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [1, 0, 0])@T_dB
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-10), [0, 1, 0])@T_dB
    # T_dB2 = trimesh.transformations.translation_matrix([1.5, 0.0, 0.0])@T_dB
    # T_dB3 = trimesh.transformations.translation_matrix([3.0, 1.5, 0.0])
    # T_db3 = trimesh.transformations.rotation_matrix(np.radians(10), [1, 0, 0])@T_dB3
    # T_dB3 = trimesh.transformations.rotation_matrix(np.radians(15), [0, 0, 1])@T_dB3@T_dB2
    # roll_dirs = np.array([-20, 0, 10]) >= 0
    # transforms = np.array([np.eye(4), T_dB, T_dB2, T_dB3])


    T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [0, 1, 0])@np.linalg.inv(T_WB)
    roll_dirs = np.array([0]) >= 0
    transforms = np.array([np.linalg.inv(T_WB), T_dB])

    # roll_dirs = np.array([-20]) >= 0
    # transforms = np.array([np.eye(4), T_dB])
    new_mesh = sweep_thin_poly_mesh(blade_mesh, transforms, roll_dirs=roll_dirs, convex_interp=True, cap=True, connect=False)
    new_mesh.apply_transform(T_WB)
    # trimesh.util.concatenate(new_mesh.split(only_watertight=True))
    # new_mesh.show(smooth=False)
    # scene = trimesh.Scene([new_mesh.convex_hull])
    scene = trimesh.Scene([new_mesh])
    # Add a world coordinate frame to the scene
    world_frame = trimesh.creation.axis(origin_size=0.1, axis_length=1.0)
    scene.add_geometry(world_frame)
    # Add blade coordinate frame to the scene
    # Make origin size larger and color purple
    blade_frame = trimesh.creation.axis(origin_size=0.1, transform=T_WB, axis_length=1.0, origin_color=[160, 32, 240])
    scene.add_geometry(blade_frame)
    # Add the Final Blade Coordinate Frame
    final_frame = trimesh.creation.axis(origin_size=0.1, transform=T_WB@transforms[-1]@T_WB, axis_length=1.0)
    scene.add_geometry(final_frame)
    # Now get a bounding box for the swept volume that is aligned with the world frame
    bbox_world = new_mesh.bounding_box
    # Add the bounding box corners to the scene
    pc = trimesh.PointCloud(bbox_world.vertices)
    # Visualize the bounding box
    scene.add_geometry(pc)
    scene.show(smooth=False)