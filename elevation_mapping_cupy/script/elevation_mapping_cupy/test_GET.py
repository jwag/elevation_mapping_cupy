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

def get_poly_mesh_boundary(poly_mesh):
    # Extract the boundary (edges, and vertices) of a thin convex polygon mesh
    org_vertices = poly_mesh.vertices
    org_edges = poly_mesh.edges

    # edges which only occur once are on the boundary of the polygon
    # since the triangulation may have subdivided the boundary of the
    # polygon, we need to find it again
    edges_unique = grouping.group_rows(np.sort(org_edges, axis=1), require_count=1)
    # subset the vertices to only ones included in the boundary
    unique, inverse = np.unique(org_edges[edges_unique].reshape(-1), return_inverse=True)
    n_unique = len(unique)
    # take only the vertices in the boundary
    vertices = org_vertices[unique]
    # the indices of vertices in the boundary
    edges = inverse.reshape((-1, 2))
    return {"edges": edges, "vertices": vertices}, unique, n_unique

def hom_inv(T):
    # Get the inverse of a homogeneous transformation matrix
    # The inverse of a homogeneous transformation matrix is the transpose of the rotation matrix
    # and the negative of the translation vector
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T@t
    return T_inv

def line_mesh_intersection(line, mesh, coincidence_tol=1e-6):
    # Define rays for each edge of boundary1
    # The rays are defined by the vertices of the edges
    ray_dirs = line[:,0] - line[:,1]
    ray_origins = line[:,1]

    # Visualize the rays and the mesh
    # stack rays into line segments for visualization as Path3D
    # ray_visualize = trimesh.load_path(np.hstack((ray_origins,
    #                                          ray_origins + ray_dirs*1.0)).reshape(-1, 2, 3))
    # scene = trimesh.Scene([mesh,
    #                    ray_visualize])
    # scene.show()

    # Find the intersection between the edges of boundary1 with mesh2
    # run the mesh- ray query
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs)
    if len(locations) == 0:
        intersections = []
        intersected_lines = []
    else:
        # Now need to determine if the intersection is on the line segment
        # First compute the distance from the origin of the ray to the intersection point
        # Then compute the distance from the origin of the ray to the end of the ray
        # If the distance to the intersection point is less than the distance to the end of the ray
        # then the intersection point is on the line segment
        # Otherwise, the intersection point is not on the line segment
        # Compute the distance from the origin of the ray to the intersection point
        dist_to_intersection = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
        # Compute the distance from the origin of the ray to the end of the ray
        dist_to_end = np.linalg.norm(ray_dirs[index_ray], axis=1)
        # Check if the intersection point is on the line segment
        # If the ray is on the line, but closer than coincedence_tol to the end, 
        # then we treat as a non-intersection. This is to handle numerical issues
        # where the edge is actually coincident with the mesh surface
        # Set coincidence_tol to 0 to disable to detect coincident edges
        on_line = dist_to_end - dist_to_intersection >= coincidence_tol

        if np.any(on_line):
            # Get the indices of the intersected lines
            intersected_lines = index_ray[on_line]
            # Get the intersection point on the line segment
            intersections = locations[intersected_lines]
        else:
            intersections = []
            intersected_lines = []
      
    return intersections, intersected_lines

def side_of_plane(points, plane_normal, plane_origin=None):
    """
    Similar to trimesh.intersections.point_plane_distance
    Determines which side of a plane a set of points are on

    Parameters
    -----------
    points : (n, 3) float
      Points in space
    plane_normal : (3,) float
      Unit normal vector
    plane_origin : (3,) float
      Plane origin in space

    Returns
    ------------
    side : (n,) bool
      Side of plane points are on, True if on positive side, False if on negative side
    """
    points = np.asanyarray(points, dtype=np.float64)
    if plane_origin is None:
        w = points
    else:
        w = points - plane_origin
    # Handle use of this function for a single normal
    # or multiple normals per point
    if plane_normal.shape == (3,):
      vdot = np.dot(plane_normal, w.T)
    else:
      vdot = np.multiply(plane_normal, w).sum(axis=1)
    # Treat points on the plane as positive side
    side = vdot >= 0.0
    return side


def side_of_surface(points, stride):
    """
    Determines which side of a surface defined by points[i*stride:(i+1)*stride]
    the points[(i+1)*stride:(i+2)*stride] are on. The positive normal is defined
    by the cross product of the vector from points[i*stride] to points[i*stride+1]
    and the vector from points[i*stride+1] to points[i*stride+2]
    """
    n_points = len(points)
    n_surfaces = n_points//stride
    inds = np.arange(n_surfaces)*stride
    vec1 = points[inds+1] - points[inds]
    vec2 = points[inds+2] - points[inds+1]
    # Remove the last point from vec1 and vec2 because we only want to check against the prior surface
    vec1 = vec1[:-1]
    vec2 = vec2[:-1]
    normals = np.repeat(np.cross(vec1, vec2), stride, 0)
    origins = points[:-stride]
    sides = side_of_plane(points[stride:], normals, origins).reshape(n_surfaces-1, stride)
    # Check if the points are on the same side of the plane
    # intersected will be true if the points for a given surface are on different sides of the plane
    intersected = np.logical_not(np.all(np.logical_not(np.logical_xor(sides.T,sides.T[0]).T),axis=1))
    # sides will be true if all the points for a given surface are on the positive side of the plane
    # Values at sides[intersected] are not valid and will be false
    sides = np.all(sides, axis=1)
    return sides, intersected

def simple_polygon_triangulation(n_verts):
    # simple triangulation of the faces with cutting out slices of the shape
    # starting with the first vertex and the next two, then the next two, etc
    # Will only work for convex shapes not concave shapes
    faces = np.concatenate((np.zeros((n_verts-2,1),dtype=int),np.lib.stride_tricks.sliding_window_view(np.arange(1,n_verts), 2)), axis=1)
    return faces

def boundary_edges(n_verts):
    # Get the edges of the boundary
    verts = np.zeros((n_verts+1),dtype=int)
    verts[0:n_verts] = np.arange(0,n_verts)
    edges = np.lib.stride_tricks.sliding_window_view(verts, 2)
    return edges

def find_intersections(T12, poly_mesh1=None, poly_mesh2=None, boundary=None, face=None, verts1=None, verts2=None, process=False, separate_surfs=True):
    # Find the intersection between two thin convex polygon meshes
    mesh1_pierces_mesh2 = False
    mesh2_pierces_mesh1 = False
    # TODO: Add checks for sizes of inputs
    if poly_mesh1 is None:
        assert face is not None and boundary is not None and verts1 is not None, "boundary, faces, and verts1 must be provided if poly_mesh1 is not provided"
        poly_mesh1 = Trimesh(vertices=verts1, faces=face, process=process)
    else:
        if boundary is None or verts1 is None or face is None:
            bnd, bnd_face, _ = get_poly_mesh_boundary(poly_mesh1)
            # If any of the mesh boundary, verts, or faces are not provided,
            # Then overwrite them with the boundary, vertices, computed from the mesh
            boundary = bnd["edges"]
            verts1 = bnd["vertices"]
            face = bnd_face[None,:]
    if poly_mesh2 is None:
        assert verts2 is not None, "verts2 must be provided if poly_mesh2 is not provided"
        poly_mesh2 = Trimesh(vertices=verts2, faces=face, process=process)
    else:
        if verts2 is None:
            bnd, _, _ = get_poly_mesh_boundary(poly_mesh2)
            verts2 = bnd["vertices"]
    
    # Face should be a single face of the polygon (not a trimesh face, but a face of the polygon)
    # This will help us to define two new 3D faces/polygons in the case of a self-intersection
    assert face.shape[0] == 1, "faces must be a single face of the polygon"

    # Define lines for each edge of boundary where the first column lines1[:,0,:] is the origin of the ray
    # and the second column lines1[:,1,:] is the end of the ray
    lines1 = np.concatenate((verts1[boundary[:,None, 0]], verts1[boundary[:,None, 1]]), axis=1)

    # Find the intersection between the edges of boundary1 with mesh2
    intersections, intersected_lines = line_mesh_intersection(lines1, poly_mesh2)
    if len(intersections) > 0:
        mesh1_pierces_mesh2 = True
        # Now get the points 
    # If there are no intersections, then check for the intersection between the edges of boundary2 with mesh1
    else:
        # Define lines for each edge of boundary
        lines2 = np.concatenate((verts2[boundary[:,None, 0]], verts2[boundary[:, None, 1]]), axis=1)
        intersections, intersected_lines = line_mesh_intersection(lines2, poly_mesh1)
        if len(intersections) > 0:
            mesh2_pierces_mesh1 = True

    valid_intersect = (mesh1_pierces_mesh2 or mesh2_pierces_mesh1)
    if valid_intersect and not separate_surfs:
        return (), valid_intersect
    elif mesh1_pierces_mesh2:
        pass
    elif mesh2_pierces_mesh1:
        # Split mesh2 into two parts
        print("Mesh2 pierces Mesh1")
        # First let us add the intersection points as vertices to the mesh
        # Add the intersection points to the mesh
        # The suffix indicates wheter or not the mesh is on the pierced side or the piercing side
        intersected_edges = boundary[intersected_lines]
        # Find the face that contain the intersected edges
        # Determine where to split face based on the intersected edges
        def find_intersected_face_inds(face, intersected_edges):
            # Find the indices of the intersected edges in the face
            intersected_face_inds = np.zeros(len(intersected_edges), dtype=int)
            # Add the first vertex to the end of the face to make it a closed loop
            face_circ = np.concatenate((face[0], face[0,0:1]), axis=0)
            for e, edge in enumerate(intersected_edges):
                # TODO: May have to deal with inverted edges too
                    for i in np.arange(len(face_circ)-1):
                        if np.all(face_circ[i:i+2] == edge):
                            # This face contains the intersected edge
                            intersected_face_inds[e] = i
            return intersected_face_inds
        intersected_face_inds = find_intersected_face_inds(face, intersected_edges)
        # The face now needs split into two faces
        # The first face will start with the first intersection point, include points from the face
        # between the first and second intersection points, and end with the second intersection point
        face_circ = np.concatenate((face[0], face[0,0:1]), axis=0)
        mesh2_face1_part = face_circ[intersected_face_inds[0]+1:intersected_face_inds[1]+1]
        mesh2_verts1_part = verts2[mesh2_face1_part]
        # Determine which side of the mesh1 the mesh2_verts1_part are on
        # This indicates whether this is a negative swept volume or a positive swept volume
        mesh2_side1 = side_of_plane(mesh2_verts1_part, poly_mesh1.face_normals[0], poly_mesh1.vertices[0])
        # Make sure all the points are on the same side of the plane
        if not np.all(mesh2_side1 == mesh2_side1[0]):
            # If the points are not all on the same side of the plane, then the intersection is not valid
            raise ValueError("The intersection is not valid. The intersection points are not all on the same side of the plane")
        mesh2_side1 = mesh2_side1[0]
        # Add the intersection points to the vertices
        mesh2_verts1 = np.concatenate((intersections[:1],mesh2_verts1_part, intersections[1:]), axis=0)
        # face1 = np.arange(len(mesh2_verts1), dtype=int)[None,:] # Define a single face
        face1 = simple_polygon_triangulation(len(mesh2_verts1))
        boundary1 = boundary_edges(len(mesh2_verts1))

        # The second face will be from the first vertex of the face to the first intersected edge
        # with the new vertices inserted, then the remaining vertices of the face that were not intersected
        mesh2_face2_part1 = face_circ[:intersected_face_inds[0]+1]
        mesh2_face2_part2 = face_circ[intersected_face_inds[1]+1:-1]
        mesh2_verts2_part1 = verts2[mesh2_face2_part1]
        mesh2_verts2_part2 = verts2[mesh2_face2_part2]
        # Determine which side of the mesh1 the mesh2_verts1_part are on
        # This indicates whether this is a negative swept volume or a positive swept volume
        mesh2_side2_part1 = side_of_plane(mesh2_verts2_part1, poly_mesh1.face_normals[0], poly_mesh1.vertices[0])
        mesh2_side2_part2 = side_of_plane(mesh2_verts2_part2, poly_mesh1.face_normals[0], poly_mesh1.vertices[0])
        # Make sure all the points are on the same side of the plane
        if not np.all(mesh2_side2_part1 == mesh2_side2_part1[0]) or not np.all(mesh2_side2_part2 == mesh2_side2_part2[0]):
            # If the points are not all on the same side of the plane, then the intersection is not valid
            raise ValueError("The intersection is not valid. The intersection points are not all on the same side of the plane")
        mesh2_side2 = mesh2_side2_part1[0]

        mesh2_verts2 = np.concatenate((mesh2_verts2_part1, intersections, mesh2_verts2_part2), axis=0)
        # face2 = np.arange(len(mesh2_verts2), dtype=int)[None,:] # Define a single face
        face2 = simple_polygon_triangulation(len(mesh2_verts2))
        boundary2 = boundary_edges(len(mesh2_verts2))
        # TODO: Define mapping between the original face and the two new faces
        # Now create the two new meshes
        # mesh2_part1 = Trimesh(vertices=mesh2_verts1, faces=face1, face_colors=[255, 0, 0, 255])
        # mesh2_part2 = Trimesh(vertices=mesh2_verts2, faces=face2, face_colors=[0, 255, 0, 255],vertex_colors=[0, 0, 255, 255])
        # Visualize the two new meshes
        # scene = trimesh.Scene([mesh2_part1, mesh2_part2])
        # scene.show()

        ########################################
        # Now we need to split mesh1
        # The faces will be the same as mesh2_part1, but the vertices will be different
        mesh1_face1_part = mesh2_face1_part
        mesh1_verts1_part = verts1[mesh1_face1_part]
        # Add the intersection points to the vertices
        T21 = hom_inv(T12)
        # May need to use the length along the ray to determine the corresponding intersection
        # point on the other mesh if there are numerical issues with using the transformed intersection points
        intersections_proj = (T21[:3, :3]@intersections.T + T21[:3, 3:]).T
        mesh1_verts1 = np.concatenate((intersections_proj[:1],mesh1_verts1_part, intersections_proj[1:]), axis=0)
        # mesh1_faces1 = np.arange(len(mesh1_verts1), dtype=int)[None,:] # Define a single face # should be same as face1
        # Edges of the swpet volume for the first surfaces will be defined 1:1 as they are the same geometry.
        # Define the second face of mesh1
        # This is a bit more complicated since this face has been pierced and the intersection edge is in the middle
        # of the face. If we treated this as one surface, then this shape would be concave and not convex.
        # This means that specifying the face as a single face would not work as the triangulation does not support
        # concave shapes. Alternatively, we could not remove this hole in the face between intersectons and intersections_proj.
        # Then the face would be convex and mirror that of mesh2_part2. The other solution is to triangulate the concave
        # shape using trimesh.creation.triangulate_polygon() for example.
        mesh1_face2_part1 = mesh2_face2_part1
        mesh1_verts2_part1 = verts1[mesh1_face2_part1]
        mesh1_face2_part2 = mesh2_face2_part2
        mesh1_verts2_part2 = verts1[mesh1_face2_part2]
        # method that includes the hole in the face commented out below. Not working for concave shapes right now
        # mesh1_verts2 = np.concatenate((mesh1_verts2_part1, intersections_proj[:1],
        #                                 intersections, intersections_proj[1:],mesh1_verts2_part2), axis=0)
        # Method that removes the hole in the face. This avoids the concave shape issue
        mesh1_verts2 = np.concatenate((mesh1_verts2_part1, intersections_proj, mesh1_verts2_part2), axis=0)
        # mesh1_faces2 = np.arange(len(mesh1_verts2), dtype=int)[None,:] # Define a single face # should be same as face2

        # Now create the two new meshes
        # mesh1_part1 = Trimesh(vertices=mesh1_verts1, faces=mesh1_faces1, face_colors=[255, 0, 0, 255])
        # mesh1_part2 = Trimesh(vertices=mesh1_verts2, faces=mesh1_faces2, face_colors=[0, 255, 0, 255],vertex_colors=[0, 0, 255, 255])
        # Visualize the two new meshes
        # scene = trimesh.Scene([mesh1_part1, mesh1_part2, mesh2_part1, mesh2_part2])
        # scene.show()

        # TODO: figure out how to identify positive and negative sweeps given the normal maybe 
        verts1 = np.concatenate((mesh1_verts1, mesh2_verts1), axis=0)
        verts2 = np.concatenate((mesh1_verts2, mesh2_verts2), axis=0)

        assert mesh2_side1 ==  (not mesh2_side2), "mesh2_side1 must be opposite to mesh2_side2"

        if mesh2_side1:
            pos_verts = verts1
            pos_boundary = boundary1
            neg_verts = verts2
            neg_boundary = boundary2
        else:
            pos_verts = verts2
            pos_boundary = boundary2
            neg_verts = verts1
            neg_boundary = boundary1
    
    return (pos_verts, pos_boundary, neg_verts, neg_boundary), valid_intersect

def get_swept_edges(boundary_edges, roll_dir, convex_interp=True, flip_normals=False):
    """
    Get the faces of a swept volume given the boundary edges and the roll direction
    face_sweep can be concatenated with initial and transformed boundary faces to create the swept volume
    This requires appropriate offsets to be applied to the face_sweep
    Creates two faces for each extruded edge of the boundary to produce appropriate triangulation
    """
    n_edges = boundary_edges.shape[0]
    boundary_next = boundary_edges + n_edges
    if not convex_interp:
        # if we're interpolating concave hulls we need to flip the sign of roll_dirs
        roll_dir = not roll_dir
    if roll_dir:
        # positive roll
        face_sweep = np.column_stack(
            [boundary_edges, boundary_next[:, :1], boundary_next[:, ::-1], boundary_edges[:, 1:]]
        ).reshape((-1, 3))
    else:
        # negative roll
        face_sweep = np.column_stack(
            [boundary_edges, boundary_next[:, -1:], boundary_next[:, ::-1], boundary_edges[:, :-1]]
        ).reshape((-1, 3))
    if flip_normals:
        face_sweep = np.fliplr(face_sweep)
    return face_sweep, n_edges

def get_cap_face(boundary_edges, flip_normals=False):
    """
    Get the faces (a single surface) of a cap for a swept volume given the boundary edges
    """
    unique = np.unique(boundary_edges.ravel())
    face_cap = simple_polygon_triangulation(len(unique))
    if flip_normals:
        face_cap = np.fliplr(face_cap)
    return face_cap


def sweep_thin_poly_mesh(
    poly_mesh: Trimesh,
    transforms: ArrayLike,
    roll_dirs: ArrayLike,
    convex_interp: bool = True,
    cap: bool = True,
    check_intersects: bool = True,
    separate_surfs: bool = True,
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
    transforms : (n, 4, 4) float
      A sequence of transforms to apply to the poly_mesh. Must be at least 1 transform.
    roll_dirs : (n,) bool
      The direction of roll between each slice. True for positive roll, False for negative roll.
    convex_interp : bool
      If True, interpolate convex hulls between each slice. If False, interpolate concave hulls.
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
    if not (util.is_shape(transforms, (-1, 4, 4)) and len(transforms) >= 1):
        raise ValueError("transforms must be (n, 4, 4)!")
    
    n_sweeps = len(transforms)

    # check to see if path is closed i.e. first and last vertex are the same
    closed = np.linalg.norm(transforms[0] - transforms[-1]) < tol.merge
    # Extract 2D vertices and triangulation
    org_faces = poly_mesh.faces
    
    # Get boundary of the polygon
    bnd, unique, n_unique = get_poly_mesh_boundary(poly_mesh)
    boundary = bnd["edges"]
    verts = bnd["vertices"]
    stride = n_unique

    # take only the vertices in the boundary
    # and stack them with zeros and ones so we can use dot
    # products to transform them all over the place
    vertices_tf = np.column_stack(
        (verts, np.ones(stride))
    )

    # apply transforms to prebaked homogeneous coordinates
    # TODO: Speed this up with einsum or similar
    vertices_3D = np.concatenate(
        [np.dot(vertices_tf, matrix.T) for matrix in transforms], axis=0
    )[:, :3]
    # Add in the original vertices to the beginning of the vertices_3D array
    vertices_3D = np.concatenate((verts, vertices_3D), axis=0)

    sides, intersected = side_of_surface(vertices_3D, stride)

    # Check for self-intersections between the slices
    # TODO: Loop through them all
    # TODO: Figure out this whole transform issue where initial mesh is translated and rotated...
    # Using unique to define the single face of the polygon (not a trimesh face, but a face of the polygon)
    # This will help us to define two new 3D faces/polygons in the case of a self-intersection
    if check_intersects:
        # Intializations
        pos_verts = np.empty((0,3))
        pos_faces = np.empty((0,3))
        neg_verts = np.empty((0,3))
        neg_faces = np.empty((0,3))
        last_pos_sweep, last_neg_sweep = 0, 0
        for i in range(n_sweeps):
            # Only look for intersections where we already know they are
            if intersected[i]:
                if i == 0:
                    poly_mesh1 = poly_mesh
                else:
                    poly_mesh1 = None
                intersects, valid_intersect = find_intersections(transforms[i], poly_mesh1=poly_mesh1,
                                                                  boundary=boundary, face=unique[None,:],
                                                                  verts1=vertices_3D[stride*(i):stride*(i+1)],
                                                                  verts2 = vertices_3D[stride*(i+1):stride*(i+2)],
                                                                  process=False, separate_surfs=True)
                if valid_intersect:
                    p_verts, p_boundary, n_verts, n_boundary = intersects
                    pos_face_sweep, n_new_pos_points = get_swept_edges(p_boundary, roll_dirs[i], convex_interp, flip_normals = False)
                    offset = len(pos_verts)
                    # Add the verticies for the positive sweep
                    pos_verts = np.concatenate((pos_verts, p_verts[0:n_new_pos_points]), axis=0)
                    # Add the start cap faces (always?)
                    cap_face = get_cap_face(p_boundary, flip_normals = True)
                    pos_faces = np.concatenate((pos_faces, cap_face+offset), axis=0)
                    # Add the swept faces
                    offset = len(pos_verts)-n_new_pos_points
                    # TODO: Could combine with above concat
                    pos_verts = np.concatenate((pos_verts, p_verts[n_new_pos_points:]), axis=0)
                    pos_faces = np.concatenate((pos_faces, pos_face_sweep+offset), axis=0)
                    # Add cap faces at the end of the sweep to close the volume
                    cap_face = get_cap_face(p_boundary, flip_normals = False)
                    pos_faces = np.concatenate((pos_faces, cap_face+len(pos_verts)-n_new_pos_points), axis=0)

                    # Add the verticies for the negative sweep
                    neg_face_sweep, n_new_neg_points = get_swept_edges(n_boundary, roll_dirs[i], convex_interp, flip_normals = True)
                    offset = len(neg_verts)
                    # Add the verticies for the negative sweep
                    neg_verts = np.concatenate((neg_verts, n_verts[0:n_new_neg_points]), axis=0)
                    # Add the start cap faces (always?)
                    cap_face = get_cap_face(n_boundary, flip_normals = False)
                    neg_faces = np.concatenate((neg_faces, cap_face+offset), axis=0)
                    # Add the swept faces
                    offset = len(neg_verts)-n_new_neg_points
                    neg_verts = np.concatenate((neg_verts, n_verts[n_new_neg_points:]), axis=0)
                    neg_faces = np.concatenate((neg_faces, neg_face_sweep+offset), axis=0)
                    # Add cap faces at the end of the sweep to close the volume
                    cap_face = get_cap_face(n_boundary, flip_normals = True)
                    neg_faces = np.concatenate((neg_faces, cap_face+len(neg_verts)-n_new_neg_points), axis=0)

            else:
                face_sweep, n_new_points = get_swept_edges(boundary, roll_dirs[i], convex_interp, flip_normals = not sides[i])
                assert n_new_points == stride, "The number of new points must be equal to the stride"
                if sides[i]:
                    # Add the past verticies if either this is the first sweep, the previous sweep was negative, or the previous sweep intersected
                    if len(pos_verts) == 0 or not sides[i-1] or intersected[i-1]:
                        offset = len(pos_verts)
                        pos_verts = np.concatenate((pos_verts, vertices_3D[stride*(i):stride*(i+1)]),axis=0)
                        # Add cap faces at the beginning of the sweep to close the volume on one end
                        cap_face = get_cap_face(boundary, flip_normals = True)
                        pos_faces = np.concatenate((pos_faces, cap_face+offset), axis=0)
                    offset = len(pos_verts)-stride
                    pos_verts = np.concatenate((pos_verts, vertices_3D[stride*(i+1):stride*(i+2)]),axis=0)
                    pos_faces = np.concatenate((pos_faces, face_sweep+offset), axis=0)
                    last_pos_sweep = i+1
                else:
                    # Add the past verticies if either this is the first sweep, the previous sweep was positive, or the previous sweep intersected
                    if len(neg_verts) == 0 or sides[i-1] or intersected[i-1]:
                        offset = len(neg_verts)
                        neg_verts = np.concatenate((neg_verts, vertices_3D[stride*(i):stride*(i+1)]),axis=0)
                        # Add cap faces at the beginning of the sweep to close the volume on one end
                        cap_face = get_cap_face(boundary, flip_normals = False)
                        neg_faces = np.concatenate((neg_faces, cap_face+offset), axis=0)
                    offset = len(neg_verts)-stride
                    neg_verts = np.concatenate((neg_verts, vertices_3D[stride*(i+1):stride*(i+2)]),axis=0)
                    neg_faces = np.concatenate((neg_faces, face_sweep+offset), axis=0)
                    last_neg_sweep = i+1

        # Cap Faces of both volumes
        # Only cap the end of the positive sweep if the last sweep was positive
        # otherwise, the cap face will be added when the negative sweep is added
        if len(pos_verts) != 0 and last_pos_sweep != 0 and last_pos_sweep == n_sweeps:
            # Handle differently if dealing with intersections
            cap_face = get_cap_face(boundary, flip_normals = False)
            pos_faces = np.concatenate((pos_faces, cap_face+len(pos_verts)-stride*last_pos_sweep), axis=0)
        elif len(neg_verts) != 0 and last_neg_sweep != 0 and last_neg_sweep == n_sweeps:
            cap_face = get_cap_face(boundary, flip_normals = True)
            neg_faces = np.concatenate((neg_faces, cap_face+len(neg_verts)-stride*last_neg_sweep), axis=0)


    # # Cap the ends of the swept volumes
    # # these are indices of `vertices_2D` that were not on the boundary
    # # which can happen for triangulation algorithms that added vertices
    # # we don't currently support that but you could append the unconsumed
    # # vertices and then update the mapping below to reflect that
    # unconsumed = set(unique).difference(org_faces.ravel())
    # if len(unconsumed) > 0:
    #     raise NotImplementedError("triangulation added vertices: no logic to cap!")

    # # map the 2D faces to the order we used
    # mapped = np.zeros(unique.max() + 2, dtype=np.int64)
    # mapped[unique] = np.arange(len(unique))

    # # now should correspond to the first vertex block
    # cap_zero = mapped[org_faces]
    # # winding will be along +Z so flip for the bottom cap
    # # faces.append(np.fliplr(cap_zero))
    # faces = np.concatenate((faces, np.fliplr(cap_zero)), axis=0)
    # # offset the end cap
    # # faces.append(cap_zero + stride * n_sweeps)
    # faces = np.concatenate((faces, cap_zero + stride * n_sweeps), axis=0)
    # # Define face_colors for mesh where the original face is green,
    # # the swept faces are grey, and the final face is red
    # alpha = 125
    # n_faces = len(faces)
    # n_org_faces = len(org_faces)
    # face_colors = np.ones((n_faces, 4), dtype=int) * 169 # Set color to grey
    # face_colors[n_faces-n_org_faces*2:-n_org_faces,:3] = [0, 255, 0]
    # face_colors[-n_org_faces:,:3] = [255, 0, 0]
    # face_colors[:, 3] = alpha # Set transparency to alpha
    # # face_colors[0:n_faces-n_org_faces*2,3] = 0 # Set transparency to 0 for swept faces

    if kwargs is None:
        kwargs = {}

    if "process" not in kwargs:
        # we should be constructing clean meshes here
        # so we don't need to run an expensive verex merge
        kwargs["process"] = False

    # generate the mesh from the face data
    alpha = 126
    if len(pos_verts) != 0:
        pos_face_color = [0, 255, 0, alpha]
        pos_swept_mesh = Trimesh(vertices=pos_verts, faces=pos_faces, face_colors=pos_face_color, **kwargs)
    else:
        pos_swept_mesh = None
    if len(neg_verts) != 0:
        neg_face_color = [255, 0, 0, alpha]
        neg_swept_mesh = Trimesh(vertices=neg_verts, faces=neg_faces, face_colors=neg_face_color, **kwargs)
    else:
        neg_swept_mesh = None

    # if tol.strict:
    #     # we should not have included any unused vertices
    #     assert len(np.unique(faces)) == len(vertices_3D)

    #     if cap:
    #         # mesh should always be a volume if cap is true
    #         assert swept_mesh.is_volume

    #     if closed and connect:
    #         assert swept_mesh.is_volume
    #         assert swept_mesh.body_count == 1

    return pos_swept_mesh, neg_swept_mesh

if __name__ == "__main__":
    # Create the blade geometry
    blade_origin=[1.634, 0.0, 0.060+0.265]
    # blade_origin=[0.0, 0.0, 0.0]
    blade_mesh, T_WB, faces_front = simble_blade_geometry(blade_width=3.0, blade_height=0.6, blade_angle_deg=-10, blade_origin=blade_origin)

    T_BW = hom_inv(T_WB)
    # Define the path of the blade
    # T_dB = trimesh.transformations.translation_matrix([1.5, -1.5, 0.0])
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [1, 0, 0])@T_dB
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-10), [0, 1, 0])@T_dB
    # T_dB2 = trimesh.transformations.translation_matrix([1.5, 0.0, 0.0])@T_dB
    # T_dB3 = trimesh.transformations.translation_matrix([1.0, 0.0, 0.0])
    # # T_db3 = trimesh.transformations.rotation_matrix(np.radians(10), [1, 0, 0])@T_dB3
    # # T_dB3 = trimesh.transformations.rotation_matrix(np.radians(15), [0, 0, 1])@T_dB3@T_dB2
    # roll_dirs = np.array([-20, 0, 0]) >= 0
    # transforms = np.array([T_dB, T_dB2, T_dB3])
    # # roll_dirs = np.array([-20]) >= 0
    # # transforms = np.array([T_dB])


    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [0, 0, 1])@T_BW
    # roll_dirs = np.array([-20]) >= 0
    # transforms = np.array([T_BW, T_dB])

    # Applying T_BW to transforms so that i can apply the transforms to the blade surface
    # directly. This is because the blade surface is defined in the blade frame
    # The blade_mesh is defined in the world frame so the transform T_BW must 
    T_dB = trimesh.transformations.translation_matrix([0.3, 0.4, 0.2])
    T_dB = trimesh.transformations.rotation_matrix(np.radians(22), [1, 0, 0])@T_dB
    T_dB = trimesh.transformations.rotation_matrix(np.radians(-15), [0, 1, 0])@T_dB
    T_dB = trimesh.transformations.rotation_matrix(np.radians(-15), [0, 0, 1])@T_dB
    # T_dB2 = trimesh.transformations.translation_matrix([-1.5, 0.0, 0.0])@T_dB
    # roll_dirs = np.array([22,0]) >= 0
    # transforms = np.array([T_dB, T_dB2])
    roll_dirs = np.array([22]) >= 0
    transforms = np.array([T_dB])

    pos_new_mesh, neg_new_mesh = sweep_thin_poly_mesh(blade_mesh.apply_transform(T_BW), transforms, roll_dirs=roll_dirs, convex_interp=True, cap=True, connect=False)
    meshes = []
    if pos_new_mesh is not None:
        pos_new_mesh.apply_transform(T_WB)
        meshes.append(pos_new_mesh)
    if neg_new_mesh is not None:
        neg_new_mesh.apply_transform(T_WB)
        meshes.append(neg_new_mesh)
    # trimesh.util.concatenate(new_mesh.split(only_watertight=True))
    # new_mesh.show(smooth=False)
    # scene = trimesh.Scene([new_mesh.convex_hull])
    scene = trimesh.Scene([meshes])
    # Add a world coordinate frame to the scene
    world_frame = trimesh.creation.axis(origin_size=0.1, axis_length=1.0)
    scene.add_geometry(world_frame)
    # Add blade coordinate frame to the scene
    # Make origin size larger and color purple
    blade_frame = trimesh.creation.axis(origin_size=0.1, transform=T_WB, axis_length=1.0, origin_color=[160, 32, 240])
    scene.add_geometry(blade_frame)
    # Add the Final Blade Coordinate Frame
    final_frame = trimesh.creation.axis(origin_size=0.1, transform=T_WB@transforms[-1], axis_length=1.0)
    scene.add_geometry(final_frame)
    # Now get a bounding box for the swept volume that is aligned with the world frame
    # bbox_world = new_mesh.bounding_box
    # # Add the bounding box corners to the scene
    # pc = trimesh.PointCloud(bbox_world.vertices)
    # # Visualize the bounding box
    # scene.add_geometry(pc)
    use_wireframe = False
    scene.show(smooth=False, flags={'wireframe': use_wireframe})