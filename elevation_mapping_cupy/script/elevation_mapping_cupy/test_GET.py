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
    

def find_intersections(T12, poly_mesh1=None, poly_mesh2=None, boundary=None, face=None, verts1=None, verts2=None):
    # Find the intersection between two thin convex polygon meshes
    mesh1_pierces_mesh2 = False
    mesh2_pierces_mesh1 = False
    # TODO: Add checks for sizes of inputs
    if poly_mesh1 is None:
        assert face is not None and boundary is not None and verts1 is not None, "boundary, faces, and verts1 must be provided if poly_mesh1 is not provided"
        poly_mesh1 = Trimesh(vertices=verts1, faces=face)
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
        poly_mesh2 = Trimesh(vertices=verts2, faces=face)
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
    
    if mesh1_pierces_mesh2:
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
        # Add the intersection points to the vertices
        mesh2_verts1 = np.concatenate((intersections[:1],mesh2_verts1_part, intersections[1:]), axis=0)
        mesh2_faces1 = np.arange(len(mesh2_verts1), dtype=int)[None,:] # Define a single face
        # The second face will be from the first vertex of the face to the first intersected edge
        # with the new vertices inserted, then the remaining vertices of the face that were not intersected
        mesh2_face2_part1 = face_circ[:intersected_face_inds[0]+1]
        mesh2_face2_part2 = face_circ[intersected_face_inds[1]+1:-1]
        mesh2_verts2_part1 = verts2[mesh2_face2_part1]
        mesh2_verts2_part2 = verts2[mesh2_face2_part2]
        mesh2_verts2 = np.concatenate((mesh2_verts2_part1, intersections, mesh2_verts2_part2), axis=0)
        mesh2_faces2 = np.arange(len(mesh2_verts2), dtype=int)[None,:] # Define a single face
        # TODO: Define mapping between the original face and the two new faces
        # Now create the two new meshes
        mesh2_part1 = Trimesh(vertices=mesh2_verts1, faces=mesh2_faces1, face_colors=[255, 0, 0, 255])
        mesh2_part2 = Trimesh(vertices=mesh2_verts2, faces=mesh2_faces2, face_colors=[0, 255, 0, 255],vertex_colors=[0, 0, 255, 255])
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
        mesh1_faces1 = np.arange(len(mesh1_verts1), dtype=int)[None,:] # Define a single face
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
        # method that includes the hole in the face. Not working for concave shapes right now
        # mesh1_verts2 = np.concatenate((mesh1_verts2_part1, intersections_proj[:1],
        #                                 intersections, intersections_proj[1:],mesh1_verts2_part2), axis=0)
        # Method that removes the hole in the face. This avoids the concave shape issue
        mesh1_verts2 = np.concatenate((mesh1_verts2_part1, intersections_proj, mesh1_verts2_part2), axis=0)
        mesh1_faces2 = np.arange(len(mesh1_verts2), dtype=int)[None,:] # Define a single face
        # Now create the two new meshes
        mesh1_part1 = Trimesh(vertices=mesh1_verts1, faces=mesh1_faces1, face_colors=[255, 0, 0, 255])
        mesh1_part2 = Trimesh(vertices=mesh1_verts2, faces=mesh1_faces2, face_colors=[0, 255, 0, 255],vertex_colors=[0, 0, 255, 255])
        # Visualize the two new meshes
        # scene = trimesh.Scene([mesh1_part1, mesh1_part2])#,mesh2_part1, mesh2_part2])
        # scene.show()

        # Debug mesh
        # debug_mesh_verts = np.concatenate((intersections_proj[:1],
        #                                 intersections, intersections_proj[1:]),axis=0)
        # debug_mesh_faces = np.arange(len(debug_mesh_verts), dtype=int)[None,:]
        # # flip the face so it is visible from the front
        # debug_mesh_faces = np.fliplr(debug_mesh_faces)
        # debug_mesh = Trimesh(vertices=debug_mesh_verts, faces=debug_mesh_faces, face_colors=[255, 0, 255, 255])
        # scene = trimesh.Scene([debug_mesh, mesh1_part1, mesh1_part2, mesh2_part1, mesh2_part2])
        # scene.show()

        scene = trimesh.Scene([mesh1_part1, mesh1_part2, mesh2_part1, mesh2_part2])
        scene.show()

        test=1

        pass

    raise NotImplementedError("Function not fully implemented yet")


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
    transforms : (n, 4, 4) float
      A sequence of transforms to apply to the poly_mesh. Must be at least 1 transform.
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

    # take only the vertices in the boundary
    # and stack them with zeros and ones so we can use dot
    # products to transform them all over the place
    vertices_tf = np.column_stack(
        (verts, np.ones(n_unique))
    )

    # apply transforms to prebaked homogeneous coordinates
    # TODO: Speed this up with einsum or similar
    vertices_3D = np.concatenate(
        [np.dot(vertices_tf, matrix.T) for matrix in transforms], axis=0
    )[:, :3]
    # Add in the original vertices to the beginning of the vertices_3D array
    vertices_3D = np.concatenate((verts, vertices_3D), axis=0)

    # Check for self-intersections between the slices
    # TODO: Loop through them all
    # TODO: Figure out this whole transform issue where initial mesh is translated and rotated...
    # Using unique to define the single face of the polygon (not a trimesh face, but a face of the polygon)
    # This will help us to define two new 3D faces/polygons in the case of a self-intersection
    find_intersections(transforms[0],
                       poly_mesh1=poly_mesh, boundary=boundary, face=unique[None,:],
                       verts1=vertices_3D[0:n_unique],
                       verts2 = vertices_3D[n_unique:n_unique*2])

    # now construct the faces with one group of boundary faces per slice
    stride = n_unique
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
        # face_colors[0:n_faces-n_org_faces*2,3] = 0 # Set transparency to 0 for swept faces

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

    T_BW = hom_inv(T_WB)
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
    roll_dirs = np.array([22]) >= 0
    transforms = np.array([T_dB])

    new_mesh = sweep_thin_poly_mesh(blade_mesh.apply_transform(T_BW), transforms, roll_dirs=roll_dirs, convex_interp=True, cap=True, connect=False)
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
    final_frame = trimesh.creation.axis(origin_size=0.1, transform=T_WB@transforms[-1], axis_length=1.0)
    scene.add_geometry(final_frame)
    # Now get a bounding box for the swept volume that is aligned with the world frame
    bbox_world = new_mesh.bounding_box
    # Add the bounding box corners to the scene
    pc = trimesh.PointCloud(bbox_world.vertices)
    # Visualize the bounding box
    scene.add_geometry(pc)
    use_wireframe = True
    scene.show(smooth=False, flags={'wireframe': use_wireframe})