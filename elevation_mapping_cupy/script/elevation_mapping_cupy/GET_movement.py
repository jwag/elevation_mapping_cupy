import numpy as np
import trimesh

from trimesh.base import Trimesh
from trimesh import grouping, util
from trimesh.typed import ArrayLike, Dict, Optional
from trimesh.constants import tol
import time
import warnings

import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


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
            intersections = locations
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

def split_intersected_meshes(face, intersections, intersected_edges, piercing_verts, pierced_verts, pierced_mesh, T_piercing_pierced):
    # Pierced mesh is the mesh that is being pierced by the piercing mesh, i.e. it creates a line of intersection in the middle of the face
    # of the pierced mesh. Both meshes must be split into two parts to create a positive and negative swept volume.

    # Convenience function
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
                        break
        # # sort the indices
        # intersected_face_inds = np.sort(intersected_face_inds)
        return intersected_face_inds
    
    # First split piercing_mesh into two parts
    # Let us add the intersection points as vertices to the mesh
    # Find the face that contain the intersected edges
    # Determine where to split face based on the intersected edges
    intersected_face_inds = find_intersected_face_inds(face, intersected_edges)
    # The face now needs split into two faces
    # The first face will start with the first intersection point, include points from the face
    # between the first and second intersection points, and end with the second intersection point
    face_circ = np.concatenate((face[0], face[0,0:1]), axis=0)
    piercing_mesh_face1_part = face_circ[intersected_face_inds[0]+1:intersected_face_inds[1]+1]
    piercing_mesh_verts1_part = piercing_verts[piercing_mesh_face1_part]
    # Determine which side of the mesh1 the piercing_mesh_verts1_part are on
    # This indicates whether this is a negative swept volume or a positive swept volume
    piercing_mesh_side1 = side_of_plane(piercing_mesh_verts1_part, pierced_mesh.face_normals[0], pierced_mesh.vertices[0])
    # Make sure all the points are on the same side of the plane
    if not np.all(piercing_mesh_side1 == piercing_mesh_side1[0]):
        # If the points are not all on the same side of the plane, then the intersection is not valid
        raise ValueError("The intersection is not valid. The intersection points are not all on the same side of the plane")
    piercing_mesh_side1 = piercing_mesh_side1[0]
    # Add the intersection points to the vertices
    piercing_mesh_verts1 = np.concatenate((intersections[:1],piercing_mesh_verts1_part, intersections[1:]), axis=0)
    # face1 = np.arange(len(piercing_mesh_verts1), dtype=int)[None,:] # Define a single face
    face1 = simple_polygon_triangulation(len(piercing_mesh_verts1))
    boundary1 = boundary_edges(len(piercing_mesh_verts1))

    # The second face will be from the first vertex of the face to the first intersected edge
    # with the new vertices inserted, then the remaining vertices of the face that were not intersected
    piercing_mesh_face2_part1 = face_circ[:intersected_face_inds[0]+1]
    piercing_mesh_face2_part2 = face_circ[intersected_face_inds[1]+1:-1]
    piercing_mesh_verts2_part1 = piercing_verts[piercing_mesh_face2_part1]
    piercing_mesh_verts2_part2 = piercing_verts[piercing_mesh_face2_part2]
    # Determine which side of the mesh1 the piercing_mesh_verts1_part are on
    # This indicates whether this is a negative swept volume or a positive swept volume
    piercing_mesh_side2_part1 = side_of_plane(piercing_mesh_verts2_part1, pierced_mesh.face_normals[0], pierced_mesh.vertices[0])
    piercing_mesh_side2_part2 = side_of_plane(piercing_mesh_verts2_part2, pierced_mesh.face_normals[0], pierced_mesh.vertices[0])
    # Make sure all the points are on the same side of the plane
    part1_invalid = False
    part2_invalid = False
    if piercing_mesh_side2_part1.shape[0] !=0:
        part1_invalid =  (not np.all(piercing_mesh_side2_part1 == piercing_mesh_side2_part1[0]))
    if piercing_mesh_side2_part2.shape[0] != 0:
        part2_invalid = (not np.all(piercing_mesh_side2_part2 == piercing_mesh_side2_part2[0]))
    if part1_invalid or part2_invalid:
        # If the points are not all on the same side of the plane, then the intersection is not valid
        raise ValueError("The intersection is not valid. The intersection points are not all on the same side of the plane")
    piercing_mesh_side2 = piercing_mesh_side2_part1[0]

    piercing_mesh_verts2 = np.concatenate((piercing_mesh_verts2_part1, intersections, piercing_mesh_verts2_part2), axis=0)
    # face2 = np.arange(len(piercing_mesh_verts2), dtype=int)[None,:] # Define a single face
    face2 = simple_polygon_triangulation(len(piercing_mesh_verts2))
    boundary2 = boundary_edges(len(piercing_mesh_verts2))
    # TODO: Define mapping between the original face and the two new faces
    # Now create the two new meshes
    # piercing_mesh_part1 = Trimesh(vertices=piercing_mesh_verts1, faces=face1, face_colors=[255, 0, 0, 255])
    # piercing_mesh_part2 = Trimesh(vertices=piercing_mesh_verts2, faces=face2, face_colors=[0, 255, 0, 255],vertex_colors=[0, 0, 255, 255])
    # Visualize the two new meshes
    # scene = trimesh.Scene([piercing_mesh_part1, piercing_mesh_part2])
    # scene.show()

    ########################################
    # Now we need to split mesh1
    # The faces will be the same as piercing_mesh_part1, but the vertices will be different
    pierced_mesh_face1_part = piercing_mesh_face1_part
    pierced_mesh_verts1_part = pierced_verts[pierced_mesh_face1_part]
    # Add the intersection points to the vertices
    # T21 = hom_inv(T12)
    # May need to use the length along the ray to determine the corresponding intersection
    # point on the other mesh if there are numerical issues with using the transformed intersection points
    intersections_proj = (T_piercing_pierced[:3, :3]@intersections.T + T_piercing_pierced[:3, 3:]).T
    pierced_mesh_verts1 = np.concatenate((intersections_proj[:1],pierced_mesh_verts1_part, intersections_proj[1:]), axis=0)
    # pierced_mesh_faces1 = np.arange(len(pierced_mesh_verts1), dtype=int)[None,:] # Define a single face # should be same as face1
    # Edges of the swpet volume for the first surfaces will be defined 1:1 as they are the same geometry.
    # Define the second face of mesh1
    # This is a bit more complicated since this face has been pierced and the intersection edge is in the middle
    # of the face. If we treated this as one surface, then this shape would be concave and not convex.
    # This means that specifying the face as a single face would not work as the triangulation does not support
    # concave shapes. Alternatively, we could not remove this hole in the face between intersectons and intersections_proj.
    # Then the face would be convex and mirror that of piercing_mesh_part2. The other solution is to triangulate the concave
    # shape using trimesh.creation.triangulate_polygon() for example.
    pierced_mesh_face2_part1 = piercing_mesh_face2_part1
    pierced_mesh_verts2_part1 = pierced_verts[pierced_mesh_face2_part1]
    pierced_mesh_face2_part2 = piercing_mesh_face2_part2
    pierced_mesh_verts2_part2 = pierced_verts[pierced_mesh_face2_part2]
    # method that includes the hole in the face commented out below. Not working for concave shapes right now
    # pierced_mesh_verts2 = np.concatenate((pierced_mesh_verts2_part1, intersections_proj[:1],
    #                                 intersections, intersections_proj[1:],pierced_mesh_verts2_part2), axis=0)
    # Method that removes the hole in the face. This avoids the concave shape issue
    pierced_mesh_verts2 = np.concatenate((pierced_mesh_verts2_part1, intersections_proj, pierced_mesh_verts2_part2), axis=0)
    # pierced_mesh_faces2 = np.arange(len(pierced_mesh_verts2), dtype=int)[None,:] # Define a single face # should be same as face2

    # Now create the two new meshes
    # pierced_mesh_part1 = Trimesh(vertices=pierced_mesh_verts1, faces=pierced_mesh_faces1, face_colors=[255, 0, 0, 255])
    # pierced_mesh_part2 = Trimesh(vertices=pierced_mesh_verts2, faces=pierced_mesh_faces2, face_colors=[0, 255, 0, 255],vertex_colors=[0, 0, 255, 255])
    # Visualize the two new meshes
    # scene = trimesh.Scene([pierced_mesh_part1, pierced_mesh_part2, piercing_mesh_part1, piercing_mesh_part2])
    # scene.show()

    # TODO: figure out how to identify positive and negative sweeps given the normal maybe 
    verts1 = np.concatenate((pierced_mesh_verts1, piercing_mesh_verts1), axis=0)
    verts2 = np.concatenate((pierced_mesh_verts2, piercing_mesh_verts2), axis=0)

    assert piercing_mesh_side1 ==  (not piercing_mesh_side2), "piercing_mesh_side1 must be opposite to piercing_mesh_side2"

    if piercing_mesh_side1:
        pos_verts = verts1
        pos_boundary = boundary1
        neg_verts = verts2
        neg_boundary = boundary2
    else:
        pos_verts = verts2
        pos_boundary = boundary2
        neg_verts = verts1
        neg_boundary = boundary1
    
    return pos_verts, pos_boundary, neg_verts, neg_boundary

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
    intersections1, intersected_lines1 = line_mesh_intersection(lines1, poly_mesh2)
    # For convex shapes, there should be at most 2 intersections, but there could be 0 or 1
    # If there are 2 intersections, then mesh1 pierces mesh2
    if len(intersections1) == 2:
        mesh1_pierces_mesh2 = True
        intersections = intersections1
        intersected_lines = intersected_lines1
    else:
        # If there are no intersections, then check for the intersection between the edges of boundary2 with mesh1
        # Define lines for each edge of boundary
        lines2 = np.concatenate((verts2[boundary[:,None, 0]], verts2[boundary[:, None, 1]]), axis=1)
        intersections2, intersected_lines2 = line_mesh_intersection(lines2, poly_mesh1)
        if len(intersections2) == 2:
            mesh2_pierces_mesh1 = True
            intersections = intersections2
            intersected_lines = intersected_lines2
        # If there is only 1 intersection, then the two meshes are piercing each other
        elif len(intersections1) == 1 and len(intersections2) == 1:
            mesh1_pierces_mesh2 = True
            mesh2_pierces_mesh1 = True
        elif len(intersections1) == 0 and len(intersections2) == 0:
            pass
        else:
            raise ValueError("The intersections are not valid. intersections1: {}, intersections2: {}".format(len(intersections1), len(intersections2)))

    valid_intersect = (mesh1_pierces_mesh2 or mesh2_pierces_mesh1)
    if valid_intersect and not separate_surfs:
        return (), valid_intersect
    elif mesh1_pierces_mesh2 and mesh2_pierces_mesh1:
        print ("Mesh1 and Mesh2 pierce each other")
        # Form intersections and and intersected_edges
        # Choose to treat as mesh1 piercing mesh2
        # Arbitrary choice to use edges of mesh1 intersecting wiht mesh2. Will change the surface 

        # Calculate the line of intersection between the two meshes
        # If the two meshes are piercing each other, then only one edge is intersected on each
        # We therefore need to find where the line intersects another edge of each mesh in order to split the meshes
        # between positive and negative surfaces
        m1_intersections, m1_valid = trimesh.intersections.plane_lines(poly_mesh2.vertices[0], poly_mesh2.face_normals[0], verts1[boundary.T], line_segments=True)
        intersections = m1_intersections
        intersected_edges = boundary[m1_valid]
        T21 = hom_inv(T12)
        pos_verts, pos_boundary, neg_verts, neg_boundary = split_intersected_meshes(face, intersections, intersected_edges,
                                                                                    piercing_verts=verts1, pierced_verts=verts2,
                                                                                    pierced_mesh=poly_mesh2, T_piercing_pierced = T12)
        # Plot positive and negative surfaces
        pstride = len(pos_verts)//2
        pfaces = simple_polygon_triangulation(pstride)
        p1 = Trimesh(vertices=pos_verts[0:pstride], faces=pfaces, process=process, face_colors=[0, 255, 0, 255])
        # Add vertices to the plot
        # add first vertex to the plot as blue
        p1_first_vert = trimesh.points.PointCloud(pos_verts[0:1], colors=[0, 0, 255, 255])
        p1_second_vert = trimesh.points.PointCloud(pos_verts[1:2], colors=[255, 0, 0, 255])
        p1_third_vert = trimesh.points.PointCloud(pos_verts[2:3], colors=[0, 255, 0, 255])
        p1_verts = trimesh.points.PointCloud(pos_verts[3:pstride])
        # scene = trimesh.Scene([p1, p1_first_vert, p1_second_vert, p1_third_vert, p1_verts])
        # scene.show()
        # p2 = Trimesh(vertices=pos_verts[pstride:], faces=pfaces, process=process, face_colors=[255, 0, 0, 255])
        # scene.add_geometry(p2)
        # scene.show()
        # nstride = len(neg_verts)//2
        # nfaces = simple_polygon_triangulation(nstride)
        # n1 = Trimesh(vertices=neg_verts[0:nstride], faces=nfaces, process=process, face_colors=[0, 255, 0, 100])
        # scene.add_geometry(n1)
        # scene.show()
        # n2 = Trimesh(vertices=neg_verts[nstride:], faces=nfaces, process=process, face_colors=[255, 0, 0, 100])
        # scene.add_geometry(n2)
        # scene.show()
        # scene = trimesh.Scene([p1, p2, n1, n2])
        # scene.show()
        
        test=1

    elif mesh1_pierces_mesh2:
        print("Mesh1 pierces Mesh2")
        intersected_edges = boundary[intersected_lines]
        pos_verts, pos_boundary, neg_verts, neg_boundary = split_intersected_meshes(face, intersections, intersected_edges,
                                                                                    piercing_verts=verts1, pierced_verts=verts2,
                                                                                    pierced_mesh=poly_mesh2, T_piercing_pierced = T12)
        test = 1
    elif mesh2_pierces_mesh1:
        # Split mesh2 into two parts
        print("Mesh2 pierces Mesh1")
        # First let us add the intersection points as vertices to the mesh
        # Add the intersection points to the mesh
        # The suffix indicates wheter or not the mesh is on the pierced side or the piercing side
        intersected_edges = boundary[intersected_lines]
        pos_verts, pos_boundary, neg_verts, neg_boundary = split_intersected_meshes(face, intersections, intersected_edges,
                                                                                    piercing_verts=verts2, pierced_verts=verts1,
                                                                                    pierced_mesh=poly_mesh1, T_piercing_pierced = hom_inv(T12))
    
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
    roll_dirs: ArrayLike = None,
    convex_interp: bool = True,
    alpha: int = 255,
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
    if roll_dirs is None:
        roll_dirs = np.ones(n_sweeps, dtype=bool)
        warnings.warn("roll_dirs not provided. Assuming all rolls are positive.")
    assert len(roll_dirs) == n_sweeps, "roll_dirs must be the same length as transforms"
    
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

    # Plot the first and second surfaces for debugging
    # mesh_1_test = Trimesh(vertices=verts, faces=org_faces, process=False, face_colors=[0, 255, 0, 255])
    # mesh_2_test = Trimesh(vertices=vertices_3D[stride:2*stride], faces=org_faces, process=False, face_colors=[255, 0, 0, 255])
    # scene = trimesh.Scene([mesh_1_test, mesh_2_test])
    # # add vertex for the 0th vertex of the first surface
    # scene.add_geometry(trimesh.points.PointCloud(verts[0][None,:], colors=[0, 255, 0, 255]))
    # # add vertex for the 1st vertex of the first surface
    # scene.add_geometry(trimesh.points.PointCloud(verts[1][None,:], colors=[0, 0, 255, 255]))
    # scene.show()

    sides, intersected = side_of_surface(vertices_3D, stride)

    # Check for self-intersections between the slices
    # TODO: Loop through them all
    # TODO: Figure out this whole transform issue where initial mesh is translated and rotated...
    # Using unique to define the single face of the polygon (not a trimesh face, but a face of the polygon)
    # This will help us to define two new 3D faces/polygons in the case of a self-intersection
    # Intializations
    pos_verts = np.empty((0,3))
    pos_faces = np.empty((0,3))
    neg_verts = np.empty((0,3))
    neg_faces = np.empty((0,3))
    last_pos_sweep, last_neg_sweep, lost_pos_sweep_cap_offset, lost_neg_sweep_cap_offset = 0, 0, 0, 0
    # TODO: Add functionality to do all sweeps in parallel (as is in the original function)
    # This could speed up things in the case that there are no intersections and the movement is assumed to be 
    # all positive or negative
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
                last_pos_sweep_cap_offset = len(pos_verts) - stride
            else:
                # Add the past verticies if either this is the first sweep, the previous sweep was positive, or the previous sweep intersected
                if len(neg_verts) == 0 or sides[i-1] or intersected[i-1]:
                    offset = len(neg_verts)
                    neg_verts = np.concatenate((neg_verts, vertices_3D[stride*(i):stride*(i+1)]),axis=0)
                    # Add cap faces at the beginning of the sweep to close the volume on one end
                    cap_face = get_cap_face(boundary, flip_normals = False)# true
                    neg_faces = np.concatenate((neg_faces, cap_face+offset), axis=0)
                offset = len(neg_verts)-stride
                neg_verts = np.concatenate((neg_verts, vertices_3D[stride*(i+1):stride*(i+2)]),axis=0)
                neg_faces = np.concatenate((neg_faces, face_sweep+offset), axis=0)
                last_neg_sweep = i+1
                last_neg_sweep_cap_offset = len(neg_verts) - stride

    # Cap Faces of both volumes
    # Only cap the end of the positive sweep if the last sweep was positive
    # otherwise, the cap face will be added when the negative sweep is added
    if len(pos_verts) != 0 and last_pos_sweep != 0 and last_pos_sweep == n_sweeps:
        # Handle differently if dealing with intersections
        cap_face = get_cap_face(boundary, flip_normals = False)
        pos_faces = np.concatenate((pos_faces, cap_face+last_pos_sweep_cap_offset), axis=0)
    elif len(neg_verts) != 0 and last_neg_sweep != 0 and last_neg_sweep == n_sweeps:
        cap_face = get_cap_face(boundary, flip_normals = True)
        neg_faces = np.concatenate((neg_faces, cap_face+last_neg_sweep_cap_offset), axis=0)


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
    alpha = alpha
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

def projected_mesh_boundary(mesh: Trimesh, axis: int = 2) -> Dict:
    """
    Get the boundary of a 2D mesh projected onto a plane

    Parameters
    ----------
    mesh : trimesh.Trimesh
      2D mesh to get the boundary of
    axis : int
      The axis to project the mesh onto

    Returns
    -------
    boundary_poly : shapely.geometry.Polygon
      A polygon representing the boundary of the mesh
      projected onto the plane
    """
    # First project the vertices of the mesh onto the plane
    proj_axes = np.array([0, 1, 2], dtype=int)
    proj_axes = np.delete(proj_axes, axis)
    verts2D = mesh.vertices[:, proj_axes]
    # Define a shapely polygon for each face
    faces = mesh.faces

    multi_poly = MultiPolygon([Polygon(verts2D[face]) for face in faces])
    # Get the boundary of the projected mesh
    boundary_poly = unary_union(multi_poly)
    # Get the boundary of the polygon
    return boundary_poly


# TODO: This function is taken from SensorProcessor. It should be moved to a utility file
# TODO: May need to make this its own kernel...
def get_ext_euler_angles(C, xp=np):
    """
    Extract Extrinsically defined Euler angles from rotation matrix
    The transformation matrix C in terms of
    extrinsically defined Euler angles (roll, pitch, yaw)
    is defined as C = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    i.e. first roll about x, then pitch about y, and then yaw about z (in that order)
    i.e. the rotation matrix is defined as a sequence of
    rotations about the axes of the original coordinate
    system where the axes are fixed in space.
    For reference Basic rotation matrices are defined as:
    Rx = lambda a: xp.array([[1, 0, 0],
                            [0, xp.cos(a), xp.sin(a)],
                            [0, -xp.sin(a), xp.cos(a)]])

    Ry = lambda b: xp.array([[xp.cos(b), 0, -xp.sin(b)],
                            [0, 1, 0],
                            [xp.sin(b), 0, xp.cos(b)]])

    Rz = lambda c: xp.array([[xp.cos(c), xp.sin(c), 0],
                            [-xp.sin(c), xp.cos(c), 0],
                            [0, 0, 1]])
    Note that this may be refered to as Extrinsic Tait-Bryan angles in x-y-z order.
    Using Reference: Computing Euler Angles from a Rotation Matrix by Gregory G. Slabaugh
    https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

    2 solutions for these Extrinsic Tait-Bryan angles always exist. 
    The Eigen geometry module function eulerAngles(ax1,ax2,ax3) returns the intrinsically
    defined Euler angles (a,b,c) applied in the order ax1, ax2, ax3. and ensures that the angles
    (a,b,c) are in the ranges [0:pi]x[-pi:pi]x[-pi:pi].
    (https://eigen.tuxfamily.org/dox/group__Geometry__Module.html#title20)
    In order to obtain Extrinsic Tait-Bryan angles applied in the order x-y-z, robot_localization
    uses the Eigen function y,p,r = eulerAngles(2,1,0)
    (https://github.com/cra-ros-pkg/robot_localization/blob/49aab740c0b66c9f266141522e86d64dc86c8939/src/ros_robot_localization_listener.cpp#L456)
    which means that the orientation state being tracked by robot_localization
    (r,p,y) are in [-pi:pi]x[-pi:pi]x[0:pi].
    Given this assumption we can select the correct set of angles
    """
    if xp.abs(C[2, 0]) != 1.0:
        pitch1 = -xp.arcsin(C[2,0]) # Different compared to paper due to 
        roll1 = xp.arctan2(C[2,1]/xp.cos(pitch1), C[2,2]/xp.cos(pitch1))
        yaw1 = xp.arctan2(C[1,0]/xp.cos(pitch1), C[0,0]/xp.cos(pitch1))
        # Select the correct set of angles
        a1_valid = roll1 >= -xp.pi and roll1 <= xp.pi
        a1_valid = a1_valid and (pitch1 >= -xp.pi and pitch1 <= xp.pi)
        a1_valid = a1_valid and (yaw1 >= 0 and yaw1 <= xp.pi)
        if a1_valid:
            roll, pitch, yaw = roll1, pitch1, yaw1
            return roll, pitch, yaw
        # else:
        pitch2 = xp.pi - pitch1
        roll2 = xp.arctan2(C[2,1]/xp.cos(pitch2), C[2,2]/xp.cos(pitch2))
        yaw2 = xp.arctan2(C[1,0]/xp.cos(pitch2), C[0,0]/xp.cos(pitch2))
        a2_valid = roll2 >= -xp.pi and roll2 <= xp.pi
        a2_valid = a2_valid and (pitch2 >= -xp.pi and pitch2 <= xp.pi)
        a2_valid = a2_valid and (yaw2 >= 0 and yaw2 <= xp.pi)
        if a2_valid:
            roll, pitch, yaw = roll2, pitch2, yaw2
            return roll, pitch, yaw
        else:
            raise ValueError("No valid set of Euler angles found")
            
    else: # Gimbal lock: pitch is at -90 or 90 degrees
        yaw = 0.0 # This can be any value, but we choose 0.0 for consistency
        if C[2, 0] == -1.0:
            pitch = xp.pi/2
            roll = yaw + xp.arctan2(C[0,1], C[0,2])
        else:
            pitch = -xp.pi/2
            roll = -yaw + xp.arctan2(-C[0,1], -C[0,2])
    return roll, pitch, yaw


class GETMovement:
    """
    This class defines the ground engaging tool (GET) and provides
    methods for generating the swept volume of the GET as it moves
    through soil or other materials. This class also provides methods
    for updating the elevation map as the GET moves through the material.

    NOTE: The marjority of the computation done here is done on the CPU as
    it relies on the trimesh library. Initial testing has shown that this
    is adequate for the current application. If performance becomes an issue,
    then the code can be modified to use the cupy library for GPU acceleration.
    This would require either a custom implementation of portions of the trimesh
    library or a fully custom implementation of the swept volume generation.
    """

    def __init__(self, GET_ID, GET_model_name, GET_params, xp=np, data_type=np.float32):
        """Initialize GET for a specific sensor.

        Args:
            GET_ID (str):           GET ID. Should be unique for each instance of a GETMovement
            GET_model_name (str):   Type of GET. Only "simple_blade" is supported for now.
            GET_params (dict):      Parameters for the chosen GET type specified in GET_name.
            xp (module):            Numpy or CuPy module. Default is numpy. Will not be used for trimesh operations.
            data_type (dtype):      Data type for the GET geometry. Default is np.float32.
        """
        self.xp = xp
        self.data_type = data_type
        self.GET_ID = GET_ID

        # TODO: Add support for more complex GETs
        self.GET_models = {"simple_blade": self.simple_blade_geometry,}
        assert GET_model_name in self.GET_models.keys(), "GET_name should be chosen from {}".format(self.GET_models.keys())
        self.GET_model_name = GET_model_name
        self.GET_model = self.GET_models[self.GET_model_name]
        self.GET_params = {}
        # Make sure params are compatible with class
        for key in GET_params.keys():
            self.GET_params[key] =np.array(GET_params[key], dtype=self.data_type)
        
        # Now generate the GET geometry
        # TODO: Add support for multiple planar GET surfaces at different angles
        self.GET_mesh = self.GET_model(**self.GET_params)

    def simple_blade_geometry(self, blade_width=3.0, blade_height=0.6):
        """
        Define GET Geometry and Sequence of thin 4 point convex polygons
        using the trimesh library
        Could do this with a CAD file, but this is a simple test
        Blade consists of a single plane with 4 points
        Define the 4 points of the blade
        The blade is a thin 4 point polygon
        The blade is defined in the ZY plane
        where the top of the blade is in the positive Z direction,
        the left side of the blade is in the positive Y direction,
        and front of the blade facing the positive X direction when the blade is at 0 degrees.
        The origin of the blade is at the center of the rectangle in the YZ plane at [0,0,0].
        This origin is important to ensure proper sweeping of the blade.
        """

        # Define the 4 vertices of the blade
        vertices = np.array([[0, blade_width/2.0, blade_height/2.0],
                            [0, -blade_width/2.0, blade_height/2.0],
                            [0, -blade_width/2.0, -blade_height/2.0],
                            [0, blade_width/2.0, -blade_height/2.0]])
        
        # curved_blade_vertices = np.array([[0.4, blade_width/2.0+0.4, blade_height/2.0],
        #                                 [0.4, blade_width/2.0+0.4, -blade_height/2.0],
        #                                 [0.4, -blade_width/2.0-0.4, -blade_height/2.0],
        #                                 [0.4, -blade_width/2.0-0.4, blade_height/2.0]])
        
        # vertices = np.concatenate((vertices, curved_blade_vertices), axis=0)
        

        # Define the single face of the blade
        # The blade will be visible from the front since the normal is pointing in the positive X direction
        # of the untransformed blade
        faces_front = np.array([[0, 1, 2, 3]])

        # # faces_curved = np.array([[0, 3, 5, 4], [1, 7, 6, 2]])
        # # faces_front = np.concatenate((faces_front, faces_curved), axis=0) 

        # Create the trimesh object
        blade = trimesh.Trimesh(vertices=vertices, faces=faces_front)

        # # Rotate the blade about the Y axis at the origin by blade_angle degrees ccw
        # # The blade is rotated about the Y axis at the origin by blade_angle degrees ccw about the Y axis.
        # # blade_angle_deg=-10, blade_origin=[1.634, 0.0, 0.060+0.265]
        # rot = trimesh.transformations.rotation_matrix(np.radians(blade_angle_deg), [0, 1, 0])
        # # Translate the blade so that the origin matches the blade_origin
        # # The transform from chassis frame to blade frame
        # T_CB = trimesh.transformations.translation_matrix(blade_origin)@rot
        
        # if apply_transform:
        #     blade.apply_transform(T_CB)

        return blade
    
    def map_index_to_point_xy(self, indices, center, cell_n, resolution):
        """
        Convert map indices to points in the map frame.

        Args:
            indices (np.ndarray) (n,2):         The indices of the points in the map
            center (np.ndarray) (3,):           The center of the map in the map frame
            cell_n (int):                       The number of cells in the map
            resolution (float):                 The resolution of the map
        Returns:
            points_xy (np.ndarray) (n,2):          The 2D points in the map frame
        """
        # Convert indices to points
        points_xy = (indices - cell_n / 2) * resolution + center[:2].reshape(1, 2)
        return points_xy
    
    def get_map_index(self, points, center, cell_n, resolution):
        """
        Convert points to map indices.
        See custom_kernels.py map_utils kernel: get_x_idx() for more information.
        Args:
            points (np.ndarray) (n,3):          The points in the map frame
            center (np.ndarray) (3,):           The center of the map in the map frame
            cell_n (int):                       The number of cells in the map
            resolution (float):                 The resolution of the map
        Returns:
            indices (np.ndarray) (n,2):         The indices of the points in the map
            points_centered (np.ndarray) (n,3): The points represented in the map frame
        """
        points_centered = points - center.reshape(1, 3)
        # Get the indices of the points in the map
        indices = (points_centered[:,0:2] / resolution + cell_n / 2).astype(self.xp.int32)
        indices = self.xp.clip(indices, 0, cell_n - 1)
        return indices, points_centered
    
    def bounding_box_to_map_index(self, points, center, cell_n, resolution):
        """
        Convert map aligned bounding box points to map indices.
        This produces indicies for cells that encompass the bounding box.

        Args:
            points (np.ndarray) (n,3):  The bounding box points in the map frame
            center (np.ndarray) (3,):   The center of the map in the map frame
            cell_n (int):               The number of cells in the map
            resolution (float):         The resolution of the map
        Returns:
            indices (np.ndarray) (2,2): The indices of the bounding box corners corresponding to 
                                        the minimum uvz coordinates in the map and the maximum uvz coordinates
        """
        # Since bounding box is axis aligned we only need the bottom corners which are where the z values are minimum
        n = points.shape[0]
        # Make sure n is 8
        assert n == 8, "Bounding box should have 8 points"
        # Since this is an axis aligned bounding box we can get the min and max coordinates
        # by finding the min and max of each coordinate
        points_minmax = np.array([np.min(points, axis=0), np.max(points, axis=0)])
        # Alternatively: Sort points to find min point along x, y, and z and max point along x, y, and z
        # sort_inds = np.lexsort((points[:,1], points[:,0], points[:,2]))
        indices, points_centered = self.get_map_index(points_minmax, center, cell_n, resolution)
        return indices, points_centered
    
    def update_map_with_swept_volume(self, swept_mesh, T_MG0, elevation_map, map_center, cell_n, resolution):
        """
        Update the elevation map in place with the a swept volume derived from the GET
        Args:
            swept_mesh (trimesh.Trimesh):    The swept volume of the GET
            T_MG0 (np.ndarray):              The initial pose of the GET in the map frame
            elevation_map (xp.ndarray):      The full starting elevation map to update in place
            map_center (np.ndarray):         The center of the map in the map frame
            cell_n (int):                    The number of cells in the map
            resolution (float):              The resolution of the map
        """
        swept_mesh.apply_transform(T_MG0)
        # Obtain a map frame aligned bounding box for the swept volume
        bbox = swept_mesh.bounding_box
        # Find cells in the elevation map that are within the bounding box of the swept volume
        bb_indices, points_centered = self.bounding_box_to_map_index(bbox.vertices, map_center, cell_n, resolution)
        min_sv_z = points_centered[0, 2]
        max_sv_z = points_centered[1, 2]
        # Extract submap from the elevation map and convert to a numpy array to enable ray casting with trimesh
        inds_i, inds_j = np.meshgrid(np.arange(bb_indices[0,0], bb_indices[1,0]+1), np.arange(bb_indices[0,1], bb_indices[1,1]+1), indexing='ij')
        submap = elevation_map[:,inds_i, inds_j]
        # Deal with fact that these cells may not be valid. If we have no elevation data for those cells then we can't cut them
        update_elevation = True
        update_upper_bound = False # This should be a parameter probably
        valid_cells = submap[2] > 0.5
        if not self.xp.any(valid_cells):
            print("No valid cells in the swept volume")
            # TODO: We could however update the upper bound and is upper bound status of the cells...
            update_elevation = False
        else:
            # Check to see if we are likely to have an intersection by comparing the max_z of the submap and the min_z of the swept volume bounding box
            min_em_z = self.xp.min(submap[0, valid_cells])
            max_em_z = self.xp.max(submap[0, valid_cells])
            if min_sv_z > max_em_z:
                print("No intersection with swept volume")
                # TODO: We could also update the variance of the cells that are not intersected,
                #       e.g. if the variance is high then we can reduce it if our swept volume is close to the ground
                update_elevation = False
        # Debugging Override
        # update_elevation = True
        if update_elevation:
            # Convert the submap to a numpy array
            submap = self.xp.asnumpy(submap)
            valid_cells = self.xp.asnumpy(valid_cells)
            # Ray cast from each submap cell center to the swept volume
            # Use the
            # Small epsilon to avoid self intersection
            epsilon_z = 1e-1
            start_z = np.min([min_em_z.item(), min_sv_z]) - epsilon_z
            # Get the cell centers in the map frame
            # Combine inds_i and inds_j to get the indices of the cells in the map
            cell_inds = np.stack((inds_i[valid_cells], inds_j[valid_cells]), axis=1)
            cell_centers = self.map_index_to_point_xy(cell_inds, map_center, cell_n, resolution)
            n_cells = cell_centers.shape[0]
            lines = np.zeros((n_cells, 2, 3), dtype=self.data_type)
            lines[:,0,0:2] = cell_centers
            lines[:,1,0:2] = cell_centers
            lines[:,0,2] = start_z
            lines[:,1,2] = submap[0, valid_cells]
            intersections, intersected_lines = line_mesh_intersection(lines, swept_mesh, coincidence_tol=1e-6)
            if len(intersections) > 0:
                print("Intersections found")
                # Update the elevation map
                # Get the indices of the intersected cells
                intersected_cells = cell_inds[intersected_lines] - bb_indices[0]
                # Update the elevation map with the new heights
                submap[0, intersected_cells[:,0], intersected_cells[:,1]] = intersections[:,2]
                # TODO: Update the variance of the cells
                # TODO: Update the upper bound status of the cells
                # Copy the updated submap back to the elevation map
                elevation_map[:,inds_i, inds_j] = submap


        # TODO: pull relavent cells from elevation_map, check validity, update upper bound, changed occupied status,
        # remove intersecting heights, move material to new location, update occupied status of cells where material was moved
        # boundary_poly = projected_mesh_boundary(new_mesh, axis=2)
        # x,y = boundary_poly.exterior.xy
        return

    def update_map_with_GET_movement(self, elevation_map, map_center, cell_n, resolution, T_MG0, T_MG1, roll=None):
        """
        Update the elevation map with the movement of the GET from T_MG0 to T_MG1
        Args:
            elevation_map (xp.ndarray):     The full starting elevation map to update in place
            map_center (np.ndarray):        The center of the map in the map frame
            cell_n (int):                   The number of cells in the map
            resolution (float):             The resolution of the map
            T_MG0 (np.ndarray):             The initial pose of the GET in the map frame
            T_MG1 (np.ndarray):             The final pose of the GET in the map frame
        """
        # First define swept volume of the GET
        # The swept volume is the volume of the material that the GET has moved through
        # as it moves from T_MG0 to T_MG1
        # Sweep the volume in the frame defined by T_MG0, i.e. relative to the initial position of the GET
        transforms = (hom_inv(T_MG0)@T_MG1)[None, :, :]
        if roll is None:
            # Get relative roll between the two poses, this is used to generate a convex sweep
            roll, _, _ = get_ext_euler_angles(transforms[0,:3,:3], xp=np)
        roll_dirs = np.array([roll]) >= 0
        pos_swept_mesh, neg_swept_mesh = sweep_thin_poly_mesh(self.GET_mesh, transforms, roll_dirs=roll_dirs, convex_interp=True)
        if pos_swept_mesh is not None:
            self.update_map_with_swept_volume(pos_swept_mesh, T_MG0, elevation_map, map_center, cell_n, resolution)
        if neg_swept_mesh is not None:
            self.update_map_with_swept_volume(neg_swept_mesh, T_MG0, elevation_map, map_center, cell_n, resolution)
        return


if __name__ == "__main__":
    GET_ID = "simple_flat_blade"
    GET_name = "simple_blade"
    GET_params = {
        "blade_width": 3.0,
        "blade_height": 0.6,
    }
    GET_m = GETMovement(GET_ID, GET_name, GET_params)

    
    # Specify starting pose of blade
    # blade_angle_deg = -10
    # blade_pos = [1.634, 0.0, 0.060+0.265]
    blade_roll_deg = 0
    blade_pitch_deg = -10
    blade_yaw_deg = 0
    blade_pos = [3.0, 5.0, 0.060+0.265]
    # Rotate the blade about the Y axis at the origin by blade_angle degrees ccw
    rot = trimesh.transformations.rotation_matrix(np.radians(blade_roll_deg), [1, 0, 0])
    rot = trimesh.transformations.rotation_matrix(np.radians(blade_pitch_deg), [0, 1, 0])@rot
    rot = trimesh.transformations.rotation_matrix(np.radians(blade_yaw_deg), [0, 0, 1])@rot
    # Translate the blade so that the origin matches the blade_origin
    # The transform from chassis frame to blade frame
    T_WG = trimesh.transformations.translation_matrix(blade_pos)@rot
    T_GW = hom_inv(T_WG)

    # Define the path of the blade
    # T_dB = trimesh.transformations.translation_matrix([1.5, -1.5, 0.0])
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [1, 0, 0])@T_dB
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-10), [0, 1, 0])@T_dB
    # T_dB2 = trimesh.transformations.translation_matrix([1.5, 0.0, 0.0])@T_dB
    # T_dB3 = trimesh.transformations.translation_matrix([-0.5, 0.0, 0.0])@T_dB2
    # # T_db3 = trimesh.transformations.rotation_matrix(np.radians(10), [1, 0, 0])@T_dB3
    # # T_dB3 = trimesh.transformations.rotation_matrix(np.radians(15), [0, 0, 1])@T_dB3@T_dB2
    # roll_dirs = np.array([-20, 0, 10]) >= 0
    # transforms = np.array([T_dB, T_dB2, T_dB3])
    # # Test with edge case of same transform provided twice
    # # roll_dirs = np.array([-20, 0, 0, 0, 0, 0, 10]) >= 0
    # # transforms = np.array([T_dB, T_dB, T_dB, T_dB, T_dB2, T_dB2, T_dB3])
    # # roll_dirs = np.array([-20]) >= 0
    # # transforms = np.array([T_dB])


    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [0, 0, 1])
    # roll_dirs = np.array([-20]) >= 0
    # transforms = np.array([T_dB])
    # # roll_dirs = np.array([0,0,-20]) >= 0
    # # transforms = np.array([T_BW, T_WB@T_BW, T_dB])

    # Applying T_BW to transforms so that i can apply the transforms to the blade surface
    # directly. This is because the blade surface is defined in the blade frame
    # The blade_mesh is defined in the world frame so the transform T_BW must 
    # Testing piercing
    # T_dB = trimesh.transformations.translation_matrix([0.3, 0.4, 0.2])
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(22), [1, 0, 0])@T_dB
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-15), [0, 1, 0])@T_dB
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-15), [0, 0, 1])@T_dB
    # T_dB2 = trimesh.transformations.translation_matrix([1.5, 0.0, 0.0])@T_dB
    # roll_dirs = np.array([22,0]) >= 0
    # transforms = np.array([T_dB, T_dB2])
    # # roll_dirs = np.array([22]) >= 0
    # # transforms = np.array([T_dB])
    # # Flipping direction to test mesh 1 piercing mesh 2
    # # roll_dirs = np.array([-22]) >= 0
    # # transforms = np.array([hom_inv(T_dB)])

    # Testing intersection
    # T_dB = trimesh.transformations.translation_matrix([0.3, 0.4, 0.2])
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(22), [1, 0, 0])@T_dB
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [0, 1, 0])@T_dB
    # T_dB = trimesh.transformations.rotation_matrix(np.radians(-15), [0, 0, 1])@T_dB
    # # Flipping direction to match paper test case
    # roll_dirs = np.array([-22]) >= 0
    # transforms = np.array([hom_inv(T_dB)])

    # Testing intersection
    T_dB = trimesh.transformations.translation_matrix([0.1, 0,0])
    T_dB = trimesh.transformations.rotation_matrix(np.radians(22), [1, 0, 0])@T_dB
    T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [0, 1, 0])@T_dB
    T_dB = trimesh.transformations.rotation_matrix(np.radians(-20), [0, 0, 1])@T_dB # This makes it strange
    # Flipping direction to match paper test case
    # roll_dirs = np.array([0]) >= 0
    roll_dirs = np.array([-20]) >= 0
    transforms = np.array([T_dB])

    alpha=255
    use_wireframe = False
    plot_pos = True
    plot_neg = True

    # Get transforms in the world frame
    # The above are just a way of generating transforms easily because we can think about them wrt the starting GET frame
    transforms = T_WG@ transforms

    # Time this function
    start = time.time()
    # Must represent transforms as relative to the starting pose of the GET
    # Must apply T_BW to the transforms so that the sweep is applied properly
    # Could get rid of this requirement by transforming mesh to T_GW frame first
    pos_new_mesh, neg_new_mesh = sweep_thin_poly_mesh(GET_m.GET_mesh, T_GW@transforms, roll_dirs=roll_dirs, convex_interp=True, cap=True, connect=False, alpha=alpha)
    end = time.time()

    # Plot the boundary of the swept volumes
    fig, ax = plt.subplots()
    print("Time taken to sweep the blade: ", end-start)
    meshes = []
    if pos_new_mesh is not None and plot_pos:
        pos_new_mesh.apply_transform(T_WG)
        meshes.append(pos_new_mesh)
        pos_boundary_poly = projected_mesh_boundary(pos_new_mesh, axis=2)
        x,y = pos_boundary_poly.exterior.xy
    ax.plot(x, y, color='g')
    if neg_new_mesh is not None and plot_neg:
        neg_new_mesh.apply_transform(T_WG)
        meshes.append(neg_new_mesh)
        neg_boundary_poly = projected_mesh_boundary(neg_new_mesh, axis=2)
        x,y = neg_boundary_poly.exterior.xy
        ax.plot(x, y, color='r')
    plt.show()

    # trimesh.util.concatenate(new_mesh.split(only_watertight=True))
    # new_mesh.show(smooth=False)
    # scene = trimesh.Scene([new_mesh.convex_hull])
    scene = trimesh.Scene([meshes])
    # Add a world coordinate frame to the scene
    world_frame = trimesh.creation.axis(origin_size=0.1, axis_length=1.0)
    scene.add_geometry(world_frame)
    # Add blade coordinate frame to the scene
    # Make origin size larger and color purple
    blade_frame = trimesh.creation.axis(origin_size=0.1, transform=T_WG, axis_length=1.0, origin_color=[160, 32, 240])
    scene.add_geometry(blade_frame)
    # Add the Final Blade Coordinate Frame
    final_frame = trimesh.creation.axis(origin_size=0.1, transform=transforms[-1], axis_length=1.0)
    scene.add_geometry(final_frame)
    # Now get a bounding box for the swept volume that is aligned with the world frame
    # bbox_world = new_mesh.bounding_box
    # # Add the bounding box corners to the scene
    # pc = trimesh.PointCloud(bbox_world.vertices)
    # # Visualize the bounding box
    # scene.add_geometry(pc)
    scene.show(smooth=False, flags={'wireframe': use_wireframe})