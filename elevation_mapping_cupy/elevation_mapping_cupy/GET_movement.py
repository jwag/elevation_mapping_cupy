import numpy as np
import trimesh

from trimesh.base import Trimesh
from trimesh import grouping, util
from trimesh.typed import ArrayLike, Dict, Optional
from trimesh.constants import tol
import time
import warnings

from elevation_mapping_cupy.parameter import Parameter

import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.simplefilter('always', UserWarning)


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
    ray_dirs = line[:,1] - line[:,0]
    ray_origins = line[:,0]

    # Visualize the rays and the mesh
    # stack rays into line segments for visualization as Path3D
    # ray_visualize = trimesh.load_path(np.hstack((ray_origins,
    #                                          ray_origins + ray_dirs*1.0)).reshape(-1, 2, 3))
    # scene = trimesh.Scene([mesh,
    #                    ray_visualize])
    # scene.show()

    # Find the intersection between the edges of boundary1 with mesh2
    # run the mesh- ray query
    # We only want the first intersection
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        multiple_hits=False) # I think this will return the hit that is the shortest distance, but not sure
    if len(locations) == 0:
        intersections = []
        intersected_lines = []
        pierce_dist = []
        invalid_intersections = []
        invalid_lines = []
    else:
        # Now need to determine if the intersection is on the line segment
        # First compute the distance from the origin of the ray to the intersection point
        # Then compute the distance from the origin of the ray to the end of the ray
        # If the distance to the intersection point is less than the distance to the end of the ray
        # then the intersection point is on the line segment
        # Otherwise, the intersection point is not on the line segment
        # Compute the distance from the origin of the ray to the intersection point
        # TODO: Only compute norm with z value because we are ray casting in z
        dist_to_intersection = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
        # Compute the distance from the origin of the ray to the end of the ray
        dist_to_end = np.linalg.norm(ray_dirs[index_ray], axis=1)
        # Check if the intersection point is on the line segment
        # If the ray is on the line, but closer than coincedence_tol to the end, 
        # then we treat as a non-intersection. This is to handle numerical issues
        # where the edge is actually coincident with the mesh surface
        # Set coincidence_tol to 0 to disable to detect coincident edges
        pierce_dist = dist_to_end - dist_to_intersection
        valid_intersect = pierce_dist >= coincidence_tol
        invalid_intersect = np.logical_not(valid_intersect)

        if np.any(valid_intersect):
            # Get the indices of the intersected lines
            intersected_lines = index_ray[valid_intersect]
            # Get the intersection point on the line segment
            intersections = locations[valid_intersect]
            pierce_dist = pierce_dist[valid_intersect]
            invalid_intersections = locations[invalid_intersect]
            invalid_lines = index_ray[invalid_intersect]
        else:
            intersections = []
            intersected_lines = []
            pierce_dist = []
            invalid_intersections = locations
            invalid_lines = index_ray
      
    # intersections is location of valid intersections of the the lines intersected_lines with the mesh
    # invalid_intersections is the location of the ray intersections that were outside of the bounds of 
    # the line(useful for upper bound calcs), and invalid_lines is the index of the rays that had invalid intersections
    return intersections, intersected_lines, pierce_dist, invalid_intersections, invalid_lines

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

def split_intersected_meshes(face, intersections, intersected_edges, piercing_verts, pierced_verts, pierced_mesh, T_piercing_pierced, relative_to="piercing"):
    # Pierced mesh is the mesh that is being pierced by the piercing mesh, i.e. it creates a line of intersection in the middle of the face
    # of the pierced mesh. Both meshes must be split into two parts to create a positive and negative swept volume.
    # The relative_to arg is use to determine which of the split meshes is the positive and negative swept volume

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
        # Adding the sort back in to ensure that indexing of the face_circ is correct
        # Otherwise when indexing backwards e.g. face_circ[3:2] will result in an empty array
        intersected_face_inds.sort()
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

    if relative_to == "piercing":
        pos_is_verts1 = not piercing_mesh_side1
    else:
        pos_is_verts1 = piercing_mesh_side1

    if pos_is_verts1:
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
    intersections1, intersected_lines1, _, _, _ = line_mesh_intersection(lines1, poly_mesh2)
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
        intersections2, intersected_lines2, _, _, _ = line_mesh_intersection(lines2, poly_mesh1)
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
            # TODO: Figure out why this happens sometimes. Maybe a rounding issue
            warnings.warn("Only one intersection found between the two meshes. Not sure what could cause this intersections1: {}, intersections2: {}".format(len(intersections1), len(intersections2)))
            # raise ValueError("The intersections are not valid. intersections1: {}, intersections2: {}".format(len(intersections1), len(intersections2)))

    valid_intersect = (mesh1_pierces_mesh2 or mesh2_pierces_mesh1)
    if not valid_intersect or not separate_surfs:
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
                                                                                    pierced_mesh=poly_mesh2, T_piercing_pierced = T12,
                                                                                    relative_to="piercing")
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
                                                                                    pierced_mesh=poly_mesh2, T_piercing_pierced = T12,
                                                                                    relative_to="piercing")
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
                                                                                    pierced_mesh=poly_mesh1, T_piercing_pierced = hom_inv(T12),
                                                                                    relative_to="pierced")
    
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
    pos_translation, neg_translation = None, None
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
                # This might mean that i need to rethink how the indexing works here
                # I think it is possible for it to fail when the intersected meshes have different length boundries...
                if n_new_pos_points != stride:
                    warnings.warn("If this condition fails that means that I need to debug this start end centroid calculation.")
                # The new points are the first n_new_pos_points of the p_verts I think
                pos_start_centroid = p_verts[n_new_pos_points:].mean(axis=0)
                pos_end_centroid = p_verts[0:n_new_pos_points].mean(axis=0)
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
                pos_translation = pos_end_centroid - pos_start_centroid
                

                # Add the verticies for the negative sweep
                neg_face_sweep, n_new_neg_points = get_swept_edges(n_boundary, roll_dirs[i], convex_interp, flip_normals = True)
                offset = len(neg_verts)
                # Add the verticies for the negative sweep
                neg_verts = np.concatenate((neg_verts, n_verts[0:n_new_neg_points]), axis=0)
                if n_new_neg_points != stride:
                    warnings.warn("If this condition fails that means that I need to debug this start end centroid calculation.")
                # The new points are the first n_new_neg_points of the n_verts I think
                neg_start_centroid = n_verts[n_new_neg_points:].mean(axis=0)
                neg_end_centroid = n_verts[0:n_new_neg_points].mean(axis=0)
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
                neg_translation = neg_end_centroid - neg_start_centroid

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

    return pos_swept_mesh, pos_translation, neg_swept_mesh, neg_translation

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
    tol = 1e-10
    if xp.abs(C[2, 0]) != 1.0:
        pitch1 = -xp.arcsin(C[2,0]) # Different compared to paper due to 
        roll1 = xp.arctan2(C[2,1]/xp.cos(pitch1), C[2,2]/xp.cos(pitch1))
        yaw1 = xp.arctan2(C[1,0]/xp.cos(pitch1), C[0,0]/xp.cos(pitch1))
        # Select the correct set of angles
        a1_valid = roll1 >= -xp.pi - tol and roll1 <= xp.pi + tol
        a1_valid = a1_valid and (pitch1 >= -xp.pi - tol and pitch1 <= xp.pi + tol)
        a1_valid = a1_valid and (yaw1 >= 0 - tol and yaw1 <= xp.pi + tol)
        if a1_valid:
            roll, pitch, yaw = roll1, pitch1, yaw1
            return roll, pitch, yaw
        # else:
        pitch2 = xp.pi - pitch1
        roll2 = xp.arctan2(C[2,1]/xp.cos(pitch2), C[2,2]/xp.cos(pitch2))
        yaw2 = xp.arctan2(C[1,0]/xp.cos(pitch2), C[0,0]/xp.cos(pitch2))
        a2_valid = roll2 >= -xp.pi - tol and roll2 <= xp.pi +tol
        a2_valid = a2_valid and (pitch2 >= -xp.pi - tol and pitch2 <= xp.pi + tol)
        a2_valid = a2_valid and (yaw2 >= 0 - tol and yaw2 <= xp.pi + tol)
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

    def __init__(self, GET_ID, GET_model_name, param: Parameter, GET_params, xp=np, data_type=np.float32):
        """Initialize GET for a specific sensor.

        Args:
            GET_ID (str):           GET ID. Should be unique for each instance of a GETMovement
            GET_model_name (str):   Type of GET. Only "simple_blade" is supported for now.
            param (dict):          Parameters for the elevation map.
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
        self.param = param # Parameters of elevtion map
        self.GET_params = {}
        # Make sure params are compatible with class
        for key in GET_params.keys():
            self.GET_params[key] =np.array(GET_params[key], dtype=self.data_type)
        
        if self.param.l_surcharge_max > self.param.l_fit_max:
            warnings.warn("l_surcharge_max must be less than or equal to l_fit_max right now. Setting equal. Possibly fix this")
            self.param.l_surcharge_max = self.param.l_fit_max
        
        # Now generate the GET geometry
        # TODO: Add support for multiple planar GET surfaces at different angles
        # The GET_origin is the origin of the GET geometry in the blade frame (should be zeros for now)
        self.GET_mesh, self.GET_geometry_origin, self.cutting_edge_origin, self.cutting_edge_vector = self.GET_model(**self.GET_params)

        # Initialize Parameters Used for FEE Projection
        self.pos_swept_mesh_FEE_projection_params = None
        self.neg_swept_mesh_FEE_projection_params = None
        self.ground_proj_params = None

    def simple_blade_geometry(self, blade_width=3.0, blade_height=0.6, **kwargs):
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
                            [0, blade_width/2.0, -blade_height/2.0]], dtype=self.data_type)
        
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

        # Origin is assumed to be at center of the GET currently
        blade_origin = np.array([0.0, 0.0, 0.0], dtype=self.data_type)

        # Define the primary cutting edge of the blade so that it can be tracked for erosion purposes
        cutting_edge_origin = vertices[2, :].copy()
        cutting_edge_vector = vertices[3, :] - vertices[2, :]

        # # Rotate the blade about the Y axis at the origin by blade_angle degrees ccw
        # # The blade is rotated about the Y axis at the origin by blade_angle degrees ccw about the Y axis.
        # # blade_angle_deg=-10, blade_origin=[1.634, 0.0, 0.060+0.265]
        # rot = trimesh.transformations.rotation_matrix(np.radians(blade_angle_deg), [0, 1, 0])
        # # Translate the blade so that the origin matches the blade_origin
        # # The transform from chassis frame to blade frame
        # T_CB = trimesh.transformations.translation_matrix(blade_origin)@rot
        
        # if apply_transform:
        #     blade.apply_transform(T_CB)

        return blade, blade_origin, cutting_edge_origin, cutting_edge_vector
    
    def map_index_to_point_xy(self, indices, center, cell_n, resolution):
        """
        Convert map indices to points in the map frame (not map origin frame).

        Args:
            indices (np.ndarray) (n,2):         The indices of the points in the map
            center (np.ndarray) (3,):           The center of the map in the map frame
            cell_n (int):                       The number of cells in the map
            resolution (float):                 The resolution of the map
        Returns:
            points_xy (np.ndarray) (n,2):          The 2D points in the map frame
        """
        # Convert indices to points
        # Need to handle the case where cell_n is even or odd
        if cell_n % 2 == 0:
            points_xy = (indices - cell_n / 2.0 + 0.5) * resolution + center[:2].reshape(1, 2)
        else:
            points_xy = (indices - cell_n / 2.0) * resolution + center[:2].reshape(1, 2)
        
        points_xy = points_xy.astype(self.data_type)
        return points_xy
    
    def get_map_index(self, points, center, cell_n, resolution, round_dir=None):
        """
        Convert points in map frame to map indices.
        Since elevation_map is itself represented in the map origin frame, O, 
        which is translated from the map frame by center, we need to account for this
        when converting points to indices. This is done by centering the points
        at the map center before using the resolution to convert to indices.
        Additionally, (0,0) in the map origin frame corresponds to the middle of
        elevation_map with cell_n/2 cells in each direction.
        See custom_kernels.py map_utils kernel: get_x_idx() for more information.
        The tricky part about all of this is that the origin frame is at a cell center
        if cell_n is odd, but at a cell corner if cell_n is even. This means that the indices
        need to be rounded differently depending on the resolution and the cell_n.
        Args:
            points (np.ndarray) (n,2 or 3):          The points in the map frame
            center (np.ndarray) (2 or 3,):           The center of the map in the map frame
            cell_n (int):                       The number of cells in the map
            resolution (float):                 The resolution of the map
            round_dir (str):                    The direction to round the indices. Default is None
                                                which means the cell_n will be used to determine the rounding
                                                direction.
        Returns:
            indices (np.ndarray) (n,2):         The indices of the points in the map (rounded down)
            points_centered (np.ndarray) (n,3): The points represented in the map origin frame
        """
        ps = points.shape
        if len(ps) == 1:
            pd = ps[0]
        elif len(ps) == 2:
            pd = ps[1]
        else:
            raise ValueError("points should be 1D or 2D array")
        if center is None:
            center = np.zeros((1,pd))
            points_centered = points
        else: # Make sure center is the right shape
            points_centered = points - center.reshape(1, pd)
        # Get the indices of the points in the map
        inds = (points_centered[:,0:2] / resolution + cell_n / 2.0)
        if round_dir is None:
            # If cell_n is even then round down, if odd then round to nearest
            if cell_n % 2 == 0:
                round_dir = "floor"
            else:
                round_dir = "round"
        if round_dir == "floor":
            indices = inds.astype(np.int32)
        elif round_dir == "ceil":
            indices = np.ceil(inds).astype(np.int32)
        elif round_dir == "round":
            indices = np.round(inds).astype(np.int32)
        elif round_dir == "float":
            indices = inds
        else:
            raise ValueError("round_dir should be 'floor', 'ceil', or 'round'")
        indices = np.clip(indices, 0, cell_n - 1)
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
        points_min = np.min(points, axis=0)
        points_max = np.max(points, axis=0)
        # Alternatively: Sort points to find min point along x, y, and z and max point along x, y, and z
        # sort_inds = np.lexsort((points[:,1], points[:,0], points[:,2]))
        # Round down for min and up for max to ensure a conservative bounding box with a minimum 1 cell border
        indices_min, point_min_centered = self.get_map_index(points_min, center, cell_n, resolution, round_dir="floor")
        indices_max, point_max_centered = self.get_map_index(points_max, center, cell_n, resolution, round_dir="ceil")
        # Increase size of bounding box by 1 cell in each direction to ensure that the bounding box captures deposit locations
        indices_min = np.clip(indices_min - 1, 0, cell_n - 1)
        indices_max = np.clip(indices_max + 1, 0, cell_n - 1)
        indices = np.concatenate((indices_min, indices_max), axis=0)
        points_centered = np.concatenate((point_min_centered, point_max_centered), axis=0)
        return indices, points_centered
    
    def find_deposit_locations(self, intersected_cells, invalid_deposit_cells, move_dir, map_size):
        dx = move_dir[0]
        dy = move_dir[1]

        deposit_location = np.zeros_like(intersected_cells)
        occ_map = np.zeros(map_size, dtype=bool)
        occ_map[intersected_cells[:,0], intersected_cells[:,1]] = True
        occ_map[invalid_deposit_cells[:,0], invalid_deposit_cells[:,1]] = True


        # This is a digitial differential analyzer (DDA) line algorithm
        # see https://en.wikipedia.org/wiki/Digital_differential_analyzer_(graphics_algorithm)
        if (abs(dx) >= abs(dy)):
            step = abs(dx)
        else:
            step = abs(dy)

        dx = dx / step
        dy = dy / step
        # Need to round differently depending on whether cell_n is even or odd
        # If cell_n is even then round down, if odd then round to nearest integer
        if map_size[0] % 2 == 0:
            round_fn = int
        else:
            round_fn = round
        # Debugging: override to what it was
        # round_fn = round
        for i, intersected_cell_ind in enumerate(intersected_cells):
            x = intersected_cell_ind[0]
            y = intersected_cell_ind[1]
            while True:
                xind = round_fn(x)
                yind = round_fn(y)
                if (occ_map[xind, yind] == False):
                    deposit_location[i] = np.array([xind, yind])
                    break
                x = x + dx
                y = y + dy
                if (x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1]):
                    warnings.warn("Deposit location not found! Debug this. Setting to intersected cell index. This should not happen if the map is large enough to capture the entire cutting edge.")
                    deposit_location[i] = intersected_cell_ind
                    break
        return deposit_location
    
    def find_cutting_edge(self, T_MG, center, map_size, resolution):
        # Use the cutting edge origin and vector to find the cutting edge in the map origin frame
        start_pos = T_MG[:3,:3] @ self.cutting_edge_origin + T_MG[:3,3]
        end_pos = T_MG[:3,:3] @ (self.cutting_edge_origin + self.cutting_edge_vector) + T_MG[:3,3]
        # Get the cutting edge vector in the map origin frame
        # The cutting edge vector is in the map frame so we need to transform it to the map origin frame
        # Since it is a direction vector we only need to transform the direction
        move_dir = T_MG[:3,:3] @ self.cutting_edge_vector
        # Normalize the direction vector, but based on the xy components only
        move_dir = move_dir / np.linalg.norm(move_dir[0:2])

        dx = move_dir[0]
        dy = move_dir[1]

        # Get the cutting edge origin in the map origin frame
        edge_origin_inds, start_pos = self.get_map_index(start_pos, center, map_size[0], self.param.resolution, round_dir="float")
        # Get the cutting edge end in the map origin frame
        edge_end_inds, end_pos = self.get_map_index(end_pos, center, map_size[0], self.param.resolution, round_dir="float")

        # Obtain the slope of the cutting edge in the map frame with respect to the map xy plane
        edge_slope = move_dir[2] / np.linalg.norm(move_dir[0:2])
        edge_length_xy = np.linalg.norm(self.cutting_edge_vector[0:2])
        # This is a digitial differential analyzer (DDA) line algorithm
        # see https://en.wikipedia.org/wiki/Digital_differential_analyzer_(graphics_algorithm)
        if (abs(dx) >= abs(dy)):
            step = abs(dx)
        else:
            step = abs(dy)

        dx = dx / step
        dy = dy / step
        x = edge_origin_inds[0,0]
        y = edge_origin_inds[0,1]
        edge_inds = np.empty((0,2), dtype=int)
        edge_heights = np.empty((0,1), dtype=self.data_type)
        # Need to round differently depending on whether cell_n is even or odd
        # If cell_n is even then round down, if odd then round to nearest integer
        if map_size[0] % 2 == 0:
            round_fn = int
        else:
            round_fn = round
        # Debug override
        # round_fn = round
        while True:
            xind = round_fn(x)
            yind = round_fn(y)
            edge_inds = np.append(edge_inds, np.array([[xind, yind]]), axis=0)
            edge_dist_xy = resolution * np.sqrt((x - edge_origin_inds[0,0])**2 + (y - edge_origin_inds[0,1])**2)
            # Get the height of the cutting edge at this point using z = mx + b
            edge_height = edge_slope * edge_dist_xy + start_pos[0,2]
            edge_heights = np.append(edge_heights, np.array([[edge_height]]), axis=0)
            # Check if the point is the end point
            if (xind == round_fn(edge_end_inds[0,0]) and yind == round_fn(edge_end_inds[0,1])):
                break
            x = x + dx
            y = y + dy
            # Check if the length of the cutting edge is exceeded, i.e. we have reached
            # the end of the cutting edge. This is needed in addition to the edge_end_inds check
            # for some reason that i don't quite understand.
            if (edge_dist_xy > edge_length_xy):
                break
            if (x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1]):
                raise ValueError("Blade cutting edge outside of map bounds")
                break
        return edge_inds, edge_heights
    
    def fit_plane_near_GET(self, elevation_map, T_OG, normal, map_center, cell_n, resolution, fit_dir, height_layer_ind=0):
        """Fit a plane to a surface layer of the elevation map in the direction of blade movement.

        This can be used as a reference for a blade controller

        Args:
            elevation_map (np.ndarray):     The elevation map that provides the layer to fit the plane to
            T_OG (np.ndarray) (4,4):        The transform from the map origin to the GET frame (should be at end of sweep)
            normal (np.ndarray) (3,):       The normal vector of the plane in the map frame
            map_center (np.ndarray) (3,):   The center of the map in the map frame
            cell_n (int):                   The number of cells in the map
            resolution (float):             The resolution of the map
            fit_dir (int):                  The direction to fit the plane in. 1 for forward, 0 for centered, -1 for negative
            height_layer_ind (int):         The index of the height layer to fit the plane to. Default is 0.

        Returns:
            plane_fit_params (dict):    The parameters for the plane fit with the keys:
                                        "M_r_MP": The point on the plane in the map frame
                                        "P_normal": The normal vector of the plane in the map frame
                                        "G_r_GC0": The bottom right corner (C0) of the GET in the GET frame
                                        "G_r_C0C1": The vector from the bottom right corner (C0) of the GET to the bottom left corner (C1) of the GET in the GET frame
                                        "G_normal": The normal vector of the GET in the map frame
        """
        # Construct ROI as shapely polygon
        # Then use the polgon bounds to find the inidicies of the map to check for being within the region using the contains function
        # Then fit a plane to those points
        # Use the cutting edge origin and vector to find the cutting edge in the map origin frame
        br_pos = T_OG[:3,:3] @ self.cutting_edge_origin + T_OG[:3,3]
        bl_pos = T_OG[:3,:3] @ (self.cutting_edge_origin + self.cutting_edge_vector) + T_OG[:3,3]
        nxy = np.array([normal[0], normal[1]])/np.linalg.norm(normal[0:2])
        plane_fit_ROI_length = 0.5 # meters
        # Now construct the ROI polygon depending on the fit_dir
        if fit_dir == 1:
            # Fit plane to region in front of the blade
            p0 = br_pos[:2]
            p1 = bl_pos[:2]
            p2 = bl_pos[:2] + nxy * plane_fit_ROI_length
            p3 = br_pos[:2] + nxy * plane_fit_ROI_length
        elif fit_dir == 0:
            # Fit plane to region around the blade
            p0 = br_pos[:2] - nxy * (plane_fit_ROI_length / 2.0)
            p1 = bl_pos[:2] - nxy * (plane_fit_ROI_length / 2.0)
            p2 = bl_pos[:2] + nxy * (plane_fit_ROI_length / 2.0)
            p3 = br_pos[:2] + nxy * (plane_fit_ROI_length / 2.0)
        elif fit_dir == -1:
            # Fit plane to region behind the blade
            p0 = br_pos[:2] - nxy * plane_fit_ROI_length
            p1 = bl_pos[:2] - nxy * plane_fit_ROI_length
            p2 = bl_pos[:2]
            p3 = br_pos[:2]
        else:
            raise ValueError("fit_dir should be 1, 0, or -1")
        # Construct the ROI polygon
        ROI = Polygon([p0, p1, p2, p3])
        # Get the bounding box of the ROI
        # (minx, miny, maxx, maxy) 
        ROI_bounds = ROI.bounds
        # Get the indices of the bounding box corners corresponding to the minimum uvz coordinates in the map and the maximum uvz coordinates
        indices_min, _ = self.get_map_index(np.array([[ROI_bounds[0], ROI_bounds[1]]]), None, cell_n, resolution, round_dir="floor")
        indices_max, _ = self.get_map_index(np.array([[ROI_bounds[2], ROI_bounds[3]]]), None, cell_n, resolution, round_dir="ceil")

        # Get the indices of all cells within the ROI
        ROI_inds = np.meshgrid(np.arange(indices_min[0,0], indices_max[0,0] + 1), np.arange(indices_min[0,1], indices_max[0,1] + 1))
        ROI_inds = np.stack((ROI_inds[0].flatten(), ROI_inds[1].flatten()), axis=-1)
        # Get the points in the map frame
        ROI_points = self.map_index_to_point_xy(ROI_inds, map_center, cell_n, resolution)
        # Get the height of the ROI points
        ROI_points_z = elevation_map[height_layer_ind, ROI_inds[:,0], ROI_inds[:,1]].get() + map_center[2]
        ROI_points = np.concatenate((ROI_points, ROI_points_z.reshape(-1,1)), axis=1)
        # Fit a plane to the points using trimesh SVD method
        C, N = trimesh.points.plane_fit(ROI_points)

        # Plane fit here is in map coordinates not map origin coordinates. Decide if that is what we want
        return {"M_r_MP": C, "P_normal": N,
                "G_r_GC0": self.cutting_edge_origin, "G_r_C0C1": self.cutting_edge_vector,
                "G_normal": self.GET_mesh.face_normals[0].astype(self.data_type)}
        
        
    def get_xy_GET_distance(self, inds, point_z, t_dir, GET_plane_origin, normal, cell_n, resolution):
        """
        Get the distance from a query cell, inds, at the height point_z, along -t_dir to the GET plane.

        Args:
            inds (np.ndarray) (2,):             The query point xy indicies of the elevation map
            point_z (float):                    The z coordinate of the query point in the map origin frame O
            t_dir (np.ndarray) (2,):            The direction of translation in 2d
            GET_plane_origin (np.ndarray) (3,): The origin of the GET plane in the map frame
            normal (np.ndarray) (3,):           The normal of the GET plane
            map_center (np.ndarray) (3,):       The center of the map in the map frame
            cell_n (int):                       The number of cells in the map
            resolution (float):                 The resolution of the map
        Returns:
            d (float):                          The distance from the query cell to the GET plane
        """
        # Get xy coordinates (we want it in the map origin frame so make center = 0,0,0)
        center = np.array([0.0, 0.0, 0.0], dtype=self.data_type)
        point_xy = self.map_index_to_point_xy(inds, center, cell_n, resolution)
        # Combine the xy coordinates with the z height of the intersected cell
        point = np.append(point_xy, point_z)
        # Trace back in the opposite direction of the translation in horizontal plane
        # to find the intersection with the GET plane
        line_dir = np.append(-t_dir, 0.0)[np.newaxis]
        # Normalize line_dir bc plane_lines expects it
        line_dir = line_dir / np.linalg.norm(line_dir)
        _, v, d = trimesh.intersections.planes_lines(GET_plane_origin[np.newaxis], normal[np.newaxis], point[np.newaxis], line_dir, return_distance=True)
        assert v == 1.0, "The line should always intersect the plane"
        # For some reason this returns negative distances sometimes
        if d < 0:
            # TODO: Figure out why this occurs
            d = abs(d)
        return d
    
    def get_surface_points(self, intersected_cells, t_dir, elevation_map, GET_plane_origin, normal, pierce_dist, cell_n, resolution):
        """
        For each intersected cell, find the set of surface points in the direction of movement, t_dir,
        and return the points in the form of (x_t, z) where x_t is the distance along the movement direction.
        Also find the volume of the surcharge, V_Q, in the region in front of the intersected cells.

        Args:
            intersected_cells (np.ndarray) (n,2):           The indices of the intersected cells
            t_dir (np.ndarray) (3,):                        The direction of blade translation (not normalized)
            elevation_map (np.ndarray) (8,cell_n,cell_n):   The elevation map
            GET_plane_origin (np.ndarray) (3,):             The origin of the GET plane in the map frame
            normal (np.ndarray) (3,):                       The normal of the GET plane
            pierce_dist (np.ndarray) (n,):                  The pierce distance of the blade for each intersected cell
            cell_n (int):                                   The number of cells in the map
            resolution (float):                             The resolution of the map
        Returns:
            surf_points (list):                             A list of arrays of surface points (x_t, z) for each intersected cell
            surf_points_inds (list):                        A list of arrays of surface point indices (x_ind, y_ind) for each intersected cell (corresponds to surf_points)
            valid (np.ndarray) (n,):                        A boolean array indicating if the cell is valid (intersected with compact soil)
            V_Q (float):                                    The volume of the surcharge in the region in front of the intersected cells


        """
        dx = t_dir[0]
        dy = t_dir[1]

        n = intersected_cells.shape[0]
        valid = np.ones(n, dtype=bool)
        # Array holding indices of cells where we want to include the surcharge from
        # Warning: l_surcharge_max must be less than or equal to l_fit_max right now, possibly fix this
        surcharge_inds = np.zeros((0,2), dtype=int)

        # Obtain the points to fit the plane to in coordinates of (x_t, z)
        # where x_t is the distance along the movement direction and z is the height of the compacted soil surface
        surf_points = [np.zeros((0,2),dtype=self.data_type) for _ in range(n)]
        surf_points_inds = [np.zeros((0,2),dtype=int) for _ in range(n)]

        # This is a digitial differential analyzer (DDA) line algorithm
        # see https://en.wikipedia.org/wiki/Digital_differential_analyzer_(graphics_algorithm)
        if (abs(dx) >= abs(dy)):
            step = abs(dx)
        else:
            step = abs(dy)

        dx = dx / step
        dy = dy / step
        for i, intersected_cell_ind in enumerate(intersected_cells):
            x = intersected_cell_ind[0]
            y = intersected_cell_ind[1]
            # Need to determine if the intersection is only with the loose soil or with the compacted surface too
            if (elevation_map[7, intersected_cell_ind[0], intersected_cell_ind[1]] > pierce_dist[i]):
                # If the pierce distance is smaller than loose soil height, then the blade doesn't contact the compacted surface
                # for cell ind intersected_cell_ind[i]. Mark as invalid for downstream use
                valid[i] = False
            # Only trace the line if the cell is valid
            while valid[i]:
                # Was a floor/int(), but since we are dealing with cell centers this should round to the nearest cell center
                xind = round(x)
                yind = round(y)
                # We want to trace the to the blade surface plane from query cell at the height of the 
                # intersection of for the intersected cell to best obtain the distance along the movement direction
                # TODO: Check that this is the same as the intersection height
                # Obtain the height of the intersection for projection onto the blade surface in get_xy_GET_distance()
                point_z = elevation_map[0, intersected_cell_ind[0], intersected_cell_ind[1]].get() - pierce_dist[i]
                inds = np.array([xind, yind])
                x_t = self.get_xy_GET_distance(inds, point_z, t_dir, GET_plane_origin, normal, cell_n, resolution)
                if (x_t >  self.param.l_fit_max): 
                    if surf_points[i].shape[0] > 0:
                        warnings.warn("No surface points found at distance < l_fit_max = {}. May indicate too large a sweep translation.".format(self.param.l_fit_max))
                    break
                if (elevation_map[2, xind, yind] > 0.5):
                    # Only add valid points to the surface points
                    # Obtain the point on the surface of the compacted soil, i.e. cell_height - loose_soil_height
                    q = elevation_map[7, xind, yind].get()
                    z = elevation_map[0, xind, yind].get() - q
                    # If the distance is less than or equal to the maximum length, include in Q calculation
                    if x_t <= self.param.l_surcharge_max:
                        surcharge_inds = np.concatenate((surcharge_inds, np.array([[xind, yind]])), axis=0)
                    surf_points[i] = np.concatenate((surf_points[i], np.array([[x_t[0], z]])), axis=0, dtype=self.data_type)
                    surf_points_inds[i] = np.concatenate((surf_points_inds[i], np.array([[xind, yind]])), axis=0, dtype=int)
                x = x + dx
                y = y + dy
                if (x < 0 or x >= cell_n or y < 0 or y >= cell_n):
                    warnings.warn("Surface points outside of map bounds. Stopping line trace.")
                    break
            # Check if more than 1 surface point was found, i.e. a point beyond the intsection point
            if surf_points[i].shape[0] <= 1 and valid[i]:
                valid[i] = False
                warnings.warn("Insufficient surface points found for FEE. Cell may be invalid.")
            
        # First get unique surcharge inds so we don't double count cells between slices
        surcharge_inds = np.unique(surcharge_inds, axis=0)
        # Now calculate the volume of the surcharge
        V_Q = elevation_map[7, surcharge_inds[:,0], surcharge_inds[:,1]].sum()*resolution**2
        V_Q = V_Q.get()
        return surf_points, surf_points_inds, valid, V_Q
    
    def obtain_FEE_em_params(self, intersected_inds, translation, elevation_map, GET_plane_origin, normal, pierce_dist, cell_n, resolution):
        """
        Obtain the elevation mapping derived parameters (alpha, rho, d, w, Q) for the FEE.
        This is accomplished by finding the parameters for each "slice" and then combining them using
        a weighted average. The surface "in front" of a slice is approximated as a plane with 0 roll, i.e. fit a 
        line to the points "in front" of each section of the blade (represented by intersected cells), where the
        front is defined as the cells in the direction of the translation vector up to a threshold distance, l_fit_max. 
        This line-fit yields the slope of the surface in the direction of movement, i.e. alpha_hat, and the
        height of the point where the approximated surface contacts the blade. The horizontal depth of cut,
        i.e. d_prime, can be derived from this point by subtracting the intersection depth. The blade inclination
        wrt the horizontal plane, i.e. rho_prime, is obtained by projecting the GET surface normal vector onto the 
        slicing plane. The blade inclination wrt the terrain surface is then obtained as rho_hat = alpha_hat + rho_prime.
        From the FEE geometry, the depth of cut per slice is d_hat = d_prime * sin(rho_hat) / sin(rho_hat - alpha_hat).
        Then a weighted average of d_hat and alpha_hat is taken to obtain d_ and alpha_ where the weights increase
        exponentially with d_hat. The blade inclination wrt the terrain surface, rho_, is then obtained from alpha_
        and rho_prime. The blade width, w, is found by finding the extent of the cells centers along the approximated
        FEE blade width, i.e. perpendicular to translation. This is found by projecting the cell centers onto the
        perp t direction and finding the difference of the min and max values. Due to the discretization of the map,
        and the use of cell_center intersection witht he swept volume, the blade width will always be underestimated.
        To help ensure the blade with estimation error has a mean closer to 0, the cell resolution is added to w.
        The volume of the surcharge, V_Q, is obtained by summing the the loose soil "in front" of the blade
        up to a threshold distance from the blade surface, l_surcharge_max. The surcharge force, Q, is then obtained
        by multiplying V_Q by the compacted_soil_moist_unit_weight taking into account the assumed swell factor.
        The parameters are then returned as a dictionary.

        Args:
            intersected_inds (np.ndarray) (n,2): The indices of the intersected cells
            translation (np.ndarray) (3,):       The direction of translation
            elevation_map (cp.ndarray) (8,n,n):  The elevation map
            GET_plane_origin (np.ndarray) (3,):  The origin of the GET plane in the map frame
            normal (np.ndarray) (3,):            The normal of the GET plane
            pierce_dist (np.ndarray) (n,):       The pierce distance for each intersected cell
            cell_n (int):                        The number of cells in the map
            resolution (float):                  The resolution of the map

        Returns:
            FEE_params (dict):                   The FEE parameters alpha, rho, d, w, V_Q (prior to soil failure)
            surf_points_dict (dict):             A dictionary of the surface points for each intersected cell {[points[n,2]], [map_inds[n,2]]}
                                                 where points has the form (x_t, z) and n is the number of identified points per intersected cell
                                                 and there lists are length N for N intersected cells.
        """
        surf_points, surf_points_inds, valid, V_Q = self.get_surface_points(intersected_inds, translation[0:2], elevation_map, GET_plane_origin, normal, pierce_dist, cell_n, resolution)
        if np.sum(valid) == 0:
            # Can be caused by an unitialized height map, or only intersections with loose soil
            # warnings.warn("No valid surface points found for FEE")
            return None, None
        # Only use valid surface points in FEE calc
        surf_points = [surf_point for surf_point, is_valid in zip(surf_points, valid) if is_valid]
        surf_points_inds = [surf_point_ind for surf_point_ind, is_valid in zip(surf_points_inds, valid) if is_valid]
        # Create a dictionary for surface points for later identification of which cells to apply estiamted soil-props to
        surf_points_dict = {}
        # Only need to store the x_t values for the surface points right now, but we will leave the z values for now
        surf_points_dict['points'] = surf_points
        surf_points_dict['map_inds'] = surf_points_inds
        # Now fit surface points to a line to get the slope of the surface in the direction of movement (alpha_i)
        # First compute weights for the points
        W = [np.exp(-self.param.surf_interp_coeff * surf_point[:,0]) for surf_point in surf_points]
        # Scale so that W sums to 1 (I don't think this is necessary because the weighted line fit doesn't assume this)
        W = [Wi / np.sum(Wi) for Wi in W]
        # Now find the slope using weighted least squares
        # https://en.wikipedia.org/wiki/Weighted_least_squares
        X = [np.concatenate((np.ones((surf_point.shape[0], 1)),surf_point[:,0:1]),axis=1) for surf_point in surf_points]

        A = [(np.linalg.inv(Xi.T@np.diag(Wi)@Xi)@Xi.T@np.diag(Wi)) for Xi, Wi in zip(X, W)]
        H = [Xi@Ai for Xi, Ai in zip(X, A)]
        # Compute the variance of the residuals for each line fit (aka. the Standard Error of the Estimate)^2
        # Using ddof=2 because we are estimating the variance of the residuals from the data ( I'm not sure about this, maybe should be 1)
        # We want it to be an unbiased estimator
        var_r = np.array([np.var((np.eye(Hi.shape[0]) - Hi)@surf_point[:,1],ddof=2) for Hi, surf_point in zip(H, surf_points)], dtype=self.data_type)
        # Should be equivalent to below (with numerical differences)
        # std_err = np.array([np.sum((surf_point[:,1] - (Xi @ beta_hat_i))**2)/(surf_point.shape[0]-2) for Xi, Wi, surf_point, beta_hat_i in zip(X, W, surf_points, beta_hat)], dtype=self.data_type)
        # Note: Beta here is the intercept and slope of the line of best fit, not the soil failure angle
        beta_hat = np.array([Ai@surf_point[:,1] for Ai, surf_point in zip(A, surf_points)], dtype=self.data_type)
        # Compute the average variances of the line fit parameters
        M_beta = np.array([Ai*var_ri@Ai.T for Ai, var_ri, surf_point in zip(A, var_r, surf_points)], dtype=self.data_type)
        alpha_hat = np.arctan(beta_hat[:,1])
        # Compute the average variances of alpha (variance along the translation direction)
        var_alpha_t = np.mean(np.arctan(np.sqrt(M_beta[:,1,1]))**2)

        # Obtain rho_prime
        # Obtain the (unscaled) normal vector of the slicing plane
        # which is defined as the cross product of the translation vector and the vertical/gravity vector
        # The direction shouldn't matter since we are using this for the projection onto the plane
        sl_n = np.cross(translation, np.array([0.0, 0.0, 1.0]))
        # Project the GET surface normal vector onto the slicing plane
        n_t = normal - np.dot(normal, sl_n) / np.dot(sl_n, sl_n) * sl_n
        # Obtain the blade inclination wrt the horizontal plane
        rho_prime = np.abs(np.arctan2(n_t[0],n_t[2]))
        # rho_i = alpha_i + rho_prime
        rho_hat = alpha_hat + rho_prime

        # Find the difference between the lowest contacted cell and the point where the compact surface
        # approximated by the line fit intersects the blade surface
        intersections = (elevation_map[0, intersected_inds[valid,0], intersected_inds[valid,1]].get() - pierce_dist[valid])
        # d_prime_prime_hat = beta_hat_i[0] - intersections[i]
        d_hat = (beta_hat[:,0]-intersections)*np.cos(alpha_hat)
        # TODO: Debug this and possibly deal with change in indexing of lower calcs
        # Handle possible negative d_hats by excluding them from the average
        d_valid = d_hat >= 0
        if np.any(~d_valid):
            warnings.warn("Negative depth of cut found in FEE calculation. Excluding from average. Consider reducing sweep distance.")
            if np.all(~d_valid):
                warnings.warn("All depth of cuts are negative. Returning None.")
                return None, surf_points_dict
            d_hat = d_hat[d_valid]
            alpha_hat = alpha_hat[d_valid]
            rho_hat = rho_hat[d_valid]
            M_beta = M_beta[d_valid]
        # For later use in determing d_w = d_prime_prime + x_t*(np.tan(alpha)-np.tan(beta+alpha))
        # Compute the average variances of d (variance along the translation direction)
        var_d_t = np.mean(((np.sqrt(M_beta[:,0,0]))*np.sin(rho_hat)/np.sin(rho_hat - alpha_hat))**2)

        # Perform another weighted average to obtain d_ and alpha_. Then compute rho_ from rho_prime and alpha_
        # Define weights as a function of the depth of cut d_hat.
        d_hat_np = np.array(d_hat)
        W_d = np.exp(self.param.depth_weight_avg_coeff*d_hat_np)
        W_d  = W_d/np.sum(W_d) # Must sum to 1 for weighted average
        d_ = np.dot(W_d, d_hat_np)
        alpha_ = np.dot(W_d, alpha_hat)
        rho_ = alpha_ + rho_prime
        # Obtain the depth of cut wrt the horizontal plane that intersects the blade at at the blade-surface intersection point
        d_prime = d_ *  np.sin(rho_ - alpha_) /np.sin(rho_)
        d_prime_prime = d_ / np.cos(alpha_)
        # TODO: Figure out some way to check the quality of the fit and the validity of the parameters combining the individual
        # line fit accuracy with the avearaging of the slices. This information could be valuable to a network that is learning
        # to augment these parameters and could help with error propagation.
        N = d_hat_np.shape[0]
        if N > 1:
            var_d_perp_t = np.sum((d_hat_np - d_)**2)/(N-1)
            # TODO: Think about how to include the variance of alpha_hat[i] in the variance of alpha_
            # This way we could account for uneven terrain along the direction of travel
            var_alpha_perp_t = np.sum((alpha_hat - alpha_)**2)/(N-1)
        else:
            var_d_perp_t = np.array(np.nan, dtype=self.data_type)
            var_alpha_perp_t = np.array(np.nan, dtype=self.data_type)
        # print("d_Std: {}, alpha_std: {}".format(np.sqrt(var_d_perp_t), np.sqrt(var_alpha_perp_t)))

        # Find w by finding the extent of the cells centers along the perp t direction by projecting the cell centers onto the 
        # perp t direction and finding the min and max values.
        perp_t_norm = np.array([translation[1], -translation[0]])
        perp_t_norm = perp_t_norm / np.linalg.norm(perp_t_norm)
        # Get xy coordinates (we want it in the map origin frame so make center = 0,0,0)
        center = np.array([0.0, 0.0, 0.0], dtype=self.data_type)
        # TODO: Perform this elsewhere so we don't have to repeat this calc which is done in get_surface_points()
        points_xy = self.map_index_to_point_xy(intersected_inds[valid][d_valid], center, cell_n, resolution)
        # Project the points onto the perp t direction
        points_perp_t = np.dot(points_xy, perp_t_norm)
        w = np.max(points_perp_t) - np.min(points_perp_t)
        # This will always be an underestimate of the true blade width due to the discretization
        # We can compensate for this partially by increasing w so that the errors are more centered around the true value on average
        # Set the corrected width as the new width
        w = w + resolution
        # Approximate uncertainty in w using a uniform distribution
        # TODO: THis assumes movement along the axes of the map, which may not be the case. Should ideally
        #  consider higher uncertainties in that case.
        var_w = 1/12*(2*resolution)**2

        # Compute the direction of of translation in the xy plane for use by depth controller
        t_dir = translation[0:2] / np.linalg.norm(translation[0:2])

        # Create dictionary of parameters to return
        FEE_em_params = {
            "alpha": alpha_.astype(self.data_type),
            "rho": rho_.astype(self.data_type),
            "d": d_.astype(self.data_type),
            "w": w.astype(self.data_type),
            "V_Q": V_Q.astype(self.data_type),
            "d_prime": d_prime.astype(self.data_type),
            "d_prime_prime": d_prime_prime.astype(self.data_type),
            "t_dir": t_dir.astype(self.data_type),
            "var_d_perp_t": var_d_perp_t.astype(self.data_type),
            "var_alpha_perp_t": var_alpha_perp_t.astype(self.data_type),
            "var_d_t": var_d_t.astype(self.data_type),
            "var_alpha_t": var_alpha_t.astype(self.data_type),
            "var_w": var_w.astype(self.data_type)
        }

        return FEE_em_params, surf_points_dict
    
    def compute_delta_surcharge(self, loose_remaining, compact_swelled_moved, loose_spilled, resolution):
        """
        The change in surcharge over the sweep is computed using the difference between the compacted soil
        that has been swelled and moved and the loose soil that remains. 

        Args:
            loose_remaining (np.ndarray):       The loose soil that remains in the intersected cells
            compact_swelled_moved (np.ndarray): The compacted soil that has been swelled and moved
            loose_spilled (np.ndarray):         The loose soil that has been spilled (not deposited anywere rn)
            resolution (float):                 The resolution of the map
        Returns:
            dV_Q (float):                       The change in surcharge volume over the sweep
        """
        # TODO: Double check that this makes sense
        dV_Q = (np.sum(compact_swelled_moved) - np.sum(loose_remaining) - np.sum(loose_spilled))*resolution**2
        dV_Q = dV_Q.astype(self.data_type)
        return dV_Q
    
    def project_FEE_surcharge(self, FEE_em_params, dV_Q, n_steps):
        """
        Project/Interpolate the surcharge volume for the FEE.
        The initial V_Q, prior to the sweep is provided by the FEE_em_params as V_Q.
        This difference in surcharge volume over the sweep is distributed evenly across the time steps of the
        sweep to obtain the surcharge volume at each time step. The surcharge volume is then limited by the
        maximum surcharge volume per unit width to help deal with not modelling erosion/spill.
        
        Args:
            FEE_em_params (dict):               The elevation mapping derived FEE parameters
            dV_Q (float):                       The change in surcharge volume over the sweep
            n_steps (int):                      The number of time steps in the sweep
        Returns:
            V_Q (np.ndarray) (n_steps,):        The surcharge volume at each time step
            Q (np.ndarray) (n_steps,):          The surcharge force at each time step
        """

        if FEE_em_params is None or dV_Q == None:
            return None, None
        # The width that corresponds to V_Q in the FEE_em_params corresponds to the uncorrected width
        w = FEE_em_params['w']

        # Interpolate the surcharge volume for the FEE
        V_Q = FEE_em_params['V_Q'] + np.linspace(0, dV_Q, n_steps)
        
        # Limit V_Q by the maximum surcharge volume per unit width to help deal with not modelling erosion/spill
        V_q = V_Q/(w)
        V_q_lim = self.param.max_surcharge_vol_per_unit_width
        if V_q_lim >= 0:
            V_q = np.minimum(V_q, V_q_lim)
        V_q = np.maximum(V_q, 0.0) # Ensure that the surcharge volume is non-negative
        V_Q = V_q*w
        
        # Compute Q
        # In math - compacted_soil_moist_unit_weight: gamma (fixed value for elevation mapping),
        #           swell_factor: epsilon
        Q = V_Q * self.param.compacted_soil_moist_unit_weight / self.param.swell_factor
        Q = Q.astype(self.data_type)
        V_Q = V_Q.astype(self.data_type)

        return V_Q, Q
    
    def compute_T_OD(self, GET_plane_origin, translation):

        # Set the rotation matrix as defined by the translation direction
        # First normalize the translation vector
        mag = np.linalg.norm(translation[0:2])
        if mag == 0:
            raise ValueError("Translation vector cannot be zero")
        t_dir = translation[0:2] / mag
        c_yaw = t_dir[0]
        s_yaw = t_dir[1]
        # The transformation matrix from the map origin frame to the blade depth calculation frame
        R_OD = np.array([[c_yaw, -s_yaw, 0],
                        [s_yaw, c_yaw, 0],
                        [0, 0, 1]], dtype=self.data_type)
        # Set the translation as the average of the two GET positions
        trans = -(GET_plane_origin[0:3] + translation[0:3]/2.0)
        T_OD = np.eye(4, dtype=self.data_type)
        T_OD[0:3,0:3] = R_OD
        T_OD[0:3,3] = R_OD@trans

        return T_OD
    
    # TODO: Review that this modificaiton in place works
    def set_blade_depth_calc_params(self, FEE_em_params, GET_plane_origin, translation):
        """
        Assign the variables for the blade depth calculation into the params dictionary.
        These are used to determine the blade depth given the current position of the blade.
        Define a new coordinate system D for the blade depth calculation where the origin is at the center of the GET
        at the halfway point between the two GET positions. This will ensure that the blade depth (d_prime) is returned
        via a linear interpolation of the blade depth in the D coordinate system with
        d_prime = x * tan(alpha) - z + d_prime_offset,
        where x and z are the x and z coordinates in the D coordinate system.
        The x axis is in the direction of the blade translation,the y axis is perpendicular to the translation, and the z axis is vertical.
        The blade depth interpolation is done in this coordinate system as an approximation of the blade depth,
        to enable higher density sampling of the blade depth.
        Args:
            FEE_em_params (dict):           The elevation mapping derived FEE parameters
            GET_plane_origin (np.ndarray):  The origin of the GET plane in the map origin frame
            translation (np.ndarray):       The translation vector from the start to the end of the sweep in the map frame
        Returns:
            FEE_proj_params (dict):         The FEE projection parameters
        """

        T_OD = self.compute_T_OD(GET_plane_origin, translation)
        # Obtain the starting x location of the blade in the D coordinate system at the beginning of the sweep
        G0_O = np.array([0.0, 0.0, 0.0, 1.0], dtype=self.data_type)
        G0_O[0:3] = GET_plane_origin[0:3]
        G0_D = (T_OD@G0_O)[0:3]
        FEE_proj_params = {'T_OD': T_OD, 'G0_D': G0_D}
        # Append the FEE EM parameters to the FEE projection parameters
        FEE_proj_params.update(FEE_em_params)
        return FEE_proj_params
    
    def set_blade_ground_dist_calc_params(self, dist_to_ground, GET_plane_origin, translation):
        """
        Record the distance from the swept volume to the map with the blade position along the sweep
        to be used for the blade ground distance calculation. This calculation can provide a depth
        of cut wrt to the hoizontal.
        Args:
            dist_to_ground (float):         The distance from the swept volume to the map along the blade sweep
            GET_plane_origin (np.ndarray):  The origin of the GET plane in the map origin frame
            translation (np.ndarray):       The translation vector from the start to the end of the sweep in the map frame
        Returns:
            ground_proj_params (dict):         The ground projection parameters
        """
        try:
            T_OD = self.compute_T_OD(GET_plane_origin, translation)
        except ValueError as e:
            # Handle the case where the translation vector is zero
            # This may happen if the blade is not moving or if the movement is too small to be detected
            # In this case, we can set T_OD to None or some default value
            T_OD = None
            warnings.warn(f"Translation vector is zero. Unable to set Transform for interpolation: {e}")
        ground_proj_params = {'T_OD': T_OD, 'dist_to_ground': dist_to_ground}
        return ground_proj_params
        

    def project_blade_depth(self, params, O_r_OG):
        """
        Project the blade depth given the current position of the blade.
        Obtain the blade depth wrt horizontal, d_prime, (useful for control purposes),
        and d, using the provided depth of cut parameters.
        # TODO: Consider modifying this to use depth of cut at blade tip wrt. horizontal instead surface intersection
        #       i.e. d_prime_prime instead of d_prime
        Args:
            params (dict):              The parameters for the FEE projection
            O_r_OG (np.ndarray) (n,3):  The position of the blade (center) in the map origin frame
        Returns:
            d_prime (np.ndarray) (n,):  The blade depth wrt the horizontal plane
            d (np.ndarray) (n,):        The blade depth wrt the terrain surface (i.e. d in FEE)
        """
        if params is None: # Double check this
            # warnings.warn("Blade depth parameters not set. Please call set_blade_depth_calc_params() before calling get_blade_depth()")
            valid = False
            return None, None
        else:
            n = O_r_OG.shape[0]
            # Transform the blade position to the blade depth calculation frame
            O_r_OG = np.concatenate((O_r_OG, np.ones((n,1), dtype=self.data_type)), axis=1)
            D_r_DG = (params['T_OD']@O_r_OG.T).T
            # Currently using the distance along all axes to determine if the blade is too far from the GET,
            # but this could be changed to only use the x and z axes or to also account for changed orientation.
            dist = np.linalg.norm(D_r_DG[:,0:3], axis=1)
            # Keep it simple for now by ensuring all points are within the max projection distance
            if np.any(dist > self.param.em_FEE_max_projection_dist):
                # If the distance is too far, don't project the depth
                return None, None
        # Return d_prime given the assumed surface and current position
        d_prime = np.tan(params['alpha']) * D_r_DG[:,0] - D_r_DG[:,2] + params['d_prime']
        # Compute d from d_prime assuming the same surface angle and blade angle
        # TODO: modify this to handle varying blade angle (need to rework math as it assumes fixed rho)
        d = d_prime * np.sin(params['rho'])/np.sin(params['rho']-params['alpha'])

        # Mask out points that are behind where the blade was at the start of the sweep
        # This is only useful for the single step live computing of the blade depth prior to the sweep
        # as otherwise the sweep ensures that D_r_DG[:,0] > params['G0_D'][0].
        # And the only use of the live update right now is for the blade depth controller during data collection
        # and the data collection is going to assume forward motion with a desired depth of cut for the forward motion
        # we don't need this. Leaving it here in case it is useful in the future, but commenting out.
        # back_pts = D_r_DG[:,0] < params['G0_D'][0]
        # if np.any(back_pts):
        #     debug = 1
        # d_prime[back_pts] = params['G0_D'][2] - D_r_DG[back_pts,2] 
        # # Assume flat terrain for back points, i.e. alpha = 0, rho = rho_prime, d=d_prime
        # d[back_pts] = d_prime[back_pts]

        return d_prime, d
    
    def project_ground_depth(self, O_r_OG):
        """
        Project the blade depth given the current position of the blade.
        Obtain the blade depth wrt the terrain surface, d, using the provided depth of cut parameters.
        Args:
            O_r_OG (np.ndarray) (n,3):  The position of the blade (center) in the map origin frame
        Returns:
            d_prime (np.ndarray) (n,):  The blade depth wrt the horizontal plane
        """
        params = self.ground_proj_params
        if params is None:
            return None
        n = O_r_OG.shape[0]
        # Transform the blade position to the blade depth calculation frame
        O_r_OG = np.concatenate((O_r_OG, np.ones((n,1), dtype=self.data_type)), axis=1)
        D_r_DG = (params['T_OD']@O_r_OG.T).T
        d_prime = -D_r_DG[:,2] - params['dist_to_ground']
        return d_prime
    
    def get_blade_depth(self ,M_r_MG, vel_xy, map_center):
        '''
        Obtain blade depth given the current position of the blade. This may be useful for control purposes.
        This function determines the swept mesh projection parameters to use based on the velocity direction.
        The default is to use the positive (swept volume along the blade normal), but if the velocity is in the opposite
        direction then the negative swept volume parameters are used. If the velocity is not in either direction then
        the blade depth is not projected.
        Args:
            M_r_MG (np.ndarray)(3,):        The origin of the GET in the map frame over the sweep
            vel_xy (np.ndarray)(2,):        The velocity of the blade in the xy plane in the map frame
            map_center (np.ndarray)(3,):   The center of the map in the map frame
        Returns:
            d_prime (np.ndarray)(1,):       The blade depth wrt the horizontal plane
            d (np.ndarray)(1,):             The blade depth wrt the terrain surface
        '''
        # Select the appropriate FEE_proj_params based on the t_dir
        FEE_proj_params = None
        # Default to using positive direction FEE projection parameters, if available
        if self.pos_swept_mesh_FEE_projection_params is not None:
            pos_vel_dot = np.dot(vel_xy, self.pos_swept_mesh_FEE_projection_params['t_dir'])
            if pos_vel_dot > 0:
                # Use the positive direction FEE projection parameters
                FEE_proj_params = self.pos_swept_mesh_FEE_projection_params
        # Only use the negative direction FEE projection parameters if the positive direction is not valid
        if self.neg_swept_mesh_FEE_projection_params is not None and FEE_proj_params is None:
            neg_vel_dot = np.dot(vel_xy, self.neg_swept_mesh_FEE_projection_params['t_dir'])
            if neg_vel_dot > 0:
                # Use the negative direction FEE projection parameters
                FEE_proj_params = self.neg_swept_mesh_FEE_projection_params
       
        # Move the swept volume poistion to the map origin frame
        O_r_OG = M_r_MG - map_center
        # Make (n,3) for compatibility with project_blade_depth
        O_r_OG = O_r_OG[None]
        d_prime = None
        if FEE_proj_params is not None:
            d_prime, d = self.project_blade_depth(FEE_proj_params, O_r_OG)
        if d_prime is None:
            # If we are unable to project the blade_depth with the FEE params then use the ground projection params
            d_prime = self.project_ground_depth(O_r_OG)
            d = d_prime

        return d_prime, d
    
    @staticmethod
    def get_rel_blade_pose(T_MG, plane_fit_params):
        """ Get the relative pose of the blade wrt the surface defined by the plane_fit_params
        Args:
            T_MG (np.ndarray) (4,4):        The transformation matrix from the map frame to the GET frame
            plane_fit_params (dict):        The parameters for the plane fit with the keys:
                                                "M_r_MP": The point on the plane in the map frame
                                                "P_normal": The normal vector of the plane in the map frame
                                                "G_r_GC0": The bottom right corner (C0) of the GET in the GET frame
                                                "G_r_C0C1": The vector from the bottom right corner (C0) of the GET to the bottom left corner (C1) of the GET in the GET frame
                                                "G_normal": The normal vector of the GET in the map frame
        Returns:
            pose_dict (dict):    The relative pose of the blade wrt the surface defined by the plane_fit_params with the keys:
                                    "roll": The roll angle of the blade wrt the surface in radians
                                    "pitch": The pitch angle of the blade wrt the surface in radians
                                    "height": The distance of the blade above the surface in meters
                                    "M_r_MBEC": The position of the blade edge center in the map frame

        """
        # TODO: Move this into a control class at some point to enable 
        # Blade Edge parameters
        G_r_C0C1 = plane_fit_params["G_r_C0C1"]
        G_r_GC0 = plane_fit_params["G_r_GC0"]
        # Get blade normal vector in map frame
        G_normal = T_MG[:3, :3]@plane_fit_params["G_normal"]

        # Plane Fit parameters
        P_normal = plane_fit_params['P_normal']
        M_r_MP = plane_fit_params['M_r_MP']

        # Get a normalized blade edge vector in the map frame
        blade_edge_vector = T_MG[:3, :3] @ G_r_C0C1
        blade_edge_vector = blade_edge_vector / np.linalg.norm(blade_edge_vector)

        # Blade edge center in map frame
        M_r_MBEC = T_MG[:3,:3] @ (G_r_GC0 + G_r_C0C1/2.0) + T_MG[:3,3]

        # Vector from plane point to blade edge center in map frame
        M_r_Cbec = M_r_MBEC - M_r_MP

        # Extract the relative pose
        # roll_angle is the angle between the blade edge and the surface
        theta = np.arccos(np.dot(P_normal, blade_edge_vector))
        roll_angle = np.pi / 2 - theta

        # pitch_angle is the angle between the blade normal and the plane 
        pitch_angle = np.pi / 2 -np.arccos(np.dot(P_normal, G_normal))
        
        # blade_cent_dist distance is the distance between the blade and the blane at the center
        # Plane equation from normal and point on the plane is 
        # P_normal[0](x-C[0]) + P_normal[1](y-C[1]) + P_normal[2](z-C[2]) = 0
        # Convert to ax + by + cz + d = 0
        a = P_normal[0]
        b = P_normal[1]
        c = P_normal[2]
        d = -P_normal[0]*M_r_MP[0] - P_normal[1]*M_r_MP[1] - P_normal[2]*M_r_MP[2]
        dist = np.abs(a*M_r_MBEC[0] + b*M_r_MBEC[1] + c*M_r_MBEC[2] + d)/ np.sqrt(a**2+b**2+c**2)
        # Get sign of dist based on dot product between vector from plane point to blade edge center point
        sgn = np.sign(np.dot(P_normal, M_r_Cbec))
        height = sgn * dist

        return {"roll": roll_angle, "pitch": pitch_angle, "height": height, "M_r_MBEC": M_r_MBEC}
        
    
    def update_map_with_swept_volume(self, swept_mesh, normal, translation, O_r_OG, n_steps, var_h, elevation_map, cell_n, resolution, FEE_proj_params={}, obtain_FEE_em_params=True, GET_plane_origin=None):
        """
        Update the elevation map in place with the a swept volume derived from the GET. The coordinate frame is
        assumed to be the map origin frame O to reduce coordinate conversions. This means that the swept_mesh, 
        GET_plane_origin, normal vector, and translational vector are in the map origin frame.
        Args:
            swept_mesh (trimesh.Trimesh):    The swept volume of the GET in the map origin frame
            normal (np.ndarray):             The normal vector of the starting plane of the swept volume
            translation (np.ndarray):        The translation vector of the swept volume in the map frame
            O_r_OG (np.ndarray) (n_steps,3): The position of the blade (center) in the map origin frame
            n_steps (int):                   The number of time steps taken over the translation
            var_h (float):                   The variance of the height of the swept volume
            elevation_map (xp.ndarray):      The full starting elevation map to update in place
            cell_n (int):                    The number of cells in the map
            resolution (float):              The resolution of the map
            FEE_proj_params (dict):          The parameters for the FEE projection
            obtain_FEE_em_params (bool):     Whether to obtain the EM derived parameters for the FEE
            GET_plane_origin (np.ndarray):   The origin of the GET plane in the map frame (any point on the surface of the GET)
        Returns:
            FEE_em_params_proj (dict):       The FEE EM parameters for the projected cells
            FEE_proj_params (dict):          The possibly updated FEE projection parameters
            FEE_valid (bool):                Whether the returned FEE parameters are valid for computing a force,
                                             i.e. not just a projection.
            surf_points_dict (dict):         A dictionary of the surface points for each intersected cell {[points[n,2]], [map_inds[n,2]]}
            intersected_inds (np.ndarray) (:,2): The indices of the intersected cells
            move_dir (np.ndarray) (3,):     The direction of movement of the blade
        """
        assert n_steps == O_r_OG.shape[0], "The number of time steps must match the number of blade positions"
        # Obtain a map frame aligned bounding box for the swept volume
        bbox = swept_mesh.bounding_box
        bbox_verts = bbox.vertices.astype(self.data_type)
        # Find cells in the elevation map that are within the bounding box of the swept volume
        map_center = np.array([0.0, 0.0, 0.0], dtype=self.data_type)
        bb_indices, points_minmax = self.bounding_box_to_map_index(bbox_verts, map_center, cell_n, resolution)
        # Note that these z values are in the map origin frame
        min_sv_z = points_minmax[0, 2]
        max_sv_z = points_minmax[1, 2]
        # Extract submap from the elevation map and convert to a numpy array to enable ray casting with trimesh
        inds_i, inds_j = np.meshgrid(np.arange(bb_indices[0,0], bb_indices[1,0]+1), np.arange(bb_indices[0,1], bb_indices[1,1]+1), indexing='ij')
        submap = elevation_map[:,inds_i, inds_j]
        # Deal with fact that these cells may not be valid. If we have no elevation data for those cells then we can't cut them
        update_elevation = True
        # TODO: Could perform ray cast even if there is no possibility of intersection if we just want to update the upper bound
        valid_cells = submap[2] > 0.5
        if not self.xp.any(valid_cells):
            print("No valid cells in the swept volume")
            # TODO: We could however update the upper bound and is upper bound status of the cells...
            update_elevation = False
        else:
            # Check to see if we are likely to have an intersection by comparing the max_z of the submap and the min_z of the swept volume bounding box
            min_em_z = self.xp.min(submap[0, valid_cells])
            max_em_z = self.xp.max(submap[0, valid_cells])
            max_compact_em_z = self.xp.max(submap[0, valid_cells] - submap[7, valid_cells])
            dist_to_ground = min_sv_z - max_em_z
            dist_to_ground_compact = min_sv_z - max_compact_em_z
            self.ground_proj_params = self.set_blade_ground_dist_calc_params(dist_to_ground_compact.get(), GET_plane_origin, translation)
            if dist_to_ground > 0:
                print("No intersection with swept volume")
                # TODO: We could also update the variance of the cells that are not intersected,
                #       e.g. if the variance is high then we can reduce it if our swept volume is close to the ground
                update_elevation = False
        elevation_updated = False
        FEE_em_params_proj = None
        FEE_em_params = None
        FEE_valid = False
        surf_points_dict = None
        intersected_inds = None
        move_dir = None
        deposit_inds = None
        GET_heights = None
        GET_inds = None
        dV_Q = 0.0
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
            # Z is in map origin frame
            start_z = np.min([min_em_z.item(), min_sv_z]) - epsilon_z
            # Get the cell centers in the map frame
            # Combine inds_i and inds_j to get the indices of the cells in the map
            cell_inds = np.stack((inds_i[valid_cells], inds_j[valid_cells]), axis=1)
            # Get cell_centers in the map origin frame given map_center is provided as [0,0,0]
            cell_centers = self.map_index_to_point_xy(cell_inds, map_center, cell_n, resolution)
            n_cells = cell_centers.shape[0]
            lines = np.zeros((n_cells, 2, 3), dtype=self.data_type)
            lines[:,0,0:2] = cell_centers
            lines[:,1,0:2] = cell_centers
            lines[:,0,2] = start_z
            lines[:,1,2] = submap[0, valid_cells]
            intersections, intersected_lines, pierce_dist, invalid_intersections, invalid_lines = line_mesh_intersection(lines, swept_mesh, coincidence_tol=1e-6)
            intersected_inds = cell_inds[intersected_lines]
            non_intersected_inds = cell_inds[invalid_lines]
            map_update = False
            if len(intersections) > 0:
                # print("Intersections found")
                # Make sure datatypes are consistent
                intersections = intersections.astype(self.data_type)
                pierce_dist = pierce_dist.astype(self.data_type)
                invalid_intersections = invalid_intersections.astype(self.data_type)
                if obtain_FEE_em_params:
                    # Obtain the geometry parameters for the FEE
                    assert GET_plane_origin is not None, "GET_plane_origin must be provided to obtain FEE geometry parameters"
                    FEE_em_params, surf_points_dict = self.obtain_FEE_em_params(intersected_inds, translation, elevation_map, GET_plane_origin, normal, pierce_dist, cell_n, resolution)
                if FEE_em_params is not None:
                    FEE_proj_params = self.set_blade_depth_calc_params(FEE_em_params, GET_plane_origin, translation)
                    FEE_valid = True
                # Find the direction of material movement
                move_dir = self.material_movement_direction(normal, translation, normal_weight=self.param.move_dir_normal_weight)
                if move_dir is None:
                    warnings.warn("Invalid movement direction, skipping update of elevation map")
                else:
                    # Remove the material from the cells that are intersected
                    # Get the indices of the intersected cells
                    intersected_cells = intersected_inds - bb_indices[0]
                    # Update the elevation map with the new heights
                    delta_h = submap[0, intersected_cells[:,0], intersected_cells[:,1]] - intersections[:,2]
                    submap[0, intersected_cells[:,0], intersected_cells[:,1]] = intersections[:,2]
                    # Ensure that loose material is removed from the cells if there is any
                    loose_remaining = np.maximum(submap[7, intersected_cells[:,0], intersected_cells[:,1]]- pierce_dist, 0.0)
                    loose_moved = submap[7, intersected_cells[:,0], intersected_cells[:,1]] - loose_remaining
                    compact_moved = delta_h - loose_moved # Should be gaurateed to be positive or 0
                    submap[7, intersected_cells[:,0], intersected_cells[:,1]] = loose_remaining
                    # ["elevation": 0, "variance": 1, "is_valid": 2, "traversability": 3, "time": 4, "upper_bound": 5, "is_upper_bound": 6, "elevation_loose": 7]` 
                    # Update the variance of the cells. Using simple variance update for now
                    start_var = submap[1, intersected_cells[:,0], intersected_cells[:,1]].copy()
                    # Compute an updated variance based on the change in elevation
                    # The idea here is that the standard deviation of the existing cell height should be reduced by the change in height
                    # The choice of 1 sigma is a bit arbitrary, but it is a reasonable starting point
                    # If the change in height is greater than the 1-sigma bound then the existing variance is set to 0
                    # The variance of the cell is then updated based on the remaining variance and the variance of the blade height
                    # The maximum of the two is taken as the new variance
                    remaining_h_std = np.maximum((np.sqrt(start_var) - delta_h), 0.0)
                    submap[1, intersected_cells[:,0], intersected_cells[:,1]] = np.maximum(remaining_h_std, np.sqrt(var_h))**2
                    # TODO: Consider updating the time layer here as well. Do we that layer to only be for lidar observations, or also for GET updates?
                    submap[4, intersected_cells[:,0], intersected_cells[:,1]] = 0.0
                    # Update the upper bound of the overlapping cells (set to the updated elevation)
                    submap[5, intersected_cells[:,0], intersected_cells[:,1]] = intersections[:,2]
                    # Update the is_upper_bound status of the cells
                    submap[6, intersected_cells[:,0], intersected_cells[:,1]] = 0.0

                    # Compute the change in surcharge over the sweep prior to dealing with errors in deposit locations
                    # in the deposited locations
                    compact_swelled_moved = compact_moved*self.param.swell_factor
                    loose_spilled = loose_moved * self.param.spill_factor
                    dV_Q = self.compute_delta_surcharge(loose_remaining, compact_swelled_moved, loose_spilled, resolution)
                    delta_h_swelled = compact_moved*self.param.swell_factor + loose_moved - loose_spilled

                    # Compute the range of possible changes in elevaiton at the deposit location given the standard deviations of the blade height and the cell height
                    # The lower portion of the range is given by the change in cell height plus the blade height standard deviation
                    # The upper portion of the range can be one of 3 values:
                    # 1. The standard deviation of the cell height
                    # 2. The change in cell height due to swell
                    # 3. The standard deviation of the blade height minus the change in cell height (minimum of 0 if negative)
                    # Which ever of these is the largest is used as the upper bound as it the highest possible value of the soil in the deposited cell for a 1 sigma bound
                    deposit_delta_std_h = (delta_h + np.sqrt(var_h) + np.max([np.sqrt(start_var), delta_h_swelled - delta_h, np.maximum(np.sqrt(var_h)-delta_h, 0.0)]))/2.0
                    deposit_delta_var_h = deposit_delta_std_h**2
                    # Deposit the material in the a new location
                    # Get the indices of the cells that lie under the projection of the swept volume onto the xy plane
                    # Note: using GET_cells instead of intersected_cells because it is possible for soil to be deposited behind
                    # the blade if only a portion of the blade contacts a the terrain. This leads to issues with lekage.
                    # THis solution isn't ideal. I would rather modify find deposit locations to find the non-intersected cells
                    # along move_dir and then deposit up to the swept volume height. If the deposit amount was higher then the algorithm
                    # would continue until the deposit material was exausted. Leaving this as a TODO
                    non_intersected_cells = non_intersected_inds - bb_indices[0]
                    # Find where to deposit the material based on the movement direction
                    deposit_inds = self.find_deposit_locations(intersected_cells, non_intersected_cells, move_dir, submap.shape[1:3])

                    # Deal with fact that some deposit locations may be the same. Need to ensure deposits are conserved
                    unique_deposit_inds, unique_inds, unique_cnt = np.unique(deposit_inds, axis=0, return_index=True, return_counts=True)
                    if np.any(unique_cnt > 1):
                        debug = 1
                        duplicates = unique_deposit_inds[unique_cnt > 1]
                        # Combine the material moved to the same location to avoid overwritting
                        # and ensure material is conserved
                        for dup in duplicates:
                            dup_inds = np.arange(deposit_inds.shape[0])
                            dup_inds = dup_inds[np.all(dup == deposit_inds, axis=1)]
                            # dup_inds[0] should be the first index of the duplicates and in the unique_inds
                            delta_h_swelled[dup_inds[0]] += delta_h_swelled[dup_inds[1:]].sum()
                            deposit_delta_var_h[dup_inds[0]] += deposit_delta_var_h[dup_inds[1:]].sum()
                        # Now update deposit_inds
                        deposit_inds = unique_deposit_inds
                        # And remove the non-unique elements from the moved arrays
                        delta_h_swelled = delta_h_swelled[unique_inds]
                        deposit_delta_var_h = deposit_delta_var_h[unique_inds]
                    # Check if the deposit locations are valid
                    valid_deposit_inds = submap[2, deposit_inds[:,0], deposit_inds[:,1]] > 0.5
                    # Handle case where the material is deposited outside the valid portion of the map
                    # a deposition locaiton may need to be a a cell that is within the map,
                    if not np.all(valid_deposit_inds):
                        warnings.warn("Some deposit locations are not valid. Material not conseved")
                        deposit_inds = deposit_inds[valid_deposit_inds]
                        delta_h_swelled = delta_h_swelled[valid_deposit_inds]
                        deposit_delta_var_h = deposit_delta_var_h[valid_deposit_inds]
                    # Deposit the material in the new location for elevation and loose material
                    # If swell factor is 1 then should be equal to pierce_dist
                    submap[0, deposit_inds[:,0], deposit_inds[:,1]] += delta_h_swelled
                    submap[7, deposit_inds[:,0], deposit_inds[:,1]] += delta_h_swelled
                    # Update the variance of the cells where material was deposited
                    # TODO: Review this method and compare to d'Adamo pg. 114 
                    # Should the pierce_dist factor in here?
                    # submap[1, deposit_inds[:,0], deposit_inds[:,1]] += var_h * self.param.swell_factor**2 
                    submap[1, deposit_inds[:,0], deposit_inds[:,1]] += deposit_delta_var_h
                    # TODO: Consider updating the time layer here as well.
                    submap[4, deposit_inds[:,0], deposit_inds[:,1]] = 0.0
                    # Update the upper bound
                    submap[5, deposit_inds[:,0], deposit_inds[:,1]] = submap[0, deposit_inds[:,0], deposit_inds[:,1]]
                    # Update the is_upper_bound status of the cells
                    submap[6, deposit_inds[:,0], deposit_inds[:,1]] = 0.0
                    elevation_updated = True
                    map_update = True
            if len(invalid_intersections) > 0:
                # print("Updating Upper Bound for non-overlapping cells")
                # Update the elevation map
                # Get the indices of the intersected cells
                ub_cells = non_intersected_inds - bb_indices[0]
                # Update the upper bound of the overlapping cells (set to the updated elevation)
                submap[5, ub_cells[:,0], ub_cells[:,1]] = invalid_intersections[:,2]
                # Update the is_upper_bound status of the cells
                submap[6, ub_cells[:,0], ub_cells[:,1]] = 1.0
                # Will need to update the original elevation map with the updated submap now
                map_update = True
            

            if len(intersections) != 0 or len(invalid_intersections) != 0:
                if len(intersections) == 0:
                    intersections = np.zeros((0, 3), dtype=self.data_type)
                elif len(invalid_intersections) == 0:
                    invalid_intersections = np.zeros((0, 3), dtype=self.data_type)
                # Combine intersections and invalid_intersections to get the bottom surface heights of the swept volume
                GET_heights = np.concatenate((intersections[:,2], invalid_intersections[:,2]), axis=0)
                # Get the indices of the cells that are intersected
                GET_inds = np.concatenate((intersected_inds, non_intersected_inds), axis=0)
            

            if map_update:
                # Copy the updated submap back to the elevation map
                elevation_map[:,inds_i, inds_j] = submap
                # If the map was updated at all that means that we had some sort of intersection with the swept volume
                # and therefore the depth of cut for this movement can be computed
                # d_trans = np.linspace(0, 1, n_steps)
                # O_r_OG = GET_plane_origin[None,:] + translation[None,:]*d_trans[:,None]
                d_prime, d = self.project_blade_depth(FEE_proj_params, O_r_OG)
                V_Q, Q = self.project_FEE_surcharge(FEE_proj_params, dV_Q, n_steps)
                # Now update FEE_proj_params with the final projected surcharge value so that it can be used for the next sweep
                if FEE_proj_params is not None and V_Q is not None:
                    FEE_proj_params['V_Q'] = V_Q[-1]
                # Make sure we can compute the blade depth (using d_prime as a valid flag for both surcharge and blade depth interp)
                if d_prime is not None:
                    FEE_em_params_proj = FEE_proj_params.copy()
                    # Could pull out T_OD if it isn't necessary
                    # Overwrite with the projected values
                    FEE_em_params_proj['d_prime'] = d_prime
                    FEE_em_params_proj['d'] = d
                    FEE_em_params_proj['V_Q'] = V_Q
                    FEE_em_params_proj['Q'] = Q
                    FEE_em_params_proj['d_step'] = np.arange(n_steps)
            # Convert to global index
            if deposit_inds is not None:
                deposit_inds = deposit_inds + bb_indices[0]

        return elevation_updated, FEE_em_params_proj, FEE_proj_params, FEE_valid, surf_points_dict, intersected_inds, move_dir, deposit_inds, GET_heights, GET_inds

    def material_movement_direction(self, normal, translation, normal_weight=0.5):
        """
        Determine the direction of material movement based on the normal of the swept volume face
        and the translation of the swept volume face. This is a heuristic that is not physically based.
        In a real GET-soil interaction, the movement of material would be based on pysical properties of
        the soil and the blade such as cohesion and friction. This is a simple heuristic that assumes
        that material is moved in an a direction between the starting surface normal and the translation vector.
        The z component of the vectors is ignored and movement is only in the xy plane since we are assuming
        a height-grid representation.
        A normal_weight of 0.5 gives equal weight to the normal and translation vectors. When selecting a normal_weight,
        it is useful to think of the normal vector as representing the movement of material in the scenario where no
        cohesion exists between the material and the blade (e.g. sand). Vice versa, the translation vector represents
        the movement of material in the scenario where material is cohesive and "sticks" to the blade (e.g. clay).
        
        Args:
            normal (np.ndarray) (3,):       The normal of the swept volume face
            translation (np.ndarray) (3,):  The translation of the swept volume face
            normal_weight (float):          The weight to give to the normal vector. Default is 0.5
        Returns:
            move_dir (np.ndarray) (2,):     The direction of material movement in the xy plane (normalized)
                                            will return none if the movement direction is invalid
        """
        # Normalize the vectors
        normal_norm = np.linalg.norm(normal)
        translation_norm = np.linalg.norm(translation)
        # TODO: Clean thus up with the below assumption. If this occurs then we should abort?
        assert normal_norm != 0, "Normal vector norm is zero. This shoud never occur"
        normal = normal/ normal_norm
        if translation_norm != 0:
            translation = translation / translation_norm
        # else: The translation vector is zero, averaging will result in normal vector
        
        # Get the xy component of the normal and translation
        normal_xy = normal[:2]
        translation_xy = translation[:2]

        if np.linalg.norm(translation_xy) == 0 and normal_weight == 0:
            warnings.warn("Translation in xy is zero and normal weight is zero. Don't move material.")
        
        # If normal is in the z direction then a movement direction will be dictated by translation. Should't occur in this application
        if np.linalg.norm(normal_xy) == 0:
            warnings.warn("Normal is in z direction (zero in xy). May observe unexpected behavior.")
        
        # Determine the direction of movement as a vector between the normal and translation vectors
        move_dir = normal_xy*normal_weight + translation_xy*(1-normal_weight)
        # Normalize the movement direction
        move_dir_norm = np.linalg.norm(move_dir)
        if move_dir_norm != 0:
            move_dir = move_dir / move_dir_norm
        else:
            warnings.warn("Movement direction is zero. Don't move material")
            # This could be because the swept volue is actually a plane and the normal and translation vectors are orthogonal
            # Then we have pierced the soil, but not moved any material
            move_dir = None
        
        return move_dir

    def update_map_with_GET_movement(self, elevation_map, map_center, cell_n, resolution, T_MG0, T_MG1, M_r_MG, n_steps, var_h, roll=None, FEE=True):
        """
        Update the elevation map with the movement of the GET from T_MG0 to T_MG1
        Args:
            elevation_map (xp.ndarray):     The full starting elevation map to update in place
            map_center (np.ndarray):        The center of the map in the map frame
            cell_n (int):                   The number of cells in the map
            resolution (float):             The resolution of the map
            T_MG0 (np.ndarray):             The initial pose of the GET in the map frame
            T_MG1 (np.ndarray):             The final pose of the GET in the map frame
            M_r_MG (np.ndarray)(n_steps x3):The origin of the GET in the map frame over the sweep
            n_steps (int):                  The number of steps to interpolate between T_MG0 and T_MG1
            var_h (float):                  The variance of the height of the swept volume
            roll (float):                   The roll of the GET in degrees
            FEE (bool):                     Whether to obtain the geometry parameters for the FEE
        """
        resolution = np.float32(resolution)
        var_h = np.float32(var_h)
        # First define swept volume of the GET
        # The swept volume is the volume of the material that the GET has moved through
        # as it moves from T_MG0 to T_MG1
        # Sweep the volume in the frame defined by T_MG0, i.e. relative to the initial position of the GET
        transforms = (hom_inv(T_MG0)@T_MG1)[None, :, :]
        if roll is None:
            # Get relative roll between the two poses, this is used to generate a convex sweep
            roll, _, _ = get_ext_euler_angles(transforms[0,:3,:3], xp=np)
        roll_dirs = np.array([roll]) >= 0
        pos_swept_mesh, pos_translation, neg_swept_mesh, neg_translation = sweep_thin_poly_mesh(self.GET_mesh, transforms, roll_dirs=roll_dirs, convex_interp=True)
        if pos_swept_mesh is not None:
            print("Positive swept volume found")
        if neg_swept_mesh is not None:
            print("Negative swept volume found")
        # T_OG0 = T_OM @ T_MG0
        # Where a point in represented in O can be obtained from a point represented in M by translating by -map_center
        T_OG0 = T_MG0.copy()
        T_OG1 = T_MG1.copy()
        map_center = map_center.reshape(3,1)
        T_OG0[:3,3:] -= map_center
        T_OG1[:3,3:] -= map_center
        # Starting face normal and translation vector are used to determine the direction of material movement (a heuristic)
        # Obtain the normal of the original surface of the GET and put in map origin frame
        normal_G0 = T_OG0[:3, :3]@self.GET_mesh.face_normals[0].astype(self.data_type)
        normal_G1 = T_OG1[:3, :3]@self.GET_mesh.face_normals[0].astype(self.data_type)
        # Obtain a point on the plane of the GET in the map origin frame, used for obtaining FEE geometry parameters
        # Using the center point of the geometry for now
        GET_plane_origin_0 = T_OG0[:3, :3]@self.GET_geometry_origin + T_OG0[:3, 3]
        GET_plane_origin_1 = T_OG1[:3, :3]@self.GET_geometry_origin + T_OG1[:3, 3]
        # Also get the translation between the two poses of the GET
        translation = T_MG1[:3, 3] - T_MG0[:3, 3]

        # Determine the direction we want to fit the plane to based on the motion.
        # If the we only have a positive swept volume then we want set the fit_dir to 1
        # If we only have a negative swept volume then we want to set the fit_dir to -1
        # If we have both then we want to set the fit_dir to 0
        if pos_swept_mesh is not None and neg_swept_mesh is not None:
            fit_dir = 0
        elif pos_swept_mesh is not None:
            fit_dir = 1
        elif neg_swept_mesh is not None:
            fit_dir = -1
        else:
            fit_dir = 0
        
        # Fit a plane to the map surface near the GET
        # TODO: Fit a plane to the desired surface and the frozen reference surface
        plane_fit_params = self.fit_plane_near_GET(elevation_map, T_OG1, normal_G1, map_center, cell_n, resolution, fit_dir, height_layer_ind=0)

        # Check that the translation is in the direction of the normal when we don't have a self intersection
        if (pos_swept_mesh is not None) != (neg_swept_mesh is not None):
            if pos_swept_mesh is not None:
                n = normal_G0
            elif neg_swept_mesh is not None:
                n = -normal_G0
            dot_prod = np.dot(n[:2], translation[:2])
            if dot_prod < 0:
                warnings.warn(
                "Translation is in the opposite direction of the normal. "
                "t o n = {} o {} = {}".format(translation[:2], n[:2], dot_prod)
            )

        # If the mesh had a self intersection then we want to use the translations of the split meshes at the centroids
        # But if not then just use the translation of the full GET centroid
        if pos_translation is None:
            pos_translation = translation
        else:
            # Rotate the translation to the map origin frame
            pos_translation = T_MG0[:3, :3] @ pos_translation
            p_dot_prod = np.dot(normal_G0[:2], pos_translation[:2])
            if p_dot_prod < 0:
                warnings.warn(
                "Positive translation is in the opposite direction of the normal. "
                "t_p o n = {} o {} = {}".format(pos_translation[:2], normal_G0[:2], p_dot_prod)
            )

        if neg_translation is None:
            neg_translation = translation
        else:
            # Rotate the translation to the map origin frame
            neg_translation = T_MG0[:3, :3] @ neg_translation
            n_dot_prod = np.dot(-normal_G0[:2], neg_translation[:2])
            if n_dot_prod < 0:
                warnings.warn(
                "Negative translation is in the opposite direction of the normal. "
                "t_n o n = {} o {} = {}".format(neg_translation[:2], -normal_G0[:2], n_dot_prod)
            )
        
        # Move the swept volume poistion to the map origin frame
        O_r_OG = M_r_MG - map_center.T


        # Initialize in case of no intersections
        # TODO: Figure out how to handle the negative swept volume and the direction of the forces
        elevation_updated = False
        FEE_em_params = None
        FEE_valid = False
        surf_points_dict = None
        intersected_inds = None
        move_dir = None
        deposit_inds = None
        GET_heights = None
        GET_inds = None
        FEE_em_params_pos = None
        FEE_em_params_neg = None
        if pos_swept_mesh is not None and neg_swept_mesh is not None and FEE==True:
            warnings.warn("Self collision detected. No FEE parameters obtained. Updating map.")
            FEE = False
        if pos_swept_mesh is not None:
            # Move the swept volume to the map origin frame
            pos_swept_mesh.apply_transform(T_OG0)
            elevation_updated_pos, FEE_em_params_pos, self.pos_swept_mesh_FEE_projection_params, FEE_valid_pos, surf_points_dict_pos, intersected_inds_pos, move_dir_pos, deposit_inds_pos, GET_heights_pos, GET_inds_pos = self.update_map_with_swept_volume(pos_swept_mesh, normal_G0, pos_translation, O_r_OG, n_steps, var_h, elevation_map, cell_n, resolution, FEE_proj_params=self.pos_swept_mesh_FEE_projection_params, obtain_FEE_em_params=FEE, GET_plane_origin=GET_plane_origin_0)
        if neg_swept_mesh is not None:
            # Flip the direction of the normal for the negative swept volume
            normal_G0 = -normal_G0
            # Move the swept volume to the map origin frame
            neg_swept_mesh.apply_transform(T_OG0)
            elevation_updated_neg, FEE_em_params_neg, self.neg_swept_mesh_FEE_projection_params, FEE_valid_pos, surf_points_dict_neg, intersected_inds_neg, move_dir_neg, deposit_inds_neg, GET_heights_neg, GET_inds_neg = self.update_map_with_swept_volume(neg_swept_mesh, normal_G0, neg_translation, O_r_OG, n_steps, var_h, elevation_map, cell_n, resolution, FEE_proj_params=self.neg_swept_mesh_FEE_projection_params, obtain_FEE_em_params=FEE, GET_plane_origin=GET_plane_origin_0)
        if FEE_em_params_pos is not None and FEE_em_params_neg is not None:
            # Could support this elsewhere by returning both and then combining them after computing the FEE force
            warnings.warn("Combining the FEE parameters for the positive and negative swept volumes is not yet implemented. Not using either.")
            FEE_em_params = None
            FEE_valid = False
            surf_points_dict = None
        elif FEE_em_params_pos is not None:
            FEE_em_params = FEE_em_params_pos
            FEE_valid = FEE_valid_pos
            surf_points_dict = surf_points_dict_pos
        elif FEE_em_params_neg is not None:
            # Need to support this by including the translation direction of the portion of the swept volume
            warnings.warn("Negative swept volume FEE parameters are only partially tested")
            FEE_em_params = FEE_em_params_neg
            FEE_valid = FEE_valid_pos
            surf_points_dict = surf_points_dict_neg
        
        if pos_swept_mesh is not None and neg_swept_mesh is not None:
            elevation_updated = elevation_updated_pos or elevation_updated_neg
            if intersected_inds_pos is None:
                intersected_inds_pos = np.zeros((0,2), dtype=np.int32)
            if intersected_inds_neg is None:
                intersected_inds_neg = np.zeros((0,2), dtype=np.int32)
            # Combine the intersected indices of the positive and negative swept volumes
            intersected_inds = np.concatenate((intersected_inds_pos, intersected_inds_neg), axis=0)
            # Combine the movement directions of the positive and negative swept volumes
            move_dir = [move_dir_pos, move_dir_neg]
            if deposit_inds_pos is None:
                deposit_inds_pos = np.zeros((0,2), dtype=np.int32)
            if deposit_inds_neg is None:
                deposit_inds_neg = np.zeros((0,2), dtype=np.int32)
            deposit_inds = np.concatenate((deposit_inds_pos, deposit_inds_neg), axis=0)
            if GET_heights_pos is None:
                GET_heights_pos = np.zeros((0,), dtype=self.data_type)
            if GET_heights_neg is None:
                GET_heights_neg = np.zeros((0,), dtype=self.data_type)
            GET_heights = np.concatenate((GET_heights_pos, GET_heights_neg), axis=0)
            if GET_inds_pos is None:
                GET_inds_pos = np.zeros((0,2), dtype=np.int32)
            if GET_inds_neg is None:
                GET_inds_neg = np.zeros((0,2), dtype=np.int32)
            GET_inds = np.concatenate((GET_inds_pos, GET_inds_neg), axis=0)
        elif pos_swept_mesh is not None:
            elevation_updated = elevation_updated_pos
            intersected_inds = intersected_inds_pos
            move_dir = [move_dir_pos]
            deposit_inds = deposit_inds_pos
            GET_heights = GET_heights_pos
            GET_inds = GET_inds_pos
        elif neg_swept_mesh is not None:
            elevation_updated = elevation_updated_neg
            intersected_inds = intersected_inds_neg
            move_dir = [move_dir_neg]
            deposit_inds = deposit_inds_neg
            GET_heights = GET_heights_neg
            GET_inds = GET_inds_neg
        
        # Only return positive FEE parameters for now
        return elevation_updated, FEE_em_params, FEE_valid, surf_points_dict, intersected_inds, move_dir, deposit_inds, GET_heights, GET_inds, plane_fit_params


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