""" Speed up the gridding code """
# cython: language_level=3

import math

import numpy as np

from sklearn.neighbors import BallTree

cimport cython
cimport numpy as np

INT_TYPE = np.int64
FLOAT_TYPE = np.float64

ctypedef np.int64_t INT_TYPE_t
ctypedef np.float64_t FLOAT_TYPE_t


cdef extern from "math.h":
    long double atan2(long double a, double b)


# Internal functions


cdef _area_of_polygon(np.ndarray[FLOAT_TYPE_t, ndim=2] verts):
    """ Area of arbitrary irregular polygons

    Calculated via the shoelace formula

    :param verts:
        The 2D coordinates of a polygon
    :returns:
        The area of the polygon
    """
    cdef FLOAT_TYPE_t prod_left, prod_right
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] verts_x, verts_y

    if not np.allclose(verts[0, :], verts[-1, :]):
        verts = np.concatenate([verts, verts[0:1, :]])

    verts_x = verts[:, 0]
    verts_y = verts[:, 1]

    prod_left = np.sum(verts_x[:-1]*verts_y[1:])
    prod_right = np.sum(verts_x[1:]*verts_y[:-1])
    return abs(prod_left - prod_right)/2


cdef _center_of_polygon(np.ndarray[FLOAT_TYPE_t, ndim=2] verts):
    """ Center of a polygon

    :param ndarray verts:
        The n x 2 set of coordinates to calculate the center of
    :returns:
        The x, y coordinates of the center of mass
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] verts_x, verts_y, difference, cx_prod, cy_prod
    cdef float prod_left, prod_right, area, cx, cy

    if not np.allclose(verts[0, :], verts[-1, :]):
        verts = np.concatenate([verts, verts[0:1, :]])

    verts_x = verts[:, 0]
    verts_y = verts[:, 1]

    # Area from the shoelace formula, but we need the sign
    prod_left = np.sum(verts_x[:-1]*verts_y[1:])
    prod_right = np.sum(verts_x[1:]*verts_y[:-1])
    area = (prod_left - prod_right)/2

    # Now calculate the signed centroid for each triangle
    difference = (verts_x[:-1]*verts_y[1:] - verts_x[1:]*verts_y[:-1])
    cx_prod = (verts_x[:-1] + verts_x[1:]) * difference
    cy_prod = (verts_y[:-1] + verts_y[1:]) * difference

    # Sum and scale by the signed area
    cx = 1/(6*area) * np.sum(cx_prod)
    cy = 1/(6*area) * np.sum(cy_prod)
    return cx, cy

  
cdef _sort_coordinates(np.ndarray[FLOAT_TYPE_t, ndim=2] coords):
    """ Sort the coordinates in a polygon

    :param ndarray coords:
        The n x 2 coordinate array
    :returns:
        The sorted coordinate array
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] theta
    cdef float cx, cy
    if coords.shape[0] < 1:
        return coords

    # Sort the coordinates by angle
    cx, cy = _center_of_polygon(coords)

    theta = np.arctan2(coords[:, 1] - cy,
                       coords[:, 0] - cx)
    return coords[np.argsort(theta), :]


cdef merge_points_cluster(np.ndarray[FLOAT_TYPE_t, ndim=2] points1,
                          np.ndarray[FLOAT_TYPE_t, ndim=2] points2,
                          FLOAT_TYPE_t max_dist):
    """ Merge the points using radial clustering

    Cluster all points into clusters with a maximum radius, then return all
    cluster centers of mass

    :param ndarray points1:
        The first frame points to merge
    :param ndarray points2:
        The second frame points to merge
    :param float max_dist:
        The maximum distance to cluster points over
    :returns:
        A single merged point set
    """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] points
    cdef np.ndarray[INT_TYPE_t, ndim=1] ind
    cdef np.ndarray inds
    cdef list clusters, unused_inds
    cdef set used
    cdef INT_TYPE_t i

    if points1.shape[0] == 0 and points2.shape[0] == 0:
        return points1
    elif points1.shape[0] == 0:
        points = points2
    elif points2.shape[0] == 0:
        points = points1
    else:
        points = np.concatenate([points1, points2], axis=0)

    # Handle nans in arrays
    points = points[~np.any(np.isnan(points), axis=1), :]

    if points.shape[0] < 2:
        return points

    tree = BallTree(points)
    inds = tree.query_radius(points, r=max_dist)

    # Cluster each set and return centers of mass
    clusters = []
    used = set()
    for ind in inds:
        unused_inds = []
        for i in ind:
            if i in used:
                continue
            unused_inds.append(i)
            used.add(i)
        if unused_inds:
            clusters.append(np.mean(points[unused_inds, :], axis=0))
    return np.array(clusters)


# Exposed functions


def calc_delta_density(list timepoint1_points, list timepoint2_points,
                       list timepoint1_warp_points, list timepoint2_warp_points,
                       dict timepoint1_mesh, dict timepoint2_mesh,
                       dict timepoint_links):

    cdef int i, j, i0, j0
    cdef float divergence, curl
    cdef dict inv_timepoint_links, delta_divergence, delta_curl, delta_perimeters, delta_warp_perimeters
    cdef list i_to_j, j_to_i, i_to_j_warp, j_to_i_warp
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] coords, warp_coords

    inv_timepoint_links = {j: i for i, j in timepoint_links.items()}

    delta_divergence = {}
    delta_curl = {}
    delta_perimeters = {}
    delta_warp_perimeters = {}

    # Find the neighbors of the points in both sets
    for i0, j0 in timepoint_links.items():
        if i0 not in timepoint1_mesh:
            continue
        if j0 not in timepoint2_mesh:
            continue

        # Compare the local mesh around i0 and j0 - forwards
        i_neighbors = timepoint1_mesh[i0]
        i_to_j = [(timepoint1_points[i], timepoint2_points[timepoint_links[i]])
                  for i in i_neighbors if i in timepoint_links]
        if not timepoint1_warp_points or not timepoint2_warp_points:
            i_to_j_warp = []
        else:
            i_to_j_warp = [(timepoint1_warp_points[i], timepoint2_warp_points[timepoint_links[i]])
                           for i in i_neighbors if i in timepoint_links]
        i_divergence, i_curl, i_coords, i_warp_coords = calc_delta_stats(i_to_j, i_to_j_warp)

        # Compare the local mesh around i0 and j0 - backwards
        j_neighbors = timepoint2_mesh[j0]
        j_to_i = [(timepoint1_points[inv_timepoint_links[j]], timepoint2_points[j])
                  for j in j_neighbors if j in inv_timepoint_links]
        if not timepoint1_warp_points or not timepoint2_warp_points:
            j_to_i_warp = []
        else:
            j_to_i_warp = [(timepoint1_warp_points[inv_timepoint_links[j]], timepoint2_warp_points[j])
                           for j in j_neighbors if j in inv_timepoint_links]

        j_divergence, j_curl, j_coords, j_warp_coords = calc_delta_stats(j_to_i, j_to_i_warp)

        # Composite the coordinates
        if np.isnan(i_divergence) and np.isnan(j_divergence):
            continue

        if np.isnan(i_divergence) and ~np.isnan(i_divergence):
            divergence = j_divergence
            curl = j_curl
            coords = j_coords
            warp_coords = j_warp_coords
        elif ~np.isnan(i_divergence) and np.isnan(i_divergence):
            divergence = i_divergence
            curl = i_curl
            coords = i_coords
            warp_coords = i_warp_coords
        else:
            divergence = (i_divergence + j_divergence)/2
            curl = (i_curl + j_curl)/2
            coords = _sort_coordinates(merge_points_cluster(i_coords, j_coords, max_dist=1))
            warp_coords = _sort_coordinates(merge_points_cluster(i_warp_coords, j_warp_coords, max_dist=0.001))

        delta_divergence[i0, j0] = divergence
        delta_curl[i0, j0] = curl
        if coords.shape[0] > 2:
            delta_perimeters[i0, j0] = coords
        if warp_coords.shape[0] > 2:
            delta_warp_perimeters[i0, j0] = warp_coords

    return delta_divergence, delta_curl, delta_perimeters, delta_warp_perimeters


def calc_delta_stats(list coords, list warp_coords):
    """ Calculate the stats for the changing polygons

    :param list[tuple] coords:
        The list of paired coordinates from timepoint1, timepoint2
    :param list[tuple] warp_coords:
        The list of paired warped coordinates from timepoint1, timepoint2
    :returns:
        divergence, curl, the original perimeter, and the warped perimeter
    """

    cdef list lcoords1, lcoords2, lwarp_coords1, lwarp_coords2
    cdef tuple p1, p2
    cdef float curl, divergence, area1, area2

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] center1, center2, theta1, theta2
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] coords1, coords2, warp_coords1, warp_coords2
    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] sorted_coords1, sorted_coords2, sorted_coords, sorted_warp_coords

    cdef np.ndarray[INT_TYPE_t, ndim=1] index1, index2

    if len(coords) < 3:
        return np.nan, np.nan, np.zeros((0, 2), dtype=FLOAT_TYPE), np.zeros((0, 2), dtype=FLOAT_TYPE)

    # Split the coordinates for easier analysis
    lcoords1 = []
    lcoords2 = []
    for p1, p2 in coords:
        lcoords1.append(p1)
        lcoords2.append(p2)
    coords1 = np.array(lcoords1, dtype=FLOAT_TYPE)
    coords2 = np.array(lcoords2, dtype=FLOAT_TYPE)

    # Split the warp coordinates too, if we were passed them
    lwarp_coords1 = []
    lwarp_coords2 = []
    if not warp_coords:
        warp_coords1 = warp_coords2 = np.zeros((0, 2), dtype=FLOAT_TYPE)
    else:
        for p1, p2 in warp_coords:
            lwarp_coords1.append(p1)
            lwarp_coords2.append(p2)
        warp_coords1 = np.array(lwarp_coords1)
        warp_coords2 = np.array(lwarp_coords2)
        assert warp_coords1.shape[0] == coords1.shape[0]

    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)

    # Curl is proportional to the overall rotation of the points
    # FIXME: Should I add in a moment arm here?
    theta1 = np.arctan2(coords1[:, 1] - center1[1], coords1[:, 0] - center1[0])
    theta2 = np.arctan2(coords2[:, 1] - center2[1], coords2[:, 0] - center2[0])
    curl = np.mean(theta2 - theta1)

    index1 = np.argsort(theta1)
    index2 = np.argsort(theta2)

    # Sort the coordinates to make area calcs work
    sorted_coords1 = coords1[index1, :]
    sorted_coords2 = coords2[index2, :]
    sorted_coords = (sorted_coords1 + sorted_coords2)/2

    # Sort order should be valid for the warp too
    if warp_coords1.shape[0] > 0 and warp_coords2.shape[0] > 0:
        sorted_warp_coords = (warp_coords1[index1, :] + warp_coords2[index2, :])/2
    else:
        sorted_warp_coords = np.zeros((0, 2), dtype=FLOAT_TYPE)

    # Divergence is proportional to the ratio of the areas of the polygons
    area1 = _area_of_polygon(sorted_coords1)
    area1 = max([1, area1])
    area2 = _area_of_polygon(sorted_coords2)
    area2 = max([1, area2])

    divergence = np.log(area2 / area1)
    return divergence, curl, sorted_coords, sorted_warp_coords


def calc_local_density(list timepoint_points,
                       list timepoint_warp_points,
                       dict timepoint_mesh):
    """ Calculate the local density around a point

    :param list timepoint_points:
        The n x 2 list of coordinates in real space
    :param list timpoint_warp_points:
        The n x 2 list of coordinates in warp (circularized) space
    :param dict timepoint_mesh:
        The triangulated mesh to use for each point
    """

    cdef dict timepoint_local_density = {}
    cdef dict timepoint_local_perimeters = {}
    cdef dict timepoint_local_warp_perimeters = {}
    cdef set i_neighbors

    cdef np.ndarray[FLOAT_TYPE_t, ndim=2] coords, sorted_coords, warp_coords

    cdef int j, i
    cdef float area

    for j, i_neighbors in timepoint_mesh.items():

        coords = np.array([timepoint_points[i] for i in i_neighbors], dtype=FLOAT_TYPE)

        if np.any(np.isnan(coords)):
            continue

        sorted_coords = _sort_coordinates(coords)

        # Density is proportional to the ratio of the areas of the polygons
        area = _area_of_polygon(sorted_coords)
        area = max([area, 1])  # minimum possible area

        timepoint_local_density[j] = 1 / area
        timepoint_local_perimeters[j] = sorted_coords

        # Apply the sort to the warped coordinates
        if timepoint_warp_points:
            warp_coords = np.array([timepoint_warp_points[i] for i in i_neighbors], dtype=FLOAT_TYPE)
            timepoint_local_warp_perimeters[j] = _sort_coordinates(warp_coords)

    return timepoint_local_density, timepoint_local_perimeters, timepoint_local_warp_perimeters


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_average_tri_density(np.ndarray[FLOAT_TYPE_t, ndim=2] timepoint_points,
                             np.ndarray[INT_TYPE_t, ndim=2] timepoint_triangles,
                             INT_TYPE_t num_points,
                             INT_TYPE_t num_triangles):
    """ Calculate the average density based on triangle area """

    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] areas, counts
    cdef np.ndarray mask
    cdef INT_TYPE_t t0, t1, t2, idx
    cdef FLOAT_TYPE_t px0, px1, px2, py0, py1, py2, area

    areas = np.zeros((num_points, ))
    counts = areas.copy()

    for idx in range(num_triangles):
        t0 = timepoint_triangles[idx, 0]
        t1 = timepoint_triangles[idx, 1]
        t2 = timepoint_triangles[idx, 2]

        px0 = timepoint_points[t0, 0]
        py0 = timepoint_points[t0, 1]

        px1 = timepoint_points[t1, 0]
        py1 = timepoint_points[t1, 1]

        px2 = timepoint_points[t2, 0]
        py2 = timepoint_points[t2, 1]

        # Area of a triangle, given its 2D coordinates
        area = abs(px0*(py1 - py2) + px1*(py2 - py0) + px2*(py0 - py1)) / 2.0
        areas[t0] += area
        areas[t1] += area
        areas[t2] += area

        counts[t0] += 1
        counts[t1] += 1
        counts[t2] += 1

    mask = counts > 0
    areas[mask] = areas[mask] / counts[mask]
    areas[~mask] = np.nan
    return areas


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_average_segment_divergence(np.ndarray[FLOAT_TYPE_t, ndim=1] timepoint1_areas,
                                    np.ndarray[FLOAT_TYPE_t, ndim=1] timepoint2_areas,
                                    np.ndarray[INT_TYPE_t, ndim=2] timepoint_links,
                                    INT_TYPE_t num_points1,
                                    INT_TYPE_t num_points2,
                                    INT_TYPE_t num_links,
                                    FLOAT_TYPE_t time_scale):
    """ Calculate the average divergence per timepoint """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] divergences1, counts1, divergences2, counts2
    cdef np.ndarray mask1, mask2
    cdef INT_TYPE_t t1, t2, idx
    cdef FLOAT_TYPE_t area1, area2

    divergences1 = np.zeros((num_points1, ))
    counts1 = divergences1.copy()

    divergences2 = np.zeros((num_points2, ))
    counts2 = divergences2.copy()

    for idx in range(num_links):
        t1 = timepoint_links[idx, 0]
        t2 = timepoint_links[idx, 1]

        area1 = timepoint1_areas[t1]
        area2 = timepoint2_areas[t2]

        divergences1[t1] = (area2 - area1) / time_scale
        divergences2[t2] = (area1 - area2) / time_scale
        counts1[t1] += 1
        counts2[t2] += 1

    mask1 = counts1 > 0
    divergences1[mask1] = divergences1[mask1]/counts1[mask1]
    divergences1[~mask1] = np.nan

    mask2 = counts2 > 0
    divergences2[mask2] = divergences2[mask2]/counts2[mask2]
    divergences2[~mask2] = np.nan
    return divergences1, divergences2


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_average_segment_angle(np.ndarray[FLOAT_TYPE_t, ndim=2] timepoint1_points,
                               np.ndarray[FLOAT_TYPE_t, ndim=2] timepoint2_points,
                               dict timepoint1_mesh,
                               dict timepoint_links,
                               INT_TYPE_t num_points1,
                               INT_TYPE_t num_points2):
    """ Average angle between link segments """
    cdef np.ndarray[FLOAT_TYPE_t, ndim=1] angles1, counts1, angles2, counts2
    cdef np.ndarray mask1, mask2
    cdef INT_TYPE_t t1, t2, i1, i2
    cdef FLOAT_TYPE_t pxc1, pyc1, pxc2, pyc2, px1, py1, px2, py2, angle, count

    angles1 = np.zeros((num_points1, ))
    counts1 = angles1.copy()

    angles2 = np.zeros((num_points2, ))
    counts2 = angles2.copy()

    for t1, t2 in timepoint_links.items():
        pxc1 = timepoint1_points[t1, 0]
        pyc1 = timepoint1_points[t1, 1]
        pxc2 = timepoint2_points[t2, 0]
        pyc2 = timepoint2_points[t2, 1]
        angle = 0.0
        count = 0.0
        if t1 not in timepoint1_mesh:
            continue
        for i1 in timepoint1_mesh[t1]:
            if i1 not in timepoint_links:
                continue
            i2 = timepoint_links[i1]

            px1 = timepoint1_points[i1, 0]
            py1 = timepoint1_points[i1, 1]
            px2 = timepoint2_points[i2, 0]
            py2 = timepoint2_points[i2, 1]

            angle += atan2(py2 - pyc2, px2 - pxc2)
            angle -= atan2(py1 - pyc1, px1 - pxc1)
            count += 1
        if count < 1:
            continue

        angles1[t1] += angle
        counts1[t1] += count

        angles2[t2] -= angle
        counts2[t2] += count

    mask1 = counts1 > 0
    angles1[mask1] = angles1[mask1]/counts1[mask1]
    angles1[~mask1] = np.nan

    mask2 = counts2 > 0
    angles2[mask2] = angles2[mask2]/counts2[mask2]
    angles2[~mask2] = np.nan
    return angles1, angles2
