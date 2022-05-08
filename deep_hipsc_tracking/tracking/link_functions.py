""" Individual link functions

* :py:func:`pair_tracks_balltree` - Use a balltree to link tracks by distance
* :py:func:`pair_tracks_softassign` - Use the SoftAssign algorithm to link tracks
* :py:func:`pair_tracks_bipartite_match` - Pair tracks with a bipartite maximum matching (Hungarian) algorithm

To create a function, make a function named ``pair_tracks_*`` with the following
signature:

.. code-block:: python

    def pair_tracks_mylink(track1, track2, **kwargs):
        # track1 is an n x 2 array of points from the first frame
        # track2 is an m x 2 array of points from the second frame
        ...
        # Return the indicies in track1 corresponding to indices in track2
        # as k x 1 numpy arrays, k <= min([n, m])
        return ind1, ind2

API Documentation
=================

"""

# Imports

# 3rd party imports
import numpy as np

from scipy.optimize import linear_sum_assignment

from sklearn.neighbors import BallTree

# Our own imports
from ..utils import calc_delaunay_adjacency
from . import soft_assign, remove_duplicates


# Constants

MAX_DIST = 10
RADIUS_INIT = 20

# Helpers


def _add_links_for_radius(track1, track2, max_dist=MAX_DIST):
    # Add the links from track1 to track2

    graph = np.zeros((track1.shape[0], track2.shape[0]))

    tree = BallTree(track1, leaf_size=2)
    point_i, point_dist = tree.query_radius(track2, r=max_dist,
                                            count_only=False,
                                            return_distance=True)
    point_j = np.arange(track2.shape[0]).astype(np.int64)

    for j in point_j:
        i, d = point_i[j], point_dist[j]
        for ii, dd in zip(i, d):
            graph[ii, j] = dd/max_dist - 1
    return graph


# Postprocessing algorithm


def impute_points(track1, track2, sindex1, tindex1to2, unpaired):
    """ Impute points

    :param ndarray track1:
        The track to generate points for
    :param ndarray track2:
        The track to generate points with
    :param dict sindex1:
        The delaunay triangulation of track1
    :param dict tindex1to2:
        The mapping between points in track1 and points in track2
    :param ndarray unpaired:
        The indices of unpaired points in track1
    :returns:
        A dictionary of points to add to track2 corresponding to indices in track1
    """

    new_points = {}

    # Travel through the space time mapping
    for t1_index in unpaired:
        if t1_index not in sindex1:
            continue
        neighbors = sindex1[t1_index]
        mapped_neighbors = [n for n in neighbors if n in tindex1to2]

        if len(mapped_neighbors) < 3:
            continue

        x, y = track1[t1_index]

        # Estimate the transform
        dx = 0
        dy = 0
        for neighbor in mapped_neighbors:
            x1, y1 = track1[neighbor]
            x2, y2 = track2[tindex1to2[neighbor]]
            dx += x2 - x1
            dy += y2 - y1
        new_points[t1_index] = (x + dx/len(mapped_neighbors), y + dy/len(mapped_neighbors))
    return new_points


def postprocess_delaunay(track1, track2, tindex1, tindex2, min_points=10, max_distance=None):
    """ Clean up the tracks with delaunay triangulation

    :param ndarray track1:
        The points in the first timepoint
    :param ndarray track2:
        The points in the second timepoint
    :param ndarray tindex1:
        The correspondences from track1 to track2
    :param ndarray tindex2:
        The correspondences from track2 to track1
    """

    if track1.shape[0] < min_points or track2.shape[0] < min_points:
        print('Cannot triangulate with fewer than {} points'.format(min_points))
        return track1, track2, tindex1, tindex2

    # Create the forward/backward correspondence map
    assert tindex1.shape == tindex2.shape
    tindex1to2 = {}
    tindex2to1 = {}
    for ind1, ind2 in zip(tindex1, tindex2):
        tindex1to2[ind1] = ind2
        tindex2to1[ind2] = ind1

    # Work out the set of unpaired points for each track
    paired1_mask = np.zeros(track1.shape[:1], dtype=bool)
    paired1_mask[tindex1] = 1
    unpaired1_mask = ~paired1_mask
    unpaired1 = np.arange(track1.shape[0])[unpaired1_mask]

    paired2_mask = np.zeros(track2.shape[:1], dtype=bool)
    paired2_mask[tindex2] = 1
    unpaired2_mask = ~paired2_mask
    unpaired2 = np.arange(track2.shape[0])[unpaired2_mask]

    # Triangulate the points and impute
    sindex1 = calc_delaunay_adjacency(track1, max_distance=max_distance, calc_perimeters=False)[0]
    sindex2 = calc_delaunay_adjacency(track2, max_distance=max_distance, calc_perimeters=False)[0]

    new_points2 = impute_points(track1, track2, sindex1, tindex1to2, unpaired1)
    new_points1 = impute_points(track2, track1, sindex2, tindex2to1, unpaired2)

    # Append the mapped points to the tracks
    new_track1 = []
    new_track2 = []
    new_tindex1 = []
    new_tindex2 = []
    i2 = track2.shape[0]
    for i1, coords in new_points2.items():
        new_tindex1.append(i1)
        new_tindex2.append(i2)
        new_track2.append(coords)
        i2 += 1
    i1 = track1.shape[0]
    for i2, coords in new_points1.items():
        new_tindex1.append(i1)
        new_tindex2.append(i2)
        new_track1.append(coords)
        i1 += 1

    if new_track1:
        new_track1 = np.concatenate([track1, np.array(new_track1)], axis=0)
    else:
        new_track1 = track1
    if new_track2:
        new_track2 = np.concatenate([track2, np.array(new_track2)], axis=0)
    else:
        new_track2 = track2

    if new_tindex1:
        new_tindex1 = np.concatenate([tindex1, np.array(new_tindex1)], axis=0)
    else:
        new_tindex1 = tindex1
    if new_tindex2:
        new_tindex2 = np.concatenate([tindex2, np.array(new_tindex2)], axis=0)
    else:
        new_tindex2 = new_tindex2

    return new_track1, new_track2, new_tindex1, new_tindex2


# Individual linkage algorithms


def pair_tracks_bipartite_match(track1, track2, max_dist=MAX_DIST):
    """ Bipartite matching to pair tracks

    Uses the Hungarian algorithm to solve a min cost bipartite graph assignment

    :param track1:
        The m x 2 track to link to
    :param track2:
        The n x 2 track to link from
    :param max_dist:
        Maximum distance to allow a link
    :returns:
        index1, index2, k x 1 arrays mapping points in track1 to track2
    """

    # Link in the forward direction
    graph = _add_links_for_radius(track1, track2,
                                  max_dist=max_dist)

    # Link in the reverse direction
    graph += _add_links_for_radius(track2, track1,
                                   max_dist=max_dist).T

    # Throw out matches with no support
    if graph.shape[0] > graph.shape[1]:
        track1_empty = np.any(graph, axis=0)
    else:
        track1_empty = np.any(graph, axis=1)

    match1, match2 = linear_sum_assignment(graph)

    return match1[track1_empty], match2[track1_empty]


def pair_tracks_balltree(track1, track2, max_dist=MAX_DIST):
    """ Pair off the tracks with a BallTree

    :param track1:
        The m x 2 track to link to
    :param track2:
        The n x 2 track to link from
    :param max_dist:
        Maximum distance to allow a link
    :returns:
        index1, index2, k x 1 arrays mapping points in track1 to track2
    """
    if track1.shape[0] == 0 or track2.shape[0] == 0:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.uint32)

    tree = BallTree(track1, leaf_size=2)
    dist, ind = tree.query(track2, k=1)
    return remove_duplicates(dist, ind, max_dist=max_dist)


def pair_tracks_softassign(track1, track2, radius_init=RADIUS_INIT):
    """ Pair the tracks using the SoftAssign algorithm

    :param track1:
        The m x 2 track to link to
    :param track2:
        The n x 2 track to link from
    :param radius_init:
        The initial maximum radius to consider links from
    :returns:
        index1, index2, k x 1 arrays mapping points in track1 to track2
    """
    pairs = soft_assign(track1, track2, radius_init=radius_init)[0]
    if pairs.ndim != 2:
        return np.array([]), np.array([])
    return pairs[:, 0], pairs[:, 1]
