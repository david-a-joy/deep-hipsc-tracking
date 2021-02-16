# Standard lib
import unittest

# 3rd party
import numpy as np

# Our own imports
from deep_hipsc_tracking.tracking import link_functions

# Data

T1 = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
]).T
T2 = np.array([
    [1.1, 2.1, 3.1, 4.1],
    [1.1, 2.1, 3.1, 4.1],
]).T
T3 = np.array([
    [1.2, 2.2, 3.2, 4.2, 5.2],
    [1.2, 2.2, 3.2, 4.2, 5.2],
]).T
T4 = np.array([
    [1.3, 3.3, 4.3, 5.3],
    [1.3, 3.3, 4.3, 5.3],
]).T
T5 = np.array([
    [1.4, 2.4, 3.4, 5.4],
    [1.4, 2.4, 3.4, 5.4],
]).T
TRACKS = [
    (1, None, T1),
    (2, None, T2),
    (3, None, T3),
    (4, None, T4),
    (5, None, T5),
]


# Tests


class TestPostprocessDelaunay(unittest.TestCase):

    def test_pair_timepoints_circle_forwards(self):

        t = np.linspace(0, np.pi*2, 13)[:12]
        x = 1.0 * np.cos(t)
        y = 1.0 * np.sin(t)

        points2 = np.stack([x, y], axis=1)
        points1 = np.concatenate([points2, np.array([[0, 0]])])

        ind1, ind2 = link_functions.pair_tracks_balltree(points1, points2, max_dist=1.0)
        res1, res2, ind1, ind2 = link_functions.postprocess_delaunay(points1, points2, ind1, ind2)

        exp_ind1 = np.arange(13)
        exp_ind2 = np.arange(13)

        np.testing.assert_almost_equal(res1, points1)
        np.testing.assert_almost_equal(res2, points1)  # Make sure we impute the center point
        np.testing.assert_almost_equal(ind1, exp_ind1)
        np.testing.assert_almost_equal(ind2, exp_ind2)

    def test_pair_timepoints_circle_backwards(self):

        t = np.linspace(0, np.pi*2, 13)[:12]
        x = 1.0 * np.cos(t)
        y = 1.0 * np.sin(t)

        points1 = np.stack([x, y], axis=1)
        points2 = np.concatenate([points1, np.array([[0, 0]])])

        ind1, ind2 = link_functions.pair_tracks_balltree(points1, points2, max_dist=1.0)
        res1, res2, ind1, ind2 = link_functions.postprocess_delaunay(points1, points2, ind1, ind2)

        exp_ind1 = np.arange(13)
        exp_ind2 = np.arange(13)

        np.testing.assert_almost_equal(res1, points2)  # Make sure we impute the center point
        np.testing.assert_almost_equal(res2, points2)
        np.testing.assert_almost_equal(ind1, exp_ind1)
        np.testing.assert_almost_equal(ind2, exp_ind2)

    def test_pair_timepoints_expanding_circle_backwards(self):

        t = np.linspace(0, np.pi*2, 13)[:12]
        x = 1.0 * np.cos(t)
        y = 1.0 * np.sin(t)

        points1 = np.stack([x, y], axis=1)
        points2 = np.concatenate([points1, np.array([[0, 0]])]) * 1.5

        ind1, ind2 = link_functions.pair_tracks_balltree(points1, points2, max_dist=1.0)

        res1, res2, ind1, ind2 = link_functions.postprocess_delaunay(points1, points2, ind1, ind2)

        exp_ind1 = np.arange(13)
        exp_ind2 = np.arange(13)

        np.testing.assert_almost_equal(ind1, exp_ind1)
        np.testing.assert_almost_equal(ind2, exp_ind2)

        np.testing.assert_almost_equal(res1, np.concatenate([points1, np.array([[0, 0]])]))
        np.testing.assert_almost_equal(res2, points2)

    def test_pair_timepoints_shifting_circle_backwards(self):

        t = np.linspace(0, np.pi*2, 13)[:12]
        x = 1.0 * np.cos(t)
        y = 1.0 * np.sin(t)

        points1 = np.stack([x, y], axis=1)
        points2 = np.concatenate([points1, np.array([[0, 0]])])
        points2[:, 0] += 0.2

        ind1, ind2 = link_functions.pair_tracks_balltree(points1, points2, max_dist=1.0)
        res1, res2, ind1, ind2 = link_functions.postprocess_delaunay(points1, points2, ind1, ind2)

        exp_ind1 = np.arange(13)
        exp_ind2 = np.arange(13)

        np.testing.assert_almost_equal(ind1, exp_ind1)
        np.testing.assert_almost_equal(ind2, exp_ind2)

        np.testing.assert_almost_equal(res1, np.concatenate([points1, np.array([[0.0, 0.0]])]))
        np.testing.assert_almost_equal(res2, points2)


class TestPairTracksBipartiteMatch(unittest.TestCase):

    def test_pair_timepoints_large_small(self):

        ind1, ind2 = link_functions.pair_tracks_bipartite_match(T1, T2)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_small_large(self):

        ind1, ind2 = link_functions.pair_tracks_bipartite_match(T2, T3)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_same_length(self):

        ind1, ind2 = link_functions.pair_tracks_bipartite_match(T4, T5)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_same_length_low_distance(self):

        ind1, ind2 = link_functions.pair_tracks_bipartite_match(T4, T5, max_dist=1)

        exp_ind1 = np.array([0, 1, 3])
        exp_ind2 = np.array([0, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_same_length_partitioned(self):

        ind1, ind2 = link_functions.pair_tracks_bipartite_match(T4, T5, max_dist=0.1)

        exp_ind1 = np.array([])
        exp_ind2 = np.array([])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_identical(self):

        ind1, ind2 = link_functions.pair_tracks_bipartite_match(T2, T2)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)


class TestPairTracksBallTree(unittest.TestCase):

    def test_can_filter_by_max_distance(self):

        points1 = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
        ])
        points2 = np.array([
            [0, 0],
            [1, 1],
            [3, 3],
        ])
        ind1, ind2 = link_functions.pair_tracks_balltree(points1, points2, max_dist=20)

        exp_ind1 = np.array([0, 1, 2])
        exp_ind2 = np.array([0, 1, 2])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

        ind1, ind2 = link_functions.pair_tracks_balltree(points1, points2, max_dist=0.1)

        exp_ind1 = np.array([0, 1])
        exp_ind2 = np.array([0, 1])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_large_small(self):

        ind1, ind2 = link_functions.pair_tracks_balltree(T1, T2)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_small_large(self):

        ind1, ind2 = link_functions.pair_tracks_balltree(T2, T3)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_same_length(self):

        ind1, ind2 = link_functions.pair_tracks_balltree(T4, T5)

        exp_ind1 = np.array([0, 1, 3])
        exp_ind2 = np.array([0, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_identical(self):

        ind1, ind2 = link_functions.pair_tracks_balltree(T2, T2)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)


class TestPairTracksSoftAssign(unittest.TestCase):

    def test_pair_timepoints_large_small(self):

        ind1, ind2 = link_functions.pair_tracks_softassign(T1, T2)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_small_large(self):

        ind1, ind2 = link_functions.pair_tracks_softassign(T2, T3)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 3, 4])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_same_length(self):

        ind1, ind2 = link_functions.pair_tracks_softassign(T4, T5)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_identical(self):

        ind1, ind2 = link_functions.pair_tracks_softassign(T2, T2)

        exp_ind1 = np.array([0, 1, 2, 3])
        exp_ind2 = np.array([0, 1, 2, 3])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)

    def test_pair_timepoints_error(self):

        t1 = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]).T
        t2 = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]).T + 200

        ind1, ind2 = link_functions.pair_tracks_softassign(t1, t2)

        exp_ind1 = np.array([])
        exp_ind2 = np.array([])

        np.testing.assert_equal(ind1, exp_ind1)
        np.testing.assert_equal(ind2, exp_ind2)
