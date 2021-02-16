# Standard lib
import unittest

# 3rd party
import numpy as np

# Our own imports
from deep_hipsc_tracking import tracking
from .. import helpers

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


class TestFindFlatRegions(unittest.TestCase):

    def test_finds_flat_region_all_flat(self):

        tt = np.linspace(0, 100, 100)
        yy = tt * 2

        res = tracking.find_flat_regions(tt, yy, interp_points=10, cutoff=10, noise_points=5)
        exp = [np.ones((100, ), dtype=np.bool)]

        msg = 'Got {} rois expected {}'.format(len(res), len(exp))
        self.assertEqual(len(res), len(exp), msg)
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r, e)

    def test_finds_flat_region_all_spikey(self):

        tt = np.linspace(0, 100, 100)
        yy = np.array([-100, 0, 100] * 50)

        res = tracking.find_flat_regions(tt, yy, interp_points=5, cutoff=1, noise_points=1)
        exp = []

        msg = 'Got {} rois expected {}'.format(len(res), len(exp))
        self.assertEqual(len(res), len(exp), msg)
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r, e)

    def test_finds_flat_region_square_waves(self):

        tt = np.linspace(0, 100, 100)
        yy = np.array(([-100] * 10 + [100] * 10)*5)

        res = tracking.find_flat_regions(tt, yy, interp_points=5, cutoff=1, noise_points=1)
        exp = []
        for i in range(0, 100, 10):
            mask = np.zeros((100, ), dtype=np.bool)
            if i == 0:
                mask[i:i+8] = 1
            elif i == 90:
                mask[i+2:i+10] = 1
            else:
                mask[i+2:i+8] = 1
            exp.append(mask)

        msg = 'Got {} rois expected {}'.format(len(res), len(exp))
        self.assertEqual(len(res), len(exp), msg)
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r, e)


class TestRollingFuncs(unittest.TestCase):

    def test_rolling_rolling_window(self):

        xp = np.array([1, 2, 3, 4, 5])
        exp = np.array([2, 3, 4])

        res = np.mean(tracking.rolling_window(xp, window=3), axis=-1)

        np.testing.assert_almost_equal(res, exp)

        exp = np.array([1.5, 2.5, 3.5, 4.5])

        res = np.mean(tracking.rolling_window(xp, window=2), axis=-1)

        np.testing.assert_almost_equal(res, exp)

        exp = np.array([1.3333, 2, 3, 4, 4.6666])

        res = np.mean(tracking.rolling_window(xp, window=3, pad='same'), axis=-1)

        np.testing.assert_almost_equal(res, exp, decimal=3)

    def test_interpolate_window(self):

        xp = np.array([1, 2, 3, 4, 5])
        yp = np.array([5, 4, 3, 2, 1])

        x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5])
        y = np.array([5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1, 0.5])

        res = tracking.rolling_interp(x, xp, yp, 3)
        np.testing.assert_almost_equal(y, res)

    def test_slope_window(self):

        xp = np.array([1, 2, 3, 4, 5])
        yp = np.array([5, 4, 3, 2, 1])

        x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5])
        a = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        res = tracking.rolling_slope(x, xp, yp, 3)
        np.testing.assert_almost_equal(a, res)


class TestMergePointsCluster(unittest.TestCase):

    def test_merges_points_with_nans(self):

        points1 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
        ])
        points2 = np.array([
            [0.0, 0.1],
            [1.0, np.nan],
            [3.0, 3.1],
        ])

        points = tracking.tracking.merge_points_cluster(points1, points2, max_dist=0.1)

        exp_points = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        np.testing.assert_almost_equal(points, exp_points)

    def test_merges_same_set(self):

        points1 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
        ])

        points = tracking.tracking.merge_points_cluster(points1, points1, max_dist=0.1)

        np.testing.assert_almost_equal(points, points1)

    def test_merges_both_different(self):

        points1 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
        ])
        points2 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [3.0, 3.1],
        ])

        points = tracking.tracking.merge_points_cluster(points1, points2, max_dist=0.1)

        exp_points = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        np.testing.assert_almost_equal(points, exp_points)

    def test_merges_superset_left(self):

        points1 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        points2 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [3.0, 3.1],
        ])

        points = tracking.tracking.merge_points_cluster(points1, points2, max_dist=0.1)

        exp_points = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        np.testing.assert_almost_equal(points, exp_points)

        def test_merges_superset_right(self):

            points1 = np.array([
                [0.0, 0.1],
                [1.0, 1.1],
                [3.0, 3.1],
            ])
            points2 = np.array([
                [0.0, 0.1],
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
            ])

            points = tracking.tracking.merge_points_cluster(points1, points2, max_dist=0.1)

            exp_points = np.array([
                [0.0, 0.1],
                [1.0, 1.1],
                [3.0, 3.1],
                [2.0, 2.1],
            ])
            np.testing.assert_almost_equal(points, exp_points)

    def test_merges_superset_slight_motion(self):

        points1 = np.array([
            [0.0, 0.2],
            [1.0, 1.2],
            [2.0, 2.2],
            [3.0, 3.2],
        ])
        points2 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [3.0, 3.1],
            [4.0, 4.1],
        ])

        points = tracking.tracking.merge_points_cluster(points1, points2, max_dist=0.2)

        exp_points = np.array([
            [0.0, 0.15],
            [1.0, 1.15],
            [2.0, 2.2],
            [3.0, 3.15],
            [4.0, 4.1],
        ])
        np.testing.assert_almost_equal(points, exp_points)


class TestMergePointsPairwise(unittest.TestCase):

    def test_merges_same_set(self):

        points1 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
        ])

        points = tracking.tracking.merge_points_pairwise(points1, points1, max_dist=0.1)

        np.testing.assert_almost_equal(points, points1)

    def test_merges_both_different(self):

        points1 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
        ])
        points2 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [3.0, 3.1],
        ])

        points = tracking.tracking.merge_points_pairwise(points1, points2, max_dist=0.1)

        exp_points = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        np.testing.assert_almost_equal(points, exp_points)

    def test_merges_superset_left(self):

        points1 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [2.0, 2.1],
            [3.0, 3.1],
        ])
        points2 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [3.0, 3.1],
        ])

        points = tracking.tracking.merge_points_pairwise(points1, points2, max_dist=0.1)

        exp_points = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [3.0, 3.1],
            [2.0, 2.1],
        ])
        np.testing.assert_almost_equal(points, exp_points)

        def test_merges_superset_right(self):

            points1 = np.array([
                [0.0, 0.1],
                [1.0, 1.1],
                [3.0, 3.1],
            ])
            points2 = np.array([
                [0.0, 0.1],
                [1.0, 1.1],
                [2.0, 2.1],
                [3.0, 3.1],
            ])

            points = tracking.tracking.merge_points_pairwise(points1, points2, max_dist=0.1)

            exp_points = np.array([
                [0.0, 0.1],
                [1.0, 1.1],
                [3.0, 3.1],
                [2.0, 2.1],
            ])
            np.testing.assert_almost_equal(points, exp_points)

    def test_merges_superset_slight_motion(self):

        points1 = np.array([
            [0.0, 0.2],
            [1.0, 1.2],
            [2.0, 2.2],
            [3.0, 3.2],
        ])
        points2 = np.array([
            [0.0, 0.1],
            [1.0, 1.1],
            [3.0, 3.1],
            [4.0, 4.1],
        ])

        points = tracking.tracking.merge_points_pairwise(points1, points2, max_dist=0.2)

        exp_points = np.array([
            [0.0, 0.15],
            [1.0, 1.15],
            [3.0, 3.15],
            [2.0, 2.2],
            [4.0, 4.1],
        ])
        np.testing.assert_almost_equal(points, exp_points)


class TestFindLinkFunctions(unittest.TestCase):

    def test_finds_all_the_links(self):

        res = tracking.find_link_functions()
        exp = {'softassign', 'balltree', 'bipartite_match'}

        self.assertEqual(set(res.keys()), exp)


class TestLinks(unittest.TestCase):

    def test_to_padded_arrays(self):

        tt = np.array([3, 5, 7, 9, 11])
        xx = np.array([0, 1, 2, 3, 4])
        yy = np.array([1, 2, 3, 4, 5])

        chain = tracking.Link.from_arrays(tt, xx, yy)

        nan = np.nan

        in_tt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        in_xx = np.array([nan, nan, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, nan])
        in_yy = np.array([nan, nan, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, nan])

        res_tt, res_xx, res_yy = chain.to_padded_arrays(min_t=1, max_t=13)

        np.testing.assert_almost_equal(res_tt, in_tt)
        np.testing.assert_almost_equal(res_xx, in_xx)
        np.testing.assert_almost_equal(res_yy, in_yy)

        in_tt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        in_xx = np.array([0, 0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4])
        in_yy = np.array([1, 1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5])

        res_tt, res_xx, res_yy = chain.to_padded_arrays(min_t=1, max_t=13, extrapolate=True)

        np.testing.assert_almost_equal(res_tt, in_tt)
        np.testing.assert_almost_equal(res_xx, in_xx)
        np.testing.assert_almost_equal(res_yy, in_yy)

    def test_interpolate_chain_regular(self):

        tt = np.array([3, 5, 7, 9, 11])
        xx = np.array([0, 1, 2, 3, 4])
        yy = np.array([1, 2, 3, 4, 5])

        chain = tracking.Link.from_arrays(tt, xx, yy)

        self.assertEqual(len(chain), 5)

        chain.interpolate_points()

        self.assertEqual(len(chain), 9)

        tt = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
        xx = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
        yy = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

        exp_chain = tracking.Link()

        for t, x, y in zip(tt, xx, yy):
            exp_chain.add(-1, t, x, y)

        self.assertEqual(chain, exp_chain)

    def test_interpolate_chain_irregular(self):

        tt = np.array([3, 5, 9, 11])
        xx = np.array([0, 1, 3, 4])
        yy = np.array([1, 2, 4, 5])

        chain = tracking.Link.from_arrays(tt, xx, yy)

        self.assertEqual(len(chain), 4)

        chain.interpolate_points()

        self.assertEqual(len(chain), 9)

        tt = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
        xx = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
        yy = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])

        exp_chain = tracking.Link()

        for t, x, y in zip(tt, xx, yy):
            exp_chain.add(-1, t, x, y)

        self.assertEqual(chain, exp_chain)

    def test_chain_from_arrays(self):

        tt = np.array([1, 3, 5, 7, 9])
        xx = np.array([0, 1, 2, 3, 4])
        yy = np.array([1, 2, 3, 4, 5])

        chain = tracking.Link.from_arrays(tt, xx, yy)
        exp_chain = tracking.Link()

        for t, x, y in zip(tt, xx, yy):
            exp_chain.add(-1, t, x, y)

        self.assertEqual(chain, exp_chain)

    def test_to_arrays(self):

        tt = np.array([1, 3, 5, 7, 9])
        xx = np.array([0, 1, 2, 3, 4])
        yy = np.array([1, 2, 3, 4, 5])

        chain = tracking.Link.from_arrays(tt, xx, yy)

        rest, resx, resy = chain.to_arrays()

        np.testing.assert_almost_equal(rest, tt)
        np.testing.assert_almost_equal(resx, xx)
        np.testing.assert_almost_equal(resy, yy)

    def test_chain_from_tuples(self):

        chain = [(1, 0, 1),
                 (3, 1, 2),
                 (5, 2, 3),
                 (7, 3, 4),
                 (9, 4, 5)]

        chain = tracking.Link.from_tuples(chain)
        exp_chain = tracking.Link()

        for t, x, y in chain:
            exp_chain.add(-1, t, x, y)

        self.assertEqual(chain, exp_chain)

    def test_join_links(self):

        chain1 = [(1, 0, 1),
                  (3, 1, 2),
                  (5, 2, 3),
                  (7, 3, 4),
                  (9, 4, 5)]

        chain2 = [(13, 6, 7),
                  (15, 7, 8),
                  (17, 8, 9),
                  (19, 9, 10),
                  (21, 10, 11)]

        chain1 = tracking.Link.from_tuples(chain1)
        chain2 = tracking.Link.from_tuples(chain2)

        chain = tracking.Link.join(chain1, chain2, interp='linear')

        exp_t = np.arange(1, 23, 2)
        exp_x = np.arange(0, 11, 1)
        exp_y = np.arange(1, 12, 1)

        res_t, res_x, res_y = chain.to_arrays()

        np.testing.assert_almost_equal(res_t, exp_t)
        np.testing.assert_almost_equal(res_x, exp_x)
        np.testing.assert_almost_equal(res_y, exp_y)

        chain1 = [(1, 0, 1),
                  (3, 1, 2),
                  (5, 2, 3),
                  (7, 3, 4),
                  (9, 4, 5)]

        chain2 = [(15, 7, 8),
                  (17, 8, 9),
                  (19, 9, 10),
                  (21, 10, 11),
                  (23, 11, 12)]

        chain1 = tracking.Link.from_tuples(chain1)
        chain2 = tracking.Link.from_tuples(chain2)

        chain = tracking.Link.join(chain1, chain2, interp='linear')

        exp_t = np.arange(1, 24, 2)
        exp_x = np.arange(0, 12, 1)
        exp_y = np.arange(1, 13, 1)

        res_t, res_x, res_y = chain.to_arrays()

        np.testing.assert_almost_equal(res_t, exp_t)
        np.testing.assert_almost_equal(res_x, exp_x)
        np.testing.assert_almost_equal(res_y, exp_y)


class TestLinkChains(unittest.TestCase):

    def test_links_chains_simple(self):

        tp1 = 3
        tp2 = 6

        t1 = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
        ]).T
        t2 = np.array([
            [11.0, 12.0, 13.0],
            [11.1, 12.1, 13.1]
        ]).T

        ind1 = np.array([0, 2, 1])
        ind2 = np.array([0, 2, 1])

        chains = [[(tp1, x, y)] for x, y in t1]
        index = np.arange(t1.shape[0], dtype=np.uint32)

        index, chains = tracking.link_chains(tp2, t2, ind1, ind2, index, chains)

        exp_index = np.array([0, 1, 2])
        exp_chains = [
           [(3, 1.0, 1.1), (6, 11.0, 11.1)],
           [(3, 2.0, 2.1), (6, 12.0, 12.1)],
           [(3, 3.0, 3.1), (6, 13.0, 13.1)],
        ]

        np.testing.assert_almost_equal(index, exp_index)
        self.assertEqual(chains, exp_chains)

    def test_links_chains_empty(self):

        tp1 = 3
        tp2 = 6

        t1 = np.array([
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
        ]).T
        t2 = np.array([
            [11.0, 12.0, 13.0],
            [11.1, 12.1, 13.1]
        ]).T

        ind1 = np.array([])
        ind2 = np.array([])

        chains = [[(tp1, x, y)] for x, y in t1]
        index = np.arange(t1.shape[0], dtype=np.uint32)

        index, chains = tracking.link_chains(tp2, t2, ind1, ind2, index, chains)

        exp_index = np.array([3, 4, 5])
        exp_chains = [
           [(3, 1.0, 1.1)],
           [(3, 2.0, 2.1)],
           [(3, 3.0, 3.1)],
           [(6, 11.0, 11.1)],
           [(6, 12.0, 12.1)],
           [(6, 13.0, 13.1)],
        ]

        np.testing.assert_almost_equal(index, exp_index)
        self.assertEqual(chains, exp_chains)

    def test_links_chains_empty_index(self):

        tp1 = 3
        tp2 = 6

        t1 = np.array([])
        t2 = np.array([
            [11.0, 12.0, 13.0],
            [11.1, 12.1, 13.1]
        ]).T

        ind1 = np.array([])
        ind2 = np.array([])

        chains = [[(tp1, x, y)] for x, y in t1]
        index = np.arange(t1.shape[0], dtype=np.uint32)

        index, chains = tracking.link_chains(tp2, t2, ind1, ind2, index, chains)

        exp_index = np.array([0, 1, 2])
        exp_chains = [
           [(6, 11.0, 11.1)],
           [(6, 12.0, 12.1)],
           [(6, 13.0, 13.1)],
        ]

        np.testing.assert_almost_equal(index, exp_index)
        self.assertEqual(chains, exp_chains)


class TestLinkAllChains(unittest.TestCase):

    def test_links_default_tracks(self):

        chains = tracking.link_all_chains(TRACKS, merge_fxn='pairwise', processes=1)
        chains = list(sorted(chains, key=lambda c: c.line_x[0]))

        chain_coords = [
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [2.0, 2.1, 2.2],
            [2.4],
            [3.0, 3.1, 3.2, 3.3, 3.4],
            [4.0, 4.1, 4.2, 4.3],
            [5.0],
            [5.2, 5.3, 5.4],
        ]

        self.assertEqual(len(chains), len(chain_coords))

        for coords, chain in zip(chain_coords, chains):
            self.assertEqual(coords, chain.line_x)
            self.assertEqual(coords, chain.line_y)

    def test_links_default_tracks_cluster(self):

        chains = tracking.link_all_chains(TRACKS,
                                          merge_fxn='cluster',
                                          max_merge_dist=4.0,
                                          impute_steps=3,
                                          processes=1)
        chains = list(sorted(chains, key=lambda c: c.line_x[0]))

        chain_coords = [
            [1.0],
            [1.4],
            [2.0],
            [2.4],
            [3.0, 3.1, 4.2, 3.55, 3.4],
            [4.0],
            [5.0],
            [5.4],
        ]
        self.assertEqual(len(chains), len(chain_coords))

        for coords, chain in zip(chain_coords, chains):
            fmt = 'Mismatched coordinates\n expected {}\n got {}\n\n'
            msg = fmt.format(coords, chain.line_x)
            self.assertEqual(len(coords), len(chain.line_x), msg=msg)
            self.assertTrue(all([round(c, 2) == round(x, 2)
                                 for c, x in zip(coords, chain.line_x)]), msg=msg)

            fmt = 'Mismatched coordinates\n expected {}\n got {}\n\n'
            msg = fmt.format(coords, chain.line_y)
            self.assertEqual(len(coords), len(chain.line_y), msg=msg)
            self.assertTrue(all([round(c, 2) == round(y, 2)
                                 for c, y in zip(coords, chain.line_y)]), msg=msg)

    def test_links_tracks_different_step(self):

        chains = tracking.link_all_chains(TRACKS, link_step=2, merge_fxn='pairwise', processes=1)
        chains = list(sorted(chains, key=lambda c: c.line_x[0]))

        chain_coords = [
            [1.0, 1.2, 1.4],
            [1.1, 1.3],
            [2.0, 2.2, 2.4],
            [2.1],
            [3.0, 3.2, 3.4],
            [3.1, 3.3],
            [4.0, 4.2],
            [4.1, 4.3],
            [5.0, 5.2, 5.4],
            [5.3],
        ]
        self.assertEqual(len(chains), len(chain_coords))

        for coords, chain in zip(chain_coords, chains):
            self.assertEqual(coords, chain.line_x)
            self.assertEqual(coords, chain.line_y)

    def test_gets_sane_results_for_velocity(self):

        chains = tracking.link_all_chains(TRACKS, merge_fxn='pairwise', processes=1)
        chains = list(sorted(chains, key=lambda c: c.line_x[0]))

        chain_vels = [
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.1, 0.1]),
            np.array([0.0]),
            np.array([0.1, 0.1, 0.1, 0.1]),
            np.array([0.1, 0.1, 0.1]),
            np.array([0.0]),
            np.array([0.1, 0.1]),
        ]

        self.assertEqual(len(chains), len(chain_vels))

        for vel, chain in zip(chain_vels, chains):
            np.testing.assert_almost_equal(vel, chain.vel_x())
            np.testing.assert_almost_equal(vel, chain.vel_y())
            np.testing.assert_almost_equal(vel*np.sqrt(2), chain.vel_mag())


class TestTrackSaveLoad(helpers.FileSystemTestCase):

    def test_can_track_save_load(self):

        trackfile = self.tempdir / 'test.csv'

        chains = tracking.link_all_chains(TRACKS, merge_fxn='pairwise', processes=1)
        tracking.save_track_csvfile(trackfile, chains)
        load_chains = tracking.load_track_csvfile(trackfile)

        self.assertEqual(len(load_chains), len(chains))
        for lc, c in zip(load_chains, chains):
            self.assertEqual(lc, c)

    def test_can_load_min_len(self):

        trackfile = self.tempdir / 'test.csv'

        chains = tracking.link_all_chains(TRACKS, merge_fxn='pairwise', processes=1)
        tracking.save_track_csvfile(trackfile, chains)
        load_chains = tracking.load_track_csvfile(trackfile, min_len=3)

        exp_chains = [
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2, 3.3, 3.4],
            [4.0, 4.1, 4.2, 4.3],
            [5.2, 5.3, 5.4],
        ]

        self.assertEqual(len(load_chains), len(exp_chains))
        for lc, exp_coords in zip(load_chains, exp_chains):
            self.assertEqual(lc.line_x, exp_coords)
            self.assertEqual(lc.line_y, exp_coords)

        load_chains = tracking.load_track_csvfile(trackfile, min_len=4)

        exp_chains = [
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [3.0, 3.1, 3.2, 3.3, 3.4],
            [4.0, 4.1, 4.2, 4.3],
        ]

        self.assertEqual(len(load_chains), len(exp_chains))
        for lc, exp_coords in zip(load_chains, exp_chains):
            self.assertEqual(lc.line_x, exp_coords)
            self.assertEqual(lc.line_y, exp_coords)
