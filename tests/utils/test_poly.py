#!/usr/bin/env python3

# Stdlib
import unittest

# 3rd party
import numpy as np

# Our own imports
from deep_hipsc_tracking.utils import poly, _poly

# Tests


class TestGridValueExtractor(unittest.TestCase):

    def test_extracts_equivalent_mask_left_side(self):

        x = np.arange(64)
        y = np.arange(128)

        extractor = poly.GridValueExtractor(y, x)

        coords = np.array([
            [0.0, 0.0],
            [10, 0.0],
            [10, 10],
            [20, 10],
            [20, 20],
            [0.0, 20.0],
        ])
        mask1 = poly.mask_in_polygon(coords, y, x)
        mask2 = extractor.extract_mask(coords)

        self.assertTrue(np.any(mask1))
        self.assertTrue(np.any(mask2))
        self.assertTrue(np.all(mask1 == mask2))

    def test_extracts_equivalent_mask_right_side(self):

        x = np.arange(64)
        y = np.arange(128)

        extractor = poly.GridValueExtractor(y, x)

        coords = np.array([
            [70.0, 0.0],
            [80.0, 0.0],
            [80.0, 10.0],
            [90.0, 10.0],
            [90.0, 20.0],
            [70.0, 20.0],
        ])
        mask1 = poly.mask_in_polygon(coords, y, x)
        mask2 = extractor.extract_mask(coords)

        self.assertTrue(np.any(mask1))
        self.assertTrue(np.any(mask2))
        self.assertTrue(np.all(mask1 == mask2))

    def test_equivalent_but_faster(self):

        img1 = np.random.rand(64, 128)
        img2 = np.random.rand(64, 128)

        x = np.arange(64)
        y = np.arange(128)

        extractor = poly.GridValueExtractor(y, x)
        extractor.add_image(img1)
        extractor.add_image(img2)

        coords = np.array([
            [0.0, 0.0],
            [10, 0.0],
            [10, 10],
            [20, 10],
            [20, 20],
            [0.0, 20.0],
        ])
        mask = poly.mask_in_polygon(coords, y, x)

        exp1 = np.mean(img1[mask])
        exp2 = np.mean(img2[mask])

        res1, res2 = extractor.extract_values(coords)

        self.assertAlmostEqual(res1, exp1, places=4)
        self.assertAlmostEqual(res2, exp2, places=4)

        coords = np.array([
            [70.0, 0.0],
            [80.0, 0.0],
            [80.0, 10.0],
            [90.0, 10.0],
            [90.0, 20.0],
            [70.0, 20.0],
        ])
        mask = poly.mask_in_polygon(coords, y, x)

        exp1 = np.mean(img1[mask])
        exp2 = np.mean(img2[mask])

        res1, res2 = extractor.extract_values(coords)

        self.assertAlmostEqual(res1, exp1, places=4)
        self.assertAlmostEqual(res2, exp2, places=4)

    def test_extract_square_from_grid(self):

        img1 = np.zeros((64, 128))
        img1[24:48, 65:100] = 1

        img2 = np.zeros((64, 128))
        img2[24:48, 65:100] = 2

        extractor = poly.GridValueExtractor(np.arange(128), np.arange(64))
        extractor.add_image(img1)
        extractor.add_image(img2)

        coords = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [0.0, 5.0],
        ])
        values = extractor.extract_values(coords)
        np.testing.assert_almost_equal(values, [0.0, 0.0])

        coords = np.array([
            [65.1, 24.1],
            [65.1, 46.9],
            [99.9, 46.9],
            [99.9, 24.1],
        ])
        values = extractor.extract_values(coords)
        np.testing.assert_almost_equal(values, [1.0, 2.0])

    def test_extracts_mean_value(self):

        img1 = np.zeros((64, 128))
        img1[24:48, 65:100] = 1
        img1[36:48, 65:100] = 2

        extractor = poly.GridValueExtractor(np.arange(128), np.arange(64))
        extractor.add_image(img1)

        coords = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [0.0, 5.0],
        ])
        values = extractor.extract_values(coords)
        np.testing.assert_almost_equal(values, [0.0])

        coords = np.array([
            [65.1, 24.1],
            [65.1, 46.9],
            [99.9, 46.9],
            [99.9, 24.1],
        ])
        values = extractor.extract_values(coords)
        np.testing.assert_almost_equal(values, [1.5])

    def test_can_extract_different_values(self):

        img1 = np.zeros((64, 128))
        img1[24:48, 65:100] = 1
        img1[36:48, 65:100] = 2

        extractor = poly.GridValueExtractor(np.arange(128), np.arange(64))
        extractor.add_image(img1)

        coords = np.array([
            [65.1, 24.1],
            [65.1, 46.9],
            [99.9, 46.9],
            [99.9, 24.1],
        ])
        values = extractor.extract_values(coords, func=np.min)
        np.testing.assert_almost_equal(values, [1.0])

        values = extractor.extract_values(coords, func=np.max)
        np.testing.assert_almost_equal(values, [2.0])

    def test_extract_outside_image_default_value(self):

        img1 = np.zeros((64, 128))
        img1[24:48, 65:100] = 1
        img1[36:48, 65:100] = 2

        extractor = poly.GridValueExtractor(np.arange(128), np.arange(64))
        extractor.add_image(img1)

        coords = np.array([
            [-2, 0],
            [-1, 0],
            [-1, -1],
            [-2, -1],
        ])
        values = extractor.extract_values(coords)
        np.testing.assert_almost_equal(values, [np.nan])

        values = extractor.extract_values(coords, fill_value=-1)
        np.testing.assert_almost_equal(values, [-1])

    def test_segment_and_extract_values(self):

        img1 = np.zeros((64, 128))
        img1[32, 72] = 50
        img1[48, 12] = 100
        img1[24, 32] = 25
        img1[32, 24] = 75

        extractor = poly.GridValueExtractor(np.arange(128), np.arange(64))
        extractor.add_image(img1)

        coords = extractor.segment_peaks(img1)

        exp_coords = np.array([
            [12, 48],
            [24, 32],
            [72, 32],
            [32, 24],
        ])
        np.testing.assert_almost_equal(coords, exp_coords)

        values = extractor.extract_points(coords)[0]
        np.testing.assert_almost_equal(values, [100, 75, 50, 25])

    def test_segment_and_extract_masks(self):

        img1 = np.zeros((64, 128))
        img1[32, 72] = 50
        img1[48, 12] = 100
        img1[24, 32] = 25
        img1[32, 24] = 75

        extractor = poly.GridValueExtractor(np.arange(128), np.arange(64))
        extractor.add_image(img1)

        coords = extractor.segment_circles(img1, radius=5)

        # Make sure the circles are centered where they should be
        exp_centers = [
            np.array([12, 48]),
            np.array([24, 32]),
            np.array([72, 32]),
            np.array([32, 24]),
        ]
        exp_values = [100, 75, 50, 25]

        self.assertEqual(len(coords), len(exp_centers))

        for coord, exp_center in zip(coords, exp_centers):
            self.assertEqual(coord.shape, (50, 2))

            res_center = np.mean(coord, axis=0)
            np.testing.assert_almost_equal(res_center, exp_center)

        # Make sure extracting the circles gives the peak value
        self.assertEqual(len(coords), len(exp_values))
        for coord, exp_value in zip(coords, exp_values):
            max_value = extractor.extract_values(coord, func=np.max)[0]

            self.assertEqual(max_value, exp_value)


class TestArcLength(unittest.TestCase):

    def test_calc_arc_length(self):

        coords = np.array([
            [0, 0],
            [1, 0],
            [1, 2],
            [0, 2],
            [0, 0],
        ])
        length = poly.arc_length(coords)
        np.testing.assert_allclose(length, 6.0)

    def test_calc_arc_distance_along(self):

        coords = np.array([
            [0, 0],
            [1, 0],
            [1, 2],
            [0, 2],
            [0, 0],
        ])
        dist_along = poly.arc_distance_along(coords)

        exp_dist = np.array([
            0.0, 1.0, 3.0, 4.0, 6.0,
        ])
        np.testing.assert_allclose(dist_along, exp_dist)

    def test_lengths_and_scales_work_on_circles(self):

        t = np.linspace(0, 2*np.pi, 201)
        x = np.cos(t)
        y = np.sin(t)

        coords = np.stack([x, y], axis=1)

        exp_length = 2*np.pi
        length = poly.arc_length(coords)

        np.testing.assert_allclose(length, exp_length, atol=1e-3)

        dist_along = poly.arc_distance_along(coords)

        np.testing.assert_allclose(dist_along[0], 0.0, atol=1e-3)
        np.testing.assert_allclose(dist_along[50], 1/2*np.pi, atol=1e-3)
        np.testing.assert_allclose(dist_along[100], np.pi, atol=1e-3)
        np.testing.assert_allclose(dist_along[150], 3/2*np.pi, atol=1e-3)
        np.testing.assert_allclose(dist_along[200], 2*np.pi, atol=1e-3)

    def test_can_get_point_along_arc(self):

        coords = np.array([
            [0.0, 1.0],
            [4.0, 2.0],
        ])

        cx, cy = poly.arc_coords_frac_along(coords, 0.0)

        np.testing.assert_allclose((cx, cy), (0.0, 1.0))

        cx, cy = poly.arc_coords_frac_along(coords, 1.0)

        np.testing.assert_allclose((cx, cy), (4.0, 2.0))

        cx, cy = poly.arc_coords_frac_along(coords, 0.5)

        np.testing.assert_allclose((cx, cy), (2.0, 1.5))

        coords = np.array([
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [4.0, 2.0],
        ])

        cx, cy = poly.arc_coords_frac_along(coords, 0.5)

        np.testing.assert_allclose((cx, cy), (1.79283, 2.0), atol=1e-3)

        # Make an ellipse and pick points around the clock
        t = np.linspace(0, 2*np.pi, 201)
        x = np.cos(t)
        y = 2.0*np.sin(t)

        coords = np.stack([x, y], axis=1)

        cx, cy = poly.arc_coords_frac_along(coords, 0.0)

        np.testing.assert_allclose((cx, cy), (1.0, 0.0), atol=1e-3)

        cx, cy = poly.arc_coords_frac_along(coords, 0.25)

        np.testing.assert_allclose((cx, cy), (0.0, 2.0), atol=1e-3)

        cx, cy = poly.arc_coords_frac_along(coords, 0.5)

        np.testing.assert_allclose((cx, cy), (-1.0, 0.0), atol=1e-3)

        cx, cy = poly.arc_coords_frac_along(coords, 0.75)

        np.testing.assert_allclose((cx, cy), (0.0, -2.0), atol=1e-3)


class TestPolyhedronStats(unittest.TestCase):

    def test_vertices_of_polyhedron(self):

        # Works in 2D
        points = np.array([
            [-1, -1],
            [-1, 1],
            [1, 1],
            [1, -1],
            [0, 0],
            [0.5, 0.5],
        ])
        res = poly.vertices_of_polyhedron(points)

        # Come back in clockwise order
        exp = np.array([
            [-1, 1],
            [-1, -1],
            [1, -1],
            [1, 1],
        ])
        np.testing.assert_almost_equal(res, exp)

        # Works in 3D
        points = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, -1, 0],
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ])
        res = poly.vertices_of_polyhedron(points)

        # Comes back in input order
        exp = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ])
        np.testing.assert_almost_equal(res, exp)

    def test_calc_3d_centroid(self):

        # 3 x 3 x 2 cube with oversampled corner
        points = np.array([
            [0, 0, 0],
            [-1, -2, -1],
            [-1, -2, -0.9],
            [-1, -2, -0.8],
            [-1, -2, -0.7],
            [-1, -2, -0.6],
            [2, -2, -1],
            [-1, 1, -1],
            [-1, -2, 1],
            [-1, 1, 1],
            [2, -2, 1],
            [2, 1, -1],
            [2, 1, 1],
        ])
        center = poly.centroid_of_polyhedron(points)

        exp = np.array([0.5, -0.5, 0])

        np.testing.assert_almost_equal(center, exp)

    def test_calc_3d_volume(self):

        # 3 x 2 x 2 cube
        points = np.array([
            [0, 0, 0],
            [-1, -1, -1],
            [2, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [2, -1, 1],
            [2, 1, -1],
            [2, 1, 1],
        ])
        volume = poly.volume_of_polyhedron(points)

        self.assertEqual(volume, 12.0)


class TestSortCoordinates(unittest.TestCase):

    def test_sorting_works(self):

        coords = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 2.0],
            [5.0, 1.0],
        ])

        res = poly.sort_coordinates(coords)
        exp = np.array([
            [1.0, 1.0],
            [2.0, 1.0],
            [5.0, 1.0],
            [4.0, 2.0],
            [3.0, 3.0],
            [2.0, 2.0],
        ])
        np.testing.assert_almost_equal(res, exp)


class TestAlphaShape(unittest.TestCase):

    def test_finds_hull_of_square(self):

        points = np.array([
            (0, 0),
            (0, 1),
            (3, 1),
            (3, 0),
        ])

        res = poly.alpha_shape(points, alpha=5)
        exp = np.array([
            [3., 0.],
            [0., 0.],
            [3., 1.],
            [0., 1.],
        ])
        np.testing.assert_almost_equal(res, exp)

    def test_finds_hull_of_square_with_internal_points(self):

        points = np.array([
            (0, 0),
            (1, 1),
            (0.5, 0.5),
            (0, 1),
            (3, 1),
            (2, 0.5),
            (3, 0),
        ])

        res = poly.alpha_shape(points, alpha=5)
        exp = np.array([
            (0, 0),
            (3, 0),
            (3, 1),
            (1, 1),
            (0, 1),
        ])

        np.testing.assert_almost_equal(res, exp)


class TestPerimeterOfPoly(unittest.TestCase):

    def test_finds_points(self):

        polygon = np.array([
            (0, 0),
            (0, 1),
            (3, 1),
            (3, 0),
        ])

        perimeter = poly.perimeter_of_polygon(polygon)

        exp = 8
        self.assertEqual(perimeter, exp)

    def test_close_for_circle(self):

        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)

        perimeter = poly.perimeter_of_polygon(np.stack([x, y], axis=1))
        exp = 2*np.pi  # 2 * pi * R

        np.testing.assert_almost_equal(perimeter, exp, decimal=3)


class TestPointsInPolygon(unittest.TestCase):

    def test_finds_points(self):

        polygon = np.array([
            (-4, -2),
            (-4, 4),
            (-3, 4),
            (-2, 4),
            (-1, 4),
            (1, 4),
            (2, 4),
            (2, -2),
        ])

        points = np.array([
            (-4, -2),  # Corner
            (-4, 0),  # Edge
            (0, 0),  # Inside
            (-5, -2),  # Outside but edge
            (-5, -3),  # Way outside
        ])

        res = poly.points_in_polygon(polygon, points)
        exp = np.array([True, True, True, False, False], dtype=np.bool)

        np.testing.assert_almost_equal(res, exp)


class TestCalcPerimetersFromEdges(unittest.TestCase):

    def test_perimeter_no_edges(self):

        sedge_count = {}

        perimeters = poly.calc_perimeters_from_edges(sedge_count)

        self.assertEqual(perimeters, [])

    def test_perimeter_one_edge(self):

        sedge_count = {
            (0, 1): 1,
        }

        perimeters = poly.calc_perimeters_from_edges(sedge_count)

        self.assertEqual(perimeters, [])

    def test_perimeter_two_edges(self):

        sedge_count = {
            (0, 1): 1,
            (1, 2): 1,
        }

        perimeters = poly.calc_perimeters_from_edges(sedge_count)

        self.assertEqual(perimeters, [])

    def test_perimeter_three_edges_disconnected(self):

        sedge_count = {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
        }

        perimeters = poly.calc_perimeters_from_edges(sedge_count)

        self.assertEqual(perimeters, [])

    def test_perimeter_three_edges_connected(self):

        sedge_count = {
            (0, 1): 1,
            (1, 2): 1,
            (0, 2): 1,
        }

        perimeters = poly.calc_perimeters_from_edges(sedge_count)
        exp_perimeters = {(0, 1, 2)}

        self.assertEqual(len(perimeters), len(exp_perimeters))
        for perimeter in perimeters:
            self.assertIn(tuple(sorted(perimeter)), exp_perimeters)

    def test_perimeter_four_edges_connected(self):

        sedge_count = {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
            (0, 3): 1,
        }

        perimeters = poly.calc_perimeters_from_edges(sedge_count)
        exp_perimeters = {(0, 1, 2, 3)}

        self.assertEqual(len(perimeters), len(exp_perimeters))
        for perimeter in perimeters:
            self.assertIn(tuple(sorted(perimeter)), exp_perimeters)

    def test_perimeter_four_edges_pokey(self):

        sedge_count = {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
            (2, 0): 1,
        }

        perimeters = poly.calc_perimeters_from_edges(sedge_count)
        exp_perimeters = {(0, 1, 2)}

        self.assertEqual(len(perimeters), len(exp_perimeters))
        for perimeter in perimeters:
            self.assertIn(tuple(sorted(perimeter)), exp_perimeters)

    def test_perimeter_very_spikey(self):

        sedge_count = {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
            (2, 0): 1,
            (1, 5): 1,
            (0, 6): 1,
            (6, 7): 1,
            (7, 8): 1,
        }

        perimeters = poly.calc_perimeters_from_edges(sedge_count)
        exp_perimeters = {(0, 1, 2)}

        self.assertEqual(len(perimeters), len(exp_perimeters))
        for perimeter in perimeters:
            self.assertIn(tuple(sorted(perimeter)), exp_perimeters)


class TestEdgeList(unittest.TestCase):

    def test_remove_links(self):

        edges = _poly.EdgeList()
        edges.add(0, 1)
        edges.add(0, 2)
        edges.add(0, 1)

        self.assertEqual(edges[0], {1, 2})
        self.assertEqual(edges[1], {0})
        self.assertEqual(edges[2], {0})

        del edges[0]

        self.assertEqual(len(edges), 0)
        self.assertNotIn(0, edges)
        self.assertNotIn(1, edges)
        self.assertNotIn(2, edges)


class TestWarpToCircle(unittest.TestCase):

    def test_project_circle_onto_circle(self):

        theta = np.arange(0, 2*np.pi, np.pi/100)
        perimeter = np.stack([2*np.cos(theta), 2*np.sin(theta)], axis=1)

        coords = np.random.rand(100, 2) * 4 - 2
        coords = coords[(coords[:, 0]**2 + coords[:, 1]**2) < 4]

        warp_coords = poly.warp_to_circle(coords, perimeter)

        np.testing.assert_almost_equal(warp_coords, coords/2)

    def test_project_square_onto_circle(self):

        # Make a square, with enough support points to get a decent approximation
        perimeter = np.array([
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
        ])
        assert perimeter.shape[1] == 2

        # Make an X inside the square
        coords = np.array([
            [-1, -1],
            [-0.5, -0.5],
            [0, 0],
            [0.5, 0.5],
            [1, 1],
            [-0.5, 0.5],
            [-1, 1],
            [0.5, -0.5],
            [1, -1],
        ])
        warp_coords = poly.warp_to_circle(coords, perimeter, i_max=200)

        # Projects to a smaller X inside the circle
        exp_coords = np.array([
            [-0.70710678, -0.70710678],
            [-0.35355339, -0.35355339],
            [0.0, 0.0],
            [0.35355339, 0.35355339],
            [0.70710678, 0.70710678],
            [-0.35355339, 0.35355339],
            [-0.70710678, 0.70710678],
            [0.35355339, -0.35355339],
            [0.70710678, -0.70710678],
        ])
        np.testing.assert_almost_equal(warp_coords, exp_coords)

    def test_project_square_onto_circle_oversampled(self):

        # Make a square, with enough support points to get a decent approximation
        perimeter = np.array([
            [-1, -1],
            [-1, -0.9],
            [-1, -0.8],
            [-1, -0.7],
            [-1, -0.5],
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
        ])
        assert perimeter.shape[1] == 2

        # Make an X inside the square
        coords = np.array([
            [-1, -1],
            [-0.5, -0.5],
            [0, 0],
            [0.5, 0.5],
            [1, 1],
            [-0.5, 0.5],
            [-1, 1],
            [0.5, -0.5],
            [1, -1],
        ])
        warp_coords = poly.warp_to_circle(coords, perimeter, i_max=200)

        # Projects to a smaller X inside the circle
        exp_coords = np.array([
            [-0.70710678, -0.70710678],
            [-0.35355339, -0.35355339],
            [0.0, 0.0],
            [0.35355339, 0.35355339],
            [0.70710678, 0.70710678],
            [-0.35355339, 0.35355339],
            [-0.70710678, 0.70710678],
            [0.35355339, -0.35355339],
            [0.70710678, -0.70710678],
        ])
        np.testing.assert_almost_equal(warp_coords, exp_coords)

    def test_warp_inv_warp_oversampled_square(self):

        # Make a square, with enough support points to get a decent approximation
        perimeter = np.array([
            [-1, -1],
            [-1, -0.9],
            [-1, -0.8],
            [-1, -0.7],
            [-1, -0.5],
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
        ])
        assert perimeter.shape[1] == 2

        # Make an X inside the square
        coords = np.array([
            [-1, -1],
            [-0.5, -0.5],
            [0, 0],
            [0.5, 0.5],
            [1, 1],
            [-0.5, 0.5],
            [-1, 1],
            [0.5, -0.5],
            [1, -1],
        ])
        warp_coords = poly.warp_to_circle(coords, perimeter, i_max=200)
        inv_warp_coords = poly.inv_warp_to_circle(warp_coords, perimeter, i_max=200)

        np.testing.assert_almost_equal(coords, inv_warp_coords)

    def test_can_warp_and_inv_warp_beyond_the_edge(self):

        # Make a circle with radius 5
        theta = np.linspace(0, 2*np.pi, 50)
        perimeter = np.stack([np.cos(theta) * 5 + 2.0, np.sin(theta)*5 - 1.0], axis=1)

        # Ray along x == -y
        coords = np.array([
            [0, 0],
            [1, -1],
            [2, -2],
            [3, -3],
            [4, -4],
            [5, -5],
            [6, -6],
            [7, -7],
            [8, -8],
            [9, -9],
            [10, -10],
        ]) + np.array([
            [2.0, -1.0],
        ])

        warp_coords = poly.warp_to_circle(coords, perimeter, i_max=200, r_max=-1)

        # Warping in this case is just centering and scaling by 5
        exp_warp_coords = (coords - np.array([[2.0, -1.0]])) / 5

        np.testing.assert_almost_equal(warp_coords, exp_warp_coords, decimal=4)

        inv_warp_coords = poly.inv_warp_to_circle(warp_coords, perimeter, i_max=200)

        np.testing.assert_almost_equal(coords, inv_warp_coords, decimal=4)

    def test_maps_the_perimeter_to_itself(self):

        # Make a rectangle, with enough support points to get a decent approximation
        perimeter = np.array([
            [-2, -1],
            [-2, -0.9],
            [-2, -0.8],
            [-2, -0.7],
            [-2, -0.5],
            [-2, 0],
            [-2, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
        ]) + np.array([[-10, 5]])
        warp_coords = poly.warp_to_circle(perimeter, perimeter, i_max=200, r_max=-1)

        # Warp perimeter should always be (almost) on the unit circle
        warp_r = np.sqrt(warp_coords[:, 0]**2 + warp_coords[:, 1]**2)
        np.testing.assert_almost_equal(warp_r, np.ones_like(warp_r), decimal=2)

        # Inverse should project exactly back to the perimeter
        inv_warp_coords = poly.inv_warp_to_circle(warp_coords, perimeter, i_max=200)

        np.testing.assert_almost_equal(inv_warp_coords, perimeter, decimal=4)


class TestCalcDelaunayAdjacency(unittest.TestCase):

    def test_handles_square(self):

        points = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [0, 2],
            [1, 2],
            [2, 2],
        ])

        res_pairs, res_tris, res_perimeters = poly.calc_delaunay_adjacency(points, max_distance=1.1)
        exp_pairs = {
            0: {1, 3},
            1: {0, 2, 4},
            2: {1, 5},
            3: {0, 4, 6},
            4: {1, 3, 5, 7},
            5: {8, 2, 4},
            6: {3, 7},
            7: {8, 4, 6},
            8: {5, 7},
        }
        exp_tris = set()
        #exp_perimeters = {(0, 1, 2, 5, 8, 7, 6, 3)}
        exp_perimeters = set()  # FIXME: This is wrong, but we need graph tools to fix it

        self.assertEqual(res_pairs, exp_pairs)
        self.assertEqual(res_tris, exp_tris)
        self.assertEqual(len(res_perimeters), len(exp_perimeters))
        for res in res_perimeters:
            self.assertIn(tuple(sorted(res)), exp_perimeters)

    def test_handles_hex(self):

        points = np.array([
            [0, 2],
            [0.5, 0.5],
            [-0.5, 0.5],
            [0, 0],
            [-0.5, -0.5],
            [0.5, -0.5],
            [0, -2],
        ])
        res_pairs, res_tris, res_perimeters = poly.calc_delaunay_adjacency(points, max_distance=None)
        exp_pairs = {
            0: {1, 2},
            1: {0, 2, 3, 5},
            2: {0, 1, 3, 4},
            3: {1, 2, 4, 5},
            4: {2, 3, 5, 6},
            5: {1, 3, 4, 6},
            6: {4, 5},
        }
        exp_tris = {
            (0, 1, 2),
            (1, 2, 3),
            (1, 3, 5),
            (2, 3, 4),
            (3, 4, 5),
            (2, 3, 4),
            (4, 5, 6),
        }
        exp_perimeters = {tuple(sorted([0, 1, 5, 6, 4, 2]))}
        self.assertEqual(res_pairs, exp_pairs)
        self.assertEqual(res_tris, exp_tris)
        self.assertEqual(len(res_perimeters), len(exp_perimeters))
        for res in res_perimeters:
            self.assertIn(tuple(sorted(res)), exp_perimeters)

    def test_handles_hex_with_distance(self):

        points = np.array([
            [0, 2],
            [0.5, 0.5],
            [-0.5, 0.5],
            [0, 0],
            [-0.5, -0.5],
            [0.5, -0.5],
            [0, -2],
        ])
        res_pairs, res_tris, res_perimeters = poly.calc_delaunay_adjacency(points, max_distance=1.0)
        exp_pairs = {
            1: {2, 3, 5},
            2: {1, 3, 4},
            3: {1, 2, 4, 5},
            4: {2, 3, 5},
            5: {1, 3, 4},
        }
        exp_tris = {
            (1, 2, 3),
            (1, 3, 5),
            (3, 4, 5),
            (2, 3, 4),
        }
        exp_perimeters = {tuple(sorted([2, 4, 5, 1]))}
        self.assertEqual(res_pairs, exp_pairs)
        self.assertEqual(res_tris, exp_tris)
        self.assertEqual(len(res_perimeters), len(exp_perimeters))
        for res in res_perimeters:
            self.assertIn(tuple(sorted(res)), exp_perimeters)

    def test_handles_two_clusters(self):

        points = np.array([
            [0, 2],
            [0.5, 0.5],
            [-0.5, 0.5],
            [0, 0],
            [-0.5, -0.5],
            [0.5, -0.5],
            [0, -2],
        ])
        points = np.concatenate([points, points + np.array([[3.5, 2.5]])], axis=0)
        res_pairs, res_tris, res_perimeters = poly.calc_delaunay_adjacency(points, max_distance=2.0)
        exp_pairs = {
            0: {1, 2},
            1: {0, 2, 3, 5},
            2: {0, 1, 3, 4},
            3: {1, 2, 4, 5},
            4: {2, 3, 5, 6},
            5: {1, 3, 4, 6},
            6: {4, 5},
            7: {8, 9},
            8: {9, 10, 12, 7},
            9: {8, 10, 11, 7},
            10: {8, 9, 11, 12},
            11: {9, 10, 12, 13},
            12: {8, 10, 11, 13},
            13: {11, 12},
        }
        exp_tris = {
            (8, 9, 10),
            (9, 10, 11),
            (0, 1, 2),
            (7, 8, 9),
            (1, 2, 3),
            (1, 3, 5),
            (11, 12, 13),
            (10, 11, 12),
            (8, 10, 12),
            (3, 4, 5),
            (2, 3, 4),
            (4, 5, 6),
        }
        exp_perimeters = {
            tuple(sorted([4, 6, 5, 1, 0, 2])),
            tuple(sorted([11, 13, 12, 8, 7, 9])),
        }

        self.assertEqual(res_pairs, exp_pairs)
        self.assertEqual(res_tris, exp_tris)
        self.assertEqual(len(res_perimeters), len(exp_perimeters))
        for res in res_perimeters:
            self.assertIn(tuple(sorted(res)), exp_perimeters)


class TestMaskInPolygon(unittest.TestCase):

    def test_with_weird_shape(self):

        polygon = np.array([
            (-3, -2),
            (-1, 4),
            (6, 1),
            (3, 10),
            (-4, 9),
        ])

        x = np.linspace(-5, 8, 5)
        y = np.linspace(-5, 10, 5)

        res = poly.mask_in_polygon(polygon, x, y)
        exp = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.bool)
        np.testing.assert_equal(res, exp)

    def test_with_square(self):

        polygon = np.array([
            (-3, -2),
            (-3, 4),
            (4, 4),
            (4, -2),
        ])

        x = np.linspace(-5, 8, 5)
        y = np.linspace(-5, 10, 5)

        res = poly.mask_in_polygon(polygon, x, y)
        exp = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.bool)
        np.testing.assert_equal(res, exp)

    def test_with_rectangle(self):

        polygon = np.array([
            (0, 7),
            (0, 11),
            (3, 11),
            (3, 7),
        ])

        x = np.linspace(0, 5, 6)
        y = np.linspace(0, 10, 11)

        res = poly.mask_in_polygon(polygon, x, y)
        exp = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ], dtype=np.bool)
        np.testing.assert_equal(res, exp)


class TestCenterOfPolygon(unittest.TestCase):

    def test_gives_correct_center_evenly_sampled(self):

        # It's a square... so center is -1, +1
        polygon = np.array([
            (-4, -2),
            (-4, 4),
            (2, 4),
            (2, -2),
        ])

        cx, cy = poly.center_of_polygon(polygon)

        self.assertEqual(cx, -1)
        self.assertEqual(cy, 1)

    def test_gives_correct_center_clockwise(self):

        # It's a square... so center is -1, +1
        # BUUUT, we have extra samples along the line from x=(-4 to +2)
        polygon = np.array([
            (-4, -2),
            (-4, 4),
            (-3, 4),
            (-2, 4),
            (-1, 4),
            (1, 4),
            (2, 4),
            (2, -2),
        ])

        cx, cy = poly.center_of_polygon(polygon)

        self.assertEqual(cx, -1)
        self.assertEqual(cy, 1)

    def test_gives_correct_center_counter_clockwise(self):

        polygon = np.array([
            (2, -2),
            (2, 4),
            (1, 4),
            (-1, 4),
            (-2, 4),
            (-3, 4),
            (-4, 4),
            (-4, -2),
        ])

        cx, cy = poly.center_of_polygon(polygon)

        self.assertEqual(cx, -1)
        self.assertEqual(cy, 1)

    def test_gives_correct_center_circle(self):

        theta = np.linspace(0, 2*np.pi, 50)
        polygon = np.stack([
            np.cos(theta) + 5,
            np.sin(theta) - 1,
        ], axis=1)

        cx, cy = poly.center_of_polygon(polygon)

        np.testing.assert_almost_equal(cx, 5, decimal=3)
        np.testing.assert_almost_equal(cy, -1, decimal=3)


class TestAreaOfPolygon(unittest.TestCase):

    def test_gives_correct_area_scaling(self):

        polygon = np.array([
            (-3, -2),
            (-1, 4),
            (6, 1),
            (3, 10),
            (-4, 9),
        ])

        cx, cy = poly.center_of_polygon(polygon)

        res = poly.scale_polygon(polygon, 2.0)

        exp = np.array([
            (-6.36666667, -9.5333333),
            (-2.36666667,  2.4666667),
            (11.63333333, -3.5333333),
            (5.63333333, 14.4666667),
            (-8.36666667, 12.4666667),
        ])
        np.testing.assert_almost_equal(res, exp)

        rx, ry = poly.center_of_polygon(res)

        self.assertAlmostEqual(rx, cx)
        self.assertAlmostEqual(ry, cy)

        area = poly.area_of_polygon(res)
        self.assertEqual(area, 240)

    def test_gives_correct_area_clockwise(self):

        polygon = np.array([
            (-3, -2),
            (-1, 4),
            (6, 1),
            (3, 10),
            (-4, 9),
        ])

        res = poly.area_of_polygon(polygon)
        self.assertEqual(res, 60)

    def test_gives_correct_area_counter_clockwise(self):

        polygon = np.array([
            (-3, -2),
            (-1, 4),
            (6, 1),
            (3, 10),
            (-4, 9),
        ])

        res = poly.area_of_polygon(polygon[::-1, :])
        self.assertEqual(res, 60)
