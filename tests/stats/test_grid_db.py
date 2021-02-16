#!/usr/bin/env python3

# Standard lib
import unittest

# 3rd party
import numpy as np

import h5py

# Our own imports
from ..helpers import FileSystemTestCase

from deep_hipsc_tracking.stats import grid_db, _grid_db
from deep_hipsc_tracking.tracking import Link

# Constants

LINK1 = Link.from_arrays(
    tt=np.array([1, 2, 3, 4, 5]),
    xx=np.array([6.1, 7.1, 8.1, 9.1, 10.1]),
    yy=np.array([6.2, 7.2, 8.2, 9.2, 10.2]),
)
LINK2 = Link.from_arrays(
    tt=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]),
    xx=np.array([6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3, 13.3, 14.3]),
    yy=np.array([1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 6.4, 5.4, 3.4]),
)
LINK3 = Link.from_arrays(
    tt=np.array([3, 4, 5, 6, 7]),
    xx=np.array([6.5, 7.5, 8.5, 9.5, 10.5]),
    yy=np.array([5.6, 6.6, 7.6, 8.6, 9.6]),
)
LINK4 = Link.from_arrays(
    tt=np.array([4, 5, 6, 7, 8, 9]),
    xx=np.array([6.7, 7.7, 8.7, 9.7, 10.7, 11.7]),
    yy=np.array([4.8, 5.8, 6.8, 7.8, 8.8, 9.8]),
)
LINK5 = Link.from_arrays(
    tt=np.array([4, 5, 6, 7, 8, 9]),
    xx=np.array([7.7, 8.7, 9.7, 10.7, 11.7, 12.7]),
    yy=np.array([4.8, 5.8, 6.8, 7.8, 8.8, 9.8]),
)
LINK6 = Link.from_arrays(
    tt=np.array([4, 5, 6, 7, 8, 9]),
    xx=np.array([7.7, 8.7, 9.7, 10.7, 11.7, 12.7]),
    yy=np.array([5.8, 6.8, 7.8, 8.8, 9.8, 10.8]),
)

GRID_DB_ATTRS = [
    'space_scale',
    'time_scale',
    'timepoint_coords',
    'timepoint_real_coords',
    'timepoint_links',
    'track_links',
    'track_links_inv',
    'timepoint_meshes',
    'timepoint_triangles',
    'timepoint_perimeters',
    'timepoint_warp_coords',
    'timepoint_warp_radius',
    'local_densities_mesh',
    'local_cell_areas_mesh',
    'delta_divergence_mesh',
    'delta_curl_mesh',
    'local_displacement_mesh',
    'local_distance_mesh',
    'local_disp_vs_dist_mesh',
    'local_velocity_mesh',
    'local_speed_mesh',
    'local_persistence_mesh',
]

PERSISTENCE_ATTRS = [
    'pct_quiescent', 'pct_persistent', 'r_rad', 'x_rad', 'y_rad',
    'x_pos', 'y_pos', 'disp', 'dist', 'vel', 'tt', 'xx', 'yy', 'mask',
]

# Helper functions


def are_objects_equal(val1, val2, places=4, msg=''):
    """ Assert that stuff is equal given nested datatypes

    Properly handles tricky things like arrays inside dictionaries and NaNs

    :param object val1:
        The first object to test
    :param object val2:
        The second object to test
    :returns:
        True if they seem similar, False otherwise
    """

    try:
        if hasattr(val1, 'dtype') and hasattr(val2, 'dtype'):
            assert np.allclose(val1, val2, equal_nan=True), msg
        elif isinstance(val1, (int, float)) and np.isnan(val1):
            assert np.isnan(val2), msg
        elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            assert round(val1, places) == round(val2, places), msg
        elif isinstance(val1, dict):
            assert val1.keys() == val2.keys(), msg
            for key, sval1 in val1.items():
                sval2 = val2[key]
                assert are_objects_equal(sval1, sval2, places=places, msg=msg), msg
        elif isinstance(val1, (list, tuple)):
            assert len(val1) == len(val2)
            for v1, v2 in zip(val1, val2):
                assert are_objects_equal(v1, v2, places=places, msg=msg), msg
        else:
            assert val1 == val2, msg
    except AssertionError:
        raise
    except Exception as e:
        print('Error evaluating are objects equal: {}'.format(e))
        return False
    return True


# Tests


class TestFindTracksInROI(unittest.TestCase):
    """ Find tracks inside an ROI """

    LINKS = [
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([0]*5), yy=np.array([0]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([1]*5), yy=np.array([0]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([2]*5), yy=np.array([0]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([0]*5), yy=np.array([1]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3]), xx=np.array([1]*3), yy=np.array([1]*3)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([2]*5), yy=np.array([1]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([0]*5), yy=np.array([2]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([1]*5), yy=np.array([2]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([2]*5), yy=np.array([2]*5)),
    ]

    def test_finds_perimeter_points_over_time(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        point_ids = db.get_perimeter_point_ids(1, list(range(9)))

        exp_ids = [0, 2, 8, 6]

        self.assertEqual(point_ids, exp_ids)

        point_ids = db.get_perimeter_point_ids(1, [0, 1, 3, 4])

        exp_ids = [0, 1, 4, 3]

        self.assertEqual(point_ids, exp_ids)

        point_ids = db.get_perimeter_point_ids(5, list(range(8)))

        exp_ids = [0, 2, 7, 5]

        self.assertEqual(point_ids, exp_ids)

        point_timeline_ids = {
            1: list(range(9)),
            2: list(range(9)),
            3: list(range(9)),
            4: list(range(8)),
            5: list(range(8)),
        }

        res = db.get_perimeter_timeline_point_ids(point_timeline_ids)

        exp_ids = {
            1: [0, 2, 8, 6],
            2: [0, 2, 8, 6],
            3: [0, 2, 8, 6],
            4: [0, 2, 7, 5],
            5: [0, 2, 7, 5],
        }
        self.assertEqual(res, exp_ids)

    def test_finds_points_image_coords_over_time(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        coords = np.array([
            [0, 2, 2, 0],
            [0, 0, 1, 1],
        ]).T

        res = db.find_point_timelines_in_roi(coords)
        exp = {
            1: [0, 1, 2, 3, 4, 5],
            2: [0, 1, 2, 3, 4, 5],
            3: [0, 1, 2, 3, 4, 5],
            4: [0, 1, 2, 3, 4],
            5: [0, 1, 2, 3, 4],
        }

        self.assertEqual(res, exp)

        coords = np.array([
            [0, 1, 1, 0],
            [0, 0, 2, 2],
        ]).T

        res = db.find_point_timelines_in_roi(coords)
        exp = {
            1: [0, 1, 3, 4, 6, 7],
            2: [0, 1, 3, 4, 6, 7],
            3: [0, 1, 3, 4, 6, 7],
            4: [0, 1, 3, 5, 6],
            5: [0, 1, 3, 5, 6],
        }
        self.assertEqual(res, exp)

    def test_inverts_points_image_coords(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        coords = np.array([
            [0, 2, 2, 0],
            [0, 0, 1, 1],
        ]).T

        point_ids = db.find_point_timelines_in_roi(coords)
        res = db.invert_point_timeline_ids(point_ids)

        exp = {
            1: [6, 7, 8],
            2: [6, 7, 8],
            3: [6, 7, 8],
            4: [5, 6, 7],
            5: [5, 6, 7],
        }

        self.assertEqual(res, exp)

    def test_inverts_empty_set_to_all_points(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        res = db.invert_point_timeline_ids({1: set(), 5: set()})

        exp = {
            1: list(range(9)),
            5: list(range(8)),
        }

        self.assertEqual(res, exp)

    def test_inverts_point_timelines_image_coords(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        coords = np.array([
            [0, 2, 2, 0],
            [0, 0, 1, 1],
        ]).T

        point_ids = db.find_points_in_roi(coords, timepoint=1)
        res = db.invert_point_ids(point_ids, timepoint=1)

        exp = [6, 7, 8]

        self.assertEqual(res, exp)

        point_ids = db.find_points_in_roi(coords, timepoint=5)
        res = db.invert_point_ids(point_ids, timepoint=5)

        exp = [5, 6, 7]

        self.assertEqual(res, exp)

    def test_finds_points_image_coords(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        coords = np.array([
            [0, 2, 2, 0],
            [0, 0, 1, 1],
        ]).T

        res = db.find_points_in_roi(coords, timepoint=1)
        exp = [0, 1, 2, 3, 4, 5]

        self.assertEqual(res, exp)

        res = db.find_points_in_roi(coords, timepoint=5)
        exp = [0, 1, 2, 3, 4]

        self.assertEqual(res, exp)

        coords = np.array([
            [0, 1, 1, 0],
            [0, 0, 2, 2],
        ]).T

        res = db.find_points_in_roi(coords, timepoint=1)
        exp = [0, 1, 3, 4, 6, 7]

        self.assertEqual(res, exp)

        res = db.find_points_in_roi(coords, timepoint=5, use_mesh='coords')
        exp = [0, 1, 3, 5, 6]

        self.assertEqual(res, exp)

    def test_find_tracks_over_time(self):

        db = grid_db.GridDB(processes=1, time_scale=2.0, space_scale=3.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=5)
        db.warp_grid_to_circle()

        coords = np.array([
            [0, 1.1, 1.1, 0],  # From the center to the max radius to the right
            [0, 0, -0.1, -0.1],  # Narrow slice along the x-axis from 0 to -0.1
        ]).T

        res = db.find_track_timelines_in_roi(coords, use_mesh='warp_coords')
        exp = [4, 5]

        self.assertEqual(res, exp)

        coords = np.array([
            [0, 2, 2, 0],
            [0, 0, 1, 1],
        ]).T

        res = db.find_track_timelines_in_roi(coords, use_mesh='coords')
        exp = [0, 1, 2, 3, 4, 5]

        self.assertEqual(res, exp)

        coords = np.array([
            [0, 3, 3, 0],
            [0, 0, 6, 6],
        ]).T

        res = db.find_track_timelines_in_roi(coords, use_mesh='real_coords', timepoints=[4, 5])
        exp = [0, 1, 3, 6, 7]

        self.assertEqual(res, exp)

    def test_invert_tracks_image_coords(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        coords = np.array([
            [0, 2, 2, 0],
            [0, 0, 1, 1],
        ]).T

        track_ids = db.find_tracks_in_roi(coords, timepoint=1)
        res = db.invert_track_ids(track_ids)

        exp = [6, 7, 8]

        self.assertEqual(res, exp)

        track_ids = db.find_tracks_in_roi(coords, timepoint=5)
        res = db.invert_track_ids(track_ids)

        exp = [4, 6, 7, 8]

        self.assertEqual(res, exp)

    def test_finds_tracks_image_coords(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)

        coords = np.array([
            [0, 2, 2, 0],
            [0, 0, 1, 1],
        ]).T

        res = db.find_tracks_in_roi(coords, timepoint=1)
        exp = [0, 1, 2, 3, 4, 5]

        self.assertEqual(res, exp)

        res = db.find_tracks_in_roi(coords, timepoint=5)
        exp = [0, 1, 2, 3, 5]

        self.assertEqual(res, exp)

        coords = np.array([
            [0, 1, 1, 0],
            [0, 0, 2, 2],
        ]).T

        res = db.find_tracks_in_roi(coords, timepoint=1)
        exp = [0, 1, 3, 4, 6, 7]

        self.assertEqual(res, exp)

        res = db.find_tracks_in_roi(coords, timepoint=5, use_mesh='coords')
        exp = [0, 1, 3, 6, 7]

        self.assertEqual(res, exp)

    def test_finds_tracks_in_warp_coords(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        # Need large max distance so triangulation doesn't fail...
        db.triangulate_grid(max_distance=5)
        db.warp_grid_to_circle()

        coords = np.array([
            [0, 1.1, 1.1, 0],  # From the center to the max radius to the right
            [0, 0, -0.1, -0.1],  # Narrow slice along the x-axis from 0 to -0.1
        ]).T

        res = db.find_tracks_in_roi(coords, timepoint=1, use_mesh='timepoint_warp_coords')
        exp = [4, 5]  # Grabs the center and the middle right point

        self.assertEqual(res, exp)

        res = db.find_tracks_in_roi(coords, timepoint=5, use_mesh='warp_coords')
        exp = [5]  # Grabs middle right point only (center is missing)

        self.assertEqual(res, exp)

    def test_finds_tracks_in_real_coords(self):

        db = grid_db.GridDB(processes=1, time_scale=2.0, space_scale=3.0)
        for link in self.LINKS:
            db.add_track(link)

        coords = np.array([
            [0, 3, 3, 0],
            [0, 0, 6, 6],
        ]).T

        res = db.find_tracks_in_roi(coords, timepoint=1, use_mesh='timepoint_real_coords')
        exp = [0, 1, 3, 4, 6, 7]

        self.assertEqual(res, exp)

        res = db.find_tracks_in_roi(coords, timepoint=5, use_mesh='real_coords')
        exp = [0, 1, 3, 6, 7]

        self.assertEqual(res, exp)


class TestFindNeighboringTracks(unittest.TestCase):
    """ Find tracks at a timepoint using graph distances """

    LINKS = [
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([0]*5), yy=np.array([0]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([1]*5), yy=np.array([0]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([2]*5), yy=np.array([0]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([0]*5), yy=np.array([1]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3]), xx=np.array([1]*3), yy=np.array([1]*3)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([2]*5), yy=np.array([1]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([0]*5), yy=np.array([2]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([1]*5), yy=np.array([2]*5)),
        Link.from_arrays(tt=np.array([1, 2, 3, 4, 5]), xx=np.array([2]*5), yy=np.array([2]*5)),
    ]

    def test_finds_nearest_links_one_step(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=1.1)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=1)
        exp = {1, 3}

        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=3)
        exp = {1, 3, 5, 7}

        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=8, timepoint=5)
        exp = {5, 7}

        self.assertEqual(res, exp)

    def test_finds_nearest_links_missing_track(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=1.1)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=5, distance=1)
        exp = set()

        self.assertEqual(res, exp)

    def test_finds_nearest_links_disconnected_tracks(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=1.1)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=5, distance=1)
        exp = set()

        self.assertEqual(res, exp)

    def test_finds_nearest_links_two_steps(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS[:5]:
            db.add_track(link)
        db.add_track(self.LINKS[6])
        db.add_track(self.LINKS[8])
        db.triangulate_grid(max_distance=1.1)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=1, distance=1)
        exp = {1, 3}

        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=6, timepoint=1, distance=1)
        exp = set()

        self.assertEqual(res, exp)

    def test_finds_nearest_links_n_steps_poorly_connected(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=1.1)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=5, distance=0)
        exp = {0}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=5, distance=1)
        exp = {1, 3}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=5, distance=2)
        exp = {2, 6}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=5, distance=3)
        exp = {7, 5}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=5, distance=4)
        exp = {8}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=0, timepoint=5, distance=5)
        exp = set()
        self.assertEqual(res, exp)

    def test_finds_nearest_links_n_steps_well_connected(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=1.1)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=1, distance=0)
        exp = {4}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=1, distance=1)
        exp = {1, 3, 5, 7}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=1, distance=2)
        exp = {0, 2, 6, 8}
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=1, distance=3)
        exp = set()
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=1, distance=4)
        exp = set()
        self.assertEqual(res, exp)

        res = db.find_neighboring_tracks(trackidx=4, timepoint=1, distance=5)
        exp = set()
        self.assertEqual(res, exp)

    def test_finds_nearest_tracks_over_time_n_steps(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=1.1)

        res = db.find_neighboring_track_timelines(trackidx=0, distance=1)
        exp = {1, 3}
        self.assertEqual(res, exp)

        res = db.find_neighboring_track_timelines(trackidx=0, distance=2)
        exp = {2, 4, 6}
        self.assertEqual(res, exp)

        res = db.find_neighboring_track_timelines(trackidx=1, distance=1)
        exp = {0, 4, 2}
        self.assertEqual(res, exp)

        res = db.find_neighboring_track_timelines(trackidx=1, distance=2)
        exp = {3, 5, 7}
        self.assertEqual(res, exp)

    def test_finds_all_nearest_tracks_over_all_time(self):

        db = grid_db.GridDB(processes=1, time_scale=1.0, space_scale=1.0)
        for link in self.LINKS:
            db.add_track(link)
        db.triangulate_grid(max_distance=1.1)

        res = db.find_all_neighboring_track_timelines(distance=1)
        exp = {
            0: {1, 3},
            1: {0, 2, 4},
            2: {1, 5},
            3: {0, 4, 6},
            4: {1, 3, 5, 7},
            5: {2, 4, 8},
            6: {3, 7},
            7: {4, 6, 8},
            8: {5, 7},
        }
        self.assertEqual(res, exp)

        res = db.find_all_neighboring_track_timelines(distance=2, trackidxs=[1, 3, 5, 7])
        exp = {
            1: {3, 5, 7},
            3: {1, 5, 7},
            5: {1, 3, 7},
            7: {1, 3, 5},
        }
        self.assertEqual(res, exp)


class TestCalcAverageTriDensity(unittest.TestCase):

    def test_calcs_area(self):

        timepoint_points = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 4.0],
            [0.0, 0.0],
            [0.0, 3.0],
            [4.0, 0.0],
        ])
        timepoint_triangles = np.array([
            [0, 1, 2],
            [3, 1, 2],  # Average over two sets, one with area 0.5, one with area 1.5
            [4, 5, 6],  # Single triangle area == 6.0
        ])

        res = _grid_db.calc_average_tri_density(timepoint_points,
                                                timepoint_triangles,
                                                timepoint_points.shape[0],
                                                timepoint_triangles.shape[0])
        exp = np.array([0.5, 1.0, 1.0, 1.5, 6.0, 6.0, 6.0])
        np.testing.assert_almost_equal(res, exp)


class TestCalcAverageSegmentDivergence(unittest.TestCase):

    def test_calcs_divergence_different_sizes(self):

        # Area 1 to Area 2 with unlinked points
        areas1 = np.array([1.0, 2.0, 3.0])
        areas2 = np.array([2.0, 1.0, 3.0, 4.0])
        links = np.array([
            (0, 0), (1, 1)
        ])

        res1, res2 = _grid_db.calc_average_segment_divergence(
            areas1, areas2, links, areas1.shape[0], areas2.shape[0], links.shape[0], 0.5)

        exp1 = np.array([
            2.0, -2.0, np.nan,
        ])
        exp2 = np.array([
            -2.0, 2.0, np.nan, np.nan
        ])

        np.testing.assert_almost_equal(res1, exp1, decimal=4)
        np.testing.assert_almost_equal(res2, exp2, decimal=4)


class TestCalcAverageSegmentAngle(unittest.TestCase):

    def test_calcs_no_angle_change(self):

        # 30 degrees to 30 degrees, with a shift
        points1 = np.array([
            [0.0, 0.0],
            [0.866, 0.5],
        ])
        points2 = np.array([
            [1.0, 1.0],
            [1.866, 1.5],
        ])
        links = {
            0: 0,
            1: 1,
        }
        mesh = {
            0: [1],
            1: [0],
        }

        res1, res2 = _grid_db.calc_average_segment_angle(
            points1, points2, mesh, links, points1.shape[0], points2.shape[0])

        exp = np.array(
            [0, 0]
        )

        np.testing.assert_almost_equal(res1, -res2)
        np.testing.assert_almost_equal(res1, exp, decimal=4)

    def test_calcs_with_different_size_arrays(self):

        # 30 degrees to 30 degrees, with a shift
        points1 = np.array([
            [0.0, 0.0],
            [0.866, 0.5],
            [0.0, 0.1],
        ])
        points2 = np.array([
            [1.0, 1.0],
            [1.866, 1.5],
            [2.0, 2.0],
            [3.0, 3.0],
        ])
        links = {
            0: 0,
            1: 1,
        }
        mesh = {
            0: [1],
            1: [0],
        }

        res1, res2 = _grid_db.calc_average_segment_angle(
            points1, points2, mesh, links, points1.shape[0], points2.shape[0])

        exp1 = np.array(
            [0, 0, np.nan],
        )
        exp2 = np.array(
            [0, 0, np.nan, np.nan],
        )
        np.testing.assert_almost_equal(res1, exp1, decimal=4)
        np.testing.assert_almost_equal(res2, exp2, decimal=4)

    def test_calcs_pos_angle_change(self):

        # 30 degrees to 30 degrees, with a shift
        points1 = np.array([
            [0.0, 0.0],
            [0.866, 0.5],
            [0.0, 1.0],
        ])
        points2 = np.array([
            [1.0, 1.0],
            [1.5, 1.866],
            [0.134, 1.5],
        ])
        links = {
            0: 0,
            1: 1,
            2: 2,
        }
        mesh = {
            0: [1, 2],
            1: [0],
            2: [0],
        }

        res1, res2 = _grid_db.calc_average_segment_angle(
            points1, points2, mesh, links, points1.shape[0], points2.shape[0])

        exp = np.array(
            [np.pi/180*45, np.pi/180*30, np.pi/180*60]
        )

        np.testing.assert_almost_equal(res1, -res2)
        np.testing.assert_almost_equal(res1, exp, decimal=4)

    def test_calcs_multiple_angles(self):

        points1 = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
        ])
        points2 = np.array([
            [0.0, 1.0],
            [0.1, 1.2],
            [0.2, 1.4],
        ])
        links = {
            0: 0,
            1: 1,
            2: 2,
        }
        mesh = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1],
        }

        res1, res2 = _grid_db.calc_average_segment_angle(
            points1, points2, mesh, links, points1.shape[0], points2.shape[0])

        exp = np.array(
            [0.3217506, 0.3217505, 0.32175051]
        )

        np.testing.assert_almost_equal(res1, -res2)
        np.testing.assert_almost_equal(res1, exp, decimal=4)


class TestCalcDeltaStats(unittest.TestCase):

    def test_calcs_deltas_no_warp(self):

        in_coords = [
            ((0, 0), (0.1, 0.1)),
            ((0, 1), (0.1, 2.1)),
            ((1, 1), (2.1, 2.1)),
        ]
        divergence, curl, coords, warp_coords = _grid_db.calc_delta_stats(in_coords, [])

        exp_coords = np.array([
            [0.05, 0.05],
            [1.55, 1.55],
            [0.05, 1.55],
        ])
        exp_warp_coords = np.zeros((0, 2))

        self.assertAlmostEqual(divergence, np.log(2.0))
        self.assertAlmostEqual(curl, 0.0)

        np.testing.assert_almost_equal(coords, exp_coords)
        np.testing.assert_almost_equal(warp_coords, exp_warp_coords)

    def test_calcs_deltas_with_warp(self):

        in_coords = [
            ((0, 0), (0.1, 0.1)),
            ((1, 1), (2.1, 2.1)),
            ((0, 1), (0.1, 2.1)),
        ]
        in_warp_coords = [
            ((0.0, 0.0), (0.01, 0.01)),
            ((0.1, 0.1), (0.21, 0.21)),
            ((0.0, 0.1), (0.01, 0.21)),
        ]

        divergence, curl, coords, warp_coords = _grid_db.calc_delta_stats(in_coords, in_warp_coords)

        exp_coords = np.array([
            [0.05, 0.05],
            [1.55, 1.55],
            [0.05, 1.55],
        ])
        exp_warp_coords = np.array([
            [0.005, 0.005],
            [0.155, 0.155],
            [0.005, 0.155],
        ])

        self.assertAlmostEqual(divergence, np.log(2.0))
        self.assertAlmostEqual(curl, 0.0)

        np.testing.assert_almost_equal(coords, exp_coords)
        np.testing.assert_almost_equal(warp_coords, exp_warp_coords)


class TestCalcLocalDensity(unittest.TestCase):

    def test_calcs_density_no_warp(self):

        timepoint_points = [
            (2.0, 2.0),
            (1.0, 4.0),
            (4.0, 1.0),
            (4.0, 4.0),
        ]
        timepoint_warp_points = []
        timepoint_mesh = {
            0: {1, 2, 3},
        }
        res = _grid_db.calc_local_density(timepoint_points, timepoint_warp_points, timepoint_mesh)
        self.assertEqual(len(res), 3)

        res_areas, res_perimeters, res_warp_perimeters = res
        exp_areas = {
            0: 1 / 4.5,  # Triangle has area == 4.5 so density is 1/4.5
        }
        exp_perimeters = {
            0: np.array([[4.0, 1.0], [4.0, 4.0], [1.0, 4.0]]),
        }
        exp_warp_perimeters = {}

        self.assertEqual(res_areas, exp_areas)
        assert are_objects_equal(res_perimeters, exp_perimeters)
        assert are_objects_equal(res_warp_perimeters, exp_warp_perimeters)

    def test_calcs_density_with_warp(self):

        timepoint_points = [
            (2.0, 2.0),
            (1.0, 4.0),
            (4.0, 1.0),
            (4.0, 4.0),
        ]
        timepoint_warp_points = [
            (0.0, 0.0),
            (0.1, 0.3),
            (0.4, 0.1),
            (0.4, 0.4),
        ]
        timepoint_mesh = {
            0: {1, 2, 3},
        }
        res = _grid_db.calc_local_density(timepoint_points, timepoint_warp_points, timepoint_mesh)
        self.assertEqual(len(res), 3)

        res_areas, res_perimeters, res_warp_perimeters = res
        exp_areas = {
            0: 1 / 4.5,  # Triangle has area == 4.5 so density is 1/4.5
        }
        exp_perimeters = {
            0: np.array([[4.0, 1.0], [4.0, 4.0], [1.0, 4.0]]),
        }
        exp_warp_perimeters = {
            0: np.array([[0.4, 0.1], [0.4, 0.4], [0.1, 0.3]]),
        }

        self.assertEqual(res_areas, exp_areas)
        assert are_objects_equal(res_perimeters, exp_perimeters)
        assert are_objects_equal(res_warp_perimeters, exp_warp_perimeters)


class TestCalcDeltaDensity(unittest.TestCase):

    def test_calcs_delta_density_no_warp(self):

        timepoint1_points = [
            (2.0, 2.0),
            (1.0, 4.0),
            (4.0, 1.0),
            (4.0, 4.0),
        ]
        timepoint2_points = [
            (3.1, 3.1),
            (1.1, 6.1),
            (6.1, 1.1),
            (6.1, 6.1),
        ]

        timepoint1_warp_points = []
        timepoint2_warp_points = []
        timepoint1_mesh = {
            0: {1, 2, 3},
        }
        timepoint2_mesh = {
            0: {1, 2, 3},
        }
        timepoint_links = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }

        res = _grid_db.calc_delta_density(timepoint1_points, timepoint2_points,
                                          timepoint1_warp_points, timepoint2_warp_points,
                                          timepoint1_mesh, timepoint2_mesh,
                                          timepoint_links)
        self.assertEqual(len(res), 4)

        res_divergences, res_curls, res_perimeters, res_warp_perimeters = res
        exp_divergences = {
            (0, 0): 1.0217,  # Expanded by ~3x
        }
        exp_curls = {
            (0, 0): 0.0,  # No rotation between the two
        }
        exp_perimeters = {
            (0, 0): np.array([
                (5.05, 1.05),
                (5.05, 5.05),
                (1.05, 5.05),
            ]),
        }
        exp_warp_perimeters = {}

        assert are_objects_equal(res_divergences, exp_divergences)
        assert are_objects_equal(res_curls, exp_curls)
        assert are_objects_equal(res_perimeters, exp_perimeters)
        assert are_objects_equal(res_warp_perimeters, exp_warp_perimeters)

    def test_calcs_delta_density_with_warp(self):

        timepoint1_points = [
            (2.0, 2.0),
            (1.0, 4.0),
            (4.0, 1.0),
            (4.0, 4.0),
        ]
        timepoint2_points = [
            (3.1, 3.1),
            (1.1, 6.1),
            (6.1, 1.1),
            (6.1, 6.1),
        ]

        timepoint1_warp_points = [
            (0.02, 0.02),
            (0.01, 0.04),
            (0.04, 0.01),
            (0.04, 0.04),
        ]
        timepoint2_warp_points = [
            (0.031, 0.031),
            (0.011, 0.061),
            (0.061, 0.011),
            (0.061, 0.061),
        ]
        timepoint1_mesh = {
            0: {1, 2, 3},
        }
        timepoint2_mesh = {
            0: {1, 2, 3},
        }
        timepoint_links = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }

        res = _grid_db.calc_delta_density(timepoint1_points, timepoint2_points,
                                          timepoint1_warp_points, timepoint2_warp_points,
                                          timepoint1_mesh, timepoint2_mesh,
                                          timepoint_links)
        self.assertEqual(len(res), 4)

        res_divergences, res_curls, res_perimeters, res_warp_perimeters = res
        exp_divergences = {
            (0, 0): 1.0217,  # Expanded by ~3x
        }
        exp_curls = {
            (0, 0): 0.0,  # No rotation between the two
        }
        exp_perimeters = {
            (0, 0): np.array([
                (5.05, 1.05),
                (5.05, 5.05),
                (1.05, 5.05),
            ]),
        }
        exp_warp_perimeters = {
            (0, 0): np.array([
                (0.0505, 0.0105),
                (0.0505, 0.0505),
                (0.0105, 0.0505),
            ]),
        }
        assert are_objects_equal(res_divergences, exp_divergences)
        assert are_objects_equal(res_curls, exp_curls)
        assert are_objects_equal(res_perimeters, exp_perimeters)
        assert are_objects_equal(res_warp_perimeters, exp_warp_perimeters)


class TestGridDB(FileSystemTestCase):

    def assertDictTuplesEqual(self, dict1, dict2, decimals=2):

        msg = 'Invalid dict tuples\n\n Got {}\n\n Expected {}\n\n'
        msg = msg.format(dict1, dict2)
        self.assertEqual(dict1.keys(), dict2.keys(), msg=msg)
        for key in dict1:
            coords1 = dict1[key]
            coords2 = dict2[key]
            self.assertEqual(len(coords1), len(coords2), msg=msg)
            for c1, c2 in zip(coords1, coords2):
                assert are_objects_equal(c1, c2, places=decimals, msg=msg)

    def assertAttrsEqual(self, db1, db2, attrs):

        for attr in attrs:

            self.assertTrue(hasattr(db1, attr), msg='db1 missing {}'.format(attr))
            self.assertTrue(hasattr(db2, attr), msg='db2 missing {}'.format(attr))

            val1 = getattr(db1, attr)
            val2 = getattr(db2, attr)

            msg = '"{}" mismatch:\n\n db1.{}={}\n\n db2.{}={}\n\n'.format(
                attr, attr, val1, attr, val2)

            try:
                self.assertTrue(are_objects_equal(val1, val2), msg=msg)
            except Exception:
                print(msg)
                raise

    def load_full_grid_db(self, links=None,
                          time_scale: float = 2.5,
                          space_scale: float = 1.5,
                          max_distance: float = 10) -> grid_db.GridDB:
        """ Load a full grid database and calculate everything """

        if links is None:
            links = [LINK1, LINK2, LINK3, LINK4, LINK5, LINK6]

        db = grid_db.GridDB(processes=1, time_scale=time_scale, space_scale=space_scale)
        for link in links:
            db.add_track(link)
        db.triangulate_grid(max_distance=max_distance)
        db.warp_grid_to_circle()
        db.calc_radial_stats()
        db.calc_local_densities_mesh()
        db.calc_delta_divergence_mesh()
        db.calc_delta_curl_mesh()
        return db

    # Tests

    def test_interp_track_values(self):

        db = self.load_full_grid_db()

        res_tt, res_xx, res_yy = db.interp_track_values(1, 'timepoint_coords', resample_factor=2, interp_points=3)

        exp_tt = np.array([2.0, 2.47058824, 2.94117647, 3.41176471, 3.88235294, 4.35294118,
                           4.82352941, 5.29411765, 5.76470588, 6.23529412, 6.70588235, 7.17647059,
                           7.64705882, 8.11764706, 8.58823529, 9.05882353, 9.52941176, 10.0])
        exp_xx = np.array([6.3, 6.77058824, 7.24117647, 7.71176471, 8.18235294, 8.65294118,
                           9.12352941, 9.59411765, 10.06470588, 10.53529412, 11.00588235, 11.47647059,
                           11.94705882, 12.41764706, 12.88823529, 13.35882353, 13.82941176, 14.3])
        exp_yy = np.array([1.4, 1.87058824, 2.34117647, 2.81176471, 3.28235294, 3.75294118,
                           4.22352941, 4.69411765, 5.16470588, 5.63529412, 6.10588235, 6.15490196,
                           6.39019608, 6.00784314, 5.77254902, 4.97843137, 4.27254902, 3.56666667])

        assert are_objects_equal(res_tt, exp_tt)
        assert are_objects_equal(res_xx, exp_xx)
        assert are_objects_equal(res_yy, exp_yy)

        res_tt, res_density, res_cell_area = db.interp_track_values(1, 'local_densities_mesh', 'local_cell_areas_mesh',
                                                                    resample_factor=2, interp_points=3)

        exp_density = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.32319929,
                                0.37894138, 0.82776939, 1.05670427, 1.12347668, 1.35241142, 1.02106027,
                                0.84676812, 0.70297756, 0.47291232, np.nan, np.nan, np.nan])
        exp_cell_area = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 3.61029179,
                                  2.35735036, 1.80220443, 1.32573431, 1.18676404, 0.71029406, 1.14522091,
                                  1.36580945, 1.87132235, 2.35661557, np.nan, np.nan, np.nan])

        assert are_objects_equal(res_tt, exp_tt)
        assert are_objects_equal(res_density, exp_density)
        assert are_objects_equal(res_cell_area, exp_cell_area)

    def test_add_track_to_mesh(self):

        link1 = Link.from_arrays(
            tt=np.array([1, 2, 3, 4, 5]),
            xx=np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
            yy=np.array([1.1, 1.2, 1.3, 1.4, 1.5]),
        )
        link2 = Link.from_arrays(
            tt=np.array([2, 3, 4, 5]),
            xx=np.array([2.2, 2.3, 2.4, 2.5]),
            yy=np.array([2.2, 2.3, 2.4, 2.5]),
        )
        link3 = Link.from_arrays(
            tt=np.array([3, 4, 5, 6]),
            xx=np.array([3.3, 3.4, 3.5, 3.6]),
            yy=np.array([3.3, 3.4, 3.5, 3.6]),
        )

        db = self.load_full_grid_db(links=[link1, link2, link3],
                                    time_scale=0.1,
                                    space_scale=2.0)
        exp_coords = {
            1: [(1.1, 1.1)],
            2: [(1.2, 1.2), (2.2, 2.2)],
            3: [(1.3, 1.3), (2.3, 2.3), (3.3, 3.3)],
            4: [(1.4, 1.4), (2.4, 2.4), (3.4, 3.4)],
            5: [(1.5, 1.5), (2.5, 2.5), (3.5, 3.5)],
            6: [(3.6, 3.6)],
        }
        self.assertDictTuplesEqual(db.timepoint_coords, exp_coords)

        exp_velocity = {
            1: [2.8283],
            2: [2.8283, 2.8283],
            3: [2.8283, 2.8283, 2.8283],
            4: [2.8283, 2.8283, 2.8283],
            5: [np.nan, np.nan, 2.8283],
            6: [np.nan],
        }
        self.assertDictTuplesEqual(db.local_velocity_mesh, exp_velocity)

    def test_get_track_summaries(self):

        db = self.load_full_grid_db()

        res_x, res_y = db.get_all_track_summaries('timepoint_coords', func='mean')
        exp_x = np.array([8.1, 10.3,  8.5,  9.2, 10.2, 10.2])
        exp_y = np.array([8.2, 4.28888889, 7.6, 7.3, 7.3, 8.3])

        assert are_objects_equal(res_x, exp_x)
        assert are_objects_equal(res_y, exp_y)

        res_x, res_y, res_t = db.get_all_track_summaries('timepoint_real_coords', func='mean')

        exp_x = np.array([12.15, 15.45, 12.75, 13.8, 15.3, 15.3])
        exp_y = np.array([12.3, 6.43333333, 11.4, 10.95, 10.95, 12.45])
        exp_t = np.array([7.5, 15.0, 12.5, 16.25, 16.25, 16.25])

        assert are_objects_equal(res_x, exp_x)
        assert are_objects_equal(res_y, exp_y)
        assert are_objects_equal(res_t, exp_t)

        res_density = db.get_all_track_summaries('local_densities_mesh', func='mean')[0]
        exp_density = np.array([0.20576145, 0.6369991, 0.98977128, 0.66489163, 0.77777793, 0.97738294])

        assert are_objects_equal(res_density, exp_density)

        res_distance = db.get_all_track_summaries('local_distance_mesh', func='max')[0]
        exp_distance = np.array([7.81539074, 14.00685354, 7.81539074, 9.91486682, 9.91486682, 9.91486682])

        assert are_objects_equal(res_distance, exp_distance)

    def test_mesh_shapes_match(self):

        db = self.load_full_grid_db()

        counts = {}
        timepoints = None

        for attr in db.NUM_ARRAY_ARGS.keys():
            if timepoints is None:
                timepoints = set(getattr(db, attr))
            else:
                self.assertEqual(timepoints, set(getattr(db, attr)),
                                 msg='Bad timepoints {}'.format(attr))

            for timepoint, values in getattr(db, attr).items():
                if timepoint not in counts:
                    counts[timepoint] = len(values)
                else:
                    self.assertEqual(counts[timepoint], len(values),
                                     msg="Bad counts {} at {}".format(attr, timepoint))

    def test_extract_track_data_timepoint_real_coords(self):

        db = self.load_full_grid_db()

        tidx, xidx, xreal, yreal, treal = db.get_track_values(1, 'timepoint_real_coords')

        exp_tidx = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        exp_xidx = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

        exp_treal = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]) * 2.5
        exp_xreal = np.array([6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3, 13.3, 14.3]) * 1.5
        exp_yreal = np.array([1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 6.4, 5.4, 3.4]) * 1.5

        np.testing.assert_almost_equal(tidx, exp_tidx)
        np.testing.assert_almost_equal(xidx, exp_xidx)

        np.testing.assert_almost_equal(xreal, exp_xreal)
        np.testing.assert_almost_equal(yreal, exp_yreal)
        np.testing.assert_almost_equal(treal, exp_treal)

    def test_warp_to_circle_at_timepoints(self):

        # Create a circle that expands over time
        # At t=1, r=1
        # At t=2, r=2, ... etc
        t = np.arange(1, 10)

        # Shift the center in x and y
        cx = 2
        cy = -1
        theta = np.linspace(0, 2*np.pi, 50)
        x = np.cos(theta) + cx
        y = np.sin(theta) + cy

        # Points along the radius
        links = [Link.from_arrays(t, x[i]*t, y[i]*t)
                 for i in range(theta.shape[0])]
        # Point at the center
        links.append(Link.from_arrays(t, t*cx, t*cy))

        db = self.load_full_grid_db(links=links, max_distance=20)

        # Ray from the center out
        coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]) + np.array([(cx, cy)])

        warp_coords = db.warp_to_timepoint(1, coords)

        exp_coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ])

        np.testing.assert_almost_equal(warp_coords, exp_coords, decimal=4)

        coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]) + np.array([(cx*5, cy*5)])

        warp_coords = db.warp_to_timepoint(5, coords)

        exp_coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]) / 5

        np.testing.assert_almost_equal(warp_coords, exp_coords, decimal=4)

        coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]) + np.array([(cx*9, cy*9)])

        warp_coords = db.warp_to_timepoint(9, coords)

        exp_coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]) / 9

        np.testing.assert_almost_equal(warp_coords, exp_coords, decimal=4)

    def test_inv_warp_to_circle_at_timepoints(self):

        # Create a circle that expands over time
        # At t=1, r=1
        # At t=2, r=2, ... etc
        t = np.arange(1, 10)

        # Shift the center in x and y
        cx = 2
        cy = -1
        theta = np.linspace(0, 2*np.pi, 50)
        x = np.cos(theta) + cx
        y = np.sin(theta) + cy

        # Points along the radius
        links = [Link.from_arrays(t, x[i]*t, y[i]*t)
                 for i in range(theta.shape[0])]
        # Point at the center
        links.append(Link.from_arrays(t, t*cx, t*cy))

        db = self.load_full_grid_db(links=links, max_distance=20)

        coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ])

        warp_coords = db.inv_warp_to_timepoint(1, coords)

        exp_coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]) + np.array([[cx, cy]])

        np.testing.assert_almost_equal(warp_coords, exp_coords, decimal=4)

        coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ])

        warp_coords = db.inv_warp_to_timepoint(5, coords)

        exp_coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ]) * 5 + np.array([[cx, cy]]) * 5

        np.testing.assert_almost_equal(warp_coords, exp_coords, decimal=4)

    def test_warp_inv_warp_roundtrips(self):

        # Create a circle that expands over time
        # At t=1, r=1
        # At t=2, r=2, ... etc
        t = np.arange(1, 10)

        # Shift the center in x and y
        cx = 2
        cy = -1
        theta = np.linspace(0, 2*np.pi, 50)
        x = np.cos(theta) + cx
        y = np.sin(theta) + cy

        # Points along the radius
        links = [Link.from_arrays(t, x[i]*t, y[i]*t)
                 for i in range(theta.shape[0])]
        # Point at the center
        links.append(Link.from_arrays(t, t*cx, t*cy))

        db = self.load_full_grid_db(links=links, max_distance=20)

        # Ray from the center out
        coords = np.array([
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
        ])

        for timepoint in t:
            warp_coords = db.warp_to_timepoint(timepoint, coords)
            inv_warp_coords = db.inv_warp_to_timepoint(timepoint, warp_coords)

            np.testing.assert_almost_equal(inv_warp_coords, coords, decimal=4)

    def test_extract_track_data_timepoint_warp_coords(self):

        db = self.load_full_grid_db()

        tidx, xidx, xwarp, ywarp, rwarp = db.get_track_values(
            1, 'timepoint_warp_coords', 'timepoint_warp_radius')

        exp_tidx = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        exp_xidx = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

        exp_xwarp = np.array([np.nan, -1.01725220e-06, 1.14761399e-01, 1.14761377e-01,
                              4.63480483e-01, 4.63480483e-01, 3.55550202e-01, 2.19104332e-01, np.nan])
        exp_ywarp = np.array([np.nan, -1.00000043, -0.99460722, -0.99460722, -0.88833763,
                              -0.88833763, -0.93735939, -0.97601022, np.nan])

        np.testing.assert_almost_equal(tidx, exp_tidx)
        np.testing.assert_almost_equal(xidx, exp_xidx)

        np.testing.assert_almost_equal(xwarp, exp_xwarp, decimal=4)
        np.testing.assert_almost_equal(ywarp, exp_ywarp, decimal=4)
        np.testing.assert_almost_equal(rwarp, np.sqrt(xwarp**2 + ywarp**2))

    def test_extract_track_data_density_curl_divergence(self):

        db = self.load_full_grid_db()

        tidx, xidx, density, cell_area, curl, div = db.get_track_values(
            1, 'local_densities_mesh', 'local_cell_areas_mesh', 'delta_curl_mesh', 'delta_divergence_mesh')

        exp_tidx = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        exp_xidx = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

        exp_density = np.array([np.nan, 0.1234568, 0.3603607, 0.3603607, 1.3333339, 1.3333333,
                                0.5925923, 0.355556, np.nan])
        exp_cell_area = np.array([np.nan, 8.1, 2.775, 2.775, 0.75, 0.75, 1.6875, 2.8125, np.nan])
        exp_curl = np.array([np.nan,  3.5132221e-08,  2.0213393e-08,  2.6045209e-08,
                             2.3076574e-07, -8.2363701e-02, -1.5114689e-01, -1.3756596e-01, np.nan])
        exp_div = np.array([np.nan, -2.1300004, -1.0650002, -0.4049996, -0.4049995,
                            0.1875003,  0.4124992,  0.4499981, np.nan])

        np.testing.assert_almost_equal(tidx, exp_tidx)
        np.testing.assert_almost_equal(xidx, exp_xidx)

        np.testing.assert_almost_equal(cell_area, exp_cell_area, decimal=4)
        np.testing.assert_almost_equal(density, exp_density, decimal=4)
        np.testing.assert_almost_equal(curl, exp_curl, decimal=4)
        np.testing.assert_almost_equal(div, exp_div, decimal=4)

    def test_extract_track_data_density_subset_timeline(self):

        db = self.load_full_grid_db()

        roi = np.array([
            [5, 1],
            [5, 9],
            [8, 9],
            [8, 1],
        ])
        points = db.find_point_timelines_in_roi(perimeter=roi, timepoints=[4, 5])

        density = db.find_values_for_point_timeline('local_densities_mesh', points=points)

        exp_density = {
            4: np.array([0.6349, 0.7843, 0.8889, 0.4535]),
            5: np.array([0.7843]),
        }
        for key, res in density.items():
            exp = exp_density[key]
            np.testing.assert_almost_equal(res, exp, decimal=4)

    def test_extract_track_data_density_curl_divergence_subset(self):

        db = self.load_full_grid_db()

        roi = np.array([
            [5, 1],
            [5, 9],
            [8, 9],
            [8, 1],
        ])
        points = db.find_points_in_roi(perimeter=roi, timepoint=4)

        density = db.find_values_for_points('local_densities_mesh', points=points, timepoint=4)

        exp_density = np.array([0.6349, 0.7843, 0.8889, 0.4535])
        np.testing.assert_almost_equal(density, exp_density, decimal=4)

        cell_area = db.find_values_for_points('local_cell_areas_mesh', points=points, timepoint=4)

        exp_cell_area = np.array([1.575, 1.275, 1.125, 2.205])
        np.testing.assert_almost_equal(cell_area, exp_cell_area, decimal=4)

        curl = db.find_values_for_points('delta_curl_mesh', points=points, timepoint=4)

        exp_curl = np.array([-1.6653e-16,  2.2204e-16,  2.9606e-16, -1.7764e-16])
        np.testing.assert_almost_equal(curl, exp_curl, decimal=4)

        div = db.find_values_for_points('delta_divergence_mesh', points=points, timepoint=4)

        exp_div = np.array([-1.3050e+00, -1.5099e-15, -1.3323e-15, -5.3291e-16])
        np.testing.assert_almost_equal(div, exp_div, decimal=4)

    def test_extract_track_velocity_distance(self):

        db = self.load_full_grid_db()

        res = db.get_track_values(
            1, 'local_velocity_mesh', 'local_speed_mesh',
            'local_distance_mesh', 'local_displacement_mesh',
            'local_disp_vs_dist_mesh', 'local_persistence_mesh')

        tidx, xidx, velocity, speed, distance, displacement, disp_vs_dist, persistence = res

        exp_tidx = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
        exp_xidx = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])

        exp_velocity = np.array([0.8043568, 0.8043568, 0.8043568, 0.7959997, 0.7536884,
                                 0.7025816, 0.6795315, 0.6657548, np.nan])
        exp_speed = np.array([0.8043568, 0.8043568, 0.8043568, 0.8031527, 0.792132,
                              0.7758652, 0.7493032, 0.7318444, np.nan])
        exp_distance = np.array([1.3788973, 3.4472433, 5.5155892, 7.5715502, 9.2798938,
                                 11.0758882, 12.6275757, 14.0068535, np.nan])
        exp_displacement = np.array([1.3788973, 3.4472433, 5.5155892, 7.497977, 8.8194847,
                                     10.0297027, 11.4520616, 12.7423756, np.nan])
        exp_disp_vs_dist = np.array([1.0, 1.0, 1.0, 0.9910537, 0.9515971, 0.9055454,
                                     0.9068893, 0.9096996, np.nan])
        exp_persistence = np.array([1.0, 1.0, 0.8, 0.0, 0.0, 0.4, 1.0, 0.75, 0.0])

        np.testing.assert_almost_equal(tidx, exp_tidx)
        np.testing.assert_almost_equal(xidx, exp_xidx)

        np.testing.assert_almost_equal(velocity, exp_velocity)
        np.testing.assert_almost_equal(speed, exp_speed)
        np.testing.assert_almost_equal(distance, exp_distance)
        np.testing.assert_almost_equal(displacement, exp_displacement)
        np.testing.assert_almost_equal(disp_vs_dist, exp_disp_vs_dist)
        np.testing.assert_almost_equal(persistence, exp_persistence)

    def test_from_to_hdf5_roundtrips(self):

        hdf5_file = self.tempdir / 'temp.h5'

        db1 = self.load_full_grid_db()

        self.assertFalse(hdf5_file.is_file())
        db1.to_hdf5(hdf5_file)

        self.assertTrue(hdf5_file.is_file())

        db2 = grid_db.GridDB.from_hdf5(hdf5_file)

        self.assertAttrsEqual(db1, db2, attrs=GRID_DB_ATTRS)
        self.assertEqual(db1.track_peristences.keys(),
                         db2.track_peristences.keys())
        for key, track1 in db1.track_peristences.items():
            track2 = db2.track_peristences[key]
            if track1 is None:
                self.assertIsNone(track2)
                continue
            self.assertAttrsEqual(track1, track2, attrs=PERSISTENCE_ATTRS)

    def test_from_to_hdf5_roundtrips_lazy(self):

        hdf5_file = self.tempdir / 'temp.h5'

        db1 = self.load_full_grid_db()

        self.assertFalse(hdf5_file.is_file())
        db1.to_hdf5(hdf5_file)

        self.assertTrue(hdf5_file.is_file())

        db2 = grid_db.GridDB.from_hdf5(hdf5_file, lazy=True)

        # By default, nothing is loaded
        for attr in GRID_DB_ATTRS:
            if attr in ('time_scale', 'space_scale'):
                continue
            self.assertTrue(hasattr(getattr(db2, attr), 'load'))

        # Force everything to load
        db2.load(GRID_DB_ATTRS)

        # Now everything should be equal
        self.assertAttrsEqual(db1, db2, attrs=GRID_DB_ATTRS)

        # Have to handle track persistence specially
        self.assertTrue(hasattr(db2.track_peristences, 'load'))
        db2.load('track_peristences')
        self.assertEqual(db1.track_peristences.keys(),
                         db2.track_peristences.keys())
        for key, track1 in db1.track_peristences.items():
            track2 = db2.track_peristences[key]
            if track1 is None:
                self.assertIsNone(track2)
                continue
            self.assertAttrsEqual(track1, track2, attrs=PERSISTENCE_ATTRS)

    def test_from_to_hdf5_subgroup_roundtrips(self):

        hdf5_file = self.tempdir / 'temp.h5'
        db = h5py.File(str(hdf5_file), 'w')
        grp1 = db.create_group('grid1')

        db1 = self.load_full_grid_db()
        db1.to_hdf5(grp1)

        self.assertTrue(hdf5_file.is_file())

        db.close()
        db = h5py.File(str(hdf5_file), 'r')

        db2 = grid_db.GridDB.from_hdf5(db['grid1'])

        self.assertAttrsEqual(db1, db2, attrs=GRID_DB_ATTRS)
        self.assertEqual(db1.track_peristences.keys(),
                         db2.track_peristences.keys())
        for key, track1 in db1.track_peristences.items():
            track2 = db2.track_peristences[key]
            if track1 is None:
                self.assertIsNone(track2)
                continue
            self.assertAttrsEqual(track1, track2, attrs=PERSISTENCE_ATTRS)

    def test_get_timepoint_values(self):

        db = self.load_full_grid_db()

        radius, velocity = db.get_timepoint_values(5, 'timepoint_warp_radius', 'local_velocity_mesh')

        exp_radius = np.array([1.00134514, 1.00120612, 0.97468799, 0.99925559, 0.51530565, 0.21895842])
        exp_velocity = np.array([np.nan, 0.79599972, 0.84852814, 0.84852814, 0.84852814, 0.84852814])

        self.assertEqual(radius.shape, velocity.shape)

        np.testing.assert_almost_equal(radius, exp_radius, decimal=4)
        np.testing.assert_almost_equal(velocity, exp_velocity, decimal=4)

    def test_get_timepoint_range_with_bounds(self):

        db = self.load_full_grid_db()

        res = db.get_timepoint_range()
        exp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.assertEqual(res, exp)

        db.min_timepoint = 3

        res = db.get_timepoint_range()
        exp = [3, 4, 5, 6, 7, 8, 9, 10]

        self.assertEqual(res, exp)

        db.max_timepoint = 8

        res = db.get_timepoint_range()
        exp = [3, 4, 5, 6, 7, 8]

        self.assertEqual(res, exp)

    def test_get_flattened_values(self):

        db = self.load_full_grid_db()

        radius, velocity = db.get_flattened_values('timepoint_warp_radius', 'local_velocity_mesh')

        exp_radius = np.array([np.nan, np.nan, np.nan, 1.0016676, 1.00000043, 0.98487195,
                               1.00134506, 1.00120612, 0.97468778, 0.99925555, 0.51530555, 0.21895845,
                               1.00134514, 1.00120612, 0.97468799, 0.99925559, 0.51530565, 0.21895842,
                               1.001977, 1.00000006, 0.99280825, 0.13994261, 0.53195755, 1.001977,
                               1.00000006, 0.99280825, 0.13994261, 0.53195755, 1.00252609, 0.99449928,
                               0.32051666, 1.00042521, 1.00030128, 0.99362276, 0.52814937, 1.0001087,
                               np.nan])
        exp_velocity = np.array([0.84852814, 0.84852814, 0.80435676, 0.84852814, 0.80435676, 0.84852814,
                                 0.84852814, 0.80435676, 0.84852814, 0.84852814, 0.84852814, 0.84852814,
                                 np.nan, 0.79599972, 0.84852814, 0.84852814, 0.84852814, 0.84852814,
                                 0.75368837, 0.84852814, 0.84852814, 0.84852814, 0.84852814, 0.7025816,
                                 np.nan, 0.84852814, 0.84852814, 0.84852814, 0.67953145, 0.84852814,
                                 0.84852814, 0.84852814, 0.66575476, np.nan, np.nan, np.nan, np.nan])

        self.assertEqual(radius.shape, velocity.shape)

        np.testing.assert_almost_equal(radius, exp_radius, decimal=4)
        np.testing.assert_almost_equal(velocity, exp_velocity, decimal=4)

    def test_add_single_track(self):

        db = grid_db.GridDB(processes=1, time_scale=2.5, space_scale=1.5)
        db.add_track(LINK1)

        # time index to x, y (in pixels)
        exp_timepoint_coords = {
            1: [(6.1, 6.2)],
            2: [(7.1, 7.2)],
            3: [(8.1, 8.2)],
            4: [(9.1, 9.2)],
            5: [(10.1, 10.2)],
        }
        # time index to x, y, t (in um, um, mins respectively)
        exp_timepoint_real_coords = {
            1: [(9.15, 9.3, 2.5)],
            2: [(10.65, 10.8, 5.0)],
            3: [(12.15, 12.3, 7.5)],
            4: [(13.65, 13.8, 10.0)],
            5: [(15.15, 15.3, 12.5)],
        }
        # track index to coordinate indices in each timestep
        exp_track_links = {
            0: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        }
        # Map of timepoints to converting indices
        exp_timepoint_links = {
            (1, 2): {0: 0},
            (2, 3): {0: 0},
            (3, 4): {0: 0},
            (4, 5): {0: 0},
        }

        self.assertDictTuplesEqual(db.timepoint_coords, exp_timepoint_coords)
        self.assertDictTuplesEqual(db.timepoint_real_coords, exp_timepoint_real_coords)

        self.assertEqual(db.track_links, exp_track_links)
        self.assertEqual(db.timepoint_links, exp_timepoint_links)

    def test_add_two_tracks(self):

        db = grid_db.GridDB(processes=1, time_scale=2.5, space_scale=1.5)
        db.add_track(LINK1)
        db.add_track(LINK2)

        # time index to x, y (in pixels)
        exp_timepoint_coords = {
            1: [(6.1, 6.2)],
            2: [(7.1, 7.2), (6.3, 1.4)],
            3: [(8.1, 8.2), (7.3, 2.4)],
            4: [(9.1, 9.2), (8.3, 3.4)],
            5: [(10.1, 10.2), (9.3, 4.4)],
            6: [(10.3, 5.4)],
            7: [(11.3, 6.4)],
            8: [(12.3, 6.4)],
            9: [(13.3, 5.4)],
            10: [(14.3, 3.4)],
        }

        # time index to x, y, t (in um, um, mins respectively)
        exp_timepoint_real_coords = {
            1: [(9.15, 9.3, 2.5)],
            2: [(10.65, 10.8, 5.0), (9.45, 2.1, 5.0)],
            3: [(12.15, 12.3, 7.5), (10.95, 3.6, 7.5)],
            4: [(13.65, 13.8, 10.0), (12.45, 5.1, 10.0)],
            5: [(15.15, 15.3, 12.5), (13.95, 6.6, 12.5)],
            6: [(15.45, 8.1, 15.0)],
            7: [(16.95, 9.6, 17.5)],
            8: [(18.45, 9.6, 20.0)],
            9: [(19.95, 8.1, 22.5)],
            10: [(21.45, 5.1, 25.0)],
        }
        # track index to coordinate indices in each timestep
        exp_track_links = {
            0: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            1: {2: 1, 3: 1, 4: 1, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
        }
        # Map of timepoints to converting indices
        exp_timepoint_links = {
            (1, 2): {0: 0},
            (2, 3): {0: 0, 1: 1},
            (3, 4): {0: 0, 1: 1},
            (4, 5): {0: 0, 1: 1},
            (5, 6): {1: 0},
            (6, 7): {0: 0},
            (7, 8): {0: 0},
            (8, 9): {0: 0},
            (9, 10): {0: 0},
        }

        self.assertDictTuplesEqual(db.timepoint_coords, exp_timepoint_coords)
        self.assertDictTuplesEqual(db.timepoint_real_coords, exp_timepoint_real_coords)

        self.assertEqual(db.track_links, exp_track_links)
        self.assertEqual(db.timepoint_links, exp_timepoint_links)
