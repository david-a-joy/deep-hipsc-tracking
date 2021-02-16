#!/usr/bin/env python3

# Standard lib
import unittest

# 3rd party
import numpy as np

# Our own imports
from deep_hipsc_tracking.tracking import tracking_pipeline, Link
from deep_hipsc_tracking.utils import save_point_csvfile

from ..helpers import FileSystemTestCase

# Tests


class TestLinkNearbyTracks(unittest.TestCase):

    def test_links_closer_of_two_tracks_in_time(self):

        tt1 = np.array([1, 3, 5, 7, 9])
        xx1 = np.array([0, 1, 2, 3, 4])
        yy1 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        track1 = Link.from_arrays(tt1, xx1, yy1)

        tt2 = np.array([1, 3, 5, 7, 9])
        xx2 = np.array([0, 1, 2, 3, 4])
        yy2 = np.array([1, 2, 3, 4, 5])
        track2 = Link.from_arrays(tt2, xx2, yy2)

        tt3 = np.array([11, 13, 15, 17, 19])
        xx3 = np.array([5, 6, 7, 8, 9])
        yy3 = np.array([6, 7, 8, 9, 10])
        track3 = Link.from_arrays(tt3, xx3, yy3)

        res = tracking_pipeline.link_nearby_tracks(
            [track1, track2, track3],
            max_track_lag=2.1,
            max_link_dist=2,
            min_track_len=5,
            max_relink_attempts=10,
        )

        exp = [track2, Link.join(track1, track3)]

        self.assertEqual(res, exp)

    def test_links_closer_of_two_tracks_in_space(self):

        tt1 = np.array([1, 3, 5, 7, 9])
        xx1 = np.array([0, 1, 2, 3, 4])
        yy1 = np.array([1, 2, 3, 4, 5])
        track1 = Link.from_arrays(tt1, xx1, yy1)

        tt2 = np.array([1.5, 3.5, 5.5, 7.5, 9.5])
        xx2 = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        yy2 = np.array([1, 2, 3, 4, 5])
        track2 = Link.from_arrays(tt2, xx2, yy2)

        tt3 = np.array([11, 13, 15, 17, 19])
        xx3 = np.array([5, 6, 7, 8, 9])
        yy3 = np.array([6, 7, 8, 9, 10])
        track3 = Link.from_arrays(tt3, xx3, yy3)

        res = tracking_pipeline.link_nearby_tracks(
            [track1, track2, track3],
            max_track_lag=2.1,
            max_link_dist=2,
            min_track_len=5,
        )

        exp = [track1, Link.join(track2, track3)]

        self.assertEqual(res, exp)

    def test_links_two_close_tracks(self):

        tt1 = np.array([1, 3, 5, 7, 9])
        xx1 = np.array([0, 1, 2, 3, 4])
        yy1 = np.array([1, 2, 3, 4, 5])
        track1 = Link.from_arrays(tt1, xx1, yy1)

        tt2 = np.array([11, 13, 15, 17, 19])
        xx2 = np.array([5, 6, 7, 8, 9])
        yy2 = np.array([6, 7, 8, 9, 10])
        track2 = Link.from_arrays(tt2, xx2, yy2)

        res = tracking_pipeline.link_nearby_tracks(
            [track1, track2],
            max_track_lag=2.1,
            max_link_dist=2,
            min_track_len=5,
        )

        exp = [Link.join(track1, track2)]

        self.assertEqual(res, exp)

    def test_doesnt_link_far_tracks_in_space(self):

        tt1 = np.array([1, 3, 5, 7, 9])
        xx1 = np.array([0, 1, 2, 3, 4])
        yy1 = np.array([1, 2, 3, 4, 5])
        track1 = Link.from_arrays(tt1, xx1, yy1)

        tt2 = np.array([11, 13, 15, 17, 19])
        xx2 = np.array([5, 6, 7, 8, 9])
        yy2 = np.array([6, 7, 8, 9, 10])
        track2 = Link.from_arrays(tt2, xx2, yy2)

        res = tracking_pipeline.link_nearby_tracks(
            [track1, track2],
            max_track_lag=2.1,
            max_link_dist=1,
            min_track_len=5,
        )

        exp = [track1, track2]

        self.assertEqual(res, exp)

    def test_doesnt_link_far_tracks_in_time(self):

        tt1 = np.array([1, 3, 5, 7, 9])
        xx1 = np.array([0, 1, 2, 3, 4])
        yy1 = np.array([1, 2, 3, 4, 5])
        track1 = Link.from_arrays(tt1, xx1, yy1)

        tt2 = np.array([11, 13, 15, 17, 19])
        xx2 = np.array([5, 6, 7, 8, 9])
        yy2 = np.array([6, 7, 8, 9, 10])
        track2 = Link.from_arrays(tt2, xx2, yy2)

        res = tracking_pipeline.link_nearby_tracks(
            [track1, track2],
            max_track_lag=1,
            max_link_dist=2,
            min_track_len=5,
        )

        exp = [track1, track2]

        self.assertEqual(res, exp)

    def test_doesnt_link_temporally_overlaping_tracks(self):

        tt1 = np.array([1, 3, 5, 7, 9])
        xx1 = np.array([0, 1, 2, 3, 4])
        yy1 = np.array([1, 2, 3, 4, 5])
        track1 = Link.from_arrays(tt1, xx1, yy1)

        tt2 = np.array([9, 11, 13, 15, 17])
        xx2 = np.array([5, 6, 7, 8, 9])
        yy2 = np.array([6, 7, 8, 9, 10])
        track2 = Link.from_arrays(tt2, xx2, yy2)

        res = tracking_pipeline.link_nearby_tracks(
            [track1, track2],
            max_track_lag=1,
            max_link_dist=2,
            min_track_len=5,
        )

        exp = [track1, track2]

        self.assertEqual(res, exp)


class TestFilterByLinkDist(unittest.TestCase):

    def test_filters_simple_tracks_no_motion(self):

        tracks = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
        ]
        res = tracking_pipeline.filter_by_link_dist(tracks, max_link_dist=1)

        exp = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
        ]
        self.assertEqual(len(res), len(exp))
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r[2], e[2])

    def test_filters_simple_tracks_some_motion(self):

        tracks = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1.1, 2], [2.1, 3], [3.1, 4]])),
            (None, None, np.array([[1.2, 2], [2.2, 3], [3.2, 4]])),
        ]
        res = tracking_pipeline.filter_by_link_dist(tracks, max_link_dist=1)

        exp = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1.1, 2], [2.1, 3], [3.1, 4]])),
            (None, None, np.array([[1.2, 2], [2.2, 3], [3.2, 4]])),
        ]
        self.assertEqual(len(res), len(exp))
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r[2], e[2])

    def test_filters_simple_tracks_too_much_motion(self):

        tracks = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[2.1, 2], [3.1, 3], [3.1, 4]])),
            (None, None, np.array([[3.2, 2], [4.2, 3], [3.2, 4]])),
        ]
        res = tracking_pipeline.filter_by_link_dist(tracks, max_link_dist=1)

        exp = [
            (None, None, np.array([[3, 4]])),
            (None, None, np.array([[3.1, 4]])),
            (None, None, np.array([[3.2, 4]])),
        ]
        self.assertEqual(len(res), len(exp))
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r[2], e[2])

    def test_filters_with_empty_detections_in_middle(self):

        tracks = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.zeros((0, 2))),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
        ]
        res = tracking_pipeline.filter_by_link_dist(tracks, max_link_dist=1)

        exp = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.zeros((0, 2))),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
        ]
        self.assertEqual(len(res), len(exp))
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r[2], e[2])

    def test_filters_with_empty_detections_at_start(self):

        tracks = [
            (None, None, np.zeros((0, 2))),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
        ]
        res = tracking_pipeline.filter_by_link_dist(tracks, max_link_dist=1)

        exp = [
            (None, None, np.zeros((0, 2))),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
        ]
        self.assertEqual(len(res), len(exp))
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r[2], e[2])

    def test_filters_with_empty_detections_at_end(self):

        tracks = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.zeros((0, 2))),
        ]
        res = tracking_pipeline.filter_by_link_dist(tracks, max_link_dist=1)

        exp = [
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.array([[1, 2], [2, 3], [3, 4]])),
            (None, None, np.zeros((0, 2))),
        ]
        self.assertEqual(len(res), len(exp))
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r[2], e[2])


class TestLoadTrack(FileSystemTestCase):

    def test_loads_csv_file_no_image_no_mask(self):

        csvfile = self.tempdir / 'test.csv'
        cx = np.array([0.0, 0.0, 0.5, 1.0])
        cy = np.array([0.0, 0.1, 0.4, 0.9])
        cv = np.array([0.09, 0.1, 0.3, 0.8])

        save_point_csvfile(csvfile, cx, cy, cv)

        exp_img = np.zeros((1000, 1000))
        exp_x = np.array([0, 500, 1000])
        exp_y = np.array([900, 600, 100])

        res = tracking_pipeline.load_track(csvfile, imagefile=None,
                                           min_point_activation=0.1)

        self.assertEqual(len(res), 3)

        np.testing.assert_almost_equal(res[0], exp_img)
        np.testing.assert_almost_equal(res[1], exp_x)
        np.testing.assert_almost_equal(res[2], exp_y)
