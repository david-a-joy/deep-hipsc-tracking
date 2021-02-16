""" Test the tracking utilities """

# Standard lib
import unittest

# 3rd party
import numpy as np

# Our own imports
from deep_hipsc_tracking import tracking


class TestCorrelateArrays(unittest.TestCase):

    def test_matches_identical_centered_complete(self):

        src_img = np.random.rand(129, 130)
        src_count = np.ones_like(src_img, dtype=np.uint32)
        match_img = src_img[1:-1, 1:-1]

        search_step = 1  # 1x1 search window
        blend_x = -1  # Complete matching in x
        blend_y = -1  # Complete matching in y

        # Correct start window
        yst, yed = 1, 128
        xst, xed = 1, 129
        rows, cols = match_img.shape
        comp_rows, comp_cols = src_img.shape

        assert (yed - yst) == rows
        assert (xed - xst) == cols

        res_y, res_x = tracking.correlate_arrays(
            src_img, src_count, match_img,
            search_step, blend_x, blend_y,
            xst, xed, yst, yed,
            rows, cols, comp_rows, comp_cols)

        # best_yst1, best_yed1, best_yst2, best_yed2
        exp_y = (1, 128, 0, 127)
        # best_xst1, best_xed1, best_xst2, best_xed2
        exp_x = (1, 129, 0, 128)

        self.assertEqual(res_y, exp_y)
        self.assertEqual(res_x, exp_x)

        np.testing.assert_almost_equal(src_img[res_y[0]:res_y[1], res_x[0]:res_x[1]],
                                       match_img[res_y[2]:res_y[3], res_x[2]:res_x[3]])

    def test_matches_identical_centered_partial(self):

        src_img = np.random.rand(129, 130)
        src_count = np.ones_like(src_img, dtype=np.uint32)
        match_img = src_img[1:-1, 1:-1]

        search_step = 1
        blend_x = 5  # Only match the top 5 columns in x
        blend_y = 5  # Only match the top 5 rows in y

        # Correct start window
        yst, yed = 1, 128
        xst, xed = 1, 129
        rows, cols = match_img.shape
        comp_rows, comp_cols = src_img.shape

        assert (yed - yst) == rows
        assert (xed - xst) == cols

        res_y, res_x = tracking.correlate_arrays(
            src_img, src_count, match_img,
            search_step, blend_x, blend_y,
            xst, xed, yst, yed,
            rows, cols, comp_rows, comp_cols)

        # best_yst1, best_yed1, best_yst2, best_yed2
        exp_y = (1, 128, 0, 127)
        # best_xst1, best_xed1, best_xst2, best_xed2
        exp_x = (1, 129, 0, 128)

        self.assertEqual(res_y, exp_y)
        self.assertEqual(res_x, exp_x)

        np.testing.assert_almost_equal(src_img[res_y[0]:res_y[1], res_x[0]:res_x[1]],
                                       match_img[res_y[2]:res_y[3], res_x[2]:res_x[3]])

    def test_matches_identical_uncentered_partial(self):

        src_img = np.random.rand(129, 130)
        src_count = np.ones_like(src_img, dtype=np.uint32)
        match_img = src_img[2:-2, 2:-2]

        search_step = 1
        blend_x = 5
        blend_y = 5

        # Inorrect start window
        yst, yed = 1, 126
        xst, xed = 1, 127
        rows, cols = match_img.shape
        comp_rows, comp_cols = src_img.shape

        assert (yed - yst) == rows
        assert (xed - xst) == cols

        res_y, res_x = tracking.correlate_arrays(
            src_img, src_count, match_img,
            search_step, blend_x, blend_y,
            xst, xed, yst, yed,
            rows, cols, comp_rows, comp_cols)

        # best_yst1, best_yed1, best_yst2, best_yed2
        exp_y = (2, 127, 0, 125)
        # best_xst1, best_xed1, best_xst2, best_xed2
        exp_x = (2, 128, 0, 126)

        self.assertEqual(res_y, exp_y)
        self.assertEqual(res_x, exp_x)

        np.testing.assert_almost_equal(src_img[res_y[0]:res_y[1], res_x[0]:res_x[1]],
                                       match_img[res_y[2]:res_y[3], res_x[2]:res_x[3]])

    def test_matches_identical_left_right(self):

        src_img = np.random.rand(256, 130)
        src_count = np.zeros_like(src_img, dtype=np.uint32)

        # Only take the first half of the source image
        src_count[0:128, 0:130] = 1

        # Match with the second half of the image
        match_img = src_img[120:247, 1:-1]

        search_step = 10
        blend_x = 10
        blend_y = 10

        yst, yed = 116, 243
        xst, xed = 0, 128
        rows, cols = match_img.shape
        comp_rows, comp_cols = src_img.shape

        assert (yed - yst) == rows
        assert (xed - xst) == cols

        res_y, res_x = tracking.correlate_arrays(
            src_img, src_count, match_img,
            search_step, blend_x, blend_y,
            xst, xed, yst, yed,
            rows, cols, comp_rows, comp_cols)

        # best_yst1, best_yed1, best_yst2, best_yed2
        exp_y = (120, 247, 0, 127)
        # best_xst1, best_xed1, best_xst2, best_xed2
        exp_x = (1, 129, 0, 128)

        self.assertEqual(res_y, exp_y)
        self.assertEqual(res_x, exp_x)


class TestSmoothVelocity(unittest.TestCase):

    def test_linear_smooth_with_resample(self):

        tt = np.linspace(1, 10, 10)
        xx = np.linspace(-10, 10, tt.shape[0])
        yy = np.linspace(-5, 15, tt.shape[0])

        sm_tt, sm_xx, sm_yy = tracking.smooth_velocity(
            tt, xx, yy,
            resample_factor=10,
            interp_points=7,
            smooth_points=0)

        exp_tt = np.linspace(1, 10, 100)
        exp_xx = np.linspace(-10, 10, 100)
        exp_yy = np.linspace(-5, 15, 100)

        np.testing.assert_almost_equal(exp_tt, sm_tt)
        np.testing.assert_almost_equal(exp_xx, sm_xx)
        np.testing.assert_almost_equal(exp_yy, sm_yy)

    def test_nonlinear_smooth_no_resample(self):

        tt = np.linspace(1, 10, 5)
        xx = np.array([0, 2, 1, 3, 2])
        yy = np.array([0, 3, 2, 5, 4])

        sm_tt, sm_xx, sm_yy = tracking.smooth_velocity(
            tt, xx, yy,
            resample_factor=2,
            interp_points=7,
            smooth_points=0)

        exp_tt = np.linspace(1, 10, 10)
        exp_xx = np.array([0.6, 0.8222222, 1.0444444, 1.2666667, 1.4888889, 1.7111111,
                           1.9333333, 2.1555556, 2.3777778, 2.6])
        exp_yy = np.array([0.8, 1.2444444, 1.6888889, 2.1333333, 2.5777778,
                           3.0222222, 3.4666667, 3.9111111, 4.3555556, 4.8])

        np.testing.assert_almost_equal(exp_tt, sm_tt)
        np.testing.assert_almost_equal(exp_xx, sm_xx)
        np.testing.assert_almost_equal(exp_yy, sm_yy)

    def test_nonlinear_smooth_with_resample(self):

        tt = np.linspace(1, 10, 5)
        xx = np.array([0, 2, 1, 3, 2])
        yy = np.array([0, 3, 2, 5, 4])

        sm_tt, sm_xx, sm_yy = tracking.smooth_velocity(
            tt, xx, yy,
            resample_factor=2,
            interp_points=7,
            smooth_points=2)

        exp_tt = np.linspace(1, 10, 10)
        exp_xx = np.array([0.6555556, 0.7666667, 0.9333333, 1.1555556, 1.3777778, 1.6,
                           1.8222222, 2.0444444, 2.2666667, 2.4333333])
        exp_yy = np.array([0.9111111, 1.1333333, 1.4666667, 1.9111111, 2.3555556, 2.8,
                           3.2444444, 3.6888889, 4.1333333, 4.4666667])

        np.testing.assert_almost_equal(exp_tt, sm_tt)
        np.testing.assert_almost_equal(exp_xx, sm_xx)
        np.testing.assert_almost_equal(exp_yy, sm_yy)
