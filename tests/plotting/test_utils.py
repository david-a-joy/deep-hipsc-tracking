#!/usr/bin/env python3

# Imports
import unittest

# 3rd party
import numpy as np

from scipy.integrate import simps

# Our own imports
from deep_hipsc_tracking.plotting import utils

# Tests


class TestGetHistogram(unittest.TestCase):

    def test_counts_kde_equal_area_counts(self):

        raw_data = np.array([
            0, 0, 0,
            1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4,
        ])
        bins_x, bins_y, hist_x, hist_y = utils.get_histogram(
            raw_data, bins=4, kernel_samples=100)

        exp_bins_x = np.array([0, 1, 2, 3, 4])
        exp_bins_y = np.array([3, 2, 4, 5])  # Last value gets eaten by the final bin

        np.testing.assert_allclose(bins_x, exp_bins_x)
        np.testing.assert_allclose(bins_y, exp_bins_y)

        self.assertEqual(hist_x.shape[0], 100)
        self.assertEqual(hist_y.shape[0], 100)

        bin_area = np.sum((exp_bins_x[1:] - exp_bins_x[:-1])*exp_bins_y)
        self.assertEqual(bin_area, raw_data.shape[0])

        kde_area = simps(hist_y, hist_x)
        np.testing.assert_allclose(kde_area, raw_data.shape[0])

    def test_counts_kde_equal_area_norm(self):

        raw_data = np.array([
            0, 0, 0,
            1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
            4,
        ])
        bins_x, bins_y, hist_x, hist_y = utils.get_histogram(
            raw_data, bins=4, kernel_samples=100, normalize=True)

        exp_bins_x = np.array([0, 1, 2, 3, 4])
        exp_bins_y = np.array([3, 2, 4, 5]) / 14  # Last value gets eaten by the final bin

        np.testing.assert_allclose(bins_x, exp_bins_x)
        np.testing.assert_allclose(bins_y, exp_bins_y)

        self.assertEqual(hist_x.shape[0], 100)
        self.assertEqual(hist_y.shape[0], 100)

        bin_area = np.sum((exp_bins_x[1:] - exp_bins_x[:-1])*exp_bins_y)
        self.assertEqual(bin_area, 1.0)

        kde_area = simps(hist_y, hist_x)
        np.testing.assert_allclose(kde_area, 1.0)


class TestBootstrapCI(unittest.TestCase):

    def test_calculates_ci_correctly_1d_small_samples(self):

        data = np.arange(100)

        low, high = utils.bootstrap_ci(data, random_seed=12345, n_samples=50)

        self.assertGreater(low, 40)
        self.assertLess(high, 60)

    def test_calculates_ci_correctly_1d(self):

        data = np.arange(100)

        low, high = utils.bootstrap_ci(data, random_seed=12345)

        self.assertGreater(low, 40)
        self.assertLess(high, 60)

    def test_calculates_ci_correctly_2d(self):

        data = np.stack([
            np.arange(100),
            np.arange(100, 200),
            np.arange(200, 300),
        ], axis=1)

        low, high = utils.bootstrap_ci(data, random_seed=12345)

        np.testing.assert_array_less(high, np.array([60, 160, 260]))
        np.testing.assert_array_less(np.array([40, 140, 240]), low)

    def test_calculates_ci_correctly_2d_different_axis(self):

        data = np.stack([
            np.arange(100),
            np.arange(100, 200),
            np.arange(200, 300),
        ], axis=1)

        low, high = utils.bootstrap_ci(data.T, random_seed=12345, axis=1)

        np.testing.assert_array_less(high, np.array([60, 160, 260]))
        np.testing.assert_array_less(np.array([40, 140, 240]), low)
