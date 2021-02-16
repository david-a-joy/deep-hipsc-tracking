""" Tests for the model utilities """

import unittest

import numpy as np

from deep_hipsc_tracking.model import utils

# Tests


class TestConvertPointsToMask(unittest.TestCase):

    def test_convert_to_points_unnormalized(self):

        img = np.random.rand(64, 64)
        points = [
            (0, 0),
            (16, 0),
            (32, 32),
            (0, 48),
            (63, 63),
        ]

        mask = utils.convert_points_to_mask(img, points,
                                            points_normalized=False,
                                            dot_radius=1,
                                            mask_type='points')

        exp_mask = np.zeros((64, 64))
        exp_mask[63, 63] = 255
        exp_mask[0, 16] = 255
        exp_mask[32, 32] = 255
        exp_mask[48, 0] = 255
        exp_mask[0, 0] = 255

        np.testing.assert_almost_equal(mask, exp_mask)

    def test_convert_to_points_normalized(self):

        img = np.random.rand(64, 64)
        points = [
            (0.0, 0.01),
            (0.0, 0.75),
            (0.75, 0.01),
            (0.5, 0.5),
            (0.99, 0.01),
        ]

        mask = utils.convert_points_to_mask(img, points,
                                            points_normalized=True,
                                            dot_radius=1,
                                            mask_type='points')

        exp_mask = np.zeros((64, 64))
        exp_mask[63, 63] = 255
        exp_mask[32, 32] = 255
        exp_mask[16, 0] = 255
        exp_mask[63, 48] = 255
        exp_mask[63, 0] = 255

        np.testing.assert_almost_equal(mask, exp_mask)
