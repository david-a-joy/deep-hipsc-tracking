#!/usr/bin/env python3

# Imports

# Standard lib
import unittest

# Our own imports
from deep_hipsc_tracking.model import SingleCellDetector
from deep_hipsc_tracking.model import _keras


# Tests


@unittest.skipIf(_keras.keras is None, 'Install keras')
class TestSingleCellDetector(unittest.TestCase):

    def test_has_expected_models(self):

        res = set(SingleCellDetector.get_detectors())
        exp = {'countception', 'unet', 'fcrn_a', 'fcrn_b', 'residual_unet',
               'fcrn_a_wide', 'fcrn_b_wide', 'residual_unet_wide'}

        self.assertEqual(res, exp)

    def test_countception_params(self):

        model = SingleCellDetector(detector='countception')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'leaky_relu')
        self.assertEqual(model.img_rows, 256)
        self.assertEqual(model.img_cols, 256)
        self.assertEqual(model.img_colors, 1)

        self.assertEqual(model.total_rows, 225)
        self.assertEqual(model.total_cols, 225)

    def test_unet_params(self):

        model = SingleCellDetector(detector='unet')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'leaky_relu')
        self.assertEqual(model.img_rows, 256)
        self.assertEqual(model.img_cols, 256)
        self.assertEqual(model.img_colors, 1)

        self.assertEqual(model.total_rows, 68)
        self.assertEqual(model.total_cols, 68)

    def test_residual_unet_params(self):

        model = SingleCellDetector(detector='residual_unet')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'elu')
        self.assertEqual(model.img_rows, 100)
        self.assertEqual(model.img_cols, 100)
        self.assertEqual(model.img_colors, 1)

        self.assertEqual(model.total_rows, 100)
        self.assertEqual(model.total_cols, 100)

    def test_residual_unet_wide_params(self):

        model = SingleCellDetector(detector='residual_unet_wide')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'elu')
        self.assertEqual(model.img_rows, 256)
        self.assertEqual(model.img_cols, 256)
        self.assertEqual(model.img_colors, 1)

        self.assertEqual(model.total_rows, 256)
        self.assertEqual(model.total_cols, 256)

    def test_fcrn_a_params(self):

        model = SingleCellDetector(detector='fcrn_a')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'leaky_relu')
        self.assertEqual(model.img_rows, 100)
        self.assertEqual(model.img_cols, 100)
        self.assertEqual(model.img_colors, 1)

        self.assertEqual(model.total_rows, 100)
        self.assertEqual(model.total_cols, 100)

    def test_fcrn_a_wide_params(self):

        model = SingleCellDetector(detector='fcrn_a_wide')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'leaky_relu')
        self.assertEqual(model.img_rows, 256)
        self.assertEqual(model.img_cols, 256)
        self.assertEqual(model.img_colors, 1)

        self.assertEqual(model.total_rows, 256)
        self.assertEqual(model.total_cols, 256)

    def test_fcrn_b_params(self):

        model = SingleCellDetector(detector='fcrn_b')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'leaky_relu')
        self.assertEqual(model.img_rows, 100)
        self.assertEqual(model.img_cols, 100)
        self.assertEqual(model.img_colors, 1)

    def test_fcrn_b_wide_params(self):

        model = SingleCellDetector(detector='fcrn_b_wide')
        model.load_optimizer()
        model.make_detector()

        self.assertEqual(model.activation_function, 'leaky_relu')
        self.assertEqual(model.img_rows, 256)
        self.assertEqual(model.img_cols, 256)
        self.assertEqual(model.img_colors, 1)

        self.assertEqual(model.total_rows, 256)
        self.assertEqual(model.total_cols, 256)
