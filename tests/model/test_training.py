""" Tests for the training data loaders/parsers """

# Imports
import unittest
import pathlib

from deep_hipsc_tracking.model import training
from ..helpers import FileSystemTestCase

# Tests


class TestPairDetectorData(FileSystemTestCase):

    def test_pairs_single_set_no_filter(self):

        rootdir = self.tempdir
        d1 = rootdir / 'ai-upsample-peaks-residual_unet-run004' / 'ai-upsample-peaks-n25000'
        (d1 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f1 = d1 / '001cell_resp.png'
        f1.touch()

        res_pairs, num_detectors = training.pair_detector_data(rootdir, data_type='train')
        self.assertEqual(num_detectors, 1)

        # Channel, tile, timepoint, but only timepoint applies to training data
        exp_pairs = {
            (None, None, 1): [f1],
        }
        self.assertEqual(set(res_pairs), set(exp_pairs))
        for key in res_pairs:
            self.assertEqual(len(res_pairs[key]), num_detectors)
            self.assertEqual(res_pairs[key], exp_pairs[key])

    def test_pairs_two_sets_no_filter(self):

        rootdir = self.tempdir
        d1 = rootdir / 'ai-upsample-peaks-residual_unet-run004' / 'ai-upsample-peaks-n25000'
        (d1 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f1 = d1 / '001cell_resp.png'
        f1.touch()

        d2 = rootdir / 'ai-upsample-peaks-countception-run002' / 'ai-upsample-peaks-n50000'
        (d2 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f2 = d2 / '001cell_resp.png'
        f2.touch()

        res_pairs, num_detectors = training.pair_detector_data(rootdir, data_type='train')
        self.assertEqual(num_detectors, 2)

        # Channel, tile, timepoint, but only timepoint applies to training data
        exp_pairs = {
            (None, None, 1): [f1, f2],
        }
        self.assertEqual(set(res_pairs), set(exp_pairs))
        for key in res_pairs:
            self.assertEqual(len(res_pairs[key]), num_detectors)
            self.assertEqual(list(sorted(res_pairs[key])),
                             list(sorted(exp_pairs[key])))

    def test_pairs_two_sets_no_filter_different_dataset(self):

        rootdir = self.tempdir
        d1 = rootdir / 'ai-upsample-confocal-residual_unet-run004' / 'ai-upsample-confocal-n25000'
        (d1 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f1 = d1 / '001cell_resp.png'
        f1.touch()

        d2 = rootdir / 'ai-upsample-peaks-countception-run002' / 'ai-upsample-peaks-n50000'
        (d2 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f2 = d2 / '001cell_resp.png'
        f2.touch()

        d3 = rootdir / 'ai-upsample-confocal-countception-run002' / 'ai-upsample-confocal-n50000'
        (d3 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f3 = d3 / '001cell_resp.png'
        f3.touch()

        res_pairs, num_detectors = training.pair_detector_data(rootdir, data_type='train', training_set='confocal')
        self.assertEqual(num_detectors, 2)

        # Channel, tile, timepoint, but only timepoint applies to training data
        exp_pairs = {
            (None, None, 1): [f1, f3],
        }
        self.assertEqual(set(res_pairs), set(exp_pairs))
        for key in res_pairs:
            self.assertEqual(len(res_pairs[key]), num_detectors)
            self.assertEqual(list(sorted(res_pairs[key])),
                             list(sorted(exp_pairs[key])))

        res_pairs, num_detectors = training.pair_detector_data(rootdir, data_type='train', training_set='inverted')
        self.assertEqual(num_detectors, 1)

        # Channel, tile, timepoint, but only timepoint applies to training data
        exp_pairs = {
            (None, None, 1): [f2],
        }
        self.assertEqual(set(res_pairs), set(exp_pairs))
        for key in res_pairs:
            self.assertEqual(len(res_pairs[key]), num_detectors)
            self.assertEqual(res_pairs[key], exp_pairs[key])

    def test_pairs_two_sets_with_filter(self):

        rootdir = self.tempdir
        d1 = rootdir / 'ai-upsample-peaks-residual_unet-run004' / 'ai-upsample-peaks-n25000'
        (d1 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f1 = d1 / '001cell_resp.png'
        f1.touch()

        d2 = rootdir / 'ai-upsample-peaks-countception-run002' / 'ai-upsample-peaks-n50000'
        (d2 / 'snapshot').mkdir(exist_ok=True, parents=True)

        f2 = d2 / '001cell_resp.png'
        f2.touch()

        res_pairs, num_detectors = training.pair_detector_data(rootdir, data_type='train', detectors='countception-r2-50k')
        self.assertEqual(num_detectors, 1)

        # Channel, tile, timepoint, but only timepoint applies to training data
        exp_pairs = {
            (None, None, 1): [f2],
        }
        self.assertEqual(set(res_pairs), set(exp_pairs))
        for key in res_pairs:
            self.assertEqual(len(res_pairs[key]), num_detectors)
            self.assertEqual(res_pairs[key], exp_pairs[key])

    def test_pairs_two_sets_any_data(self):

        rootdir = self.tempdir
        d1 = rootdir / 'SingleCell-countception'
        f1 = d1 / 's01' / 's01t001cell_resp.png'
        f1.parent.mkdir(parents=True, exist_ok=True)
        f1.touch()
        f2 = d1 / 's01' / 's01t002cell_resp.png'
        f2.parent.mkdir(parents=True, exist_ok=True)
        f2.touch()
        f3 = d1 / 's01t002cell_resp.png'
        f3.parent.mkdir(parents=True, exist_ok=True)
        f3.touch()

        d2 = rootdir / 'SingleCell-unet'
        f4 = d2 / 's01' / 's01t001cell_resp.png'
        f4.parent.mkdir(parents=True, exist_ok=True)
        f4.touch()
        f5 = d2 / 's01' / 's01t002cell_resp.png'
        f5.parent.mkdir(parents=True, exist_ok=True)
        f5.touch()
        f6 = d2 / 's01t002cell_resp.png'
        f6.parent.mkdir(parents=True, exist_ok=True)
        f6.touch()

        res_pairs, num_detectors = training.pair_detector_data(
            rootdir, data_type='any', detectors=('countception', 'unet'))
        self.assertEqual(num_detectors, 2)

        # Channel, tile, timepoint, but only channel applies to "any" data
        exp_pairs = {
            (pathlib.Path('s01/s01t001cell'), None, None): [f1, f4],
            (pathlib.Path('s01/s01t002cell'), None, None): [f2, f5],
            (pathlib.Path('s01t002cell'), None, None): [f3, f6],
        }
        self.assertEqual(set(res_pairs), set(exp_pairs))
        for key in res_pairs:
            self.assertEqual(len(res_pairs[key]), num_detectors)
            self.assertEqual(len(exp_pairs[key]), num_detectors)
            self.assertEqual(set(res_pairs[key]), set(exp_pairs[key]))


class TestParseDetectors(unittest.TestCase):

    def test_parses_default(self):
        res = training.parse_detectors(None)
        self.assertIsNone(res)

    def test_parses_full_spec_real(self):

        res = training.parse_detectors('countception', data_type='real')
        exp = ['countception']

        self.assertEqual(res, exp)

    def test_parses_full_spec_training(self):

        res = training.parse_detectors('countception-run003-n50k', data_type='train')
        exp = [('countception', 3, 50000)]

        self.assertEqual(res, exp)

    def test_parses_multiple_specs_training(self):

        res = training.parse_detectors(['countception-run003-n50k', 'fcrn_a-run1-10m', 'bees-run2-50'], data_type='train')
        exp = [('countception', 3, 50000),
               ('fcrn_a', 1, 10000000),
               ('bees', 2, 50)]

        self.assertEqual(res, exp)
