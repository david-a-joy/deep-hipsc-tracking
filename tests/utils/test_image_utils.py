#!/usr/bin/env python3

# Stdlib
import pathlib
import unittest

# 3rd party
import numpy as np

# Our own imports
from deep_hipsc_tracking.utils import image_utils
from .. import helpers

# Tests


class TestFixContrast(unittest.TestCase):

    def test_fixes_flat_image(self):

        img = np.zeros((64, 64))

        # If the image has no contrast, return zeros by default
        res = image_utils.fix_contrast(img)
        exp = np.zeros((64, 64))

        self.assertEqual(res.dtype, np.float64)
        np.testing.assert_almost_equal(res, exp)

        # If the image is smaller than max, return all ones
        res = image_utils.fix_contrast(img, cmax=-1)
        exp = np.ones((64, 64))
        np.testing.assert_almost_equal(res, exp)

        # If the image is lower than min, return 0s
        res = image_utils.fix_contrast(img, cmin=1)
        exp = np.zeros((64, 64))
        np.testing.assert_almost_equal(res, exp)

    def test_fixes_image_no_filter(self):

        img = np.zeros((64, 64))
        img[24:48, 24:48] = 3

        res = image_utils.fix_contrast(img)
        exp = np.zeros((64, 64))
        exp[24:48, 24:48] = 1

        self.assertEqual(res.dtype, np.float64)
        np.testing.assert_almost_equal(res, exp)

    def test_fixes_image_raw_with_clipping(self):

        img = np.zeros((64, 64))
        img[24:48, 24:48] = 5
        img[24:32, 24:32] = 3

        res = image_utils.fix_contrast(img, mode='raw')
        exp = np.zeros((64, 64))
        exp[24:48, 24:48] = 1
        exp[24:32, 24:32] = 3/5
        np.testing.assert_almost_equal(res, exp)

        res = image_utils.fix_contrast(img, mode='raw', cmin=1)
        exp = np.zeros((64, 64))
        exp[24:48, 24:48] = 1
        exp[24:32, 24:32] = 0.5
        np.testing.assert_almost_equal(res, exp)

        res = image_utils.fix_contrast(img, mode='raw', cmax=4)
        exp = np.zeros((64, 64))
        exp[24:48, 24:48] = 1
        exp[24:32, 24:32] = 3/4
        np.testing.assert_almost_equal(res, exp)

        res = image_utils.fix_contrast(img, mode='raw', cmax=4, cmin=1)
        exp = np.zeros((64, 64))
        exp[24:48, 24:48] = 1
        exp[24:32, 24:32] = 2/3
        np.testing.assert_almost_equal(res, exp)

        res = image_utils.fix_contrast(img, mode='raw', cmax=3, cmin=1)
        exp = np.zeros((64, 64))
        exp[24:48, 24:48] = 1
        np.testing.assert_almost_equal(res, exp)

    def test_fixes_image_with_filter(self):

        img = np.zeros((64, 64))
        img[24:48, 24:48] = 3
        img += np.random.rand(64, 64) * 1e-3

        res = image_utils.fix_contrast(img, filter_size=3)
        exp = np.zeros((64, 64))

        # We recover 1s in the middle, 0s on the edge
        exp[24:48, 24:48] = 1

        # We lose the corners because of smoothing
        exp[24, 24] = 0
        exp[24, 47] = 0
        exp[47, 24] = 0
        exp[47, 47] = 0

        self.assertEqual(res.dtype, np.float64)
        np.testing.assert_almost_equal(res, exp)


class TestLoadSavePoints(helpers.FileSystemTestCase):

    def test_save_load_no_filter(self):

        csvfile = self.tempdir / 'test.csv'

        cx = np.array([-0.1, 0, 0.5, 0.5, 1.0, 1.1])
        cy = np.array([-0.2, 0, 0.4, 0.6, 1.1, 1.0])
        cv = np.array([0, 1, 2, 3, 4, 5])

        self.assertFalse(csvfile.is_file())

        image_utils.save_point_csvfile(csvfile, cx, cy, cv)

        self.assertTrue(csvfile.is_file())

        res = image_utils.load_point_csvfile(csvfile)

        self.assertEqual(len(res), 3)

        rx, ry, rv = res
        np.testing.assert_almost_equal(rx, cx)
        np.testing.assert_almost_equal(ry, cy)
        np.testing.assert_almost_equal(rv, cv)

    def test_save_load_y_filter(self):

        csvfile = self.tempdir / 'test.csv'

        cx = np.array([-0.1, 0, 0.5, 0.5, 1.0, 1.1])
        cy = np.array([-0.2, 0, 0.4, 0.6, 1.1, 1.0])
        cv = np.array([0, 1, 2, 3, 4, 5])

        self.assertFalse(csvfile.is_file())

        image_utils.save_point_csvfile(csvfile, cx, cy, cv, ylim=(0, 1))

        self.assertTrue(csvfile.is_file())

        res = image_utils.load_point_csvfile(csvfile)

        self.assertEqual(len(res), 3)

        ex = np.array([0, 0.5, 0.5, 1.1])
        ey = np.array([0, 0.4, 0.6, 1.0])
        ev = np.array([1, 2, 3, 5])

        rx, ry, rv = res
        np.testing.assert_almost_equal(rx, ex)
        np.testing.assert_almost_equal(ry, ey)
        np.testing.assert_almost_equal(rv, ev)

    def test_save_load_x_filter(self):

        csvfile = self.tempdir / 'test.csv'

        cx = np.array([-0.1, 0, 0.5, 0.5, 1.0, 1.1])
        cy = np.array([-0.2, 0, 0.4, 0.6, 1.1, 1.0])
        cv = np.array([0, 1, 2, 3, 4, 5])

        self.assertFalse(csvfile.is_file())

        image_utils.save_point_csvfile(csvfile, cx, cy, cv, xlim=(0, 1))

        self.assertTrue(csvfile.is_file())

        res = image_utils.load_point_csvfile(csvfile)

        self.assertEqual(len(res), 3)

        ex = np.array([0, 0.5, 0.5, 1.0])
        ey = np.array([0, 0.4, 0.6, 1.1])
        ev = np.array([1, 2, 3, 4])

        rx, ry, rv = res
        np.testing.assert_almost_equal(rx, ex)
        np.testing.assert_almost_equal(ry, ey)
        np.testing.assert_almost_equal(rv, ev)

    def test_save_load_xy_filter(self):

        csvfile = self.tempdir / 'test.csv'

        cx = np.array([-0.1, 0, 0.5, 0.5, 1.0, 1.1])
        cy = np.array([-0.2, 0, 0.4, 0.6, 1.1, 1.0])
        cv = np.array([0, 1, 2, 3, 4, 5])

        self.assertFalse(csvfile.is_file())

        image_utils.save_point_csvfile(csvfile, cx, cy, cv,
                                       xlim=(0, 1), ylim=(0, 1))

        self.assertTrue(csvfile.is_file())

        res = image_utils.load_point_csvfile(csvfile)

        self.assertEqual(len(res), 3)

        ex = np.array([0, 0.5, 0.5])
        ey = np.array([0, 0.4, 0.6])
        ev = np.array([1, 2, 3])

        rx, ry, rv = res
        np.testing.assert_almost_equal(rx, ex)
        np.testing.assert_almost_equal(ry, ey)
        np.testing.assert_almost_equal(rv, ev)


class TestLoadImage(helpers.FileSystemTestCase):

    def test_load_2d_image(self):

        imgfile = helpers.DATADIR / 'test2d.tif'

        assert imgfile.is_file()

        img = image_utils.load_image(imgfile)

        assert img.shape == (260, 347)

    def test_load_3d_image_2d(self):

        imgfile = helpers.DATADIR / 'test3d.tif'

        assert imgfile.is_file()

        img = image_utils.load_image(imgfile, ctype='gray')

        assert img.shape == (260, 347)

    def test_load_3d_image_3d(self):

        imgfile = helpers.DATADIR / 'test3d.tif'

        assert imgfile.is_file()

        img = image_utils.load_image(imgfile, ctype='color')

        assert img.shape == (260, 347, 4)

    def test_load_save_image(self):

        img = np.random.rand(120, 120)

        imgfile = self.tempdir / 'test.png'

        image_utils.save_image(imgfile, img,
                               cmin=0, cmax=1.0)

        assert imgfile.is_file()

        res_img = image_utils.load_image(imgfile)/255.0

        assert res_img.ndim == 2
        assert res_img.shape == (120, 120, )

        np.testing.assert_almost_equal(img, res_img, decimal=2)

    def test_load_save_image_forced_rounding(self):

        img = np.random.rand(120, 120) * 2.0

        imgfile = self.tempdir / 'test.png'

        image_utils.save_image(imgfile, img,
                               cmin=0, cmax=1.0)

        assert imgfile.is_file()

        res_img = image_utils.load_image(imgfile)/255.0
        exp_img = img.copy()
        exp_img[img > 1.0] = 1.0

        assert res_img.ndim == 2
        assert res_img.shape == (120, 120, )

        np.testing.assert_almost_equal(exp_img, res_img, decimal=2)


class TestAlignTimepoints(unittest.TestCase):

    def test_align_no_data(self):

        agg_data = {'foo': []}
        agg_timepoints = []

        res = image_utils.align_timepoints(agg_data, agg_timepoints)
        exp = {'foo': np.array([])}

        self.assertEqual(set(res.keys()), set(exp.keys()))
        for key in res:
            np.testing.assert_almost_equal(res[key], exp[key])

    def test_align_one_record_no_gaps(self):

        agg_data = {'foo': [
            np.array([1, 2, 3, 4, 5]),
        ]}
        agg_timepoints = [np.array([1, 2, 3, 4, 5])]

        res = image_utils.align_timepoints(agg_data, agg_timepoints)
        exp = {'foo': np.array([
                [1, 2, 3, 4, 5],
            ]),
        }

        self.assertEqual(set(res.keys()), set(exp.keys()))
        for key in res:
            np.testing.assert_almost_equal(res[key], exp[key])

    def test_align_two_records_no_gaps(self):

        agg_data = {'foo': [
            np.array([1, 2, 3, 4, 5]),
            np.array([2, 4, 6, 8, 10]),
        ]}
        agg_timepoints = [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
        ]

        res = image_utils.align_timepoints(agg_data, agg_timepoints)
        exp = {'foo': np.array([
                [1, 2, 3, 4, 5],
                [2, 4, 6, 8, 10],
            ]),
        }

        self.assertEqual(set(res.keys()), set(exp.keys()))
        for key in res:
            np.testing.assert_almost_equal(res[key], exp[key])

    def test_align_several_records_with_gaps(self):

        agg_data = {'foo': [
            np.array([2, 3, 4, 5]),
            np.array([4, 6, 8, 10]),
            np.array([3, 6, 12, 15]),
        ]}
        agg_timepoints = [
            np.array([2, 3, 4, 5]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 4, 5]),
        ]

        res = image_utils.align_timepoints(agg_data, agg_timepoints)
        exp = {'foo': np.array([
                [np.nan, 2, 3, 4, 5],
                [4, 6, 8, 10, np.nan],
                [3, 6, np.nan, 12, 15],
            ]),
        }

        self.assertEqual(set(res.keys()), set(exp.keys()))
        for key in res:
            print(res[key])
            print(exp[key])
            np.testing.assert_almost_equal(res[key], exp[key])

    def test_align_several_records_with_gaps_all_lists_diff_lengths(self):

        agg_data = {'foo': [
            [2, 3, 4],
            [4, 6, 8, 10],
            [3, 6, 12, 15, 18],
        ]}
        agg_timepoints = [
            [2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 4, 5, 6],
        ]

        res = image_utils.align_timepoints(agg_data, agg_timepoints)
        exp = {'foo': np.array([
                [np.nan, 2, 3, 4, np.nan, np.nan],
                [4, 6, 8, 10, np.nan, np.nan],
                [3, 6, np.nan, 12, 15, 18],
            ]),
        }

        self.assertEqual(set(res.keys()), set(exp.keys()))
        for key in res:
            print(res[key])
            print(exp[key])
            np.testing.assert_almost_equal(res[key], exp[key])

    def test_handles_records_with_no_data(self):

        agg_data = {'foo': [
            np.array([]),
            np.array([4, 6, 8, 10]),
            np.array([3, 6, 12, 15]),
        ]}
        agg_timepoints = [
            np.array([]),
            np.array([1, 2, 3, 4]),
            np.array([1, 2, 4, 5]),
        ]

        res = image_utils.align_timepoints(agg_data, agg_timepoints)
        exp = {'foo': np.array([
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [4, 6, 8, 10, np.nan],
                [3, 6, np.nan, 12, 15],
            ]),
        }

        self.assertEqual(set(res.keys()), set(exp.keys()))
        for key in res:
            print(res[key])
            print(exp[key])
            np.testing.assert_almost_equal(res[key], exp[key])


class TestToJSONTypes(unittest.TestCase):

    def test_handles_arrays(self):

        data = np.array([1, 2, 3, 4])

        res = image_utils.to_json_types(data)
        exp = [1, 2, 3, 4]

        self.assertEqual(res, exp)

        data = np.array([0, 1, 0, 0], dtype=np.bool)

        res = image_utils.to_json_types(data)
        exp = [0, 1, 0, 0]

        self.assertEqual(res, exp)

        data = np.array([1.5, 2.5, 3.5])

        res = image_utils.to_json_types(data)
        exp = [1.5, 2.5, 3.5]

        self.assertEqual(res, exp)

    def test_handles_nested_tuple_dicts(self):

        data = [('foo', {'bar': np.array([1, 2, 3])}),
                ('bif', {'bof': np.array([4, 5, 6])})]

        res = image_utils.to_json_types(data)
        exp = [['foo', {'bar': [1, 2, 3]}],
               ['bif', {'bof': [4, 5, 6]}]]

        self.assertEqual(res, exp)

    def test_handles_dicts_with_str_keys(self):

        data = {1: 'foo', 2: 'bar'}

        res = image_utils.to_json_types(data)
        exp = {'1': 'foo', '2': 'bar'}

        self.assertEqual(res, exp)

    def test_handles_paths(self):

        data = ('foo.txt', pathlib.Path('bar.txt'))

        res = image_utils.to_json_types(data)
        exp = ['foo.txt', 'bar.txt']

        self.assertEqual(res, exp)


class TestMaskFromContours(unittest.TestCase):

    def test_roundtrip_mask(self):

        mask = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ])

        contours = image_utils.contours_from_mask(mask)
        self.assertEqual(len(contours), 2)

        res = image_utils.mask_from_contours(contours, mask.shape)

        np.testing.assert_almost_equal(mask, res)


class TestContoursFromMask(unittest.TestCase):

    def test_contours_with_no_mask(self):

        mask = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])

        res = image_utils.contours_from_mask(mask)

        self.assertEqual(res, [])

    def test_contours_with_centered_mask(self):

        mask = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        res = image_utils.contours_from_mask(mask)

        self.assertEqual(len(res), 1)

        exp = np.array([
            [3.0, 2.5],
            [2.0, 2.5],
            [1.5, 2.0],
            [1.0, 1.5],
            [0.5, 1.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [2.5, 1.0],
            [3.0, 1.5],
            [3.5, 2.0],
            [3.0, 2.5],
        ])
        np.testing.assert_almost_equal(res[0], exp)

    def test_contours_with_edge_mask(self):

        mask = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        res = image_utils.contours_from_mask(mask)

        self.assertEqual(len(res), 1)

        exp = np.array([
            [3.0, 2.5],
            [2.0, 2.5],
            [1.5, 2.0],
            [1.0, 1.5],
            [0.0, 1.5],
            [-0.5, 1.0],
            [-0.5, 0.0],
            [0.0, -0.5],
            [1.0, -0.5],
            [2.0, -0.5],
            [2.5, 0.0],
            [2.5, 1.0],
            [3.0, 1.5],
            [3.5, 2.0],
            [3.0, 2.5],
        ])

        np.testing.assert_almost_equal(res[0], exp)

    def test_contours_with_edge_mask_3d(self):

        mask = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        res = image_utils.contours_from_mask(mask[np.newaxis, :, :])

        self.assertEqual(len(res), 1)

        exp = np.array([
            [3.0, 2.5],
            [2.0, 2.5],
            [1.5, 2.0],
            [1.0, 1.5],
            [0.0, 1.5],
            [-0.5, 1.0],
            [-0.5, 0.0],
            [0.0, -0.5],
            [1.0, -0.5],
            [2.0, -0.5],
            [2.5, 0.0],
            [2.5, 1.0],
            [3.0, 1.5],
            [3.5, 2.0],
            [3.0, 2.5],
        ])

        np.testing.assert_almost_equal(res[0], exp)

        res = image_utils.contours_from_mask(mask[:, :, np.newaxis])

        np.testing.assert_almost_equal(res[0], exp)

    def test_contours_with_two_regions(self):

        mask = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ])

        res = image_utils.contours_from_mask(mask)
        self.assertEqual(len(res), 2)

        exp0 = np.array([
            [1.0, 1.5],
            [0.0, 1.5],
            [-0.5, 1.0],
            [-0.5, 0.0],
            [0.0, -0.5],
            [1.0, -0.5],
            [1.5, 0.0],
            [1.5, 1.0],
            [1.0, 1.5],
        ])
        exp1 = np.array([
            [4.0, 3.5],
            [3.0, 3.5],
            [2.5, 3.0],
            [2.5, 2.0],
            [3.0, 1.5],
            [4.0, 1.5],
            [4.5, 2.0],
            [4.5, 3.0],
            [4.0, 3.5],
        ])

        np.testing.assert_almost_equal(res[0], exp0)
        np.testing.assert_almost_equal(res[1], exp1)

    def test_contours_with_two_regions_tolerance(self):

        mask = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ])

        res = image_utils.contours_from_mask(mask, tolerance=0.5)
        self.assertEqual(len(res), 2)

        exp0 = np.array([
            [1., 1.5],
            [-0.5, 1.],
            [0., -0.5],
            [1.5, 0.],
            [1., 1.5],
        ])
        exp1 = np.array([
            [4., 3.5],
            [2.5, 3.],
            [3., 1.5],
            [4.5, 2.],
            [4., 3.5],
        ])

        np.testing.assert_almost_equal(res[0], exp0)
        np.testing.assert_almost_equal(res[1], exp1)

    def test_contours_max_only(self):

        mask = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 0, 2, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        res = image_utils.contours_from_mask(mask, min_level=None, max_level=1.5)
        self.assertEqual(len(res), 2)

        exp0 = np.array([
            [4.0, 3.5],
            [3.0, 3.5],
            [2.0, 3.5],
            [1.0, 3.5],
            [0.0, 3.5],
            [-0.5, 3.0],
            [-0.5, 2.0],
            [-0.5, 1.0],
            [-0.5, 0.0],
            [0.0, -0.5],
            [1.0, -0.5],
            [2.0, -0.5],
            [3.0, -0.5],
            [4.0, -0.5],
            [4.5, 0.0],
            [4.5, 1.0],
            [4.5, 2.0],
            [4.5, 3.0],
            [4.0, 3.5],
        ])
        exp1 = np.array([
            [2.5, 2.0],
            [2.5, 1.0],
            [2.0, 0.5],
            [1.5, 1.0],
            [1.5, 2.0],
            [2.0, 2.5],
            [2.5, 2.0],
        ])

        np.testing.assert_almost_equal(res[0], exp0)
        np.testing.assert_almost_equal(res[1], exp1)


class TestTrimZeros(unittest.TestCase):
    """ Trim all the zeros out of an array """

    def test_trim_2d(self):

        arr = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        exp = np.array([
            [1, 1, 0],
            [1, 0, 1],
        ])

        res = image_utils.trim_zeros(arr, padding=0)
        np.testing.assert_almost_equal(res, exp)

    def test_trim_2d_axis(self):

        arr = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ])

        exp = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
        ])

        res = image_utils.trim_zeros(arr, padding=0, axis=0)
        np.testing.assert_almost_equal(res, exp)

        exp = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ])

        res = image_utils.trim_zeros(arr, padding=0, axis=1)
        np.testing.assert_almost_equal(res, exp)

        exp = np.array([
            [1, 1, 0],
            [1, 0, 1],
        ])

        res = image_utils.trim_zeros(arr, padding=0, axis=(0, 1))
        np.testing.assert_almost_equal(res, exp)

    def test_trim_2d_symmetric_axis(self):

        arr = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])

        exp = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
        ])

        res = image_utils.trim_zeros(arr, symmetric_axis=0)
        np.testing.assert_almost_equal(res, exp)

        exp = np.array([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ])

        res = image_utils.trim_zeros(arr, symmetric_axis=1)
        np.testing.assert_almost_equal(res, exp)

        exp = np.array([
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ])

        res = image_utils.trim_zeros(arr, symmetric_axis=(0, 1))
        np.testing.assert_almost_equal(res, exp)

    def test_trim_2d_symmetric_axis_no_trim_needed(self):

        arr = np.array([
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        exp = np.array([
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        res = image_utils.trim_zeros(arr, symmetric_axis=0)
        np.testing.assert_almost_equal(res, exp)

        res = image_utils.trim_zeros(arr, symmetric_axis=1)
        np.testing.assert_almost_equal(res, exp)

        res = image_utils.trim_zeros(arr, symmetric_axis=(0, 1))
        np.testing.assert_almost_equal(res, exp)

    def test_trim_2d_erode(self):

        arr = np.array([
            [0, 0, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
        ])

        exp = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])

        res = image_utils.trim_zeros(arr, erode=1)
        np.testing.assert_almost_equal(res, exp)

    def test_trim_2d_erode_padding(self):

        arr = np.array([
            [0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
        ])

        exp = np.array([
            [0, 0, 1, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ])

        res = image_utils.trim_zeros(arr, erode=1, padding=1)
        np.testing.assert_almost_equal(res, exp)

    def test_trim_2d_pad1(self):

        arr = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
        ])

        exp = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
        ])

        res = image_utils.trim_zeros(arr, padding=1)
        np.testing.assert_almost_equal(res, exp)

    def test_trim_2d_pad2(self):

        arr = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
        ])

        exp = arr
        res = image_utils.trim_zeros(arr, padding=2)
        np.testing.assert_almost_equal(res, exp)

    def test_trim_3d_pad0(self):

        arr = np.zeros((4, 7, 5))
        arr[1:3, 1:6, 1] = np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0],
        ])
        arr[1:3, 1:6, 2] = np.array([
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
        ])
        arr[1:3, 1:6, 3] = np.array([
            [0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
        ])

        exp = arr[1:3, 1:6, 1:4]
        res = image_utils.trim_zeros(arr)
        np.testing.assert_almost_equal(res, exp)


if __name__ == '__main__':
    unittest.main()
