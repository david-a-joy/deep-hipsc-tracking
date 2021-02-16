#!/usr/bin/env python3

# Imports

# Standard lib
import unittest
import pathlib

# 3rd party
import numpy as np

from PIL import Image

# Our own imports
from deep_hipsc_tracking.model import preproc
from deep_hipsc_tracking.model._preproc import composite_mask
from .. import helpers


# Helper Classes


class FakeDetector(object):
    """ Mock a classic one output detector """

    def __init__(self, predict=None):
        if predict is None:
            self.predict = self.predict_ones
        else:
            self.predict = predict

    def predict_ones(self, batch_slab, batch_size):
        return np.ones((batch_size, 1), dtype=np.float32)


class FakeConvDetector(object):
    """ Mock a convolutional detector """

    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.start_x = (in_shape[0] - out_shape[0])//2
        self.start_y = (in_shape[1] - out_shape[1])//2

        self.end_x = self.start_x + out_shape[0]
        self.end_y = self.start_y + out_shape[1]

    def predict(self, batch_slab):
        assert batch_slab.ndim == 4
        assert batch_slab.shape[1:3] == self.in_shape
        return batch_slab[:, self.start_x:self.end_x, self.start_y:self.end_y, 0:1]


# Tests


class TestPredictWithSteps(unittest.TestCase):

    def test_predicts_same_size_input_output(self):

        img = np.random.ranf((256, 256))
        detector = FakeConvDetector((256, 256), (256, 256))
        res = preproc.predict_with_steps(img, detector, (256, 256), (256, 256))

        self.assertEqual(res.shape, (256, 256))
        np.testing.assert_almost_equal(res, img)

    def test_predicts_one_off_input_output(self):

        img = np.random.ranf((257, 257))
        detector = FakeConvDetector((256, 256), (256, 256))
        res = preproc.predict_with_steps(img, detector, (256, 256), (256, 256))

        self.assertEqual(res.shape, (257, 257))
        np.testing.assert_almost_equal(res, img)

    def test_predicts_input_output_all_different(self):

        img = np.random.ranf((257, 257))
        detector = FakeConvDetector((256, 256), (225, 225))
        res = preproc.predict_with_steps(img, detector, (256, 256), (225, 225))

        self.assertEqual(res.shape, (257, 257))
        np.testing.assert_almost_equal(res, img)

    def test_predicts_input_output_countception_shape(self):

        img = np.random.ranf((260, 347))
        detector = FakeConvDetector((256, 256), (225, 225))
        res = preproc.predict_with_steps(img, detector, (256, 256), (225, 225))

        self.assertEqual(res.shape, (260, 347))
        np.testing.assert_almost_equal(res, img)

    def test_predicts_input_output_unet_shape(self):

        img = np.random.ranf((260, 347))
        detector = FakeConvDetector((256, 256), (68, 68))
        res = preproc.predict_with_steps(img, detector, (256, 256), (68, 68))

        self.assertEqual(res.shape, (260, 347))
        np.testing.assert_almost_equal(res, img)

    def test_predicts_input_output_with_small_overlap(self):

        img = np.random.ranf((260, 347))
        detector = FakeConvDetector((256, 256), (68, 68))
        res = preproc.predict_with_steps(img, detector, (256, 256), (68, 68), overlap=1)

        self.assertEqual(res.shape, (260, 347))
        np.testing.assert_almost_equal(res, img)

    def test_predicts_input_output_with_large_overlap(self):

        img = np.random.ranf((260, 347))
        detector = FakeConvDetector((256, 256), (68, 68))
        res = preproc.predict_with_steps(img, detector, (256, 256), (68, 68), overlap=(10, 8))

        self.assertEqual(res.shape, (260, 347))
        np.testing.assert_almost_equal(res, img)


class TestCalculatePeakImage(unittest.TestCase):

    def test_peaks_with_single_dot_equal_padding(self):

        target_img = np.zeros((64, 64))
        target_img[32, 32] = 1

        x = np.arange(64) - 32
        y = np.arange(64) - 32

        xx, yy = np.meshgrid(x, y)
        rr = np.sqrt(xx**2 + yy**2)

        exp_img = (1 - rr/4)
        exp_img[exp_img < 0] = 0

        peak_img = preproc.calculate_peak_image(target_img,
                                                img_rows=32, img_cols=32,
                                                zero_padding=32,
                                                peak_sharpness=8)
        self.assertEqual(peak_img.shape, target_img.shape)
        np.testing.assert_almost_equal(exp_img, peak_img)


class TestRandomSplit(unittest.TestCase):

    def test_without_replacement(self):

        ind = np.random.rand(16)
        ind.sort()

        samp, rem = preproc.random_split(ind, 8)

        self.assertEqual(samp.shape, (8, ))
        self.assertEqual(rem.shape, (8, ))

        res = np.concatenate((samp, rem))
        res.sort()

        np.testing.assert_almost_equal(res, ind)

    def test_without_replacement_too_many_samples(self):

        ind = np.random.rand(16)
        ind.sort()

        samp, rem = preproc.random_split(ind, 20)

        self.assertEqual(samp.shape, (16, ))
        self.assertEqual(rem.shape, (0, ))

        res = np.concatenate((samp, rem))
        res.sort()

        np.testing.assert_almost_equal(res, ind)

    def test_with_replacement(self):

        ind = np.random.rand(16)
        ind.sort()

        samp, rem = preproc.random_split(ind, 8, with_replacement=True)

        self.assertEqual(samp.shape, (8, ))
        self.assertEqual(rem.shape, (16, ))

        np.testing.assert_almost_equal(ind, rem)
        self.assertTrue(all([s in ind for s in samp]))

    def test_with_replacement_too_many_samples(self):

        ind = np.random.rand(16)
        ind.sort()

        samp, rem = preproc.random_split(ind, 20, with_replacement=True)
        self.assertEqual(samp.shape, (20, ))
        self.assertEqual(rem.shape, (16, ))

        np.testing.assert_almost_equal(ind, rem)
        self.assertTrue(all([s in ind for s in samp]))


class TestCompositeMask(unittest.TestCase):

    def test_composite_one_sample_mean(self):

        srows, scols = 16, 16

        img = np.random.rand(16, 16, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             mode='mean')
        exp = np.ones((16, 16))

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_full_field_mean(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             mode='mean')
        exp = np.ones((32, 32))

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_full_field_mean_small_batches(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             batch_size=2,
                             mode='mean')
        exp = np.ones((32, 32))

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             batch_size=3,
                             mode='mean')
        exp = np.ones((32, 32))

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_full_field_mean_strided(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=5,
                             batch_size=2,
                             mode='mean')
        exp = np.ones((32, 32))
        exp[:, -1] = np.nan
        exp[-1, :] = np.nan

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_full_field_mean_masked(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)
        mask = np.zeros((32, 32), dtype=np.bool)
        mask[:4, :4] = 1
        mask[-4:, -4:] = 1

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             mask=mask,
                             srows=srows, scols=scols,
                             batch_stride=5,
                             batch_size=2,
                             mode='mean')
        exp = np.ones((32, 32))
        exp[:, -1] = np.nan
        exp[-1, :] = np.nan
        exp[~mask] = np.nan

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_one_field_peaks(self):

        srows, scols = 16, 16

        img = np.random.rand(16, 16, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             mode='peak')
        exp = np.full((16, 16), np.nan)
        exp[8, 8] = 1

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_full_field_peaks(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             mode='peaks')
        exp = np.full((32, 32), np.nan)
        exp[8:25, 8:25] = 1

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_full_field_peaks_rotations(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             mode='peaks',
                             transforms='rotations')

        exp0 = np.full((32, 32), np.nan)
        exp0[8:25, 8:25] = 1

        exp1 = np.full((32, 32), np.nan)
        exp1[8:25, 7:24] = 1

        exp2 = np.full((32, 32), np.nan)
        exp2[7:24, 7:24] = 1

        exp3 = np.full((32, 32), np.nan)
        exp3[7:24, 8:25] = 1

        exp = [exp0, exp1, exp2, exp3]

        self.assertEqual(len(res), len(exp))
        for r, e in zip(res, exp):
            np.testing.assert_almost_equal(r, e)

    def test_composite_full_field_peaks_small_batches(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             batch_size=2,
                             mode='peaks')
        exp = np.full((32, 32), np.nan)
        exp[8:25, 8:25] = 1

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=1,
                             batch_size=3,
                             mode='peaks')
        exp = np.full((32, 32), np.nan)
        exp[8:25, 8:25] = 1

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

    def test_composite_full_field_peaks_strided(self):

        srows, scols = 16, 16

        img = np.random.rand(32, 32, 3)

        detector = FakeDetector()

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=5,
                             batch_size=2,
                             mode='peaks')
        exp = np.full((32, 32), np.nan)
        exp[6:26, 6:26] = 1

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)

        res = composite_mask(img, detector,
                             srows=srows, scols=scols,
                             batch_stride=5,
                             batch_size=3,
                             mode='peaks')
        exp = np.full((32, 32), np.nan)
        exp[6:26, 6:26] = 1

        self.assertEqual(len(res), 1)
        np.testing.assert_almost_equal(res[0], exp)


class TestCompleteSampler(unittest.TestCase):

    def test_samples_upper_corner(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.CompleteSampler(files=[],
                                          image_layout='tensorflow',
                                          batch_size=1,
                                          input_shape=(64, 96, 3),
                                          size_range=(128, 256),
                                          rotation_range=(-10, 10),
                                          flip_horizontal=True,
                                          noise_type='none',
                                          noise_fraction=0.1,
                                          cache_size=None)
        sampler.current_index = 0
        sampler.current_slice = 0
        out_img = sampler.slice_next(1, img)

        self.assertEqual(sampler.current_index, 0)
        self.assertEqual(sampler.current_slice, 1)
        self.assertEqual(out_img.shape, (1, 64, 96, 3))
        np.testing.assert_almost_equal(out_img[0, ...], img[:64, :96, :])

        out_img = sampler.slice_next(1, img)

        self.assertEqual(sampler.current_index, 0)
        self.assertEqual(sampler.current_slice, 2)
        self.assertEqual(out_img.shape, (1, 64, 96, 3))
        np.testing.assert_almost_equal(out_img[0, ...], img[:64, 1:97, :])

    def test_samples_over_whole_image(self):

        img = np.random.rand(100, 100, 3)

        sampler = preproc.CompleteSampler(files=[],
                                          image_layout='tensorflow',
                                          batch_size=1,
                                          input_shape=(64, 96, 3),
                                          size_range=(128, 256),
                                          rotation_range=(-10, 10),
                                          flip_horizontal=True,
                                          noise_type='none',
                                          noise_fraction=0.1,
                                          cache_size=None)
        sampler.current_index = 0
        sampler.current_slice = 0
        out_img = sampler.slice_next(185, img)

        self.assertEqual(sampler.current_index, 1)
        self.assertEqual(sampler.current_slice, 0)

        self.assertEqual(out_img.shape, (185, 64, 96, 3))

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img[i:i+64, j:j+96, :])

    def test_samples_over_whole_image_color_to_gray(self):

        img = np.random.rand(100, 100, 3)
        img_gray = np.mean(img, axis=2)[..., np.newaxis]

        sampler = preproc.CompleteSampler(files=[],
                                          image_layout='tensorflow',
                                          batch_size=1,
                                          input_shape=(64, 96, 1),
                                          size_range=(128, 256),
                                          rotation_range=(-10, 10),
                                          flip_horizontal=True,
                                          noise_type='none',
                                          noise_fraction=0.1,
                                          cache_size=None)
        sampler.current_index = 0
        sampler.current_slice = 0
        out_img = sampler.slice_next(185, img)

        self.assertEqual(sampler.current_index, 1)
        self.assertEqual(sampler.current_slice, 0)

        self.assertEqual(out_img.shape, (185, 64, 96, 1))

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img_gray[i:i+64, j:j+96, :])

    def test_samples_as_much_as_it_can(self):

        img = np.random.rand(100, 100, 3)

        sampler = preproc.CompleteSampler(files=[],
                                          image_layout='tensorflow',
                                          batch_size=1,
                                          input_shape=(64, 96, 3),
                                          size_range=(128, 256),
                                          rotation_range=(-10, 10),
                                          flip_horizontal=True,
                                          noise_type='none',
                                          noise_fraction=0.1,
                                          cache_size=None)
        sampler.current_index = 0
        sampler.current_slice = 0
        out_img = sampler.slice_next(187, img)

        self.assertEqual(sampler.current_index, 1)
        self.assertEqual(sampler.current_slice, 0)

        self.assertEqual(out_img.shape, (185, 64, 96, 3))

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img[i:i+64, j:j+96, :])

    def test_samples_as_much_as_it_can_with_an_offset(self):

        img = np.random.rand(100, 100, 3)

        sampler = preproc.CompleteSampler(files=[],
                                          image_layout='tensorflow',
                                          batch_size=1,
                                          input_shape=(64, 96, 3),
                                          size_range=(128, 256),
                                          rotation_range=(-10, 10),
                                          flip_horizontal=True,
                                          noise_type='none',
                                          noise_fraction=0.1,
                                          cache_size=None)
        sampler.current_index = 0
        sampler.current_slice = 5
        out_img = sampler.slice_next(187, img)

        self.assertEqual(sampler.current_index, 1)
        self.assertEqual(sampler.current_slice, 0)

        self.assertEqual(out_img.shape, (180, 64, 96, 3))

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j - 5
                if idx < 0:
                    continue
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img[i:i+64, j:j+96, :])

    def test_samples_multiple_whole_images(self):

        img1 = np.random.rand(100, 100, 3)
        img2 = np.random.rand(100, 100, 3)

        sampler = preproc.CompleteSampler(files=[],
                                          image_layout='tensorflow',
                                          batch_size=1,
                                          input_shape=(64, 96, 3),
                                          size_range=(128, 256),
                                          rotation_range=(-10, 10),
                                          flip_horizontal=True,
                                          noise_type='none',
                                          noise_fraction=0.1,
                                          cache_size=None)
        sampler.current_index = 0
        sampler.current_slice = 0
        out_img1, out_img2 = sampler.slice_next(185, img1, img2)

        self.assertEqual(sampler.current_index, 1)
        self.assertEqual(sampler.current_slice, 0)

        self.assertEqual(out_img1.shape, (185, 64, 96, 3))
        self.assertEqual(out_img2.shape, (185, 64, 96, 3))

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j
                np.testing.assert_almost_equal(out_img1[idx, ...],
                                               img1[i:i+64, j:j+96, :])
                np.testing.assert_almost_equal(out_img2[idx, ...],
                                               img2[i:i+64, j:j+96, :])

    def test_resample_all_over_several_images(self):

        img1 = np.random.rand(100, 100, 3)
        img2 = np.random.rand(110, 110, 3)

        class FakeCompleteSampler(preproc.CompleteSampler):

            def load_file(self, filename):
                if filename.name == '001.jpg':
                    return img1
                elif filename.name == '003.jpg':
                    return img2
                else:
                    return None

        sampler = FakeCompleteSampler(files=['001.jpg', '002.jpg', '003.jpg'],
                                      image_layout='tensorflow',
                                      batch_size=1024,
                                      input_shape=(64, 96, 3),
                                      size_range=(128, 256),
                                      rotation_range=(-10, 10),
                                      flip_horizontal=True,
                                      noise_type='none',
                                      noise_fraction=0.1,
                                      cache_size=None)
        out_img = sampler.resample_all(1024)

        self.assertEqual(out_img.shape, (1024, 64, 96, 3))
        self.assertEqual(sampler.current_slice, 134)
        self.assertEqual(sampler.current_index, 0)

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img1[i:i+64, j:j+96, :])

        for i in range(47):
            for j in range(15):
                idx = i * 15 + j + 185
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img2[i:i+64, j:j+96, :])

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j + 890
                if idx >= 1024:
                    break
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img1[i:i+64, j:j+96, :])

    def test_resample_all_over_several_images_with_masks(self):

        img1 = np.random.rand(100, 100, 3)
        img2 = np.random.rand(110, 110, 3)

        mask1 = np.random.rand(100, 100, 1)
        mask2 = np.random.rand(110, 110, 1)

        class FakeCompleteSampler(preproc.CompleteSampler):

            def load_file(self, filename):
                if filename.name == '001.jpg':
                    return img1
                elif filename.name == '003.jpg':
                    return img2
                else:
                    return None

            def load_mask(self, filename, img):
                if filename.name == '001.jpg':
                    return mask1
                elif filename.name == '003.jpg':
                    return mask2
                else:
                    return None

        sampler = FakeCompleteSampler(files=['001.jpg', '002.jpg', '003.jpg'],
                                      masks=['001.npz', '002.npz', '003.npz'],
                                      image_layout='tensorflow',
                                      batch_size=1024,
                                      input_shape=(64, 96, 3),
                                      size_range=(128, 256),
                                      rotation_range=(-10, 10),
                                      flip_horizontal=True,
                                      noise_type='none',
                                      noise_fraction=0.1,
                                      cache_size=None)
        out_img, out_mask = sampler.resample_all(1024)

        self.assertEqual(out_img.shape, (1024, 64, 96, 3))
        self.assertEqual(out_mask.shape, (1024, 64, 96, 1))
        self.assertEqual(sampler.current_slice, 134)
        self.assertEqual(sampler.current_index, 0)

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img1[i:i+64, j:j+96, :])
                np.testing.assert_almost_equal(out_mask[idx, ...],
                                               mask1[i:i+64, j:j+96, :])

        for i in range(47):
            for j in range(15):
                idx = i * 15 + j + 185
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img2[i:i+64, j:j+96, :])
                np.testing.assert_almost_equal(out_mask[idx, ...],
                                               mask2[i:i+64, j:j+96, :])

        for i in range(37):
            for j in range(5):
                idx = i * 5 + j + 890
                if idx >= 1024:
                    break
                np.testing.assert_almost_equal(out_img[idx, ...],
                                               img1[i:i+64, j:j+96, :])
                np.testing.assert_almost_equal(out_mask[idx, ...],
                                               mask1[i:i+64, j:j+96, :])


class TestRandomSampler(unittest.TestCase):

    def test_load_mask(self):

        img = np.random.rand(300, 300, 3)
        masks = {
            'foo': [
                [0.0, 0.0, 0.4, 0.5],
                [0.9, 0.9, 1.0, 1.0],
            ],
        }

        sampler = preproc.RandomSampler(files=[],
                                        masks=masks,
                                        image_layout='theano',
                                        batch_size=1,
                                        input_shape=(64, 96, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_mask = sampler.load_mask(pathlib.Path('grr/foo.jpg'), img)

        self.assertEqual(out_mask.shape, (300, 300, 1))

        exp_mask = np.zeros((300, 300, 1), dtype=np.bool)
        exp_mask[150:, :120, :] = True
        exp_mask[:30, 270:, :] = True

        np.testing.assert_almost_equal(exp_mask, out_mask)

    def test_resample_image_theano(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='theano',
                                        batch_size=1,
                                        input_shape=(64, 96, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img = sampler.resample_image(img)

        self.assertEqual(out_img.shape, (3, 96, 64))

    def test_resample_image_tensorflow(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(64, 96, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img = sampler.resample_image(img)

        self.assertEqual(out_img.shape, (64, 96, 3))

    def test_resample_image_tensorflow_color_to_gray(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(64, 96, 1),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img = sampler.resample_image(img)

        self.assertEqual(out_img.shape, (64, 96, 1))

    def test_can_resample_with_fixed_params_no_change(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(300, 300, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img = sampler.resample_image(
            img, size=300, theta=0, shift=[0, 0],
            flip_horizontal=False)
        exp_img = img

        np.testing.assert_almost_equal(exp_img, out_img, decimal=4)

    def test_can_resample_with_fixed_params_zero_pad_no_change(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(300, 300, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        zero_padding=10,
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img = sampler.resample_image(
            img, size=300, theta=0, shift=[10, 10],
            flip_horizontal=False)
        exp_img = img

        np.testing.assert_almost_equal(exp_img, out_img, decimal=4)

    def test_can_resample_with_fixed_params_shifts_flips(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(200, 200, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        flip_vertical=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img = sampler.resample_image(
            img, size=200, theta=0, shift=[10, 10],
            flip_horizontal=True, flip_vertical=True)
        exp_img = img[10:-90, 10:-90, :]
        exp_img = exp_img[::-1, ::-1, :]

        np.testing.assert_almost_equal(exp_img, out_img, decimal=4)

    def test_can_resample_with_fixed_params_only_resize(self):

        img = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(64, 96, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img = sampler.resample_image(
            img, size=300, theta=0, shift=[0, 0],
            flip_horizontal=False)

        exp_img = preproc.resample_in_box(
            img, 300, np.eye(2), np.array([[150.0], [150.0]]),
            input_shape=(64, 96, 3))

        np.testing.assert_almost_equal(exp_img, out_img)

    def test_can_resample_multiple_images_with_same_transform(self):

        img1 = np.random.rand(300, 300, 3)
        img2 = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(200, 200, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img1, out_img2 = sampler.resample_image(
            img1, img2, size=200, theta=0, shift=[10, 10],
            flip_horizontal=True)
        exp_img1 = img1[10:-90, 10:-90, :]
        exp_img1 = exp_img1[:, ::-1, :]

        np.testing.assert_almost_equal(exp_img1, out_img1, decimal=4)

        exp_img2 = img2[10:-90, 10:-90, :]
        exp_img2 = exp_img2[:, ::-1, :]

        np.testing.assert_almost_equal(exp_img2, out_img2, decimal=4)

    def test_can_resample_multiple_images_with_same_transform_padding(self):

        img1 = np.random.rand(300, 300, 3)
        img2 = np.random.rand(300, 300, 3)

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(200, 200, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None,
                                        zero_padding=5)
        out_img1, out_img2 = sampler.resample_image(
            img1, img2, size=200, theta=0, shift=[10, 10],
            flip_horizontal=True)
        exp_img1 = img1[5:205, 5:205, :]
        exp_img1 = exp_img1[:, ::-1, :]

        np.testing.assert_almost_equal(exp_img1, out_img1, decimal=4)

        exp_img2 = img2[5:205, 5:205, :]
        exp_img2 = exp_img2[:, ::-1, :]

        np.testing.assert_almost_equal(exp_img2, out_img2, decimal=4)

    def test_can_resample_multiple_images_random_transform(self):

        img1 = np.random.rand(300, 300, 3)
        img2 = img1.copy()

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(200, 200, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        flip_vertical=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img1, out_img2 = sampler.resample_image(
            img1, img2)

        np.testing.assert_almost_equal(out_img1, out_img2, decimal=4)

    def test_can_resample_mask_with_image_same_transform(self):

        img1 = np.random.rand(300, 300, 3)
        img2 = np.zeros((300, 300), dtype=np.bool)
        img2[10:-90, 10:-90] = False

        sampler = preproc.RandomSampler(files=[],
                                        image_layout='tensorflow',
                                        batch_size=1,
                                        input_shape=(200, 200, 3),
                                        size_range=(128, 256),
                                        rotation_range=(-10, 10),
                                        flip_horizontal=True,
                                        noise_type='none',
                                        noise_fraction=0.1,
                                        cache_size=None)
        out_img1, out_img2 = sampler.resample_image(
            img1, img2, size=200, theta=0, shift=[10, 10],
            flip_horizontal=True)
        exp_img1 = img1[10:-90, 10:-90, :]
        exp_img1 = exp_img1[:, ::-1, :]

        np.testing.assert_almost_equal(exp_img1, out_img1, decimal=4)

        exp_img2 = img2[10:-90, 10:-90]
        exp_img2 = exp_img2[:, ::-1]

        np.testing.assert_almost_equal(exp_img2, out_img2, decimal=4)
        self.assertTrue(np.all(exp_img2 == 0))


class TestResampleInBox(unittest.TestCase):

    def test_resample_grayscale_2d(self):

        img = np.random.random((512, 512, 3))
        img = np.mean(img, axis=2)

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        out_img = preproc.resample_in_box(
            img, scale, rot, shift, input_shape=256)

        self.assertEqual(out_img.shape, (256, 256))

    def test_resample_grayscale_3d(self):

        img = np.random.random((512, 512, 3))
        img = np.mean(img, axis=2)
        img = img[:, :, np.newaxis]

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        out_img = preproc.resample_in_box(
            img, scale, rot, shift, input_shape=256)

        self.assertEqual(out_img.shape, (256, 256, 1))

    def test_resample_grayscale_3d_colors(self):

        img = np.random.random((512, 512, 3))

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        out_img = preproc.resample_in_box(
            img, scale, rot, shift, input_shape=256)

        self.assertEqual(out_img.shape, (256, 256, 3))

    def test_resample_grayscale_3d_colors_x_y_diff(self):

        img = np.random.random((512, 512, 3))

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        out_img = preproc.resample_in_box(
            img, scale, rot, shift, input_shape=(256, 128))

        self.assertEqual(out_img.shape, (256, 128, 3))

    def test_resample_grayscale_2d_to_colors(self):

        img = np.random.random((512, 512, ))

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        out_img = preproc.resample_in_box(
            img, scale, rot, shift, input_shape=(256, 128, 3))

        self.assertEqual(out_img.shape, (256, 128, 3))

    def test_resample_grayscale_3d_to_colors(self):

        img = np.random.random((512, 512, 1))

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        out_img = preproc.resample_in_box(
            img, scale, rot, shift, input_shape=(256, 128, 3))

        self.assertEqual(out_img.shape, (256, 128, 3))

    def test_resample_colors_to_grayscale(self):

        img = np.random.random((512, 512, 3))

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        out_img = preproc.resample_in_box(
            img, scale, rot, shift, input_shape=(256, 128, 1))

        self.assertEqual(out_img.shape, (256, 128, 1))

    def test_4d_input_shape_raises_errors(self):

        img = np.random.random((512, 512, 3))

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        with self.assertRaises(ValueError):
            preproc.resample_in_box(
                img, scale, rot, shift, input_shape=(256, 128, 1, 1))

    def test_input_shape_with_crazy_dims_raises_errors(self):

        img = np.random.random((512, 512, 3))

        scale = 2
        rot = np.array([
            [1, 0],
            [0, 1],
        ])

        shift = np.array([
            [1], [1]
        ])
        with self.assertRaises(ValueError):
            preproc.resample_in_box(
                img, scale, rot, shift, input_shape=(256, 128, 2))


class TestImageResampler(helpers.FileSystemTestCase):

    def fullpath(self, *args):
        r = self.tempdir
        for a in args:
            r = r / a
        return r

    def make_image(self, *args, **kwargs):

        image_path = self.fullpath(*args)

        size = kwargs.pop('size', (512, 512, 3))

        # Random noise image
        img = np.random.random(size)
        img = np.round(img * 255)
        img[img < 0] = 0
        img[img > 255] = 255
        img = Image.fromarray(img.astype(np.uint8))

        image_path.parent.mkdir(exist_ok=True, parents=True)
        img.save(str(image_path))
        return image_path

    def make_mask(self, *args, **kwargs):
        mask_path = self.fullpath(*args)

        size = kwargs.pop('size', (512, 512))

        # Random noise image
        mask = np.random.random(size) > 0.5

        mask_path.parent.mkdir(exist_ok=True, parents=True)

        np.savez(str(mask_path), mask=mask, refined_mask=mask)
        return mask_path

    def make_resampler(self,
                       datadir=None,
                       data_finder=None,
                       mask_finder=None,
                       mask_type=None,
                       test_fraction=None,
                       validation_fraction=None,
                       **kwargs):
        """ Make the ImageResampler object

        :param Path datadir:
            The data directory or self.tempdir
        :param float validation_fraction:
            How many images in the validation set (default 0)
        :param \\*\\* kwargs:
            Arguments to pass to the load_samplers method of the resampler object
        :returns:
            The loaded ImageResampler object
        """
        if datadir is None:
            datadir = self.tempdir
        if data_finder is None:
            data_finder = preproc.find_raw_data

        proc = preproc.ImageResampler()
        proc.set_data_loader(datadir, data_finder=data_finder)
        if mask_finder is not None:
            proc.set_mask_loader(mask_finder=mask_finder, mask_type=mask_type)
        proc.load_files()
        proc.calc_train_test_split(test_fraction=test_fraction,
                                   validation_fraction=validation_fraction)
        proc.load_samplers(**kwargs)
        return proc

    def test_is_split_under_datadir(self):

        self.make_image('foo', '001.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=1)

        self.assertTrue(proc.is_split_under_datadir(self.tempdir / 'foo'))
        self.assertFalse(proc.is_split_under_datadir(self.tempdir / 'bees'))
        self.assertFalse(proc.is_split_under_datadir(self.tempdir))  # FIXME: This should work

    def test_resample_one_image(self):

        self.make_image('foo', '001.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=1)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (1, 1, 256, 256))

    def test_resample_several_images(self):

        self.make_image('foo', '001.jpg')
        self.make_image('foo', '002.jpg')
        self.make_image('foo', '003.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=0.333,
                                   batch_size=2)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (2, 1, 256, 256))

        with self.assertRaises(ValueError):
            imgs = next(proc.validation_data)

        proc.validation_data.batch_size = 1

        imgs = next(proc.validation_data)

        self.assertEqual(imgs.shape, (1, 1, 256, 256))

        self.assertEqual(len(proc.train_data), 2)
        self.assertEqual(len(proc.validation_data), 1)

    def test_resample_several_images_colored(self):

        self.make_image('foo', '001.jpg')
        self.make_image('foo', '002.jpg')
        self.make_image('foo', '003.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=0.333,
                                   batch_size=2,
                                   input_shape=(256, 256, 3))
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (2, 3, 256, 256))

        with self.assertRaises(ValueError):
            imgs = next(proc.validation_data)

        proc.validation_data.batch_size = 1

        imgs = next(proc.validation_data)

        self.assertEqual(imgs.shape, (1, 3, 256, 256))

        self.assertEqual(len(proc.train_data), 2)
        self.assertEqual(len(proc.validation_data), 1)

    def test_resample_several_images_one_deleted(self):

        i1 = self.make_image('foo', '001.jpg')
        i2 = self.make_image('foo', '002.jpg')
        i3 = self.make_image('foo', '003.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=0.0,
                                   batch_size=3,
                                   cache_size=0)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (3, 1, 256, 256))

        i1.unlink()

        with self.assertRaises(ValueError):
            next(proc.train_data)

        proc.train_data.batch_size = 2

        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (2, 1, 256, 256))
        self.assertEqual(set(proc.train_data.files), {i2, i3})

        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (2, 1, 256, 256))
        self.assertEqual(set(proc.train_data.files), {i2, i3})

    def test_resample_several_images_several_deleted(self):

        i1 = self.make_image('foo', '001.jpg')
        i2 = self.make_image('foo', '002.jpg')
        i3 = self.make_image('foo', '003.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=0.0,
                                   batch_size=3,
                                   cache_size=0)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (3, 1, 256, 256))

        i1.unlink()
        i3.unlink()

        with self.assertRaises(ValueError):
            next(proc.train_data)

        proc.train_data.batch_size = 1

        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (1, 1, 256, 256))
        self.assertEqual(proc.train_data.files, [i2])

        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (1, 1, 256, 256))
        self.assertEqual(proc.train_data.files, [i2])

    def test_resample_several_images_large_cache(self):

        self.make_image('foo', '001.jpg')
        self.make_image('foo', '002.jpg')
        self.make_image('foo', '003.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=3,
                                   cache_size=5)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (3, 1, 256, 256))
        self.assertEqual(len(proc.train_data), 3)
        self.assertEqual(len(proc.train_data.image_cache), 3)

    def test_resample_several_images_no_cache(self):

        self.make_image('foo', '001.jpg')
        self.make_image('foo', '002.jpg')
        self.make_image('foo', '003.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=3,
                                   cache_size=None)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (3, 1, 256, 256))
        self.assertEqual(len(proc.train_data), 3)
        self.assertEqual(len(proc.train_data.image_cache), 0)

    def test_resample_several_images_small_cache(self):

        self.make_image('foo', '001.jpg')
        self.make_image('foo', '002.jpg')
        self.make_image('foo', '003.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=3,
                                   cache_size=2)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (3, 1, 256, 256))
        self.assertEqual(len(proc.train_data), 3)
        self.assertEqual(len(proc.train_data.image_cache), 2)

    def test_resample_several_images_deduplicated_cache(self):

        self.make_image('foo', '001.jpg')
        self.make_image('bar', '001.jpg')
        self.make_image('baz', '001.jpg')

        proc = self.make_resampler(test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=3,
                                   cache_size=5)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (3, 1, 256, 256))
        self.assertEqual(len(proc.train_data), 3)
        self.assertEqual(len(proc.train_data.image_cache), 1)

    def test_resample_several_images_alternate_finder(self):

        def find_data(datadir, blacklist=None):
            channeldir = datadir / 'TL Brightfield'
            for tiledir in channeldir.iterdir():
                for image_file in tiledir.iterdir():
                    yield image_file

        self.make_image('TL Brightfield', 's01', 's01-001.jpg')
        self.make_image('TL Brightfield', 's02', 's02-001.jpg')
        self.make_image('TL Brightfield', 's02', 's02-002.jpg')

        proc = self.make_resampler(data_finder=find_data,
                                   test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=3,
                                   cache_size=0)
        imgs = next(proc.train_data)

        self.assertEqual(imgs.shape, (3, 1, 256, 256))
        self.assertEqual(len(proc.train_data), 3)

    def test_resample_several_images_and_masks(self):

        def find_data(datadir, blacklist=None):
            channeldir = datadir / 'Corrected' / 'TL Brightfield'
            for tiledir in channeldir.iterdir():
                for image_file in tiledir.iterdir():
                    yield image_file

        def find_masks(datadir, blacklist=None):
            channeldir = datadir / 'colony_mask' / 'TL Brightfield'
            for tiledir in channeldir.iterdir():
                for image_file in tiledir.iterdir():
                    yield image_file.stem, image_file

        self.make_image('Corrected', 'TL Brightfield', 's01', 's01-001.jpg')
        self.make_image('Corrected', 'TL Brightfield', 's02', 's02-001.jpg')
        self.make_image('Corrected', 'TL Brightfield', 's02', 's02-002.jpg')

        self.make_mask('colony_mask', 'TL Brightfield', 's01', 's01-001.npz')
        self.make_mask('colony_mask', 'TL Brightfield', 's02', 's02-001.npz')

        proc = self.make_resampler(data_finder=find_data,
                                   mask_finder=find_masks,
                                   mask_type='file',
                                   test_fraction=None,
                                   validation_fraction=None,
                                   batch_size=2,
                                   cache_size=0)
        imgs, masks = next(proc.train_data)

        self.assertEqual(imgs.shape, (2, 1, 256, 256))
        self.assertEqual(masks.shape, (2, 1, 256, 256))
        self.assertEqual(len(proc.train_data), 2)
