# Standard lib
import pathlib

# 3rd party
from PIL import Image

import numpy as np

# Our own imports
from deep_hipsc_tracking.utils import (
    image_volume, load_image, save_image)

from .. import helpers

# Tests


class TestFindImageVolumes(helpers.FileSystemTestCase):

    def make_image_file(self, image_file: pathlib.Path,
                        slices: int, rows: int, cols: int) -> pathlib.Path:
        """ Make an image file of a known shape """

        image_file.parent.mkdir(parents=True, exist_ok=True)

        img = np.random.randint(0, 32, (slices, rows, cols))

        # Note that it's "append_images" for TIFF format, not "append" like for GIF
        slices = [Image.fromarray(img[i, :, :].astype(np.uint8)) for i in range(slices)]
        slices[0].save(image_file, format='TIFF', save_all=True, append_images=slices[1:])
        return image_file

    def make_image_dir(self, image_dir: pathlib.Path,
                       slices: int, rows: int, cols: int) -> pathlib.Path:
        """ Make an image file of a known shape """

        image_dir.mkdir(parents=True, exist_ok=False)

        img = np.random.randint(0, 32, (rows, cols))
        for i in range(slices):

            image_file = image_dir / f'frame{i:05d}.tif'
            save_image(image_file, img + i, cmin=0, cmax=255, ctype='grey')
        return image_dir

    def test_doesnt_find_empty_dirs(self):

        rootdir = self.tempdir

        d1 = self.tempdir / 'foo'
        d2 = self.tempdir / 'grr' / 'arg'

        d1.mkdir(parents=True, exist_ok=True)
        d2.mkdir(parents=True, exist_ok=True)

        image_files = image_volume.find_image_volumes(rootdir)

        self.assertEqual(image_files, [])

    def test_finds_nested_image_files(self):

        rootdir = self.tempdir

        f1 = self.tempdir / 'foo' / 'bar.tif'
        f2 = self.tempdir / 'grr' / 'arg.tif'

        self.make_image_file(f1, 10, 32, 33)
        self.make_image_file(f2, 11, 63, 64)

        image_files = image_volume.find_image_volumes(rootdir)

        image_file_paths = {res.image_file for res in image_files}

        self.assertEqual(len(image_files), 2)
        self.assertEqual({f1, f2}, image_file_paths)

        image_shapes = {res.shape for res in image_files}
        self.assertEqual({(10, 32, 33), (11, 63, 64)}, image_shapes)

    def test_finds_nested_image_dirs(self):

        rootdir = self.tempdir

        d1 = self.tempdir / 'foo' / 'bar'
        d2 = self.tempdir / 'grr' / 'arg'

        self.make_image_dir(d1, 10, 32, 33)
        self.make_image_dir(d2, 11, 63, 64)

        image_dirs = image_volume.find_image_volumes(rootdir)

        image_dir_paths = {res.image_dir for res in image_dirs}

        self.assertEqual(len(image_dir_paths), 2)
        self.assertEqual({d1, d2}, image_dir_paths)

        image_shapes = {res.shape for res in image_dirs}
        self.assertEqual({(10, 32, 33), (11, 63, 64)}, image_shapes)

    def test_finds_nested_mixed_image_dirs_files(self):

        rootdir = self.tempdir

        d1 = self.tempdir / 'foo' / 'bar'
        d2 = self.tempdir / 'grr' / 'arg'

        f1 = self.tempdir / 'foo' / 'bar.tif'
        f2 = self.tempdir / 'grr' / 'arg.tif'

        self.make_image_dir(d1, 10, 32, 33)
        self.make_image_dir(d2, 11, 63, 64)
        self.make_image_file(f1, 12, 55, 44)
        self.make_image_file(f2, 13, 44, 55)

        image_paths = image_volume.find_image_volumes(rootdir)

        self.assertEqual(len(image_paths), 4)

        image_dir_paths = {res.image_dir for res in image_paths if hasattr(res, 'image_dir')}
        image_file_paths = {res.image_file for res in image_paths if hasattr(res, 'image_file')}

        self.assertEqual(len(image_dir_paths), 2)
        self.assertEqual(len(image_file_paths), 2)

        self.assertEqual({d1, d2}, image_dir_paths)
        self.assertEqual({f1, f2}, image_file_paths)

        image_shapes = {res.shape for res in image_paths}
        self.assertEqual({(10, 32, 33), (11, 63, 64), (12, 55, 44), (13, 44, 55)},
                         image_shapes)

    def test_finds_nested_mixed_image_dirs_files_matching_pattern(self):

        rootdir = self.tempdir

        d1 = self.tempdir / 'foo' / 'bar1234'
        d2 = self.tempdir / 'grr' / 'arg2345'

        f1 = self.tempdir / 'foo' / 'bar2345.tif'
        f2 = self.tempdir / 'grr' / 'arg3456.tif'

        self.make_image_dir(d1, 10, 32, 33)
        self.make_image_dir(d2, 11, 63, 64)
        self.make_image_file(f1, 12, 55, 44)
        self.make_image_file(f2, 13, 44, 55)

        image_paths = image_volume.find_image_volumes(rootdir, pattern='bar[0-9]+')

        self.assertEqual(len(image_paths), 2)

        image_dir_paths = {res.image_dir for res in image_paths if hasattr(res, 'image_dir')}
        image_file_paths = {res.image_file for res in image_paths if hasattr(res, 'image_file')}

        self.assertEqual(len(image_dir_paths), 1)
        self.assertEqual(len(image_file_paths), 1)

        self.assertEqual({d1, }, image_dir_paths)
        self.assertEqual({f1, }, image_file_paths)

        image_shapes = {res.shape for res in image_paths}
        self.assertEqual({(10, 32, 33), (12, 55, 44)},
                         image_shapes)

        image_paths = image_volume.find_image_volumes(rootdir, pattern='arg[0-9]+')

        self.assertEqual(len(image_paths), 2)

        image_dir_paths = {res.image_dir for res in image_paths if hasattr(res, 'image_dir')}
        image_file_paths = {res.image_file for res in image_paths if hasattr(res, 'image_file')}

        self.assertEqual(len(image_dir_paths), 1)
        self.assertEqual(len(image_file_paths), 1)

        self.assertEqual({d2, }, image_dir_paths)
        self.assertEqual({f2, }, image_file_paths)

        image_shapes = {res.shape for res in image_paths}
        self.assertEqual({(11, 63, 64), (13, 44, 55)},
                         image_shapes)

    def test_can_select_image_volume_type(self):

        rootdir = self.tempdir

        d1 = self.tempdir / 'foo' / 'bar1234'
        d2 = self.tempdir / 'grr' / 'arg2345'

        f1 = self.tempdir / 'foo' / 'bar2345.tif'
        f2 = self.tempdir / 'grr' / 'arg3456.tif'

        self.make_image_dir(d1, 10, 32, 33)
        self.make_image_dir(d2, 11, 63, 64)
        self.make_image_file(f1, 12, 55, 44)
        self.make_image_file(f2, 13, 44, 55)

        image_paths = image_volume.find_image_volumes(rootdir, volume_type='dir')

        self.assertEqual(len(image_paths), 2)

        image_dir_paths = {res.image_dir for res in image_paths if hasattr(res, 'image_dir')}
        image_file_paths = {res.image_file for res in image_paths if hasattr(res, 'image_file')}

        self.assertEqual(len(image_dir_paths), 2)
        self.assertEqual(len(image_file_paths), 0)

        self.assertEqual({d1, d2}, image_dir_paths)

        image_shapes = {res.shape for res in image_paths}
        self.assertEqual({(10, 32, 33), (11, 63, 64)},
                         image_shapes)

        image_paths = image_volume.find_image_volumes(rootdir, volume_type='file')

        self.assertEqual(len(image_paths), 2)

        image_dir_paths = {res.image_dir for res in image_paths if hasattr(res, 'image_dir')}
        image_file_paths = {res.image_file for res in image_paths if hasattr(res, 'image_file')}

        self.assertEqual(len(image_dir_paths), 0)
        self.assertEqual(len(image_file_paths), 2)

        self.assertEqual({f1, f2, }, image_file_paths)

        image_shapes = {res.shape for res in image_paths}
        self.assertEqual({(12, 55, 44), (13, 44, 55)},
                         image_shapes)

        image_paths = image_volume.find_image_volumes(rootdir, volume_type='both')

        self.assertEqual(len(image_paths), 4)

        image_dir_paths = {res.image_dir for res in image_paths if hasattr(res, 'image_dir')}
        image_file_paths = {res.image_file for res in image_paths if hasattr(res, 'image_file')}

        self.assertEqual(len(image_dir_paths), 2)
        self.assertEqual(len(image_file_paths), 2)

        self.assertEqual({d1, d2, }, image_dir_paths)
        self.assertEqual({f1, f2, }, image_file_paths)

        image_shapes = {res.shape for res in image_paths}
        self.assertEqual({(10, 32, 33), (11, 63, 64), (12, 55, 44), (13, 44, 55)},
                         image_shapes)


class TestLazyImageFile(helpers.FileSystemTestCase):

    def make_image_file(self, slices: int, rows: int, cols: int) -> pathlib.Path:
        """ Make an image file of a known shape """

        outfile = self.tempdir / 'img.tif'
        outfile.parent.mkdir(parents=True, exist_ok=True)

        img = np.random.randint(0, 32, (slices, rows, cols))

        # Note that it's "append_images" for TIFF format, not "append" like for GIF
        image_slices = []
        for i in range(slices):
            # Mark the top left pixel so we can test stack order
            img[i, 0, 0] = i
            image_slices.append(Image.fromarray(img[i, :, :].astype(np.uint8)))
        image_slices[0].save(outfile, format='TIFF', save_all=True, append_images=image_slices[1:])
        return outfile

    def test_can_iterate_over_frames(self):

        image_file = self.make_image_file(10, 64, 63)

        vol = image_volume.LazyImageFile(image_file)

        total_files = 0
        for i, res in enumerate(vol):
            self.assertEqual(res.shape, (64, 63))
            self.assertAlmostEqual(res[0, 0], i)
            total_files += 1
        self.assertEqual(total_files, 10)
        self.assertEqual(len(vol), 10)

    def test_single_index_frames(self):

        image_file = self.make_image_file(10, 64, 63)
        vol = image_volume.LazyImageFile(image_file)

        for i in range(10):
            res = vol[i]
            self.assertEqual(res.shape, (64, 63))
            self.assertAlmostEqual(res[0, 0], i)

    def test_negative_index_frames(self):

        image_file = self.make_image_file(10, 64, 63)

        vol = image_volume.LazyImageFile(image_file)

        for i in range(-10, 0):
            res = vol[i]
            self.assertEqual(res.shape, (64, 63))
            self.assertAlmostEqual(res[0, 0], i+10)

    def test_can_subset_with_bounding_box(self):

        image_file = self.make_image_file(10, 64, 63)
        vol = image_volume.LazyImageFile(image_file)
        subvol = vol.crop([(10, -10), (15, -15)])

        self.assertEqual(subvol.shape, (10, 44, 33))
        self.assertEqual(subvol[2, :, :].shape, (44, 33))

        # Make sure that each individual index matches
        for res in subvol:
            self.assertEqual(res.shape, (44, 33))


class TestLazyImageDir(helpers.FileSystemTestCase):

    def make_image_dir(self, slices: int, rows: int, cols: int) -> pathlib.Path:
        """ Make an image volume of a known shape """

        outdir = self.tempdir / 'img'
        outdir.mkdir(parents=True, exist_ok=False)

        img = np.random.randint(0, 32, (rows, cols))
        for i in range(slices):

            outfile = outdir / f'frame{i:05d}.tif'
            save_image(outfile, img + i, cmin=0, cmax=255, ctype='grey')
        return outdir

    def test_can_iterate_over_frames(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        vol = image_volume.LazyImageDir(image_dir)

        total_files = 0
        for res, framefile in zip(vol, framefiles):
            exp = load_image(framefile)

            np.testing.assert_allclose(res, exp)

            total_files += 1
        self.assertEqual(total_files, len(framefiles))

    def test_single_index_frames(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        vol = image_volume.LazyImageDir(image_dir)

        for i in range(len(framefiles)):
            res = vol[i]
            exp = load_image(framefiles[i])

            np.testing.assert_allclose(res, exp)

    def test_negative_index_frames(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        vol = image_volume.LazyImageDir(image_dir)

        for i in range(-len(framefiles), 0):
            res = vol[i]
            exp = load_image(framefiles[i])

            np.testing.assert_allclose(res, exp)

    def test_frames_in_order(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        self.assertEqual(len(framefiles), 10)

        vol = image_volume.LazyImageDir(image_dir)

        self.assertEqual(vol.shape, (10, 64, 63))

        # Make sure that each individual index matches
        for i in range(len(framefiles)):
            self.assertIn(framefiles[i], vol)

            res = vol[i, ...]
            exp = load_image(framefiles[i])

            np.testing.assert_allclose(res, exp)

        self.assertEqual(len(vol), len(framefiles))

    def test_can_subset_with_bounding_box(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        self.assertEqual(len(framefiles), 10)

        vol = image_volume.LazyImageDir(image_dir)
        subvol = vol.crop([(10, -10), (15, -15)])

        self.assertEqual(subvol.shape, (10, 44, 33))

        # Make sure that each individual index matches
        for i in range(len(framefiles)):
            res = subvol[i, ...]
            exp = load_image(framefiles[i])[10:-10, 15:-15]

            np.testing.assert_allclose(res, exp)

    def test_can_subset_twice_with_bounding_box(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        self.assertEqual(len(framefiles), 10)

        vol = image_volume.LazyImageDir(image_dir)
        subvol = vol.crop([(5, -5), (10, -10)]).crop([(5, -5), (5, -5)])

        self.assertEqual(subvol.shape, (10, 44, 33))

        # Make sure that each individual index matches
        for i in range(len(framefiles)):
            res = subvol[i, ...]
            exp = load_image(framefiles[i])[10:-10, 15:-15]

            np.testing.assert_allclose(res, exp)

    def test_can_subset_with_bounding_box_positive_indices(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        self.assertEqual(len(framefiles), 10)

        vol = image_volume.LazyImageDir(image_dir)
        subvol = vol.crop([(10, 54), (15, 48)])

        self.assertEqual(subvol.shape, (10, 44, 33))

        # Make sure that each individual index matches
        for i in range(len(framefiles)):
            res = subvol[i, ...]
            exp = load_image(framefiles[i])[10:-10, 15:-15]

            np.testing.assert_allclose(res, exp)

    def test_can_subset_with_bounding_box_starting_negative_indices(self):

        image_dir = self.make_image_dir(10, 64, 63)
        framefiles = [p for p in sorted(image_dir.iterdir()) if p.suffix == '.tif']

        self.assertEqual(len(framefiles), 10)

        vol = image_volume.LazyImageDir(image_dir)
        subvol = vol.crop([(-59, 55), (-53, 49)]).crop([(5, -1), (5, -1)])

        self.assertEqual(subvol.shape, (10, 44, 33))

        # Make sure that each individual index matches
        for i in range(len(framefiles)):
            res = subvol[i, ...]
            exp = load_image(framefiles[i])[10:-10, 15:-15]

            np.testing.assert_allclose(res, exp)
