# 3rd party
import numpy as np

# Our own imports
from .helpers import FileSystemTestCase

from deep_hipsc_tracking import frame_tools

# Tests


class TestWriteTiffVolumes(FileSystemTestCase):

    def test_writes_volume_to_file(self):

        indir = self.tempdir / 'test'
        indir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            imagefile = indir / f'{i:02d}.tif'
            img = np.ones((64, 64))*i
            frame_tools.save_image(imagefile, img, cmin=0, cmax=5)

        outfile = self.tempdir / 'test.tif'
        self.assertFalse(outfile.is_file())

        frame_tools.write_tiff_volume(indir, outfile)

        self.assertTrue(outfile.is_file())

    def test_can_write_read_volume(self):

        indir = self.tempdir / 'test'
        indir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            imagefile = indir / f'{i:02d}.tif'
            img = np.ones((64, 64), dtype=np.float32)*(i+1)
            frame_tools.save_image(imagefile, img, cmin=0, cmax=5)

        outfile = self.tempdir / 'test.tif'
        self.assertFalse(outfile.is_file())

        frame_tools.write_tiff_volume(indir, outfile)

        self.assertTrue(outfile.is_file())

        ct = 0
        for i, img in frame_tools.tiff_extractor(outfile):
            self.assertTrue(np.all(np.abs(img - i/5.0) < 1e-2))
            ct += 1
        self.assertEqual(ct, 5)
