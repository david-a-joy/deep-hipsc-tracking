""" Tests for the data finders library """

from deep_hipsc_tracking.model import finders
from ..helpers import FileSystemTestCase

# Constants
DATADIR = finders.DATADIR

# Tests


class TestDataFinders(FileSystemTestCase):

    def test_finds_real(self):

        finder = finders.DataFinders('real')
        self.assertEqual(finder.data_finder, finder.find_data_real)
        self.assertIsNone(finder.mask_finder)
        self.assertEqual(finder.mask_type, 'file')
        self.assertEqual(finder.rootdir, DATADIR)

    def test_finds_training(self):

        finder = finders.DataFinders('training')

        self.assertEqual(finder.data_finder, finder.find_data_training)
        self.assertEqual(finder.mask_finder, finder.find_masks_training)
        self.assertEqual(finder.mask_type, 'file')
        self.assertEqual(finder.rootdir, DATADIR)

    def test_finds_training_inverted(self):

        finder = finders.DataFinders('training_inverted')

        self.assertEqual(finder.data_finder, finder.find_data_training)
        self.assertEqual(finder.mask_finder, finder.find_masks_training)
        self.assertEqual(finder.mask_type, 'file')
        self.assertEqual(finder.rootdir, DATADIR / 'training_inverted')

    def test_finds_training_confocal(self):

        finder = finders.DataFinders('training_confocal')

        self.assertEqual(finder.data_finder, finder.find_data_training)
        self.assertEqual(finder.mask_finder, finder.find_masks_training)
        self.assertEqual(finder.mask_type, 'file')
        self.assertEqual(finder.rootdir, DATADIR / 'training_confocal')

    def test_finds_image_files_real(self):

        datadir = self.tempdir
        subdir = datadir / 'sub'
        subdir.mkdir(parents=True)

        c1 = datadir / '001cell.png'
        c2 = subdir / '002cell.png'
        c1.touch()
        c2.touch()

        finder = finders.DataFinders('real')
        imagefiles = finder.data_finder(datadir)

        self.assertEqual(set(imagefiles), {c1, c2})

    def test_finds_image_files_training(self):

        datadir = self.tempdir
        c1 = datadir / '001cell.png'
        c2 = datadir / '002cell.png'
        c1.touch()
        c2.touch()

        finder = finders.DataFinders('training')
        imagefiles = finder.data_finder(datadir)

        self.assertEqual(set(imagefiles), {c1, c2})

    def test_finds_mask_files_training(self):

        datadir = self.tempdir
        c1 = datadir / '001cell.png'
        c2 = datadir / '002cell.png'
        c1.touch()
        c2.touch()

        m1 = datadir / '001dots.png'
        m2 = datadir / '002dots.png'
        m1.touch()
        m2.touch()

        finder = finders.DataFinders('training')
        maskfiles = finder.mask_finder(datadir)

        self.assertEqual(dict(maskfiles), {'001cell': m1, '002cell': m2})
