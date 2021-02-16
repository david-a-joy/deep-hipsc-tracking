""" Define the paths to the training datasets

Finder Class:

* :py:class:`DataFinders`: Define all the finders for each type of data

API Documentation
-----------------

"""

# Imports
import pathlib
from typing import Tuple, Generator

# Types
DataFileGenerator = Generator[pathlib.Path, None, None]
MaskFileGenerator = Generator[Tuple[str, pathlib.Path], None, None]

# Constants
THISDIR = pathlib.Path(__file__).parent
DATADIR = THISDIR.parent / 'data'

IMAGE_SUFFIXES = ('.tif', '.png', '.jpg')

DATA_FINDER_MODE = 'real'  # Default finder mode

# data_finder, mask_finder, mask_type, rootdir
DATA_FINDER_MODES = {
    'training': {
        'data_finder': 'find_data_training',
        'mask_finder': 'find_masks_training',
        'mask_type': 'file',
        'rootdir': DATADIR,
    },
    'training_confocal': {
        'data_finder': 'find_data_training',
        'mask_finder': 'find_masks_training',
        'mask_type': 'file',
        'rootdir': DATADIR / 'training_confocal',
    },
    'training_inverted': {
        'data_finder': 'find_data_training',
        'mask_finder': 'find_masks_training',
        'mask_type': 'file',
        'rootdir': DATADIR / 'training_inverted',
    },
    'real': {
        'data_finder': 'find_data_real',
        'mask_finder': None,
        'mask_type': 'file',
        'rootdir': DATADIR,
    }
}


# Classes


class DataFinders(object):
    """ Class to hold the data finder objects

    :param data_finder_mode:
        The data finder template to load from DATA_FINDER_MODES

    Example:

        finder = DataFinders('real')
        finder.data_finder  # function to load the data files
        finder.mask_finder  # function to load the mask files
        finder.mask_type  # either 'file' or 'selection'

    """

    default_mode = DATA_FINDER_MODE

    modes = DATA_FINDER_MODES

    def __init__(self, data_finder_mode: str = DATA_FINDER_MODE):

        if data_finder_mode in DATA_FINDER_MODES:
            finder_data = DATA_FINDER_MODES[data_finder_mode]
            data_finder = finder_data['data_finder']
            mask_finder = finder_data['mask_finder']

            print(f'Loading data_finder: {data_finder}')
            print(f'Loading mask finder: {mask_finder}')

            if isinstance(data_finder, str):
                self.data_finder = getattr(self, data_finder)
            else:
                self.data_finder = data_finder
            if isinstance(mask_finder, str):
                self.mask_finder = getattr(self, mask_finder)
            else:
                self.mask_finder = mask_finder
            self.mask_type = finder_data['mask_type']
            self.rootdir = finder_data['rootdir']
        else:
            raise KeyError(f'Unknown data finder mode: {data_finder_mode}')

    # Data Finders

    def find_data_real(self, datadir: pathlib.Path) -> DataFileGenerator:
        """ Try to analyze any image under the directory

        :param Path datadir:
            The data directory
        :returns:
            A generator of paths to image files under that directory
        """
        targets = [datadir]
        while targets:
            p = targets.pop()
            if p.name.startswith('.'):
                continue
            if p.is_dir():
                targets.extend(p.iterdir())
                continue
            if p.suffix in IMAGE_SUFFIXES:
                yield p

    def find_data_training(self, datadir: pathlib.Path) -> DataFileGenerator:
        """ Find the training data files

        :param Path datadir:
            The data directory
        :returns:
            A generator of paths to image files under that directory
        """
        for infile in datadir.iterdir():
            if infile.name.endswith('cell.png'):
                yield infile

    # Mask Finders

    def find_masks_training(self, datadir: pathlib.Path) -> MaskFileGenerator:
        """ Find the training mask files

        :param Path datadir:
            The data directory
        :returns:
            A generator of (key, file) tuples for each corresponding mask file
        """
        for infile in datadir.iterdir():
            if infile.name.endswith('dots.png'):
                key = infile.stem.replace('dots', 'cell')
                yield key, infile
