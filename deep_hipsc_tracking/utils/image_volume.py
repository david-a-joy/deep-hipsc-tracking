""" Tools for manipulating image volumes

Base Class:

* :py:class:`LazyImageVolume`: Imitate an iterator for image stacks

Main Classes:

* :py:class:`LazyImageDir`: Lazily load images from an image directory
* :py:class:`LazyImageFile`: Lazily load images from a 3D image

Main Functions:

* :py:func:`find_image_volumes`: Find all image files and dirs under a path
* :py:func:`is_image_dir`: Return True if the path appears to be an image directory
* :py:func:`is_image_file`: Return True if the path appears to be an image file

"""

# Imports
import re
import pathlib
import operator
from typing import Optional, List, Tuple

# 3rd party
import numpy as np

from PIL import Image, ImageSequence

from skimage.transform import resize

# Our own imports
from .image_utils import load_image


# Constants

IMAGE_SUFFIXES = ('.png', '.jpg', '.tif')


# Base Class


class LazyImageVolume(object):
    """ Generic methods for :py:class:`LazyImageDir` and :py:class:`LazyImageFile`

    * :py:meth:`crop`: Generate a new class with a cropped view of the image
    * :py:meth:`transform`: Apply the crops, resizes, transposes, and subset indices

    """

    # Properties

    @property
    def ndim(self) -> int:
        """ Number of dimensions for the n-dimensional array """
        if self._ndim is not None:
            return self._ndim
        img = self._load_image(0)
        self._ndim = img.ndim + 1
        return self._ndim

    @property
    def shape(self) -> Tuple[int]:
        """ Shape of the n-dimensional array """
        if self._shape is not None:
            return self._shape
        shape = self._load_image(0).shape
        if self.bbox is None:
            crop_shape = list(shape)
        else:
            assert len(shape) == len(self.bbox)
            crop_shape = []
            for s, (st, ed) in zip(shape, self.bbox):
                if st < 0:
                    st += s
                if ed < 0:
                    ed += s
                crop_shape.append(ed - st)
        self._shape = tuple([len(self)] + crop_shape)
        return self._shape

    # Generic methods

    def crop(self, bbox: List[Tuple]) -> "LazyImageVolume":
        """ Crop the image volume as it loads

        :param list[tuple] bbox:
            A list of start, end slice indices for each dimension
        :returns:
            The same volume, but with a crop filter applied
        """
        shape = self.shape[1:]
        if self.bbox is None:
            # Default bounding box is just everything
            orig_bbox = [(0, s) for s in shape]
        else:
            orig_bbox = self.bbox
        assert len(bbox) == len(shape)

        # Fuse the two bounding boxes together
        final_bbox = []
        for orig, new in zip(orig_bbox, bbox):
            ost, oed = orig
            nst, ned = new
            if ned < 0:
                ned = oed + ned
            final_bbox.append((ost+nst, ned))

        # Make a copy of the class, ignoring caches
        kwargs = {k: v for k, v in self._kwargs.items()}
        kwargs['bbox'] = final_bbox
        return self.__class__(**kwargs)

    def transform(self, img: np.ndarray, jk: Optional[Tuple[int]]) -> np.ndarray:
        """ Apply the specified transformations to the image

        :param ndarray img:
            The raw 2D image to transform
        :param index jk:
            The sub-slices to extract from the image
        :returns:
            The transformed, cropped, sliced image
        """
        if self.transpose:
            img = img.T

        if self.scale != 1.0:
            rows, cols = img.shape[:2]
            rows = int(np.round(rows * self.scale))
            cols = int(np.round(cols * self.scale))
            img = resize(img, (rows, cols) + img.shape[2:])
        if self.bbox is not None:
            slices = tuple(slice(st, ed) for st, ed in self.bbox)
            assert len(slices) == img.ndim
            img = operator.getitem(img, slices)
        if jk is None:
            return img
        return operator.getitem(img, jk)

    # Magic methods

    def __getitem__(self, idx: Tuple) -> np.ndarray:
        """ Get the image of interest from the table """
        if isinstance(idx, int):
            i = idx
            jk = None
        else:
            i, jk = idx[0], idx[1:]
        img = self._load_image(i)
        return self.transform(img, jk)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.image_path})'

    # Override methods

    def __len__(self) -> int:
        """ Length of the first dimension """
        raise NotImplementedError('Implement the "__len__" magic method')

    def _load_image(self, idx: int) -> np.ndarray:
        """ Load the frame with the first index idx and return it as an array

        :param int idx:
            The index for the frame of the movie/frame in the directory/etc to load
        :returns:
            The image, without applying any of the additional transforms
        """
        raise NotImplementedError('Implement the "_load_image" method')


# Classes


class LazyImageDir(LazyImageVolume):
    """ Lazy load a volume of images from a directory

    :param Path image_dir:
        Directory of images, one per frame
    :param float scale:
        If not 1.0, fraction to scale images up or down by
    :param list[tuple[int]] bbox:
        A list of [(start, end)] indices to crop the image volume with
    :param str suffix:
        A suffix the image name should end with like '_resp' for 'foo_resp.tif'
    :param bool transpose:
        If True, transpose the images when reading them
    """

    def __init__(self,
                 image_dir: pathlib.Path,
                 scale: float = 1.0,
                 bbox: Optional[List[Tuple[int]]] = None,
                 suffix: str = '',
                 transpose: bool = False):
        self.image_dir = image_dir
        self.image_path = self.image_dir

        self.suffix = suffix

        self.scale = scale
        self.bbox = bbox
        self.transpose = transpose

        self._kwargs = {
            'image_dir': image_dir,
            'scale': scale,
            'bbox': bbox,
            'suffix': suffix,
            'transpose': transpose,
        }

        self._shape = None
        self._ndim = None

        self._file_index = None

    @property
    def file_index(self) -> List[pathlib.Path]:
        """ Index the image files

        :returns:
            The list of files in the image directory
        """
        if self._file_index is None:
            file_index = []
            for imgfile in sorted(self.image_dir.iterdir()):
                if imgfile.name.startswith('.'):
                    continue
                if not imgfile.name.endswith(IMAGE_SUFFIXES):
                    continue
                if not imgfile.stem.endswith(self.suffix):
                    continue
                if not imgfile.is_file():
                    continue
                file_index.append(imgfile)
            self._file_index = file_index
        return self._file_index

    def _load_image(self, index: int) -> np.ndarray:
        """ Load a single image from a file """
        return load_image(self.file_index[index])

    def __contains__(self, idx: int) -> bool:
        """ Return true if a file is in the index """
        return idx in self.file_index

    def __len__(self) -> int:
        return len(self.file_index)


class LazyImageFile(LazyImageVolume):
    """ Lazy load a volume of images from a 3D image file

    :param Path image_file:
        3D image file to load
    :param float scale:
        If not 1.0, fraction to scale images up or down by
    :param list[tuple[int]] bbox:
        A list of [(start, end)] indices to crop the image volume with
    :param bool transpose:
        If True, transpose the images when reading them
    """
    def __init__(self,
                 image_file: pathlib.Path,
                 scale: float = 1.0,
                 bbox: Optional[List[Tuple[int]]] = None,
                 transpose: bool = False):
        self.image_file = image_file
        self.image_path = image_file

        self.scale = scale
        self.bbox = bbox
        self.transpose = transpose

        self._kwargs = {
            'image_file': image_file,
            'scale': scale,
            'bbox': bbox,
            'transpose': transpose,
        }

        self._shape = None
        self._ndim = None

        self._img = None
        self._len = None

    @property
    def img(self) -> np.ndarray:
        """ Handle for the image """
        if self._img is not None:
            return self._img
        self._img = ImageSequence.Iterator(Image.open(self.image_file))
        return self._img

    def _load_image(self, index: int) -> np.ndarray:
        """ Load a single image from a 3D image """
        if index < 0:
            index += len(self)
        return np.asarray(self.img[index])

    def __len__(self) -> int:
        if self._len is not None:
            return self._len
        num_frames = 0
        for _ in self:
            num_frames += 1
        self._len = num_frames
        return self._len

# Functions


def is_image_dir(path: pathlib.Path) -> bool:
    """ Detect if a path appears to be an image directory

    :param Path path:
        The path to inspect
    :returns:
        True if the path appears to be an image dir, False otherwise
    """
    if not path.is_dir():
        return False
    subpaths = [p for p in path.iterdir() if not p.name.startswith('.')]
    subfiles = [p for p in subpaths if p.is_file() and p.suffix in IMAGE_SUFFIXES]
    subdirs = [p for p in subpaths if p.is_dir()]

    if len(subdirs) > 0:
        return False
    return len(subfiles) > 1


def is_image_file(path: pathlib.Path) -> bool:
    """ Detect if a path appears to be an image file

    :param Path path:
        The path to inspect
    :returns:
        True if the path appears to be an image file, False otherwise
    """
    if not path.is_file():
        return False
    if path.name.startswith('.'):
        return False
    return path.suffix in IMAGE_SUFFIXES


def find_image_volumes(rootdir: pathlib.Path,
                       pattern: Optional[str] = None,
                       volume_type: str = 'both') -> List[LazyImageVolume]:
    """ Find and load all volumes under a directory

    :param Path rootdir:
        The directory to search
    :param str pattern:
        The regex string for the file **stem** (no file extension) to match
    :param str volume_type:
        Select either "file", "dir", or "both" volumes
    :returns:
        A list of LazyImageFile and LazyImageDir objects
    """
    # Try to match file stems to a pattern
    if pattern is not None:
        if not pattern.startswith('^'):
            pattern = '^' + pattern
        if not pattern.endswith('$'):
            pattern = pattern + '$'
        pattern = re.compile(pattern, re.IGNORECASE)

    # Work out which selectors we were passed
    if volume_type.lower().strip() in ('all', 'both'):
        volume_types = ('file', 'dir')
    elif volume_type.lower().strip() in ('dir', 'directory'):
        volume_types = ('dir', )
    elif volume_type.lower().strip() in ('file', ):
        volume_types = ('file', )
    else:
        raise ValueError(f'Unknown image volume format "{volume_type}"')

    volumes = []
    rootdirs = [rootdir]
    while rootdirs:
        path = rootdirs.pop()

        # See if the path name is something we're looking for
        if path.name.startswith('.'):
            continue
        if pattern is None:
            pattern_matches = True
        else:
            pattern_matches = pattern.match(path.stem) is not None

        # Switch based on the type of image volume
        if pattern_matches and is_image_dir(path):
            if 'dir' in volume_types:
                volumes.append(LazyImageDir(path))
            continue
        if pattern_matches and is_image_file(path):
            if 'file' in volume_types:
                volumes.append(LazyImageFile(path))
            continue

        # Nothing, so recurse into sub-directories
        if path.is_dir():
            rootdirs.extend(path.iterdir())
            continue
    return volumes
