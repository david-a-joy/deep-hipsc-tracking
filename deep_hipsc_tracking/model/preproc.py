""" Preprocessing tools

Core Classes:

* :py:class:`ImageResampler`: Resample a collection of images, handling train/test split

Core Functions:

* :py:func:`calculate_peak_image`: Convert a point mask to a peak mask
* :py:func:`check_nvidia`: Make sure the nvidia driver appears to have loaded properly

Sampler Framework:

* :py:class:`BaseSampler`: Base class for implementating samplers
* :py:class:`RandomSampler`: Class for sampling random rotated, scaled crops from images
* :py:class:`CompleteSampler`: Class for sampling strided crops from images with small, pixel-wise steps
* :py:class:`ConvSampler`: Similar to :py:class:`CompleteSampler`, but with strides over images instead of pixels

Utility Functions:

* :py:func:`random_split`: Do a random split over a numpy array
* :py:func:`clamp`: Clamp a value to a range
* :py:func:`pad_with_zeros`: Pad the array with zeros on either side

API Documentation
-----------------

"""

# Standard lib
import time
import json
import random
import pathlib
import itertools
import subprocess
from collections import OrderedDict
from typing import Tuple, Callable, Optional, List

# 3rd party
import numpy as np

from scipy.ndimage import distance_transform_edt
from scipy.ndimage.interpolation import map_coordinates

import matplotlib.pyplot as plt

# Our own imports
from ..utils.image_utils import load_image

# Constants
BATCH_SIZE = 200

NOISE_FRACTION = 0.1
NOISE_TYPE = 'saltpepper'

INPUT_SHAPE = (256, 256, 1)  # Shape of the images to produce
TEST_FRACTION = 0.1  # Percent of samples to use as test data
VALIDATION_FRACTION = 0.1  # Percent of samples to use as validation

TRAINING_SAMPLES = 10000  # Number of training views in an epoch
TEST_SAMPLES = 1000  # Number of test views in an epoch
VALIDATION_SAMPLES = 1000  # Number of validation views in an epoch

ROTATION_RANGE = (-20, 20)  # degrees
SIZE_RANGE = (256, 400)  # pixels square
FLIP_HORIZONTAL = True  # if True, flip the images horizontally
FLIP_VERTICAL = False  # if True, flip the images vertically
ZERO_PADDING = 0  # Number of pixels to zero pad on a side

MASK_TYPE = 'selection'  # Selection or file

IMAGE_LAYOUT = 'theano'  # Figure out which image convention to use

PLOT_SAMPLE_WINDOWS = False  # if True, display the sample window

GPU_NAMES = ['GeForce GTX 970M', 'GeForce GTX 1080']

# Functions


def random_split(arr: np.ndarray,
                 num_samples: int,
                 with_replacement: bool = False) -> Tuple[np.ndarray]:
    """ Make a random split of an array

    :param ndarray arr:
        The array to randomly split
    :param int num_samples:
        The number of samples to select
    :param bool with_replacement:
        If True, sample with replacement
    :returns:
        A random sample of arr with shape == (num_samples),
        the remainder of arr
    """
    if with_replacement:
        return arr[np.random.randint(low=0, high=arr.shape[0],
                                     size=(num_samples, ))], arr
    else:
        shuffle = arr[np.random.permutation(arr.shape[0])]
        return np.split(shuffle, (num_samples, ))


def clamp(val: float, minval: float, maxval: float) -> float:
    """ Clamp a value to within a range

    :param float val:
        The value to clamp
    :param float minval:
        The minimum value
    :param float maxval:
        The maximum value
    :returns:
        The value, clamped to the range specified
    """
    if val < minval:
        return minval
    if val > maxval:
        return maxval
    return val


def pad_with_zeros(img: np.ndarray, padding: int) -> np.ndarray:
    """ Pad a 2D image with 0s

    :param ndarray img:
        The 2D image to pad
    :param int padding:
        The number of pixels to zero-pad on either side
    :returns:
        A new 2D array with the appropriate padding
    """

    if padding <= 0:
        return img

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        was_2d = True
    else:
        was_2d = False
    rows, cols, colors = img.shape

    new_img = np.zeros((rows + padding*2, cols + padding*2, colors),
                       dtype=img.dtype)
    new_img[padding:-padding, padding:-padding, :] = img
    if was_2d:
        new_img = new_img[:, :, 0]
    return new_img


def calculate_peak_image(target_img: np.ndarray,
                         img_rows: int = 32,
                         img_cols: int = 32,
                         zero_padding: int = 32,
                         peak_sharpness: float = 8):
    """ Calculate the peaks from a target image

    :param ndarray target_img:
        The user selected points image to convert to peaks
    :param int img_rows:
        The rows for the imaging window
    :param int img_cols:
        The cols for the imaging window
    :param int zero_padding:
        How much to zero pad the target image
    :param float peak_sharpness:
        How sharp of peaks to make
    :returns:
        The new peak image
    """

    # FIXME: We don't really use the zero_padding anymore
    sampling = (peak_sharpness / img_rows, peak_sharpness / img_cols)

    rows, cols = target_img.shape[:2]
    if target_img.ndim == 3:
        target_img = np.mean(target_img, axis=2)
    assert target_img.ndim == 2

    target_peaks = 1.0 - distance_transform_edt(target_img == 0, return_distances=True, sampling=sampling)
    target_peaks[target_peaks < 0] = 0
    target_peaks[target_peaks > 1] = 1

    return target_peaks


def find_raw_data(datadir: pathlib.Path):
    """ Find all the data files under datadir

    :param Path datadir:
        The path to the base of the raw data directory
    :returns:
        A generator yielding raw data files to process
    """

    for subdir in datadir.iterdir():
        if subdir.name.startswith('.'):
            continue
        if not subdir.is_dir():
            continue
        for imgfile in subdir.iterdir():
            if imgfile.name.startswith('.'):
                continue
            if imgfile.suffix not in ('.jpg', '.tif', '.png'):
                continue
            if not imgfile.is_file():
                continue
            yield imgfile


def plot_sampler(img: np.ndarray,
                 verts: np.ndarray,
                 rot_verts: np.ndarray,
                 shift_verts: np.ndarray):
    """ Plot the steps in the sampler operation

    :param ndarray img:
        The m x n image to plot
    :param ndarray verts:
        The n x 4 list of original vertices to plot
    :param ndarray rot_verts:
        The n x 4 list of rotated vertices to plot
    :param ndarray shift_verts:
        The n x 4 list of rotated and shifted vertices to plot
    """

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    assert img.ndim == 3

    rows, cols, colors = img.shape
    cx = cols / 2
    cy = rows / 2

    if colors == 1:
        img = np.concatenate([img, img, img], axis=2)
    assert img.shape[2] == 3

    img = np.round(img)
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)

    fig, axes = plt.subplots(1, 1)
    ax0 = axes

    ax0.imshow(img)
    ax0.plot(verts[0, [0, 1, 2, 3, 0]] + cx,
             verts[1, [0, 1, 2, 3, 0]] + cy,
             '-g', linewidth=3)
    ax0.plot(rot_verts[0, [0, 1, 2, 3, 0]] + cx,
             rot_verts[1, [0, 1, 2, 3, 0]] + cy,
             '-r', linewidth=3)
    ax0.plot(shift_verts[0, [0, 1, 2, 3, 0]],
             shift_verts[1, [0, 1, 2, 3, 0]],
             '-b', linewidth=3)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.axis([0, cols, rows, 0])

    plt.show()


def resample_in_box(img: np.ndarray,
                    scale: float,
                    rotation: np.ndarray,
                    translation: np.ndarray,
                    input_shape: Tuple[int] = INPUT_SHAPE):
    """ Resample the image inside a box

    :param ndarray img:
        The image to resample (n x m)
    :param float scale:
        The scale factor for the images
    :param ndarray rotation:
        The 2x2 rotation matrix to rotate the image by
    :param ndarray translation:
        The 2x1 translation matrix to move the image by
    :returns:
        The shifted, rotated, translated image
    """

    was_2d = False
    if img.ndim == 2:
        was_2d = True
        img = img[:, :, np.newaxis]
    assert img.ndim == 3
    rows, cols, colors = img.shape

    if isinstance(input_shape, (int, float)):
        xshape = input_shape
        yshape = input_shape
        cshape = colors
    elif len(input_shape) == 1:
        xshape = yshape = input_shape[0]
        cshape = colors
    elif len(input_shape) == 2:
        xshape, yshape = input_shape
        cshape = colors
    elif len(input_shape) == 3:
        xshape, yshape, cshape = input_shape
    else:
        raise ValueError(f'Cannot parse input shape: {input_shape}')

    # Rearrange the image for color vs grayscale
    is_color_grayscale = False
    if cshape != colors:
        if cshape == 1 and colors == 3:
            img = np.mean(img, axis=2)[:, :, np.newaxis]
        elif cshape == 3 and colors == 1:
            is_color_grayscale = True
            cshape = 1
        else:
            raise ValueError(f'No conversion from {colors} colors to {cshape}')

    x_delta = np.linspace(0, scale - 1, xshape) - scale/2
    y_delta = np.linspace(0, scale - 1, yshape) - scale/2

    xx_delta, yy_delta = np.meshgrid(x_delta, y_delta)
    vert_delta = np.stack([xx_delta.flatten(),
                           yy_delta.flatten()])

    vert_delta = rotation @ vert_delta + translation
    vert_delta = vert_delta[[1, 0], :]

    final_img = []
    for ci in range(cshape):
        out_img = map_coordinates(img[:, :, ci], vert_delta, order=1)
        final_img.append(np.reshape(out_img, (xshape, yshape)))

    if is_color_grayscale:
        cshape = 3
        final_img = [final_img[0] for _ in range(3)]

    final_img = np.stack(final_img, axis=2)
    if was_2d and cshape == 1:
        final_img = final_img[:, :, 0]
    return final_img


def predict_with_steps(img: np.ndarray,
                       detector,
                       img_shape: Tuple[int],
                       response_shape: Tuple[int],
                       overlap: Tuple[int] = 0):
    """ Predict all the steps on the image

    :param ndarray img:
        The 2D image to predict results on
    :param object detector:
        The keras detector object with a ``.predict()`` method
    :param tuple[int] img_shape:
        The rows, cols shape for the input to the detector
    :param tuple[int] response_shape:
        The rows, cols shape for the output of the detector
    :param tuple[int] overlap:
        How many pixels of overlap (minimum) in the output in each direction
    :returns:
        The response image, the same shape as the input image
    """

    # Crush the input image so it's the right shape
    img = np.squeeze(img)
    assert img.ndim == 2
    rows, cols = img.shape

    img_rows, img_cols = img_shape
    resp_rows, resp_cols = response_shape

    if isinstance(overlap, int):
        overlap_rows = overlap_cols = overlap
    else:
        overlap_rows, overlap_cols = overlap

    assert overlap_rows*2 < resp_rows
    assert overlap_cols*2 < resp_cols

    resp_small_rows = resp_rows - overlap_rows*2
    resp_small_cols = resp_cols - overlap_cols*2

    # Pad the image so that the output covers the entire image
    num_row_steps = int(np.ceil(rows / resp_small_rows))
    num_col_steps = int(np.ceil(cols / resp_small_cols))

    resp_row_padding = resp_small_rows * num_row_steps - rows
    resp_col_padding = resp_small_cols * num_col_steps - cols
    assert resp_row_padding >= 0
    assert resp_col_padding >= 0

    resp_row_left = resp_row_padding // 2 + overlap_rows
    resp_row_right = resp_row_left + rows
    resp_col_left = resp_col_padding // 2 + overlap_cols
    resp_col_right = resp_col_left + cols

    out_rows = rows + resp_row_padding
    out_cols = cols + resp_col_padding

    # Create blocks to store the output images
    final_response = np.zeros((out_rows + overlap_rows*2, out_cols + overlap_cols*2))
    final_counts = np.zeros((out_rows + overlap_rows*2, out_cols + overlap_cols*2))

    # Now pad the image so that the corresponding inputs fit inside it
    in_row_padding = img_rows - resp_small_rows
    in_col_padding = img_cols - resp_small_cols

    assert in_row_padding >= 0
    assert in_col_padding >= 0

    in_rows = out_rows + in_row_padding
    in_cols = out_cols + in_col_padding

    in_row_left = in_row_padding // 2 + resp_row_padding // 2
    in_row_right = in_row_left + rows
    in_col_left = in_col_padding // 2 + resp_col_padding // 2
    in_col_right = in_col_left + cols

    pad_img = np.zeros((in_rows, in_cols))
    pad_img[in_row_left:in_row_right, in_col_left:in_col_right] = img

    # Now, break that padded image into output image sized chunks
    row_st = np.linspace(0, out_rows - resp_small_rows, num_row_steps)
    col_st = np.linspace(0, out_cols - resp_small_cols, num_col_steps)
    row_ed = row_st + resp_rows
    col_ed = col_st + resp_cols

    for i, j in itertools.product(range(num_row_steps), range(num_col_steps)):
        # We constructed the map such that the input FOV exactly matches the output FOV
        ox_st = int(row_st[i])
        ox_ed = int(row_ed[i])
        oy_st = int(col_st[j])
        oy_ed = int(col_ed[j])

        ix_st, iy_st = ox_st, oy_st
        ix_ed = ix_st + img_rows
        iy_ed = iy_st + img_cols

        subset = pad_img[ix_st:ix_ed, iy_st:iy_ed]
        assert subset.shape == (img_rows, img_cols)

        response = detector.predict(subset[np.newaxis, :, :, np.newaxis])
        assert response.shape[1:3] == (resp_rows, resp_cols)

        final_response[ox_st:ox_ed, oy_st:oy_ed] += response[0, :, :, 0]
        final_counts[ox_st:ox_ed, oy_st:oy_ed] += 1

    # Calculate the mean and strip off the padding
    return (final_response / final_counts)[resp_row_left:resp_row_right, resp_col_left:resp_col_right]


def check_nvidia(verify_names: bool = False):
    """ Make sure the GPU driver is loaded and seems to be working

    :param bool verify_names:
        If True, verify that the GPUs are the specific cards we expect
    """

    cmd = ['nvidia-smi']
    res = subprocess.check_output(cmd).decode('utf-8')

    if 'CUDA Version:' not in res:
        print('#' * 10 + ' Cannot detect CUDA driver ' + '#' * 10)
        print(res)
        raise RuntimeError('Cannot detect CUDA driver')

    if verify_names and not any([gpu in res for gpu in GPU_NAMES]):
        print('#' * 10 + ' Cannot open video card ' + '#' * 10)
        print(res)
        raise RuntimeError('Cannot open video card')

    import tensorflow as tf
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpu_list)}")
    for gpu in gpu_list:
        print(f'* {gpu}')
    assert len(gpu_list) > 0

# Classes


class LRUDict(OrderedDict):
    """ LRU Dictionary """
    def __init__(self, *args, maxsize=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxsize = 0 if maxsize is None else maxsize
        self.purge()

    def purge(self):
        """Removes expired or overflowing entries."""
        if self.maxsize > 0:
            # pop until maximum capacity is reached
            overflowing = max(0, len(self) - self.maxsize)
            for _ in range(overflowing):
                self.popitem(last=False)

    def __getitem__(self, key):
        # retrieve item
        value = super().__getitem__(key)
        # update lru time
        super().__setitem__(key, value)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if self.maxsize < 1:
            return
        super().__setitem__(key, value)
        self.purge()


class ImageResampler(object):
    """ Image resampler class

    Handles the shuffling and the train/test split of the data

    Basic Usage:

    .. code-block:: python

        sampler = ImageResampler()
        sampler.set_data_loader(datadir='path/to/data',
                                data_finder=data_finder_func)
        sampler.set_mask_loader(masks=None,
                                mask_finder=mask_finder_func,
                                mask_type='selection')
        sampler.load_files()
        sampler.calc_train_test_split()
        sampler.load_samplers(sampler_type='random')

        x_train = sampler.train_data
        x_test = sampler.test_data
        x_validation = sampler.validation_data

    """

    def set_data_loader(self,
                        datadir: Optional[pathlib.Path] = None,
                        data_finder: Callable = find_raw_data):
        """ Set the data loader function

        :param Path datadir:
            The raw datadir to process
        :param function data_finder:
            A function to actually find the image files
        """
        if datadir is not None:
            datadir = pathlib.Path(datadir).resolve()
            datadir.mkdir(parents=True, exist_ok=True)
        self.datadir = datadir
        self.data_finder = data_finder

    def set_mask_loader(self,
                        masks: Optional[List[pathlib.Path]] = None,
                        mask_finder: Optional[Callable] = None,
                        mask_type: str = MASK_TYPE):
        """ Set the mask loader function

        :param masks:
            The masks to load for resampling
        :param function mask_finder:
            A function to find the masks in datadir
        :param str mask_type:
            What kinds of masks are being loaded (either 'selection' or 'file')
        """

        if mask_finder is not None and masks is not None:
            raise ValueError('Provide only one of `mask_finder` or `masks`')

        self.masks = masks
        self.mask_finder = mask_finder
        self.mask_type = mask_type

    def load_files(self):
        """ Load the file and mask data """

        if self.datadir is None:
            files = []
        else:
            files = list(self.data_finder(self.datadir))
        print(f'Loaded {len(files)} files...')

        if getattr(self, 'mask_finder', None) is not None and self.datadir is not None:
            masks = dict(self.mask_finder(self.datadir))
        else:
            masks = getattr(self, 'masks', None)

        if masks is not None:
            files = [f for f in files if f.stem in masks]
            print(f'Loaded {len(files)} masks...')

        self.files = files
        self.masks = masks

    def calc_train_test_split(self,
                              test_fraction: float = TEST_FRACTION,
                              validation_fraction: float = VALIDATION_FRACTION):
        """ Calculate the train/test split

        :param float test_fraction:
            The fraction of train/test/validation files to use for testing
        :param float validation_fraction:
            The fraction of train/test/validation files to use for final validation
        """
        if test_fraction is None:
            test_fraction = 0.0
        if validation_fraction is None:
            validation_fraction = 0.0
        assert test_fraction + validation_fraction < 1.0

        # Discrete split the files into bins
        num_files = len(self.files)
        random.shuffle(self.files)

        files = self.files
        split_point = int(np.round(num_files*test_fraction))
        self.test_files, files = files[:split_point], files[split_point:]

        split_point = int(np.round(num_files*validation_fraction))
        self.validation_files, files = files[:split_point], files[split_point:]
        self.train_files = files

    def load_train_test_split(self, split_file: pathlib.Path):
        """ Load the old split from a file

        :param Path split_file:
            The split JSON file
        """

        split_file = pathlib.Path(split_file)
        with split_file.open('rt') as fp:
            split = json.load(fp)
        self.train_files = [pathlib.Path(p) for p in split.get('train_files', [])]
        self.test_files = [pathlib.Path(p) for p in split.get('test_files', [])]
        self.validation_files = [pathlib.Path(p) for p in split.get('validation_files', [])]

    def save_train_test_split(self, split_file: pathlib.Path):
        """ Save a train/test split to a file

        :param Path split_file:
            The split JSON file
        """
        split = {
            'train_files': [str(p) for p in self.train_files],
            'test_files': [str(p) for p in self.test_files],
            'validation_files': [str(p) for p in self.validation_files],
        }
        split_file.parent.mkdir(exist_ok=True, parents=True)
        with split_file.open('wt') as fp:
            json.dump(split, fp)

    def is_split_under_datadir(self, datadir: pathlib.Path):
        """ Test if the split is all under a specific directory

        :param Path datadir:
            The directory the files should live under
        :returns:
            True if all train/validation files are there, False otherwise
        """
        num_invalid = {
            'train_files': 0,
            'test_files': 0,
            'validation_files': 0,
        }
        for attr in num_invalid:
            for imagefile in getattr(self, attr):
                if imagefile.parent != datadir:
                    num_invalid[attr] += 1
        for attr, value in num_invalid.items():
            if value > 0:
                print(f'Got {value} bad {attr} under {datadir}')
        return all(v == 0 for v in num_invalid.values())

    def load_samplers(self, sampler_type='random',
                      sample_with_replacement=False,
                      image_layout=IMAGE_LAYOUT,
                      batch_size=BATCH_SIZE,
                      input_shape=INPUT_SHAPE,
                      size_range=SIZE_RANGE,
                      rotation_range=ROTATION_RANGE,
                      flip_horizontal=FLIP_HORIZONTAL,
                      flip_vertical=FLIP_VERTICAL,
                      zero_padding=ZERO_PADDING,
                      noise_type=NOISE_TYPE,
                      noise_fraction=NOISE_FRACTION,
                      training_samples=TRAINING_SAMPLES,
                      test_samples=TEST_SAMPLES,
                      validation_samples=VALIDATION_SAMPLES,
                      cache_size=None):
        """ Load the sampler objects

        :param str sampler_type:
            The type of the sampler, or a subclass of BaseSampler
        :param bool sample_with_replacement:
            If True, sample the data with replacement
        """
        sampler_cls = BaseSampler.get_sampler_cls(sampler_type)

        if isinstance(input_shape, (int, float)):
            input_shape = (input_shape, input_shape, 1)
        elif len(input_shape) == 1:
            input_shape = (input_shape[0], input_shape[0], 1)
        elif len(input_shape) == 2:
            input_shape = (input_shape[0], input_shape[1], 1)
        assert len(input_shape) == 3
        assert input_shape[2] in (1, 3)

        if self.test_files is None:
            self.test_data = None
        else:
            random.shuffle(self.test_files)
            self.test_data = sampler_cls(
                self.test_files,
                masks=getattr(self, 'masks', None),
                mask_type=getattr(self, 'mask_type', MASK_TYPE),
                image_layout=image_layout,
                batch_size=batch_size,
                input_shape=input_shape,
                size_range=size_range,
                rotation_range=rotation_range,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                zero_padding=zero_padding,
                noise_type=noise_type,
                noise_fraction=noise_fraction,
                cache_size=cache_size,
                sample_with_replacement=sample_with_replacement)

        if self.validation_files is None:
            self.validation_data = None
        else:
            random.shuffle(self.validation_files)
            self.validation_data = sampler_cls(
                self.validation_files,
                masks=getattr(self, 'masks', None),
                mask_type=getattr(self, 'mask_type', MASK_TYPE),
                image_layout=image_layout,
                batch_size=batch_size,
                input_shape=input_shape,
                size_range=size_range,
                rotation_range=rotation_range,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                zero_padding=zero_padding,
                noise_type=noise_type,
                noise_fraction=noise_fraction,
                cache_size=cache_size,
                sample_with_replacement=sample_with_replacement)

        random.shuffle(self.train_files)
        self.train_data = sampler_cls(
            self.train_files,
            masks=getattr(self, 'masks', None),
            mask_type=getattr(self, 'mask_type', MASK_TYPE),
            image_layout=image_layout,
            batch_size=batch_size,
            input_shape=input_shape,
            size_range=size_range,
            rotation_range=rotation_range,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            zero_padding=zero_padding,
            noise_type=noise_type,
            noise_fraction=noise_fraction,
            cache_size=cache_size,
            sample_with_replacement=sample_with_replacement)

        if self.validation_data is None:
            num_validation = 0
        else:
            num_validation = len(self.validation_data)
        if self.test_data is None:
            num_test = 0
        else:
            num_test = len(self.test_data)
        num_training = len(self.train_data)
        num_total = num_training + num_test + num_validation
        if num_total < 1:
            print('No valid samples found, assuming no training?!?')
        else:
            print('Using {:.1%} of the data for training'.format(num_training/num_total))
            print('Using {:.1%} of the data for testing'.format(num_test/num_total))
            print('Using {:.1%} of the data for validation'.format(num_validation/num_total))
            print('')
            print('{} Training files'.format(num_training))
            print('{} Test files'.format(num_test))
            print('{} Validation files'.format(num_validation))

            self.samples_per_epoch = min([training_samples, len(self.train_data)])

            if self.test_data is None:
                self.test_samples_per_epoch = None
            else:
                self.test_samples_per_epoch = min([test_samples, len(self.test_data)])

            if self.validation_data is None:
                self.val_samples_per_epoch = None
            else:
                self.val_samples_per_epoch = min([validation_samples, len(self.validation_data)])


# Sampler Classes


class BaseSampler(object):
    """ Base class for the various samplers """

    def __init__(self, files,
                 masks=None,
                 mask_type=MASK_TYPE,
                 image_layout=IMAGE_LAYOUT,
                 batch_size=BATCH_SIZE,
                 input_shape=INPUT_SHAPE,
                 size_range=SIZE_RANGE,
                 rotation_range=ROTATION_RANGE,
                 flip_horizontal=FLIP_HORIZONTAL,
                 flip_vertical=FLIP_VERTICAL,
                 zero_padding=ZERO_PADDING,
                 noise_type=NOISE_TYPE,
                 noise_fraction=NOISE_FRACTION,
                 cache_size=None,
                 expand_mask=1.0,
                 sample_with_replacement=False,
                 seed=None):
        self.files = [pathlib.Path(f) for f in files]
        self.masks = masks
        self.mask_type = mask_type

        self.indicies = np.arange(len(files))

        self.image_layout = image_layout

        if cache_size is None or cache_size < 0:
            cache_size = 0
        self.cache_size = cache_size
        self.image_cache = LRUDict(maxsize=cache_size)

        self.rnd = np.random.RandomState(seed)

        self.batch_size = batch_size

        self.input_shape = input_shape
        self.size_range = size_range
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.zero_padding = zero_padding

        self.noise_type = noise_type
        self.noise_fraction = noise_fraction
        self.noise_salt_vs_pepper = 0.5

        self.expand_mask = expand_mask

        self.sample_with_replacement = sample_with_replacement

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        return self

    def __next__(self):
        return self.resample_all(self.batch_size)

    next = __next__

    @classmethod
    def get_sampler_cls(cls, sampler_type):
        """ Do a dynamic lookup on the sampler class name

        :param str sampler_type:
            Either a subclass of BaseSampler, or the name of a sampler class
        :returns:
            The appropriate sampler class
        """
        if isinstance(sampler_type, type) and issubclass(sampler_type, BaseSampler):
            return sampler_type
        samplers = {subcls.__name__.lower()[:-len('sampler')]: subcls
                    for subcls in cls.__subclasses__()
                    if subcls.__name__.lower().endswith('sampler')}
        sampler_type = sampler_type.lower()
        if sampler_type.endswith('sampler'):
            sampler_type = sampler_type[:-len('sampler')]
        return samplers[sampler_type]

    @property
    def shape(self):
        # Stupid Tensorflow convention
        if self.image_layout == 'theano':
            return (self.batch_size, self.input_shape[2], self.input_shape[0], self.input_shape[1])
        else:
            return (self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2])

    def load_random_image(self):
        """ Load a random image from the set """
        image_idx = np.random.randint(0, len(self.files))
        image_file = self.files[int(image_idx)]

        img = self.load_file(image_file)
        if self.masks is None:
            return image_file, img
        mask = self.load_mask(image_file, img)
        return image_file, img, mask

    def load_file(self, imgfile):
        """ Load the image file """
        # Try the cache
        if imgfile.name in self.image_cache:
            return self.image_cache[imgfile.name]

        try:
            img = load_image(imgfile, ctype='color')
        except FileNotFoundError:
            return None

        # Convert to grayscale
        if self.input_shape[2] == 1:
            if img.ndim == 3 and img.shape[2] > 1:
                img = np.mean(img, axis=2)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        assert img.ndim == 3
        img = img / 255.0

        self.image_cache[imgfile.name] = img
        return img

    def load_mask(self, *args, **kwargs):
        """ Load the mask for the image """

        if self.mask_type == 'selection':
            return self.load_mask_selection(*args, **kwargs)
        elif self.mask_type == 'file':
            return self.load_mask_file(*args, **kwargs)
        else:
            raise ValueError('Unknown mask type: {}'.format(self.mask_type))

    def load_mask_file(self, imgfile, img):
        """ Load the mask from a numpy file """

        rows, cols = img.shape[:2]

        mask = np.zeros((rows, cols, 1), dtype=np.bool)

        if self.masks is None:
            return mask

        mask_file = self.masks.get(imgfile.stem, None)
        if mask_file is None:
            return mask

        if mask_file.suffix in ('.npz', ):
            real_mask = np.load(str(mask_file))['mask']
        elif mask_file.suffix in ('.png', '.tif', '.jpg'):
            real_mask = load_image(mask_file, ctype='gray')
        else:
            raise OSError('Unknown mask file type: {}'.format(mask_file.suffix))

        if real_mask.shape[:2] != (rows, cols):
            err = 'Mask {} from Image {}, expected {}x{}, got {}x{}'
            err = err.format(mask_file, imgfile,
                             rows, cols,
                             real_mask.shape[0], real_mask.shape[1])
            raise ValueError(err)
        if real_mask.ndim == 2:
            real_mask = real_mask[:, :, np.newaxis]
        if real_mask.ndim != 3:
            raise ValueError('Got non-3D mask: {}'.format(real_mask.shape))
        return real_mask

    def load_mask_selection(self, imgfile, img):
        """ Load the mask for the image

        From a set of selection ROIs"""

        rows, cols = img.shape[:2]

        mask = np.zeros((rows, cols, 1), dtype=np.bool)

        if self.masks is None:
            return mask

        selections = self.masks.get(imgfile.stem, [])
        for x0, y0, x1, y1 in selections:
            # Expand selection so we can segment
            xctr = (x0 + x1)/2
            xrng = abs(x1 - x0) * self.expand_mask
            x0 = xctr - xrng/2
            x1 = xctr + xrng/2

            y0 = 1 - y0
            y1 = 1 - y1

            yctr = (y0 + y1)/2
            yrng = abs(y1 - y0) * self.expand_mask
            y0 = yctr - yrng/2
            y1 = yctr + yrng/2

            xst = int(round(cols * x0))
            yst = int(round(rows * y0))
            xed = int(round(cols * x1))
            yed = int(round(rows * y1))

            yst = clamp(yst, 0, rows-1)
            yed = clamp(yed, 0, rows)
            xst = clamp(xst, 0, cols-1)
            xed = clamp(xed, 0, cols)

            mask[yst:yed, xst:xed, :] = True

        return mask

    def resample_all(self, num_images):
        """ Extract num_images worth of patches """
        raise NotImplementedError('Implement a sampling function')


class ConvSampler(BaseSampler):
    """ Sample an image into large blocks for convolutional nets """

    def resample_all(self, num_images):
        """ Sample num_images whole images from the current database """

        raw_images = []
        raw_masks = []
        missing_images = []

        print('Resampling {} data points...'.format(self.batch_size))
        t0 = time.time()

        self.current_index = getattr(self, 'current_index', 0)

        while len(raw_images) < num_images and len(self.files) > 0:

            if self.current_index >= len(self.files):
                self.current_index = 0

            img = self.load_file(self.files[self.current_index])
            if img is None:
                missing_images.append(self.files[self.current_index])
                self.files.pop(self.current_index)
                continue
            raw_images.append(img)
            if self.masks is not None:
                mask = self.load_mask(self.files[self.current_index], img)
                raw_masks.append(mask)

        if len(raw_images) < num_images:
            raise ValueError('Not enough files, have {} need {}'.format(len(self.files), num_images))

        print('Finished in {:1.2f} seconds'.format(time.time() - t0))

        if self.masks is None:
            return np.stack(raw_images, axis=0)
        else:
            return (np.stack(raw_images, axis=0),
                    np.stack(raw_masks, axis=0))


class CompleteSampler(BaseSampler):
    """ Sample an image by sampling every sub-patch of a given size """

    def slice_next(self, num_images, *args):
        """ Slice the next n-slices out of an image or image set """

        args = tuple(pad_with_zeros(a.copy(), self.zero_padding) for a in args)

        rows, cols = args[0].shape[:2]
        for a in args:
            assert a.shape[:2] == (rows, cols)

        current_index = self.current_index
        current_slice = self.current_slice

        img_rows, img_cols, img_colors = self.input_shape

        if img_rows > rows or img_cols > cols:
            err = 'Cannot slice {}x{} from image {}x{}'
            err = err.format(img_rows, img_cols, rows, cols)
            raise ValueError(err)

        row_steps = rows - img_rows + 1
        col_steps = cols - img_cols + 1

        num_images = min([row_steps * col_steps - current_slice, num_images])

        out_args = []
        for _ in args:
            out_args.append([])

        num_steps = 0
        for idx in range(current_slice, num_images + current_slice):
            i = idx // col_steps
            j = idx % col_steps

            for arg, oarg in zip(args, out_args):
                oslice = arg[i:i+img_rows, j:j+img_cols, ...]
                if oslice.ndim == 3:
                    if oslice.shape[2] == 3 and img_colors == 1:
                        oslice = np.mean(oslice, axis=2)[..., np.newaxis]
                oarg.append(oslice)
            num_steps += 1

        current_slice += num_steps
        if current_slice >= row_steps * col_steps:
            current_slice = 0
            current_index += 1
        self.current_slice = current_slice
        self.current_index = current_index

        final_args = []
        for oargs in out_args:
            try:
                final_args.append(np.stack(oargs, axis=0))
            except ValueError:
                # for a in args:
                #     print(a.shape)
                # for oa in oargs:
                #     print(oa.shape)
                raise

        if len(final_args) == 1:
            return final_args[0]
        return final_args

    def resample_all(self, num_images):
        """ Sample num_images patches from the current database """

        raw_images = []
        raw_masks = []
        missing_images = []

        print('Resampling {} data points...'.format(self.batch_size))
        t0 = time.time()

        self.current_index = getattr(self, 'current_index', 0)
        self.current_slice = getattr(self, 'current_slice', 0)

        while len(raw_images) < num_images and len(self.files) > 0:

            if self.current_index >= len(self.files):
                self.current_index = 0

            img = self.load_file(self.files[self.current_index])
            if img is None:
                missing_images.append(self.files[self.current_index])
                self.files.pop(self.current_index)
            else:
                need_images = num_images - len(raw_images)
                if self.masks is None:
                    img_slices = self.slice_next(need_images, img)
                    raw_images.extend(img_slices)
                else:
                    mask = self.load_mask(self.files[self.current_index], img)
                    img_slices, mask_slices = self.slice_next(need_images, img, mask)

                    raw_images.extend(img_slices)
                    raw_masks.extend(mask_slices)

        if len(raw_images) < num_images:
            raise ValueError('Not enough files, have {} need {}'.format(len(self.files), num_images))

        print('Finished in {:1.2f} seconds'.format(time.time() - t0))

        if self.masks is None:
            return np.stack(raw_images, axis=0)
        else:
            return (np.stack(raw_images, axis=0),
                    np.stack(raw_masks, axis=0))


class RandomSampler(BaseSampler):
    """ Sample an image by sampling distored crops of the image """

    def _add_saltpepper_noise(self, x_train):
        # Salt and pepper noise

        samples, rows, cols, _ = x_train.shape

        noise_fraction = self.noise_fraction
        salt_vs_pepper = self.noise_salt_vs_pepper

        x_noisy = x_train.copy()
        num_salt = int(np.ceil(rows*cols*noise_fraction*salt_vs_pepper))
        for i in range(samples):
            x_coords = np.random.randint(0, rows, size=num_salt)
            y_coords = np.random.randint(0, cols, size=num_salt)
            x_noisy[i, x_coords, y_coords, 0] = 1

        num_pepper = int(np.ceil(rows*cols*noise_fraction*(1.0-salt_vs_pepper)))
        for i in range(samples):
            x_coords = np.random.randint(0, rows, size=num_pepper)
            y_coords = np.random.randint(0, cols, size=num_pepper)
            x_noisy[i, x_coords, y_coords, 0] = 0

        return x_noisy

    def resample_all(self, num_images):
        """ Resample all the images we need """

        indicies = self.indicies
        raw_images = []
        raw_masks = []
        missing_images = []

        print('Resampling {} data points...'.format(self.batch_size))
        t0 = time.time()

        while len(raw_images) < num_images and len(indicies) > 0:
            batch_indicies, indicies = random_split(indicies, num_images,
                                                    with_replacement=self.sample_with_replacement)
            for idx in batch_indicies:
                img = self.load_file(self.files[idx])
                if img is None:
                    missing_images.append(self.files[idx])
                else:
                    if self.masks is not None:
                        raw_masks.append(self.load_mask(self.files[idx], img))
                    raw_images.append(img)

        # Clear out any invalid images
        if len(missing_images) > 0:
            print('Clearing {} invalid images'.format(len(missing_images)))
            self.files = [f for f in self.files if f not in missing_images]
            self.indicies = np.arange(len(self.files))

        if len(raw_images) < num_images:
            raise ValueError('Not enough files, have {} need {}'.format(len(self.files), num_images))

        images = []
        masks = []
        for i, img in enumerate(raw_images):
            if self.masks is None:
                images.append(self.resample_image(img))
            else:
                rimg, rmask = self.resample_image(img, raw_masks[i])
                images.append(rimg)
                masks.append(rmask)

        print('Finished in {:1.2f} seconds'.format(time.time() - t0))
        if self.cache_size > 0:
            print('Cache size {} of {}'.format(len(self.image_cache), self.cache_size))

        if self.masks is None:
            return np.array(images)
        else:
            return np.array(images), np.array(masks)

    def add_noise(self, x_train):
        """ Add some noise to the training data """

        if self.noise_type == 'none':
            return x_train
        elif self.noise_type == 'saltpepper':
            return self._add_saltpepper_noise(x_train)
        elif self.noise_type == 'gaussian':
            return self._add_gaussian_noise(x_train)
        else:
            raise ValueError('Unknown noise type: {}'.format(self.noise_type))

    def resample_image(self, img, *args, **kwargs):
        """ Resample the image

        :param np.array img:
            The image to resample
        :param float size:
            The size of the sqaure sample region to extract
        :param float theta:
            The angle (in degrees) to rotate the sample region
        :param np.array shift:
            The x, y shifts to apply to the sample region
        :returns:
            The sample of the image
        """
        img = pad_with_zeros(img, self.zero_padding)
        args = tuple(pad_with_zeros(a, self.zero_padding) for a in args)

        rows, cols, colors = img.shape

        # Allow deterministic selection of parameters
        size = kwargs.pop('size', None)
        theta = kwargs.pop('theta', None)
        shift = kwargs.pop('shift', None)
        flip_horizontal = kwargs.pop('flip_horizontal', None)
        flip_vertical = kwargs.pop('flip_vertical', None)

        # Randomize the horizontal and vertical flips
        if flip_horizontal is None:
            flip_horizontal = self.flip_horizontal and self.rnd.rand(1) >= 0.5

        if flip_vertical is None:
            flip_vertical = self.flip_vertical and self.rnd.rand(1) >= 0.5

        # Figure out the box size
        if size is None:
            size_min, size_max = self.size_range
            size_rng = size_max - size_min
            size = self.rnd.rand(1) * size_rng + size_min

        # Figure out the rotation angle
        if theta is None:
            theta_min, theta_max = self.rotation_range
            theta_rng = theta_max - theta_min
            theta = self.rnd.rand(1) * theta_rng + theta_min
        theta = theta / 180 * np.pi

        # Make a rotated bounding box
        R = np.squeeze(np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]))
        verts = np.array([
            [-1, -1, 1, 1],
            [-1, 1, 1, -1],
        ]) * 0.5 * size

        rot_verts = R @ verts

        min_verts = np.min(rot_verts, axis=1)
        max_verts = np.max(rot_verts, axis=1)

        # Work out how much we can shift the rotated bounding box
        rng_verts = max_verts - min_verts
        rng_shift = np.array([[cols - 1, rows - 1]]) - rng_verts
        rng_shift[rng_shift < 0] = 0

        if shift is None:
            shift = self.rnd.rand(2) * rng_shift
        else:
            shift = np.array(shift, dtype=np.float)
        shift = np.squeeze(shift)[:, np.newaxis]
        final_shift = shift + rng_verts[..., np.newaxis]/2

        if PLOT_SAMPLE_WINDOWS:
            plot_sampler(img, verts, rot_verts, rot_verts + final_shift)

        rimgs = []
        for i in (img, ) + args:
            if i.ndim == 2 or i.shape[2] == 1:
                input_shape = self.input_shape[:2]
            else:
                input_shape = self.input_shape

            rimg = resample_in_box(i, size, R, final_shift,
                                   input_shape=input_shape)
            if flip_horizontal:
                rimg = np.fliplr(rimg)
            if flip_vertical:
                rimg = np.flipud(rimg)

            # Switch to the stupid Theano convention
            if self.image_layout == 'theano':
                rimg = np.swapaxes(rimg, 0, 2)
            rimgs.append(rimg.astype(np.float32))

        if len(rimgs) == 1:
            return rimgs[0]
        return rimgs
