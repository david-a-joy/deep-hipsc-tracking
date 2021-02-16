""" Readers and writers for training/test data generation

Useful functions:

* :py:func:`write_point_outfile`: Write the point/image pairs for the training set
* :py:func:`write_mask_outfile`: Write the mask/image pairs for the training set
* :py:func:`reduce_image`: Reduce an image by a given factor
* :py:func:`expand_image`: Expand an image by a given factor

"""

# Imports

# Standard lib
import pathlib
from typing import Optional

# 3rd party
import numpy as np

from skimage.transform import rescale, resize

from skimage.draw import disk

# Our own imports
from ..utils import load_image, save_image
from . import calculate_peak_image

# Constants

DATA_RESIZE = 1  # Factor to downsample the masks by (e.g. 2 = 2x downsample)
MASK_TYPE = 'peaks'  # One of 'points' or 'peaks'
DOT_RADIUS = 0.5  # How big to make the dots


# Functions


def reduce_image(img: np.ndarray,
                 data_resize: int = DATA_RESIZE) -> np.ndarray:
    """ Reduce the image by a scale value

    :param ndarray img:
        The image to reduce
    :param int data_resize:
        The amount to scale the image by
    :returns:
        A scaled and normalized image
    """

    if data_resize != 1:
        print(f'Reducing image by: {data_resize}')
        print(f'Original shape: {img.shape}')
        rows, cols = img.shape[:2]
        new_rows, new_cols = int(np.round(rows / data_resize)), int(np.round(cols / data_resize))
        img = resize(img, (new_rows, new_cols) + img.shape[2:])
        print(f'Reduced shape:  {img.shape}')

    img_min = np.min(img)
    img_max = np.max(img)
    img_rng = img_max - img_min

    if img_rng < 1e-5:
        return np.zeros_like(img)
    return (img - img_min) / img_rng


def expand_image(img: np.ndarray,
                 data_resize: int = DATA_RESIZE) -> np.ndarray:
    """ Expand the image by a scale value

    :param ndarray img:
        The image to reduce
    :param int data_resize:
        The amount to scale the image by
    :returns:
        A scaled and normalized image
    """

    if data_resize != 1:
        print(f'Expanding image by: {data_resize}')
        print(f'Original shape: {img.shape}')
        if img.ndim == 2:
            scales = (data_resize, data_resize)
        else:
            scales = (data_resize, data_resize, 1)
        img = rescale(img, scales, order=3)
        print(f'Expanded shape:  {img.shape}')

    img_min = np.min(img)
    img_max = np.max(img)
    img_rng = img_max - img_min

    if img_rng < 1e-5:
        return np.zeros_like(img)
    return (img - img_min) / img_rng


def convert_points_to_mask(img: np.ndarray,
                           points: np.ndarray,
                           dot_radius: float = DOT_RADIUS,
                           mask_type: str = MASK_TYPE,
                           points_normalized: bool = True) -> np.ndarray:
    """ Convert the points to a mask

    :param ndarray img:
        The 2D image to create a mask for
    :param list[tuple] points:
        The list of x, y coordinates for the point centers
    :param float dot_radius:
        The radius of dots to draw around each point
    :param str mask_type:
        One of 'peaks' or 'points', what kind of mask to draw
    :param bool points_normalized:
        If True, the points are in normalized coordinates (0 - 1)
    :returns:
        The 2D, 8-bit mask with the same shape as img
    """

    # Convert the point coordinates to locations in a mask image
    mask = np.zeros_like(img, dtype=np.uint8)
    rows, cols = mask.shape

    for point in points:
        if hasattr(point, 'x'):
            x, y = point.x, point.y
        else:
            x, y = point

        # Convert from normal to image coords
        if points_normalized:
            px = x * cols
            py = (1.0 - y) * rows
        else:
            px = x
            py = y
        # print(px, py)

        coords_x, coords_y = disk((py, px), dot_radius, shape=(rows, cols))
        # Turns out the circle function can return empty arrays
        if coords_x.shape[0] > 0 and coords_y.shape[0] > 0:
            mask[coords_x, coords_y] = 255

    # If we got told to make peaks, convert to a peak mask
    if mask_type == 'peaks':
        mask = calculate_peak_image(mask, img_rows=32, img_cols=32, zero_padding=32, peak_sharpness=8)
        mask = np.round(mask * 255)
        mask[mask > 255] = 255
        mask[mask < 0] = 0
        mask = mask.astype(np.uint8)
    return mask


def write_point_outfile(imagefile: pathlib.Path,
                        points: np.ndarray,
                        img_outfile: pathlib.Path,
                        mask_outfile: pathlib.Path,
                        data_resize: int = DATA_RESIZE,
                        flip: str = 'none',
                        rot90: int = 0,
                        dot_radius: float = DOT_RADIUS,
                        mask_type: str = MASK_TYPE,
                        min_image_size: Optional[int] = None):
    """ Write the training image/point pairs to a file

    :param Path imagefile:
        The input imagefile to convert to training data
    :param ndarray points:
        The n x 2 array of points to convert to a training mask
    :param Path img_outfile:
        The path to write the output image file to
    :param Path mask_outfile:
        The path to write the output mask file to
    :param int data_resize:
        What factor to reduce the image by
    :param str flip:
        How to mirror the image: one of horizontal/vertical/none
    :param int rot90:
        How many 90 degree rotations to apply to the image
    :param float dot_radius:
        The radius of dots to generate for a given point. All points will be
        drawn a minimum of one pixel large
    :param str mask_type:
        One of 'points' or 'peaks', the type of mask to write
    :param int min_image_size:
        Minimum size for rows/columns for the final image
    """

    img = load_image(imagefile, ctype='gray')

    # Downsample the image
    img = reduce_image(img, data_resize=data_resize)

    # Convert to 8-bit
    img = np.round(img*255)
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)

    rows, cols = img.shape

    mask = convert_points_to_mask(img, points, dot_radius=dot_radius, mask_type=mask_type)

    # Pad to the minimum size
    if min_image_size is not None:
        new_rows = min_image_size if rows < min_image_size else rows
        new_cols = min_image_size if cols < min_image_size else cols
        delta_rows = new_rows - rows
        delta_cols = new_cols - cols

        st_rows = delta_rows // 2
        st_cols = delta_cols // 2
        ed_rows = st_rows + rows
        ed_cols = st_cols + cols

        new_img = np.zeros((new_rows, new_cols), dtype=np.uint8)
        new_img[st_rows:ed_rows, st_cols:ed_cols] = img

        new_mask = np.zeros((new_rows, new_cols), dtype=np.uint8)
        new_mask[st_rows:ed_rows, st_cols:ed_cols] = mask

        img, mask = new_img, new_mask

    # Apply the current set of transforms to the image and mask
    if rot90 != 0:
        img = np.rot90(img, rot90)
        mask = np.rot90(mask, rot90)
    if flip not in ('none', None):
        if flip == 'horizontal':
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        elif flip == 'vertical':
            img = np.flipud(img)
            mask = np.flipud(mask)

    save_image(img_outfile, img)
    save_image(mask_outfile, mask)


def write_mask_outfile(imagefile: pathlib.Path,
                       maskfile: pathlib.Path,
                       img_outfile: pathlib.Path,
                       mask_outfile: pathlib.Path,
                       data_resize: int = DATA_RESIZE):
    """ Write the training image/mask pairs to a file

    :param Path imagefile:
        The image file to load
    :param Path maskfile:
        The mask file to load
    :param Path img_outfile:
        The path to save the resampled image to
    :param Path mask_outfile:
        The path to save the resampled mask to
    :param int data_resize:
        How much to resize the image/mask pair by
    """

    img = load_image(imagefile)

    # FIXME: Allow loading from mask images to
    mask = np.load(maskfile)['mask']

    # Resize the image and mask by the same rate
    img = reduce_image(img, data_resize=data_resize)
    mask = reduce_image(mask, data_resize=data_resize)

    # Convert to 8-bit
    img = np.round(img*255)
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)

    # Convert to 8-bit black and white
    mask = np.round((mask > 0.5)*255)
    mask[mask < 0] = 0
    mask[mask > 255] = 255
    mask = mask.astype(np.uint8)

    save_image(img_outfile, img)
    save_image(mask_outfile, mask)
