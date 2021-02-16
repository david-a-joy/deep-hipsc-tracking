""" Image processing utilities

Convert polygons to images:

* :py:func:`contours_from_mask`: Convert a boolean mask to a set of contours
* :py:func:`mask_from_contours`: Convert a set of contours to a boolean mask

Useful functions:

* :py:func:`align_timepoints`: Grid time series to the same scales
* :py:func:`calc_histogram`: Calculate an image histogram
* :py:func:`correlation_coeff`: Calculate the correlation between two images
* :py:func:`fix_contrast`: Use adaptive equalization to flatten the image contrast
* :py:func:`trim_zeros`: Strip zeros off of the edge of an image

File readers and writers:

* :py:func:`load_image`: Load an image from disk
* :py:func:`save_image`: Save an image to disk
* :py:func:`load_point_csvfile`: Load the point values from the peak detector
* :py:func:`save_point_csvfile`: Save the point values from the peak detector

Data type conversion:

* :py:func:`to_json_types`: Convert a python type to a JSON serializable type

API Documentation
-----------------
"""

# Standard lib
import pathlib
from typing import Optional, List, Tuple, Dict

# 3rd Party
import numpy as np

from PIL import Image

from skimage import exposure, measure, filters

from scipy.ndimage.morphology import binary_erosion

# Functions


def contours_from_mask(mask: np.ndarray,
                       min_level: float = 0.5,
                       max_level: Optional[float] = None,
                       tolerance: float = 0.0) -> List[np.ndarray]:
    """ Extract the contour from the mask image

    The border is padded so contours can extend outside of it

    :param ndarray mask:
        The mask or image to contour
    :param float min_level:
        The minimum level for the mask or None for no minimum
    :param float max_level:
        The maximum level for the mask or None for no maximum
    :param float tolerance:
        The tolerance, in pixels, to simplify the contour
        (0.0 for the original marching cubes result)
    :returns:
        A list of closed contour lines
    """
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f'Only 2D masks supported, got shape: {mask.shape}')

    if max_level is None and min_level is None:
        mask = mask.astype(np.bool)
    elif min_level is None:
        mask = mask <= max_level
    elif max_level is None:
        mask = mask >= min_level
    else:
        mask = np.logical_and(mask >= min_level, mask <= max_level)

    pad_mask = np.pad(mask, 1, mode='constant', constant_values=0)
    if not np.any(pad_mask):
        return []

    sneks = measure.find_contours(pad_mask, 0.5, fully_connected='high')
    final_sneks = []
    for snek in sneks:
        snek = snek[:, [1, 0]]
        snek = snek - np.array([1, 1])  # Remove the padding
        snek = measure.approximate_polygon(snek, tolerance)
        final_sneks.append(snek)
    return final_sneks


def mask_from_contours(contours: List[np.ndarray],
                       shape: Tuple[int]) -> np.ndarray:
    """ Make a mask from a set of contours

    :param list[ndarray] contours:
        The list of contours for the image
    :param tuple[float] shape:
        The rows, cols for the new boolean image
    :returns:
        A 2D ndarray with the specified shape
    """

    rows, cols = shape

    final_mask = np.zeros((rows, cols), dtype=np.bool)
    for contour in contours:
        contour[contour < 0] = 0
        contour[contour[:, 0] > cols, 0] = cols
        contour[contour[:, 1] > rows, 1] = rows

        contour_mask = measure.grid_points_in_poly((rows, cols), contour[:, [1, 0]])
        final_mask = np.logical_or(contour_mask, final_mask)
    return final_mask


def to_json_types(data):
    """ Convert things to JSON serializable types

    :param object data:
        Data that may or may not be JSON serializable
    :returns:
        A new, more JSON-y object
    """

    if isinstance(data, str):
        return data
    elif isinstance(data, pathlib.Path):
        return str(data)
    elif isinstance(data, (int, np.bool_, np.int64, np.int32, np.uint8)):
        return int(data)
    elif isinstance(data, (float, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (list, tuple, np.ndarray)):
        return [to_json_types(d) for d in data]
    elif isinstance(data, dict):
        return {str(k): to_json_types(v) for k, v in data.items()}
    else:
        raise ValueError(f'Unknown data type: {type(data)}: {data}')


def calc_histogram(img: np.ndarray, bins: int = 40) -> np.ndarray:
    """ Calculate a histogram

    Easier to plot this than the numpy one

    :param ndarray img:
        The image to plot
    :param int bins:
        Number of bins to use in the histogram
    :returns:
        The x and y histogram bins
    """
    y_hist, edges = np.histogram(img, bins=bins)
    x_hist = (edges[1:] + edges[:-1])/2

    y_hist = y_hist / np.sum(y_hist)
    return x_hist, y_hist


def correlation_coeff(img1: np.ndarray, img2: np.ndarray) -> float:
    """ Raw correlation coefficient

    :param ndarray img1:
        First image to correlate
    :param ndarray img2:
        Second image to correlate
    :returns:
        The correlation coefficient between the two
    """

    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)

    sum1 = np.sum(img1**2)
    sum2 = np.sum(img2**2)
    sum12 = np.sum(img1 * img2)
    return sum12 / np.sqrt(sum1*sum2)


def fix_contrast(img: np.ndarray,
                 filter_size: int = 1,
                 cmin: Optional[float] = None,
                 cmax: Optional[float] = None,
                 mode: str = 'equalize_adapthist',
                 atol: float = 1e-5) -> np.ndarray:
    """ Fix the contrast of an image

    Uses adaptive histogram equalization to remove most contrast issues

    :param ndarray img:
        The image to fix
    :param int filter_size:
        If > 1, width of the median filter to run over the image
    :param float img_min:
        Min value to use when rescaling the image
    :param float img_max:
        Max value to use when rescaling the image
    :param float atol:
        Absolute contrast tolarance (max - min) below which an image is considered black
    :returns:
        The image with newly corrected contrast
    """
    if mode not in (None, 'none', 'raw') and not mode.startswith('equalize_'):
        mode = 'equalize_' + mode

    # Convert the image to floating point
    img = img.astype(np.float64)
    if cmin is None:
        cmin = np.min(img)
    if cmax is None:
        cmax = np.max(img)
    crng = cmax - cmin
    if crng < atol:
        if np.all(img > cmax):
            return np.ones(img.shape, dtype=np.float64)
        else:
            return np.zeros(img.shape, dtype=np.float64)

    # Normalize the image to clean up the contrast
    img = (img - cmin) / crng
    img[img < 0] = 0
    img[img > 1] = 1

    # Run a median filter over the image to remove shot noise
    if filter_size > 1:
        img = np.round(img * 255)  # Convert to an 8-bit image
        img[img < 0] = 0
        img[img > 255] = 255
        img = img.astype(np.uint8)

        img = filters.median(img, selem=np.ones((filter_size, filter_size)))
        img = img.astype(np.float64) / 255

    # Allow different equalization methods
    if mode in (None, 'none', 'raw'):
        pass
    elif mode == 'equalize_adapthist':
        img = exposure.equalize_adapthist((img*255).astype(np.uint8), clip_limit=0.03)
    elif mode == 'equalize_hist':
        img = exposure.equalize_hist((img*255).astype(np.uint8), nbins=256)
    elif mode == 'equalize_percentile':
        pct2, pct98 = np.percentile(img, [2, 98])
        if (pct98 - pct2) < atol:
            return np.zeros(img.shape, dtype=np.float64)
        img = (img - pct2) / (pct98 - pct2)
        img[img < pct2] = pct2
        img[img > pct98] = pct98
    else:
        raise KeyError(f'Unknown equalization method: {mode}')

    assert img.dtype == np.float64
    return img


def trim_zeros(data: np.ndarray,
               padding: int = 0,
               cutoff: float = 0,
               erode: int = 0,
               axis: Optional[int] = None,
               symmetric_axis: Optional[int] = None) -> np.ndarray:
    """ Trim edge zeros out of the data

    :param ndarray data:
        A numpy nd-array to trim zeros off of
    :param int padding:
        Number of padded layers to add around the edges
    :param float cutoff:
        Data greater than this is considered non-zero
    :param int erode:
        How many pixels to erode from the mask before calculating bounds
    :param int axis:
        Only apply the trim to this axis or axes
    :param int symmetric_axis:
        A tuple of axis to trim symmetrically (take the minimum of the left or
        right trim)
    :returns:
        The new image with all zeros trimmed off the edges
    """

    # Work out which image axis/axes to trim
    if axis is None:
        axis = tuple(i for i in range(data.ndim))
    elif isinstance(axis, (int, float)):
        axis = (int(axis), )
    else:
        axis = tuple(int(i) for i in axis)

    # Work out which axis/axes need to be trimmed equally from both sides
    if symmetric_axis is None:
        symmetric_axis = ()
    elif isinstance(symmetric_axis, (int, float)):
        symmetric_axis = (int(symmetric_axis), )
    else:
        symmetric_axis = tuple(int(s) for s in symmetric_axis)

    # Create the mask and erode it
    mask = data > cutoff
    for _ in range(erode):
        mask = binary_erosion(mask)
    where = np.argwhere(mask)

    mindex = tuple(np.min(where[:, i]) for i in range(data.ndim))
    maxdex = tuple(np.max(where[:, i]) for i in range(data.ndim))

    # Now actually back out the extreme bounds of the trim mask
    affine_offset = []
    index = []
    for i, item in enumerate(zip(mindex, maxdex)):

        if i not in axis:
            affine_offset.append(0)
            index.append(slice(0, data.shape[i]))
            continue

        mn, mx = item

        mn -= padding
        mx += padding + 1

        mn = mn if mn > 0 else 0
        mx = mx if mx < data.shape[i] else data.shape[i]

        if i in symmetric_axis:
            ox = data.shape[i] - mx
            ox = max((min((mn, ox)), 0))

            mn = ox
            mx = data.shape[i] - ox

            mn = mn if mn > 0 else 0
            mx = mx if mx < data.shape[i] else data.shape[i]
        index.append(slice(mn, mx))
    return data[tuple(index)]


def align_timepoints(agg_data: Dict, agg_timepoints: List) -> Dict:
    """ Align the timepoints

    :param dict agg_data:
        A mapping of key: [list of timepoint stats]
    :param list agg_timepoints:
        A list of timepoints corresponding to each stat
    :returns:
        A dictionary of 2D arrays.
        Each data set is padded with nans to the same length.
    """

    if all(len(tp) == 0 for tp in agg_timepoints):
        full_timepoints = np.array([])
    else:
        min_timepoint = np.min([np.min(tp) for tp in agg_timepoints if len(tp) > 0])
        max_timepoint = np.max([np.max(tp) for tp in agg_timepoints if len(tp) > 0])
        full_timepoints = np.arange(min_timepoint, max_timepoint+1)

    final_data = {}
    for key in agg_data:
        final_recs = []
        for i, rec in enumerate(agg_data[key]):
            timepoint = agg_timepoints[i]
            timepoint_mask = np.in1d(full_timepoints, timepoint)
            final_rec = np.full_like(full_timepoints, np.nan, dtype=np.float)
            final_rec[timepoint_mask] = rec
            final_recs.append(final_rec)
        final_data[key] = np.array(final_recs)
    return final_data


# Readers and Writers


def save_point_csvfile(csvfile: pathlib.Path,
                       cx: np.ndarray,
                       cy: np.ndarray,
                       cv: np.ndarray,
                       xlim: Optional[Tuple[float]] = None,
                       ylim: Optional[Tuple[float]] = None):
    """ Save the points to a csvfile

    :param Path csvfile:
        The csvfile to save to
    :param ndarray cx:
        The x coordinates to save
    :param ndarray cy:
        The y coordinates to save
    :param ndarray cv:
        The confidence score for the detection
    :param tuple[float] xlim:
        Minimum, maximum x values to write
    :param tuple[float] ylim:
        Minimum, maximum y values to write
    """

    print(f'Writing points to: {csvfile}')
    csvfile.parent.mkdir(exist_ok=True, parents=True)

    assert len(cx) == len(cv)
    assert len(cy) == len(cv)

    if xlim is None:
        xmin, xmax = min(cx), max(cx)
    else:
        xmin, xmax = xlim

    if ylim is None:
        ymin, ymax = min(cy), max(cy)
    else:
        ymin, ymax = ylim

    with csvfile.open('wt') as fp:
        fp.write('#point_x,point_y,point_score\n')
        for px, py, pv in zip(cx, cy, cv):
            if px < xmin or px > xmax:
                continue
            if py < ymin or py > ymax:
                continue
            fp.write(f'{float(px):f},{float(py):f},{float(pv):f}\n')


def load_point_csvfile(csvfile: pathlib.Path) -> Tuple[np.ndarray]:
    """ Load the csvfile

    :param Path csvfile:
        A file, as written by load_point_csvfile
    :returns:
        Three numpy arrays: x coords, y coords, point score
    """

    print(f'Reading points from: {csvfile}')

    cx, cy, cv = [], [], []
    with csvfile.open('rt') as fp:
        for line in fp:
            line = line.split('#', 1)[0].strip()
            if line == '':
                continue
            x, y, v = line.split(',')
            cx.append(float(x.strip()))
            cy.append(float(y.strip()))
            cv.append(float(v.strip()))
    return np.array(cx), np.array(cy), np.array(cv)


def load_image(imgfile: pathlib.Path, ctype: str = 'gray') -> np.ndarray:
    """ Load an image as grayscale...ish

    :param Path imgfile:
        The path to the file to load
    :param str ctype:
        If "gray", convert the RGB image to 'grayscale'
    :returns:
        The image as an ndarray
    """
    ctype = ctype.lower()
    img = np.asarray(Image.open(str(imgfile))).astype(np.float64)
    if ctype in ('gray', 'grey'):
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        if img.ndim != 2:
            raise ValueError(f'Expected 2D gray image, got image with shape {img.shape}')
    elif ctype == 'color':
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(f'Expected 3D color image, got image with shape {img.shape}')
    elif ctype == 'rgb':
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f'Expected 3D RGB image, got image with shape {img.shape}')
    elif ctype == 'rgba':
        if img.ndim != 3 or img.shape[2] != 4:
            raise ValueError(f'Expected 3D RGBA image, got image with shape {img.shape}')
    else:
        raise KeyError(f'Unknown color type "{ctype}"')
    return img


def save_image(imgfile: pathlib.Path,
               img: np.ndarray,
               cmin: Optional[float] = None,
               cmax: Optional[float] = None,
               ctype: str = 'gray',
               atol: float = 1e-5):
    """ Save an image to a file

    :param Path imgfile:
        The image file to save
    :param ndarray img:
        The 2D or 3D (RGB or RGBA) image to save
    :param float cmin:
        The minimum value in the image to save (the black level)
    :param float cmax:
        The maximum value in the image to save (the while level)
    :param str ctype:
        Check that the image is grey, RGB, or RGBA
    """
    # Clean up the input file types
    ctype = ctype.lower()
    imgfile = pathlib.Path(imgfile)

    # Force the image to be float for math purposes
    img = img.astype(np.float64)
    if cmin is None:
        cmin = np.nanmin(img)
    if cmax is None:
        cmax = np.nanmax(img)
    img[np.isnan(img)] = cmin

    # Scale the image using the contrast bounds
    crange = cmax - cmin
    if crange < atol:
        img = np.zeros_like(img)
    else:
        img = np.round((img - cmin) / crange * 255)

    # Force the image to be 8-bit
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)

    # Make sure that the image has the correct shape
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)

    # Make sure the expected type has the correct dimensions
    if ctype in ('grey', 'gray'):
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(f'Expected gray image, got image with shape {img.shape}')
    elif ctype in ('color', 'colour'):
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(f'Expected color image, got image with shape {img.shape}')
    elif ctype == 'rgb':
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f'Expected RGB image, got image with shape {img.shape}')
    elif ctype == 'rgba':
        if img.ndim != 3 or img.shape[2] != 4:
            raise ValueError(f'Expected RGBA image, got image with shape {img.shape}')
    else:
        raise KeyError(f'Unknown color type "{ctype}"')

    # Write the actual images
    imgfile.parent.mkdir(exist_ok=True, parents=True)

    img = Image.fromarray(img)
    img.save(str(imgfile))
