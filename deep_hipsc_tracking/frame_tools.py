""" Tools to extract and process the frames from input data

Main function:

* :py:func:`extract_frames`: Extract all the frames under a root directory

Functions:

* :py:func:`find_tiles`: Find all the tiles under the input directory
* :py:func:`extract_single_frames`: Extract all the frames for a single tile

Extractors:

* :py:func:`directory_extractor`: Generate frames in order from a directory
* :py:func:`tiff_extractor`: Generate frames in order from a 3D tiff
* :py:func:`movie_extractor`: Generate frames in order from a movie

"""

# Standard lib
import shutil
import pathlib
from typing import Generator, Tuple, Optional

# 3rd party
import numpy as np

from tifffile import TiffFile, TiffWriter

# Our own imports
from .utils import Hypermap, load_image, save_image, fix_contrast, read_movie

# Types
ExtractedFrame = Tuple[int, np.ndarray]

# Constants
IMAGE_SUFFIXES = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')

# Frame extractors


def directory_extractor(inpath: pathlib.Path) -> Generator[ExtractedFrame, None, None]:
    """ Extract all the frames from a directory, in order

    :param Path inpath:
        Directory to load
    :returns:
        A Generator with one numpy array for each frame of the movie
    """
    ct = 0
    for p in sorted(inpath.iterdir()):
        if not p.is_file():
            continue
        if p.name.startswith('.'):
            continue
        if p.suffix not in IMAGE_SUFFIXES:
            continue
        ct += 1
        yield ct, load_image(p, ctype='gray')


def tiff_extractor(inpath: pathlib.Path) -> Generator[ExtractedFrame, None, None]:
    """ Extract all the frames from a 3D tiff, in order

    :param Path inpath:
        Tiff file to load
    :returns:
        A Generator with one numpy array for each frame of the movie
    """
    with TiffFile(inpath) as tif:
        for ct, page in enumerate(tif.pages):
            frame = fix_contrast(page.asarray(), mode='raw',
                                 cmin=0, cmax=255)  # Only works for RGB images
            if frame.ndim == 3:
                frame = np.mean(frame, axis=2)
            assert frame.ndim == 2
            yield ct+1, frame


def movie_extractor(inpath: pathlib.Path) -> Generator[ExtractedFrame, None, None]:
    """ Extract all the frames from a movie, in order

    :param Path inpath:
        Movie file to load
    :returns:
        A Generator with one numpy array for each frame of the movie
    """
    for ct, frame in enumerate(read_movie(inpath)):
        frame = fix_contrast(frame, cmin=0, cmax=255, mode='raw')
        if frame.ndim == 3:
            frame = np.mean(frame, axis=2)
        assert frame.ndim == 2
        yield ct+1, frame

# Functions


def write_tiff_volume(imagedir: pathlib.Path,
                      imagefile: pathlib.Path,
                      space_scale: float = 1.0,
                      time_scale: float = 1.0):
    """ Write a directory of tiff files out to a volume

    :param Path imagedir:
        The directory to process
    :param Path imagefile:
        The path to write the tiff volume to
    :param float space_scale:
        The width of a single pixel (in um)
    :param float time_scale:
        The number of minutes between frames
    :param float cmin:
        The minimum grey value for the input volume
    :param float cmax:
        The maximum grey value for the input volume
    """
    # Write out the 3D tiff, using compression to reduce the file size
    imagefile.parent.mkdir(parents=True, exist_ok=True)

    # Save metadata on each frame
    options = {
        'resolution': (1.0/space_scale, 1.0/space_scale),
        'metadata': {
            'frame interval': time_scale,
            'unit': 'um',
            'axes': 'TYX',
        },
    }
    input_pct_low = []
    input_pct_high = []
    with TiffWriter(imagefile, bigtiff=True, byteorder='<') as tif:
        # Read all the files in, in order
        for p in sorted(imagedir.iterdir()):
            if p.name.startswith('.'):
                continue
            if not p.is_file():
                continue
            if p.suffix not in IMAGE_SUFFIXES:
                continue
            print(f'Loading {p}')

            # Load and clip it to range
            img = load_image(p, ctype='gray').astype(np.uint16)
            pct_low, pct_high = np.percentile(img, [2, 98])
            input_pct_low.append(pct_low)
            input_pct_high.append(pct_high)

            # Convert the image to 8-bit
            tif.write(img, contiguous=True, dtype=np.uint16, **options)
    print(f'Wrote {imagefile}')

    print(f'Low/High: {np.min(input_pct_low)} {np.max(input_pct_high)}')
    print(f'Time scale:  {time_scale:0.2f} min/frame')
    print(f'Space scale: {space_scale:0.2f} um/pixel')


def extract_single_frames(inpath: pathlib.Path,
                          outpath: pathlib.Path,
                          do_fix_contrast: bool = False):
    """ Extract frames for a single tile inside a subprocess

    :param Path inpath:
        The input tile path to process
    :param Path outpath:
        The output tile path to write to
    :param bool do_fix_contrast:
        If True, fix the contrast for each frame, otherwise just write the raw data
    """
    # Switch on the frame extractor method
    if inpath.is_dir():
        extract_func = directory_extractor
    elif inpath.suffix in ('.tif', '.tiff'):
        extract_func = tiff_extractor
    elif inpath.suffix == '.mp4':
        extract_func = movie_extractor
    else:
        raise OSError(f'Unknown tile data file format: {inpath}')
    print(f'Extracting frames from {inpath}')

    tilename = inpath.stem

    for timepoint, frame in extract_func(inpath):
        outname = f'{tilename}t{timepoint:03d}.tif'
        outfile = outpath / outname
        if do_fix_contrast:
            frame = fix_contrast(frame, mode='equalize_adapthist')
        else:
            frame = fix_contrast(frame, mode='raw')
        save_image(outfile, frame, cmin=0, cmax=1, ctype='gray')


def find_tiles(indir: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """ Find all the tiles under the input directory

    :param Path indir:
        The path to the ``RawData`` directory
    :returns:
        A generator generating all the paths to various tile dirs and files
    """
    for p in indir.iterdir():
        if p.name.startswith('.'):
            continue
        if p.is_dir():
            if len([subp for subp in p.iterdir()
                    if subp.is_file() and subp.suffix in ('.tif', '.tiff')]) > 0:
                yield p
        elif p.is_file():
            if p.suffix in ('.mp4', '.tif', '.tiff'):
                yield p

# Main function


def extract_frames(rootdir: pathlib.Path,
                   do_fix_contrast: bool = False,
                   processes: int = 1,
                   overwrite: bool = False,
                   config_file: Optional[pathlib.Path] = None):
    """ Extract the frames for all tiles under a directory

    :param Path rootdir:
        The directory to extract
    :param bool do_fix_contrast:
        If True, contrast correct each frame
    :param int processes:
        How many worker processes to use to correct frames
    :param bool overwrite:
        If True, overwrite old directories
    """
    indir = rootdir / 'RawData'
    outdir = rootdir / 'Corrected'

    # Make sure the inputs look good
    if not indir.is_dir():
        raise OSError(f'No input directory under {indir}')

    # Convert the paths and options into a dictionary for multiprocessing
    items = [{'inpath': p, 'do_fix_contrast': do_fix_contrast, 'outpath': outdir / p.stem}
             for p in find_tiles(indir)]
    if len(items) < 1:
        raise OSError(f'No valid tiles under {indir}')

    # Make sure the outputs look okay
    if overwrite and outdir.is_dir():
        shutil.rmtree(outdir)
    if not outdir.is_dir():
        outdir.mkdir(parents=True, exist_ok=True)

    # Analyze all the samples, potentially running in parallel
    with Hypermap(processes=processes, lazy=False, wrapper='dict') as proc:
        res = proc.map(extract_single_frames, items)

    # See how many jobs processed successfully
    total_jobs = len(res)
    successful_jobs = sum(res)
    pct_jobs = successful_jobs/total_jobs
    print(f'{successful_jobs} tiles processed successfully ({pct_jobs:0.1%})')

    if total_jobs != successful_jobs:
        raise ValueError('Some tiles could not be processed!')
