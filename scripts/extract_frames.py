#!/usr/bin/env python3
""" Convert a movie to frames, optionally contrast correcting each image

.. code-block:: bash

    $ ./extract_frames.py /path/to/rootdir

Files are assumed to be in a folder under ``/path/to/rootdir`` called ``RawData``.

Three types of file structures are supported:

* Individual folders per tile, containing 2D tiff frames: e.g. ``RawData/s01/001.tif``
* OME-Tiff files, one per tile, containing 3D tiff movies: e.g. ``RawData/s01.tif``
* MPEG encoded movie files, one per tile: e.g. ``RawData/s01.mp4``

Frames will be written to corresponding directories under ``Corrected/s01/001.tif``.
Note that files under the ``Corrected`` directory are always 2D Tiffs, even if their
source is 3D.

The lossy compression used in movie file formats reduces segmentation quality, so
raw Tiff files are preferred for highest accuracy. The option to process movie
files is provided for demonstration purposes.

"""

# Standard lib
import sys
import pathlib
import argparse
import multiprocessing

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))

# Our own imports
from deep_hipsc_tracking.frame_tools import extract_frames

# Command line interface


def parse_args(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', type=pathlib.Path,
                        help='Root directory to process')
    parser.add_argument('--do-fix-contrast', action='store_true',
                        help='If True, correct the contrast of each frame')
    parser.add_argument('--config-file', type=pathlib.Path,
                        help='Path to the global configuration file')

    parser.add_argument('-p', '--processes', type=int, default=multiprocessing.cpu_count(),
                        help='Number of parallel workers to use')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite previously created files')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    extract_frames(**vars(args))


if __name__ == '__main__':
    main()
