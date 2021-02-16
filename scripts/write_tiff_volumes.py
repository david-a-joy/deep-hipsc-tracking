#!/usr/bin/env python3

""" Write image directories to an OME Tiff file

.. code-block:: bash

    $ ./write_tiff_volume.py /path/to/image/dir

This will convert all the frames under the image dir into a single, 3D Tiff

To convert the images in the "confocal" data set:

.. code-block:: bash

    $ ./write_tiff_volumes.py \\
        --space-scale 0.91 \\
        --time-scale 3.0 \\
        example_confocal/RawData/*

"""

# Imports
import sys
import pathlib
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))

# Our own imports
from deep_hipsc_tracking.frame_tools import write_tiff_volume

# Constants

SPACE_SCALE = 0.91  # um/pixel
TIME_SCALE = 3.0  # frames/minute

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--space-scale', type=float, default=SPACE_SCALE,
                        help='Scale factor for the pixels (um/px)')
    parser.add_argument('--time-scale', type=float, default=TIME_SCALE,
                        help='Frame rate for the volume (min/frame)')
    parser.add_argument('imagedirs', nargs='+', type=pathlib.Path,
                        help='Input directories each containing a volume of images')
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))

    # Autogenerate the output file name
    for imagedir in args.pop('imagedirs'):
        if not imagedir.is_dir():
            continue
        outfile = imagedir.parent / f'{imagedir.stem}.tif'
        if outfile.is_file():
            outfile.unlink()
        write_tiff_volume(imagedir, imagefile=outfile, **args)


if __name__ == '__main__':
    main()
