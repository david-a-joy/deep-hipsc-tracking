#!/usr/bin/env python3

""" Compress the data and calculate all the checksums

Compress the example inverted data set:

.. code-block:: bash

    $ ./compress_data.py example_confocal

This creates the ``example_confocal.zip`` file for use in ``download_data.py``.
File size and checksums will print out at the end of the script.

"""

# Imports
import pathlib
import argparse
import hashlib
from zipfile import ZipFile

# Constants
CHUNK_SIZE = 8192

# Main function


def compress_data(datadir: pathlib.Path,
                  chunk_size: int = CHUNK_SIZE):
    """ Compress the data and calculate size, shasum, etc

    :param Path datadir:
        The directory to compress
    """
    if not datadir.is_dir():
        raise OSError(f'Data directory not found: {datadir}')
    datafile = datadir.parent / f'{datadir.name}.zip'

    if datafile.is_file():
        datafile.unlink()

    # Compress everything under that directory
    with ZipFile(datafile, 'w') as zipfp:
        targets = [datadir]
        while targets:
            p = targets.pop()
            if p.name.startswith('.'):
                continue
            if p.is_dir():
                targets.extend(p.iterdir())
                continue
            if not p.is_file():
                continue
            relp = p.relative_to(datadir.parent)
            print(p, relp)
            zipfp.write(p, arcname=str(relp))

    # Calculate stats on the file
    hasher = hashlib.new('sha1')
    size = 0
    with datafile.open('rb') as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            hasher.update(chunk)
    print(f'SHA-1: {hasher.hexdigest()}')
    print(f'Size:  {size}')

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    compress_data(**vars(args))


if __name__ == '__main__':
    main()
