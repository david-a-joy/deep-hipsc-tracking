#!/usr/bin/env python3

""" Download the training data, trained weights, and example files

Download everything to the default location (../deep_hipsc_tracking/data):

.. code-block:: bash

    $ ./download_data.py

Download everything to a different location (/path/to/data)

.. code-block:: bash

    $ ./download_data.py -r /path/to/data

Note that tracking code expects weights to be in specific directories, so this may break the pipeline

Download only specific data sets

.. code-block:: bash

    $ ./download_data.py -d weights -d training_inverted -d examples_inverted

Where the individual data sets are:

* ``weights``: The pre-trained model weights for inverted and confocal segmentation
* ``training_inverted``: The human annotations for the inverted training data set
* ``training_confocal``: The human annotations for the confoccal training data set
* ``example_confocal``: A set of sample time series images from the confocal data set

To train your own model using one of the provided architectures, see ``detect_cells.py``

To package an existing data set into a form the pipeline expects, see ``write_tiff_volumes.py``

"""

# Imports
import shutil
import pathlib
import argparse
import hashlib
from zipfile import ZipFile
from typing import List
from urllib.parse import urljoin

# 3rd party
import requests

# Constants
URLROOT = 'https://github.com/david-a-joy/deep-hipsc-tracking/releases/download/data/'
THISDIR = pathlib.Path(__file__).parent.resolve()
ROOTDIR = THISDIR.parent / 'deep_hipsc_tracking' / 'data'

# Classes

class DataDownloader(object):
    """ Store metadata and urls for downloading data

    :param Path rootdir:
        The base directory to write downloaded files to
    :param str urlroot:
        The URL to download files from
    """

    # Individual download links
    weights_url = 'weights.zip'
    weights_sha1 = 'd23762e9daa72003825e23916858d4360c04c80f'
    weights_size = 207761898
    weights_filename = 'weights.zip'

    training_inverted_url = 'training_inverted.zip'
    training_inverted_size = 416145256
    training_inverted_sha1 = '469212a7cbd76ce84dd84aab46ba392ae2d2d0e6'
    training_inverted_filename = 'training_inverted.zip'

    training_confocal_url = 'training_confocal.zip'
    training_confocal_size = 237216110
    training_confocal_sha1 = 'a20734857ed14f7b099dc1623eee5f79baedf10d'
    training_confocal_filename = 'training_confocal.zip'

    example_confocal_url = 'example_confocal.zip'
    example_confocal_size = 177108473
    example_confocal_sha1 = 'b809ea3fc02095627be377313e38194dd9d9fc20'
    example_confocal_filename = 'example_confocal.zip'

    def __init__(self, rootdir: pathlib.Path,
                 urlroot: str = URLROOT):
        self.rootdir = rootdir.resolve()
        self.urlroot = urlroot

        self.verify_ssl = True

    def download(self, data_set: str,
                 chunk_size: int = 8192):
        """ Download an individual data set

        :param str data_set:
            Which data set to download
        :param int chunk_size:
            What size of memory chunks to download
        """

        datadir = self.rootdir / data_set
        print(f'Downloading {data_set} to {datadir}')

        if datadir.is_dir():
            print(f'Deleting old data {datadir}')
            shutil.rmtree(datadir)
        datadir.mkdir(parents=True, exist_ok=True)

        data_set_url = getattr(self, f'{data_set}_url')
        data_set_size = getattr(self, f'{data_set}_size')
        data_set_sha1 = getattr(self, f'{data_set}_sha1')
        data_set_filename = getattr(self, f'{data_set}_filename')

        data_set_full_url = urljoin(self.urlroot, data_set_url)
        print(f'Downloading from {data_set_full_url}')

        # Download the file and calculate the sha-1 hash
        hasher = hashlib.new('sha1')
        r = requests.get(data_set_full_url, stream=True, verify=self.verify_ssl)
        content_length = r.headers.get('Content-Length')
        if content_length is not None:
            content_length = int(content_length)

        input_length = 0
        last_pct = None
        with open(datadir / data_set_filename, 'wb') as fp:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fp.write(chunk)
                hasher.update(chunk)
                input_length += len(chunk)
                if content_length is not None:
                    pct = input_length / content_length
                    if last_pct is None or (pct - last_pct >= 0.05):
                        last_pct = pct
                        print(f'{data_set_filename} downloading... {pct:5.1%}')
        print(f'{data_set_filename} downloading... 100.0%')

        # Make sure the input length matches the expected length
        if input_length != data_set_size:
            err = f'File length mismatch on {data_set_filename}'
            print(err)
            print(f'Expected: {data_set_size}')
            print(f'Got:      {input_length}')
            raise ValueError(err)

        # Check the downloaded sha1sum
        downloaded_sha1 = hasher.hexdigest()
        if downloaded_sha1 != data_set_sha1:
            err = f'SHA-1 Sum mismatch on {data_set_filename}'
            print(err)
            print(f'Expected: {data_set_sha1}')
            print(f'Got:      {downloaded_sha1}')
            raise ValueError(err)

        # Unpack the downloaded blob
        prefix = f'{data_set}/'  # Everything in the dataset ends up one folder up
        total_files = 0
        with ZipFile(datadir / data_set_filename, 'r') as zipfp:
            for name in zipfp.namelist():
                if not name.startswith(prefix):
                    continue
                if name.endswith('/'):
                    continue
                shortname = name[len(prefix):]
                outpath = datadir / shortname
                if outpath.name.startswith('.'):
                    continue

                # Yay, unpack this file to the correct relative path
                print(outpath)
                total_files += 1
                outpath.parent.mkdir(parents=True, exist_ok=True)
                with outpath.open('wb') as gp:
                    with zipfp.open(name, 'r') as fp:
                        while True:
                            chunk = fp.read(chunk_size)
                            if len(chunk) < 1:
                                break
                            gp.write(chunk)

        # Make sure we got something back
        print(f'Unpacked {total_files} total files')
        if total_files < 1:
            raise ValueError(f'Failed to find any files in data set {data_set}')

        # Clean up the zip file
        (datadir / data_set_filename).unlink()

# Main function


def download_all_data(rootdir: pathlib.Path,
                      data_sets: List[str]):
    """ Download reference data

    :param Path rootdir:
        The directory to write the example data to
    :param list[str] data_sets:
        Which categories of data to download or 'all' for everything
    """
    if data_sets in (None, []):
        data_sets = ['all']
    elif isinstance(data_sets, str):
        data_sets = [data_sets]
    else:
        data_sets = [str(d) for d in data_sets]
    if 'all' in data_sets:
        data_sets = [
            'weights', 'example_confocal',
            'training_inverted', 'training_confocal',
        ]

    downloader = DataDownloader(rootdir)

    print(f'Downloading data sets: {data_sets}')
    for data_set in data_sets:
        downloader.download(data_set)

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rootdir', default=ROOTDIR,
                        help='Directory to write the downloaded data to')
    parser.add_argument('-d', '--data-set', action='append', dest='data_sets',
                        help='Which data set(s) to download',
                        choices=('all', 'weights', 'example_confocal',
                                 'training_inverted', 'training_confocal'))
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    download_all_data(**vars(args))


if __name__ == '__main__':
    main()
