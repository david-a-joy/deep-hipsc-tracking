#!/usr/bin/env python3

""" Composite multiple detectors into an ensamble

Ensamble real data from ``composite_cells.py``:

    $ composite_cells.py /data/Experiment/2017-01-30

Ensamble training data from ``composite_cells.py``:

    $ composite_cells.py --data-type train ~/Desktop/TrainingData

Filter detectors by name to only use some of them in the ensamble:

.. code-block:: bash

    $ composite_cells.py \\
        -d countception-r5-n50000 unet-r1-n50000 \\
        --data-type train ~/Desktop/TrainingData

This will only composite run005 of countception and run001 of unet using the
versions at 50,000 iterations each.

Specifically, to generate the composite optimal dataset, I ran:

.. code-block:: bash

    $ ./composite_cells.py \\
        -d countception-r3-50k \\
        -d fcrn_b_wide-r3-75k \\
        -d residual_unet-r4-25k \\
        --data-type train \\
        ~/Desktop/TrainingData \\
        -o ~/Desktop/TrainingData/ai-upsample-peaks-composite-d3-final \\
        --microscope inverted

"""

# Standard lib
import sys
import shutil
import pathlib
import argparse
from typing import List, Optional

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party
import numpy as np

from skimage.feature import peak_local_max

# Our own imports
from deep_hipsc_tracking.utils import load_image, save_image, Hypermap, save_point_csvfile
from deep_hipsc_tracking.tracking import DetectorThresholds
from deep_hipsc_tracking.model import convert_points_to_mask, pair_detector_data, parse_detectors
from deep_hipsc_tracking.presets import load_preset

# Constants
DETECTION_THRESHOLD = 0.01
PEAK_DISTANCE = 3  # Minimum number of pixels between peaks
COMP_TYPE = 'mean'

EXCLUDE_BORDER = 5  # number of pixels around the edge to exclude


# Parallel processing per image


def ensamble_timepoint(prefix: pathlib.Path,
                       outdir: pathlib.Path,
                       num_detectors: int,
                       imagefiles: List[pathlib.Path],
                       peak_distance: int = PEAK_DISTANCE,
                       detection_threshold: float = DETECTION_THRESHOLD,
                       comp_type: str = COMP_TYPE,
                       weights: Optional[List[float]] = None,
                       exclude_border: int = EXCLUDE_BORDER):
    """ Ensamble a single timepoint over several detectors

    :param Path prefix:
        The relative path to write the files to
    :param Path outdir:
        The base directory to write the files to
    :param int num_detectors:
        How many detectors should be in this composite
    :param list[Path] imagefiles:
        The list of files for each image from the detectors
    :param int peak_distance:
        Distance in pixels for the composite detection
    :param float detection_threshold:
        The number between 0 and 1 for the detector cutoff
    :param str comp_type:
        Which composite method to use ('mean' or 'max')
    """
    assert len(imagefiles) == num_detectors

    out_tiledir = outdir / prefix.parent
    out_imagefile = out_tiledir / f'{prefix.stem}_resp.png'
    out_trainfile = out_tiledir / f'{prefix.stem}_train.png'
    out_pointfile = out_tiledir / f'{prefix.stem}.csv'

    # Do a weird parallel mkdir because the other way is unsafe
    for _ in range(3):
        if out_imagefile.parent.is_dir():
            break
        try:
            out_imagefile.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue

    # Weights for the ensamble
    if weights is None:
        weights = [1.0 for _ in imagefiles]
    elif isinstance(weights, (int, float)):
        weights = [weights for _ in imagefiles]
    if len(weights) != len(imagefiles):
        err = f'Invalid shape for weights, got {len(weights)} weights, {len(imagefiles)} images'
        raise ValueError(err)
    weights = [float(w) for w in weights]

    # Create a composite detection
    print(out_imagefile)
    comp_image = []
    for weight, imagefile in zip(weights, imagefiles):
        in_image = load_image(imagefile) / 255.0
        comp_image.append(in_image * weight)

    if comp_type == 'mean':
        response = np.mean(comp_image, axis=0)
    elif comp_type == 'max':
        response = np.max(comp_image, axis=0)
    else:
        raise KeyError(f'Unknown composite type: {comp_type}')

    # Normalize the response
    response[response < 0] = 0
    response[response > 1] = 1
    save_image(out_imagefile, response)

    # Create the peak detection for that composite
    peaks = peak_local_max(response + np.random.rand(*response.shape)*1e-5,
                           min_distance=int(np.ceil(peak_distance)),
                           threshold_abs=detection_threshold,
                           exclude_border=exclude_border)
    rows, cols = response.shape[:2]

    assert peaks.shape[1] == 2
    norm_px = [px / cols
               for px in peaks[:, 1]]
    norm_py = [(1.0 - py / rows)
               for py in peaks[:, 0]]
    peak_values = response[peaks[:, 0], peaks[:, 1]]

    final_peaks = [(x, y) for x, y, v in zip(norm_px, norm_py, peak_values)
                   if v > 0.01]

    save_point_csvfile(out_pointfile, norm_px, norm_py, peak_values,
                       xlim=(0, 1), ylim=(0, 1))

    # Finally, synthesize a training image
    if out_trainfile is not None:
        mask = convert_points_to_mask(response, final_peaks)
        save_image(out_trainfile, mask)


def load_detector_weights(detectors=None, microscope: str = 'inverted'):
    """ Load the detector weights

    :param tuple[str] detectors:
        The detectors to load weights for or None to load the defaults
    :param str microscope:
        The microscope we're processing for this data
    :returns:
        A tuple of weights for these detectors
    """
    # Load the weights and metadata from the database
    thresholds = DetectorThresholds.by_microscope(microscope)
    detector_weights_init = thresholds.detector_weights
    training_set = thresholds.training_set

    # Force the detector keys and the stored weights to be sorted
    detector_weights = {}
    for key, val in detector_weights_init.items():
        key = tuple(sorted(d.lower() for d in key))
        if len(key) != len(val):
            raise ValueError(f'Invalid weights for {key}: {val}')
        assert key not in detector_weights
        detector_weights[key] = val

        # Add a synthetic key for the real data too
        new_key = []
        for detector in key:
            res = parse_detectors(detector, data_type='train')[0][0]
            new_key.append(res)
        new_key = tuple(sorted(new_key))
        if len(new_key) != len(val):
            raise ValueError(f'Invalid weights for {new_key}: {val}')
        assert new_key not in detector_weights
        detector_weights[new_key] = val

    # Force the detector keys and the stored weights to be sorted
    if detectors is not None:
        detectors = tuple(sorted(d.lower() for d in detectors))
        print(f'Ensambling using the detectors: {detectors}')

    weights = detector_weights.get(detectors)
    if weights is not None:
        print(f'Ensambling with weights: {weights}')
    return detectors, weights, training_set


# Main function


def ensemble_single_cell(rootdir: pathlib.Path,
                         config_file: pathlib.Path,
                         processes: int = 1,
                         outdir: Optional[pathlib.Path] = None,
                         detectors: Optional[List[str]] = None):
    """ Ensemble all the single cell detectors

    :param Path rootdir:
        The path to the experiment directory or training data directory
    :param int processes:
        Number of parallel processes to run with
    :param str comp_type:
        One of "mean", "max", how to composite the dataset
    :param Path outdir:
        The output directory to write the composites to
    """
    rootdir = rootdir.resolve()
    config_file = config_file.resolve()

    # Load defaults from the config file
    config = load_preset(config_file)
    if detectors in (None, []):
        detectors = config.segmentation['detectors']
    microscope = config.microscope

    exclude_border = config.segmentation['exclude_border']
    detection_threshold = config.segmentation['detection_threshold']
    peak_distance = config.segmentation['peak_distance']
    comp_type = config.segmentation['composite_type']

    detectors, weights, training_set = load_detector_weights(detectors=detectors, microscope=microscope)

    all_paths, num_detectors = pair_detector_data(
        rootdir, detectors=detectors, training_set=training_set, data_type='any')
    print(f'Found {num_detectors} matching detectors')

    if outdir is None:
        outdir = rootdir / 'SingleCell-Composite'

    if outdir.is_dir():
        print(f'Overwriting: {outdir}')
        shutil.rmtree(str(outdir))
    outdir.mkdir(parents=True)

    items = [{'outdir': outdir,
              'num_detectors': num_detectors,
              'imagefiles': imagefiles,
              'comp_type': comp_type,
              'detection_threshold': detection_threshold,
              'weights': weights,
              'exclude_border': exclude_border,
              'peak_distance': peak_distance,
              'prefix': prefix}
             for (prefix, _, _), imagefiles in sorted(all_paths.items())]

    with Hypermap(processes=processes, lazy=False, wrapper='dict') as pool:
        okay = pool.map(ensamble_timepoint, items)

    total_items = len(items)
    okay_items = sum(okay)

    print(f'{okay_items} processed successfully ({okay_items/total_items:0.1%})')
    if total_items != okay_items:
        raise ValueError(f'Got {total_items - okay_items} errors during processing')


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=Hypermap.cpu_count())
    parser.add_argument('--config-file', type=pathlib.Path)
    parser.add_argument('-o', '--outdir', type=pathlib.Path)
    parser.add_argument('-d', '--detector', dest='detectors', default=[], action='append',
                        help='List of detectors to use')
    parser.add_argument('rootdirs', nargs='+', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    for rootdir in args.pop('rootdirs'):
        ensemble_single_cell(rootdir, **args)


if __name__ == '__main__':
    main()
