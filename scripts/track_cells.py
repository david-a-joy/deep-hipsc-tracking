#!/usr/bin/env python3
""" Convert sequences of detected cells to cell tracks

Link the detections in experiment "2017-01-30" using the "balltree" algorithm:

.. code-block:: bash

    $ ./track_cells.py \\
        /data/Experiment/2017-01-30

Link the detections in experiment "2017-03-03" using the "softassign" algorithm.
Only output the final track plots:

.. code-block:: bash

    $ ./track_cells.py \\
        --plot track \\
        /data/Experiment/2017-03-03

API Documentation
-----------------

"""

# Standard lib
import sys
import pathlib
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))

# 3rd party
import numpy as np

# Our own imports
from deep_hipsc_tracking.presets import load_preset
from deep_hipsc_tracking.utils import find_all_detectors
from deep_hipsc_tracking.tracking import (
    make_tracks_for_experiment, DetectorThresholds, find_link_functions)

# Constants
PLOT_STYLE = 'light'
SUFFIX = '.png'

OVERWRITE = False

MIN_POINT_ACTIVATION = 0.01  # Minimal activation under the mask
MAX_LINK_DIST = 25.0  # pixels travel - 8 mm
MAX_VELOCITY = 40.0  # pixels / minute
MAX_RELINK_ATTEMPTS = 10  # Maximumm tries to relink tracks

COLORS = []

LINK_FXN = 'balltree'  # Which algorithm to link with

# Command line interface


def parse_args(args=None):
    link_fxns = find_link_functions()

    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', type=pathlib.Path)
    parser.add_argument('--config-file', type=pathlib.Path,
                        help='Path to the global configuration file')
    parser.add_argument('-p', '--plot', dest='plots',
                        choices=('none', 'track', 'track_subset'),
                        action='append', default=[])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--link-fxn', choices=tuple(link_fxns.keys()),
                        help='Which algorithm to use to link frames')
    parser.add_argument('--processes', type=int,
                        help='Number of processes to use to link tracks')
    parser.add_argument('--min-point-activation', type=float,
                        help='Minimum confidence point activation to allow')
    parser.add_argument('-d', '--detector', dest='detectors', default=[], action='append',
                        help='List of detectors to use')
    parser.add_argument('--try-all-detectors', action='store_true',
                        help='Try all the detectors we have detections for')
    parser.add_argument('--plot-style', default=PLOT_STYLE)
    parser.add_argument('--suffix', default=SUFFIX)
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    config = load_preset(args.pop('config_file'))

    magnification = config.magnification
    space_scale = config.space_scale
    time_scale = config.time_scale
    microscope = config.microscope

    link_fxn = args.pop('link_fxn', None)
    if link_fxn is None:
        link_fxn = config.tracking['link_fxn']
    link_activation_threshold = config.tracking['activation_threshold']
    link_max_link_dist = config.tracking['max_link_dist']
    link_max_track_lag = config.tracking['max_track_lag']
    link_max_velocity = config.tracking['max_velocity']
    link_max_merge_dist = config.tracking['max_merge_dist']

    # Tracking round parameters
    link_step = config.tracking['link_step']
    impute_steps = config.tracking['impute_steps']
    max_relink_attempts = config.tracking['max_relink_attempts']

    detectors = config.tracking['detectors']

    print(f'Linking cell detections using method "{link_fxn}"')
    print(f'Linking at magnification: {magnification}')

    # Maximum link distance based on magnification and sampling rate
    max_link_dist = link_max_link_dist / space_scale
    print(f'Maximum link distance: {max_link_dist} px')

    # Maximum distance to merge peaks from two detections into one
    max_merge_dist = link_max_merge_dist / space_scale
    print(f'Maximum cell merge distance: {max_merge_dist} px')

    # Maximum allowed lag to connect track fragments
    max_track_lag = int(np.ceil(link_max_track_lag / time_scale))
    print(f'Maximum frame-frame lag: {max_track_lag} frames')

    # Maximum allowed velocity
    max_velocity = link_max_velocity * time_scale / space_scale
    print(f'Maximum track velocity: {max_velocity} pixels/frame')

    # Minimum point activation cutoff
    if link_activation_threshold is not None:
        print(f'Accepting points at threshold: {link_activation_threshold}')

    # Unpack the command line args
    rootdir = args.pop('rootdir')
    min_point_activation = args.pop('min_point_activation', None)
    plot_style = args.pop('plot_style')
    suffix = args.pop('suffix')
    plots = args.pop('plots', None)
    overwrite = args.pop('overwrite', False)
    if plots in (None, []):
        plots = ['none']

    try_all_detectors = args.pop('try_all_detectors')
    if try_all_detectors:
        detectors = find_all_detectors(rootdir, prefix='SingleCell')
    elif detectors in ([], None):
        detectors = [None]
    elif isinstance(detectors, str):
        detectors = [detectors]

    print(f'Using detectors: {detectors}')
    if len(detectors) < 1:
        raise ValueError('No detectors defined!')

    # Load the thresholds for this microscope
    thresholds = DetectorThresholds.by_microscope(microscope)
    detector_thresholds = thresholds.detector_thresholds

    print(args)

    # Run the tracking for each detector
    for detector in detectors:
        if min_point_activation is None:
            if detector is not None:
                detector = detector.lower()
            detector_point_activation = detector_thresholds.get(detector, MIN_POINT_ACTIVATION)
        else:
            detector_point_activation = min_point_activation
        print(f'Using detector "{detector}" at threshold "{detector_point_activation}"')
        make_tracks_for_experiment(rootdir,
                                   detector=detector,
                                   min_point_activation=detector_point_activation,
                                   max_link_dist=max_link_dist,
                                   max_track_lag=max_track_lag,
                                   max_velocity=max_velocity,
                                   max_merge_dist=max_merge_dist,
                                   link_fxn=link_fxn,
                                   plot_style=plot_style,
                                   suffix=suffix,
                                   plots=plots,
                                   overwrite=overwrite,
                                   link_step=link_step,
                                   impute_steps=impute_steps,
                                   max_relink_attempts=max_relink_attempts)


if __name__ == '__main__':
    main()
