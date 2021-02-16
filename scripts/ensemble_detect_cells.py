#!/usr/bin/env python3
""" Composite the cell detections using optimized network weighting

Segment and time a specific experiment:

.. code-block:: bash

    $ ./ensemble_detect_cells.py /data/Experiment/2017-01-30

"""

# Imports
import sys
import time
import json
import shutil
import argparse
import pathlib
import traceback
from typing import Optional, List

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))

from deep_hipsc_tracking.utils import call
from deep_hipsc_tracking.presets import load_preset
from deep_hipsc_tracking.tracking import DetectorThresholds

# Classes


class EnsembleDetectCells(object):
    """ Run multiple detectors and do a timing test

    :param Path rootdir:
        The directory to run detectors over
    :param bool overwrite:
        If True, clear the old directory before processing
    """

    def __init__(self,
                 rootdir: pathlib.Path,
                 config_file: pathlib.Path,
                 overwrite: bool = False):

        # Force the paths to be absolute
        rootdir = pathlib.Path(rootdir).resolve()
        config_file = pathlib.Path(config_file).resolve()

        self.rootdir = rootdir
        self.config_file = config_file
        self.timing_log_file = self.rootdir / 'SingleCell-timing-composite.txt'

        self.config = load_preset(self.config_file)

        self.overwrite = overwrite

        self.timestamps = {}

        # Metadata
        self.microscope = self.config.microscope
        self.magnification = self.config.magnification

        self.single_cell_dir = None

        self.detectors = None
        self.detector_snapshot_dirs = None

    def __enter__(self):
        self.start_timing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish_timing(exc_type is None)

    def start_timing(self):
        """ Start the timing """
        self.timestamps[('all', 'start')] = time.time()

    def finish_timing(self, okay: bool):
        """ Finish timing """
        status = 'success' if okay else 'error'
        self.timestamps[('all', 'end')] = time.time()

        all_time = self.timestamps[('all', 'end')] - self.timestamps[('all', 'start')]
        if ('composite', 'end') in self.timestamps:
            composite_time = self.timestamps[('composite', 'end')] - self.timestamps[('composite', 'start')]
        else:
            composite_time = None

        log_data = {
            'status': status,
            'detector_name': 'composite',
            'sub_detectors': self.detectors,
            'composite_time': composite_time,
            'overall_time': all_time,
        }
        for key, val in self.timestamps.items():
            log_data['timestamp_' + '_'.join(key)] = val

        with self.timing_log_file.open('wt') as fp:
            fp.write(json.dumps(log_data) + '\n')
            fp.flush()

    def load_single_cell_dir(self):
        """ Figure out where the single cell directory should be """

        if not self.rootdir.is_dir():
            raise OSError(f'Cannot find root directory: {self.rootdir}')

        self.single_cell_dir = self.rootdir / 'SingleCell'

        # Make sure we don't accidentally destroy stuff
        if self.single_cell_dir.is_dir():
            if self.overwrite:
                print(f'Overwriting old single cell dir: {self.single_cell_dir}')
                shutil.rmtree(str(self.single_cell_dir))
            else:
                raise OSError(f'Cannot overwrite existing single cell data: {self.single_cell_dir}')

    def load_detectors(self,
                       detectors=None,
                       with_default_detectors=False,
                       with_all_detectors=False):
        """ Make sure we have all the detectors before stating

        :param list[str] detectors:
            Names of the detectors to use
        :param str microscope:
            Name of the microscope to detect on
        :param bool with_default_detectors:
            If True, also add in the default detectors for this scope
        :param bool with_all_detectors:
            If True, use every single detector defined for this scope
        """
        print(f'Loading detectors for scope: {self.microscope}')
        thresholds = DetectorThresholds.by_microscope(self.microscope)

        if with_all_detectors:
            print('Loading all available detectors for scope...')
            detectors = thresholds.all_detectors

        if detectors in (None, []):
            print('Loading default detectors for scope...')
            detectors = thresholds.default_detectors
        elif isinstance(detectors, str):
            detectors = [detectors]
        if 'composite' in detectors:
            detectors.remove('composite')
        if len(detectors) < 1:
            raise ValueError("Need to load at least one detector")

        # Optionally, add in the default detectors for this scope
        if with_default_detectors:
            detectors.extend(d for d in thresholds.default_detectors if d not in detectors)
        print(f'Loaded {len(detectors)} detectors: {detectors}')

        detectors_for_scope = {d: thresholds.training_detectors[d] for d in detectors}

        detector_snapshot_dirs = {}
        for detector, snapshot_dir in sorted(detectors_for_scope.items()):
            snapshot_dir = pathlib.Path(snapshot_dir)
            if not snapshot_dir.is_dir():
                raise OSError(f'Cannot find neural net snapshot {snapshot_dir}')

            exp_files = ['config.json', 'single_cell_weights.hdf5']
            if not all([(snapshot_dir / f).is_file() for f in exp_files]):
                raise OSError(f'Neural net snapshot missing files: {snapshot_dir}')
            detector_snapshot_dirs[detector] = snapshot_dir
            print(f'* {detector}: {snapshot_dir}')

        self.detectors = detectors
        self.detector_snapshot_dirs = detector_snapshot_dirs

    def time_all_detectors(self):
        """ Run all detectors and time each of them """

        # Run a timing test
        for detector in sorted(self.detector_snapshot_dirs):
            self.time_detector(detector)

    def time_detector(self, detector: str):
        """ Time a single detector

        :param str detector:
            The detector name in detector_snapshot_dirs
        """
        snapshot_dir = self.detector_snapshot_dirs[detector]
        timing_log_file = self.rootdir / f'SingleCell-timing-{detector}.txt'
        single_cell_out_dir = self.rootdir / f'SingleCell-{detector}'

        if not self.overwrite and timing_log_file.is_file() and single_cell_out_dir.is_dir():
            print(f'Skipping already processed data {single_cell_out_dir}')
            return

        if timing_log_file.is_file():
            if self.overwrite:
                timing_log_file.unlink()
            else:
                raise OSError(f'Cannot overwrite timing data {timing_log_file}')

        cmd = [THISDIR / 'detect_cells.py',
               '--config-file', self.config_file,
               '--load-snapshot', snapshot_dir,
               '--image-dir', self.rootdir / 'Corrected',
               '--save-plots',
               '--composite-transforms', 'none',
               '--detector', detector,
               '--plot-timing-log-file', timing_log_file,
               '0']

        self.timestamps[(detector, 'start')] = time.time()
        call(cmd)
        self.timestamps[(detector, 'end')] = time.time()

        if not timing_log_file.is_file():
            raise OSError(f'Failed to get timings for {timing_log_file}')
        if not self.single_cell_dir.is_dir():
            raise OSError(f'Failed to get detections for {self.single_cell_dir}')

        single_cell_out_dir = self.rootdir / f'SingleCell-{detector}'
        print(f'Moving {self.single_cell_dir} to {single_cell_out_dir}')
        if single_cell_out_dir.is_dir():
            if self.overwrite:
                shutil.rmtree(str(single_cell_out_dir))
            else:
                raise OSError(f'Cannot overwrite output dir {single_cell_out_dir}')

        shutil.move(str(self.single_cell_dir), str(single_cell_out_dir))

    def time_composite(self):
        """ Run the composite over all detectors and time that """

        # Final timing test for compositing
        single_cell_out_dir = self.rootdir / 'SingleCell-composite'
        timing_log_file = self.timing_log_file

        if not self.overwrite and timing_log_file.is_file() and single_cell_out_dir.is_dir():
            print(f'Skipping already composited data {single_cell_out_dir}')
            return

        if timing_log_file.is_file():
            if self.overwrite:
                timing_log_file.unlink()
            else:
                raise OSError(f'Cannot overwrite timing file {timing_log_file}')

        cmd = [
            THISDIR / 'composite_cells.py',
            '--config-file', self.config_file,
        ]
        for detector in self.detector_snapshot_dirs.keys():
            cmd.extend(['--detector', detector])
        cmd.extend([
            self.rootdir,
        ])

        self.timestamps[('composite', 'start')] = time.time()
        call(cmd)
        self.timestamps[('composite', 'end')] = time.time()


# Main function


def ensemble_detect_cells(rootdir: pathlib.Path,
                          config_file: pathlib.Path,
                          overwrite: bool = False,
                          detectors: Optional[List[str]] = None,
                          skip_composite: bool = False,
                          with_default_detectors: bool = False,
                          with_all_detectors: bool = False):
    """ Run the timing test on this directory

    :param Path rootdir:
        The directory to analyze
    :param Path config_file:
        The configuration to load
    :param bool overwrite:
        If True, overwrite the timing data
    """
    # Load defaults from the config file
    config = load_preset(config_file)
    if detectors in (None, []):
        detectors = config.segmentation['detectors']

    with EnsembleDetectCells(rootdir=rootdir,
                             config_file=config_file,
                             overwrite=overwrite) as proc:
        proc.load_single_cell_dir()
        proc.load_detectors(detectors=detectors,
                            with_default_detectors=with_default_detectors,
                            with_all_detectors=with_all_detectors)
        proc.time_all_detectors()

        if not skip_composite:
            proc.time_composite()


# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdirs', nargs='+', type=pathlib.Path)
    parser.add_argument('--config-file', type=pathlib.Path,
                        help='Path to the configuration to load')
    parser.add_argument('--overwrite', action='store_true',
                        help='If True, overwrite existing composites')
    parser.add_argument('-d', '--detector', dest='detectors', default=[], action='append',
                        help='Detectors to use to analyze the data')
    parser.add_argument('--with-default-detectors', action='store_true',
                        help='Also add in the default detectors')
    parser.add_argument('--with-all-detectors', action='store_true',
                        help='Use all the available detectors')
    parser.add_argument('--skip-composite', action='store_true',
                        help='If True, skip the composite step')
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    for rootdir in args.pop('rootdirs'):
        try:
            ensemble_detect_cells(rootdir, **args)
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    main()
