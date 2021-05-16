""" Run all the stages of the Deep Colony Tracking pipeline

Functions:

* :py:func:`log_method`: Wrap methods in a logging and timing calls

Classes:

* :py:class:`ImagePipeline`: The actual pipeline steps

"""

# Standard lib
import sys
import time
import shutil
import pathlib
import datetime
import functools
import traceback
from typing import Callable, Optional, List

# Our own imports
from .presets import get_preset, load_preset
from .utils import call, print_cmd, Hypermap, is_nonempty_dir

# Constants
PRESET = 'confocal'  # Which scope settings to use as default

PLOT_STYLE = 'light'
SUFFIX = '.png'

DATE_FMT = '%Y-%m-%d %H:%M:%S'

# Functions


def log_method(fxn: Callable, date_fmt: str = DATE_FMT) -> Callable:
    """ Log the method to the class log

    :param func fxn:
        The method to wrap with logging
    :param str date_fmt:
        The datetime format string to pass to ``datetime.strftime()``
    :returns:
        A log wrapped function
    """

    @functools.wraps(fxn)
    def inner(self, *args, **kwargs):
        start_t0 = time.time()
        start_t0_dt = datetime.datetime.fromtimestamp(start_t0)
        self.log('##### Starting {} #####'.format(fxn.__name__))
        self.log('Start Time {} ({})\n'.format(start_t0_dt.strftime(date_fmt), start_t0))
        try:
            ret = fxn(self, *args, **kwargs)
        except Exception:
            error_t0 = time.time()
            error_t0_dt = datetime.datetime.fromtimestamp(error_t0)
            self.log('##### Error {} #####'.format(fxn.__name__))
            self.log('Error Time {} ({})\n'.format(error_t0_dt.strftime(date_fmt), time.time()))
            self.log('Error After {} secs\n'.format(error_t0 - start_t0))
            self.log(traceback.format_exc())
            raise
        finish_t0 = time.time()
        finish_t0_dt = datetime.datetime.fromtimestamp(finish_t0)
        self.log('##### Finished {} #####'.format(fxn.__name__))
        self.log('Finished Time {} ({})\n'.format(finish_t0_dt.strftime(date_fmt), time.time()))
        self.log('Finished After {} secs\n'.format(finish_t0 - start_t0))
        return ret
    return inner


# Main Processing Steps


class ImagePipeline(object):
    """ Image segmentation pipeline

    Run all the stages of the pipeline under a given directory.

    The default stages are listed under ``pipeline_stages``

    :param Path rootdir:
        The base data directory to process
    :param bool dry_run:
        If True, print out the commands but don't actually run
    :param bool overwrite:
        If True, overwrite already created data
    :param float magnification:
        If not None, the magnification for this image, as a floating point factor
        (e.g. 10.0 for 10x, 20.0 for 20x, etc)
    :param float time_scale:
        If not None, the number of minutes in between frames of a time series
    :param float space_scale:
        If not None, the width of a single (square) pixel in an image
    :param str suffix:
        Suffix for the plots to generate
    :param str plot_style:
        The plot style to use for generated plots
    :param str link_fxn:
        Frame to frame linking function to use
    :param float link_activation_threshold:
        Minimum activation level to link two cells
    :param float link_max_link_dist:
        Maximum spatial distance to allow a link between cells
    :param float link_max_track_lag:
        Maximum temporal distance to allow a link between cells
    :param Path nn_snapshot_dir:
        Directory where neural net snapshots are stored
    :param str nn_transforms:
        Which transforms to apply to each image when processing
    :param float nn_activation_threshold:
        Activation threshold to use for each neural net, or None for the default
    :param float nn_peak_distance:
        Minimum cell to cell spacing for non-maximum supression, or None for the default
    :param bool skip_tri_plots:
        If True, skip the plots from colony triangulation (faster, but less output)
    """

    # Run order for the pipeline
    pipeline_stages = [
        'write_config_file',
        'extract_frames',
        'ensemble_detect_cells',
        'track_cells',
        'mesh_cells',
    ]

    # Special short-cut stages
    # Name of stage: Stage it should run before
    pipeline_special_stages = {
        'detect_cells': 'track_cells',
    }

    def __init__(self,
                 rootdir: pathlib.Path,
                 preset: str = PRESET,
                 dry_run: bool = False,
                 overwrite: bool = False,
                 suffix: str = SUFFIX,
                 plot_style: str = PLOT_STYLE):

        # Have to resolve the input directory in case it's a relative path
        rootdir = pathlib.Path(rootdir).resolve()
        self.rootdir = rootdir
        self.preset = preset
        self.script_dir = pathlib.Path(__file__).resolve().parent.parent / 'scripts'
        self.dry_run = dry_run
        self.overwrite = overwrite

        assert self.script_dir.is_dir()

        # Plot controls
        self.plot_style = plot_style
        self.suffix = suffix

        # Paths
        self.python_bin = pathlib.Path(sys.executable)

        self.raw_data_dir = rootdir / 'RawData'
        self.extract_frames_dir = rootdir / 'Corrected'
        self.detect_cells_dir = rootdir / 'SingleCell'
        self.track_dir = rootdir / 'CellTracking'
        self.mesh_dir = rootdir / 'GridCellTracking'

        self.log_file = rootdir / 'deep_tracking.log'
        self.config_file = rootdir / 'deep_tracking.ini'
        self._log_fp = None
        self._start_t0 = None

        self._config = None

    @property
    def detectors(self) -> List[str]:
        """ Detectors specified in the config file for detecting cells """
        return self._config.segmentation['detectors']

    @property
    def link_detectors(self) -> List[str]:
        """ Detectors specified in the config file for linking cells """
        return self._config.tracking['detectors']

    @property
    def mesh_detectors(self) -> List[str]:
        """ Detectors specified in the config file for meshing cells """
        return self._config.meshing['detectors']

    # Utilities

    def open(self):
        """ Open the log and start processing """
        self.log_file.parent.mkdir(exist_ok=True, parents=True)
        self._log_fp = self.log_file.open('at')

        start_t0 = time.time()
        start_t0_dt = datetime.datetime.fromtimestamp(start_t0)

        self.log('#' * 10 + 'Start Processing' + '#' * 10)
        self.log('Start Time {} ({})\n'.format(start_t0_dt.strftime(DATE_FMT), start_t0))

        self._start_t0 = start_t0

        if self.overwrite:
            self.log('!!!!! OVERWRITING !!!!!')
        if self.dry_run:
            self.log('Dry Run...')

    def close(self):
        """ Close the log and finish processing """
        finish_t0 = time.time()
        finish_t0_dt = datetime.datetime.fromtimestamp(finish_t0)

        self.log('#' * 10 + 'End Processing' + '#' * 10)
        self.log('End Time {} ({})\n'.format(finish_t0_dt.strftime(DATE_FMT), finish_t0))
        if self._start_t0 is not None:
            self.log('Ended After {} secs\n'.format(finish_t0 - self._start_t0))
            self._start_t0 = None

        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None

    def run(self,
            start_stage: Optional[str] = None,
            stop_stage: Optional[str] = None):
        """ Run the pipeline

        :param str start_stage:
            If not None, the stage or stage index to start on
        :param str stop_stage:
            If not None, the stage or stage index to end on (inclusive)
        """
        pipeline_stages = list(self.pipeline_stages)

        special_stage = None
        if start_stage is None:
            start_stage = 0
        elif isinstance(start_stage, str):
            if start_stage in self.pipeline_special_stages:
                special_stage, start_stage = start_stage, self.pipeline_special_stages[start_stage]
            start_stage = pipeline_stages.index(start_stage)
        else:
            start_stage = int(start_stage)
        if stop_stage is None:
            stop_stage = len(pipeline_stages)
        elif isinstance(stop_stage, str):
            stop_stage = pipeline_stages.index(stop_stage) + 1  # 0-based indexing
        else:
            stop_stage = int(stop_stage) + 1

        # Jump to the correct point, and append a special_stage if we have one
        pipeline_stages = pipeline_stages[start_stage:stop_stage]
        if special_stage is not None:
            pipeline_stages = [special_stage] + pipeline_stages

        self.log(f'Running pipeline from: {pipeline_stages[0]}')
        self.log(f'Running pipeline to:   {pipeline_stages[-1]}')

        for i, pipeline_stage in enumerate(pipeline_stages):
            # Make sure the configuration was set up before running other stages
            if pipeline_stage != 'write_config_file':
                self.check_config_file()
            self.log(f'Running stage ({i+1} of {len(pipeline_stages)}): {pipeline_stage}')
            getattr(self, pipeline_stage)()

    def call(self, *args):
        """ Call the function

        :param str \\*args:
            The string arguments to the script to call
        """

        cmd = (self.python_bin, self.script_dir / args[0]) + args[1:]
        if self.dry_run:
            print_cmd(cmd)
            return
        call(cmd, cwd=self.rootdir)

    def log(self, msg: str):
        """ Log a message for this class

        :param str msg:
            The message to write to the log
        """
        print(msg)
        if self._log_fp is not None:
            self._log_fp.write(msg + '\n')
            self._log_fp.flush()

    def check_detector_dirs(self,
                            modality_dir: pathlib.Path,
                            detectors: Optional[List[str]] = None):
        """ See if we have all the expected detector subdirs

        :param Path modality_dir:
            The base name for the detector directory
        :param list[str] detectors:
            The list of detectors to look for
        :returns:
            True if they all seem present, False otherwise
        """
        if detectors is None:
            detectors = []
        elif isinstance(detectors, str):
            detectors = [detectors.lower()]

        print('Checking directories...')
        if detectors == []:
            all_okay = is_nonempty_dir(modality_dir)
            print('* {}: [{}]'.format(modality_dir, 'DONE' if all_okay else 'NOT DONE'))
            print('')
            return all_okay

        all_okay = True
        for detector in detectors:
            okay = False
            for try_detector in [detector.lower(), detector.capitalize()]:
                try_dir = modality_dir.parent / f'{modality_dir.name}-{try_detector}'
                if is_nonempty_dir(try_dir):
                    okay = True
                    break
            all_okay = all_okay and okay
            print('* {}: [{}]'.format(try_dir, 'DONE' if okay else 'NOT DONE'))
        print('* Overall: [{}]'.format('DONE' if all_okay else 'NOT DONE'))
        print('')
        return all_okay

    def clean_detector_dirs(self,
                            modality_dir: pathlib.Path,
                            detectors: Optional[List[str]] = None):
        """ Remove all the detector subdirs

        :param Path modality_dir:
            The base name for the detector directory
        :param list[str] detectors:
            The list of detectors to look for
        """
        if detectors is None:
            detectors = []
        elif isinstance(detectors, str):
            detectors = [detectors.lower()]

        print('Cleaning directories...')
        if detectors == []:
            shutil.rmtree(str(modality_dir))
            print(f'* {modality_dir}')
            return

        for detector in detectors:
            for try_detector in [detector.lower(), detector.capitalize()]:
                try_dir = modality_dir.parent / f'{modality_dir.name}-{try_detector}'
                if is_nonempty_dir(try_dir):
                    shutil.rmtree(str(try_dir))
                    print(f'* {try_dir}')

    def check_config_file(self):
        """ Make sure the config file exists and is loaded """

        if not self.config_file.is_file():
            if self.dry_run:
                print(f'Cannot find required config file at {self.config_file}')
            else:
                raise OSError(f'Cannot find required config file at {self.config_file}')

        if self._config is None:
            if self.dry_run:
                self._config = get_preset(self.preset)
            else:
                self._config = load_preset(self.config_file)

    # Individual scripts

    @log_method
    def write_config_file(self):
        """ Write the initial config file """

        if self.config_file.is_file():
            print(f'Loading configuration from "{self.config_file}"')
            config = load_preset(self.config_file)
        else:
            print(f'Loading preset configuration for "{self.preset}"')
            config = get_preset(self.preset)

        if config is None:
            err = f'No preset configuration available for "{self.preset}" or at {self.config_file}'
            if self.dry_run:
                print(err)
            else:
                raise OSError(err)

        print(f'Writing preset to {self.config_file}')
        if not self.dry_run:
            config.to_file(self.config_file)

        self._config = config

    @log_method
    def extract_frames(self):
        """ Run contrast correction on the data """

        if is_nonempty_dir(self.extract_frames_dir):
            if self.overwrite and not self.dry_run:
                print(f'Overwriting extract frames dir: {self.extract_frames_dir}')
                shutil.rmtree(str(self.extract_frames_dir))
            else:
                self.log(f'Skipping extract frames: {self.rootdir}')
                return

        # Make sure we have raw data
        if not self.dry_run:
            if not is_nonempty_dir(self.raw_data_dir):
                raise OSError(f'Cannot find raw data dir at: {self.raw_data_dir}')

        self.call('extract_frames.py',
                  '--config-file', self.config_file,
                  '--do-fix-contrast',
                  self.rootdir)

        # Make sure we got the frames extracted
        if not self.dry_run and not is_nonempty_dir(self.extract_frames_dir):
            raise OSError(f'Cannot find contrast corrected dir: {self.extract_frames_dir}')

    @log_method
    def detect_cells(self):
        """ Run a single deep neural net segmentation """

        if is_nonempty_dir(self.detect_cells_dir):
            if self.overwrite and not self.dry_run:
                print(f'Overwriting single cell dir: {self.detect_cells_dir}')
                shutil.rmtree(str(self.detect_cells_dir))
            else:
                self.log(f'Skipping detect_cells: {self.rootdir}')
                return

        if len(self.detectors) != 1:
            raise ValueError(f'Expected 1 detector, got {self.detectors}')

        self.call('detect_cells.py',
                  '--config-file', self.config_file,
                  '--detector', self.detectors[0],
                  '--image-dir', self.rootdir / 'Corrected',
                  '--save-plots',
                  '0')

        if not self.dry_run and not is_nonempty_dir(self.detect_cells_dir):
            raise OSError(f'Cannot find single cell detection dir: {self.detect_cells_dir}')

    @log_method
    def ensemble_detect_cells(self):
        """ Run several machine learning algorthms in an ensamble """

        if self.check_detector_dirs(self.detect_cells_dir, self.detectors):
            if self.overwrite and not self.dry_run:
                self.clean_detector_dirs(self.detect_cells_dir, self.detectors)
            else:
                self.log(f'Skipping composite_detect_cells: {self.rootdir}')
                return

        self.call('ensemble_detect_cells.py',
                  '--config-file', self.config_file,
                  self.rootdir)

        if not self.dry_run:
            if not self.check_detector_dirs(self.detect_cells_dir, self.detectors):
                raise OSError('Some single cell detections failed')

    @log_method
    def track_cells(self):
        """ Combine cell detections into cell tracks """

        if self.check_detector_dirs(self.track_dir, self.link_detectors):
            if self.overwrite:
                if self.dry_run:
                    print(f'Dry run... Not cleaning {self.track_dir}')
                else:
                    self.clean_detector_dirs(self.track_dir, self.link_detectors)
            else:
                self.log(f'Skipping track_cells: {self.rootdir}')
                return

        cmd = [
            'track_cells.py',
            self.rootdir,
            '--config-file', self.config_file,
            '--overwrite',
            '--plot', 'track',
            '--plot-style', self.plot_style,
            '--suffix', self.suffix,
        ]
        self.call(*cmd)

        if not self.dry_run:
            if not self.check_detector_dirs(self.track_dir, self.link_detectors):
                raise OSError('Some tracking detectors failed')

    @log_method
    def mesh_cells(self):
        """ Combine cell detections into whole colony meshes """

        if self.check_detector_dirs(self.mesh_dir, self.link_detectors):
            if self.overwrite:
                if self.dry_run:
                    print(f'Dry run... Not cleaning {self.mesh_dir}')
                else:
                    self.clean_detector_dirs(self.mesh_dir, self.link_detectors)
            else:
                self.log(f'Skipping mesh_cells: {self.rootdir}')
                return

        print('Calculating colony meshes')

        # Use few processes because we run out of memory
        processes = Hypermap.cpu_count()
        processes = max([1, int(round(processes / 4))])
        print(f'Calculating with {processes} parallel processes')

        self.call('mesh_cells.py',
                  '--config-file', self.config_file,
                  '-r', self.rootdir,
                  '--processes', processes,
                  '--suffix', self.suffix,
                  '--plot-style', self.plot_style)

        if not self.dry_run:
            if not self.check_detector_dirs(self.mesh_dir, self.link_detectors):
                raise OSError('Some colony meshing failed')
