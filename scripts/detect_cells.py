#!/usr/bin/env python3
""" Detect the cells in each frame using one of the CNNs

Requires the image contrast be corrected using ``extract_frames.py``

Training the net for 50,000 iterations on the AI-L-GFP dataset

.. code-block:: bash

    ./detect_cells.py \\
        --overwrite \\
        --composite-mode peak \\
        --peak-sharpness 8 \\
        --composite-stride 1 \\
        --data-resize 1 \\
        --image-dir ~/Desktop/AI-L-GFP-CELLS \\
        --save-plots \\
        --data-finder-mode training \\
        --plotdir /home/david/Desktop/ai-training-50000 \\
        --save-snapshot-for-plots \\
        50000

To extract detections from an already contrast corrected stack of images at 20x:

.. code-block:: bash

    ./detect_cells.py \\
        --data-resize 4 \\
        --composite-mode peak \\
        --composite-stride 1 \\
        --peak-sharpness 8 \\
        --data-finder-mode real \\
        --load-snapshot ~/Desktop/TrainingData/snapshots/snapshot-single-cell-POSTER-2017-05-10 \\
        --image-dir /data/Experiment/2017-08-29 \\
        --save-plots \\
        --composite-transforms rotations \\
        0

To extract detections from a contrast corrected stack at 10x:

.. code-block:: bash

    ./detect_cells.py \\
        --data-resize 1 \\
        --composite-mode peak \\
        --composite-stride 1 \\
        --peak-sharpness 8 \\
        --data-finder-mode real \\
        --load-snapshot ~/Desktop/TrainingData/snapshots/snapshot-single-cell-POSTER-2017-05-10 \\
        --image-dir /data/Experiment/2017-08-29 \\
        --save-plots \\
        --composite-transforms rotations \\
        0

To extract from several root directories at once, see ``composite_detect_cells.py``

"""

# Our own imports
import os
import pathlib
import sys
import datetime
import argparse
import time
import shutil
from typing import Optional, List

THISFILE = pathlib.Path(__file__).resolve()
BASEDIR = THISFILE.parent.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))
os.environ['KERAS_BACKEND'] = 'tensorflow'

# 3rd party
import matplotlib
matplotlib.use('Agg')

# Our own imports
from deep_hipsc_tracking.presets import load_preset
from deep_hipsc_tracking.utils import get_rootdir
from deep_hipsc_tracking.model import check_nvidia, SingleCellDetector, find_snapshot, DataFinders

# Constants
DETECTOR = 'countception'

DATADIR = BASEDIR / 'deep_hipsc_tracking' / 'data'
SNAPSHOT_DIR = DATADIR / 'model-snapshots'
ROI_OUTFILE = DATADIR / 'single_cell_weights.hdf5'

COMPOSITE_MODE = 'peak'
COMPOSITE_STRIDE = 32
COMPOSITE_BATCH_SIZE = 12
COMPOSITE_TRANSFORMS = 'none'

PEAK_SHARPNESS = 8
PEAK_DISTANCE = 3

RESPONSE_MIN = 0.0
RESPONSE_MAX = 0.8

BATCH_SIZE = 8  # Number of samples in a main batch

NB_EPOCH = 1000  # Number of steps to train

CUR_OPT_STAGE = 0  # Current optimizer stage

DATA_RESIZE = 4  # How much to reduce the sampling images by
DETECTION_THRESHOLD = 0.3  # 0.5 for a well trained net
DETECTION_EROSION = 0  # How many pixels of the heatmap to erode

# Optimizer stages
OPT_STAGES = [
    1e-4,
    5e-5,
    2e-5,
    1e-5,
    1e-6,
]

# Functions


def get_names_for_plots(image_file: pathlib.Path,
                        plotdir: Optional[pathlib.Path] = None):
    """ Get the names for the plotfile

    :param Path image_file:
        The image to evaluate
    :param Path plotdir:
        The directory to save plots to
    :returns:
        A tuple of plotfile, pointfile, response_file
    """

    if plotdir is None:
        rootdir = get_rootdir(image_file)
        plotrel = image_file.relative_to(rootdir)
        plotdir = rootdir / 'SingleCell' / plotrel.parent

    plotfile = plotdir / (image_file.stem + '.png')
    response_file = plotdir / (image_file.stem + '_resp.png')
    pointfile = plotdir / (image_file.stem + '.csv')
    return plotfile, pointfile, response_file


# Main function


def train_single_cell(config_file: pathlib.Path,
                      save_snapshot: bool = False,
                      save_snapshot_for_plots: bool = False,
                      load_snapshot: Optional[pathlib.Path] = None,
                      detector: str = DETECTOR,
                      overwrite: bool = False,
                      nb_epoch: int = NB_EPOCH,
                      opt_stage: int = CUR_OPT_STAGE,
                      image_files: Optional[List[pathlib.Path]] = None,
                      image_dirs: Optional[List[pathlib.Path]] = None,
                      composite_mode: str = COMPOSITE_MODE,
                      composite_stride: int = COMPOSITE_STRIDE,
                      composite_transforms: int = COMPOSITE_TRANSFORMS,
                      composite_batch_size: int = COMPOSITE_BATCH_SIZE,
                      save_plots: bool = False,
                      data_finder_mode: str = DataFinders.default_mode,
                      data_resize: int = DATA_RESIZE,
                      batch_size: int = BATCH_SIZE,
                      skip_existing_images: bool = False,
                      plotdir: Optional[pathlib.Path] = None,
                      max_plots: int = -1,
                      peak_sharpness: int = PEAK_SHARPNESS,
                      peak_distance: int = PEAK_DISTANCE,
                      detection_threshold: float = DETECTION_THRESHOLD,
                      detection_erosion: int = DETECTION_EROSION,
                      plot_timing_log_file: Optional[pathlib.Path] = None):

    config_file = config_file.resolve()

    print(f'Loading presets from {config_file}')
    config = load_preset(config_file)
    data_resize = config.segmentation['data_resize']
    detection_threshold = config.segmentation['detection_threshold']
    composite_transforms = config.segmentation['composite_transforms']
    composite_mode = config.segmentation['composite_mode']
    composite_stride = config.segmentation['composite_stride']

    peak_distance = config.segmentation['peak_distance']
    peak_sharpness = config.segmentation['peak_sharpness']

    skip_gpu_check = config.segmentation.get('skip_gpu_check', False)

    snapshot_time = datetime.datetime.now()
    snapshot_name = 'snapshot-single-cell-%Y%m%d-%H%M%S'
    snapshot_name = snapshot_time.strftime(snapshot_name)
    snapshot_dir = SNAPSHOT_DIR / snapshot_name

    if save_snapshot:
        print(f'Saving snapshot to: {snapshot_dir}')
        snapshot_dir.mkdir(parents=True)
        if ROI_OUTFILE.is_file():
            shutil.copy2(ROI_OUTFILE, snapshot_dir / 'single_cell_weights.hdf5')

    if overwrite:
        print('Overwriting!')
        if ROI_OUTFILE.is_file():
            ROI_OUTFILE.unlink()

    if skip_gpu_check:
        print("Skipping GPU check. Hope you know what you're doing...")
    else:
        print("Making sure the GPU looks okay...")
        check_nvidia()
        print("GPU seems: [GOOD]")

    print(f'Running for {nb_epoch} steps')

    # Load the GAN
    print('Loading Single Cell Detector...')
    t0 = time.time()
    load_snapshot = find_snapshot(snapshot_dir=load_snapshot,
                                  snapshot_prefix=SingleCellDetector.model_name,
                                  snapshot_root=SNAPSHOT_DIR)
    if load_snapshot is None:
        net = SingleCellDetector(data_finder_mode=data_finder_mode,
                                 peak_sharpness=peak_sharpness,
                                 peak_distance=peak_distance,
                                 detection_threshold=detection_threshold,
                                 detection_erosion=detection_erosion,
                                 batch_size=batch_size,
                                 detector=detector)
        net.load(snapshot_dir=snapshot_dir)
        # Snapshot the current configuration
        if save_snapshot:
            net.save_snapshot(snapshot_dir)
    else:
        net = SingleCellDetector.from_snapshot(snapshot_dir=load_snapshot,
                                               data_finder_mode=data_finder_mode,
                                               peak_sharpness=peak_sharpness,
                                               peak_distance=peak_distance,
                                               detection_threshold=detection_threshold,
                                               detection_erosion=detection_erosion,
                                               batch_size=batch_size,
                                               detector=detector)
    delta_t = time.time() - t0
    print(f'Loaded in {delta_t:1.2f} seconds')
    print(f'Training data under: {net.rootdir}')
    print(f'Training data finder: {net.data_finder_mode}')
    print(f'Training detector: {net.detector_name}')

    # Set the optimizer to the right level of aggressiveness
    opt = OPT_STAGES[opt_stage]
    print(f'Optimizing stage {opt_stage}')
    print(f'opt:  {opt}')
    net.set_opt_lr(opt=opt)

    # Training schedule
    if nb_epoch > 0:
        net.train_for_n(nb_epoch=nb_epoch)
        net.eval_model()

    if image_files is None:
        image_files = []
    if image_dirs is None:
        image_dirs = []

    for image_dir in image_dirs:
        print(f'Looking for {data_finder_mode} data in: {image_dir}')
        finder = DataFinders(data_finder_mode)
        for image_file in finder.data_finder(image_dir):
            image_files.append(image_file)
        print(f'Found {len(image_files)} files to test')

    # Write out the peak detections for each input image
    if plot_timing_log_file is None:
        timing_log_fp = None
    else:
        plot_timing_log_file.parent.mkdir(exist_ok=True, parents=True)
        timing_log_fp = plot_timing_log_file.open('wt')

    try:
        for i, image_file in enumerate(sorted(image_files)):
            if max_plots > 0 and i >= max_plots:
                break

            if save_plots:
                plotfile, pointfile, response_file = get_names_for_plots(image_file, plotdir=plotdir)
                if skip_existing_images and pointfile.is_file():
                    print(f'Skipping {pointfile}')
                else:
                    print(f'Detecting {pointfile}')
            else:
                plotfile = None
                pointfile = None
                response_file = None

            show = plotfile is None
            net.plot_response(image_file,
                              show=show,
                              plotfile=plotfile,
                              pointfile=pointfile,
                              response_file=response_file,
                              composite_stride=composite_stride,
                              composite_mode=composite_mode,
                              composite_transforms=composite_transforms,
                              composite_batch_size=composite_batch_size,
                              data_resize=data_resize,
                              timing_log_fp=timing_log_fp)
    finally:
        if timing_log_fp is not None:
            timing_log_fp.close()

    if save_snapshot_for_plots:
        if plotdir is None:
            # Shouldn't happen
            raise OSError(f'Provide a plot directory: {plotdir}')

        # Snapshot the model
        snapshot_dir = plotdir / 'snapshot'
        net.save_snapshot(snapshot_dir)

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--opt-stage', type=int, default=CUR_OPT_STAGE)

    parser.add_argument('--save-snapshot', action='store_true')
    parser.add_argument('--save-snapshot-for-plots', action='store_true',
                        help='Save a snapshot alongside the plots')
    parser.add_argument('--load-snapshot', type=pathlib.Path,
                        help='Path to the snapshot to load')

    parser.add_argument('--detector', choices=tuple(SingleCellDetector.get_detectors().keys()),
                        default=DETECTOR, help='Which neural net detector to use')

    parser.add_argument('--composite-mode', choices=('peak', 'mean'),
                        default=COMPOSITE_MODE)
    parser.add_argument('--composite-stride', type=int,
                        default=COMPOSITE_STRIDE)
    parser.add_argument('--composite-transforms', choices=('none', 'rotations'),
                        default=COMPOSITE_TRANSFORMS)
    parser.add_argument('--composite-batch-size', type=int, default=COMPOSITE_BATCH_SIZE,
                        help='Batch size for compositing the data')
    parser.add_argument('--peak-sharpness', type=int, default=PEAK_SHARPNESS,
                        help='1 - maximally smooth, 32 - pointwise sharp')
    parser.add_argument('--peak-distance', type=int, default=PEAK_DISTANCE,
                        help='Minimum distance between peaks')

    parser.add_argument('--data-resize', default=DATA_RESIZE, type=int,
                        help="Amount to resize the raw images by")
    parser.add_argument('--detection-threshold', default=DETECTION_THRESHOLD, type=float,
                        help='What detection level to cut off the detector at')
    parser.add_argument('--detection-erosion', default=DETECTION_EROSION, type=int,
                        help='How many pixels of the output image to erode')

    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int,
                        help='Batch size for training on the data')

    parser.add_argument('--data-finder-mode', default=DataFinders.default_mode,
                        choices=tuple(DataFinders.modes.keys()))

    parser.add_argument('-f', '--image-file', dest='image_files', action='append',
                        default=[], type=pathlib.Path)
    parser.add_argument('--image-dir', dest='image_dirs', action='append',
                        default=[], type=pathlib.Path)
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--skip-existing-images', action='store_true',
                        help='If True, dont plot images that already exist')
    parser.add_argument('--plotdir', type=pathlib.Path)
    parser.add_argument('--max-plots', type=int, default=-1,
                        help='Maximum number of plots to generate')

    parser.add_argument('--plot-timing-log-file', type=pathlib.Path,
                        help='Path to save the detection timings to')

    parser.add_argument('--config-file', type=pathlib.Path,
                        help='Path to the config file to load')

    parser.add_argument('nb_epoch', type=int, default=NB_EPOCH, nargs='?',
                        help='How many batches of training to run')

    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)
    train_single_cell(**vars(args))


if __name__ == '__main__':
    main()
