#!/usr/bin/env python3

""" Run all the stages of the Deep Colony Tracking pipeline

Analyze the example confocal data:

.. code-block:: bash

    $ ./deep_track.py --preset confocal ../deep_hipsc_tracking/data/example_confocal/

To try the code out without running anything:

.. code-block:: bash

    $ ./deep_track.py --dry-run /path/to/rootdir

To run the pipeline

.. code-block:: bash

    $ ./deep_track.py /path/to/rootdir

To run the pipeline on several root directories at once

.. code-block:: bash

    $ ./deep_track.py /path/to/rootdir1 /path/to/rootdir2 ...

Run the pipeline starting at a particular stage

.. code-block:: bash

    $ ./deep_track.py --start-stage $STAGE_NAME ...

See which pipeline stages are available

.. code-block:: bash

    $ ./deep_track.py --list-stages

See ``deep_hipsc_tracking.pipeline`` for the detailed pipeline code

"""

# Standard lib
import sys
import pathlib
import argparse

THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
if (BASEDIR / 'deep_hipsc_tracking').is_dir():
    sys.path.insert(0, str(BASEDIR))

# Our own imports
from deep_hipsc_tracking.pipeline import ImagePipeline
from deep_hipsc_tracking.presets import list_presets

# Constants
PRESET = 'confocal'  # Which scope settings to use as default

PLOT_STYLE = 'light'
SUFFIX = '.png'

DATE_FMT = '%Y-%m-%d %H:%M:%S'

# Command line interface


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate, but do not actually run the pipeline')

    parser.add_argument('--plot-style', default=PLOT_STYLE,
                        help='Style for the output plots')
    parser.add_argument('--suffix', default=SUFFIX,
                        help='Suffix for the output plot')

    # Pipeline staging control
    pipeline_stages = tuple(ImagePipeline.pipeline_stages) + tuple(ImagePipeline.pipeline_special_stages.keys())

    parser.add_argument('-s', '--start-stage', choices=pipeline_stages,
                        help='First stage to run')
    parser.add_argument('--stop-stage', choices=pipeline_stages,
                        help='Last stage to run (inclusive)')
    parser.add_argument('-l', '--list-stages', action='store_true',
                        help='List available stages then exit')

    # File and directory cleaning
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing data')

    # Key arguments
    parser.add_argument('--preset', default=PRESET, choices=tuple(list_presets()),
                        help='Which config file preset to load for this analysis')
    parser.add_argument('rootdirs', nargs='+', type=pathlib.Path)
    return parser.parse_args(args=args)


def main(args=None):
    args = vars(parse_args(args=args))
    # Quick query for available pipeline stages
    list_stages = args.pop('list_stages')
    if list_stages:
        print('Pipeline stages:')
        for i, stage in enumerate(ImagePipeline.pipeline_stages):
            print(f'* [{i}] {stage}')
        print('')
        print('Special stages:')
        for stage, next_stage in sorted(ImagePipeline.pipeline_special_stages.items()):
            print(f'* "{stage}" runs before "{next_stage}"')
        print('')
        return

    start_stage = args.pop('start_stage')
    stop_stage = args.pop('stop_stage')

    for rootdir in args.pop('rootdirs'):
        try:
            # Run the whole processing pipeline
            pipeline = ImagePipeline(rootdir=rootdir, **args)
            pipeline.open()

            pipeline.run(start_stage=start_stage,
                         stop_stage=stop_stage)
        except Exception:
            pipeline.log(f'Error processing {rootdir}')
        finally:
            pipeline.close()


if __name__ == '__main__':
    main()
