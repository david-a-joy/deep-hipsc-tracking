""" Tools for dealing with the training data

* :py:func:`pair_detector_data`: Pair off the data for the detectors
* :py:func:`parse_detectors`: Parse detector specification strings

API Documentation
-----------------

"""

# Imports
import re
import pathlib
from typing import Optional, List, Dict, Tuple

# Our own imports
from ..utils import guess_channel_dir, parse_image_name

# Types
DetectorPairs = Dict[Tuple, List[pathlib.Path]]
DetectorSpec = Optional[List[str]]

# Constants

SKIP_DETECTORS = ('poster', 'composite', 'orig')

reDETECTOR = re.compile(r'^(?P<detector>[a-z0-9_]+)-r(un)?(?P<run_number>[0-9]+)-n?(?P<iter_number>[0-9]+[a-z]?)$', re.IGNORECASE)
reITERDIR = re.compile(r'^ai-upsample-(?P<training_set>[a-z]+)-n(?P<iter_number>[0-9]+)$', re.IGNORECASE)
reSINGLE_CELL = re.compile(r'^singlecell-(?P<name>[a-z0-9_-]+)$', re.IGNORECASE)
reRUNDIR = re.compile(r'^ai-upsample-(?P<training_set>[a-z]+)-(?P<detector>[a-z0-9_-]+)-run(?P<run_number>[0-9]+)$', re.IGNORECASE)
reCELL_FILE = re.compile(r'^(?P<cell_number>[0-9]+)cell[a-z0-9_-]*\.[a-z]+$', re.IGNORECASE)

# Functions


def parse_detectors(detectors: DetectorSpec,
                    data_type: str = 'real') -> List[str]:
    """ Parse the detector spec

    :param list[str] detectors:
        A list of detectors to process
    :param str data_type:
        One of 'real' or 'train', what detector type to process
    :returns:
        A list of detectors to use
    """

    if detectors in (None, []):
        return None
    if isinstance(detectors, str):
        detectors = [detectors]

    parsed_detectors = []
    for detector in detectors:
        if data_type != 'train':
            parsed_detectors.append(detector.lower().strip())
            continue

        match = reDETECTOR.match(detector)
        if not match:
            raise ValueError(f'Unparsable detector: {detector}')
        iter_number = match.group('iter_number').lower()
        if iter_number.endswith('k'):
            iter_number = int(iter_number[:-1]) * 1000
        elif iter_number.endswith('m'):
            iter_number = int(iter_number[:-1]) * 1000000
        else:
            iter_number = int(iter_number)

        parsed_detectors.append((
            match.group('detector'),
            int(match.group('run_number')),
            iter_number,
        ))
    return parsed_detectors


def pair_data_real(rootdir: pathlib.Path,
                   detectors: DetectorSpec = None,
                   training_set: str = 'inverted') -> DetectorPairs:
    """ Pair real data from running the detector

    :param Path rootdir:
        Path to the experiment dir to load all composites for (e.g. /data/Experiment/2017-01-30)
    :param list[str] detectors:
        The detector names to look for (otherwise return everything)
    :param str training_set:
        The training set to look for (one of 'inverted', 'confocal')
    :returns:
        A dictionary mapping channel, tile, timepoint: file for each detector and the number of detectors
    """

    if detectors is None:
        detectors = []
    elif isinstance(detectors, str):
        detectors = [detectors]
    detectors = [d.lower() for d in detectors]

    all_paths = {}
    num_detectors = 0

    for subdir in rootdir.iterdir():
        if not subdir.is_dir():
            continue
        match = reSINGLE_CELL.match(subdir.name)
        if not match:
            continue
        detector = match.group('name')
        if detector.lower() in SKIP_DETECTORS:
            print(f'Skipping invalid detector: {detector}')
            continue
        if detector is not None and detectors != [] and detector.lower() not in detectors:
            print(f'Skipping unmatched detector: {detector}')
            continue

        num_detectors += 1
        print(f'Loading {detector}: {subdir}')
        channel, channeldir = guess_channel_dir(subdir / 'Corrected', 'gfp')

        for tiledir in channeldir.iterdir():
            if not tiledir.is_dir():
                continue
            for imagefile in tiledir.iterdir():
                if not imagefile.is_file():
                    continue
                if not imagefile.name.endswith('_resp.png'):
                    continue
                imagedata = parse_image_name(imagefile.name[:-len('_resp.png')] + '.png')
                key = (channel, imagedata['tile'], imagedata['timepoint'])
                all_paths.setdefault(key, []).append(imagefile)
    return all_paths, num_detectors


def pair_data_any(rootdir: pathlib.Path,
                  detectors: DetectorSpec = None,
                  training_set: str = 'inverted') -> DetectorPairs:
    """ Pair single directory data from running the detector

    :param Path rootdir:
        Path to the experiment dir to load all composites for (e.g. /data/Experiment/2017-01-30)
    :param list[str] detectors:
        The detector names to look for (otherwise return everything)
    :param str training_set:
        The training set to look for (one of 'inverted', 'confocal')
    :returns:
        A dictionary mapping channel, tile, timepoint: file for each detector and the number of detectors
    """

    if detectors is None:
        detectors = []
    elif isinstance(detectors, str):
        detectors = [detectors]
    detectors = [d.lower() for d in detectors]

    all_paths = {}
    num_detectors = 0

    for subdir in rootdir.iterdir():
        if not subdir.is_dir():
            continue
        match = reSINGLE_CELL.match(subdir.name)
        if not match:
            continue
        detector = match.group('name')
        if detector.lower() in SKIP_DETECTORS:
            print(f'Skipping invalid detector: {detector}')
            continue
        if detector is not None and detectors != [] and detector.lower() not in detectors:
            print(f'Skipping unmatched detector: {detector}')
            continue

        num_detectors += 1
        print(f'Loading {detector}: {subdir}')

        targets = [subdir]
        while targets:
            p = targets.pop()
            if p.name.startswith('.'):
                continue
            if p.is_dir():
                targets.extend(p.iterdir())
                continue
            if not p.is_file():
                continue
            if not p.name.endswith('_resp.png'):
                continue
            relp = p.relative_to(subdir)

            # Just use the relative path as the key, no channel or tile
            key = (relp.parent / p.name[:-len('_resp.png')], None, None, )
            all_paths.setdefault(key, []).append(p)
    return all_paths, num_detectors


def find_all_training_dirs(rootdir: pathlib.Path,
                           detectors: DetectorSpec = None,
                           training_set: str = 'inverted'):
    """ Parse the training directory structure

    :param Path rootdir:
        The root directory to parse
    :param list[tuple] detectors:
        The list of detectors to search for as loaded by :py:func:`parse_detectors`
    :param str training_set:
        The training set to find (one of 'inverted', 'confocal')
    :returns:
        A generator yeilding meta, path pairs for each training directory
    """
    training_set_val = {
        'inverted': 'peaks',
        'confocal': 'confocal'
    }[training_set]

    # Recursively walk all the training directories
    subdirs = [({}, p) for p in rootdir.iterdir()]

    while subdirs != []:
        prefix, path = subdirs.pop()
        if not path.is_dir():
            continue

        # Try and parse the run directory
        rundir_match = reRUNDIR.match(path.name)
        if rundir_match:
            if training_set_val != rundir_match.group('training_set').lower():
                continue
            detector = rundir_match.group('detector')
            if detector in SKIP_DETECTORS:
                print(f'Skipping invalid detector: {path}')
                continue

            run_number = int(rundir_match.group('run_number'))
            assert 'detector' not in prefix
            assert 'run_number' not in prefix
            prefix['detector'] = detector
            prefix['run_number'] = int(run_number)
            subdirs.extend((dict(prefix), subpath)
                           for subpath in path.iterdir())
            continue

        # Try and parse the train directory
        iterdir_match = reITERDIR.match(path.name)
        if iterdir_match and (path / 'snapshot').is_dir():
            if training_set_val != iterdir_match.group('training_set').lower():
                continue
            iter_number = int(iterdir_match.group('iter_number'))
            assert 'iter_number' not in prefix
            prefix['iter_number'] = iter_number

            key = (prefix['detector'], prefix['run_number'], prefix['iter_number'])
            if detectors is None or key in detectors:
                yield prefix, path


def pair_data_train(rootdir: pathlib.Path,
                    detectors: DetectorSpec = None,
                    training_set: str = 'inverted') -> Tuple[DetectorPairs, int]:
    """ Pair training data from running the detector

    :param Path rootdir:
        Path to the experiment dir to load all composites for (e.g. ~/Desktop/TrainingData)
    :param list[str] detectors:
        The detector names to look for (otherwise return everything)
    :param str training_set:
        The training set to look for (one of 'inverted', 'confocal')
    :returns:
        A dictionary mapping channel, tile, timepoint: file for each detector and the number of detectors
    """
    target_dirs = [p for _, p in find_all_training_dirs(rootdir, detectors=detectors, training_set=training_set)]

    # Now pair off the final images
    num_detectors = len(target_dirs)
    all_paths = {}
    for target_dir in target_dirs:
        for imagefile in target_dir.iterdir():
            if not imagefile.name.endswith('cell_resp.png'):
                continue
            if not imagefile.is_file():
                continue
            match = reCELL_FILE.match(imagefile.name)
            if not match:
                continue
            timepoint = int(match.group('cell_number'))
            # Channel, tile, timepoint, where we ignore all but timepoint
            key = (None, None, timepoint)
            all_paths.setdefault(key, []).append(imagefile)
    return all_paths, num_detectors


# Main function


def pair_detector_data(rootdir: pathlib.Path,
                       detectors: DetectorSpec = None,
                       data_type: str = 'real',
                       training_set: str = 'inverted') -> Tuple[DetectorPairs, int]:
    """ Pair detector results

    :param Path rootdir:
        Path to the experiment dir to load all composites for (e.g. ~/Desktop/TrainingData)
    :param list[str] detectors:
        A list of detectors to process
    :param str data_type:
        One of 'real' or 'train', what detector type to process
    :param str training_set:
        Which training set data to search for (either 'inverted' or 'confocal')
    :returns:
        A dictionary mapping channel, tile, timepoint: file for each detector and the number of detectors
    """
    detectors = parse_detectors(detectors, data_type=data_type)

    if data_type == 'real':
        all_paths, num_detectors = pair_data_real(rootdir, detectors=detectors, training_set=training_set)
    elif data_type == 'any':
        all_paths, num_detectors = pair_data_any(rootdir, detectors=detectors, training_set=training_set)
    elif data_type == 'train':
        all_paths, num_detectors = pair_data_train(rootdir, detectors=detectors, training_set=training_set)
    else:
        raise KeyError(f'Unknown data type: {data_type}')
    return all_paths, num_detectors
