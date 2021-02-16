""" Thresholds for all the tracking data

Classes:

* :py:class:`InvertedThresholds`: Thresholds for the inverted data
* :py:class:`ConfocalThresholds`: Thresholds for the confocal data

To add a new microscope or modality:

Subclass :py:class:`DetectorThresholds` and add the required attributes.

API Documentation
-----------------

"""

import pathlib
from typing import List

# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
BASEDIR = THISDIR.parent
MODELDIR = BASEDIR / 'data' / 'weights'

# Classes


class DetectorThresholds(object):
    """ Find the parameters for the detectors """

    @classmethod
    def all_thresholds(cls):
        return [c for c in cls.__subclasses__()]

    @classmethod
    def by_microscope(cls, microscope: str) -> 'DetectorThresholds':
        """ Find the thresholds by microscope

        :param str microscope:
            The microscope to load
        :returns:
            The proper subclass of DetectorThresholds to use
        """
        classes = [c for c in cls.all_thresholds() if c.microscope == microscope]
        if len(classes) != 1:
            raise KeyError(f'Invalid microscope: "{microscope}"')
        return classes[0]()

    @classmethod
    def by_training_set(cls, training_set: str) -> 'DetectorThresholds':
        """ Find the thresholds by training set

        :param str training_set:
            The training set name to load
        :returns:
            The proper subclass of DetectorThresholds to use
        """
        classes = [c for c in cls.all_thresholds() if c.training_set == training_set]
        if len(classes) != 1:
            raise KeyError(f'Invalid training_set: "{training_set}"')
        return classes[0]()

    @property
    def default_detectors(self) -> List[str]:
        """ Return the default detectors for this class """
        detectors = [d.split('-', 1)[0] for d in self.detectors]
        assert all([d in self.training_detectors for d in detectors])
        return detectors

    @property
    def all_detectors(self) -> List[str]:
        """ Return all detectors for this class """
        return [d for d in self.training_detectors if d != 'composite']


class InvertedThresholds(DetectorThresholds):
    """ Thresholds for the inverted detectors """

    # Which microscope to use these thresholds for
    microscope = 'inverted'

    # Identifier for the training directories for this scope
    training_set = 'peaks'

    # Where the training directory lives
    train_rootdir = MODELDIR / 'inverted'

    # Dict of the top detectors for this training set (from ``make_train_curve``)
    training_detectors = {
        'composite': train_rootdir / 'composite',
        'fcrn_a_wide': train_rootdir / 'fcrn_a_wide-r3-75k-snapshot',
        'fcrn_b_wide': train_rootdir / 'fcrn_b_wide-r3-75k-snapshot',
        'countception': train_rootdir / 'countception-r3-50k-snapshot',
        'unet': train_rootdir / 'unet-r1-50k-snapshot',
        'residual_unet': train_rootdir / 'residual_unet-r4-25k-snapshot',
    }

    # Detectors to use for the combinatoral optimization step (for ``optimize_single_cell_detectors``)
    detectors = ['countception-r3-50k', 'residual_unet-r4-25k', 'fcrn_b_wide-r3-75k']

    # Weights to combine those detectors (from ``optimize_single_cell_detectors``)
    detector_weights = {
        ('countception-r3-50k', 'residual_unet-r4-25k', 'fcrn_b_wide-r3-75k'): (0.3, 0.8, 0.5),
    }

    # Thresholds for the final segmentation for each detector
    detector_thresholds = {
        'composite': 0.01,
        'countception': 0.55,
        'fcrn_a_wide': 0.55,
        'fcrn_b_wide': 0.3,
        'residual_unet': 0.4,
        'unet': 0.45,
    }


class ConfocalThresholds(DetectorThresholds):
    """ Thresholds for the confocal detectors """

    # Which microscope to use these thresholds for
    microscope = 'confocal'

    # Identifier for the training directories for this scope
    training_set = 'confocal'

    # Where the training directory lives
    train_rootdir = MODELDIR / 'confocal'

    # Dict of the top detectors for this training set (from ``make_train_curve``)
    training_detectors = {
        'composite': train_rootdir / 'composite-d3-final',
        'fcrn_b_wide': train_rootdir / 'fcrn_b_wide-r1-25k-snapshot',
        'countception': train_rootdir / 'countception-r1-10k-snapshot',
        'residual_unet': train_rootdir / 'residual_unet-r1-5k-snapshot',
    }

    # Detectors to use for the combinatoral optimization step (for ``optimize_single_cell_detectors``)
    detectors = ['countception-r1-10k', 'residual_unet-r1-5k', 'fcrn_b_wide-r1-25k']

    # Weights to combine those detectors (from ``optimize_single_cell_detectors``)
    detector_weights = {
        ('countception-r1-10k', 'residual_unet-r1-5k', 'fcrn_b_wide-r1-25k'): (0.3, 1.7, 0.1),
    }

    # Thresholds for the final segmentation for each detector
    detector_thresholds = {
        'composite': 0.01,
        'countception': 0.35,
        'fcrn_b_wide': 0.3,
        'residual_unet': 0.1,
    }
