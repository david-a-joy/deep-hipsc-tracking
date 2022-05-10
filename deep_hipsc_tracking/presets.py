""" Presets for various microscopes

Store configuration for the default confocal and inverted microscopes

Classes:

* :py:func:`BasePreset`: Base class for presets. Subclass this to define new presets
* :py:func:`ConfocalPreset`: Parameters for the spinning disc confocal experiments
* :py:func:`InvertedPreset`: Parameters for the inverted microscope experiments

Functions:

* :py:func:`load_preset`: Load the preset from a file on disk
* :py:func:`get_preset`: Load a preset based on its name
* :py:func:`list_presets`: List all the defined presets in this file

"""

# Imports

import pathlib
import json
import configparser
from typing import List

# Classes


class BasePreset(object):
    """ Base class for preset data

    To use, define a subclass with at least the following attributes:

    * ``name``: Name for this class
    * ``space_scale``: Space scale factor in um/pixel
    * ``time_scale``: Time scale factor in min/frame
    * ``magnification``: Magnification as a multiplier (i.e. 5, 10, 20, etc)

    """

    def __init__(self, **kwargs):
        for key, val in vars(self.__class__).items():
            if key.startswith('_') or key in ('from_file', 'to_file'):
                continue
            setattr(self, str(key), val)

        for key, val in kwargs.items():
            if key.startswith('_'):
                continue
            setattr(self, str(key), val)

    def __eq__(self, other) -> bool:
        self_attrs = {k: v for k, v in vars(self).items()
                      if not (k.startswith('_') or k in ('from_file', 'to_file'))}
        other_attrs = {k: v for k, v in vars(other).items()
                       if not (k.startswith('_') or k in ('from_file', 'to_file'))}
        if self_attrs.keys() != other_attrs.keys():
            return False
        for key, self_val in self_attrs.items():
            other_val = other_attrs[key]
            if self_val != other_val:
                return False
        return True

    @classmethod
    def from_file(cls, config_file: pathlib.Path) -> 'BasePreset':
        """ Load the presets from a file

        :param Path config_file:
            Input config file, an ini formatted file where keys are strings and
            values are JSON encoded.
        """
        config = configparser.ConfigParser()
        sections = {}
        config.read(config_file)
        for section in config.sections():
            if section == 'base':
                sections.update({k: json.loads(v) for k, v in config[section].items()})
            else:
                subsection = sections.setdefault(section, {})
                subsection.update({k: json.loads(v) for k, v in config[section].items()})
        return cls(**sections)

    def to_file(self, config_file: pathlib.Path):
        """ Write the presets to a file

        :param Path config_file:
            Output config file, an ini formatted file where keys are strings and
            values are JSON encoded.
        """
        config = configparser.ConfigParser()
        sections = {}

        # Convert the attributes into config sections
        for key, val in vars(self).items():
            key = str(key)
            assert key != 'base'
            if key.startswith('_'):
                continue
            if not isinstance(val, dict):
                sections.setdefault('base', {})
                sections['base'][key] = json.dumps(val)
            else:
                sections.setdefault(key, {})
                for k, v in val.items():
                    sections[key][k] = json.dumps(v)

        for key, val in sections.items():
            config[key] = val

        # Write out the config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with config_file.open('wt') as fp:
            config.write(fp)


class ConfocalPreset(BasePreset):
    """ Presets for the spinning disc confocal """

    name = 'confocal'
    microscope = 'confocal'
    space_scale = 0.91
    time_scale = 3.0
    magnification = 10

    segmentation = {
        'data_resize': 1,  # Downsample factor for input images (1 for no downsampling, 2 for half size, etc)
        'composite_transforms': 'none',  # Transforms to apply to each image before detection
        'composite_type': 'mean',  # How to merge multiple segmentations
        'peak_distance': 3.0,  # um - spacing between the peaks
        'composite_mode': 'peak',  # How to fuse images
        'composite_stride': 1,  # Stride size for compositing
        'peak_sharpness': 8,  # pixels - radius of the cone around detections
        'exclude_border': 5,  # pixels - number of border pixels to ignore
        'detection_threshold': 0.01,  # Minimum pixel value for cell detection
        'skip_gpu_check': True,  # If True, don't check for a GPU before running
        'detectors': ['countception', 'residual_unet', 'fcrn_b_wide', 'composite'],
    }
    tracking = {
        'link_fxn': 'balltree',  # Which function to use to connect detections
        'activation_threshold': None,  # Use the hand-tuned neural net activations
        'max_track_lag': 15.0,  # minutes - maximum lag to connect tracks across frames
        'max_link_dist': 8.0,  # um - maximum distance to connect tracks between frames
        'max_velocity': 5.0,  # um/min - maximum velocity to plot
        'link_step': 1,  # How many frames to step while linking (1 - every frame, 2 - every other frame, etc)
        'impute_steps': 1,  # How many rounds of link imputation to run
        'max_relink_attempts': 10,  # How many rounds of linking to run
        'max_merge_dist': 7.0,  # um - Maximum distance to merge cells
        'detectors': ['composite'],
    }
    meshing = {
        'skip_plots': False,  # If True, skip generating the mesh plots period
        'skip_single_timepoints': False,  # If True, don't generate mesh plots for individual frames
        'max_distance': 50,  # um - maximum distance for cell neighbors
        'detectors': ['composite'],
    }


class InvertedPreset(BasePreset):
    """ Presets for the inverted microscope """

    name = 'inverted'
    microscope = 'confocal'
    space_scale = 0.323  # um/pixel
    time_scale = 5.0  # min/frame
    magnification = 20  # objective magnification

    segmentation = {
        'data_resize': 4,  # Downsample factor for input images (1 for no downsampling, 2 for half size, etc)
        'composite_transforms': 'none',  # Transforms to apply to each image before detection
        'composite_type': 'mean',  # How to merge multiple segmentations
        'peak_distance': 3.0,  # um - spacing between the peaks
        'composite_mode': 'peak',  # How to fuse images
        'composite_stride': 1,  # Stride size for compositing
        'peak_sharpness': 8,  # pixels - radius of the cone around detections
        'exclude_border': 5,  # pixels - number of border pixels to ignore
        'detection_threshold': 0.01,  # Minimum pixel value for cell detection
        'skip_gpu_check': False,  # If True, don't check for a GPU before running
        'detectors': ['countception', 'residual_unet', 'fcrn_b_wide', 'composite'],
    }
    tracking = {
        'link_fxn': 'balltree',  # Which function to use to connect detections
        'activation_threshold': None,  # Use the hand-tuned neural net activations
        'max_track_lag': 15.0,  # minutes - maximum lag to connect tracks across frames
        'max_link_dist': 8.0,  # um - maximum distance to connect tracks between frames
        'max_velocity': 5.0,  # um/min - maximum velocity to plot
        'link_step': 1,  # How many frames to step while linking (1 - every frame, 2 - every other frame, etc)
        'impute_steps': 1,  # How many rounds of link imputation to run
        'max_relink_attempts': 10,  # How many rounds of linking to run
        'max_merge_dist': 7.0,  # um - Maximum distance to merge cells
        'detectors': ['composite'],
    }
    meshing = {
        'skip_plots': False,  # If True, skip generating the mesh plots period
        'skip_single_timepoints': False,  # If True, don't generate mesh plots for individual frames
        'max_distance': 50,  # um - maximum distance for cell neighbors
        'detectors': ['composite'],
    }

# Functions


def list_presets() -> List[str]:
    """ List all the valid presets

    :returns:
        A list of all preset names that can be loaded
    """
    presets = []
    for cls in BasePreset.__subclasses__():
        name = getattr(cls, 'name', None)
        if name is not None:
            presets.append(name)
    return presets


def get_preset(preset: str) -> BasePreset:
    """ Load a specific preset by name

    :param str preset:
        The name of the preset to load
    :returns:
        The preset or None if no presets match
    """
    for cls in BasePreset.__subclasses__():
        name = getattr(cls, 'name', None)
        if name == preset:
            return cls()
    return None


def load_preset(config_file: pathlib.Path) -> BasePreset:
    """ Load the preset from a file

    :param Path config_file:
        Input config file, an ini formatted file where keys are strings and
        values are JSON encoded.
    :returns:
        A preset class with attributes matching the file
    """
    return BasePreset.from_file(config_file)
