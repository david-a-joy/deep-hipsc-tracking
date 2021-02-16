""" Statistics Utilities

T-tests and Power calculations:

* :py:func:`calc_pairwise_power`: Calculate the pair-wise power level to replicate a p-value over categories
* :py:func:`calc_pairwise_significance`: Calculate pair-wise significance over all categories in a dataframe
* :py:func:`calc_pairwise_anova`: Calculate one-way or two-way ANOVA over categories in a dataframe
* :py:func:`calc_pairwise_effect_size`: Calculate the effect size using Cohen's d in a dataframe
* :py:func:`calc_pairwise_batch_effect`: Calculate a batch effect using 2-way ANOVA in a dataframe
* :py:func:`calc_effect_size`: Calculate the effect size using Cohen's d

Filters and signal processing:

* :py:func:`calc_frequency_domain`: Convert a signal from time to frequency
* :py:func:`score_points`: Calculate the score vectors and IRR based on point correspondences
* :py:func:`bin_by_radius`: Bin warped data by radial group

Grouping and Pairing:

* :py:func:`groups_to_dataframe`: Convert group dictionaries to DataFrames
* :py:func:`group_by_contrast`: Group objects using a categorical attribute
* :py:func:`pair_all_tile_data`: Pair tile objects by tile and timepoint
* :py:func:`pair_train_test_data`: Use the cell index to pair off training and test files

Filesystem search and I/O:

* :py:func:`find_all_train_files`: Find all the training data under a directory, indexed by cell number
* :py:func:`load_points_from_maskfile`: Convert a mask or probability field into a point array
* :py:func:`load_training_data`: Transform training data to experiment/tile/timepoint
* :py:func:`load_train_test_split`: Load the train/test split for the data

Score Objects

* :py:class:`PointScore`: Manage ROC, TP/FP and precision/recall data for points
* :py:class:`ROCScore`: Store ROC/PR curves for a given set of scores

API Documentation
-----------------

"""

# Imports

import re
import json
import pathlib
import itertools
from collections import namedtuple
from typing import Dict, Tuple, Union, List, Optional

# 3rd party
import numpy as np

from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
from scipy.fftpack import fft

from statsmodels.stats.power import tt_ind_solve_power
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

from skimage.feature import peak_local_max, match_descriptors

from sklearn.metrics import roc_curve, precision_recall_curve, auc

import pandas as pd

# Our own imports
from . import load_image, guess_channel_dir, find_tiledirs, parse_image_name, to_json_types

# Constants
reTRAINFILE = re.compile(r'^(?P<index>[0-9]+)[a-z_-]+\.[a-z]+$', re.IGNORECASE)
reCELLNUM = re.compile(r'^(?P<cell_number>[0-9]+)(cell|cell_resp|dots)\.[a-z]+$', re.IGNORECASE)

MAX_DISTANCE = 1.0

CategoryType = Union[str, List[str]]

# Classes

CellPoints = namedtuple('CellPoints', 'x, y')
CellIndex = namedtuple('CellIndex', 'experiment, tile, timepoint, rot90, flip')


class TileMetaData(object):
    """ Meta data for individual tiles

    :param Path imagefile:
        The image file path
    :param Path rootdir:
        Experiment root directory
    :param str experiment:
        The name of the experiment
    :param int tile:
        The tile number
    :param int timepoint:
        The timepoint
    """

    def __init__(self,
                 imagefile: pathlib.Path,
                 rootdir: pathlib.Path,
                 experiment: str,
                 tile: int,
                 timepoint: int):

        self.imagefile = imagefile
        self.rootdir = rootdir

        self.cell_num = int(reCELLNUM.match(self.imagefile.name).group('cell_number'))

        self.experiment = experiment
        self.tile = tile
        self.timepoint = timepoint

        self.prev_tile = None
        self.next_tile = None
        self.ref_tile = None

        self.points_x = None
        self.points_y = None
        self.points_v = None

    def load_image(self, cutoff=0.01, keep_scale=True):
        if all([p is not None for p in (self.points_x, self.points_y, self.points_v)]):
            return
        self.points_x, self.points_y, self.points_v = load_points_from_maskfile(
            self.imagefile, cutoff=cutoff, keep_scale=keep_scale)

    def get_points(self, threshold):
        points = np.stack([self.points_x, self.points_y], axis=1)
        return points[self.points_v >= threshold, :]

    def __repr__(self):
        return f'TileMetaData({self.experiment},{self.tile},{self.timepoint})'


class PointScore(object):
    """ Score object for point data

    :param tuple train_points:
        The x, y coordinates of the positive reference points (score == 1.0)
    :param tuple test_points:
        The x, y (potentially v) coordinates of the results from an individual scorer
        If passed, v is the probability assigned to the given points
    :param float max_distance:
        The maximum inter-point distance considered a match
    """

    def __init__(self, train_points, test_points, max_distance=MAX_DISTANCE):

        # Unpack the train points
        if isinstance(train_points, tuple) and len(train_points) == 2:
            train_x, train_y = train_points
            train_xy = np.stack([train_x, train_y], axis=1)
        else:
            train_xy = np.array(train_points)
        assert train_xy.ndim == 2
        assert train_xy.shape[1] == 2

        # Unpack the test points
        if isinstance(test_points, tuple) and len(test_points) == 2:
            test_x, test_y = test_points
            test_xyv = np.stack([test_x, test_y], axis=1)
        elif isinstance(test_points, tuple) and len(test_points) == 3:
            test_x, test_y, test_v = test_points
            test_xyv = np.stack([test_x, test_y, test_v], axis=1)
        else:
            test_xyv = np.array(test_points)
        assert test_xyv.ndim == 2

        # Pad the test points if they have scores
        if test_xyv.shape[1] == 2:
            test_xyv = np.concatenate([test_xyv, np.ones((test_xyv.shape[0], 1))], axis=1)
        assert test_xyv.shape[1] == 3
        test_xy, test_v = test_xyv[:, :2], test_xyv[:, 2]

        self.train_xy = train_xy
        self.test_xy = test_xy
        self.test_v = test_v

        self.max_distance = max_distance

    def score_points(self):
        """ Calculate point matches """

        # Match points by distance
        if self.train_xy.shape[0] == 0 or self.test_xy.shape[0] == 0:
            matches = np.zeros((0, 2), dtype=np.int)
        else:
            matches = match_descriptors(self.train_xy, self.test_xy,
                                        metric='euclidean',
                                        max_distance=self.max_distance,
                                        cross_check=True)
        num_matches = matches.shape[0]

        train_mask = np.zeros((self.train_xy.shape[0], ), dtype=np.bool)
        test_mask = np.zeros((self.test_xy.shape[0], ), dtype=np.bool)

        train_mask[matches[:, 0]] = True
        test_mask[matches[:, 1]] = True

        num_missed = np.sum(~train_mask)
        num_extra = np.sum(~test_mask)
        num_total = num_matches + num_missed + num_extra

        print('{:.1%} Matched'.format(num_matches/num_total))
        print('{} Matches (True Positives)'.format(num_matches))
        print('{} Missed (False Negatives)'.format(num_missed))
        print('{} Extra (False Positives)'.format(num_extra))

        self.num_matches = num_matches
        self.num_missed = num_missed
        self.num_extra = num_extra
        self.num_total = num_total

        self.irr = num_matches / num_total
        self.precision = num_matches / (num_matches + num_extra)
        self.recall = num_matches / (num_matches + num_missed)

        print('IRR:       {:.1%}'.format(self.irr))
        print('Precision: {:.1%}'.format(self.precision))
        print('Recall:    {:.1%}'.format(self.recall))

        # Create the masks for ROC curves
        y_real = np.zeros((num_total, ), dtype=np.bool)
        y_score = np.zeros((num_total, ), dtype=np.float)

        y_real[:num_matches] = True
        y_score[:num_matches] = self.test_v[test_mask]

        y_real[num_matches:train_mask.shape[0]] = True
        y_score[num_matches:train_mask.shape[0]] = 0.0

        y_real[train_mask.shape[0]:num_total] = False
        y_score[train_mask.shape[0]:num_total] = self.test_v[~test_mask]

        self.y_real = y_real
        self.y_score = y_score


class ROCData(object):
    """ ROC and PR curves for data

    :param ndarray y_train:
        An Nx1 array of the True labels for the dataset
    :param ndarray y_test:
        An Nx1 array of the output of the net for the dataset
    """

    # Nice human readable aliases for some training runs
    FINAL_LABELS = {
        'countception-r3-50k': 'Count-ception',
        'fcrn_a_wide-r3-75k': 'FCRN-A',
        'fcrn_b_wide-r3-75k': 'FCRN-B',
        'residual_unet-r4-25k': 'Residual U-net',
        'unet-r1-50k': 'U-net',
    }

    def __init__(self, y_train, y_test):
        self.y_train = y_train
        self.y_test = y_test

        self.metadata = {}

        self.roc_data = {}
        self.precision_recall_data = {}

    @property
    def roc_auc(self):
        return self.roc_data['roc_auc']

    @property
    def pr_auc(self):
        return self.precision_recall_data['pr_auc']

    @property
    def label(self):
        return self.metadata['label']

    @property
    def detector(self):
        return self.metadata['detector']

    @property
    def data_split(self):
        return self.metadata['data_split']

    @classmethod
    def from_json(cls, datafile):
        """ Reload the class from JSON

        :param Path datafile:
            The data file to load from
        """
        with datafile.open('rt') as fp:
            stat_data = json.load(fp)

        # Load the raw arrays
        obj = cls(np.array(stat_data.pop('y_train')),
                  np.array(stat_data.pop('y_test')))

        # Reload the ROC data
        roc_data = stat_data.pop('roc_data')
        for key in ('true_positive_rate', 'false_positive_rate', 'thresholds'):
            if key in roc_data:
                roc_data[key] = np.array(roc_data[key])
        obj.roc_data = roc_data

        # Reload the P/R data
        precision_recall_data = stat_data.pop('precision_recall_data')
        for key in ('precision_rate', 'recall_rate', 'thresholds'):
            if key in precision_recall_data:
                precision_recall_data[key] = np.array(precision_recall_data[key])
        obj.precision_recall_data = precision_recall_data

        # Anything else must have been metadata
        obj.add_metadata(**stat_data.pop('metadata'))
        assert stat_data == {}
        return obj

    def to_json(self, datafile: pathlib.Path):
        """ Save the current state of the class to JSON

        :param Path datafile:
            The JSON data file to save to
        """
        stat_data = {
            'y_train': to_json_types(self.y_train),
            'y_test': to_json_types(self.y_test),
            'metadata': to_json_types(self.metadata),
            'roc_data': to_json_types(self.roc_data),
            'precision_recall_data': to_json_types(self.precision_recall_data),
        }
        with datafile.open('wt') as fp:
            json.dump(stat_data, fp)

    def to_plotdata(self):
        """ Only save the values that we need to re-plot everything

        :returns:
            A JSON serializable dictionary of the plot data
        """
        return {
            'metadata': to_json_types(self.metadata),
            'roc_data': to_json_types(self.roc_data),
            'precision_recall_data': to_json_types(self.precision_recall_data),
        }

    @classmethod
    def from_plotdata(cls, plotdata):
        """ Load only the values we need to plot things

        :param dict plotdata:
            The JSON data to load from
        """

        # No raw array data, just load empty values
        obj = cls(None, None)

        # Reload the ROC data
        roc_data = plotdata.pop('roc_data')
        for key in ('true_positive_rate', 'false_positive_rate', 'thresholds'):
            if key in roc_data:
                roc_data[key] = np.array(roc_data[key])
        obj.roc_data = roc_data

        # Reload the P/R data
        precision_recall_data = plotdata.pop('precision_recall_data')
        for key in ('precision_rate', 'recall_rate', 'thresholds'):
            if key in precision_recall_data:
                precision_recall_data[key] = np.array(precision_recall_data[key])
        obj.precision_recall_data = precision_recall_data

        # Anything else must have been metadata
        obj.add_metadata(**plotdata.pop('metadata'))
        assert plotdata == {}
        return obj

    def clear_raw_data(self):
        """ Clear raw data """
        self.y_train = None
        self.y_test = None

    def add_metadata(self, **kwargs):
        """ Add some metadata to this score object

        If present, the key 'label' is used when logging scores for this object
        """
        self.metadata.update(kwargs)

    def calc_roc_curve(self):
        """ Calculate the ROC score """

        # Calculate the FP and TP vectors and the thresholds
        false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_train, self.y_test)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        # Find the knee of the curve
        knee_dist = (true_positive_rate - 1.0)**2 + false_positive_rate**2
        knee_index = np.argmin(knee_dist)

        # Print some debugging so I don't think the process died
        label = str(self.metadata.get('label', ''))
        prefix = '{}: '.format(label) if label else ''
        print('{}ROC AUC {:0.3f}'.format(prefix, roc_auc))

        self.roc_data = {
            'false_positive_rate': false_positive_rate,
            'true_positive_rate': true_positive_rate,
            'roc_auc': roc_auc,
            'thresholds': thresholds,
            'knee_index': knee_index,
            'knee_dist': knee_dist,
            'knee_fpr': false_positive_rate[knee_index],
            'knee_tpr': true_positive_rate[knee_index],
            'knee_threshold': thresholds[knee_index],
        }

    def calc_precision_recall_curve(self):
        """ Calculate the P/R curve for this data """

        # Calculate the Precision and Recall vectors and the thresholds
        precision_rate, recall_rate, thresholds = precision_recall_curve(self.y_train, self.y_test)
        pr_auc = auc(recall_rate, precision_rate)

        # Find the knee of the curve
        knee_dist = precision_rate**2 + recall_rate**2
        knee_index = np.argmax(knee_dist)

        # Print some debugging so I don't think the process died
        label = str(self.metadata.get('label', ''))
        prefix = '{}: '.format(label) if label else ''
        print('{}P/R AUC {:0.3f}'.format(prefix, pr_auc))

        self.precision_recall_data = {
            'precision_rate': precision_rate,
            'recall_rate': recall_rate,
            'pr_auc': pr_auc,
            'thresholds': thresholds,
            'knee_index': knee_index,
            'knee_dist': knee_dist,
            'knee_precision': precision_rate[knee_index],
            'knee_recall': recall_rate[knee_index],
            'knee_threshold': thresholds[knee_index],
        }

    def plot_roc(self, ax, linewidth=3, color='k', annotate_knee=False):
        """ Plot ROC curve for this line

        :param Axes ax:
            The matplotlib axis object
        :param float linewidth:
            Width of the line to plot
        :param str color:
            Color of the line to plot
        :param bool annotate_knee:
            If True, annotate the knee of the curve
        """
        false_positive_rate = self.roc_data['false_positive_rate']
        true_positive_rate = self.roc_data['true_positive_rate']
        roc_auc = self.roc_data['roc_auc']

        knee_fpr = self.roc_data['knee_fpr']
        knee_tpr = self.roc_data['knee_tpr']
        knee_threshold = self.roc_data['knee_threshold']

        label = self.metadata.get('label')
        label = self.FINAL_LABELS.get(label, label)
        prefix = '{} '.format(label) if label else ''
        final_label = '{}(Area: {:0.3f})'.format(prefix, roc_auc)

        # Plot the actual ROC curve
        ax.plot(false_positive_rate, true_positive_rate,
                linewidth=linewidth, color=color,
                label=final_label)

        # Plot the knee point
        ax.plot([knee_fpr], [knee_tpr], 'o', color=color)
        if annotate_knee:
            ax.text(knee_fpr, knee_tpr-0.02, '{:0.3f}'.format(knee_threshold),
                    color=color, fontsize=24)

    def plot_precision_recall(self, ax, linewidth=3, color='k', annotate_knee=False):
        """ Plot P/R curve for this line

        :param Axes ax:
            The matplotlib axis object
        :param float linewidth:
            Width of the line to plot
        :param str color:
            Color of the line to plot
        :param bool annotate_knee:
            If True, annotate the knee of the curve
        """
        precision_rate = self.precision_recall_data['precision_rate']
        recall_rate = self.precision_recall_data['recall_rate']
        pr_auc = self.precision_recall_data['pr_auc']

        knee_precision = self.precision_recall_data['knee_precision']
        knee_recall = self.precision_recall_data['knee_recall']
        knee_threshold = self.precision_recall_data['knee_threshold']

        label = self.metadata.get('label')
        label = self.FINAL_LABELS.get(label, label)
        prefix = '{} '.format(label) if label else ''
        final_label = '{}(area: {:0.3f})'.format(prefix, pr_auc)

        # Plot the actual ROC curve
        ax.plot(recall_rate, precision_rate,
                linewidth=linewidth, color=color,
                label=final_label)

        # Plot the knee point
        ax.plot([knee_recall], [knee_precision], 'o', color=color)
        if annotate_knee:
            ax.text(knee_recall, knee_precision-0.02, '{:0.3f}'.format(knee_threshold),
                    color=color, fontsize=24)


# Functions


def groups_to_dataframe(groups, attr=None, column='Value', labels=None):
    """ Convert a group dictionary to a two-column dataframe

    :param dict[str, data] groups:
        The data map for the dataframe
    :param str attr:
        The attribute to extract (or None to use the groups directly)
    :param str column:
        The name for the column to create
    :param dict labels:
        The mapping between group_key and final condition name
    :returns:
        A DataFrame that can be fed into sns.violinplot
    """

    if labels is None:
        labels = {}

    group_name = []
    group_value = []

    for group_key, group in groups.items():
        group_label = str(labels.get(group_key, group_key))
        if attr is None:
            group_data = group
        else:
            group_data = []
            for g in group:
                val = getattr(g, attr, None)
                if val is None:
                    continue
                for v in val:
                    if isinstance(v, list):
                        group_data.extend(v)
                    else:
                        group_data.append(v)
            group_data = np.array(group_data)
        group_name.extend(group_label for _ in range(group_data.shape[0]))
        group_value.append(group_data)
    return pd.DataFrame({'Group': group_name, column: np.concatenate(group_value)})


def bin_by_radius(radius: np.ndarray,
                  value: np.ndarray,
                  num_bins: int = 4,
                  label: str = 'Value',
                  bin_type: str = 'uniform',
                  r_min: float = 0.0,
                  r_max: float = 1.0,
                  r_overflow: float = 1.1,
                  category_type: str = 'index') -> pd.DataFrame:
    """ Bin by radius into a dataframe

    :param ndarray radius:
        The radius vector to bin
    :param ndarray value:
        The value vector to make bins for
    :param int num_bins:
        The number of bins to generate
    :param str label:
        The label for the value vector
    :param str bin_type:
        One of 'uniform', 'area' - how to generate bin edges
    :param float r_min:
        The minimum radius to bin
    :param float r_max:
        The maximum radius to bin
    :param float r_overflow:
        The overflow radius for the final bin
    :param str category_type:
        Return value for the radius, one of "index" or "mean"
    :returns:
        A DataFrame with binned radii and all values detected at those radii
    """
    if not label:
        label = 'Value'

    df = {
        'Radius': [],
        label: [],
    }
    if bin_type in ('radius', 'uniform'):
        # Equally spaced edges
        edges = np.linspace(r_min, r_max, num_bins+1)
    elif bin_type == 'area':
        # Edges spaced to give annuli with equal area
        area = np.pi * (r_max**2 - r_min**2) / num_bins
        edges = [r_min]
        for i in range(num_bins):
            edges.append(np.sqrt(area / np.pi + edges[-1]**2))
        edges = np.array(edges)
    else:
        raise KeyError(f'Unknown bin type: "{bin_type}"')
    assert edges.shape[0] == num_bins + 1
    print(f'Generating bins at: {edges}')

    real_mask = ~np.isnan(radius)
    radius = radius[real_mask]
    value = value[real_mask]

    if radius.shape[0] < 1 or value.shape[0] < 1:
        raise ValueError('No valid measurements for radius')

    for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
        if category_type == 'index':
            cat = f'{e0:0.1f}-{e1:0.1f}'
        elif category_type == 'mean':
            cat = np.mean([e0, e1])
        else:
            raise ValueError(f'Unknown category type: {category_type}')

        if i == num_bins - 1:
            e1 = r_overflow
        mask = np.logical_and(radius >= e0, radius < e1)

        df['Radius'].extend([cat] * int(np.sum(mask)))
        df[label].extend(value[mask])
    return pd.DataFrame(df)


def group_by_contrast(tiles, contrast=None, categories=None, fixed=None,
                      only_experiments=None, filter_uniform_experiments=False):
    """ Group the tiles by the experimental contrasts

    :param tiles:
        The list of tiles to group
    :param str contrast:
        The attribute to group by or None to collect everything into one group
    :param list categories:
        If not None, a list of allowed categories for the contrast
    :param dict fixed:
        A dictionary of key: value pairs. Only tiles where key == value will be used
    :param bool filter_uniform_experiments:
        If True, ignore tiles beloning to experiments where contrast never changes
    :param list[str] only_experiments:
        Only load the data for these experiments
    :returns:
        A dictionary of group: tiles for all unique group values
    """

    groups = {}
    if fixed is None:
        fixed = {}

    print('Grouping contrast {}'.format(contrast))
    print('Loading {} tiles'.format(len(tiles)))
    if contrast is not None:
        print([getattr(tile, contrast, None) for tile in tiles])

    # Throw out experiments that don't perturb this variable
    if contrast is not None and (filter_uniform_experiments or only_experiments is not None):
        tiles_by_experiment = {}
        for tile in tiles:
            tiles_by_experiment.setdefault(tile.rootname, []).append(tile)
        # Throw out any experiments that don't match a requested experiment
        if only_experiments is not None:
            tiles_by_experiment = {experiment: exp_tiles
                                   for experiment, exp_tiles in tiles_by_experiment.items()
                                   if experiment in only_experiments}
        # Throw out any experiments that have only one level of our contrast
        tiles = []
        for experiment, exp_tiles in tiles_by_experiment.items():
            if filter_uniform_experiments:
                exp_contrast = [getattr(t, contrast, None) for t in exp_tiles]
                exp_contrast = [t for t in exp_contrast if t is not None]
                if exp_contrast == [] or all([e == exp_contrast[0] for e in exp_contrast]):
                    print('Filtering: {}'.format(experiment))
                    continue
            tiles.extend(exp_tiles)
        print('{} tiles survived filtering'.format(len(tiles)))

    # Now break the tiles up by experimental group
    skiped_contrasts = set()
    for tile in tiles:
        # Make sure the tile matches all required constant values
        should_use = True
        for key, value in fixed.items():
            if getattr(tile, key, None) != value:
                should_use = False
                break
        if not should_use:
            continue
        if contrast is None:
            groups.setdefault('all', []).append(tile)
        else:
            # Split the tiles up by requested contrasts
            group = getattr(tile, contrast, None)
            if categories is not None and group not in categories:
                skiped_contrasts.add(group)
                continue
            groups.setdefault(group, []).append(tile)

    print('Skipped contrasts: {}'.format(skiped_contrasts))
    print('Got {} groups:'.format(len(groups)))
    for group, tiles in groups.items():
        print('{}: {} tiles'.format(group, len(tiles)))
    return groups


def calc_frequency_domain(time: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray]:
    """ Calculate the frequency domain data

    :param ndarray time:
        The time array in seconds
    :param ndarray signal:
        The signal intensity
    :returns:
        The frequency array, the power at each frequency
    """

    dt = time[1] - time[0]
    num_points = time.shape[0]

    yf = 2.0 / num_points * np.abs(fft(signal))
    xf = np.linspace(0.0, 0.5 / dt, num_points//2)
    return xf, 10 * np.log10(yf[:num_points//2])


def load_training_data(training_dirs, rev_index):
    """ Load the training data

    :param list[Path] training_dirs:
        The list of training directories to load the data for
    :param dict rev_index:
        The reverse index returned by load_index
    :returns:
        A dictionary mapping transform: pairs of data
    """

    collected_images = {}
    for training_dir in training_dirs:
        for p in training_dir.iterdir():
            if not p.name.endswith(('dots.png', 'cell_resp.png')):
                continue
            if not p.is_file():
                continue
            match = reCELLNUM.match(p.name)
            if not match:
                continue
            cell_num = int(match.group('cell_number'))

            key = rev_index[cell_num]
            assert key.experiment is not None
            rec = TileMetaData(imagefile=p,
                               rootdir=training_dir,
                               experiment=key.experiment,
                               tile=key.tile,
                               timepoint=key.timepoint)

            transform_key = (key.rot90, key.flip)
            transform_images = collected_images.setdefault(transform_key, {})

            time_key = (key.experiment, key.tile, key.timepoint)
            transform_images.setdefault(time_key, []).append(rec)

    # Cross link all the data in space
    num_time_recs = len(training_dirs)
    for transform_recs in collected_images.values():
        for time_recs in transform_recs.values():
            assert len(time_recs) == num_time_recs
            for rec in time_recs[1:]:
                assert rec.ref_tile is None
                rec.ref_tile = time_recs[0]

    # Cross link all the data in time
    for transform_recs in collected_images.values():
        for pair_idx in range(num_time_recs):
            all_recs = [t[pair_idx] for t in transform_recs.values()]
            pair_all_tile_data(all_recs)

    return collected_images


def pair_all_tile_data(all_tile_data):
    """ Connect all the tile data together into pairs

    :param list[TileMetaData] all_tile_data:
        The list of tile data objects to link
    :returns:
        A list containing the first member of every tile data pair
    """

    # Sort the data so that proximate tiles and timepoints are near eachother
    all_tile_data = list(sorted(all_tile_data, key=lambda x: (x.experiment, x.tile, x.timepoint)))

    linked_tile_data = []
    while all_tile_data != []:
        tile_data = all_tile_data.pop(0)
        link_data = None
        for td in all_tile_data:
            if td.experiment == tile_data.experiment and td.tile == tile_data.tile:
                if td.timepoint == tile_data.timepoint:
                    err = 'Duplicate tile for ({}, {}, {})'.format(td.experiment, td.tile, td.timepoint)
                    raise ValueError(err)

                if abs(td.timepoint - tile_data.timepoint) == 1:
                    link_data = td
                    break

        if link_data is None:
            # No match
            continue

        if link_data.timepoint > tile_data.timepoint:
            link_data.prev_tile = tile_data
            tile_data.next_tile = link_data
            linked_tile_data.append(tile_data)
        else:
            link_data.next_tile = tile_data
            tile_data.prev_tile = link_data
            linked_tile_data.append(link_data)

    return linked_tile_data


def calc_pairwise_power(data: pd.DataFrame, category: str, score: str,
                        power: float = 0.8,
                        alpha: float = 0.05,
                        ratio: float = 1.0) -> Dict[Tuple[str], str]:
    """ Work out the sample sizes needed to get stat sig results

    :param DataFrame data:
        The data frame containging the records to evaluate
    :param str category:
        The column to categorize over
    :param str score:
        The column to evaluate over
    :returns:
        A dict with pairwise keys, and values corresponding to the required number of samples
    """

    categories = {}
    for c, s in zip(data[category], data[score]):
        categories.setdefault(str(c), []).append(s)
    print('Got {} unique categories'.format(len(categories)))

    print('Solving pairwise power...')

    # Do lexographic sorting
    samples = {}
    keys = list(sorted([k for k in categories.keys()], key=lambda x: str(x)))
    for key1 in keys:

        cat1_mean = np.mean(categories[key1])
        cat1_std = np.std(categories[key1])

        for key2 in keys:
            if key1 == key2:
                continue
            if (key1, key2) in samples or (key2, key1) in samples:
                continue
            cat2_mean = np.mean(categories[key2])
            cat2_std = np.std(categories[key2])

            # Params we want
            cat_std = np.sqrt(cat2_std**2 + cat1_std**2)
            cat_delta = abs(cat1_mean - cat2_mean)

            print('{} vs {}: delta {}'.format(key1, key2, cat_delta))

            needed_samples = tt_ind_solve_power(effect_size=cat_delta/cat_std,
                                                nobs1=None,
                                                power=power,
                                                alpha=alpha,
                                                ratio=ratio,
                                                alternative='two-sided')
            print('{} vs {} - {}'.format(key1, key2, needed_samples))
            samples[(key1, key2)] = needed_samples
    return samples


def calc_pairwise_significance(data: pd.DataFrame,
                               category: CategoryType,
                               score: str,
                               alpha: float = 0.05,
                               test_fxn: str = 't-test',
                               control: Optional[CategoryType] = None,
                               group: Optional[CategoryType] = None,
                               correction_fxn: str = 'holm',
                               keep_non_significant: bool = False) -> Dict[Tuple[str], float]:
    """ Work out the significance of the differences

    :param DataFrame data:
        The data frame containging the records to evaluate
    :param str category:
        The column or columns to categorize over
    :param str score:
        The column to evaluate over
    :param float alpha:
        The significance threshold to test for
    :param str method:
        Which `method <http://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html>`_
        to use to correct for multiple comparisons
    :param str test_fxn:
        Which statistical test function to use ('t-test', 'u-test', 'ks-test') or a
        function that takes two one dimensional sets of data and returns an uncorrected pvalue
    :param list[str] control:
        If not None, the control or list of controls to compare to (default is all vs all)
    :param list[str] group:
        If not None, collect each comparison over all the unique values for this category
    :param bool keep_non_significant:
        If True, keep all values, otherwise only return values with p < alpha
    :returns:
        A dictionary with (cat1, cat2): pvalue keys for each pairwise comparison
    """
    if isinstance(test_fxn, str):
        test_fxn = {
            't-test': lambda x, y: ttest_ind(x, y, equal_var=True)[1],
            'welch-t-test': lambda x, y: ttest_ind(x, y, equal_var=False)[1],
            'u-test': lambda x, y: mannwhitneyu(x, y, alternative='two-sided')[1],
            'ks-test': lambda x, y: ks_2samp(x, y, alternative='two-sided')[1],
        }[test_fxn.lower()]
    if not callable(test_fxn):
        raise KeyError(f'Test function "{test_fxn}" does not appear to be callable')

    if isinstance(category, str):
        category = [category]
    else:
        category = list(category)
    if control is None:
        control = []
    elif isinstance(control, str):
        control = [control]
    else:
        control = list(control)
    if group is None:
        group = []
    elif isinstance(group, str):
        group = [group]
    else:
        group = list(group)

    vectors = [data[column] for column in group + category + [score]]

    # Unpack the DataFrame into a dictionary of (key1, key2, ...): value
    group_categories = {}
    for v in zip(*vectors):
        group_key = v[:len(group)]
        cat_key = v[len(group):len(group)+len(category)]
        val = v[-1]

        group_key = tuple(str(k) for k in group_key)
        categories = group_categories.setdefault(group_key, {})

        cat_key = tuple(str(k) for k in cat_key)
        categories.setdefault(cat_key, []).append(val)
    print(f'Got {len(group_categories)} unique groups')
    unique_categories = set()
    for categories in group_categories.values():
        unique_categories.update(categories.keys())
    print(f'Got {len(unique_categories)} unique categories')

    # Handle the case where there are no contrasts?
    if len(unique_categories) < 2:
        return {}

    print('Solving pairwise significance...')

    # Do lexographic sorting
    keys = list(sorted(unique_categories))
    if len(control) < 1:
        control = keys
    significance = {}
    for group_key, categories in group_categories.items():
        for key1 in control:
            if isinstance(key1, str):
                key1 = (key1, )
            gkey1 = group_key + key1
            for key2 in keys:
                if isinstance(key2, str):
                    key2 = (key2, )
                gkey2 = group_key + key2
                if gkey1 == gkey2:
                    continue
                if (gkey1, gkey2) in significance:
                    continue
                if (gkey2, gkey1) in significance:
                    continue
                if len(categories[key1]) < 2 or len(categories[key2]) < 2:
                    continue
                pvalue = test_fxn(categories[key1], categories[key2])
                # print(f'{key1} vs {key2}: {pvalue}')
                significance[(gkey1, gkey2)] = pvalue

    # Correct for multiple comparisons
    significance_keys = list(sorted(significance.keys()))
    if len(significance_keys) < 2:
        corr_significance = {key: pvalue
                             for key, pvalue in significance.items()
                             if keep_non_significant or pvalue <= alpha}
    else:
        pvalues = np.array([significance[key] for key in significance_keys])
        corr_significance = multipletests(pvalues,
                                          alpha=alpha,
                                          method=correction_fxn,
                                          is_sorted=False)[1]
        corr_significance = {key: corr_significance[i]
                             for i, key in enumerate(significance_keys)
                             if keep_non_significant or corr_significance[i] <= alpha}
    # Write out the final significance
    final_corr_significance = {}
    for (key1, key2), pvalue in corr_significance.items():
        if len(key1) == 1:
            key1 = str(key1[0])
        if len(key2) == 1:
            key2 = str(key2[0])
        # print(f'{key1} vs {key2}: (p={pvalue})')
        final_corr_significance[(key1, key2)] = pvalue
    return final_corr_significance


def calc_pairwise_effect_size(data: pd.DataFrame,
                              category: CategoryType,
                              score: str) -> Dict[Tuple[str], float]:
    """ Calculate the effect size for a given set of data

    :param DataFrame data:
        The data frame containging the records to evaluate
    :param str category:
        The column to categorize over
    :param str score:
        The column to evaluate over
    :returns:
        A Dataframe with several measures of effect size
    """

    if isinstance(category, str):
        category = [category]
    else:
        category = list(category)
    vectors = [data[column] for column in category + [score]]

    # Unpack the DataFrame into a dictionary of (key1, key2, ...): value
    categories = {}
    for v in zip(*vectors):
        key = v[:-1]
        if len(key) == 1:
            key = str(key[0])
        else:
            key = tuple(str(k) for k in key)
        val = v[-1]
        categories.setdefault(key, []).append(val)
    print('Got {} unique categories'.format(len(categories)))

    # Handle the case where there are no contrasts?
    if len(categories) < 2:
        return {}

    print('Solving pairwise effect size...')

    # Do lexographic sorting
    keys = list(sorted([k for k in categories.keys()], key=lambda x: x))
    effect_size = {}
    for key1 in keys:
        for key2 in keys:
            if key1 == key2:
                continue
            if (key1, key2) in effect_size:
                continue
            if (key2, key1) in effect_size:
                continue
            if len(categories[key1]) < 2 or len(categories[key2]) < 2:
                continue
            mean1 = np.nanmean(categories[key1])
            mean2 = np.nanmean(categories[key2])
            std1 = np.nanstd(categories[key1])
            std2 = np.nanstd(categories[key2])
            n1 = np.sum(~np.isnan(categories[key1]))
            n2 = np.sum(~np.isnan(categories[key2]))

            effect_size[(key1, key2)] = calc_effect_size(mean1, mean2, std1, std2, n1, n2)
    return effect_size


def calc_pairwise_anova(data: pd.DataFrame,
                        category: CategoryType,
                        score: str,
                        include_interactions: bool = True) -> pd.DataFrame:
    """ Calculate one-way or two-way ANOVA over the dataframe

    :param DataFrame data:
        The data frame containging the records to evaluate
    :param str category:
        The column to categorize over
    :param str score:
        The column to evaluate over
    :returns:
        A dict of paired keys and pvalues
    """
    if isinstance(category, str):
        category = [category]
    else:
        category = list(category)

    factors = ['C({})'.format(c) for c in category]
    if include_interactions and len(factors) > 1:
        factors.extend(k1 + ':' + k2 for k1, k2 in itertools.combinations(factors, 2))
    formula = '{} ~ {}'.format(score, ' + '.join(factors))

    print('Got factors: {}'.format(factors))
    print('Fitting model: {}'.format(formula))

    linear_model = ols(formula, data).fit()
    print(linear_model.summary())

    linear_stats = anova_lm(linear_model)

    # Rename back to normal sane names
    linear_stats.rename(index=lambda x: x.replace('C(', '').replace(')', ''),
                        inplace=True)
    return linear_stats


def calc_pairwise_batch_effect(data: pd.DataFrame,
                               category: CategoryType,
                               batch: CategoryType,
                               score: str,
                               model: str = 'ols') -> pd.DataFrame:
    """ Calculate the effect due to batch, preserving variation due to category

    :param DataFrame data:
        The data frame containing the records to evaluate
    :param str category:
        The column to categorize over
    :param str batch:
        The column to calculate batch effects over
    :param str score:
        The column to evaluate over
    :returns:
        A new data frame with the batch effect removed
    """
    if isinstance(category, str):
        category = [category]
    else:
        category = list(category)
    if isinstance(batch, str):
        batch = [batch]
    else:
        batch = list(batch)

    # Figure out how many category levels are in each batch
    batch_levels = {b: np.unique(data[b].values) for b in batch}
    batch_counts = {b: len(l) for b, l in batch_levels.items()
                    if len(l) > 1}
    batch = list(batch_counts.keys())

    data = data.copy()

    if len(batch) < 1:
        return data

    factors = ['C({})'.format(c) for c in category + batch]
    formula = '{} ~ {} - 1'.format(score, ' + '.join(factors))

    print('Got factors: {}'.format(factors))
    print('Fitting model: {}'.format(formula))
    if model == 'ols':
        # This is similar to, but not exactly the same as limma regression
        linear_model = ols(formula, data).fit()
        print(linear_model.summary())

        print(linear_model.params)
        for batch_key in batch:
            # First level offset is 0.0 to start
            batch_offsets = {}
            # Deal with stupid r-like indexing conventions
            for batch_levelno in range(batch_counts[batch_key]):
                batch_levelkey = batch_levels[batch_key][batch_levelno]
                batch_cov = 0.0
                ols_key = None
                for key in linear_model.params.index:
                    # More stupid parameter re-naming stupidity
                    if key.startswith('C({})['.format(batch_key)):
                        if batch_levelkey == key.split('[', 1)[1].strip('T.').rstrip(']'):
                            batch_cov = linear_model.params[key]
                            ols_key = key
                            break
                if ols_key is not None:
                    del linear_model.params[ols_key]
                batch_offsets[batch_levelkey] = batch_cov
            batch_mean = np.mean(list(batch_offsets.values()))
            batch_offsets = {k: v-batch_mean for k, v in batch_offsets.items()}

            print(batch_offsets)

            # Now subtract the final offsets
            data_score = data[score].values
            for batch_levelkey, batch_offset in batch_offsets.items():
                data_mask = (data[batch_key] == batch_levelkey).values
                data_score[data_mask] -= batch_offset
            data.loc[:, score] = data_score
    else:
        raise KeyError('Unknown model "{}"'.format(model))
    return data


def load_train_test_split(test_dir):
    """ Load the train/test split file

    :param Path test_dir:
        The test directory to load, containing a neural net snapshot subdirectory
    :returns:
        A dictionary with the keys "train", "test", and "validation" mapping to the indices
    """
    split_file = test_dir / 'snapshot' / 'train_test_split.json'
    if not split_file.is_file():
        raise OSError('No train/test split available for {}'.format(test_dir))
    with split_file.open('rt') as fp:
        split_data = json.load(fp)

    index_data = {
        'train': [],
        'test': [],
        'validation': [],
    }
    for key in (k + '_files' for k in index_data.keys()):
        for filename in split_data.get(key, []):
            filename = pathlib.Path(filename)
            match = reTRAINFILE.match(filename.name)
            if not match:
                continue
            index_data[key.split('_', 1)[0]].append(int(match.group('index')))
    return index_data


def find_all_train_files(train_dir, suffix=''):
    """ Find all the training files under a training directory

    :param Path train_dir:
        The training directory to load
    :param str suffix:
        A suffix that trianing files need to match
    :returns:
        A dictionary mapping train_index: train_path
    """

    train_files = {}
    for p in train_dir.iterdir():
        if not p.name.endswith(suffix) or not p.is_file():
            continue
        match = reTRAINFILE.match(p.name)
        if not match:
            continue
        train_index = int(match.group('index'))
        assert train_index not in train_files
        train_files[train_index] = p
    if len(train_files) < 1:
        raise OSError('No data with suffix "{}" under data dir: {}'.format(suffix, train_dir))
    return train_files


def pair_train_test_data(train_data, test_data, datatype='real', mode='points', data_split='all'):
    """ Pair off the train and test images

    :param Path train_data:
        The path to the training data directory
    :param Path test_data:
        The path to the test data file or directory
    :param str datatype:
        'real' or 'test', which kind of data to load
    :param str mode:
        One of 'points' or 'masks', which output to load
    :param str data_split:
        One of 'train', 'test', 'validation', or 'all', which data split to load
    :returns:
        A list of paired train/test datasets
    """

    # Convert the mode and datatype to file suffix
    # mode, datatype: train_suffix, test_suffix
    train_suffix, test_suffix = {
        ('points', 'test'): ('dots.png', 'cell.csv'),
        ('points', 'real'): ('dots.png', '.csv'),
        ('masks', 'test'): ('dots.png', 'cell_resp.png'),
        ('masks', 'real'): ('dots.png', '_resp.png'),
    }[(mode, datatype)]

    train_files = find_all_train_files(train_data, suffix=train_suffix)
    for split in ['train', 'test', 'validation']:
        if data_split == split:
            train_files = {i: p for i, p in train_files.items()
                           if i in load_train_test_split(test_data)[split]}
            break
    else:
        if data_split not in ('all', None):
            raise KeyError('Unknown data split type: {}'.format(data_split))

    if datatype == 'real':
        index_file = train_data / 'index.xlsx'
        if not index_file.is_file():
            raise OSError('No index under train data dir: {}'.format(train_data))
        cell_index = load_index(index_file)
        test_files = index_all_test_files(test_data, cell_index, suffix=test_suffix)
    elif datatype == 'test':
        test_files = find_all_train_files(test_data, suffix=test_suffix)
    else:
        raise ValueError('Unknown datatype "{}"'.format(datatype))

    pairs = []

    for key in sorted(train_files):
        if key in test_files:
            pairs.append((train_files[key], test_files[key]))
    return pairs


def index_all_test_files(test_data, cell_index, detector=None, suffix=''):
    """ Use the index to assign cell numbers to the test data

    :param Path test_data:
        The points generated by the algorithm
    :param dict cell_index:
        The mapping from cell number to experiment/tile/timepoint
    :returns:
        A dictionary mapping cell number to test data
    """
    test_files = {}
    experiment_dirs = [test_data / index.experiment for index in cell_index]

    for experiment_dir in experiment_dirs:
        if detector is None:
            detector_dir = experiment_dir / 'SingleCell'
        else:
            detector_dir = experiment_dir / 'SingleCell-{}'.format(detector)
        if not detector_dir.is_dir():
            continue
        channel_dir = guess_channel_dir(detector_dir / 'Corrected', 'gfp')[1]
        for _, tiledir in find_tiledirs(channel_dir):
            for imagefile in tiledir.iterdir():
                if not imagefile.name.endswith(suffix):
                    continue
                imagedata = parse_image_name(imagefile.name)
                key = CellIndex(experiment_dir.name, imagedata['tile'], imagedata['timepoint'], 0, 'none')
                if key in cell_index:
                    test_files[cell_index[key]] = imagefile
    return test_files


def load_index(index_file):
    """ Load the mapping from cell number to experiment, tile, timepoint

    :param Path index_file:
        The index file mapping cell number to experiment/tile/timepoint

    :returns:
        A dictionary of experiment, tile, timepoint: cell number
    """

    cell_index = {}

    index_data = pd.read_excel(index_file)

    for i, cell_number in enumerate(index_data['cell_number']):
        row = index_data.iloc[i]

        experiment = str(row['experiment'])
        tile = int(row['tile'])
        timepoint = int(row['timepoint'])
        rot90 = int(row['rot90'])
        flip = str(row['flip'])

        key = CellIndex(experiment, tile, timepoint, rot90, flip)
        if key in cell_index:
            raise KeyError('Duplicate index: {} for cell number {}'.format(key, int(cell_number)))
        cell_index[key] = int(cell_number)
    return cell_index


def load_points_from_maskfile(maskfile, cutoff=0.5, keep_scale=False, min_distance=2, exclude_border=0):
    """ Load the points from a mask file

    :param Path maskfile:
        Path to the mask file to load
    :param float cutoff:
        The minimum value a peak can have
    :param bool keep_scale:
        If True, keep the image scale, otherwise return normalized coordinates
    :param int min_distance:
        Minimum distance between peaks
    :param int exclude_border:
        How many pixels around the edge to exclude
    :returns:
        The x, y coordinates for that image
    """

    img = load_image(maskfile)/255

    rows, cols = img.shape
    if keep_scale:
        xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    else:
        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, cols),
                             np.linspace(1.0, 0.0, rows))
    mask = img >= cutoff
    peak_mask = peak_local_max(img + np.random.ranf(img.shape)*1e-5,
                               min_distance=min_distance,
                               indices=False,
                               exclude_border=exclude_border,
                               labels=mask)
    return xx[peak_mask], yy[peak_mask], img[peak_mask]


def score_points(train_points, test_points, max_distance=MAX_DISTANCE,
                 return_irr=False):
    """ Score the train and test points

    :param tuple[ndarray] train_points:
        The x, y reference coordinates
    :param tuple[ndarray] test_points:
        The x, y, v test points with confidence scores
    :param float max_distance:
        The max allowed distance between correspondences
    :param bool return_irr:
        If True, only return the match and total scores, otherwise return the masks
    :returns:
        The score profile for the points
    """
    # FIXME: Is this entire function now redundant?
    scores = PointScore(train_points, test_points, max_distance=max_distance)
    scores.score_points()
    # FIXME: This is dumb
    if return_irr:
        return scores.irr, scores.num_total
    return scores.y_real, scores.y_score


def calc_effect_size(mean1: float, mean2: float,
                     std1: float, std2: float,
                     n1: int, n2: int) -> float:
    """ Calculate the effect size because all the stats are significant

    :param float mean1:
        The mean of the first distribution
    :param float mean2:
        The mean of the second distribution
    :param float std1:
        The std of the first distribution
    :param float std2:
        The std of the second distribution
    :param int n1:
        The number of samples in the first distribution
    :param int n2:
        The number of samples in the second distribution
    :returns:
        Cohen's d, where 0.01 is a small effect, 0.5 is a medium effect, etc
    """
    s1 = (n1 - 1) * std1**2
    s2 = (n2 - 1) * std2**2
    s = np.sqrt((s1 + s2)/(n1 + n2))  # ML estimator by Hedges and Olkin
    return np.abs((mean1 - mean2)/s)
