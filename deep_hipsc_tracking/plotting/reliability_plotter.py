""" Plotting class for IRR/IFR

Plots the output of :py:func:`~agg_dyn.stats.utils.score_points`

Classes:

* :py:class:`ReliabilityPlotter`: Plot IRR and IFR barplots

Functions:

* :py:class:`plot_roc_curve`: Make ROC curve plots
* :py:class:`plot_pr_curve`: Make P/R curve plots

"""

# Imports
import json

# 3rd party
import numpy as np
import matplotlib.pyplot as plt

# Our own imports
from .consts import PLOT_STYLE, BARPLOT_ORIENT, SUFFIX, PALETTE, FIGSIZE, LINEWIDTH
from .styling import set_plot_style, colorwheel
from .cat_plot import add_barplot
from .split_axes import SplitAxes

from ..utils import calc_pairwise_significance

# Classes


class ReliabilityPlotter(object):
    """ Plot reliability for different scores

    :param DataFrame score_data:
        The score data frame with multiple condition columns
    :param Path outdir:
        The output directory to write the plots to
    :param str score_type:
        One of "irr" or "ifr"
    """

    def __init__(self, score_data, outdir, score_type,
                 plot_style=PLOT_STYLE,
                 barplot_orient=BARPLOT_ORIENT,
                 suffix=SUFFIX,
                 palette=PALETTE,
                 for_paper=False,
                 annotator_column='Annotator',
                 figsize=FIGSIZE):
        self.score_data = score_data
        self.outdir = outdir
        self.score_type = score_type.lower()

        self.for_paper = for_paper
        if self.for_paper:
            plot_style = 'light'
            barplot_orient = 'vertical'
            suffix = '.svg'
            palette = 'Set1'

        # Data frame column names
        self.annotator_column = annotator_column
        self.pct_column = 'PercentLabeled'
        self.tile_column = 'Tile'

        self.ylabel = {
            'irr': '% Matches to Consensus',
            'ifr': '% Matches Between Frames',
        }[self.score_type]
        self.ycolumn = {
            'irr': 'IRR',
            'ifr': 'IFR',
        }[self.score_type]

        # Make sure we're plotting percents and check bounds
        if np.max(self.score_data[self.ycolumn]) <= 1.0:
            # Convert to percent
            self.score_data[self.ycolumn] *= 100
        score_min = np.min(self.score_data[self.ycolumn])
        score_max = np.max(self.score_data[self.ycolumn])
        if score_min < 0 or score_max > 100:
            err = 'Score {} should be between 0 and 100, got {} to {}'
            err = err.format(self.ycolumn, score_min, score_max)
            raise ValueError(err)

        self.xlimits = None
        self.ylimits = [(0, 5), (65, 100)]

        # Styling
        self.plot_style = plot_style
        self.barplot_orient = barplot_orient
        self.suffix = suffix
        self.palette = palette

        self.fig_height = 8  # in - height of the figure
        self.bar_width = 1.25  # in - width of individual bars
        self.bar_padding = 1  # in - width of the left padding to add to the figure size

        # Annotator plots
        self.annotator_label = {
            'Annotator': 'Rater',
            'Detector': '',
        }[self.annotator_column]

        # Percent plots
        self.pct_label = 'Percent Labeled'
        self.pct_order = ['10', '30', '100']
        self.pct_xticklabels = [f'{pct}%' for pct in self.pct_order]

    def calc_figsize(self, score_data, xcolumn, hue_column=None):
        """ Work out how big of a figure to generate

        :returns:
            A tuple of (figure width, figure height) in inches
        """

        categories = np.unique(score_data[xcolumn])
        num_categories = len(categories)

        if num_categories < 2:
            err = 'Got {} unique categories for column {}: {}'
            err = err.format(num_categories, xcolumn, categories)
            raise ValueError(err)

        if hue_column is None:
            num_hue_categories = 1
        else:
            hue_categories = np.unique(score_data[hue_column])
            num_hue_categories = len(hue_categories)
            if num_hue_categories < 2:
                err = 'Got {} unique categories for column {}: {}'
                err = err.format(num_hue_categories, hue_column, hue_categories)
                raise ValueError(err)

        num_bars = num_categories * num_hue_categories
        return (num_bars * self.bar_width + self.bar_padding, self.fig_height)

    def swap_limits(self, limits, orient=None):
        """ Swap the order of the tuple in limits when the orientation is reversed

        :param tuple limits:
            A tuple with elements corresponding to the order when vertical
        :returns:
            The elements swapped when horizontal
        """
        if limits is None:
            return None
        if orient is None:
            orient = self.barplot_orient
        if orient.startswith('h'):
            limits = tuple(reversed(limits))
        return limits

    def subset_score_data(self, subset=None, xcolumn=None, order=None):
        """ Subset the data

        :param dict subset:
            If not None, a dictionary of column: values to subset the dataframe by
        :returns:
            A new dataframe with only rows where (column in values)
        """
        if subset in (None, {}):
            return self.score_data, order

        score_data = self.score_data
        print('Size before filtering: {} rows'.format(score_data.shape[0]))
        for key, values in subset.items():
            mask = np.zeros((score_data.shape[0], ), dtype=bool)
            if isinstance(values, (str, int, float)):
                values = [values]

            # Only keep values matching the selected categories
            column_data = score_data[key]
            for value in values:
                mask = np.logical_or(mask, column_data == value)
            score_data = score_data[mask]
            print('After filtering "{}": {} rows'.format(key, score_data.shape[0]))

            # Mask the order as well if we're filtering the relevant category
            if order is not None and xcolumn is not None and key == xcolumn:
                order = [o for o in order if o in values]
                print('After filtering order for "{}": {}'.format(key, order))
        # Final score filter
        print('Size after filtering: {} rows'.format(score_data.shape[0]))
        return score_data, order

    def plot_comparison(self, xcolumn, xlabel,
                        subset=None, prefix='', order=None,
                        hue=None, hue_order=None,
                        barplot_orient=None,
                        xticklabels=None,
                        pathname=None,
                        sig_category=None):
        """ Plot an arbitrary comparison between different samples

        :param str xcolumn:
            Name of the column to use for the categorical data
        :param str xlabel:
            Label for the categorical data
        """

        if barplot_orient is None:
            barplot_orient = self.barplot_orient
        if pathname is None:
            pathname = xcolumn.lower()
        if sig_category is None:
            sig_category = xcolumn

        if subset in (None, {}):
            print('Scoring {} vs {}...'.format(xcolumn, self.ycolumn))
        else:
            print('Scoring filtered {} vs {}...'.format(xcolumn, self.ycolumn))

        # Filter the data and calculate a figure size
        score_data, order = self.subset_score_data(subset=subset,
                                                   xcolumn=xcolumn,
                                                   order=order)
        figsize = self.calc_figsize(score_data, xcolumn=xcolumn, hue_column=hue)

        # Swap all the axes when horizontal
        figsize = self.swap_limits(figsize, orient=barplot_orient)
        xlimits, ylimits = self.swap_limits((self.xlimits, self.ylimits),
                                            orient=barplot_orient)
        order = self.swap_limits(order, orient=barplot_orient)
        xticklabels = self.swap_limits(xticklabels, orient=barplot_orient)

        print('Scoring {} vs {}...'.format(xcolumn, self.ycolumn))
        significance = calc_pairwise_significance(score_data,
                                                  category=sig_category,
                                                  score=self.ycolumn)

        with set_plot_style(self.plot_style) as style:
            outfile = '{}_vs_{}{}'.format(self.score_type, pathname, self.suffix)
            if prefix not in ('', None):
                outfile = prefix + '_' + outfile
            outfile = self.outdir / outfile

            with SplitAxes(figsize=figsize, xlimits=xlimits, ylimits=ylimits) as ax:
                add_barplot(y=self.ycolumn, x=xcolumn, order=order,
                            data=score_data, ax=ax, significance=significance,
                            orient=barplot_orient, palette=self.palette,
                            xlabel=xlabel, xticklabels=xticklabels, ylabel=self.ylabel,
                            hue=hue, hue_order=hue_order,
                            savefile=outfile)
            style.show(outfile, transparent=True)

    def plot_annotator(self, subset=None, prefix='', hue=None, hue_order=None, pathname=None, order=None, sig_category=None):
        """ Make an annotator plot """

        self.plot_comparison(xcolumn=self.annotator_column,
                             xlabel=self.annotator_label,
                             subset=subset,
                             prefix=prefix,
                             order=order,
                             hue=hue,
                             hue_order=hue_order,
                             pathname=pathname,
                             sig_category=sig_category)

    def plot_percent(self, subset=None, prefix='', hue=None, hue_order=None, pathname=None, order=None, sig_category=None):
        """ Make a percent labeled plot """

        if self.for_paper:
            barplot_orient = 'horizontal'
        else:
            barplot_orient = self.barplot_orient

        if order is None:
            order = self.pct_order

        if pathname is None:
            pathname = 'pct'

        self.plot_comparison(xcolumn=self.pct_column,
                             xlabel=self.pct_label,
                             subset=subset,
                             prefix=prefix,
                             hue=hue,
                             hue_order=hue_order,
                             barplot_orient=barplot_orient,
                             order=order,
                             xticklabels=self.pct_xticklabels,
                             pathname=pathname,
                             sig_category=sig_category)

    def plot_tiles(self, subset=None, prefix='', hue=None, hue_order=None, pathname=None, order=None, sig_category=None):
        """ Make a plot for the individual tiles """

        self.plot_comparison(xcolumn=self.tile_column,
                             xlabel=self.tile_column,
                             subset=subset,
                             prefix=prefix,
                             order=order,
                             hue=hue,
                             hue_order=hue_order,
                             pathname=pathname,
                             sig_category=sig_category)


# Functions


def plot_roc_curve(roc_lines, xlines=None, linewidth=LINEWIDTH, outfile=None,
                   plot_style=PLOT_STYLE, figsize=FIGSIZE, palette=PALETTE,
                   savefile=None):
    """ Plot the ROC curves for sets of train/test pairs

    :param list[ROCData] roc_lines:
        The list of ROC data objects to plot
    :param list[tuple] xlines:
        List of (yvalue, label) lines
    :param float linewidth:
        Width of the lines to plot
    :param Path outfile:
        If not None, the path to save the ROC curve to
    :param str plot_style:
        The style to use for the plots
    :param tuple[float] figsize:
        The size in inches of the figure
    :param Path savefile:
        If not None, the JSON file to write containing all the plot data
    """

    if xlines is None:
        xlines = []
    savedata = []

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        wheel = colorwheel(palette, n_colors=6)
        for roc_line, color in zip(sorted(roc_lines, key=lambda x: x.label), wheel):
            roc_line.plot_roc(ax=ax, color=color, linewidth=linewidth, annotate_knee=False)
            if savefile is not None:
                savedata.append(roc_line.to_plotdata())
        ax.plot([0, 1], [0, 1], color='navy', linewidth=linewidth, linestyle='--',
                label=None)
        for (xvalue, xlabel), color in zip(xlines, colorwheel()):
            ax.plot([0, 1], [xvalue, xvalue], color=color, linewidth=linewidth,
                    linestyle='--', label=xlabel)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")

        style.show(outfile, transparent=True)

    # Cache the final output data
    if savefile is not None:
        savefile = savefile.parent / (savefile.stem + '.json')
        savefile.parent.mkdir(parents=True, exist_ok=True)
        print('Saving plot data to {}'.format(savefile))
        with savefile.open('wt') as fp:
            for rec in savedata:
                fp.write(json.dumps(rec) + '\n')


def plot_pr_curve(pr_lines, xlines=None, linewidth=LINEWIDTH, outfile=None,
                  plot_style=PLOT_STYLE, figsize=FIGSIZE, palette=PALETTE,
                  savefile=None):
    """ Plot the P/R curves for sets of train/test pairs

    :param list[ROCData] pr_lines:
        The list of ROC data objects to plot
    :param list[tuple] xlines:
        List of (yvalue, label) lines
    :param float linewidth:
        Width of the lines to plot
    :param Path outfile:
        If not None, the path to save the P/R curve to
    :param str plot_style:
        The style to use for the plots
    :param tuple[float] figsize:
        The size in inches of the figure
    """

    if xlines is None:
        xlines = []
    savedata = []

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        wheel = colorwheel(palette, n_colors=6)

        for pr_line, color in zip(sorted(pr_lines, key=lambda x: x.label), wheel):
            pr_line.plot_precision_recall(ax=ax, color=color, linewidth=linewidth, annotate_knee=False)
            if savefile is not None:
                savedata.append(pr_line.to_plotdata())
        for (xvalue, xlabel), color in zip(xlines, colorwheel()):
            ax.plot([0, 1], [xvalue, xvalue], color=color, linewidth=linewidth,
                    linestyle='--', label=xlabel)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc="lower left")

        style.show(outfile, transparent=True)

    # Cache the final output data
    if savefile is not None:
        savefile = savefile.parent / (savefile.stem + '.json')
        savefile.parent.mkdir(parents=True, exist_ok=True)
        print('Saving plot data to {}'.format(savefile))
        with savefile.open('wt') as fp:
            for rec in savedata:
                fp.write(json.dumps(rec) + '\n')
