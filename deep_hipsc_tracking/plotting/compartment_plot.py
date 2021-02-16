""" Plot data split by compartments

Classes:

* :py:class:`CompartmentPlot`: compartment plotting tool

"""

# Standard lib
from typing import Tuple, Optional, Dict

# 3rd party imports
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

# Our own imports
from .styling import set_plot_style
from .utils import bootstrap_ci, get_histogram

# Classes


class CompartmentPlot(object):
    """ Plot data split by multiple compartments

    :param int n_compartments:
        How many compartments to split the data into
    :param int topk:
        How many samples to take from each compartment
    """

    def __init__(self,
                 n_compartments: int,
                 topk: Optional[int] = None,
                 figsize: Tuple[int] = (8, 8),
                 plot_style: str = 'dark',
                 suffix: str = '.png'):
        self.n_compartments = n_compartments
        self.topk = topk

        self.figsize = figsize
        self.plot_style = plot_style
        self.suffix = suffix

        # Color palettes for the different compartments
        self.colors = (['blue', 'orange', 'green', 'red', 'purple', 'grey'])[:n_compartments]
        self.palletes = [sns.color_palette(c.capitalize()+'s', n_colors=10)
                         for c in self.colors]

        # Calculated values
        self._bin_indices = None
        self._bin_values = None
        self._xdata = None
        self._xcolumn = None
        self._ycolumn = None
        self._plotdata = None
        self._distdata = None
        self._total_count = None

    def calc_indices(self, values: np.ndarray):
        """ Calculate the indicies for each bin

        :param ndarray values:
            The values to use to generate the bins
        """
        if self.topk is None:
            self.topk = values.shape[0] // self.n_compartments

        if values.shape[0] < self.topk * self.n_compartments:
            err = 'Got too few values for {} samples of {} compartments: {}'
            err = err.format(self.topk, self.n_compartments, values.shape[0])
            raise ValueError(err)
        print(f'Spliting into {self.n_compartments} compartments of {self.topk} samples each')

        # Sort all the indices
        indices = np.argsort(values)

        # Split into even bins of size topk
        bin_start = np.floor(np.linspace(0, indices.shape[0]-self.topk, self.n_compartments))
        bin_start[bin_start < 0] = 0
        bin_end = bin_start + self.topk
        bin_end[bin_end > indices.shape[0]] = indices.shape[0]

        # Extract the sorted bins for each compartment
        self._bin_indices = [indices[int(s):int(e)] for s, e in zip(bin_start, bin_end)]

    def calc_bin(self,
                 bin_value: np.ndarray,
                 label: str,
                 total_count: int) -> Dict[str, float]:
        """ Calculate all the stats for a single bin

        :param ndarray bin_value:
            The 2D array of n timepoints x k samples
        :param str label:
            The label for this category
        :param int total_count:
            The total number of samples in this bin
        :returns:
            A dictionary of bin stats for plotting
        """

        bin_mean = np.nanmean(bin_value, axis=1)
        bin_std = np.nanstd(bin_value, axis=1)

        bin5, bin25, bin50, bin75, bin95 = np.nanpercentile(bin_value, [5, 25, 50, 75, 95], axis=1)
        bin_mean_ci0, bin_mean_ci1 = bootstrap_ci(bin_value, func=np.nanmean, axis=1)
        assert bin_mean_ci0.shape == bin_mean.shape
        assert bin_mean_ci1.shape == bin_mean.shape

        bin_median_ci0, bin_median_ci1 = bootstrap_ci(bin_value, func=np.nanmedian, axis=1)
        assert bin_median_ci0.shape == bin50.shape
        assert bin_median_ci0.shape == bin50.shape

        # Work out how many samples/bin we have in each timepoint
        bin_count = np.sum(~np.isnan(bin_value), axis=1)
        bin_support = bin_count / total_count
        bin_support[~np.isfinite(bin_support)] = 0

        # Stash all the values for later
        return {
            'mean' + label: bin_mean,
            'mean ci low' + label: bin_mean_ci0,
            'mean ci high' + label: bin_mean_ci1,
            'std' + label: bin_std,
            'p5' + label: bin5,
            'p25' + label: bin25,
            'p50' + label: bin50,
            'p50 ci low' + label: bin_median_ci0,
            'p50 ci high' + label: bin_median_ci1,
            'p75' + label: bin75,
            'p95' + label: bin95,
            'count' + label: bin_count,
            'support' + label: bin_support,
        }

    def split_comparison(self,
                         data: Dict[str, np.ndarray],
                         xcolumn: str,
                         ycolumn: str,
                         integrate_values: bool = False):
        """ Split the comparison by the bins

        :param dict[str, Any] data:
            A dictionary containing the xcolumn and ycolumn data
        :param str xcolumn:
            The column containing the shared time vector to plot along
        :param str ycolumn:
            The column containing the values to bin along
        :param bool integrate_values:
            If True, integrate the resulting statistics over the xdata range
        """
        xdata = data[xcolumn]
        plotdata = {
            xcolumn: xdata,
        }
        values = np.stack(data[ycolumn], axis=1)
        total_count = np.sum(~np.isnan(values), axis=1)

        if values.shape[0] != xdata.shape[0]:
            raise ValueError('Expected {} with shape {}, got {}'.format(ycolumn, xdata.shape[0], values.shape[0]))

        bin_values = []

        # Add a set for all the values
        plotdata.update(self.calc_bin(values, f' {ycolumn} all', total_count))
        for i, indices in enumerate(self._bin_indices):
            bin_value = values[:, indices]
            bin_values.append(bin_value)

            label = f' {ycolumn} bin{i+1}'
            plotdata.update(self.calc_bin(bin_value, label, total_count))

        self._plotdata = plotdata
        self._xdata = xdata
        self._xcolumn = xcolumn
        self._ycolumn = ycolumn
        self._bin_values = bin_values
        self._total_count = total_count

    def calc_envelope(self, label: str, envelope: str = 'std') -> Tuple[float]:
        """ Calculate the envelope (high/low) stats for a label

        :param str label:
            The label to calculate the envelope for
        :param str envelope:
            Which stats to calculate the envelope with
        :returns:
            A tuple of low, high values
        """
        plotdata = self._plotdata

        if envelope == 'std':
            value_mean = plotdata['mean' + label]
            value_std = plotdata['std' + label]
            value_st = value_mean - value_std
            value_ed = value_mean + value_std
        elif envelope == 'mean ci':
            value_st = plotdata['mean ci low' + label]
            value_ed = plotdata['mean ci high' + label]
        elif envelope == 'median ci':
            value_st = plotdata['p50 ci low' + label]
            value_ed = plotdata['p50 ci high' + label]
        elif envelope == 'iqr':
            value_st = plotdata['p25' + label]
            value_ed = plotdata['p75' + label]
        else:
            raise ValueError('Unknown envelope function "{}"'.format(envelope))
        return value_st, value_ed

    def plot_raw_tracks(self, outfile=None, xlabel=None, ylabel=None):
        """ Plot individual raw tracks """

        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            for i, bin_value in enumerate(self._bin_values):
                ax.set_prop_cycle(color=self.palletes[i])
                ax.plot(self._xdata, bin_value, '-')
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            style.show(outfile=outfile, fig=fig)

    def plot_mean_tracks(self, outfile=None, xlabel=None, ylabel=None, envelope='std', mode='split'):
        """ Mean and deviation envelope

        :param Path outfile:
            If not None, the file to write out
        :param str xlabel:
            Label for the x-axis (time)
        :param str ylabel:
            Label for the y-axis (category)
        """
        plotdata = self._plotdata

        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            if mode == 'split':
                for i in range(self.n_compartments):
                    label = ' {} bin{}'.format(self._ycolumn, i+1)
                    value_mean = plotdata['mean' + label]
                    value_st, value_ed = self.calc_envelope(label, envelope)
                    ax.fill_between(self._xdata, value_st, value_ed,
                                    facecolor=self.colors[i], alpha=0.5)
                    ax.plot(self._xdata, value_mean, '-', color=self.colors[i], linewidth=2)
            elif mode == 'all':
                label = ' {} all'.format(self._ycolumn)
                value_mean = plotdata['mean' + label]
                value_st, value_ed = self.calc_envelope(label, envelope)
                ax.fill_between(self._xdata, value_st, value_ed,
                                facecolor='b', alpha=0.5)
                ax.plot(self._xdata, value_mean, '-', color='b', linewidth=2)
            else:
                raise ValueError('Unknown mode {}'.format(mode))
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            style.show(outfile=outfile, fig=fig)

    def plot_median_tracks(self, outfile=None, xlabel=None, ylabel=None, envelope='iqr', mode='split'):
        """ Median and 25/75% envelope """

        plotdata = self._plotdata

        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            if mode == 'split':
                for i in range(self.n_compartments):
                    label = ' {} bin{}'.format(self._ycolumn, i+1)
                    value_mid = plotdata['p50' + label]
                    value_st, value_ed = self.calc_envelope(label, envelope)
                    ax.fill_between(self._xdata, value_st, value_ed,
                                    facecolor=self.colors[i], alpha=0.5)
                    ax.plot(self._xdata, value_mid, '-', color=self.colors[i], linewidth=2)
            elif mode == 'all':
                label = ' {} all'.format(self._ycolumn)
                value_mean = plotdata['p50' + label]
                value_st, value_ed = self.calc_envelope(label, envelope)
                ax.fill_between(self._xdata, value_st, value_ed,
                                facecolor='b', alpha=0.5)
                ax.plot(self._xdata, value_mean, '-', color='b', linewidth=2)
            else:
                raise ValueError('Unknown mode {}'.format(mode))

            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            style.show(outfile=outfile, fig=fig)

    def plot_track_support(self, outfile=None, xlabel=None, ylabel=None):
        """ Plot how many tracks are in a given bin at a given time """
        plotdata = self._plotdata

        with set_plot_style(self.plot_style) as style:
            fig_x, fig_y = self.figsize
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_x*2, fig_y))
            ax1.plot(self._xdata, self._total_count, '-k', linewidth=2)
            ax2.hlines([100], np.min(self._xdata), np.max(self._xdata), colors=['k'], linewidth=2)

            for i in range(self.n_compartments):
                label = ' {} bin{}'.format(self._ycolumn, i+1)
                count = plotdata['count' + label]
                support = plotdata['support' + label]
                ax1.plot(self._xdata, count, '-', color=self.colors[i], linewidth=2)
                ax2.plot(self._xdata, support*100, '-', color=self.colors[i], linewidth=2)
            if xlabel is not None:
                ax1.set_xlabel(xlabel)
                ax2.set_xlabel(xlabel)
            ax1.set_ylabel('Num Tracks')
            ax2.set_ylabel('Percent Total Tracks')
            ax1.set_ylim([0, np.max(self._total_count)*1.02])
            ax2.set_ylim([0, 102])
            style.show(outfile=outfile, fig=fig)

    def plot_dist_histogram(self, values, outfile=None, xlabel=None, ylabel=None):
        """ Plot where on the histogram each value occurs

        :param ndarray values:
            The values to generate a histogram for
        :param Path outfile:
            If not None, the path to save the plot to
        """
        # Histogram the distribution and which compartments are being labeled
        _, _, kernel_x, kernel_y = get_histogram(values, bins=10, kernel_smoothing=True)
        compartment_values = [values[indices] for indices in self._bin_indices]

        distdata = {
            'compartment': [],
            'value': [],
            'density': [],
        }

        # Now, plot each compartment on the total histogram
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.plot(kernel_x, kernel_y, '-', color='gray')
            distdata['compartment'].extend(0 for _ in kernel_x)
            distdata['value'].extend(kernel_x)
            distdata['density'].extend(kernel_y)
            for i, compartment_value in enumerate(compartment_values):
                compartment_min = np.min(compartment_value)
                compartment_max = np.max(compartment_value)
                kernel_mask = np.logical_and(kernel_x >= compartment_min,
                                             kernel_x <= compartment_max)

                compartment_x = kernel_x[kernel_mask]
                compartment_y = kernel_y[kernel_mask]
                distdata['compartment'].extend(i+1 for _ in compartment_x)
                distdata['value'].extend(compartment_x)
                distdata['density'].extend(compartment_y)

                ax.fill_between(compartment_x, 0, compartment_y,
                                facecolor=self.colors[i], alpha=0.5)
                ax.plot(compartment_x, compartment_y, '-',
                        color=self.colors[i], linewidth=2)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            style.show(outfile=outfile, fig=fig)
        self._distdata = distdata

    def save_plotdata(self, outfile, suffix='.csv'):
        """ Save the plot data """
        if self._plotdata is None:
            raise ValueError('No distribution data, call split_comparison first')

        outfile = outfile.parent / (outfile.stem + suffix)
        print('Writing distribution data to {}'.format(outfile))
        plotdata = pd.DataFrame(self._plotdata)

        if suffix == '.csv':
            plotdata.to_csv(str(outfile), header=True, index=False)
        elif suffix == '.xlsx':
            plotdata.to_excel(str(outfile), header=True, index=False)
        else:
            raise KeyError('Unknown plot data output file type: {}'.format(outfile))

    def save_distdata(self, outfile, suffix='.csv'):
        """ Save the distribution data """
        if self._distdata is None:
            raise ValueError('No distribution data, call plot_dist_histogram first')

        outfile = outfile.parent / (outfile.stem + suffix)
        print('Writing distribution data to {}'.format(outfile))
        distdata = pd.DataFrame(self._distdata)

        if suffix == '.csv':
            distdata.to_csv(str(outfile), header=True, index=False)
        elif suffix == '.xlsx':
            distdata.to_excel(str(outfile), header=True, index=False)
        else:
            raise KeyError('Unknown dist data output file type: {}'.format(outfile))
