""" Plots for categorical data frames

Categorical plots:

* :py:class:`CatPlot`: Different kinds of categorical plots with tests

Compound plotting functions:

* :py:func:`~add_barplot`: Add a barplot with significance test marks
* :py:func:`~add_boxplot`: Add a boxplot with significance test marks
* :py:func:`~add_lineplot`: Add a line plot with confidence intervals
* :py:func:`~add_violins_with_outliers`: Make violinplots with outliers

"""

# Standard lib
import itertools
import pathlib
from typing import Tuple, List, Union, Optional, Dict

# 3rd party imports
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pandas as pd

import seaborn as sns

# Our own imports
from .consts import COLOR_PALETTE
from .styling import set_plot_style, colorwheel
from .utils import bootstrap_ci, get_histogram

# Types

CategoryKeyType = Union[str, Tuple[str]]  # For category or hue keys

# Classes


class CatPlot(object):
    """ Categorical plots

    :param DataFrame data:
        The data frame where each column contains either a category or a value
    :param str xcolumn:
        The name of the categorical data column
    :param str ycolumn:
        The name of the score data column
    :param str hue_column:
        A secondary category column to group over
    :param str sample_column:
        A per-batch category column to group over
    :param list order:
        Order for the xcolumn categories
    :param list hue_order:
        Order for the hue_column categories
    :param Axes ax:
        Axes to draw the plots on
    :param str palette:
        The palette to use to draw with
    :param str orient:
        Orientation for the plot ('vertical' or 'horizontal')
    """

    def __init__(self,
                 data: pd.DataFrame,
                 xcolumn: str,
                 ycolumn: str,
                 hue_column: Optional[str] = None,
                 sample_column: Optional[str] = None,
                 order: Optional[List] = None,
                 hue_order: Optional[List] = None,
                 ax: Optional[Axes] = None,
                 palette: str = 'deep',
                 orient: str = 'v'):

        # Actual pandas DataFrame
        self.data = data

        # Which columns have interesting contrasts in them
        self.xcolumn = xcolumn
        self.ycolumn = ycolumn
        self.hue_column = hue_column
        self.sample_column = sample_column

        # Orientation for the plots
        self.orient = orient.lower()[0]
        if self.orient not in ('v', 'h'):
            raise ValueError('Invalid orientation: choose either [v]ertical or [h]orizontal')

        # Automagically calculated properties
        self._ax = ax

        self._palette_name = palette
        self._palette = None
        self._plot_style = None

        self._order = order
        self._hue_order = hue_order
        self._total_order = None

        # Caches for re-plotting data in other tools
        self._plotdata = None
        self._plotdata_dists = None
        self._plotdata_significance = None

        self.x_ticks = None

    # Properties

    @property
    def ax(self) -> Axes:
        if self._ax is not None:
            return self._ax
        self._ax = plt.gca()
        return self._ax

    @property
    def plot_style(self) -> str:
        if self._plot_style is not None:
            return self._plot_style
        self._plot_style = set_plot_style.get_active_style()
        return self._plot_style

    @property
    def order(self) -> List:
        """ Store the order for each category """
        if self._order is not None:
            return self._order
        self._order = np.sort(np.unique(self.data[self.xcolumn]))
        return self._order

    @property
    def hue_order(self) -> List:
        """ Store the order for each hue """
        if self.hue_column is None:
            return None
        if self._hue_order is not None:
            return self._hue_order
        self._hue_order = np.sort(np.unique(self.data[self.hue_column]))
        return self._hue_order

    @property
    def total_order(self) -> List:
        """ Ordered pairs of category, hue for each value """
        if self._total_order is not None:
            return self._total_order
        if self.hue_column is None:
            self._total_order = list(self.order)
        else:
            self._total_order = list(itertools.product(self.order, self.hue_order))
        return self._total_order

    @property
    def palette(self) -> colorwheel:
        """ Palette object cache """
        if self._palette is not None:
            return self._palette
        if self.hue_order is not None:
            n_colors = len(self.hue_order)
        else:
            n_colors = len(self.order)
        self._palette = colorwheel(self._palette_name, n_colors=n_colors)
        return self._palette

    # Helper functions

    def _save_dataframe(self, df: pd.DataFrame, outfile: pathlib.Path):
        # Save off a DataFrame
        suffix = outfile.suffix
        if suffix == '.csv':
            df.to_csv(str(outfile), header=True, index=False)
        elif suffix == '.xlsx':
            df.to_excel(str(outfile), header=True, index=False)
        else:
            raise KeyError('Unknown plotdata output file type: {}'.format(outfile))

    def _calc_extrema(self, ydata: np.ndarray, how: str = 'mean') -> Tuple[float, float]:
        ydata = ydata[~np.isnan(ydata)]
        if ydata.shape[0] < 3:
            return np.nan, np.nan
        if how == 'mean':
            ycenter = np.mean(ydata)
            yextent = np.std(ydata)
        elif how == 'median':
            [pct25, ycenter, pct75] = np.percentile(ydata, [25, 50, 75])
            yextent = pct75 - pct25
        else:
            raise KeyError('Unknown extreme filter method "{}", choose either mean or median'.format(how))
        return ycenter, yextent

    def _extract_points(self, index: int, key: CategoryKeyType) -> Tuple[np.ndarray, str]:
        # Select the points that contributed to this column
        if self.hue_column is None:
            mask = self.data[self.xcolumn] == key
            color = np.array(self.palette[index])
        else:
            category_key, hue_key = key
            mask = np.logical_and(
                self.data[self.xcolumn] == category_key,
                self.data[self.hue_column] == hue_key,
            )
            color = np.array(self.palette[index % len(self.hue_order)])
        return self.data[self.ycolumn][mask], color

    def _extract_dataframe(self, index: int, key: CategoryKeyType) -> Tuple[pd.DataFrame, str]:
        # Select the points that contributed to this column
        # index: int - The color order index for this key
        # key: str - The string or tuple for the category
        if self.hue_column is None:
            mask = self.data[self.xcolumn] == key
            color = np.array(self.palette[index])
        else:
            category_key, hue_key = key
            mask = np.logical_and(
                self.data[self.xcolumn] == category_key,
                self.data[self.hue_column] == hue_key,
            )
            color = np.array(self.palette[index % len(self.hue_order)])
        return self.data[mask], color

    # Main methods

    def filter_extrema(self, how: str = 'mean',
                       extremes: str = 'both',
                       max_extreme: float = 5.0):
        """ Filter extrema

        :param str how:
            One of 'mean' (+/- standard deviation) or 'median' (+/- IQR)
        :param str extremes:
            Either 'upper', 'lower', or 'both', which end of the extrema to filter
        :param float max_extreme:
            Maximum value to allow (e.g. mean + max_extreme * std)
        """

        extremes = str(extremes).lower()
        if extremes == 'none':
            return

        ydata = self.data[self.ycolumn].values
        ycenter, yextent = self._calc_extrema(ydata, how=how)

        if extremes == 'upper':
            mask = ydata <= ycenter + yextent * max_extreme
        elif extremes == 'lower':
            mask = ydata >= ycenter - yextent * max_extreme
        elif extremes == 'both':
            mask = np.logical_and(ydata <= ycenter + yextent * max_extreme,
                                  ydata >= ycenter - yextent * max_extreme)
        else:
            raise KeyError('Unknown extreme bound "{}", choose none, upper, lower, or both')
        self.data = self.data[mask]

    def extract_plotdata(self, calc_dists: bool = False):
        """ Extract the data for re-plotting in other systems

        :param bool calc_dists:
            If True, calculate the kernel density distributions per category
        """
        def extract_values(cat: CategoryKeyType) -> np.ndarray:
            """ Extract each category as a single value """
            if self.hue_column is None:
                mask = xvals == cat
                num_samples = np.sum(mask)

                plotdata.setdefault(self.xcolumn, []).append(cat)

                if num_samples > 0:
                    plotdata_samples.setdefault(self.xcolumn, []).extend([cat]*num_samples)
            else:
                xcat, hcat = cat
                mask = np.logical_and(xvals == xcat, hvals == hcat)
                num_samples = np.sum(mask)

                plotdata.setdefault(self.xcolumn, []).append(xcat)
                plotdata.setdefault(self.hue_column, []).append(hcat)

                if num_samples > 0:
                    plotdata_samples.setdefault(self.xcolumn, []).extend([xcat]*num_samples)
                    plotdata_samples.setdefault(self.hue_column, []).extend([hcat]*num_samples)
            # Pull out the selected values
            cat_vals = yvals[mask]

            if cat_vals.shape[0] > 0:
                plotdata_samples.setdefault(self.ycolumn, []).extend(cat_vals)
            return cat_vals

        # Convert the plot parameters to a single spreadsheet
        plotdata = {}
        plotdata_dists = {}
        plotdata_samples = {}

        yvals = self.data[self.ycolumn].values
        xvals = self.data[self.xcolumn].values
        if self.hue_column is None:
            hvals = None
        else:
            hvals = self.data[self.hue_column].values

        # Extract summary stats for the values
        for i, key in enumerate(self.total_order):
            # Pull out the selected values
            selvals = extract_values(key)
            if selvals.shape[0] < 1:
                p5 = p25 = p50 = p75 = p95 = np.nan
                mean = std = ci_low = ci_high = np.nan
            elif selvals.shape[0] == 1:
                p50 = mean = selvals[0]
                p5 = p25 = p75 = p95 = np.nan
                std = ci_low = ci_high = np.nan
            else:
                p5, p25, p50, p75, p95 = np.percentile(selvals, [5, 25, 50, 75, 95])
                mean = np.mean(selvals)
                std = np.std(selvals)
                ci_low, ci_high = bootstrap_ci(selvals)

            if calc_dists:
                # Add a kernel density estimate to the plots
                kernel_x, kernel_y = get_histogram(selvals, bins=10)[2:]
                if self.hue_column is None:
                    catname = '{} {}'.format(self.ycolumn, key)
                else:
                    catname = '{} {} {}'.format(self.ycolumn, *key)
                plotdata_dists[catname + ' value'] = kernel_x
                plotdata_dists[catname + ' density'] = kernel_y

            # Stash all the stats away
            plotdata.setdefault('{} p5'.format(self.ycolumn), []).append(p5)
            plotdata.setdefault('{} p25'.format(self.ycolumn), []).append(p25)
            plotdata.setdefault('{} p50'.format(self.ycolumn), []).append(p50)
            plotdata.setdefault('{} p75'.format(self.ycolumn), []).append(p75)
            plotdata.setdefault('{} p95'.format(self.ycolumn), []).append(p95)
            plotdata.setdefault('{} mean'.format(self.ycolumn), []).append(mean)
            plotdata.setdefault('{} std'.format(self.ycolumn), []).append(std)
            plotdata.setdefault('{} ci low'.format(self.ycolumn), []).append(ci_low)
            plotdata.setdefault('{} ci high'.format(self.ycolumn), []).append(ci_high)
            plotdata.setdefault('{} n'.format(self.ycolumn), []).append(selvals.shape[0])

        # Kernel density estimates for violinplot extraction
        if not calc_dists:
            plotdata_dists = None

        self._plotdata = plotdata
        self._plotdata_dists = plotdata_dists
        self._plotdata_samples = plotdata_samples

    def add_violinplot(self, **kwargs):
        """ Plot the categories using violins """

        kwargs.update({
            'ax': self.ax,
            'data': self.data,
            'order': self.order,
            'orient': self.orient,
            'hue': self.hue_column,
            'hue_order': self.hue_order,
            'palette': self.palette,
            'split': False,
        })

        # Have to swap x and y columns for horizontal violins
        if self.orient == 'v':
            sns.violinplot(x=self.xcolumn, y=self.ycolumn, **kwargs)
        else:
            sns.violinplot(x=self.ycolumn, y=self.xcolumn, **kwargs)

        # FIXME: Will this fail if we use colors too?
        if self.orient == 'v':
            new_lines = self.ax.get_xticks()
        else:
            new_lines = self.ax.get_yticks()

        if len(new_lines) != len(self.total_order):
            raise ValueError('Got {} lines but {} items to order: {} vs {}'.format(
                len(new_lines), len(self.total_order), new_lines, self.total_order))
        self.x_ticks = np.sort(np.array(new_lines))

    def add_boxplot(self, showfliers: bool = False, **kwargs):
        """ Plot the categories using boxplots

        :param bool showfliers:
            If True, show the data plotted on the boxplot
        """

        kwargs.update({
            'ax': self.ax,
            'data': self.data,
            'order': self.order,
            'orient': self.orient,
            'hue': self.hue_column,
            'hue_order': self.hue_order,
            'palette': self.palette,
            'showfliers': showfliers,
        })
        old_lines = [line for line in self.ax.lines]

        # Have to swap x and y columns for horizontal boxes
        if self.orient == 'v':
            sns.boxplot(x=self.xcolumn, y=self.ycolumn, **kwargs)
        else:
            sns.boxplot(x=self.ycolumn, y=self.xcolumn, **kwargs)

        # Reverse engineer the positions of the bars
        new_lines = []
        for line in self.ax.lines:
            if line in old_lines:
                continue
            if self.orient == 'v':
                xdata = line.get_xdata()
            else:
                xdata = line.get_ydata()
            # Only look at vertical lines
            if np.all(xdata == xdata[0]):
                if xdata[0] not in new_lines:
                    new_lines.append(xdata[0])

        if len(new_lines) != len(self.total_order):
            raise ValueError('Got {} lines but {} items to order: {} vs {}'.format(
                len(new_lines), len(self.total_order), new_lines, self.total_order))
        self.x_ticks = np.sort(np.array(new_lines))

    def add_barplot(self,
                    capsize: float = 0.15,
                    errwidth: float = 3.0,
                    errcolor: Optional[str] = None,
                    remove_legend: bool = False,
                    **kwargs):
        """ Plot the categories using barplots

        :param float capsize:
            Size of the caps for the barplot
        :param str errcolor:
            The color for the error bars
        """
        # Make sure our order columns are fine
        in_order = set(self.order)
        data_order = set(np.unique(self.data[self.xcolumn]))
        if len(in_order & data_order) == 0:
            raise ValueError(f'No intersection between order categories {in_order} and data categories {data_order}')

        if self.hue_column is not None:
            in_hue_order = set(self.hue_order)
            hue_data_order = set(np.unique(self.data[self.hue_column]))
            if len(in_hue_order & hue_data_order) == 0:
                raise ValueError(f'No intersection between hue order categories {in_hue_order} and hue data categories {hue_data_order}')

        if errcolor is None:
            errcolor = {
                'dark': '#FFFFFF',
                'dark_poster': '#FFFFFF',
            }.get(self.plot_style, '#000000')

        # Stash the old lines (if any)
        old_patches = [line for line in self.ax.patches]

        kwargs.update({
            'hue': self.hue_column,
            'data': self.data,
            'ax': self.ax,
            'orient': self.orient,
            'order': self.order,
            'hue_order': self.hue_order,
            'palette': self.palette,
            'capsize': capsize,
            'errcolor': errcolor,
            'errwidth': errwidth,
        })

        # Have to swap x and y columns for horizontal bars
        if self.orient == 'v':
            g = sns.barplot(y=self.ycolumn, x=self.xcolumn, **kwargs)
        else:
            g = sns.barplot(y=self.xcolumn, x=self.ycolumn, **kwargs)
        if remove_legend:
            g.get_legend().remove()

        # Pull out all the bars drawn by the barplot function
        new_lines = []
        for patch in self.ax.patches:
            if patch in old_patches:
                continue
            if self.orient == 'v':
                new_lines.append(patch.get_x() + patch.get_width()/2)
            else:
                new_lines.append(patch.get_y() + patch.get_height()/2)

        if len(new_lines) != len(self.total_order):
            raise ValueError('Got {} lines but {} items to order: {} vs {}'.format(
                len(new_lines), len(self.total_order), new_lines, self.total_order))
        self.x_ticks = np.sort(np.array(new_lines))

    def add_samples(self,
                    marker_color_fade: float = 0.8,
                    markerfacecolor: Optional[str] = None,
                    jitter: float = 0.5):
        """ Plot all the individual samples for each category

        :param float marker_color_fade:
            How much to fade the extrema markers
        :param float jitter:
            How much random noise to add to the x-position of each sample
        """
        for i, (key, xcoord) in enumerate(zip(self.total_order, self.x_ticks)):
            points, color = self._extract_points(i, key)
            fade_color = color * marker_color_fade

            points_x = np.full(points.shape, xcoord, dtype=np.float) + (np.random.ranf(points.shape) - 0.5)*jitter

            if markerfacecolor is not None:
                color = markerfacecolor

            if self.orient == 'v':
                self.ax.plot(points_x, points, 'o',
                             markerfacecolor=color,
                             markeredgecolor=fade_color)
            else:
                self.ax.plot(points, points_x, 'o',
                             markerfacecolor=color,
                             markeredgecolor=fade_color)

    def add_extrema(self,
                    how: str = 'mean',
                    extremes: str = 'upper',
                    min_extreme: float = 2.0,
                    max_extreme: float = 5.0,
                    marker_color_fade: float = 0.8,
                    fontsize: float = 24,
                    jitter: float = 0.5):
        """ Add extremes to the plot

        :param str how:
            One of 'mean' (+/- standard deviation) or 'median' (+/- IQR)
        :param str extremes:
            Either 'upper', 'lower', or 'both', which end of the extrema to plot
        :param float min_extreme:
            Minimum value to plot (e.g. mean + min_extreme * std)
        :param float max_extreme:
            Maximum value to plot (e.g. mean + max_extreme * std)
        :param float marker_color_fade:
            How much to fade the extrema markers
        :param float jitter:
            How much random noise to add to the x-position of each sample
        """
        if extremes in (None, 'none'):
            return

        yupper = np.max(self.data[self.ycolumn].values) * 1.05
        ylower = np.min(self.data[self.ycolumn].values) * 1.05

        for i, (key, xcoord) in enumerate(zip(self.total_order, self.x_ticks)):
            points, color = self._extract_points(i, key)
            fade_color = color * marker_color_fade

            # Calculate the bin stats
            num_points = points.shape[0]
            ycenter, yextent = self._calc_extrema(points, how=how)

            # Plot upper outliers
            if extremes in ('upper', 'both'):
                upper_mask = np.logical_and(points >= ycenter + min_extreme * yextent,
                                            points <= ycenter + max_extreme * yextent)
                upper_points = points[upper_mask].values
                if upper_points.shape[0] > 0:
                    upper_x = np.full(upper_points.shape, xcoord, dtype=np.float) + (np.random.ranf(upper_points.shape) - 0.5)*jitter

                    if self.orient == 'v':
                        self.ax.plot(upper_x, upper_points, 'o', markerfacecolor=color, markeredgecolor=fade_color)
                        self.ax.text(xcoord, yupper, 'Upper Outliers={:d} ({:0.1%})'.format(np.sum(upper_mask), np.sum(upper_mask)/num_points),
                                     horizontalalignment='center', fontsize=fontsize)
                    else:
                        self.ax.plot(upper_points, upper_x, 'o', markerfacecolor=color, markeredgecolor=fade_color)
                        self.ax.text(yupper, xcoord, 'Upper Outliers={:d} ({:0.1%})'.format(np.sum(upper_mask), np.sum(upper_mask)/num_points),
                                     horizontalalignment='center', fontsize=fontsize)

            # Plot lower outliers
            if extremes in ('lower', 'both'):
                lower_mask = np.logical_and(points <= ycenter - min_extreme * yextent,
                                            points >= ycenter - max_extreme * yextent)
                lower_points = points[lower_mask].values

                if lower_points.shape[0] > 0:
                    lower_x = np.full(lower_points.shape, xcoord, dtype=np.float) + (np.random.ranf(lower_points.shape) - 0.5)*jitter

                    if self.orient == 'v':
                        self.ax.plot(lower_x, lower_points, 'o', markerfacecolor=color, markeredgecolor=fade_color)
                        self.ax.text(xcoord, ylower, 'Lower Outliers={:d} ({:0.1%})'.format(np.sum(lower_mask), np.sum(lower_mask)/num_points),
                                     horizontalalignment='center', fontsize=fontsize)
                    else:
                        self.ax.plot(lower_points, lower_x, 'o', markerfacecolor=color, markeredgecolor=fade_color)
                        self.ax.text(ylower, xcoord, 'Lower Outliers={:d} ({:0.1%})'.format(np.sum(lower_mask), np.sum(lower_mask)/num_points),
                                     horizontalalignment='center', fontsize=fontsize)

    def add_num_samples(self, fontsize: int = 24, color: str = None):
        """ Add the number of samples

        :param int fontsize:
            Size for the text
        :param str color:
            The color to use for the annotation text
        """

        if color is None:
            color = {
                'dark': '#FFFFFF',
                'dark_poster': '#FFFFFF',
            }.get(self.plot_style, '#000000')
        ymax = np.max(self.data[self.ycolumn].values) * 1.1

        for i, (key, xcoord) in enumerate(zip(self.total_order, self.x_ticks)):
            # Work out the number of samples in each bin
            points, _ = self._extract_points(i, key)
            num_points = points.shape[0]

            # Label the number of samples
            if self.orient == 'v':
                self.ax.text(xcoord, ymax, 'n={:,}'.format(num_points),
                             horizontalalignment='center',
                             fontsize=fontsize,
                             color=color)
            else:
                self.ax.text(ymax, xcoord, 'n={:,}'.format(num_points),
                             horizontalalignment='center',
                             fontsize=fontsize,
                             color=color)

    def add_sample_means(self,
                         how: str = 'mean',
                         marker: str = 'o',
                         markersize: float = 5,
                         linewidth: float = 2,
                         marker_color_fade: float = 0.8,
                         jitter: float = 0.1,
                         showbands: bool = False):
        """ Add sample means for replicates within the data

        :param str how:
            One of 'mean' (+/- standard deviation) or 'median' (+/- IQR)
        :param str marker:
            Which marker shape to use
        :param float marker_color_fade:
            How much to fade the markers
        :param float jitter:
            How wide to make the sample scatter
        """

        if self.sample_column is None:
            return

        for i, (xdata, key) in enumerate(zip(self.x_ticks, self.total_order)):
            sample_data, color = self._extract_dataframe(i, key)
            fade_color = color * marker_color_fade

            all_samples = np.unique(sample_data[self.sample_column])
            for sample_key in all_samples:
                # Pull out each individual sample
                single_sample_data = sample_data[sample_data[self.sample_column] == sample_key]
                if single_sample_data.shape[0] < 3:
                    continue
                ycenter, yextent = self._calc_extrema(single_sample_data[self.ycolumn].values, how=how)
                xcoord = xdata + (np.random.rand(1) - 0.5)*jitter

                if self.orient == 'v':
                    if showbands:
                        self.ax.plot([xcoord, xcoord], [ycenter - yextent, ycenter + yextent],
                                     color=fade_color, linestyle='-', linewidth=linewidth)
                    self.ax.plot([xcoord], [ycenter], marker=marker, markersize=markersize, color=fade_color)
                else:
                    if showbands:
                        self.ax.plot([ycenter - yextent, ycenter + yextent], [xcoord, xcoord],
                                     color=fade_color, linestyle='-', linewidth=linewidth)
                    self.ax.plot([ycenter], [xcoord], marker=marker, markersize=markersize, color=fade_color)

    def add_significance_bars(self,
                              significance: Dict,
                              barcolor: Optional[str] = None,
                              linewidth: float = 5,
                              fontsize: float = 28,
                              markers: Optional[Dict[float, str]] = None):
        """ Add significance bars and stars to barplot and violinplot-style charts

        :param dict[tuple, float] significance:
            A mapping of (category1, category2): pvalue for each pair of categories
            Something like what is returned by :py:func:`~agg_dyn.stats.utils.calc_pairwise_significance`
        :param str barcolor:
            Color for the significance bars
        :param str linewidth:
            Linewidth for the significance bars
        :param int fontsize:
            Fontsize for the significance marks
        :param dict[float, str] markers:
            A mapping of pvalue cutoff: symbol
        """
        if significance is None:
            return

        if barcolor is None:
            barcolor = {
                'dark': '#AAAAAA',
                'dark_poster': '#AAAAAA',
            }.get(self.plot_style, '#666666')

        # Sort the significance markers
        if markers is None:
            markers = {
                0.05: '*',
                0.01: '**',
                0.001: '***',
            }
        else:
            markers = dict(markers)
        sorted_markers = list(sorted(markers.items(), key=lambda x: x[0]))

        # Convert the significance values to symbols
        total_order = self.total_order
        cast_order = [str, int, float, (lambda t: tuple(str(s) for s in t)), (lambda x: x)]
        key_cast = None  # Shenanigans to counteract normalization elsewhere

        plotdata_significance = {}
        index_significance = {}
        for (key1, key2), pvalue in significance.items():
            # Convert the significance matrix to a data frame
            if self.hue_column is None:
                sig_key1 = '{}1'.format(self.xcolumn)
                sig_key2 = '{}2'.format(self.xcolumn)
                plotdata_significance.setdefault(sig_key1, []).append(key1)
                plotdata_significance.setdefault(sig_key2, []).append(key2)
            else:
                sig_key1 = '{}1'.format(self.xcolumn)
                sig_key2 = '{}1'.format(self.hue_column)
                sig_key3 = '{}2'.format(self.xcolumn)
                sig_key4 = '{}2'.format(self.hue_column)
                plotdata_significance.setdefault(sig_key1, []).append(key1[0])
                plotdata_significance.setdefault(sig_key2, []).append(key1[1])
                plotdata_significance.setdefault(sig_key3, []).append(key2[0])
                plotdata_significance.setdefault(sig_key4, []).append(key2[1])
            plotdata_significance.setdefault('P-value', []).append(pvalue)

            # Work out where on the plot to draw the bars
            pvalue_symb = None
            for sig_level, sig_symbol in sorted_markers:
                if pvalue < sig_level:
                    pvalue_symb = sig_symbol
                    break
            if key_cast is None:
                for try_cast in cast_order:
                    try:
                        if try_cast(key1) in total_order:
                            key_cast = try_cast
                            break
                    except (TypeError, ValueError):
                        continue
            if key_cast is None:
                print('Got key1: {} of type {}'.format(key1, type(key1)))
                print('Got key2: {} of type {}'.format(key2, type(key2)))
                print('Got order:')
                for rec in total_order:
                    print('* {} of type {}'.format(rec, type(rec)))
                raise ValueError('Cannot find a valid cast for key {} of type {}'.format(key1, type(key1)))

            if key_cast(key1) not in total_order:
                raise KeyError('Cannot find {} in ordering: {}'.format(key1, total_order))
            if key_cast(key2) not in total_order:
                raise KeyError('Cannot find {} in ordering: {}'.format(key2, total_order))

            if pvalue_symb is not None:
                index_significance[total_order.index(key_cast(key1)),
                                   total_order.index(key_cast(key2))] = pvalue_symb
        print('Plotting {} values'.format(len(index_significance)))
        self._plotdata_significance = plotdata_significance

        if self.orient == 'v':
            # Draw horizontal lines above vertical bars
            cur_ymin, cur_ymax = self.ax.get_ylim()
            print('Got y {} to {}'.format(cur_ymin, cur_ymax))

            bottom_y = cur_ymax * 0.9
            delta = (cur_ymax - cur_ymin) * 0.02
            step = 1
            for (i1, i2), pvalue_symb in index_significance.items():
                cur_y = bottom_y + delta*step
                self.ax.plot([i1, i2], [cur_y, cur_y],
                             color=barcolor, linewidth=linewidth, clip_on=False)
                self.ax.text((i1+i2)/2, cur_y, pvalue_symb,
                             color=barcolor, fontsize=fontsize,
                             horizontalalignment='center')
                step += 1
        else:
            # Draw vertical lines to the right of horizontal bars
            cur_xmin, cur_xmax = self.ax.get_xlim()
            bottom_x = cur_xmax * 0.9
            delta = (cur_xmax - cur_xmin) * 0.02
            step = 1
            for (i1, i2), pvalue_symb in index_significance.items():
                cur_x = bottom_x + delta*step
                self.ax.plot([cur_x, cur_x], [i1, i2],
                             color=barcolor, linewidth=linewidth, clip_on=False)
                self.ax.text(cur_x + delta*0.25, (i1+i2)/2, pvalue_symb,
                             color=barcolor, fontsize=fontsize,
                             rotation=90, horizontalalignment='center')
                step += 1

    def set_ylim(self,
                 ymin: Optional[float] = None,
                 ymax: Optional[float] = None):
        """ Set the value axis limits, even when the axes are flipped

        :param float ymin:
            Minimum y-value to plot
        :param float ymax:
            Maximum y-value to plot
        """
        if self.orient == 'v':
            # Set the y limits (x axis corresponds to categories)
            cur_ymin, cur_ymax = self.ax.get_ylim()
            if ymin is not None:
                cur_ymin = ymin
            if ymax is not None:
                cur_ymax = ymax
            self.ax.set_ylim([cur_ymin, cur_ymax])
        else:
            # Set the x limits (y axis corresponds to categories)
            cur_ymin, cur_ymax = self.ax.get_xlim()
            if ymin is not None:
                cur_ymin = ymin
            if ymax is not None:
                cur_ymax = ymax
            self.ax.set_xlim([cur_ymin, cur_ymax])

    def set_xlabel(self, xlabel: str):
        """ Set the category axis label

        :param str xlabel:
            Label to set on the categorical axis
        """
        if xlabel is None:
            return
        if self.orient == 'v':
            self.ax.set_xlabel(xlabel)
        else:
            self.ax.set_ylabel(xlabel)

    def set_xticklabels(self, xticklabels: List[str]):
        """ Set the category axis ticks

        :param list[str] xticklabels:
            Labels for the category axis
        """
        if xticklabels is None:
            return
        if self.orient == 'v':
            self.ax.set_xticklabels(xticklabels)
        else:
            self.ax.set_yticklabels(xticklabels)

    def set_ylabel(self, ylabel: str):
        """ Set the axis y-label

        :param str ylabel:
            Label to set on the value axis
        """
        if ylabel is None:
            return
        if self.orient == 'v':
            self.ax.set_ylabel(ylabel)
        else:
            self.ax.set_xlabel(ylabel)

    def set_yticklabels(self, yticklabels: List[str]):
        """ Set the value axis ticks

        :param list[str] yticklabels:
            Labels for the value axis
        """
        if yticklabels is None:
            return
        if self.orient == 'v':
            self.ax.set_yticklabels(yticklabels)
        else:
            self.ax.set_xticklabels(yticklabels)

    def set_yscale(self, yscale: Optional[str] = None):
        """ Set the value axis scale (linear vs log)

        :param str yscale:
            One of 'linear' or 'log'
        """
        if yscale is None:
            return
        if self.orient == 'v':
            self.ax.set_yscale(yscale)
        else:
            self.ax.set_xscale(yscale)

    def save_plotdata(self,
                      outfile: pathlib.Path,
                      suffix: Optional[str] = None,
                      calc_dists: bool = False):
        """ Save the main data from a plot

        :param Path outfile:
            The file to write the output data to
        :param str suffix:
            Suffix to save the output file to
        :param bool calc_dists:
            If True, calculate the kernel density distributions
        """
        if suffix is None:
            suffix = outfile.suffix
        self.extract_plotdata(calc_dists=calc_dists)

        outfile = outfile.parent / f'{outfile.stem}{suffix}'
        print(f'Writing plot data to {outfile}')
        plotdata = pd.DataFrame(self._plotdata)
        self._save_dataframe(plotdata, outfile)

        sample_outfile = outfile.parent / f'{outfile.stem}_samples{suffix}'
        print(f'Writing plot data samples to {sample_outfile}')
        plotdata_samples = pd.DataFrame(self._plotdata_samples)
        self._save_dataframe(plotdata_samples, sample_outfile)

        if self._plotdata_dists is not None:
            dist_outfile = outfile.parent / f'{outfile.stem}_dist{suffix}'
            print(f'Writing distribution data to {dist_outfile}')
            plotdata_dists = pd.DataFrame(self._plotdata_dists)
            self._save_dataframe(plotdata_dists, dist_outfile)
        if self._plotdata_significance is not None:
            sig_outfile = outfile.parent / f'{outfile.stem}_sig{suffix}'
            print('Writing significance data to {}'.format(sig_outfile))
            plotdata_significance = pd.DataFrame(self._plotdata_significance)
            self._save_dataframe(plotdata_significance, sig_outfile)

    def format_axis(self,
                    ymin: Optional[float] = None,
                    ymax: Optional[float] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    xticklabels: Optional[List[str]] = None,
                    yticklabels: Optional[List[str]] = None,
                    yscale: Optional[str] = None):
        """ Set generic axis format parameters

        Utility method to allow each axis plot function to format things the same

        :param float ymin:
            Lower limit for the value axis
        :param float ymax:
            Upper limit for the value axis
        :param str xlabel:
            Label for the category axis
        :param str ylabel:
            Label for the value axis
        :param list[str] xticklabels:
            Labels for the category ticks
        :param list[str] yticklabels:
            Labels for the value ticks
        :param str yscale:
            Scaling for the y axis (either 'linear' or 'log')
        """
        self.set_yscale(yscale)
        self.set_ylim(ymin=ymin, ymax=ymax)
        self.set_xlabel(xlabel)
        self.set_ylabel(ylabel)
        self.set_xticklabels(xticklabels)
        self.set_yticklabels(yticklabels)

# Functions


def add_boxplot(ax: Axes,
                data: pd.DataFrame,
                x: str,
                y: str,
                hue: Optional[str] = None,
                significance: Optional[Dict] = None,
                orient: str = 'v',
                order: Optional[List] = None,
                hue_order: Optional[List] = None,
                palette: str = COLOR_PALETTE,
                sig_barcolor: Optional[str] = None,
                sig_linewidth: float = 5,
                sig_fontsize: float = 24,
                sig_markers: Optional[Dict[float, str]] = None,
                capsize: float = 0.2,
                errcolor: Optional[str] = None,
                savefile: Optional[pathlib.Path] = None,
                **kwargs):
    """ Add a seaborn-style boxplot with extra decorations

    :param Axes ax:
        The matplotlib axis to add the barplot for
    :param DataFrame data:
        The data to add a barplot for
    :param str x:
        The column to use for the categorical values
    :param str y:
        The column to use for the real values
    :param dict[tuple, float] significance:
        A mapping of (category1, category2): pvalue for each pair of categories.
        Something like what is returned by :py:func:`~agg_dyn.stats.utils.calc_pairwise_significance`
    :param str sig_barcolor:
        Color for the significance bars
    :param int sig_linewidth:
        Size of the significance bars
    :param int sig_fontsize:
        Size of the markers for the significance fonts
    :param dict sig_markers:
        A dictionary mapping significance level: marker string
    :param float capsize:
        The size of the caps on the error bars
    :param str errcolor:
        The color for the error bars
    :param \\*\\*kwargs:
        Formatting keyword args
    :returns:
        The axis we drew on
    """

    plotter = CatPlot(ax=ax, data=data, xcolumn=x, ycolumn=y, hue_column=hue,
                      order=order, hue_order=hue_order, palette=palette,
                      orient=orient)
    plotter.add_boxplot()
    plotter.add_num_samples()
    plotter.add_significance_bars(significance,
                                  barcolor=sig_barcolor,
                                  linewidth=sig_linewidth,
                                  fontsize=sig_fontsize,
                                  markers=sig_markers)
    plotter.format_axis(**kwargs)
    if savefile is not None:
        plotter.save_plotdata(savefile, suffix='.xlsx', calc_dists=False)
    return ax


def add_barplot(ax: Axes, data, x, y, hue=None, significance=None, orient='v',
                order=None, hue_order=None, palette=COLOR_PALETTE,
                sig_barcolor=None, sig_linewidth=5, bottom=0.0,
                sig_fontsize=24, sig_markers=None, capsize=0.2, errcolor=None,
                savefile=None, plot_individual_samples=False,
                plot_sample_numbers=True,
                **kwargs):
    """ Add a seaborn-style barplot with extra decorations

    :param Axes ax:
        The matplotlib axis to add the barplot for
    :param DataFrame data:
        The data to add a barplot for
    :param str x:
        The column to use for the categorical values
    :param str y:
        The column to use for the real values
    :param dict[tuple, float] significance:
        A mapping of (category1, category2): pvalue for each pair of categories.
        Something like what is returned by :py:func:`~agg_dyn.stats.utils.calc_pairwise_significance`
    :param str sig_barcolor:
        Color for the significance bars
    :param int sig_linewidth:
        Size of the significance bars
    :param int sig_fontsize:
        Size of the markers for the significance fonts
    :param dict sig_markers:
        A dictionary mapping significance level: marker string
    :param float capsize:
        The size of the caps on the error bars
    :param str errcolor:
        The color for the error bars
    :param \\*\\*kwargs:
        Formatting keyword args
    :returns:
        The axis we drew on
    """

    plotter = CatPlot(ax=ax, data=data, xcolumn=x, ycolumn=y, hue_column=hue,
                      order=order, hue_order=hue_order, palette=palette,
                      orient=orient)
    if orient == 'v':
        bar_kwargs = {'bottom': bottom}
    else:
        bar_kwargs = {'left': bottom}
    plotter.add_barplot(capsize=capsize, errcolor=errcolor, **bar_kwargs)
    if plot_sample_numbers:
        plotter.add_num_samples()
    if plot_individual_samples:
        plotter.add_samples(markerfacecolor='k')
    plotter.add_significance_bars(significance,
                                  barcolor=sig_barcolor,
                                  linewidth=sig_linewidth,
                                  fontsize=sig_fontsize,
                                  markers=sig_markers)
    plotter.format_axis(**kwargs)
    if savefile is not None:
        plotter.save_plotdata(savefile, suffix='.xlsx', calc_dists=False)
    return ax


def add_violins_with_outliers(ax: Axes,
                              data: pd.DataFrame,
                              xcolumn: str,
                              ycolumn: str,
                              order: Optional[List] = None,
                              palette: str = 'deep',
                              extremes: str = 'upper',
                              std_min: float = 2.0,
                              std_max: float = 5.0,
                              fontsize: str = 16,
                              significance: Optional[Dict[float, str]] = None,
                              graphic: str = 'violin',
                              savefile: Optional[pathlib.Path] = None,
                              **kwargs):
    """ Make a violinplot with outliers

    Make a simple violinplot with outliers above and below the violins:

    .. code-block:: python

        data = pd.DataFrame({
            'Label': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Value': [0.1, 0.2, 0.3, 0.2, 0.3, 0.4],
        })
        fig, ax = plt.subplots(1, 1)
        add_violins_with_outliers(ax, data, xcolumn='Label', ycolumn='Value')

    :param Axes ax:
        The axis to plot on
    :param DataFrame data:
        The data to plot
    :param xcolumn:
        The column name for the x plot (category)
    :param ycolumn:
        The column name for the y plot (violin distribution)
    :param list[str] order:
        The list of group names to plot in order
    :param str extremes:
        One of 'upper', 'lower', 'both'
    :param float std_min:
        Minimum factor * standard deviation to be an "outlier"
    :param float std_max:
        Maximum factor * standard deviation to be an "outlier" (more extreme values discarded)
    :param int fontsize:
        Size for the violin plot text
    :param dict significance:
        The significance values like those returned from :py:func:`calc_pairwise_significance`
    :param str graphic:
        One of 'violin' or 'box': which distribution plot style to use
    """
    plotter = CatPlot(ax=ax, data=data, xcolumn=xcolumn, ycolumn=ycolumn,
                      order=order, palette=palette)
    plotter.filter_extrema(how='mean', extremes='both', max_extreme=std_max)
    if graphic == 'violin':
        plotter.add_violinplot()
    elif graphic == 'box':
        plotter.add_boxplot()
    elif graphic == 'bar':
        plotter.add_barplot()
    else:
        raise KeyError('Unknown graphic "{}", choose one of "violin", "box", or "bar"'.format(graphic))

    plotter.add_extrema(how='mean',
                        extremes=extremes,
                        min_extreme=std_min,
                        max_extreme=std_max,
                        fontsize=fontsize)
    plotter.add_num_samples()
    plotter.add_significance_bars(significance)
    plotter.format_axis(**kwargs)
    if savefile is not None:
        plotter.save_plotdata(savefile, suffix='.xlsx', calc_dists=False)
    return ax


def add_lineplot(ax: Axes,
                 data: pd.DataFrame,
                 x: str, y: str,
                 hue: Optional[str] = None,
                 order: Optional[List[str]] = None,
                 hue_order: Optional[List[str]] = None,
                 palette: str = COLOR_PALETTE,
                 savefile: Optional[pathlib.Path] = None,
                 label: Optional[str] = None,
                 drop_missing: bool = False,
                 mirror_x: bool = False,
                 err_style: str = 'band',
                 yscale: float = 1.0):
    """ Add a seaborn-style lineplot with extra decorations

    FIXME: This could probably be integrated into the CatPlot framework

    :param Axes ax:
        The matplotlib axis to add the barplot for
    :param DataFrame data:
        The data to add a barplot for
    :param str x:
        The column to use for the categorical values
    :param str y:
        The column to use for the real values
    :param str palette:
        The palette to use
    :param Path savefile:
        If not None, save the figure data to this path
    :param bool mirror_x:
        If True, mirror the x and y data around x==0.0
    :param float yscale:
        Scale factor for the y data
    """
    bins = {}

    # Select only the columns we need, then drop missing data
    all_columns = [x, y]
    if hue is not None:
        all_columns.append(hue)
    data = data[all_columns]
    data = data.dropna(how='any', axis=0)

    if data.shape[0] < 1:
        raise ValueError(f'No valid records after removing empty rows using {all_columns}')

    if order is None:
        order = np.sort(np.unique(data[x]))
    if hue is None:
        hue_order = [None]
    elif hue_order is None:
        hue_order = np.sort(np.unique(data[hue]))

    for cat in order:
        for hue_cat in hue_order:
            if hue_cat is None:
                mask = data[x] == cat
            else:
                mask = np.logical_and(data[x] == cat, data[hue] == hue_cat)

            # Handle missing categories
            n_samples = np.sum(mask)
            if n_samples >= 3:
                catdata = data[mask]
                ydata = catdata[y].values

                ymean = np.mean(ydata)
                ylow, yhigh = bootstrap_ci(ydata)
            elif n_samples >= 1:
                catdata = data[mask]
                ydata = catdata[y].values
                ymean = ylow = yhigh = np.mean(ydata)
            else:
                ymean = ylow = yhigh = np.nan

            if hue is None:
                bins.setdefault(x, []).append(cat)
                bins.setdefault(f'{y} Mean', []).append(ymean*yscale)
                bins.setdefault(f'{y} CI Low', []).append(ylow*yscale)
                bins.setdefault(f'{y} CI High', []).append(yhigh*yscale)
                bins.setdefault('Samples', []).append(n_samples)
            else:
                bins.setdefault(x, []).append(cat)
                bins.setdefault(hue, []).append(hue_cat)
                bins.setdefault(f'{y} Mean', []).append(ymean*yscale)
                bins.setdefault(f'{y} CI Low', []).append(ylow*yscale)
                bins.setdefault(f'{y} CI High', []).append(yhigh*yscale)
                bins.setdefault('Samples', []).append(n_samples)

    # Save the background data
    bins = pd.DataFrame(bins)
    if savefile is not None:
        if savefile.suffix != '.xlsx':
            savefile = savefile.parent / f'{savefile.stem}.xlsx'
        savefile.parent.mkdir(exist_ok=True, parents=True)
        bins.to_excel(str(savefile))

    # Now draw the plots
    palette = colorwheel(palette, len(hue_order))

    for i, hue_cat in enumerate(hue_order):
        if hue_cat is None:
            xcoords = bins[x].values
            ymean = bins[f'{y} Mean'].values
            ylow = bins[f'{y} CI Low'].values
            yhigh = bins[f'{y} CI High'].values
            hue_label = label
        else:
            hue_bins = bins[bins[hue] == hue_cat]

            xcoords = hue_bins[x].values
            ymean = hue_bins[f'{y} Mean'].values
            ylow = hue_bins[f'{y} CI Low'].values
            yhigh = hue_bins[f'{y} CI High'].values
            if label is None:
                hue_label = hue_cat
            else:
                hue_label = f'{hue_cat} {label}'
        color = palette[i]

        if drop_missing:
            mask = ~np.isnan(ymean)
            xcoords = xcoords[mask]
            ylow = ylow[mask]
            yhigh = yhigh[mask]
            ymean = ymean[mask]

        if mirror_x:
            xcoords = np.concatenate([-xcoords[::-1], xcoords])
            ylow = np.concatenate([ylow[::-1], ylow])
            yhigh = np.concatenate([yhigh[::-1], yhigh])
            ymean = np.concatenate([ymean[::-1], ymean])

        if err_style in ('band', 'bands'):
            ax.fill_between(xcoords, ylow, yhigh, facecolor=color, alpha=0.5)
            ax.plot(xcoords, ymean, '-', color=color, label=hue_label)
        elif err_style in ('bar', 'bars'):
            ax.errorbar(xcoords, ymean, np.stack([ymean-ylow, yhigh-ymean], axis=0),
                        capsize=15, linewidth=3, color=color, label=hue_label)
        else:
            raise ValueError(f'Unknown error style: "{err_style}"')

    return ax
