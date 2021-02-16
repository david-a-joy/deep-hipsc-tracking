""" Enable replotting of different kinds of plot elements

Plot element generators:

* :py:func:`add_single_boxplot`: Add a single boxplot at a fixed x, y
* :py:func:`add_single_barplot`: Add a single boxplot at a fixed x, y

"""

# Imports
import pathlib
import itertools
from typing import Optional, Union, Tuple, Generator, List

# 3rd party
import numpy as np

import pandas as pd

from matplotlib.patches import Rectangle

# Our own imports
from .utils import bootstrap_ci, get_histogram
from .styling import colorwheel, set_plot_style
from .split_axes import SplitAxes

# Constants

PLOT_STYLE: str = 'poster'
PLOT_TYPE: str = 'bars'
ORIENT: str = 'vertical'
PALETTE: str = 'wheel_greywhite'

# Types

CategoryKeyType = Union[str, Tuple[str]]  # For category or hue keys


# Helper Functions


def read_data_frame(datafile: pathlib.Path) -> pd.DataFrame:
    """ Read in a data frame in a data type agnostic way

    :param Path datafile:
        The spreadsheet to read
    :returns:
        A pandas DataFrame for that spreadsheet
    """
    if datafile.suffix == '.csv':
        data = pd.read_csv(str(datafile))
    elif datafile.suffix == '.tsv':
        data = pd.read_csv(str(datafile), sep='\t')
    elif datafile.suffix in ('.xls', '.xlsx'):
        data = pd.read_excel(str(datafile))
    else:
        raise ValueError('Unknown data file type: {}'.format(datafile))
    return data


def write_data_frame(df: pd.DataFrame, outfile: pathlib.Path):
    """ Save off a DataFrame """
    suffix = outfile.suffix
    if suffix == '.csv':
        df.to_csv(str(outfile), header=True, index=False)
    elif suffix == '.xlsx':
        df.to_excel(str(outfile), header=True, index=False)
    else:
        raise KeyError('Unknown plotdata output file type: {}'.format(outfile))


def add_single_boxplot(ax,
                       xcoord: float,
                       ycoords: np.ndarray,
                       width: float,
                       color: str,
                       edgecolor: str,
                       linewidth: float,
                       orient: str,
                       label: Optional[str] = None,
                       error_kw: Optional[dict] = None):
    """ Add a single boxplot at position x with range y

    :param Axis ax:
        The matplotlib axis to plot on
    :param float xcoord:
        The x-coordinate to plot the box at
    :param tuple[float] ycoord:
        The 5%, 25%, 50%, 75%, 95% levels in y to plot
    :param float width:
        The box width
    :param str color:
        The color for the box fill
    :param str edgecolor:
        The color for the box edges
    :param float linewidth:
        The width of the borders
    :param str orient:
        The orientation of the boxes ('horizontal' or 'vertical')
    :param str label:
        **UNUSED**
    :param dict error_kw:
        Keyword arguments to pass to matplotlib.pyplot.errorbar
    """
    if error_kw is None:
        error_kw = {}

    yp5, yp25, yp50, yp75, yp95 = ycoords

    yerr = [yp50-yp5, yp95-yp50]
    xlow = xcoord - width/2.0
    xhigh = xcoord + width/2.0
    ylow = yp25
    yhigh = yp75

    if orient.startswith('v'):
        ax.errorbar(xcoord, yp50, yerr=yerr, zorder=0, **error_kw)
        rect = Rectangle((xlow, ylow),
                         width=(xhigh-xlow),
                         height=(yhigh-ylow),
                         facecolor=color,
                         edgecolor=edgecolor,
                         linewidth=linewidth,
                         alpha=1.0,
                         zorder=1)
        ax.add_patch(rect)
        ax.plot([xlow, xhigh], [yp50, yp50],
                linestyle='-',
                color=edgecolor,
                linewidth=linewidth,
                zorder=2)
    elif orient.startswith('h'):
        ax.errorbar(xcoord, yp50, yerr=yerr, zorder=0, **error_kw)
        rect = Rectangle((ylow, xlow),
                         width=(yhigh-ylow),
                         height=(xhigh-xlow),
                         facecolor=color,
                         edgecolor=edgecolor,
                         linewidth=linewidth,
                         alpha=1.0,
                         zorder=1)
        ax.add_patch(rect)
        ax.plot([yp50, yp50], [xlow, xhigh],
                linestyle='-',
                color=edgecolor,
                linewidth=linewidth,
                zorder=2)
    else:
        raise KeyError(f'Unknown orientation: {orient}')


def add_single_barplot(ax,
                       xcoord: float,
                       ycoords: np.ndarray,
                       width: float,
                       color: str,
                       edgecolor: str,
                       linewidth: float,
                       orient: str = 'v',
                       bottom: Optional[float] = None,
                       label: Optional[str] = None,
                       error_kw: Optional[dict] = None,
                       clip_on: bool = True,
                       errorbar_side: Optional[str] = None):
    """ Add a single barplot at position x with range y

    :param Axis ax:
        The matplotlib axis to plot on
    :param float xcoord:
        The x-coordinate to plot the box at
    :param tuple[float] ycoord:
        The ymean, ylow, yhigh confidence bounds
    :param float width:
        The box width
    :param str color:
        The color for the box fill
    :param str edgecolor:
        The color for the box edges
    :param float linewidth:
        The width of the borders
    :param str orient:
        The orientation of the boxes ('horizontal' or 'vertical')
    :param str label:
        Label for legends and etc
    :param dict error_kw:
        Keyword arguments to pass to matplotlib.pyplot.bar
    """
    if errorbar_side is None:
        errorbar_side = 'both'

    if error_kw is None:
        error_kw = {}

    if len(ycoords) == 1:
        yheight = ycoords
        yerr = None
    else:
        yheight, ylow, yhigh = ycoords
        if errorbar_side == 'both':
            yerr = np.array([[ylow], [yhigh]])
        elif errorbar_side in ('bottom', 'left'):
            yerr = np.array([[ylow], [0.0]])
        elif errorbar_side in ('top', 'right'):
            yerr = np.array([[0.0], [yhigh]])
        elif errorbar_side in ('dir', 'direction'):
            # Pick error bar based on bar direction
            if yheight < 0.0:
                yerr = np.array([[ylow], [0.0]])
            else:
                yerr = np.array([[0.0], [yhigh]])
        else:
            raise ValueError(f'Unknown errorbar_side: "{errorbar_side}"')

        assert yerr.shape[0] == 2

    if orient.startswith('v'):
        ax.bar(xcoord,
               height=yheight,
               width=width,
               yerr=yerr,
               color=color,
               edgecolor=edgecolor,
               linewidth=linewidth,
               label=label,
               bottom=bottom,
               error_kw=error_kw,
               clip_on=clip_on)
    elif orient.startswith('h'):
        ax.barh(xcoord,
                height=width,
                width=yheight,
                yerr=yerr,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=label,
                left=bottom,
                error_kw=error_kw,
                clip_on=clip_on)
    else:
        raise KeyError(f'Unknown orientation "{orient}"')


# Classes


class StrInt(object):
    """ Sort ints before strings, in numerical order """

    def __init__(self, val: str):
        try:
            val = int(val)
            is_num = True
        except (TypeError, ValueError):
            is_num = False
        self.is_num = is_num
        self.val = val

    def __lt__(self, other):
        if self.is_num:
            if other.is_num:
                return self.val < other.val
            return True
        else:
            if other.is_num:
                return False
            return self.val < other.val


class DataManager(object):
    """ Manage and handle all the required data elements """

    def __init__(self,
                 xcolumn: str,
                 ycolumn: str,
                 hue_column: Optional[str] = None,
                 ylimits: Optional[Tuple[Tuple[float]]] = None,
                 scale: Optional[float] = None,
                 order: Optional[List[str]] = None,
                 hue_order: Optional[List[str]] = None):

        # Slots for the various results
        self._raw_data = None
        self._cache_data = None
        self._cache_dist = None
        self._cache_sig = None

        # Column selectors
        self.xcolumn = xcolumn
        self.ycolumn = ycolumn
        self.hue_column = hue_column

        # Figure layout
        self.figure_y = 10
        self.figure_x = None

        self.bar_width = 1.0
        self.bar_padding = 0.15
        self.bar_cat_padding = 0.4

        self.box_width = 1.0
        self.box_padding = 0.15
        self.box_cat_padding = 0.4

        # Order attributes
        self._order = order
        self._hue_order = hue_order
        self._total_order = None

        # Column positions
        self._xcoords = None
        self._xcoord_map = None
        self._xtick_coords = None

    @property
    def order(self):
        """ Store the order for each category """
        if self._order is not None:
            return self._order
        if self._raw_data is not None:
            order = list(sorted(np.unique(self._raw_data[self.xcolumn]), key=StrInt))
        elif self._cache_data is not None:
            order = list(sorted(np.unique(self._cache_data[self.xcolumn]), key=StrInt))
        else:
            raise ValueError('No data frame is defined')
        self._order = order
        return self._order

    @property
    def hue_order(self):
        """ Store the order for each hue """
        if self.hue_column is None:
            return None
        if self._hue_order is not None:
            return self._hue_order
        self._hue_order = list(sorted(np.unique(self.data[self.hue_column]), key=StrInt))
        return self._hue_order

    @property
    def total_order(self):
        """ Ordered pairs of category, hue for each value """
        if self._total_order is not None:
            return self._total_order
        if self.hue_column is None:
            self._total_order = list(self.order)
        else:
            self._total_order = list(itertools.product(self.order, self.hue_order))
        return self._total_order

    def load_cache_file(self, cache_file: pathlib.Path):
        """ Load the data from a cached file """

        # Load the cache data
        self._cache_data = read_data_frame(cache_file)

        # Look for the other possible cache files
        datadir = cache_file.parent
        suffix = cache_file.suffix
        stem = cache_file.stem

        distfile = datadir / '{}_dist{}'.format(stem, suffix)
        sigfile = datadir / '{}_sig{}'.format(stem, suffix)

        if distfile.is_file():
            self._cache_dist = read_data_frame(distfile)
        if sigfile.is_file():
            self._cache_sig = read_data_frame(sigfile)

    def save_cache_file(self, cache_file: pathlib.Path):
        """ Write the results to a file """

        # Look for the other possible cache files
        datadir = cache_file.parent
        suffix = cache_file.suffix
        stem = cache_file.stem

        distfile = datadir / '{}_dist{}'.format(stem, suffix)
        sigfile = datadir / '{}_sig{}'.format(stem, suffix)

        write_data_frame(self._cache_data, cache_file)

        if distfile.is_file():
            write_data_frame(self._cache_dist, distfile)
        if sigfile.is_file():
            write_data_frame(self._cache_sig, sigfile)

    def load_raw_data(self,
                      raw_data: pd.DataFrame,
                      calc_dists: bool = False):
        """ Extract the data for re-plotting in other systems

        :param DataFrame raw_data:
            The raw data to process
        :param bool calc_dists:
            If True, calculate the kernel density distributions per category
        """

        def extract_values(cat: CategoryKeyType) -> np.ndarray:
            """ Extract each category as a single value """
            if self.hue_column is None:
                mask = xvals == cat
                cache_data.setdefault(self.xcolumn, []).append(cat)
            else:
                xcat, hcat = cat
                mask = np.logical_and(xvals == xcat, hvals == hcat)
                cache_data.setdefault(self.xcolumn, []).append(xcat)
                cache_data.setdefault(self.hue_column, []).append(hcat)
            # Pull out the selected values
            return yvals[mask]

        # Convert the plot parameters to a single spreadsheet
        self._raw_data = raw_data
        cache_data = {}
        cache_dist = {}

        yvals = raw_data[self.ycolumn].values
        xvals = raw_data[self.xcolumn].values
        if self.hue_column is None:
            hvals = None
        else:
            hvals = raw_data[self.hue_column].values

        # Extract summary stats for the values
        for i, key in enumerate(self.total_order):
            # Pull out the selected values
            selvals = extract_values(key)
            if selvals.shape[0] < 1:
                raise KeyError('No values for category: {}'.format(key))
            elif selvals.shape[0] == 1:
                p50 = mean = selvals[0]
                p5 = p25 = p75 = p95 = np.nan
                std = np.nan
                ci_low = ci_high = np.nan
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
                    cache_dist[catname + ' value'] = kernel_x
                    cache_dist[catname + ' density'] = kernel_y

            # Stash all the stats away
            cache_data.setdefault('{} p5'.format(self.ycolumn), []).append(p5)
            cache_data.setdefault('{} p25'.format(self.ycolumn), []).append(p25)
            cache_data.setdefault('{} p50'.format(self.ycolumn), []).append(p50)
            cache_data.setdefault('{} p75'.format(self.ycolumn), []).append(p75)
            cache_data.setdefault('{} p95'.format(self.ycolumn), []).append(p95)
            cache_data.setdefault('{} mean'.format(self.ycolumn), []).append(mean)
            cache_data.setdefault('{} std'.format(self.ycolumn), []).append(std)
            cache_data.setdefault('{} ci low'.format(self.ycolumn), []).append(ci_low)
            cache_data.setdefault('{} ci high'.format(self.ycolumn), []).append(ci_high)
            cache_data.setdefault('{} n'.format(self.ycolumn), []).append(selvals.shape[0])

        # Save the cached data values
        self._cache_data = pd.DataFrame(cache_data)

        # Kernel density estimates for violinplot extraction
        if calc_dists and cache_dist != {}:
            self._cache_dist = pd.DataFrame(cache_dist)
        else:
            self._cache_dist = None

    def calc_bar_xcoords(self):
        """ Work out the x positions of the bars """

        bar_width = self.bar_width
        bar_padding = self.bar_padding
        cat_padding = self.bar_cat_padding

        # Work out the number of bars
        if self.hue_order is None:
            hue_order = [None]
        else:
            hue_order = self.hue_order
        num_hue_cats = len(hue_order)

        # Lay out the data in x/y
        xcoords = []
        xtick_coords = []
        xcoord_map = {}

        # Load data, calculate positions
        x = 0.0
        for i, xcat in enumerate(self.order):
            x += cat_padding
            for j, hcat in enumerate(hue_order):

                xcoords.append(x + bar_width/2.0)
                if hcat is None:
                    xcoord_map[xcat] = xcoords[-1]
                else:
                    xcoord_map[(xcat, hcat)] = xcoords[-1]

                x += bar_width
                if j < len(hue_order)-1:
                    x += bar_padding

            # Work out where the ticks go
            xtick_coords.append(np.mean(xcoords[-num_hue_cats:]))

        self._xcoords = xcoords
        self._xtick_coords = xtick_coords
        self._xcoord_map = xcoord_map
        self.figure_x = x + cat_padding

    def calc_y_subsets(self) -> Generator:
        """ Subset the y data by x and hue """

        if self.hue_order is None:
            hue_order = [None]
        else:
            hue_order = self.hue_order

        data = self._cache_data
        xcolumn = self.xcolumn
        hue_column = self.hue_column

        for xcat in self.order:
            for hcat in hue_order:
                mask = data[xcolumn] == xcat
                if hcat is not None:
                    mask = np.logical_and(mask, data[hue_column] == hcat)
                if ~np.any(mask):
                    if hcat is None:
                        raise ValueError('Got empty category {}'.format(xcat))
                    else:
                        raise ValueError('Got empty dual category {} {}'.format(xcat, hcat))
                yield xcat, hcat, data[mask]

    def calc_bars(self):
        """ Calculate the coordinates needed for bars """

        bar_width = self.bar_width
        bar_padding = self.bar_padding
        cat_padding = self.bar_cat_padding

        # Work out the number of bars
        if self.hue_order is None:
            hue_order = [None]
        else:
            hue_order = self.hue_order
        num_hue_cats = len(hue_order)

        # Look at the data
        data = self._cache_data
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        # Lay out the data in x/y
        xcoords = []
        xtick_coords = []
        ycoords = []
        ylows = []
        yhighs = []
        ynum_samples = []
        coord_map = {}
        bar_hcat = []
        bar_color = []

        # Load data, calculate positions
        x = 0.0
        for i, xcat in enumerate(self.order):
            x += cat_padding
            for j, hcat in enumerate(hue_order):
                mask = data[xcolumn] == xcat
                if hcat is not None:
                    mask = np.logical_and(mask, data[hue_column] == hcat)
                if ~np.any(mask):
                    if hcat is None:
                        raise ValueError('Got empty category {}'.format(xcat))
                    else:
                        raise ValueError('Got empty dual category {} {}'.format(xcat, hcat))
                subdata = data[mask]

                ymean = subdata[ycolumn + ' mean'].values * self._yscale
                ylow = subdata[ycolumn + ' ci low'].values * self._yscale
                yhigh = subdata[ycolumn + ' ci high'].values * self._yscale
                ynum = subdata[ycolumn + ' n'].values

                xcoords.append(x + bar_width/2.0)
                ycoords.extend(ymean)
                ylows.extend(ymean - ylow)
                yhighs.extend(yhigh - ymean)
                ynum_samples.extend(ynum)

                if hcat is None:
                    coord_map[xcat] = (xcoords[-1], ycoords[-1])
                else:
                    coord_map[(xcat, hcat)] = (xcoords[-1], ycoords[-1])

                bar_hcat.append(hcat)
                if self._hue_order is None:
                    bar_color.append(self._color_table[i])
                else:
                    bar_color.append(self._color_table[j])

                x += bar_width
                if j < len(hue_order)-1:
                    x += bar_padding

            # Work out where the ticks go
            xtick_coords.append(np.mean(xcoords[-num_hue_cats:]))

        # Work out where the significance bars go
        ysignificance = []
        if self._sigdata is not None:
            # Fixme: handle hue columns here
            key1 = self.xcolumn + '1'
            key2 = self.xcolumn + '2'
            for i, row in self._sigdata.iterrows():
                coord1 = coord_map[row[key1]]
                coord2 = coord_map[row[key2]]
                pvalue = row['P-value']
                ysignificance.append((coord1[0], coord2[0], pvalue))

        self.figure_x = x + cat_padding
        self._ycoords = np.array(ycoords)
        self._yerr = np.stack([np.array(ylows),
                               np.array(yhighs)], axis=0)
        self._ynum = np.array(ynum_samples)
        if ysignificance != []:
            self._ysig = np.array(ysignificance)
        self._xcoords = np.array(xcoords)
        self._xtick_coords = np.array(xtick_coords)
        self._bar_hcat = bar_hcat
        self._bar_color = bar_color

    def calc_boxes(self):
        """ Calculate the coordinates needed for boxes """

        box_width = self.box_width
        box_padding = self.box_padding
        cat_padding = self.box_cat_padding

        # Work out the number of bars
        if self._hue_order is None:
            hue_order = [None]
        else:
            hue_order = self._hue_order
        num_hue_cats = len(hue_order)

        # Look at the data
        data = self._data
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        # Lay out the data in x/y
        xcoords = []
        xtick_coords = []
        yp5s = []
        yp25s = []
        yp50s = []
        yp75s = []
        yp95s = []
        yerr = []
        ynum_samples = []
        coord_map = {}
        box_hcat = []
        box_color = []

        # Load data, calculate positions
        x = 0.0
        for i, xcat in enumerate(self._order):
            x += cat_padding
            for j, hcat in enumerate(hue_order):
                mask = data[xcolumn] == xcat
                if hcat is not None:
                    mask = np.logical_and(mask, data[hue_column] == hcat)
                if ~np.any(mask):
                    if hcat is None:
                        raise ValueError('Got empty category {}'.format(xcat))
                    else:
                        raise ValueError('Got empty dual category {} {}'.format(xcat, hcat))
                subdata = data[mask]

                yp5 = subdata[ycolumn + ' p5'].values * self._yscale
                yp25 = subdata[ycolumn + ' p25'].values * self._yscale
                yp50 = subdata[ycolumn + ' p50'].values * self._yscale
                yp75 = subdata[ycolumn + ' p75'].values * self._yscale
                yp95 = subdata[ycolumn + ' p95'].values * self._yscale

                yerr.append(yp95-yp50)

                ynum = subdata[ycolumn + ' n'].values

                xcoords.append(x + box_width/2.0)
                yp5s.append(yp5)
                yp25s.append(yp25)
                yp50s.append(yp50)
                yp75s.append(yp75)
                yp95s.append(yp95)
                ynum_samples.extend(ynum)

                if hcat is None:
                    coord_map[xcat] = (xcoords[-1], yp95s[-1])
                else:
                    coord_map[(xcat, hcat)] = (xcoords[-1], yp95s[-1])

                box_hcat.append(hcat)
                if self._hue_order is None:
                    box_color.append(self._color_table[i])
                else:
                    box_color.append(self._color_table[j])

                x += box_width
                if j < len(hue_order)-1:
                    x += box_padding

            # Work out where the ticks go
            xtick_coords.append(np.mean(xcoords[-num_hue_cats:]))

        # Work out where the significance bars go
        ysignificance = []
        if self._sigdata is not None:
            # Fixme: handle hue columns here
            key1 = self.xcolumn + '1'
            key2 = self.xcolumn + '2'
            for i, row in self._sigdata.iterrows():
                coord1 = coord_map[row[key1]]
                coord2 = coord_map[row[key2]]
                pvalue = row['P-value']
                ysignificance.append((coord1[0], coord2[0], pvalue))

        self.figure_x = x + cat_padding
        self._ycoords = np.stack([yp5s, yp25s, yp50s, yp75s, yp95s], axis=1)
        self._yerr = yerr
        self._ynum = np.array(ynum_samples)
        if ysignificance != []:
            self._ysig = np.array(ysignificance)
        self._xcoords = np.array(xcoords)
        self._xtick_coords = np.array(xtick_coords)
        self._box_hcat = box_hcat
        self._box_color = box_color


class Plotter(object):
    """ Load the plot data and regenerate plots """

    def __init__(self,
                 datafile: pathlib.Path,
                 xcolumn: str,
                 ycolumn: str,
                 hue_column: Optional[str] = None,
                 outfile: Optional[pathlib.Path] = None,
                 plot_style: str = PLOT_STYLE,
                 plot_type: str = PLOT_TYPE,
                 palette: str = PALETTE,
                 orient: str = ORIENT,
                 ylimits: Optional[Tuple[Tuple[float]]] = None,
                 scale: Optional[float] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 order: Optional[List[str]] = None,
                 hue_order: Optional[List[str]] = None):
        # Paths
        self.datafile = datafile
        self.outfile = outfile

        # Column selectors
        self.xcolumn = xcolumn
        self.ycolumn = ycolumn
        self.hue_column = hue_column

        # Styles
        self.plot_type = plot_type
        self.plot_style = plot_style
        self.palette = palette
        orient = orient.lower()[0]
        if orient not in ('v', 'h'):
            raise ValueError('Orient must be one of [h]orizontal or [v]ertical')
        self.orient = orient

        self._color_table = None

        # Bar layout
        self.figure_y = 10
        self.figure_x = None

        self.bar_width = 1.0
        self.bar_padding = 0.15
        self.bar_cat_padding = 0.4

        self.box_width = 1.0
        self.box_padding = 0.15
        self.box_cat_padding = 0.4

        # Bar styles
        self.bar_linewidth = 3
        self.bar_edgecolor = 'black'
        self.bar_error_linewidth = 3
        self.bar_error_linecolor = 'black'
        self.bar_error_capsize = 15

        # Text and colors for annotations
        self.num_points_fontsize = 24
        self.num_points_color = {'dark': '#FFFFFF'}.get(self.plot_style, '#000000')

        self.significance_barcolor = {'dark': '#AAAAAA'}.get(self.plot_style, '#666666')
        self.significance_linewidth = 3
        self.significance_fontsize = 28

        # Derived
        self._data = None
        self._distdata = None
        self._sigdata = None

        self._order = order
        self._hue_order = hue_order

        self._xcoords = None
        self._xtick_coords = None
        self._xticklabel_rotation = None
        self._ycoords = None
        self._yerr = None
        self._ynum = None
        self._ysig = None
        self._yscale = scale

        self._bar_hcat = None
        self._bar_color = None

        self._ylimits = ylimits
        self._xlabel = xlabel
        self._ylabel = ylabel

    @property
    def data_distfile(self):
        """ Distribution data associated with a graph """
        datadir = self.datafile.parent
        suffix = self.datafile.suffix
        stem = self.datafile.stem
        return datadir / '{}_dist{}'.format(stem, suffix)

    @property
    def data_sigfile(self):
        """ Significance data associated with a graph """
        datadir = self.datafile.parent
        suffix = self.datafile.suffix
        stem = self.datafile.stem
        return datadir / '{}_sig{}'.format(stem, suffix)

    def read_data_frame(self, datafile: pathlib.Path) -> pd.DataFrame:
        """ Read in a data frame in a data type agnostic way """
        if datafile.suffix == '.csv':
            data = pd.read_csv(str(datafile))
        elif datafile.suffix == '.tsv':
            data = pd.read_csv(str(datafile), sep='\t')
        elif self.datafile.suffix in ('.xls', '.xlsx'):
            data = pd.read_excel(str(datafile))
        else:
            raise ValueError('Unknown data file type: {}'.format(datafile))
        return data

    def load_data(self):
        """ Load the raw data in """
        # Load the original data frame
        data = self.read_data_frame(self.datafile)
        data[self.xcolumn] = data[self.xcolumn].apply(str)
        if self.hue_column is not None:
            data[self.hue_column] = data[self.hue_column].apply(str)
        self._data = data

        # Load the distribution data, if any
        if self.data_distfile.is_file():
            self._distdata = self.read_data_frame(self.data_distfile)

        # Load the significance data
        if self.data_sigfile.is_file():
            sigdata = self.read_data_frame(self.data_sigfile)
            xcol1 = self.xcolumn + '1'
            xcol2 = self.xcolumn + '2'
            if xcol1 in sigdata.columns and xcol2 in sigdata.columns:
                sigdata[xcol1] = sigdata[xcol1].apply(str)
                sigdata[xcol2] = sigdata[xcol2].apply(str)
                self._sigdata = sigdata

        if self._order is None:
            if self.orient == 'v':
                self._order = sorted(np.unique(data[self.xcolumn]), key=StrInt)
            else:
                self._order = sorted(np.unique(data[self.xcolumn]), reverse=True, key=StrInt)
        if self.hue_column is None:
            self._color_table = colorwheel(self.palette, n_colors=len(self._order))
        else:
            if self._hue_order is None:
                self._hue_order = sorted(np.unique(data[self.hue_column]), key=StrInt)
            self._color_table = colorwheel(self.palette, n_colors=len(self._hue_order))

    def load_metadata(self):
        """ Load the metadata for this modality """

        # Y-axis attributes
        if self._ylimits is None:
            self._ylimits = {
                'pct_ifr': [(0.0, 5.0), (65, 100)],
                'pct_irr': [(0.0, 5.0), (65, 100)],
                'IFR': [(0.0, 5.0), (65, 100)],
                'IRR': [(0.0, 5.0), (65, 100)],
                'CellArea': [(0, 50), (200, 550)],  # um^2 - Area of the segmented cell
                'Curl': [(-2, 2)],  # rads/min - How much a cell rotates each frame
                'Density': [(0, 0.25), (1.5, 3.0)],  # um^-2 - Cell density
                'Displacement': [(0, 30)],  # um - How much the cells move in 6 hours
                'Distance': [(0, 30)],  # um - Long the cell track is after 6 hours
                'Divergence': [(-2, 2)],  # ratio - How much the cells spread over 6 hours
                'Persistence': [(-0.1, 1.1)],  # ratio - how much of the cell track is movement
                'Velocity': [(0.0, 0.9)],  # um/min - how fast does the cell move
            }[self.ycolumn]
        if self._ylabel is None:
            self._ylabel = {
                'pct_ifr': '% Matches Between Frames',
                'pct_irr': '% Matches To Consensus',
                'IFR': '% Matches Between Frames',
                'IRR': '% Matches To Consensus',
                'CellArea': 'Cell Area ($\\mu m^2$)',
                'Curl': 'Curl ($rads/min * 10^{-2}$)',
                'Density': 'Cell Density ($\\mu m^{-2} * 10^{-3}$)',
                'Displacement': 'Cell Displacement ($\\mu m$)',
                'Distance': 'Cell Distance ($\\mu m$)',
                'Divergence': 'Divergence ($area/min * 10^{-2}$)',
                'Persistence': 'Cell Persistence',
                'Velocity': 'Velocity Magnitude ($\\mu m/min$)',
            }[self.ycolumn]
        if self._yscale is None:
            self._yscale = {
                'Density': 1000,
                'Curl': 100,
                'Divergence': 100,
            }.get(self.ycolumn, 1.0)

        # X-axis attributes
        if self._xlabel is None:
            self._xlabel = {
                'percent_labeled': 'Percent Labeled',
                'PercentLabeled': 'Percent Labeled',
                'Annotator': 'Annotator',
                'detector': 'Detector',
                'Radius': 'Colony Radius',
                'Media': 'Culture Media',
                'Substrate': 'Substrate',
                'ColonySize': 'Colony Size',
            }[self.xcolumn]
        self._xticklabels = {
            'percent_labeled': lambda o: '{}%'.format(o),
            'PercentLabeled': lambda o: '{}%'.format(o),
            'Annotator': lambda o: o,
            'Media': lambda o: o,
            'Substrate': lambda o: o,
            'ColonySize': lambda o: o,
            'Group': lambda o: o,
        }[self.xcolumn]
        self._xticklabel_rotation = {
            'detector': 90,
            'Radius': 90,
        }.get(self.xcolumn, 0)

        # Force an ordering for specific x-columns
        self._order = {
            'detector': [
                'countception-r003-n50000',
                'fcrn_a_wide-r003-n75000',
                'fcrn_b_wide-r003-n75000',
                'residual_unet-r004-n25000',
                'unet-r001-n50000',
                'composite-d3-final',
            ],
            'Radius': [
                '0.0-0.6',
                '0.6-0.8',
                '0.8-1.0',
            ],
            'Media': [
                'mTeSR',
                'E8',
            ],
            'Substrate': [
                'Matrigel',
                'Vitronectin',
                'Laminin',
            ],
            'ColonySize': [
                '100 Cell',
                '500 Cell',
            ]
        }.get(self.xcolumn, self._order)

    def calc_bars(self):
        """ Calculate the coordinates needed for bars """

        bar_width = self.bar_width
        bar_padding = self.bar_padding
        cat_padding = self.bar_cat_padding

        # Work out the number of bars
        if self._hue_order is None:
            hue_order = [None]
        else:
            hue_order = self._hue_order
        num_hue_cats = len(hue_order)

        # Look at the data
        data = self._data
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        # Lay out the data in x/y
        xcoords = []
        xtick_coords = []
        ycoords = []
        ylows = []
        yhighs = []
        ynum_samples = []
        coord_map = {}
        bar_hcat = []
        bar_color = []

        # Load data, calculate positions
        x = 0.0
        for i, xcat in enumerate(self._order):
            x += cat_padding
            for j, hcat in enumerate(hue_order):
                mask = data[xcolumn] == xcat
                if hcat is not None:
                    mask = np.logical_and(mask, data[hue_column] == hcat)
                if ~np.any(mask):
                    if hcat is None:
                        raise ValueError('Got empty category {}'.format(xcat))
                    else:
                        raise ValueError('Got empty dual category {} {}'.format(xcat, hcat))
                subdata = data[mask]

                ymean = subdata[ycolumn + ' mean'].values * self._yscale
                ylow = subdata[ycolumn + ' ci low'].values * self._yscale
                yhigh = subdata[ycolumn + ' ci high'].values * self._yscale
                ynum = subdata[ycolumn + ' n'].values

                xcoords.append(x + bar_width/2.0)
                ycoords.extend(ymean)
                ylows.extend(ymean - ylow)
                yhighs.extend(yhigh - ymean)
                ynum_samples.extend(ynum)

                if hcat is None:
                    coord_map[xcat] = (xcoords[-1], ycoords[-1])
                else:
                    coord_map[(xcat, hcat)] = (xcoords[-1], ycoords[-1])

                bar_hcat.append(hcat)
                if self._hue_order is None:
                    bar_color.append(self._color_table[i])
                else:
                    bar_color.append(self._color_table[j])

                x += bar_width
                if j < len(hue_order)-1:
                    x += bar_padding

            # Work out where the ticks go
            xtick_coords.append(np.mean(xcoords[-num_hue_cats:]))

        # Work out where the significance bars go
        ysignificance = []
        if self._sigdata is not None:
            # Fixme: handle hue columns here
            key1 = self.xcolumn + '1'
            key2 = self.xcolumn + '2'
            for i, row in self._sigdata.iterrows():
                coord1 = coord_map[row[key1]]
                coord2 = coord_map[row[key2]]
                pvalue = row['P-value']
                ysignificance.append((coord1[0], coord2[0], pvalue))

        self.figure_x = x + cat_padding
        self._ycoords = np.array(ycoords)
        self._yerr = np.stack([np.array(ylows),
                               np.array(yhighs)], axis=0)
        self._ynum = np.array(ynum_samples)
        if ysignificance != []:
            self._ysig = np.array(ysignificance)
        self._xcoords = np.array(xcoords)
        self._xtick_coords = np.array(xtick_coords)
        self._bar_hcat = bar_hcat
        self._bar_color = bar_color

    def calc_boxes(self):
        """ Calculate the coordinates needed for boxes """

        box_width = self.box_width
        box_padding = self.box_padding
        cat_padding = self.box_cat_padding

        # Work out the number of bars
        if self._hue_order is None:
            hue_order = [None]
        else:
            hue_order = self._hue_order
        num_hue_cats = len(hue_order)

        # Look at the data
        data = self._data
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        # Lay out the data in x/y
        xcoords = []
        xtick_coords = []
        yp5s = []
        yp25s = []
        yp50s = []
        yp75s = []
        yp95s = []
        yerr = []
        ynum_samples = []
        coord_map = {}
        box_hcat = []
        box_color = []

        # Load data, calculate positions
        x = 0.0
        for i, xcat in enumerate(self._order):
            x += cat_padding
            for j, hcat in enumerate(hue_order):
                mask = data[xcolumn] == xcat
                if hcat is not None:
                    mask = np.logical_and(mask, data[hue_column] == hcat)
                if ~np.any(mask):
                    if hcat is None:
                        raise ValueError('Got empty category {}'.format(xcat))
                    else:
                        raise ValueError('Got empty dual category {} {}'.format(xcat, hcat))
                subdata = data[mask]

                yp5 = subdata[ycolumn + ' p5'].values * self._yscale
                yp25 = subdata[ycolumn + ' p25'].values * self._yscale
                yp50 = subdata[ycolumn + ' p50'].values * self._yscale
                yp75 = subdata[ycolumn + ' p75'].values * self._yscale
                yp95 = subdata[ycolumn + ' p95'].values * self._yscale

                yerr.append(yp95-yp50)

                ynum = subdata[ycolumn + ' n'].values

                xcoords.append(x + box_width/2.0)
                yp5s.append(yp5)
                yp25s.append(yp25)
                yp50s.append(yp50)
                yp75s.append(yp75)
                yp95s.append(yp95)
                ynum_samples.extend(ynum)

                if hcat is None:
                    coord_map[xcat] = (xcoords[-1], yp95s[-1])
                else:
                    coord_map[(xcat, hcat)] = (xcoords[-1], yp95s[-1])

                box_hcat.append(hcat)
                if self._hue_order is None:
                    box_color.append(self._color_table[i])
                else:
                    box_color.append(self._color_table[j])

                x += box_width
                if j < len(hue_order)-1:
                    x += box_padding

            # Work out where the ticks go
            xtick_coords.append(np.mean(xcoords[-num_hue_cats:]))

        # Work out where the significance bars go
        ysignificance = []
        if self._sigdata is not None:
            # Fixme: handle hue columns here
            key1 = self.xcolumn + '1'
            key2 = self.xcolumn + '2'
            for i, row in self._sigdata.iterrows():
                coord1 = coord_map[row[key1]]
                coord2 = coord_map[row[key2]]
                pvalue = row['P-value']
                ysignificance.append((coord1[0], coord2[0], pvalue))

        self.figure_x = x + cat_padding
        self._ycoords = np.stack([yp5s, yp25s, yp50s, yp75s, yp95s], axis=1)
        self._yerr = yerr
        self._ynum = np.array(ynum_samples)
        if ysignificance != []:
            self._ysig = np.array(ysignificance)
        self._xcoords = np.array(xcoords)
        self._xtick_coords = np.array(xtick_coords)
        self._box_hcat = box_hcat
        self._box_color = box_color

    def add_significance(self, ax, markers=None):
        """ Plot significance bars over the bars """

        if self._ysig is None:
            return

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

        barcolor = self.significance_barcolor
        linewidth = self.significance_linewidth
        fontsize = self.significance_fontsize

        ymax = (np.max(self._ycoords) + np.max(self._yerr))
        ystep = (np.max(self._ycoords) - np.min(self._ycoords)) * 0.05

        for xcoord1, xcoord2, pvalue in self._ysig:
            psymbol = None
            for pthreshold, symbol in sorted_markers:
                if pvalue <= pthreshold:
                    psymbol = symbol
                    break
            if psymbol is None:
                continue
            ymax += ystep
            if self.orient == 'v':
                ax.plot([xcoord1, xcoord2], [ymax, ymax],
                        color=barcolor, linewidth=linewidth, clip_on=False)
                ax.text((xcoord1+xcoord2)/2, ymax+ystep, psymbol,
                        color=barcolor, fontsize=fontsize,
                        horizontalalignment='center',
                        verticalalignment='top',
                        clip_on=False)
            else:
                ax.plot([ymax, ymax], [xcoord1, xcoord2],
                        color=barcolor, linewidth=linewidth, clip_on=False)
                ax.text(ymax+ystep, (xcoord1+xcoord2)/2, psymbol,
                        color=barcolor, fontsize=fontsize,
                        rotation=90,
                        horizontalalignment='right',
                        verticalalignment='center',
                        clip_on=False)

    def add_number_of_samples(self, ax):
        """ Plot the number of samples over the bars """

        color = self.num_points_color
        fontsize = self.num_points_fontsize

        ymax = (np.max(self._ycoords) + np.max(self._yerr))*1.05
        for i, xcoord in enumerate(self._xcoords):
            num_points = self._ynum[i]

            # Label the number of samples
            if self.orient == 'v':
                ax.text(xcoord, ymax, 'n={:d}'.format(num_points),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=fontsize,
                        color=color,
                        clip_on=False)
            else:
                ax.text(ymax, xcoord, 'n={:d}'.format(num_points),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=fontsize,
                        color=color,
                        clip_on=False)

    def plot(self):
        """ Generate the correct plot """
        return getattr(self, 'plot_{}'.format(self.plot_type))()

    def plot_boxes(self):
        """ Plot categories as boxes """

        self.calc_boxes()

        if self.orient == 'v':
            xlimits = [(0, self.figure_x)]
            ylimits = self._ylimits
            figsize = (self.figure_x, self.figure_y)
        else:
            xlimits = self._ylimits
            ylimits = [(0, self.figure_x)]
            figsize = (self.figure_y, self.figure_x)

        with set_plot_style(self.plot_style) as style:
            with SplitAxes(xlimits=xlimits,
                           ylimits=ylimits,
                           figsize=figsize) as ax:
                labeled = set()
                for i in range(self._ycoords.shape[0]):
                    if self._box_hcat[i] not in labeled:
                        label = self._box_hcat[i]
                        labeled.add(label)
                    else:
                        label = None
                    add_single_boxplot(
                        ax=ax,
                        xcoord=self._xcoords[i],
                        ycoords=self._ycoords[i, :],
                        width=self.bar_width,
                        color=self._box_color[i],
                        edgecolor=self.bar_edgecolor,
                        linewidth=self.bar_linewidth,
                        orient=self.orient,
                        label=label, error_kw={
                            'ecolor': self.bar_error_linecolor,
                            'capthick': self.bar_error_linewidth,
                            'capsize': self.bar_error_capsize,
                            'elinewidth': self.bar_error_linewidth,
                        })
                self.add_significance(ax)
                self.add_number_of_samples(ax)

                if self.orient == 'v':
                    ax.set_xticks(self._xtick_coords)
                    ax.set_xticklabels([self._xticklabels(o) for o in self._order])
                    ax.set_xlabel(self._xlabel)
                    ax.set_ylabel(self._ylabel)
                    style.rotate_xticklabels(ax, self._xticklabel_rotation)
                else:
                    ax.set_yticks(self._xtick_coords)
                    ax.set_yticklabels([self._xticklabels(o) for o in self._order])
                    ax.set_ylabel(self._xlabel)
                    ax.set_xlabel(self._ylabel)
                    style.rotate_yticklabels(ax, self._xticklabel_rotation)
                if self._hue_order is not None:
                    ax.legend()
            style.show(outfile=self.outfile)

    def plot_bars(self):
        """ Plot categories as bars """

        self.calc_bars()

        if self.orient == 'v':
            xlimits = [(0, self.figure_x)]
            ylimits = self._ylimits
            figsize = (self.figure_x, self.figure_y)
        else:
            xlimits = self._ylimits
            ylimits = [(0, self.figure_x)]
            figsize = (self.figure_y, self.figure_x)

        with set_plot_style(self.plot_style) as style:
            with SplitAxes(xlimits=xlimits,
                           ylimits=ylimits,
                           figsize=figsize) as ax:
                labeled = set()
                for i in range(self._ycoords.shape[0]):
                    if self._bar_hcat[i] not in labeled:
                        label = self._bar_hcat[i]
                        labeled.add(label)
                    else:
                        label = None
                    add_single_barplot(
                        ax=ax,
                        xcoord=self._xcoords[i],
                        ycoords=[self._ycoords[i], self._yerr[i, 0], self._yerr[i, 1]],
                        width=self.bar_width,
                        color=self.bar_color[i],
                        edgecolor=self.bar_edgecolor,
                        linewidth=self.bar_linewidth,
                        orient=self.orient,
                        label=label,
                        error_kw={
                            'ecolor': self.bar_error_linecolor,
                            'capthick': self.bar_error_linewidth,
                            'capsize': self.bar_error_capsize,
                            'elinewidth': self.bar_error_linewidth,
                        })
                self.add_significance(ax)
                self.add_number_of_samples(ax)

                if self.orient == 'v':
                    ax.set_xticks(self._xtick_coords)
                    ax.set_xticklabels([self._xticklabels(o) for o in self._order])
                    ax.set_xlabel(self._xlabel)
                    ax.set_ylabel(self._ylabel)
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(self._xticklabel_rotation)
                else:
                    ax.set_yticks(self._xtick_coords)
                    ax.set_yticklabels([self._xticklabels(o) for o in self._order])
                    ax.set_ylabel(self._xlabel)
                    ax.set_xlabel(self._ylabel)
                    for tick in ax.get_yticklabels():
                        tick.set_rotation(self._xticklabel_rotation)
                if self._hue_order is not None:
                    ax.legend()
            style.show(outfile=self.outfile)

    def plot_traces(self):
        """ Plot categories as a trace """

        figure_x = self.bar_width * 2 * len(self._order) + self.bar_width
        figsize = (figure_x, self.figure_y)

        data = self._data
        xcolumn = self.xcolumn
        ycolumn = self.ycolumn
        hue_column = self.hue_column

        with set_plot_style(self.plot_style) as style:
            with SplitAxes(ylimits=self._ylimits,
                           figsize=figsize) as ax:
                for j, hcat in enumerate(self._hue_order):
                    ymeans = []
                    ylows = []
                    yhighs = []
                    xcoords = []

                    for i, xcat in enumerate(self._order):
                        mask = np.logical_and(data[xcolumn] == xcat,
                                              data[hue_column] == hcat)
                        if ~np.any(mask):
                            continue
                        subdata = data[mask]

                        ymean = subdata[ycolumn + ' mean'].values * self._yscale
                        ylow = subdata[ycolumn + ' ci low'].values * self._yscale
                        yhigh = subdata[ycolumn + ' ci high'].values * self._yscale

                        xcoords.append(i)
                        ymeans.extend(ymean)
                        ylows.extend(ymean - ylow)
                        yhighs.extend(yhigh - ymean)

                    ymeans = np.array(ymeans)
                    yerr = np.stack([np.array(ylows),
                                     np.array(yhighs)], axis=0)
                    xcoords = np.array(xcoords)
                    color = self._color_table[j]

                    ax.errorbar(
                        xcoords, ymeans, yerr,
                        capsize=self.bar_error_capsize,
                        capthick=self.bar_error_linewidth,
                        elinewidth=self.bar_error_linewidth,
                        linewidth=self.bar_error_linewidth,
                        label=hcat,
                        color=color,
                        ecolor=color)
                ax.set_xticks(range(len(self._order)))
                ax.set_xticklabels([self._xticklabels(o) for o in self._order])
                ax.set_xlabel(self._xlabel)
                ax.set_ylabel(self._ylabel)
                ax.legend()
            style.show(outfile=self.outfile)
