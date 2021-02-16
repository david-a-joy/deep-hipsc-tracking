""" Plotting utilities

* :py:func:`~get_layout`: Calculate useful grid layouts
* :py:func:`~bootstrap_ci`: Calculate bootstrap confidence intervals for line plots

Compound plotting functions:

* :py:func:`~add_colorbar`: Add a colorbar to an axis
* :py:func:`~add_histogram`: Add a histogram with kernel and model fits
* :py:func:`~add_gradient_line`: Add a line with a color gradient
* :py:func:`~add_meshplot`: Plot a mesh of points and links
* :py:func:`~add_poly_meshplot`: Plot a mesh with filled polygons
* :py:func:`~add_scalebar`: Add a scale bar to an image plot

API Documentation
-----------------

"""

# Standard lib
from typing import Tuple, List, Optional, Dict, Callable

# 3rd party imports
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.font_manager
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.patches import Polygon

from scipy.stats import gamma, gaussian_kde
from scipy.integrate import simps


# Utility functions


def bootstrap_ci(data: np.ndarray,
                 n_boot: int = 1000,
                 n_samples: Optional[int] = None,
                 random_seed: Optional[int] = None,
                 ci: float = 95,
                 func: Callable = np.mean,
                 axis: int = 0) -> Tuple[np.ndarray]:
    """ Calculate a confidence interval from the input data using bootstrapping

    :param ndarray data:
        The data to bootstrap sample
    :param int n_boot:
        Number of times to sample the frame
    :param int n_samples:
        If not None, the number of samples to draw for each round (default: length of the sample axis)
    :param int random_seed:
        Seed for the random number generator
    :param float ci:
        Confidence interval to calculate (mean +/- ci/2.0)
    :param Callable func:
        Function to calculate the ci around (default: np.mean)
    :param int axis:
        Which axis to sample over
    :returns:
        The upper and lower bounds on the CI
    """
    if n_samples is None:
        n_samples = data.shape[axis]
    rs = np.random.RandomState(random_seed)
    boot_dist = []
    for i in range(n_boot):
        resampler = rs.randint(0, data.shape[axis], n_samples)
        sample = data.take(resampler, axis=axis)
        boot_dist.append(func(sample, axis=axis))
    boot_dist = np.array(boot_dist)
    return np.percentile(boot_dist, [50 - ci/2, 50 + ci/2], axis=0)


def get_layout(sequence: List, max_columns: float = 8.0) -> Tuple[int]:
    """ Get a rectangular layout that will have one subplot for each element in sequence

    :param list sequence:
        An iterable with a length, or a number representing the size of an iterable
    :param float max_columns:
        Limit the number of columns to this many
    :returns:
        n_rows, n_cols, the number of rows and columns to call plt.subplots with
    """
    # Work out the axes layout
    if hasattr(sequence, '__len__'):
        sequence = float(len(sequence))
    else:
        sequence = float(sequence)
    n_cols = np.ceil(np.sqrt(sequence))
    n_cols = min((float(max_columns), n_cols))
    n_rows = np.ceil(sequence / n_cols)
    return int(n_rows), int(n_cols)


def get_histogram(data: np.ndarray,
                  bins: int,
                  range: Optional[Tuple[float]] = None,
                  kernel_smoothing: bool = True,
                  kernel_bandwidth: Optional[str] = None,
                  kernel_samples: int = 100,
                  normalize: bool = False) -> Tuple[np.ndarray]:
    """ Get a histogram and a kernel fit for some data

    :param ndarray data:
        The data to fit
    :param int bins:
        The number of bins to generate
    :param tuple[float] range:
        The range to fit bins to (argument to np.histogram)
    :param bool kernel_smoothing:
        If True, also generate a kernel-smoothed fit. If False, xkernel, ykernel are None
    :param str kernel_bandwidth:
        If not None, the method to use to estimate the kernel smoothed fit
    :param int kernel_samples:
        The number of samples to draw for the kernel fit
    :param bool normalize:
        If True, scale everything to total area == 1.0
    :returns:
        xbins, ybins, xkernel, ykernel
    """
    bins_y, bins_x = np.histogram(data, bins=bins, range=range)
    bin_width = bins_x[1:] - bins_x[:-1]
    hist_area = np.sum(bin_width * bins_y)

    # Normalize the counts
    if normalize:
        bins_y = bins_y / hist_area
        hist_area = 1.0

    # Estimate the kernel smoothed fit
    if kernel_smoothing:
        kernel = gaussian_kde(data, bw_method=kernel_bandwidth)
        kernel_x = np.linspace(bins_x[0], bins_x[-1], kernel_samples)
        kernel_y = kernel(kernel_x)

        # Rescale for equal areas
        kernel_area = simps(kernel_y, kernel_x)
        kernel_y = kernel_y * hist_area / kernel_area
    else:
        kernel_x = kernel_y = None
    return bins_x, bins_y, kernel_x, kernel_y


# Plot Functions


def add_colorbar(ax: Axes,
                 im: AxesImage,
                 orientation: str = 'vertical',
                 append: Optional[str] = None):
    """ Attach a colorbar to an axis

    Basic Usage:

    .. code-block:: python

        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(np.random.rand(64, 64), cmap='inferno')
        add_colorbar(ax, im,
                     orientation='vertical',
                     append='left')

    Note that the colormap is taken from the ``cmap`` argument in ``imshow``

    For a useful table of colormaps, see the
    `Matplotlib Colormaps <http://matplotlib.org/examples/color/colormaps_reference.html>`_

    :param Axes ax:
        The matplotlib axis to add the colorbar for
    :param AxesImage im:
        The image data (returned from ``im = ax.imshow()``)
    :param str orientation:
        Either horizontal or vertical
    :param str append:
        Any of 'right', 'left', 'top', 'bottom'
    """

    if append is None:
        if orientation == 'vertical':
            append = 'right'
        elif orientation == 'horizontal':
            append = 'bottom'

    divider = make_axes_locatable(ax)
    fig = ax.figure
    cax = divider.append_axes(append, size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation=orientation)


def add_histogram(ax: Axes,
                  data: np.ndarray,
                  xlabel: Optional[str] = None,
                  ylabel: str = 'Counts',
                  title: str = None,
                  bins: int = 10,
                  draw_bars: bool = True,
                  bar_width: float = 0.7,
                  range: Optional[Tuple[float]] = None,
                  fit_dist: Optional[str] = None,
                  fit_dist_color: str = 'r',
                  kernel_smoothing: bool = True,
                  kernel_label: Optional[str] = None,
                  label_kernel_peaks: Optional[str] = None,
                  kernel_smoothing_color: str = 'c',
                  kernel_bandwidth: Optional[str] = None,
                  vlines: Optional[List[float]] = None,
                  vline_colors: Optional[List[str]] = 'b',
                  normalize: bool = False):
    """ Add a histogram plot

    Basic Usage:

    .. code-block:: python

        fig, ax = plt.subplots(1, 1)
        histogram(ax, np.random.rand(64, 64),
                  draw_bars=True,
                  kernel_smoothing=True,
                  fit_dist='poisson',
                  vlines=[0.25, 0.75])

    This will draw the histogram with a kernel smoothed fit, a poisson fit,
    and vertical lines at x coordinates 0.25 and 0.75.

    :param Axes ax:
        The axis to add the histogram to
    :param ndarray data:
        The data to make the histogram for
    :param str xlabel:
        Label for the x axis
    :param str ylabel:
        Label for the y axis
    :param str title:
        Title for the axis
    :param int bins:
        Number of bins in the histogram
    :param bool draw_bars:
        If True, draw the histogram bars
    :param float bar_width:
        The width of the bars to plot
    :param tuple[float] range:
        The range to fit bins to (argument to np.histogram)
    :param str fit_dist:
        The name of a distribution to fit to the data
    :param str fit_dist_color:
        The color of the fit dist line
    :param bool kernel_smoothing:
        If True, plot the kernel smoothed line over the bars
    :param str kernel_label:
        If not None, label for the kernel smoothed line
    :param str label_kernel_peaks:
        Any of min, max, both to label extrema in the kernel
    :param str kernel_smoothing_color:
        The color of the kernel smoothed fit line
    :param str kernel_bandwidth:
        The method to calculate the kernel width with
    :param list[float] vlines:
        x coords to draw vertical lines at
    :param list[str] vline_colors:
        The color or list of colors for the spectra
    :param bool normalize:
        If True, normalize the counts/KDE/models
    """

    # Estimate the histogram
    data = data[np.isfinite(data)]

    xbins, hist, kernel_x, kernel_y = get_histogram(
        data, bins=bins, range=range,
        kernel_smoothing=kernel_smoothing,
        kernel_bandwidth=kernel_bandwidth,
        normalize=normalize)

    width = bar_width * (xbins[1] - xbins[0])
    center = (xbins[:-1] + xbins[1:])/2

    # Add bars for the histogram
    if draw_bars:
        ax.bar(center, hist, align='center', width=width)

    # Estimate the kernel smoothed fit
    if kernel_smoothing:
        # Add a kernel smoothed fit
        ax.plot(kernel_x, kernel_y, color=kernel_smoothing_color,
                label=kernel_label)

        if label_kernel_peaks in ('max', 'both', True):
            maxima = (np.diff(np.sign(np.diff(kernel_y))) < 0).nonzero()[0] + 1
            kx_maxima = kernel_x[maxima]
            ky_maxima = kernel_y[maxima]

            ax.plot(kx_maxima, ky_maxima, 'oc')
            for kx, ky in zip(kx_maxima, ky_maxima):
                ax.text(kx, ky*1.05, "{}".format(float(f"{kx:.2g}")),
                        color="c", fontsize=12)

        if label_kernel_peaks in ('min', 'both', True):
            minima = (np.diff(np.sign(np.diff(kernel_y))) > 0).nonzero()[0] + 1
            kx_minima = kernel_x[minima]
            ky_minima = kernel_y[minima]

            ax.plot(kx_minima, ky_minima, 'oy')
            for kx, ky in zip(kx_minima, ky_minima):
                ax.text(kx, ky*0.88, "{}".format(float(f"{kx:.2g}")),
                        color="y", fontsize=12)

    # Fit an model distribution to the data
    if fit_dist is not None:
        opt_x = np.linspace(xbins[0], xbins[-1], 100)

        if fit_dist == 'gamma':
            fit_alpha, fit_loc, fit_beta = gamma.fit(data + 1e-5)
            # print(fit_alpha, fit_loc, fit_beta)
            opt_y = data = gamma.pdf(opt_x, fit_alpha, loc=fit_loc, scale=fit_beta) * data.shape[0]
        else:
            raise KeyError(f'Unknown fit distribution: {fit_dist}')

        ax.plot(opt_x, opt_y, fit_dist_color)

    # Add spectral lines
    if vlines is None:
        vlines = []
    if isinstance(vlines, (int, float)):
        vlines = [vlines]
    if isinstance(vline_colors, (str, tuple)):
        vline_colors = [vline_colors for _ in vlines]

    if len(vlines) != len(vline_colors):
        raise ValueError(f'Number of colors and lines needs to match: {vlines} vs {vline_colors}')

    ymin, ymax = ax.get_ylim()
    for vline, vline_color in zip(vlines, vline_colors):
        ax.vlines(vline, ymin, ymax, colors=vline_color)

    # Label the axes
    if xlabel not in (None, ''):
        ax.set_xlabel(xlabel)
    if ylabel not in (None, ''):
        ax.set_ylabel(ylabel)
    if title not in (None, ''):
        ax.set_title(f'{title} (n={data.shape[0]})')
    else:
        ax.set_title(f'n = {data.shape[0]}')


def add_gradient_line(ax: Axes,
                      x: np.ndarray,
                      y: np.ndarray,
                      v: np.ndarray,
                      cmap: str = 'gist_rainbow',
                      linewidth: float = 2,
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None):
    """ Add a set of lines with a colormap based on the coordinates

    :param Axes ax:
        The axis to add a line to
    :param ndarray x:
        The n point x coordinates
    :param ndarray y:
        The n point y coordinates
    :param ndarray v:
        The n point color matrix to plot
    :param str cmap:
        The matplotlib colormap to use
    :param int linewidth:
        The line width to plot
    :param float vmin:
        The minimum value for the color map
    :param float vmax:
        The maximum value for the color map
    """
    coords = np.stack([x, y], axis=1)
    v = np.squeeze(v)

    assert coords.ndim == 2
    assert coords.shape[1] == 2
    assert coords.shape[0] > 1
    if v.ndim != 1:
        raise ValueError(f'Expected 1D colors but got shape {v.shape}')

    if v.shape[0] != coords.shape[0]:
        raise ValueError(f'Got coords with shape {coords.shape} but colors with shape {v.shape}')

    if vmin is None:
        vmin = np.min(v)
    if vmax is None:
        vmax = np.max(v)

    # Convert from vertex-centric to edge-centric
    coords = coords[:, np.newaxis, :]
    stack_coords = np.stack([coords[:-1, 0, :], coords[1:, 0, :]], axis=1)

    assert stack_coords.shape == (coords.shape[0]-1, 2, 2)

    # Convert each segment to a color on the gradient
    cm = plt.get_cmap(cmap)
    cnorm = mplcolors.Normalize(vmin=vmin, vmax=vmax)
    scalar_map = mplcm.ScalarMappable(norm=cnorm, cmap=cm)

    index = (v[:-1] + v[1:])/2

    coll = LineCollection(stack_coords,
                          colors=scalar_map.to_rgba(index),
                          linewidth=linewidth)
    ax.add_collection(coll)
    return ax


def add_meshplot(ax: Axes,
                 points: np.ndarray,
                 mesh: Dict[int, List[int]],
                 linewidth: float = 1,
                 markersize: float = 2,
                 color: str = 'r',
                 rasterized: bool = True):
    """ Add a plot of the mesh

    :param Axes ax:
        The axis to add a meshplot to
    :param ndarray points:
        The n x 2 array of points
    :param dict[int, list] mesh:
        The dictionary mapping point indices to their neighbors in the mesh
    """

    points = np.array(points)

    meshlines = []

    # Merge all the links into a line collection
    for i, point in enumerate(points):
        if i not in mesh:
            continue
        connected_points = mesh[i]
        for j in connected_points:
            meshlines.append(np.array([
                [points[i, 0], points[i, 1]],
                [points[j, 0], points[j, 1]],
            ]))
    if rasterized:
        ax.set_rasterization_zorder(2.1)
    collection = LineCollection(meshlines, colors=color, linewidths=linewidth)
    collection.set_zorder(2)
    ax.add_collection(collection)
    ax.plot(points[:, 0], points[:, 1], linestyle='', marker='o', color=color, markersize=markersize,
            zorder=0)
    return ax


def add_poly_meshplot(ax: Axes,
                      points: np.ndarray,
                      triangles: List[Tuple[int]],
                      values: np.ndarray,
                      vmax: Optional[float] = None,
                      vmin: Optional[float] = None,
                      cmap: str = 'coolwarm',
                      rasterized: bool = True):
    """ Add meshes with faces colored by the values

    :param Axes ax:
        The axis to add a meshplot to
    :param ndarray points:
        The n x 2 array of points
    :param list[tuple[int]]:
        The set of all perimeter indicies for this mesh
    :param ndarray values:
        The values to color the perimeters with
    """
    if vmin is None:
        if isinstance(values, dict):
            vmin = np.percentile(list(values.values()), 10)
        else:
            vmin = np.percentile(list(values), 10)
    if vmax is None:
        if isinstance(values, dict):
            vmax = np.percentile(list(values.values()), 90)
        else:
            vmax = np.percentile(list(values), 90)
    cmap = mplcm.get_cmap(cmap)
    norm = mplcm.colors.Normalize(vmax=vmax, vmin=vmin)
    scores = []
    patches = []

    points = np.array(points)
    max_idx = points.shape[0]

    for indices in triangles:
        tri = []
        score = []
        for i in indices:
            if i < 0 or i > max_idx:
                continue
            tri.append(points[i, :])
            score.append(values[i])
        if len(tri) < 3:
            continue
        mean = np.nanmean(score)
        if np.isnan(mean):
            continue
        scores.append(norm(mean))
        patch = Polygon(tri, closed=True, edgecolor='none')
        patches.append(patch)

    colors = cmap(scores)

    collection = PatchCollection(patches)
    ax.add_collection(collection)
    collection.set_color(colors)
    return ax


def add_scalebar(ax: Axes,
                 img_shape: Tuple[int],
                 space_scale: float,
                 bar_len: Optional[float] = None,
                 bar_text: Optional[str] = None,
                 bar_color: Optional[str] = 'w',
                 bar_linewidth: float = 10,
                 fontsize: Optional[float] = None):
    """ Add a scale bar to an image

    :param Axes ax:
        The axis to add a meshplot to
    :param tuple[int, int] img_shape:
        Rows, columns of the image to add a scale bar to
    :param float space_scale:
        The width/pixels (in um)
    :param float bar_len:
        The bar length (in um) to use, or None to guess one
    :param str bar_text:
        The text to write on the bar or None to autogenerate
    :param str bar_color:
        The color for the bar and the bar text
    :param float bar_linewidth:
        The thickness of the bar
    :param float fontsize:
        The fontsize for the text
    """

    rows, cols = img_shape
    total_len = cols * space_scale
    if fontsize is None:
        fontsize = bar_linewidth * 5

    if bar_len is None:
        # Try and find a *nice* bar length that's between 1/3rd and 1/5th of the plot
        min_len = total_len / 5
        max_len = total_len / 3

        min_pow10 = int(np.floor(np.log10(min_len)))
        max_pow10 = int(np.ceil(np.log10(max_len)))
        for pow10 in range(min_pow10, max_pow10+1):
            for nice_step in [1, 2, 5, 4, 8]:
                nice_len = nice_step * (10 ** pow10)
                if min_len <= nice_len <= max_len:
                    bar_len = nice_len
                    break

        if bar_len is None:
            # FIXME: Make this slightly better
            bar_len = total_len / 4

    if bar_text is None:
        bar_text = f'{bar_len:0.0f} μm'

    # Work out the fraction of the plot that the bar takes up
    bar_frac = bar_len / total_len

    bar_ypos = rows * 0.95
    bar_ypos_text = rows * 0.93

    bar_xed = cols * 0.95
    bar_xst = cols * (0.95 - bar_frac)

    print(f'Bar size: {bar_xed - bar_xst} pixels')

    ax.plot([bar_xst, bar_xed], [bar_ypos, bar_ypos],
            linestyle='-', color=bar_color, linewidth=bar_linewidth,
            clip_on=False,
            in_layout=False)
    # Scale text on the first plot
    if bar_text not in ('', None):
        print(f'Bar label: {bar_text}')
        print(f'Font size: {fontsize}')
        ax.text((bar_xst+bar_xed)/2, bar_ypos_text, bar_text,
                horizontalalignment='center', verticalalignment='baseline',
                color=bar_color, fontsize=fontsize,
                clip_on=False,
                in_layout=False)
    return ax


def get_font_families() -> List:
    """ List all the installed fonts

    :returns:
        A list of font names for use in e.g. `matplotlib.rc('font', **font)`
    """
    fnames = []
    for font in matplotlib.font_manager.get_fontconfig_fonts():
        try:
            fname = matplotlib.font_manager.FontProperties(fname=font).get_name()
            fnames.append(fname)
        except Exception:
            pass
    return fnames


def set_radial_ticklabels(ax: Axes,
                          axis: str = 'x',
                          label: str = 'Normalized Radius',
                          mirror: bool = True,
                          labelpad: Optional[float] = None,
                          tickpad: Optional[float] = None,
                          ticks: List[float] = None):
    """ Set the ranges and labels for the axis

    :param Axes ax:
        The axis object to add ticklabels to
    :param str axis:
        Which of x or y axis to add the ticks to
    :param str label:
        Label to apply to the axis
    :param bool mirror:
        If True, reflect labels across x/y == 0.0
    :param float labelpad:
        Padding for the labels
    :param float tickpad:
        Padding for the ticks
    """
    if ticks is None:
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ticks = list(sorted(t for t in ticks if t >= 0.0 and t <= 1.0))

    # If the axis is mirrored, set reflected bounds
    if mirror:
        limits = [-1.01, 1.01]
        major_ticks = []
        major_labels = []

        left_ticks = [-t for t in ticks[::-1]]
        if left_ticks[-1] > -0.01:
            left_ticks = left_ticks[:-1]
        minor_ticks = left_ticks + ticks
    else:
        # Otherwise just go from center to edge
        limits = [0.0, 1.01]
        major_ticks = []
        major_labels = []

        minor_ticks = ticks

    # Format the ticks to minimize trailing zeros, but always make a decimal number
    minor_labels = [f"{np.abs(t):0.2f}".rstrip('0') for t in minor_ticks]
    minor_labels = [f"{t}0" if t.endswith('.') else t for t in minor_labels]

    # Matplotlib weirdness means minor ticks don't work in axis3d
    ticks = major_ticks + minor_ticks
    labels = major_labels + minor_labels

    # Set all the labels with axis specific methods
    if axis == 'x':
        ax.set_xlabel(label, labelpad=labelpad)
        ax.set_xlim(limits)

        ax.set_xticks(ticks, minor=False)
        ax.set_xticklabels(labels, minor=False)
    elif axis == 'y':
        ax.set_ylabel(label, labelpad=labelpad)

        ax.set_ylim(limits)

        ax.set_yticks(ticks, minor=False)
        ax.set_yticklabels(labels, minor=False)
    else:
        raise KeyError(f'Unknown axis "{axis}"')
    if tickpad is not None:
        ax.tick_params(axis=axis, pad=tickpad)
    return ax
