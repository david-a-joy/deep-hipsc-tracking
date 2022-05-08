""" Plots for examining the tracking data

Plot functions:

* :py:func:`plot_top_k_waveforms`: Plot the k longest waveforms
* :py:func:`plot_all_tracks`: Plot the traces for whole colonies
* :py:func:`plot_dist_vs_displacement`: Plot the distance vs displacement graph

API Documentation
-----------------

"""

# Imports
import pathlib
from typing import Optional, Tuple, List, Callable

# 3rd party
import numpy as np

import matplotlib.pyplot as plt

# Our own imports
from .consts import PLOT_STYLE, FIGSIZE, LINEWIDTH, MARKERSIZE, PALETTE
from .styling import set_plot_style, colorwheel


# Plot Functions


def plot_all_tracks(final_chains,
                    plotfile: Optional[pathlib.Path] = None,
                    title: Optional[str] = None,
                    image: Optional[np.ndarray] = None,
                    image_cmap: str = 'gray',
                    track_style: str = 'tracks',
                    rows: int = 1000,
                    cols: int = 1000,
                    max_tracks: int = -1,
                    max_velocity: float = 50,
                    linewidth: float = 1,
                    plot_style: str = PLOT_STYLE,
                    figsize: Tuple[float, float] = FIGSIZE,
                    rasterized: bool = True):
    """ Plot all the tracks in one graphic

    :param list[Link] final_chains:
        The list of final track Link objects
    :param str plotfile:
        File to save the plot to
    :param str title:
        The title to save the track with
    :param ndarray image:
        If not None, the image to save the plot with
    :param str track_style:
        One of 'tracks' or 'arrows', how to plot the traces
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    num_tracks = 0
    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if rasterized:
            ax.set_rasterization_zorder(1)
        if image is not None:
            ax.imshow(image, cmap=image_cmap, zorder=-1)
        for i, chain in enumerate(sorted(final_chains, key=lambda c: len(c), reverse=True)):
            if max_tracks > 0 and num_tracks > max_tracks:
                break
            if np.max(chain.vel_mag()) > max_velocity:
                continue
            if track_style == 'tracks':
                ax.plot(chain.line_x, chain.line_y,
                        linewidth=linewidth,
                        color=colors[i % len(colors)],
                        zorder=0)
            elif track_style == 'arrows':
                ax.arrow(chain.line_x[0], chain.line_y[0],
                         chain.line_x[-1]-chain.line_x[0],
                         chain.line_y[-1]-chain.line_y[0],
                         linewidth=linewidth,
                         color=colors[i % len(colors)],
                         head_width=3,
                         zorder=0)
            else:
                raise KeyError(f'Unknown track style: "{track_style}"')

            num_tracks += 1

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([0, cols])
        ax.set_ylim([rows, 0])
        ax.set_axis_off()
        ax.set_aspect('equal')

        if title is not None:
            fig.suptitle(f'Tracks for {title}')
        style.show(outfile=plotfile, transparent=True)


def plot_top_k_waveforms(timelines: List[np.ndarray],
                         waveforms: List[np.ndarray],
                         topk: Optional[int] = None,
                         plot_style: str = PLOT_STYLE,
                         palette: str = PALETTE,
                         figsize: Tuple[int] = FIGSIZE,
                         xlabel: str = 'Time (mins)',
                         ylabel: str = 'Persistence Velocity ($\\mu$m/min)',
                         outfile: Optional[pathlib.Path] = None,
                         step_padding: float = 0.2,
                         trace_linewidth: float = 2.5,
                         zero_masked_regions: bool = True,
                         max_timepoint: Optional[float] = None,
                         min_timepoint: Optional[float] = None,
                         plot_max_timepoint_line: bool = True,
                         sort_key: Optional[Callable] = None,
                         wave_order: str = 'top_to_bottom'):
    """ Plot the waveforms as a trace plot

    :param list[ndarray] timelines:
        Either the shared timeline for all the tracks, or a list of timelines, one for each track
    :param list[ndarray] waveforms:
        The list of magnitudes for each track
    :param int topk:
        If not None, the maximum number of tracks to plot (default is plot all)
    :param str plot_style:
        The stylesheet to use for the plot
    :param tuple[int] figsize:
        The figure size for the plot
    :param str xlabel:
        Label for the trace x-axis
    :param str ylabel:
        Label for the trace y-axis
    :param Path outfile:
        The path to the file to save the plot to
    :param float step_padding:
        Pad each trace by this amount, as a fraction of the maximum trace height
    :param bool zero_masked_regions:
        If True, replace np.nan waveform values with zero
    :param bool plot_max_timepoint_line:
        If True, plot a red line marking the max timepoint
    :param float max_timepoint:
        The maximum timepoint, or None to plot all timepoints
    :param float min_timepoint:
        The minimum timepoint, or None to plot all timepoints
    :param callable sort_key:
        The function to sort the waveforms by. It takes a tuple of (timeline, waveform)
        and returns a value to rank each waveform. HIGHER values are plotted at the top,
        and LOWER values are plotted at the bottom.

        Example: order by wave height (the default):

        .. code-block:: python

            def sort_key(time_wave):
                _, wave = time_wave
                return np.nanmax(wave) - np.nanmin(wave)

        Example: order by track duration:

        .. code-block:: python

            def sort_key(time_wave):
                timepoint, _ = time_wave
                return np.max(timepoint) - np.min(timepoint)

    :param str wave_order:
        Either 'top_to_bottom' or 'bottom_to_top'. How should the HIGHEST to
        LOWEST values be plotted (i.e. either HIGHEST on top to LOWEST on bottom
        or vice versa)
    """
    # If we only got one waveform, make it a list of waveforms
    if isinstance(waveforms, np.ndarray):
        waveforms = [waveforms]
    # If we only got one timeline, expand it to the length of all the waveforms
    if isinstance(timelines, np.ndarray):
        timelines = [timelines for _ in waveforms]
    # Make sure we got the same number of times as waveforms
    assert len(timelines) == len(waveforms)

    # For empty k, take all the timepoints
    if topk is None or topk < 1:
        topk = len(timelines)
    # For empty sort key, take the k-largest traces
    if sort_key is None:
        def sort_key(time_wave):
            _, wave = time_wave
            return np.nanmax(wave) - np.nanmin(wave)

    # Sort the tracks and take the k-longest
    sorted_timelines = []
    sorted_waveforms = []
    waveform_step = []
    waveform_baseline = []
    for timeline, waveform in sorted(zip(timelines, waveforms), key=sort_key, reverse=True):
        if len(timeline) < 2 or len(waveform) < 2:
            continue
        if len(sorted_timelines) >= topk:
            break
        sorted_timelines.append(timeline)
        sorted_waveforms.append(waveform)

        # Work out the wave extents so we can keep them from overlapping
        wave_max = np.nanmax(waveform)
        wave_min = np.nanmin(waveform)
        waveform_step.append(wave_max - wave_min)
        waveform_baseline.append(wave_min)

    # Work out the step size for each wave
    step = np.nanmax(waveform_step)
    padded_step = step * (1.0 + step_padding)
    baseline = np.nanmin(waveform_baseline)

    min_time = np.nanmin(np.concatenate(sorted_timelines))
    max_time = np.nanmax(np.concatenate(sorted_timelines))

    # Reverse the waves if we plot "top_to_bottom"
    if wave_order.lower().startswith('top'):
        sorted_timelines.reverse()
        sorted_waveforms.reverse()
    elif not wave_order.lower().startswith('bottom'):
        # Bottom to top doesn't need any reversing
        raise KeyError(f'Unknown wave order: {wave_order}')

    palette = colorwheel(palette, n_colors=len(sorted_timelines))

    # Actually plot the traces, now that we've worked out the geometry
    yticks = []
    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for i, (timeline, waveform) in enumerate(zip(sorted_timelines, sorted_waveforms)):

            # Mask timepoint boundaries if requested
            if max_timepoint is None:
                tp_mask = np.ones_like(timeline, dtype=bool)
            else:
                tp_mask = timeline <= max_timepoint
            if min_timepoint is not None:
                tp_mask = np.logical_and(tp_mask, timeline >= min_timepoint)

            # Fill the masked values out with 0s
            if zero_masked_regions:
                waveform[np.isnan(waveform)] = 0
            # Remove the baseline so we can plot everything nicely
            waveform -= baseline

            # Only plot if we get at least two real values
            if np.sum(tp_mask) > 1:
                color = palette[i]
                ax.plot(timeline[tp_mask] - min_time, waveform[tp_mask] + i*padded_step,
                        color=color, linewidth=trace_linewidth, linestyle='-')
            yticks.append((i+0.5)*padded_step)

        # Plot a max_timepoint line to mark the max
        if plot_max_timepoint_line and max_timepoint is not None:
            ax.axvline(x=max_timepoint, ymin=0, ymax=1, color='red')

        # Constant size for waveform comparing
        ax.set_xlim([min_time, max_time])
        ax.set_ylim([-step*step_padding, len(sorted_timelines)*padded_step + step*step_padding])

        # If we get too many ticks, just blank the ticklabels
        if len(yticks) > 20:
            ax.set_yticks([])
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels(['' for _ in yticks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        style.show(outfile=outfile, transparent=True)


def plot_dist_vs_displacement(track_distances: np.ndarray,
                              track_displacements: np.ndarray,
                              track_class: np.ndarray,
                              outfile: Optional[pathlib.Path] = None,
                              plot_style: str = PLOT_STYLE,
                              figsize: Tuple[float] = FIGSIZE,
                              linewidth: float = LINEWIDTH,
                              markersize: float = MARKERSIZE,
                              cmap: Optional[str] = None,
                              colors: Optional[Tuple] = None,
                              vmin: Optional[float] = None,
                              vmax: Optional[float] = None,
                              xmin: Optional[float] = None,
                              xmax: Optional[float] = None,
                              ymin: Optional[float] = None,
                              ymax: Optional[float] = None):
    """ Plot distance vs displacement

    :param ndarray track_distances:
        Total distance along the track
    :param ndarray track_displacement:
        Total displacement along the track
    """
    if colors is not None and cmap is not None:
        raise ValueError('Pass one of colors or cmap, not both!')

    if colors is not None:
        if isinstance(colors, (str)):
            colors = (colors, )
        track_levels = np.unique(track_class[~np.isnan(track_class)])
        print('Got {} track levels: {}'.format(len(track_levels), track_levels))
        print('With {} colors: {}'.format(len(colors), colors))
        if len(track_levels) != len(colors):
            raise ValueError('Got {} track levels but {} colors'.format(len(track_levels), len(colors)))
    else:
        track_levels = None

    if markersize is None:
        markersize = 1.0
    if track_levels is not None:
        if isinstance(markersize, (int, float)):
            markersize = [markersize for _ in track_levels]
        assert len(markersize) == len(track_levels)

    # Throw out invalid values
    track_mask = np.logical_and(~np.isnan(track_distances),
                                ~np.isnan(track_displacements))
    track_mask = np.logical_and(track_mask, ~np.isnan(track_class))

    # Split into distance, displacement, color
    track_distances = track_distances[track_mask]
    track_displacements = track_displacements[track_mask]
    track_class = track_class[track_mask]

    if cmap is not None:
        if vmin is None:
            vmin = np.percentile(track_class, [1])[0]
        if vmax is None:
            vmax = np.percentile(track_class, [99])[0]
        print('Coloring tracks "{}" range: {} to {}'.format(cmap, vmin, vmax))
    else:
        vmin = vmax = None

    print('Distance vs displacement...')
    print('Total points: {}'.format(track_distances.shape[0]))

    # Plot bounds
    max_dist = np.percentile(track_distances, [99])[0]
    max_disp = np.percentile(track_displacements, [99])[0]

    if xmin is None:
        xmin = 0
    if xmax is None:
        xmax = max([max_dist, max_disp])
    if ymin is None:
        ymin = 0
    if ymax is None:
        ymax = max([max_dist, max_disp])

    with set_plot_style(plot_style) as style:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if colors is not None:
            for color, level, mksz in zip(colors, track_levels, markersize):
                mask = track_class == level
                ax.plot(track_distances[mask],
                        track_displacements[mask],
                        color=color,
                        marker='.',
                        linestyle='none',
                        markersize=mksz)
        elif cmap is not None:
            ax.scatter(track_distances, track_displacements,
                       marker='.',
                       c=track_class,
                       s=markersize,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax)
        else:
            ax.plot(track_distances,
                    track_displacements,
                    color='k',
                    marker='.',
                    linestyle='none',
                    markersize=markersize)

        # Put theory curves on
        if plot_style == 'dark':
            marker = '--w'
        else:
            marker = '--k'

        distance_rand_diffusion = np.linspace(xmin, xmax, 50)
        displacement_rand_diffusion = np.sqrt(distance_rand_diffusion)
        dist_disp_rand_diffusion = 0.5*(distance_rand_diffusion**2 - distance_rand_diffusion)/(distance_rand_diffusion - np.sqrt(distance_rand_diffusion))

        ax.plot(distance_rand_diffusion, displacement_rand_diffusion, marker, label='Random Diffusion')
        ax.plot(distance_rand_diffusion, distance_rand_diffusion, marker, label='Persistent Motion')
        ax.plot(distance_rand_diffusion, dist_disp_rand_diffusion, '--', color='#AAAAAA', linewidth=2)

        ax.set_xlabel('Track distance ($\\mu m$)')
        ax.set_ylabel('Track displacement ($\\mu m$)')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.legend()

        style.show(outfile, transparent=True)
