""" Tools for ploting individual traces

* :py:class:`TraceDB`: Store a map between individual traces and image stacks

"""

# Imports
import pathlib
from typing import Dict, Optional, Tuple, List

# 3rd party imports
import numpy as np

import matplotlib.pyplot as plt

from skimage.feature import peak_local_max

# Our own imports
from ..plotting import colorwheel, set_plot_style
from ..utils import (
    guess_channel_dir, find_tiledirs, parse_tile_name, write_movie, LazyImageDir)
from .utils import load_track_csvfile


# Classes


class TraceDB(object):
    """ Map between images and traces

    :param Path image_dir:
        The image directory to load
    :param Path track_file:
        The parallel track file to load
    :param Path activation_dir:
        The directory of activation images to load
    :param float time_scale:
        Scale factor for time (mins/frame)
    :param float space_scale:
        Scale factor for space (um/min)
    :param int padding:
        Padding on all sides by this many pixels
    :param float linewidth:

    """

    def __init__(self,
                 image_dir: pathlib.Path,
                 track_file: pathlib.Path,
                 activation_dir: Optional[pathlib.Path] = None,
                 time_scale: float = 1.0,
                 space_scale: float = 1.0,
                 padding: int = 50,
                 plot_style: str = 'light',
                 figsize: Tuple[int] = (8, 8),
                 linewidth: float = 3,
                 markersize: float = 12,
                 raw_imgs_cmap: str = 'viridis',
                 act_imgs_cmap: str = 'inferno',
                 track_cmap: str = 'Set1',
                 transpose: bool = False):
        self.image_dir = image_dir
        self.activation_dir = activation_dir  # FIXME: This is redundant with image_dir
        self.track_file = track_file

        self.time_scale = time_scale
        self.space_scale = space_scale

        self.padding = padding
        self.transpose = transpose

        self.linewidth = linewidth
        self.markersize = markersize
        self.figsize = figsize
        self.plot_style = plot_style

        self.raw_imgs_cmap = raw_imgs_cmap
        self.act_imgs_cmap = act_imgs_cmap
        self.track_cmap = track_cmap

        self.num_tail_points = 10

        self.vmin = 0.0
        self.vmax = 255.0

        self.tracks = None
        self.track_bboxes = None
        self.min_track_len = None
        self.max_track_len = None

        self.raw_imgs = None
        self.act_imgs = None
        self.rows = self.cols = None

    def load_track_file(self,
                        min_samples: int = 5,
                        min_distance: float = 1.0,
                        min_displacement: float = 1.0):
        """ Load the track from a file

        :param int min_samples:
            Minimum samples to keep a track
        :param float min_distance:
            Minimum distance along a track to keep
        :param float min_displacement:
            Minimum displacement along a track to keep
        """
        filtered_tracks = []
        for track in load_track_csvfile(self.track_file):
            if len(track) < min_samples:
                continue
            distance = track.get_track_length(self.space_scale)
            displacement = track.get_track_displacement(self.space_scale)

            if distance < min_distance:
                continue
            if displacement < min_displacement:
                continue
            filtered_tracks.append(track)
        if len(filtered_tracks) < 1:
            raise ValueError(f'No sufficiently long tracks under: {self.track_file}')

        filtered_tracks = list(sorted(filtered_tracks, key=lambda t: len(t), reverse=True))
        print(f'Got {len(filtered_tracks)} tracks')

        min_len = min([len(t) for t in filtered_tracks])
        max_len = max([len(t) for t in filtered_tracks])
        print(f'Min track len: {min_len}')
        print(f'Max track len: {max_len}')

        self.min_track_len = min_len
        self.max_track_len = max_len
        self.tracks = filtered_tracks

        track_bboxes = []
        for track in filtered_tracks:
            bbox = track.get_bbox()
            track_bboxes.append([bbox.x0, bbox.x1, bbox.y0, bbox.y1])
        self.track_bboxes = np.array(track_bboxes)

    def load_raw_images(self, scale: float = 1.0, suffix: str = ''):
        """ Load the raw image database

        :param float scale:
            If not 1, the rescaling factor to resize images by
        :param str suffix:
            The suffix for each image to find
        """

        self.raw_imgs = LazyImageDir(self.image_dir,
                                     scale=scale,
                                     suffix=suffix,
                                     transpose=self.transpose)
        if self.rows is None or self.cols is None:
            _, self.rows, self.cols = self.raw_imgs.shape

    def load_activation_images(self, scale: float = 1.0, suffix: str = '_resp'):
        """ Load the activation image database

        :param float scale:
            If not 1, the rescaling factor to resize images by
        :param str suffix:
            The suffix for each image to find
        """
        if self.activation_dir is None:
            raise ValueError('Cannot load activation images without a valid activation_dir')
        self.act_imgs = LazyImageDir(self.activation_dir,
                                     scale=scale,
                                     suffix=suffix,
                                     transpose=self.transpose)
        if self.rows is None or self.cols is None:
            _, self.rows, self.cols = self.act_imgs.shape

    def check_image_alignment(self):
        """ Make sure the images and tracks are concordant """

        if self.raw_imgs is not None:
            print(f'Raw image shape: {self.raw_imgs.shape}')
            if self.raw_imgs.shape[0] < self.max_track_len:
                raise ValueError(f'Got images with {self.raw_imgs.shape[0]} frames but tracks with {self.max_track_len} timepoints')

        if self.act_imgs is not None:
            print(f'Act image shape: {self.act_imgs.shape}')
            if self.act_imgs.shape[0] < self.max_track_len:
                raise ValueError(f'Got activations with {self.act_imgs.shape[0]} frames but tracks with {self.max_track_len} timepoints')

        if self.act_imgs is not None and self.raw_imgs is not None:
            if self.act_imgs.shape != self.raw_imgs.shape:
                raise ValueError(f'Images have shape {self.raw_imgs.shape} but activations are shape {self.act_imgs.shape}')

    def plot_all_single_traces(self,
                               outdir: pathlib.Path,
                               image_type: str = 'raw',
                               track_start: int = 0,
                               track_end: int = -1,
                               track_step: int = 1,
                               min_timepoint: int = 0,
                               max_timepoint: int = -1,
                               write_to_movie: bool = False,
                               frames_per_second: int = 5):
        """ Plot all traces over a single track

        :param Path outdir:
            Directory to write traces under
        :param int track_start:
            Which track index to start with
        :param int track_end:
            Which track index to end with
        :param int track_step:
            Step size for iterating over tracks
        :param int min_timepoint:
            Minimum timepoint to plot
        :param int max_timepoint:
            Maximum timepoint to plot
        :param bool write_to_movie:
            If True, write the frames to a movie
        :param int frames_per_second:
            Frames per second for the trace plot
        """
        if track_start < 0:
            track_start = len(self.tracks) + track_start
        if track_end < 0:
            track_end = len(self.tracks) + track_end
        if track_start < 0 or track_start >= len(self.tracks):
            raise IndexError(f'Invalid start track index {track_start} for {len(self.tracks)} tracks')
        if track_end < 0 or track_end >= len(self.tracks):
            raise IndexError(f'Invalid end track index {track_end} for {len(self.tracks)} tracks')

        for trackid in range(track_start, track_end, track_step):
            if image_type == 'raw':
                framefiles = self.plot_raw_single_trace(
                    outdir, trackid,
                    min_timepoint=min_timepoint,
                    max_timepoint=max_timepoint)
            elif image_type in ('act', 'activation'):
                framefiles = self.plot_act_single_trace(
                    outdir, trackid,
                    min_timepoint=min_timepoint,
                    max_timepoint=max_timepoint)
            else:
                raise KeyError(f'Unknown image type: "{image_type}"')

            # Write out the track to a movie
            if write_to_movie and outdir is not None:
                moviefile = outdir / f'{image_type}-tr{trackid:03d}.mp4'
                print(f'Writing to movie: {moviefile}')
                write_movie(framefiles, moviefile,
                            frames_per_second=frames_per_second,
                            get_size_from_frames=True)

    def plot_roi_traces(self,
                        outdir: Optional[pathlib.Path],
                        bbox: List[Tuple],
                        image_type: str = 'raw',
                        min_timepoint: int = 0,
                        max_timepoint: int = -1,
                        write_to_movie: bool = False,
                        frames_per_second: float = 5) -> List[pathlib.Path]:
        """ Plot all traces in a single ROI

        :param Path outdir:
            The directory to write the traces out to
        :param list[tuple] bbox:
            The list of bounding box coordinates to use
        :param str image_type:
            One of "raw" or "act" to select raw images or activations
        :param int min_timepoint:
            Minimum timepoint index to use
        :param int max_timepoint:
            Maximum timepoint index to use
        :returns:
            The list of files written, if any
        """

        if image_type == 'raw':
            block_img = self.raw_imgs.crop(bbox)
            cmap = self.raw_imgs_cmap
            prefix = 'raw'
        elif image_type in ('act', 'activation'):
            block_img = self.act_imgs.crop(bbox)
            cmap = self.act_imgs_cmap
            prefix = 'act'
        else:
            raise KeyError(f'Unknown image type: "{image_type}"')

        # Figure out the name of the ROI movie
        if outdir is None:
            trackdir = None
        else:
            roi_id = 0
            while True:
                trackdir = outdir / f'{prefix}-roi{roi_id:03d}'
                if not trackdir.is_dir():
                    break
                roi_id += 1
            trackdir.mkdir(exist_ok=True, parents=True)

        palette = colorwheel(self.track_cmap)

        xmin = bbox[0][0]
        xmax = bbox[0][1]
        ymin = bbox[1][0]
        ymax = bbox[1][1]

        # Aspect ratio for the tracks
        aspect = (ymax - ymin) / (xmax - xmin)
        fig_x = self.figsize[0] * aspect
        fig_y = self.figsize[0]

        track_ids = self.calc_tracks_in_bbox(bbox)
        tracks = [self.tracks[i].to_arrays() for i in track_ids]
        if len(tracks) < 1:
            print(f'No tracks found in {bbox}')
            return
        print(f'Found {len(tracks)} tracks in ROI')

        # Render animated because that runs much faster
        framefiles = []
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))
            art1 = ax.imshow(block_img[0, :, :], cmap=cmap, aspect='equal',
                             vmin=self.vmin, vmax=self.vmax)
            artists = [art1]
            for i, (timepoints, ys, xs) in enumerate(tracks):
                color = palette[i]
                art_line = ax.plot(ys[:1] - ymin, xs[:1] - xmin, '-', color=color,
                                   linewidth=self.linewidth)[0]
                art_point = ax.plot(ys[0] - ymin, xs[0] - xmin, 'o', color=color,
                                    markersize=self.markersize)[0]
                artists.extend([art_line, art_point])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, ymax-ymin])
            ax.set_ylim([xmax-xmin, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            for art in artists:
                art.set_animated(True)
            fig.canvas.draw()
            bg_cache = fig.canvas.copy_from_bbox(ax.bbox)

            if min_timepoint < 0:
                min_timepoint = block_img.shape[0] + min_timepoint
            if max_timepoint < 0:
                max_timepoint = block_img.shape[0] + max_timepoint

            for i in range(min_timepoint, max_timepoint):
                fig.canvas.restore_region(bg_cache)

                if i not in block_img:
                    continue

                art1.set_data(block_img[i, :, :])
                for j, (timepoints, ys, xs) in enumerate(tracks):
                    art_line = artists[j*2 + 1]
                    art_point = artists[j*2 + 2]
                    t_ed = i - np.min(timepoints)
                    t_st = t_ed - self.num_tail_points
                    if t_ed < 0:
                        art_line.set_visible(False)
                        art_point.set_visible(False)
                    elif t_ed >= timepoints.shape[0]:
                        art_line.set_visible(True)
                        art_point.set_visible(False)
                        t_ed = timepoints.shape[0]

                        art_line.set_data(ys[t_st:t_ed+1] - ymin, xs[t_st:t_ed+1] - xmin)
                    elif t_st >= timepoints.shape[0]:
                        art_line.set_visible(False)
                        art_point.set_visible(False)
                    else:
                        art_line.set_visible(True)
                        art_point.set_visible(True)

                        t_st = max([0, t_st])
                        art_line.set_data(ys[t_st:t_ed+1] - ymin, xs[t_st:t_ed+1] - xmin)
                        art_point.set_data(ys[t_ed] - ymin, xs[t_ed] - xmin)

                for art in artists:
                    art.axes.draw_artist(art)
                fig.canvas.blit(ax.bbox)

                if trackdir is None:
                    plt.pause(0.1)
                else:
                    outfile = trackdir / f'{self.image_dir.name}-roi{roi_id:03d}t{i:03d}.tif'
                    print(f'Saving frame: {outfile}')
                    style.savefig(str(outfile), transparent=True,
                                  bbox_inches='tight', pad_inches=0)
                    framefiles.append(outfile)
            plt.close()

        if write_to_movie and outdir is not None:
            moviefile = trackdir.parent / (trackdir.name + '.mp4')
            print(f'Writing frames to {moviefile}')
            write_movie(framefiles, moviefile,
                        frames_per_second=frames_per_second,
                        get_size_from_frames=True)
        return framefiles

    def plot_raw_single_trace(self, outdir: Optional[pathlib.Path],
                              trackid: int,
                              min_timepoint: int = 0,
                              max_timepoint: int = -1) -> List[pathlib.Path]:
        """ Plot a single trace

        :param Path outdir:
            The root directory where traces will be stored
        :param int trackid:
            Index of the track to load
        :returns:
            The list of files written, if any
        """

        if outdir is None:
            trackdir = None
        else:
            trackdir = outdir / f'raw-tr{trackid:03d}'
            trackdir.mkdir(exist_ok=True, parents=True)

        print(f'Track {trackid}')

        track = self.tracks[trackid]
        timepoints, ys, xs = track.to_arrays()

        distance = np.sum(np.sqrt((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2))
        displacement = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        print(f'Distance: {distance * self.space_scale}')
        print(f'Displacement: {displacement * self.space_scale}')

        # Work out bounding box around points
        bbox = self.calc_image_bbox(xs, ys)
        xmin = bbox[0][0]
        xmax = bbox[0][1]
        ymin = bbox[1][0]
        ymax = bbox[1][1]

        # Aspect ratio for the tracks
        aspect = (ymax - ymin) / (xmax - xmin)
        fig_x = self.figsize[0] * aspect
        fig_y = self.figsize[0]
        block_raw_img = self.raw_imgs.crop(bbox)

        # Render animated because that runs much faster
        framefiles = []
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))
            art1 = ax.imshow(block_raw_img[timepoints[0], :, :],
                             cmap=self.raw_imgs_cmap,
                             aspect='equal',
                             vmin=self.vmin,
                             vmax=self.vmax)
            art2 = ax.plot(ys[:1] - ymin, xs[:1] - xmin, '-r',
                           linewidth=self.linewidth)[0]
            art3 = ax.plot(ys[0] - ymin, xs[0] - xmin, 'ro',
                           markersize=self.markersize)[0]

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, ymax-ymin])
            ax.set_ylim([xmax-xmin, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            artists = [art1, art2, art3]
            for art in artists:
                art.set_animated(True)
            fig.canvas.draw()
            bg_cache = fig.canvas.copy_from_bbox(ax.bbox)

            if min_timepoint < 0:
                min_timepoint = timepoints.shape[0] + min_timepoint
            if max_timepoint < 0:
                max_timepoint = timepoints.shape[0] + max_timepoint

            for i in range(min_timepoint, max_timepoint):
                fig.canvas.restore_region(bg_cache)

                if timepoints[i] not in block_raw_img:
                    continue

                art1.set_data(block_raw_img[timepoints[i], :, :])
                art2.set_data(ys[:i+1] - ymin, xs[:i+1] - xmin)
                art3.set_data(ys[i] - ymin, xs[i] - xmin)

                for art in artists:
                    art.axes.draw_artist(art)
                fig.canvas.blit(ax.bbox)

                if trackdir is None:
                    plt.pause(0.1)
                else:
                    outfile = trackdir / f'{self.image_dir.name}-tr{trackid:03d}t{i:03d}.tif'
                    print(f'Saving frame: {outfile}')
                    framefiles.append(outfile)
                    style.savefig(str(outfile), transparent=True,
                                  bbox_inches='tight', pad_inches=0)
            plt.close()
        return framefiles

    def plot_act_single_trace(self, outdir: Optional[pathlib.Path],
                              trackid: int,
                              min_timepoint: int = 0,
                              max_timepoint: int = -1) -> List[pathlib.Path]:
        """ Plot the traces over the activation images

        :returns:
            The list of files written, if any
        """

        if outdir is None:
            trackdir = None
        else:
            trackdir = outdir / f'act-tr{trackid:03d}'
            trackdir.mkdir(exist_ok=True, parents=True)

        print(f'Track {trackid}')

        track = self.tracks[trackid]
        timepoints, ys, xs = track.to_arrays()

        bbox = self.calc_image_bbox(xs, ys)
        xmin = bbox[0][0]
        xmax = bbox[0][1]
        ymin = bbox[1][0]
        ymax = bbox[1][1]

        # Aspect ratio for the tracks
        aspect = (ymax - ymin) / (xmax - xmin)
        fig_x = self.figsize[0] * aspect
        fig_y = self.figsize[0]

        block_act_img = self.act_imgs.crop(bbox)
        framefiles = []
        with set_plot_style(self.plot_style) as style:
            fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))

            art4 = ax.imshow(block_act_img[timepoints[0], :, :],
                             cmap=self.act_imgs_cmap,
                             aspect='equal',
                             vmin=self.vmin,
                             vmax=self.vmax)

            peaks = peak_local_max(block_act_img[timepoints[0], :, :],
                                   min_distance=3,
                                   threshold_abs=50,
                                   exclude_border=0)
            art5 = ax.plot(peaks[:, 1], peaks[:, 0],
                           'go', markersize=20)[0]

            art6 = ax.plot(ys[:1] - ymin, xs[:1] - xmin, '-r',
                           linewidth=self.linewidth)[0]
            art7 = ax.plot(ys[0] - ymin, xs[0] - xmin, 'ro',
                           markersize=self.markersize)[0]

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([0, ymax-ymin])
            ax.set_ylim([xmax-xmin, 0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            artists = [art4, art5, art6, art7]
            for art in artists:
                art.set_animated(True)
            fig.canvas.draw()
            bg_cache = fig.canvas.copy_from_bbox(ax.bbox)

            if min_timepoint < 0:
                min_timepoint = timepoints.shape[0] + min_timepoint
            if max_timepoint < 0:
                max_timepoint = timepoints.shape[0] + max_timepoint

            for i in range(min_timepoint, max_timepoint):
                fig.canvas.restore_region(bg_cache)

                if timepoints[i] not in block_act_img:
                    continue

                peaks = peak_local_max(block_act_img[timepoints[i], :, :],
                                       min_distance=3,
                                       threshold_abs=50,
                                       exclude_border=0)

                art4.set_data(block_act_img[timepoints[i], :, :])
                art5.set_data(peaks[:, 1], peaks[:, 0])
                art6.set_data(ys[:i+1] - ymin, xs[:i+1] - xmin)
                art7.set_data(ys[i] - ymin, xs[i] - xmin)

                for art in artists:
                    art.axes.draw_artist(art)
                fig.canvas.blit(ax.bbox)

                if trackdir is None:
                    plt.pause(0.1)
                else:
                    outfile = trackdir / f'{self.activation_dir.name}-tr{trackid:03d}t{i:03d}.png'
                    print(f'Saving frame: {outfile}')
                    style.savefig(str(outfile), transparent=True,
                                  bbox_inches='tight', pad_inches=0)
                    framefiles.append(outfile)
            plt.close()
        return framefiles

    def calc_bbox_from_trackid(self, trackid: int) -> List[Tuple[int]]:
        """ Calculate the bounding box from a track identifier

        :param int trackid:
            The track index to use
        :returns:
            A bounding box around that track in image coordinates
        """
        track = self.tracks[trackid]
        _, ys, xs = track.to_arrays()
        return self.calc_image_bbox(xs, ys)

    def calc_image_bbox(self, xs: np.ndarray, ys: np.ndarray) -> List[Tuple[int]]:
        """ Calculate the bounding box around a set of tracks

        :param ndarray xs:
            The x coordinates for the track
        :param ndarray ys:
            The y coordinates for the track
        :returns:
            A bounding box around that track in image coordinates
        """

        xmin = np.floor(np.min(xs)) - self.padding
        xmax = np.ceil(np.max(xs)) + self.padding
        xmin = int(max(xmin, 0))
        xmax = int(min(xmax, self.rows))

        ymin = np.floor(np.min(ys)) - self.padding
        ymax = np.ceil(np.max(ys)) + self.padding
        ymin = int(max(ymin, 0))
        ymax = int(min(ymax, self.cols))

        print(f'X Range: {xmin} to {xmax}')
        print(f'Y Range: {ymin} to {ymax}')
        return [(xmin, xmax), (ymin, ymax)]

    def calc_tracks_in_bbox(self, bbox: List[Tuple]) -> np.ndarray:
        """ Calculate the tracks that fall within an image bounding box

        :returns:
            A list of track ids that fall in this bounding box
        """
        ymin = bbox[0][0]
        ymax = bbox[0][1]
        xmin = bbox[1][0]
        xmax = bbox[1][1]
        track_bboxes = self.track_bboxes

        xmask = np.logical_and(np.any(track_bboxes[:, 0:2] >= xmin, axis=1),
                               np.any(track_bboxes[:, 0:2] <= xmax, axis=1))
        ymask = np.logical_and(np.any(track_bboxes[:, 2:4] >= ymin, axis=1),
                               np.any(track_bboxes[:, 2:4] <= ymax, axis=1))
        return np.where(np.logical_and(xmask, ymask))[0]

    # Helper methods

    @staticmethod
    def find_image_dir(rootdir: pathlib.Path, tileno: int, channel: str) -> pathlib.Path:
        """ Find the image directory """

        channel, channel_dir = guess_channel_dir(rootdir / 'Corrected', channel)
        print(f'Following on channel: {channel}')

        tiledirs = list(find_tiledirs(channel_dir, tiles=tileno))
        if len(tiledirs) == 0:
            raise OSError(f'No tile data for tile {tileno}: {rootdir}')
        if len(tiledirs) > 1:
            raise OSError(f'Multiple tiles match {tileno}: {rootdir}')
        return tiledirs[0][1]

    @staticmethod
    def find_track_file(track_dir: pathlib.Path, tile_data: Dict) -> pathlib.Path:
        """ Find the track file for a given set of tile data """

        track_filenames = ['s{tile:02d}-{condition}_traces.csv'.format(**tile_data),
                           's{tile:02d}_traces.csv'.format(**tile_data)]
        track_file = None
        for track_filename in track_filenames:
            if (track_dir / track_filename).is_file():
                track_file = track_dir / track_filename
                break
        if track_file is None or not track_file.is_file():
            raise OSError(f'Cannot find track data under {track_dir} matching {tile_data}')
        return track_file

    @staticmethod
    def find_activation_dir(activation_rootdir: pathlib.Path, tile_data: Dict) -> pathlib.Path:
        """ Find where the activations are for this track """
        act_dirnames = ['s{tile:02d}-{condition}'.format(**tile_data),
                        's{tile:02d}'.format(**tile_data)]
        activation_dir = None
        for act_dirname in act_dirnames:
            if (activation_rootdir / act_dirname).is_dir():
                activation_dir = activation_rootdir / act_dirname
                break
        if activation_dir is None:
            raise OSError(f'Cannot find activations under {activation_rootdir} matching {tile_data}')
        return activation_dir

    @classmethod
    def from_tileno(cls,
                    rootdir: pathlib.Path,
                    tileno: int,
                    channel: pathlib.Path = 'gfp',
                    detector: pathlib.Path = 'composite',
                    **kwargs) -> 'TraceDB':
        """ Load a database from a rootdir/tileno combination

        :param Path rootdir:
            The experiment directory to look at
        :param int tileno:
            The tile number to look at
        :param str channel:
            The channel to look for
        :param str detector:
            The detector to look for
        """
        image_dir = cls.find_image_dir(rootdir, tileno, channel)

        # Go fishing for matching tracks
        tile_data = parse_tile_name(image_dir.name)

        activation_rootdir = guess_channel_dir(rootdir / f'SingleCell-{detector}' / 'Corrected', channel)[1]
        track_dir = rootdir / f'CellTracking-{detector}' / 'Tracks'

        track_file = cls.find_track_file(track_dir, tile_data)
        try:
            activation_dir = cls.find_activation_dir(activation_rootdir, tile_data)
        except OSError:
            print(f'Activation dir not found under {activation_rootdir}')
            activation_dir = None
        return cls(image_dir=image_dir,
                   track_file=track_file,
                   activation_dir=activation_dir,
                   **kwargs)
