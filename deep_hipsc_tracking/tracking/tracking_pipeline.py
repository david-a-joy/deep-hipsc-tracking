""" Run the tracking algorithm to link detections

Main tracking function:

* :py:func:`make_tracks_for_experiment`: Make an entire experiment worth of tracks

Functions:

* :py:func:`link_nearby_tracks`: Merge track fragments into tracks
* :py:func:`find_all_csvfiles`: Find all CSV track files in a directory

API Documentation
-----------------

"""

# Standard lib
import time
import random
import shutil
import pathlib
from typing import List, Tuple, Optional

# 3rd party
import numpy as np

from PIL import Image

from sklearn.neighbors import BallTree

# Our own imports
from . import link_all_chains, load_track_csvfile, save_track_csvfile, Link
from ..utils import load_point_csvfile
from ..plotting import plot_all_tracks

# Constants

PLOT_STYLE = 'light'
SUFFIX = '.png'

OVERWRITE = False

FIGSIZE = (24, 24)

TRACE_LINEWIDTH = 2  # Width of the lines for the traces

MARKERSIZE = 30

MAX_TRACKS = 5000  # Maximum individual tracks to plot
MAX_TRACK_LAG = 3  # frames - Maximum lag between endpoints to link a track - 15 minutes
MIN_TRACK_LEN = 10  # frames
MIN_POINT_ACTIVATION = 0.01  # Minimal activation under the mask
MAX_LINK_DIST = 25.0  # pixels travel - 8 mm
MAX_VELOCITY = 40.0  # pixels / minute
MAX_RELINK_ATTEMPTS = 10  # Maximumm tries to relink tracks
MAX_MERGE_DIST = 8.0  # pixels - maximum distance to merge points

DEFAULT_ROWS, DEFAULT_COLS = 1000, 1000  # Default image shape for scale-free images

COLORS = []

LINK_FXN = 'balltree'  # Which algorithm to link with
LINK_STEP = 1  # How many frames to step while linking (1 - every frame, 2 - every other frame, etc...)
IMPUTE_STEPS = 1  # How many rounds of imputation to run while linking

MIN_TIMEPOINT = 0
MAX_TIMEPOINT = -1

# Functions


def subset_chains(chains: List[Link], subset_pct: float) -> List[Link]:
    """ Randomly subset chains to simulate different mixing percentages

    :param list[Link] chains:
        A set of linked tracks to subset
    :param float subset_pct:
        The subset percentage between 0 (no chains) and 100 (keep all chains)
    :returns:
        A subset list of links, randomly chosen
    """
    num_chains = len(chains)
    num_subset = int(round(subset_pct / 100 * num_chains))
    print(f'Selecting {num_subset} of {num_chains} chains, {subset_pct}%')
    return random.sample(chains, num_subset)


def find_all_csvfiles(tiledir: pathlib.Path,
                      reverse: bool = False) -> List[pathlib.Path]:
    """ Find all the CSV files in the tile directory

    :param Path tiledir:
        The tile directory to load all the point detections from
    :param bool reverse:
        If true, reverse the track files
    :returns:
        The list of csv files to load for all tiles
    """
    csvfiles = []
    for subfile in sorted(tiledir.iterdir(), reverse=reverse):
        if subfile.suffix not in ('.csv', ):
            continue
        if not subfile.is_file():
            continue
        csvfiles.append(subfile)
    return csvfiles


def pair_with_image(csvfiles: List[pathlib.Path],
                    single_cell_dir: pathlib.Path,
                    datadir: pathlib.Path,
                    plotdir: pathlib.Path,
                    suffix: str = SUFFIX) -> List[Tuple[pathlib.Path]]:
    """ Pair an image off with the dataset

    :param list[Path] csvfiles:
        The list of CSV files to pair off
    :param Path single_cell_dir:
        The base directory where the CSV files live
    :param Path datadir:
        The data directory to pair off with
    :param Path plotdir:
        The directory to save plots to
    :param str suffix:
        Suffix to use for the plots
    :returns:
        A list of file locations corresponding to the CSV file
    """
    imagefiles = []
    for csvfile in csvfiles:
        stem = csvfile.stem
        rel_csvfile = csvfile.relative_to(single_cell_dir)

        imagefile = datadir / rel_csvfile.parent / f'{stem}.tif'
        if not imagefile.is_file():
            raise OSError(f'Cannot find expected image file at {imagefile}')
        plotfile = plotdir / rel_csvfile.parent / f'merged_gfp_{stem}{suffix}'

        imagefiles.append((csvfile, imagefile, plotfile))
    return imagefiles


def pair_all_images(rootdir: pathlib.Path,
                    detector: str = None,
                    overwrite: bool = OVERWRITE,
                    reverse: bool = False,
                    suffix: str = SUFFIX):
    """ Pair all the images from a nested dataset

    :param Path rootdir:
        The path to the base directory to load
    :param str detector:
        Which detector to use to track the images
    :param bool overwrite:
        If True, clear the old directories
    :param bool reverse:
        If True, reverse the order of the images
    :param Path experiment_root_dir:
        The base directory for the experiments
    """

    if detector in (None, ''):
        single_cell_name = 'SingleCell'
        tracking_name = 'CellTracking'
    else:
        for try_detector in [detector.lower(), detector.capitalize()]:
            single_cell_name = f'SingleCell-{try_detector}'
            if (rootdir / single_cell_name).is_dir():
                break
        tracking_name = f'CellTracking-{try_detector}'

    single_cell_dir = rootdir / single_cell_name / 'Corrected'
    raw_image_dir = rootdir / 'Corrected'

    if not single_cell_dir.is_dir():
        raise OSError(f'Cannot find single cell tracking under {single_cell_dir}')
    if not raw_image_dir.is_dir():
        raise OSError(f'Cannot find raw images under {raw_image_dir}')

    plotdir = rootdir / tracking_name / 'Plots'
    trackdir = rootdir / tracking_name / 'Tracks'

    if overwrite:
        if plotdir.is_dir():
            shutil.rmtree(str(plotdir))

        if trackdir.is_dir():
            shutil.rmtree(str(trackdir))

    targets = [single_cell_dir]
    while targets:
        tiledir = targets.pop()
        if not tiledir.is_dir():
            continue
        targets.extend(p for p in tiledir.iterdir()
                       if p.is_dir() and not p.name.startswith('.'))
        csvfiles = find_all_csvfiles(tiledir, reverse=reverse)
        if len(csvfiles) < 1:
            continue
        pairfiles = pair_with_image(csvfiles, single_cell_dir, raw_image_dir, plotdir, suffix=suffix)
        yield plotdir, trackdir, tiledir, pairfiles


def load_track(csvfile: pathlib.Path,
               imagefile: Optional[pathlib.Path] = None,
               score_level: Optional[float] = None,
               min_point_activation: float = MIN_POINT_ACTIVATION):
    """ Load in a single point track, along with the raw image and the mask

    :param Path csvfile:
        The path to the csvfile to load
    :param Path imagefile:
        The path to the image file to load, or None for an empty image
    :param float score_level:
        If not None, the minimum image intensity to accept
    :param float min_point_activation:
        The minimum point probability to accept
    :returns:
        The image, xcoords, ycoords, mask_contours
    """

    cx, cy, cv = load_point_csvfile(csvfile)

    if imagefile is None:
        print(f'No image for {csvfile.name}')
        img = np.zeros((DEFAULT_ROWS, DEFAULT_COLS))
    else:
        print(f'Loading image for {csvfile.name}: {imagefile}')
        img = Image.open(str(imagefile))
        img = np.asarray(img)

    if img.ndim == 3:
        img = np.mean(img, axis=2)
    assert img.ndim == 2

    rows, cols = img.shape
    print(f'Image size: {rows}x{cols}')

    # Filter by score in the point mask
    point_mask = cv >= min_point_activation
    total_points = point_mask.shape[0]
    num_good_points = np.sum(point_mask)
    num_bad_points = total_points - num_good_points

    print(f'Accepting {num_good_points} points')
    print(f'Rejecting {num_bad_points} points')

    cx = cx[point_mask]
    cy = cy[point_mask]

    cx = cx * cols
    cy = (1.0 - cy) * rows

    # Filter points using a score cutoff
    fx, fy = [], []
    for x, y in zip(cx, cy):
        if score_level is not None:
            ix, iy = int(np.round(x)), int(np.round(y))

            score = np.mean(img[iy-2:iy+2, ix-2:ix+2])
            if score < score_level:
                continue
        fx.append(x)
        fy.append(y)
    return img, np.array(fx, dtype=np.float64), np.array(fy, dtype=np.float64)


def filter_by_link_dist(tracks: List[Link], max_link_dist: float = MAX_LINK_DIST):
    """ Filter links that are too far from eachother

    :param list tracks:
        The list of tracks to filter
    :param float max_link_dist:
        The maximum distance allowed across a link
    :returns:
        The filtered list of tracks
    """

    print('Filtering the tracks by distance to nearest point...')
    tree_rev = None
    tree_cur = None
    tree_fwd = None

    output_tracks = []

    for i, track in enumerate(tracks):
        points_cur = track[2]

        tree_rev = tree_cur
        tree_cur = tree_fwd

        if i < len(tracks) - 1:
            points_fwd = tracks[i+1][2]
            if points_fwd.shape[0] > 0:
                tree_fwd = BallTree(points_fwd)
            else:
                tree_fwd = None
        else:
            tree_fwd = None

        okay = np.ones((points_cur.shape[0], 1), dtype=bool)

        if points_cur.shape[0] > 0:
            if tree_fwd is not None:
                dist_fwd = tree_fwd.query(points_cur, k=1, return_distance=True)[0]
                okay = np.logical_and(okay, dist_fwd < max_link_dist)

            if tree_rev is not None:
                dist_rev = tree_rev.query(points_cur, k=1, return_distance=True)[0]
                okay = np.logical_and(okay, dist_rev < max_link_dist)

        okay = okay[:, 0]
        num_good_points = np.sum(okay)
        num_bad_points = okay.shape[0] - num_good_points

        print(f'Keeping {num_good_points} points')
        print(f'Rejecting {num_bad_points} points')

        output_tracks.append(track[:2] + (points_cur[okay, :], ))

    return output_tracks


def link_first_last(first_tracks: np.ndarray,
                    last_tracks: np.ndarray,
                    num_tracks: int,
                    max_lag: int,
                    max_dist: float):
    """ Link track fragments by their ends

    :param ndarray first_tracks:
        The n x 3 coordinates (t, x, y) of points for the first part of tracks
    :param ndarray last_tracks:
        The n x 3 coordinates (t, x, y) of points for the last part of tracks
    :param int num_tracks:
        **UNUSED**
    :param int max_lag:
        Maximum allowed distance for a link in time
    :param float max_dist:
        Maximum allowed distance in space
    :returns:
        The best matches for each track set
    """
    tree = BallTree(last_tracks[:, 1:])
    inds, dists = tree.query_radius(first_tracks[:, 1:],
                                    r=max_dist,
                                    return_distance=True)
    assert len(inds) == first_tracks.shape[0]
    new_tracks = []
    # FIXME: This is O(n**2) but could be faster if sorted by time
    for j, (ind, dist) in enumerate(zip(inds, dists)):
        if ind.shape[0] < 1:
            continue
        assert ind.shape == dist.shape
        first_t = first_tracks[j, 0]
        best_matches = []
        for i, d in zip(ind, dist):
            if i == j:
                continue
            last_t = last_tracks[i, 0]
            if first_t <= last_t or first_t - last_t > max_lag:
                continue
            best_matches.append((first_t - last_t, d, i))
        if len(best_matches) > 0:
            new_tracks.append((sorted(best_matches)[0][2], j))
    return new_tracks


def link_nearby_tracks(tracks: List[Link],
                       min_track_len: int = MIN_TRACK_LEN,
                       max_track_lag: int = MAX_TRACK_LAG,
                       max_link_dist: float = MAX_LINK_DIST,
                       max_relink_attempts: int = MAX_RELINK_ATTEMPTS):
    """ Link near tracks

    :param list[Link] tracks:
        A list of all the tracks to potentially link
    :param int min_track_len:
        Minimum number of timepoints for a track
    :param int max_track_lag:
        Maximum spacing between tracks in time steps
    :param float max_link_dist:
        Maximum distance to link tracks over
    :param int max_relink_attempts:
        The maximum number of relink attempts before giving up
    :returns:
        A new list of tracks
    """

    for tstep in range(max_relink_attempts):
        orig_track_len = len(tracks)

        first_tracks = np.array([t.first for t in tracks])
        last_tracks = np.array([t.last for t in tracks])

        if first_tracks.shape[0] == 0 or last_tracks.shape[0] == 0:
            print('No tracks to connect...')
            return tracks

        print(f'Connecting {first_tracks.shape[0]} track fragments...')
        t0 = time.time()
        new_tracks = link_first_last(first_tracks, last_tracks,
                                     num_tracks=first_tracks.shape[0],
                                     max_lag=max_track_lag,
                                     max_dist=max_link_dist)
        print(f'Fragments linked in {time.time() - t0} secs')
        print(f'Created {len(new_tracks)} links')

        link_tracks = []
        assigned_tracks = set()
        for i, j in new_tracks:
            if i in assigned_tracks or j in assigned_tracks:
                continue
            link_tracks.append(Link.join(tracks[i], tracks[j]))
            tracks[i] = None
            tracks[j] = None
            assigned_tracks.add(i)
            assigned_tracks.add(j)

        tracks = [t for t in tracks + link_tracks if t is not None]
        print(f'After {tstep} steps, track count: {len(tracks)}')
        if len(tracks) == orig_track_len:
            break
    else:
        print(f'Track linking did not converge in {max_relink_attempts} steps')

    # Interpolate tracks and filter by final size
    print(f'Filtering by final min size {min_track_len}: before: {len(tracks)}')
    final_tracks = []
    for track in tracks:
        if len(track) >= min_track_len:
            final_tracks.append(track)
    print(f'Filtering by final min size {min_track_len}: after: {len(final_tracks)}')
    return final_tracks

# Main function


def make_tracks_for_experiment(rootdir: pathlib.Path,
                               detector: Optional[str] = None,
                               overwrite: bool = OVERWRITE,
                               plots: Optional[List[str]] = None,
                               max_tracks: int = MAX_TRACKS,
                               min_track_len: int = MIN_TRACK_LEN,
                               max_velocity: float = MAX_VELOCITY,
                               max_merge_dist: float = MAX_MERGE_DIST,
                               min_point_activation: float = MIN_POINT_ACTIVATION,
                               max_link_dist: float = MAX_LINK_DIST,
                               max_track_lag: int = MAX_TRACK_LAG,
                               link_fxn: str = LINK_FXN,
                               link_step: int = LINK_STEP,
                               impute_steps: int = IMPUTE_STEPS,
                               processes: Optional[int] = None,
                               min_timepoint: int = MIN_TIMEPOINT,
                               max_timepoint: int = MAX_TIMEPOINT,
                               reverse: bool = False,
                               plot_style: str = PLOT_STYLE,
                               suffix: str = SUFFIX,
                               max_relink_attempts: int = MAX_RELINK_ATTEMPTS):
    """ Generate all the tracking plots

    :param Path rootdir:
        The name of the experiment directory to track
    :param str detector:
        If not None, the name of the detector to use for track linking
    :param bool overwrite:
        If True, overwrite existing data
    :param list[str] plots:
        The list of plots to generate (one or more of "point", "track")
    :param int max_tracks:
        The maximum number of tracks to plot
    :param int min_track_len:
        The minimum number of track frames for a link
    :param float min_point_activation:
        The minimum activation allowed at the segmentation point
    :param float max_link_dist:
        The maximum link distance allowed for each point
    :param int max_track_lag:
        Maximum number of frames of lag to allow when linking track fragments
    """
    if plots is None or plots == []:
        plots = ['track', 'track_subset']
    if isinstance(plots, str):
        plots = [plots]
    if 'none' in plots:
        plots = []

    track_subset_percents = [10, 30]

    for tile_group in pair_all_images(rootdir,
                                      detector=detector,
                                      overwrite=overwrite,
                                      reverse=reverse,
                                      suffix=suffix):

        plotdir, trackdir, tiledir, tile_files = tile_group
        print(f'Processing: {tiledir}')

        rows, cols = None, None
        first_img = None

        # Load all the points into a volume
        tracks = []
        for i, files in enumerate(tile_files):
            csvfile, imagefile, plotfile = files
            img, fx, fy = load_track(csvfile, imagefile,
                                     min_point_activation=min_point_activation)

            if rows is None and cols is None:
                rows, cols = img.shape[:2]
                print(f'Loading {rows}x{cols} images')
            elif (rows, cols) != img.shape[:2]:
                err = 'Got new image shape. Expected {}x{}, got {}x{} in {}'
                err = err.format(rows, cols, img.shape[0], img.shape[1], csvfile)
                raise ValueError(err)
            if first_img is None:
                first_img = img

            timepoint = i+1
            tracks.append((timepoint, img, np.stack([fx, fy], axis=1)))

        # Write out the trace files
        track_csvfile = trackdir / (tiledir.stem + '_traces.csv')
        if track_csvfile.is_file():
            final_chains = load_track_csvfile(track_csvfile)
        else:
            tracks = tracks[min_timepoint:max_timepoint]
            final_chains = link_all_chains(tracks,
                                           link_fxn=link_fxn,
                                           link_step=link_step,
                                           processes=processes,
                                           impute_steps=impute_steps,
                                           max_merge_dist=max_merge_dist)
            final_chains = link_nearby_tracks(final_chains,
                                              min_track_len=min_track_len,
                                              max_link_dist=max_link_dist,
                                              max_track_lag=max_track_lag,
                                              max_relink_attempts=max_relink_attempts)
            interp_chains = []
            for chain in final_chains:
                if len(chain) < 4:
                    # Can't interpolate with fewer than 4 points
                    continue
                chain.interpolate_points()
                interp_chains.append(chain)
            save_track_csvfile(track_csvfile, interp_chains)

        if rows is None and cols is None:
            print(f'No images loaded, assuming default scaling: {DEFAULT_ROWS}x{DEFAULT_COLS}')
            rows, cols = DEFAULT_ROWS, DEFAULT_COLS

        trackfile = trackdir / f'{tiledir.stem}_traces{suffix}'
        track_imagefile = trackdir / f'{tiledir.stem}_image{suffix}'
        track_arrowfile = trackdir / f'{tiledir.stem}_arrows{suffix}'
        if 'track' in plots and (not trackfile.is_file() or not track_arrowfile.is_file()):
            if min_track_len > 0:
                final_chains = [c for c in final_chains if len(c) >= min_track_len]

            if final_chains == []:
                print('No chains survived filtering')
                continue

            print('Min track length: {}'.format(min([len(c) for c in final_chains])))
            print('Max track length: {}'.format(max([len(c) for c in final_chains])))
            print('Total tracks:     {}'.format(len(final_chains)))

            min_t = min([min(c.line_t) for c in final_chains])
            max_t = max([max(c.line_t) for c in final_chains])

            min_x = min([min(c.line_x) for c in final_chains])
            max_x = max([max(c.line_x) for c in final_chains])

            min_y = min([min(c.line_y) for c in final_chains])
            max_y = max([max(c.line_y) for c in final_chains])

            print('T Limits: {} to {}'.format(min_t, max_t))
            print('X Limits: {} to {}'.format(min_x, max_x))
            print('Y Limits: {} to {}'.format(min_y, max_y))

            plot_all_tracks(final_chains, trackfile,
                            track_style='tracks',
                            title=tiledir.stem,
                            max_tracks=max_tracks,
                            max_velocity=max_velocity,
                            rows=rows, cols=cols,
                            plot_style=plot_style)

            if first_img is not None:
                plot_all_tracks(final_chains, track_imagefile,
                                track_style='tracks',
                                image=first_img,
                                title=tiledir.stem,
                                max_tracks=max_tracks,
                                max_velocity=max_velocity,
                                rows=rows, cols=cols,
                                plot_style=plot_style)

            plot_all_tracks(final_chains, track_arrowfile,
                            track_style='arrows',
                            title=tiledir.stem,
                            max_tracks=max_tracks,
                            max_velocity=max_velocity,
                            rows=rows, cols=cols,
                            plot_style=plot_style)

        if 'track_subset' in plots:
            for track_subset_percent in track_subset_percents:
                track_pct_dir = trackdir / f'{track_subset_percent:d}pct'
                track_pct_csv = track_pct_dir / (tiledir.stem + '.csv')
                track_pct_file = track_pct_dir / f'{tiledir.stem}_traces.png'
                track_pct_arrowfile = track_pct_dir / f'{tiledir.stem}_arrows.png'

                if not track_pct_file.is_file() or not track_pct_arrowfile.is_file():

                    final_pct_chains = subset_chains(final_chains, track_subset_percent)
                    save_track_csvfile(track_pct_csv, final_pct_chains)

                    if min_track_len > 0:
                        final_pct_chains = [c for c in final_pct_chains if len(c) >= min_track_len]

                    if final_pct_chains == []:
                        print('No subset chains survived filtering')
                        continue

                    print('Min track length: {}'.format(min([len(c) for c in final_pct_chains])))
                    print('Max track length: {}'.format(max([len(c) for c in final_pct_chains])))
                    print('Total tracks:     {}'.format(len(final_pct_chains)))

                    min_t = min([min(c.line_t) for c in final_pct_chains])
                    max_t = max([max(c.line_t) for c in final_pct_chains])

                    min_x = min([min(c.line_x) for c in final_pct_chains])
                    max_x = max([max(c.line_x) for c in final_pct_chains])

                    min_y = min([min(c.line_y) for c in final_pct_chains])
                    max_y = max([max(c.line_y) for c in final_pct_chains])

                    print('T Limits: {} to {}'.format(min_t, max_t))
                    print('X Limits: {} to {}'.format(min_x, max_x))
                    print('Y Limits: {} to {}'.format(min_y, max_y))

                    plot_all_tracks(final_pct_chains, track_pct_file,
                                    track_style='tracks',
                                    title=tiledir.stem,
                                    max_tracks=max_tracks,
                                    max_velocity=max_velocity,
                                    rows=rows, cols=cols,
                                    plot_style=plot_style)

                    plot_all_tracks(final_pct_chains, track_pct_arrowfile,
                                    track_style='arrows',
                                    title=tiledir.stem,
                                    max_tracks=max_tracks,
                                    max_velocity=max_velocity,
                                    rows=rows, cols=cols,
                                    plot_style=plot_style)
