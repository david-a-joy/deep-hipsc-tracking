""" Tools for tracking cells

Main function:

* :py:func:`link_all_chains` - Run a parallel frame to frame link for all input frames

Helper functions:

* :py:func:`find_link_functions` - Find all the link algorithms to try
* :py:func:`merge_points_pairwise` - Merge point sets by pairing points
* :py:func:`merge_points_cluster` - Merge point sets by clustering around a mean

API Documentation
-----------------

"""

# Imports
import time
import inspect
import traceback
from collections import namedtuple
from typing import List, Optional, Callable, Dict

# 3rd party imports
import numpy as np

from sklearn.neighbors import BallTree

# Our own imports
from ..utils import Hypermap
from . import link_chains, Link
from . import link_functions

# Constants

MAX_MERGE_DIST = 2.0  # Cutoff for distance between linked points

IMPUTE_STEPS = 1  # number of imputation iterations

LINK_STEP = 1  # Spacing between frames to link

# Classes


LinkItem = namedtuple('LinkItem', 'tp1, tp2, t1, t2, link_fxn, do_postprocessing')


# Main frame to frame linker


def _link_frames(item):
    # Inside the worker, run the frame-to-frame linkage and report the correspondences

    tp1, tp2, t1, t2, link_fxn, do_postprocessing = item
    try:
        print(f'Linking {tp1} to {tp2}')

        if tp1 == tp2:
            raise ValueError(f"Got duplicate timepoints: {tp1} to {tp2}")

        t0 = time.perf_counter()
        ind1, ind2 = link_fxn(t1, t2)
        print(f'Link took {(time.perf_counter() - t0):0.2f} secs')

        assert ind1.shape == ind2.shape
    except Exception:
        print(f'Error during link: {tp1} to {tp2}')
        traceback.print_exc()
        ind1 = ind2 = None

    if do_postprocessing and ind1 is not None and ind2 is not None:
        try:
            print(f'Postprocessing link: {tp1} to {tp2}')
            t1, t2, ind1, ind2 = link_functions.postprocess_delaunay(t1, t2, ind1, ind2)
        except Exception:
            print(f'Error during postprocessing: {tp1} to {tp2}')
            traceback.print_exc()
            ind1 = ind2 = None

    return tp1, tp2, t1, t2, ind1, ind2


def _link_and_merge_frames(items, link_fxn, link_step=LINK_STEP,
                           do_postprocessing=False,
                           processes=None,
                           max_merge_dist=MAX_MERGE_DIST,
                           merge_fxn='cluster'):
    # Link and merge all the frames
    pairs = {}
    with Hypermap(processes=processes) as pool:
        for linked_item in pool.map(_link_frames, items):
            tp1, tp2, t1, t2, ind1, ind2 = linked_item
            if ind1 is None or ind2 is None:
                raise ValueError(f'Invalid link {tp1} to {tp2}')
            pairs.setdefault(tp1, []).append(t1)
            pairs.setdefault(tp2, []).append(t2)

    # Reduce pass to merge the pairwise results
    # FIXME: Maybe make this run in parallel too?
    merged = {}
    for timepoint, track_pairs in pairs.items():
        assert timepoint not in merged
        if len(track_pairs) == 1:
            merged[timepoint] = track_pairs[0].copy()
        elif len(track_pairs) == 2:
            if merge_fxn == 'cluster':
                merged[timepoint] = merge_points_cluster(
                    track_pairs[0].copy(), track_pairs[1].copy(), max_dist=max_merge_dist)
            elif merge_fxn == 'pairwise':
                merged[timepoint] = merge_points_pairwise(
                    track_pairs[0].copy(), track_pairs[1].copy(), max_dist=max_merge_dist)
            else:
                raise KeyError(f'Unknown merge function {merge_fxn}')
        else:
            raise ValueError(f'Got invalid track_pairs at {timepoint}: {len(track_pairs)}')

    # Recycle into link items
    items = []
    for tp1 in sorted(merged):
        tp2 = tp1 + link_step
        if tp2 not in merged:
            continue
        t1 = merged[tp1]
        t2 = merged[tp2]
        assert t1.shape[1] == 2
        assert t2.shape[1] == 2
        items.append(LinkItem(tp1, tp2, t1, t2, link_fxn, do_postprocessing))
    return items


def _maybe_link_frames(tracks: List,
                       link_fxn: Callable,
                       link_step: int = LINK_STEP,
                       processes: Optional[int] = None,
                       impute_steps: int = IMPUTE_STEPS,
                       max_merge_dist: float = MAX_MERGE_DIST,
                       merge_fxn: str = 'cluster'):
    """ Create a parallel generator that runs the frame-to-frame linkage in parallel

    :param list tracks:
        The individual point sets to link
    :param callable link_fxn:
        The link function to use
    :param int link_step:
        Number of frames to step in the link
    :param int processes:
        Number of parallel processes to use while linking
    :param int impute_steps:
        Number of frames to impute over
    :param float max_merge_dist:
        Maximum distance to merge points
    :param str merge_fxn:
        Which function to use to merge points with
    """

    # Pack the tracks up so we can link in parallel
    items = []
    for track1, track2 in zip(tracks[:-link_step], tracks[link_step:]):
        tp1, _, t1 = track1
        tp2, _, t2 = track2
        items.append(LinkItem(tp1, tp2, t1, t2, link_fxn, impute_steps > 0))

    for i in range(impute_steps):
        items = _link_and_merge_frames(items,
                                       do_postprocessing=(i < impute_steps - 1),
                                       link_step=link_step,
                                       link_fxn=link_fxn,
                                       processes=processes,
                                       max_merge_dist=max_merge_dist,
                                       merge_fxn=merge_fxn)

    # Final pass is a link without a merge
    with Hypermap(processes=processes) as pool:
        return pool.map(_link_frames, items)


# Functions


def merge_points_pairwise(points1: np.ndarray,
                          points2: np.ndarray,
                          max_dist: float = MAX_MERGE_DIST) -> np.ndarray:
    """ Merge the points using pairwise matching

    Match each set of points in points1 and points2, then return the union
    of all matched and unmatched sets

    :param ndarray points1:
        The first frame points to merge
    :param ndarray points2:
        The second frame points to merge
    :param float max_dist:
        The maximum distance to pair two points across
    :returns:
        A single merged point set
    """

    match1, match2 = link_functions.pair_tracks_balltree(points1, points2, max_dist=max_dist)

    unmatched1 = np.ones(points1.shape[0], dtype=bool)
    unmatched1[match1] = 0

    unmatched2 = np.ones(points2.shape[0], dtype=bool)
    unmatched2[match2] = 0

    points_merge = (points1[match1, :] + points2[match2, :])/2
    return np.concatenate([points_merge, points1[unmatched1, :], points2[unmatched2, :]], axis=0)


def merge_points_cluster(points1: np.ndarray,
                         points2: np.ndarray,
                         max_dist: float = MAX_MERGE_DIST) -> np.ndarray:
    """ Merge the points using radial clustering

    Cluster all points into clusters with a maximum radius, then return all
    cluster centers of mass

    :param ndarray points1:
        The first frame points to merge
    :param ndarray points2:
        The second frame points to merge
    :param float max_dist:
        The maximum distance to cluster points over
    :returns:
        A single merged point set
    """
    if points1 is None:
        points1 = np.array([])
    if points2 is None:
        points2 = np.array([])

    if points1.shape[0] == 0 and points2.shape[0] == 0:
        return points1
    elif points1.shape[0] == 0:
        points = points2
    elif points2.shape[0] == 0:
        points = points1
    else:
        points = np.concatenate([points1, points2], axis=0)

    # Handle nans in arrays
    mask = np.any(np.isnan(points), axis=1)
    points = points[~mask, :]

    if points.shape[0] < 2:
        return points

    tree = BallTree(points)
    inds = tree.query_radius(points, r=max_dist)

    # Cluster each set and return centers of mass
    clusters = []
    used = set()
    for ind in inds:
        ind = [i for i in ind if i not in used]
        if ind:
            clusters.append(np.mean(points[ind, :], axis=0))
            used.update(ind)
    return np.array(clusters)


def find_link_functions() -> Dict[str, Callable]:
    """ Find all the link functions available

    :returns:
        A dictionary mapping func_name: function
    """
    link_fxns = {}
    for name, func in inspect.getmembers(link_functions, inspect.isfunction):
        if not name.startswith('pair_tracks_'):
            continue
        link_fxns[name[len('pair_tracks_'):]] = func
    return link_fxns


def link_all_chains(tracks: List,
                    link_fxn: Optional[Callable] = None,
                    processes: Optional[int] = None,
                    link_step: int = LINK_STEP,
                    impute_steps: int = IMPUTE_STEPS,
                    max_merge_dist: float = MAX_MERGE_DIST,
                    merge_fxn: str = 'cluster') -> List:
    """ Link a bunch of points into tracks

    :param list tracks:
        The points to pair up. This is a list of (timepoint, image, points) tuples
    :param str link_fxn:
        The link function name or function to use for linkage
    :param int processes:
        The number of parallel links to run
    :param int link_step:
        How many time steps to take to run each link
    :param int impute_steps:
        Number of steps to try and impute missing points
    :param float max_merge_dist:
        Maximum distance to merge imputed points during linkage
    :returns:
        A list of chains resulting from the linkage
    """

    index = None
    chains = None

    # Parse the link function
    if link_fxn is None:
        link_fxn = link_functions.pair_tracks_balltree
    elif isinstance(link_fxn, str):
        link_fxn = find_link_functions()[link_fxn.lower()]
    elif not callable(link_fxn):
        err = f'Invalid link function, expected name or callable, got {link_fxn}'
        raise TypeError(err)

    t0 = time.perf_counter()
    link_kwargs = {
        'link_fxn': link_fxn,
        'link_step': link_step,
        'processes': processes,
        'max_merge_dist': max_merge_dist,
        'merge_fxn': merge_fxn,
        'impute_steps': impute_steps,
    }

    # Link tracks between two movies
    multi_chains = []
    for i, links in enumerate(_maybe_link_frames(tracks, **link_kwargs)):
        tp1, tp2, t1, t2, ind1, ind2 = links

        # Initialize the first independent link
        if i // link_step == 0:
            chains = [[(tp1, x, y)] for x, y in t1]
            index = np.arange(t1.shape[0], dtype=np.uint32)
            multi_chains.append((index, chains))

        index, chains = multi_chains[i % link_step]
        # Debug the index when the links are weird
        # print(f'Link {tp1} to {tp2}')
        # print(f'Ind 1: {ind1.shape}, {np.min(ind1)} to {np.max(ind1)}')
        # print(f'Ind 2: {ind2.shape}, {np.min(ind2)} to {np.max(ind2)}')
        # print(f'Index: {index.shape}, {np.min(index)} to {np.max(index)}')
        multi_chains[i % link_step] = link_chains(tp2, t2, ind1, ind2, index, chains)

    final_chains = []
    for _, chains in multi_chains:
        final_chains.extend(Link.from_tuples(chain) for chain in chains)
    print(f'Total linking time: {(time.perf_counter() - t0):0.2f} secs')
    return final_chains
